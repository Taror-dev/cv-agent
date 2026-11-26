import base64
import difflib
import hashlib
import json
import re
import time
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from urllib.parse import urlsplit, urlunsplit

import requests
from bs4 import BeautifulSoup

from .models import Vacancy
from .resource_limits import network_request_limit

BA_API = "https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v6/jobs"
BA_DETAIL = "https://rest.arbeitsagentur.de/jobboerse/jobsuche-service/pc/v4/jobdetails/{ref}"
BA_API_KEY = "jobboerse-jobsuche"
STEPSTONE_SEARCH = "https://www.stepstone.de/jobs/{keywords}/in-{location}"
# id заканчивается на -<digits>; допустимы суффиксы .htm/.html и query-string
STEPSTONE_JOB_RE = re.compile(r"/jobs/[^\"?#]*?-(\d+)(?:\.html?)?(?:[?#].*)?$")
STEPSTONE_STATE_RE = re.compile(
    r'window\.__PRELOADED_STATE__\["app-unifiedResultlist"\]\s*=\s*'
)
LINKEDIN_SEARCH = "https://www.linkedin.com/jobs/search"
LINKEDIN_URN_RE = re.compile(r"urn:li:jobPosting:(\d+)")
LINKEDIN_REMOVED_MARKERS = (
    "job posting has been removed",
    "job id provided may not be valid",
    "no longer accepting applications",
    "this job is no longer available",
)
GETINIT_API = "https://www.get-in-it.de/api/v2/open/job/search"
GETINIT_BASE = "https://www.get-in-it.de"
# Конкретная вакансия всегда живёт по /jobsuche/p<id>; прочие URL
# (например, страница компании) не являются отдельным объявлением.
GETINIT_JOB_URL_RE = re.compile(r"/jobsuche/p(\d+)")
GETINIT_CATEGORIES = {
    "Anwendungsentwicklung": 36,
    "Datenbankentwicklung/BI": 39,
    "IT-Security": 51,
    "Webentwicklung": 5,
    "Consulting": 37,
    "Embedded Systems": 45,
    "System Engineering / Admin": 35,
}
BROWSER_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)
MAX_SOURCE_ATTEMPTS = 3
SOURCE_BACKOFF_BASE = 0.5
DEFAULT_MAX_KEYWORDS = 12
DEFAULT_PER_QUERY_LIMIT = 25
DEFAULT_TOTAL_LIMIT = 100
DEFAULT_SIMILARITY_THRESHOLD = 0.9
REMOTE_LOCATION_MARKERS = {
    "bundesweit",
    "deutschlandweit",
    "remote",
    "homeoffice",
    "home office",
    "ortsunabhangig",
}
GETINIT_CATEGORY_SIGNALS = {
    "Anwendungsentwicklung": {
        "software",
        "entwickl",
        "developer",
        "engineer",
        "python",
        "java",
        "api",
        "automatisierung",
        "automation",
        "ai",
        "ki",
        "rag",
        "prompt",
    },
    "Datenbankentwicklung/BI": {
        "datenbank",
        "database",
        "data",
        "sql",
        "postgres",
        "mysql",
        "business intelligence",
        "bi",
    },
    "IT-Security": {
        "security",
        "sicherheit",
        "cyber",
        "soc",
        "pentest",
    },
    "Webentwicklung": {
        "web",
        "frontend",
        "backend",
        "fullstack",
        "javascript",
        "typescript",
        "node",
        "react",
        "angular",
        "vue",
    },
    "Consulting": {"consult", "beratung", "berater"},
    "Embedded Systems": {
        "embedded",
        "firmware",
        "microcontroller",
        "mikrocontroller",
        "c++",
    },
    "System Engineering / Admin": {
        "systemintegration",
        "system engineer",
        "administrator",
        "admin",
        "devops",
        "docker",
        "kubernetes",
        "linux",
        "cloud",
        "network",
        "netzwerk",
    },
}


@dataclass(frozen=True)
class SearchQuery:
    keyword: str
    location: str
    radius: int
    limit: int
    label: str = ""
    category_id: int | None = None


class SourceError(RuntimeError):
    """Normalisierter Adapterfehler ohne Response-Body oder Request-Geheimnisse."""

    def __init__(
        self,
        source: str,
        operation: str,
        message: str,
        *,
        retryable: bool = False,
        status_code: int | None = None,
    ):
        self.source = source
        self.operation = operation
        self.message = message
        self.retryable = retryable
        self.status_code = status_code
        super().__init__(f"{source}/{operation}: {message}")


class VacancyUnavailable(Exception):
    """Объявление больше не существует у источника (удалено или истекло)."""

    def __init__(self, source: str, message: str):
        self.source = source
        self.message = message
        super().__init__(f"{source}: {message}")


class SearchAdapter(ABC):
    source: str
    local_geo_filter = False

    def queries(
        self,
        keywords: list[str],
        location: str,
        radius: int,
        limit: int,
    ) -> list[SearchQuery]:
        return [
            SearchQuery(keyword, location, radius, limit, label=keyword)
            for keyword in keywords
        ]

    @abstractmethod
    def search(self, query: SearchQuery) -> list[Vacancy]:
        raise NotImplementedError

    @abstractmethod
    def detail(self, vacancy: Vacancy) -> str:
        raise NotImplementedError

    @abstractmethod
    def exists(self, vacancy: Vacancy) -> bool:
        raise NotImplementedError

    def refresh(self, vacancy: Vacancy) -> Vacancy:
        vacancy.description = self.detail(vacancy)
        return vacancy


class HTTPSourceAdapter(SearchAdapter):
    def _get(
        self,
        url: str,
        *,
        operation: str,
        accepted_statuses: set[int] | None = None,
        **kwargs,
    ):
        accepted = accepted_statuses or set()
        for attempt in range(1, MAX_SOURCE_ATTEMPTS + 1):
            try:
                with network_request_limit():
                    response = self.session.get(url, timeout=self.timeout, **kwargs)
            except (requests.ConnectionError, requests.Timeout) as exc:
                if attempt < MAX_SOURCE_ATTEMPTS:
                    time.sleep(SOURCE_BACKOFF_BASE * (2 ** (attempt - 1)))
                    continue
                raise SourceError(
                    self.source,
                    operation,
                    f"Network error after {attempt} attempts ({type(exc).__name__})",
                    retryable=True,
                ) from exc
            except requests.RequestException as exc:
                raise SourceError(
                    self.source,
                    operation,
                    f"Invalid HTTP request ({type(exc).__name__})",
                ) from exc

            status = int(response.status_code)
            if status in accepted:
                return response
            if 500 <= status <= 599:
                if attempt < MAX_SOURCE_ATTEMPTS:
                    time.sleep(SOURCE_BACKOFF_BASE * (2 ** (attempt - 1)))
                    continue
                raise SourceError(
                    self.source,
                    operation,
                    f"HTTP {status} after {attempt} attempts",
                    retryable=True,
                    status_code=status,
                )
            try:
                response.raise_for_status()
            except requests.RequestException as exc:
                raise SourceError(
                    self.source,
                    operation,
                    f"HTTP {status}",
                    status_code=status,
                ) from exc
            return response
        raise AssertionError("unerreichbarer Retry-Zustand")

    def exists(self, vacancy: Vacancy) -> bool:
        """True nur bei eindeutiger Verfuegbarkeit; 999/403 bleiben unbekannt."""
        response = self._get(
            vacancy.url,
            operation="exists",
            accepted_statuses={404, 410, 999},
        )
        return response.status_code not in {404, 410}


def normalize_url(url: str) -> str:
    try:
        parsed = urlsplit((url or "").strip())
    except ValueError:
        return ""
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return ""
    path = re.sub(r"/+", "/", parsed.path or "/").rstrip("/") or "/"
    return urlunsplit(
        (parsed.scheme.lower(), parsed.netloc.lower(), path, "", "")
    )


def stable_url_ref(prefix: str, url: str) -> str:
    normalized = normalize_url(url)
    if not normalized:
        return ""
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}-url-{digest}"


def normalize_match_text(value: str) -> str:
    """Unicode- und interpunktionsstabile Form für lokale Vergleiche."""
    decomposed = unicodedata.normalize("NFKD", str(value or "")).casefold()
    without_marks = "".join(
        char for char in decomposed if not unicodedata.combining(char)
    )
    return " ".join(re.sub(r"[^\w]+", " ", without_marks).split())


@dataclass(frozen=True)
class KeywordSelection:
    keywords: list[str]
    input_count: int
    duplicate_count: int
    omitted_count: int


def normalize_keywords(
    keywords: list[str], max_keywords: int = DEFAULT_MAX_KEYWORDS
) -> KeywordSelection:
    """Bereinigt die aktive deutsche Account-Liste und begrenzt den Request-Fanout."""
    if not 1 <= max_keywords <= 50:
        raise ValueError("max_keywords must be between 1 and 50")
    unique: list[str] = []
    seen: set[str] = set()
    duplicates = 0
    input_count = len(keywords)
    for raw in keywords:
        keyword = " ".join(unicodedata.normalize("NFKC", str(raw or "")).split())
        if not keyword:
            continue
        key = keyword.casefold()
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        unique.append(keyword[:100])
    selected = unique[:max_keywords]
    return KeywordSelection(
        keywords=selected,
        input_count=input_count,
        duplicate_count=duplicates,
        omitted_count=max(0, len(unique) - len(selected)),
    )


def select_getinit_categories(
    keywords: list[str], configured: list[str] | tuple[str, ...] | None = None
) -> list[str]:
    """Wählt Berufsfelder aus Account-Keywords oder einer expliziten Konfiguration."""
    if configured is not None:
        unknown = [name for name in configured if name not in GETINIT_CATEGORIES]
        if unknown:
            raise ValueError(
                "Unbekannte get-in-it-Kategorie(n): " + ", ".join(unknown)
            )
        return list(dict.fromkeys(configured))
    normalized = [normalize_match_text(keyword) for keyword in keywords]
    selected = [
        category
        for category, signals in GETINIT_CATEGORY_SIGNALS.items()
        if any(
            signal in keyword
            for keyword in normalized
            if keyword
            for signal in signals
        )
    ]
    return selected or ["Anwendungsentwicklung"]


def location_matches(candidate_location: str, requested_location: str) -> bool:
    """Lokaler Stadt/Remote-Filter für Quellen ohne funktionierenden Geo-Filter."""
    if not requested_location.strip():
        return True
    candidate = normalize_match_text(candidate_location)
    if not candidate:
        return False
    if any(marker in candidate for marker in REMOTE_LOCATION_MARKERS):
        return True
    requested_without_postcode = re.sub(r"\b\d{5}\b", " ", requested_location)
    requested = normalize_match_text(requested_without_postcode)
    if not requested:
        return True
    return requested in candidate


def validate_vacancy(vacancy: Vacancy, expected_source: str) -> tuple[Vacancy | None, str]:
    """Normalisiert eine Source-Antwort oder liefert einen begrenzten Skip-Grund."""
    vacancy.refnr = str(vacancy.refnr or "").strip()
    vacancy.title = " ".join(str(vacancy.title or "").split())
    vacancy.company = " ".join(str(vacancy.company or "").split())
    vacancy.location = " ".join(str(vacancy.location or "").split())
    vacancy.url = normalize_url(str(vacancy.url or ""))
    vacancy.source = str(vacancy.source or "").strip()
    if vacancy.source != expected_source:
        return None, "source_mismatch"
    if not vacancy.refnr or len(vacancy.refnr) > 200 or any(
        ord(char) < 32 for char in vacancy.refnr
    ):
        return None, "invalid_refnr"
    invalid_ids = {
        "stepstone": {"ss-", "ss-none", "ss-null"},
        "get-in-it": {"gi-", "gi-none", "gi-null"},
        "linkedin": {"li-", "li-none", "li-null"},
    }
    if vacancy.refnr.lower() in invalid_ids.get(expected_source, set()):
        return None, "unstable_refnr"
    if not vacancy.title:
        return None, "missing_title"
    if not vacancy.url:
        return None, "invalid_url"
    return vacancy, ""


@dataclass
class SourceSearchResult:
    source: str
    vacancies: list[Vacancy]
    queries: int
    skipped: int
    errors: list[SourceError]
    duration: float

    def summary(self) -> str:
        return (
            f"Source {self.source}: queries={self.queries} "
            f"found={len(self.vacancies)} skipped={self.skipped} "
            f"errors={len(self.errors)} duration={self.duration:.2f}s"
        )


def run_adapter(
    adapter: SearchAdapter,
    keywords: list[str],
    location: str | list[str],
    radius: int,
    limit: int,
    seen_refnrs: set[str] | None = None,
    total_limit: int | None = None,
) -> SourceSearchResult:
    """Führt alle Queries einer Quelle unabhängig aus und validiert ihre Ergebnisse.

    ``location`` — одна строка или список локаций. Для адаптеров с локальным
    geo-filter (запросы не зависят от локации, например get-in-it) запросы
    выполняются один раз и фильтруются против всех запрошенных локаций;
    для остальных генерируется набор запросов на каждую локацию.
    """
    started = time.monotonic()
    seen = seen_refnrs if seen_refnrs is not None else set()
    vacancies: list[Vacancy] = []
    skipped = 0
    errors: list[SourceError] = []
    locations = (
        [str(location)] if isinstance(location, str) else [str(loc) for loc in location]
    ) or [""]
    if getattr(adapter, "local_geo_filter", False):
        queries = adapter.queries(keywords, locations[0], radius, limit)
        queries = [
            replace(query, location=",".join(locations)) for query in queries
        ]
    else:
        queries = [
            query
            for loc in locations
            for query in adapter.queries(keywords, loc, radius, limit)
        ]
    executed_queries = 0
    for query in queries:
        remaining = None if total_limit is None else total_limit - len(vacancies)
        if remaining is not None and remaining <= 0:
            break
        if remaining is not None and query.limit > remaining:
            query = replace(query, limit=remaining)
        executed_queries += 1
        try:
            found = adapter.search(query)
        except SourceError as exc:
            label = (query.label or query.keyword or "query")[:80]
            errors.append(
                SourceError(
                    exc.source,
                    exc.operation,
                    f"Query '{label}': {exc.message}",
                    retryable=exc.retryable,
                    status_code=exc.status_code,
                )
            )
            continue
        except Exception as exc:  # noqa: BLE001 — parser contract boundary
            label = (query.label or query.keyword or "query")[:80]
            errors.append(
                SourceError(
                    adapter.source,
                    "search",
                    f"Query '{label}': unexpected adapter error ({type(exc).__name__})",
                )
            )
            continue
        for vacancy in found:
            valid, _reason = validate_vacancy(vacancy, adapter.source)
            if (
                valid is None
                or valid.refnr in seen
                or (
                    adapter.local_geo_filter
                    and not any(
                        location_matches(valid.location, loc) for loc in locations
                    )
                )
            ):
                skipped += 1
                continue
            seen.add(valid.refnr)
            vacancies.append(valid)
            if total_limit is not None and len(vacancies) >= total_limit:
                break
    return SourceSearchResult(
        source=adapter.source,
        vacancies=vacancies,
        queries=executed_queries,
        skipped=skipped,
        errors=errors,
        duration=time.monotonic() - started,
    )


def _normalize_title(value: str) -> str:
    value = re.sub(
        r"\([mwdifx/\-\s]{3,}\)|\b[mwdifx]\s*/\s*[mwdifx](?:\s*/\s*[mwdifx])+\b",
        " ",
        value,
        flags=re.I,
    )
    return normalize_match_text(value)


def _normalize_company(value: str) -> str:
    tokens = normalize_match_text(value).split()
    legal_forms = {"gmbh", "ag", "se", "kg", "mbh", "ug", "inc", "ltd", "llc"}
    return " ".join(token for token in tokens if token not in legal_forms)


def _normalize_location(value: str) -> str:
    return normalize_match_text(re.sub(r"\b\d{5}\b", " ", value))


def _locations_compatible(a: str, b: str) -> bool:
    if not a or not b:
        return True
    if any(marker in a or marker in b for marker in REMOTE_LOCATION_MARKERS):
        return True
    return a == b or a in b or b in a


def dedup(
    vacs: list[Vacancy],
    similarity: float = DEFAULT_SIMILARITY_THRESHOLD,
    existing: list[Vacancy] | None = None,
) -> list[Vacancy]:
    """Entfernt normalisierte Duplikate innerhalb neuer und gespeicherter Stellen."""
    if not 0 <= similarity <= 1:
        raise ValueError("similarity must be between 0 and 1")
    out: list[Vacancy] = []
    stored = existing or []
    for v in vacs:
        if any(_is_duplicate(v, keep, similarity) for keep in out):
            continue
        if any(
            old.refnr != v.refnr and _is_duplicate(v, old, similarity)
            for old in stored
        ):
            continue
        out.append(v)
    return out


def _is_duplicate(
    a: Vacancy,
    b: Vacancy,
    similarity: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> bool:
    company_a, company_b = _normalize_company(a.company), _normalize_company(b.company)
    if not company_a or company_a != company_b:
        return False
    location_a, location_b = _normalize_location(a.location), _normalize_location(b.location)
    if not _locations_compatible(location_a, location_b):
        return False
    ta, tb = _normalize_title(a.title), _normalize_title(b.title)
    if not ta or not tb or ta == tb:
        return ta == tb
    return difflib.SequenceMatcher(None, ta, tb).ratio() >= similarity


class BASearch(HTTPSourceAdapter):
    source = "arbeitsagentur"

    def __init__(self, timeout: int = 20):
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "cv-agent/0.1", "X-API-Key": BA_API_KEY}
        )
        self.timeout = timeout

    def search(
        self,
        query: SearchQuery | str,
        location: str = "",
        radius: int = 50,
        limit: int = 25,
    ) -> list[Vacancy]:
        if not isinstance(query, SearchQuery):
            query = SearchQuery(str(query), location, radius, limit, label=str(query))
        params = {
            "was": query.keyword,
            "wo": query.location,
            "umkreis": query.radius,
            "angebotsart": 1,
            "size": min(query.limit, 50),
            "seite": 1,
        }
        resp = self._get(BA_API, operation="search", params=params)
        vacancies = self._parse(resp.json())
        for vacancy in vacancies:
            vacancy.source_metadata["search_keyword"] = query.keyword
        return vacancies

    def detail(self, vacancy: Vacancy | str) -> str:
        refnr = vacancy.refnr if isinstance(vacancy, Vacancy) else str(vacancy)
        enc = base64.urlsafe_b64encode(refnr.encode()).decode().rstrip("=")
        resp = self._get(
            BA_DETAIL.format(ref=enc),
            operation="detail",
            accepted_statuses={404},
        )
        if resp.status_code == 404:
            raise VacancyUnavailable(
                self.source, "Referenznummer no longer available"
            )
        return resp.json().get("stellenangebotsBeschreibung") or ""

    def exists(self, vacancy: Vacancy | str) -> bool:
        """True если вакансия ещё доступна на BA API, False если удалена."""
        refnr = vacancy.refnr if isinstance(vacancy, Vacancy) else str(vacancy)
        enc = base64.urlsafe_b64encode(refnr.encode()).decode().rstrip("=")
        resp = self._get(
            BA_DETAIL.format(ref=enc),
            operation="exists",
            accepted_statuses={404},
        )
        if resp.status_code == 200:
            return True
        if resp.status_code == 404:
            return False
        return True

    @staticmethod
    def _parse(data: dict) -> list[Vacancy]:
        out = []
        for item in data.get("ergebnisliste", []):
            lok = item.get("stellenlokationen") or [{}]
            adresse = (lok[0] or {}).get("adresse", {})
            out.append(
                Vacancy(
                    refnr=item.get("referenznummer", ""),
                    title=item.get("stellenangebotsTitel", ""),
                    company=item.get("firma", ""),
                    location=f"{adresse.get('plz', '')} {adresse.get('ort', '')}".strip(),
                    url=item.get("externeUrl")
                    or f"https://www.arbeitsagentur.de/jobsuche/jobdetail/{item.get('referenznummer')}",
                    source="arbeitsagentur",
                    published=item.get("datumErsteVeroeffentlichung", ""),
                    description="",
                    source_metadata={
                        "provider_refnr": item.get("referenznummer", "")
                    },
                )
            )
        return out


def fetch_description(v: Vacancy) -> str:
    """Описание вакансии из любого источника (BA API / StepStone / LinkedIn / get-in-it)."""
    adapter_class = {
        "arbeitsagentur": BASearch,
        "stepstone": StepStoneSearch,
        "linkedin": LinkedInSearch,
        "get-in-it": GetInItSearch,
    }.get(v.source)
    if adapter_class is None:
        raise SourceError(v.source or "unknown", "detail", "Unsupported source")
    return adapter_class().detail(v)


def ensure_description(
    store,
    vacancy: Vacancy,
    *,
    ttl_hours: int = 168,
    fetcher=None,
) -> str:
    """Lädt Details nur bei fehlendem/abgelaufenem Cache und persistiert Erfolg."""
    checker = getattr(store, "description_needs_refresh", None)
    needs_refresh = (
        checker(vacancy.refnr, ttl_hours)
        if callable(checker)
        else not vacancy.description
    )
    if not needs_refresh:
        return vacancy.description
    try:
        description = (fetcher or fetch_description)(vacancy)
    except VacancyUnavailable:
        marker = getattr(store, "mark_unavailable", None)
        if callable(marker):
            marker(vacancy.refnr)
        raise
    vacancy.description = str(description or "")
    updater = getattr(store, "update_description", None)
    if callable(updater):
        updater(vacancy.refnr, vacancy.description)
    return vacancy.description


class StepStoneSearch(HTTPSourceAdapter):
    source = "stepstone"

    def __init__(self, timeout: int = 20):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": BROWSER_UA})
        self.timeout = timeout

    def detail(self, vacancy: Vacancy | str) -> str:
        """Описание со страницы вакансии (ld+json JobPosting)."""
        url = vacancy.url if isinstance(vacancy, Vacancy) else str(vacancy)
        search_url = STEPSTONE_SEARCH.format(
            keywords="python", location="deutschland"
        )
        headers = {"Referer": search_url}
        resp = self._get(
            url,
            operation="detail",
            accepted_statuses={403, 404},
            headers=headers,
        )
        if resp.status_code in (403, 404):
            self._get(search_url, operation="warmup")
            resp = self._get(
                url,
                operation="detail",
                accepted_statuses={404, 410},
                headers=headers,
            )
            if resp.status_code in (404, 410):
                raise VacancyUnavailable(
                    self.source, f"HTTP {resp.status_code}: posting removed"
                )
        return self._parse_detail(resp.text)

    def exists(self, vacancy: Vacancy | str) -> bool:
        """404/410 nach Warm-up-Header bedeutet: Stelle entfernt."""
        url = vacancy.url if isinstance(vacancy, Vacancy) else str(vacancy)
        search_url = STEPSTONE_SEARCH.format(
            keywords="python", location="deutschland"
        )
        headers = {"Referer": search_url}
        resp = self._get(
            url,
            operation="exists",
            accepted_statuses={403, 404, 410},
            headers=headers,
        )
        if resp.status_code == 403:
            self._get(search_url, operation="warmup")
            resp = self._get(
                url,
                operation="exists",
                accepted_statuses={404, 410},
                headers=headers,
            )
        return resp.status_code not in (404, 410)

    @staticmethod
    def _parse_detail(html: str) -> str:
        for m in re.finditer(
            r'<script type="application/ld\+json">(.*?)</script>', html, re.S
        ):
            try:
                data = json.loads(m.group(1).strip())
            except (json.JSONDecodeError, ValueError):
                continue
            items = data if isinstance(data, list) else [data]
            for it in items:
                if isinstance(it, dict) and it.get("@type") == "JobPosting":
                    desc = it.get("description") or it.get("descriptionHTML") or ""
                    if desc:
                        return BeautifulSoup(desc, "html.parser").get_text(
                            "\n", strip=True
                        )
        return ""

    def search(
        self,
        query: SearchQuery | str,
        location: str = "",
        limit: int = 25,
    ) -> list[Vacancy]:
        if not isinstance(query, SearchQuery):
            query = SearchQuery(str(query), location, 50, limit, label=str(query))
        url = STEPSTONE_SEARCH.format(
            keywords="-".join(query.keyword.split()),
            location=query.location.replace(" ", "-"),
        )
        resp = self._get(url, operation="search")
        vacs = self._parse_embedded(resp.text)
        if not vacs:
            vacs = self._parse_links(resp.text, query.location)
        for vacancy in vacs:
            vacancy.source_metadata["search_keyword"] = query.keyword
        return vacs[: query.limit]

    @staticmethod
    def _parse_embedded(html: str) -> list[Vacancy]:
        """Вакансии из встроенного JSON (window.__PRELOADED_STATE__['app-unifiedResultlist'])."""
        m = STEPSTONE_STATE_RE.search(html)
        if not m:
            return []
        try:
            obj, _ = json.JSONDecoder().raw_decode(html[m.end():])
        except ValueError:
            return []
        out = []
        for it in obj.get("searchResults", {}).get("items", []):
            if not isinstance(it, dict):
                continue
            url = it.get("url") or ""
            if not url.startswith("http"):
                url = "https://www.stepstone.de" + url
            harmonised_id = it.get("harmonisedId") or it.get("id") or ""
            refnr = (
                f"ss-{harmonised_id}"
                if harmonised_id
                else stable_url_ref("ss", url)
            )
            out.append(
                Vacancy(
                    refnr=refnr,
                    title=it.get("title") or "",
                    company=it.get("companyName") or "",
                    location=it.get("location") or "",
                    url=url.split("?")[0],
                    source="stepstone",
                    published=(it.get("datePosted") or "")[:10],
                    source_metadata=(
                        {"harmonised_id": str(harmonised_id)}
                        if harmonised_id
                        else {"id_strategy": "url_sha256"}
                    ),
                )
            )
        return out

    @staticmethod
    def _parse_links(html: str, location: str) -> list[Vacancy]:
        soup = BeautifulSoup(html, "html.parser")
        out = []
        for a in soup.select("a[href*='/jobs/']"):
            href = a.get("href", "")
            # query-string/fragment не мешают распознаванию id
            m = STEPSTONE_JOB_RE.search(href.split("?")[0].split("#")[0])
            if not m:
                continue
            if not href.startswith("http"):
                href = "https://www.stepstone.de" + href
            job_id = m.group(1)
            out.append(
                Vacancy(
                    refnr=f"ss-{job_id}",
                    title=a.get_text(" ", strip=True)[:120] or job_id,
                    company="",
                    location=location,
                    url=href,
                    source="stepstone",
                    source_metadata={"job_id": job_id},
                )
            )
        return out


class GetInItSearch(HTTPSourceAdapter):
    """get-in-it.de — публичное API `/api/v2/open/job/search` (нужен заголовок X-Requested-With).

    Портал не поддерживает полнотекстовый поиск и фильтр города (ломается на их
    стороне), единственное рабочее измерение — категории Berufsfeld
    (`filter[thematic_priority][]`). Ищем по категориям, релевантным профилю.
    """

    source = "get-in-it"
    local_geo_filter = True

    def __init__(
        self,
        timeout: int = 20,
        categories: list[str] | tuple[str, ...] | None = None,
    ):
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": BROWSER_UA, "X-Requested-With": "XMLHttpRequest"}
        )
        self.timeout = timeout
        self.categories = (
            None
            if categories is None
            else tuple(select_getinit_categories([], categories))
        )

    def queries(
        self,
        keywords: list[str],
        location: str,
        radius: int,
        limit: int,
    ) -> list[SearchQuery]:
        categories = select_getinit_categories(keywords, self.categories)
        return [
            SearchQuery(
                "",
                location,
                radius,
                limit,
                label=category,
                category_id=category_id,
            )
            for category in categories
            for category_id in [GETINIT_CATEGORIES[category]]
        ]

    def detail(self, vacancy: Vacancy | str) -> str:
        """Описание со страницы вакансии (__NEXT_DATA__ → jobJob.job.content)."""
        url = vacancy.url if isinstance(vacancy, Vacancy) else str(vacancy)
        resp = self._get(url, operation="detail")
        if self._redirected_away(resp, url):
            raise VacancyUnavailable(
                self.source, "Job posting redirects to company page"
            )
        return self._parse_detail(resp.text)

    @staticmethod
    def _redirected_away(resp, original_url: str) -> bool:
        """True, если удалённое объявление перебросило на страницу компании.

        get-in-it отвечает на просроченные /jobsuche/p<id> редиректом (307)
        на профиль компании; requests следует ему автоматически, поэтому
        сверяем итоговый URL с исходным идентификатором вакансии.
        """
        if not getattr(resp, "history", None):
            return False
        expected = GETINIT_JOB_URL_RE.search(original_url or "")
        if expected is None:
            return False
        final = GETINIT_JOB_URL_RE.search(getattr(resp, "url", "") or "")
        return final is None or final.group(1) != expected.group(1)

    def exists(self, vacancy: Vacancy | str) -> bool:
        """False при 404/410 или редиректе на страницу компании."""
        url = vacancy.url if isinstance(vacancy, Vacancy) else str(vacancy)
        try:
            resp = self._get(url, operation="exists")
        except SourceError as exc:
            if exc.status_code in (404, 410):
                return False
            raise
        return not self._redirected_away(resp, url)

    @staticmethod
    def _parse_detail(html: str) -> str:
        m = re.search(
            r'<script id="__NEXT_DATA__" type="application/json">(.*?)</script>',
            html,
            re.S,
        )
        if not m:
            return ""
        try:
            data = json.loads(m.group(1))
        except (json.JSONDecodeError, ValueError):
            return ""
        content = (
            data.get("props", {})
            .get("initialState", {})
            .get("jobJob", {})
            .get("job", {})
            .get("content", "")
        )
        if not content:
            return ""
        return BeautifulSoup(content, "html.parser").get_text("\n", strip=True)

    def search(
        self,
        query: SearchQuery | str = "",
        limit: int = 25,
    ) -> list[Vacancy]:
        if isinstance(query, SearchQuery):
            if query.category_id is None:
                raise SourceError(self.source, "search", "Kategorie fehlt")
            return self._search_category(query)

        location = str(query)
        out = []
        seen = set()
        for item in self.queries([], location, 50, limit):
            try:
                found = self._search_category(item)
            except SourceError:
                continue
            for v in found:
                if v.refnr not in seen:
                    seen.add(v.refnr)
                    out.append(v)
        return out

    def _search_category(self, query: SearchQuery) -> list[Vacancy]:
        params = {
            "filter[thematic_priority][]": query.category_id,
            "start": 0,
            "limit": min(query.limit, 100),
        }
        resp = self._get(GETINIT_API, operation="search", params=params)
        vacancies = self._parse(resp.json())
        for vacancy in vacancies:
            vacancy.source_metadata.update(
                {"category": query.label, "category_id": query.category_id}
            )
        return vacancies

    @staticmethod
    def _parse(data: dict) -> list[Vacancy]:
        out = []
        for it in data.get("items", {}).get("results", []):
            if not isinstance(it, dict):
                continue
            locs = "; ".join(
                str(location.get("name") or "").strip()
                for location in (it.get("locations") or [])
                if isinstance(location, dict) and location.get("name")
            )
            company = (it.get("company") or {}).get("title", "")
            raw_id = it.get("id")
            url = it.get("url") or ""
            if raw_id is not None:
                # канонический URL конкретной вакансии независимо от того,
                # какую ссылку вернул API (иногда это страница компании)
                url = f"{GETINIT_BASE}/jobsuche/p{raw_id}"
            else:
                if not GETINIT_JOB_URL_RE.search(url):
                    continue  # не отдельное объявление
                url = url if url.startswith("http") else GETINIT_BASE + url
            out.append(
                Vacancy(
                    refnr=f"gi-{raw_id}" if raw_id is not None else stable_url_ref("gi", url),
                    title=it.get("title") or "",
                    company=company,
                    location=locs,
                    url=url,
                    source="get-in-it",
                    source_metadata=(
                        {"job_id": raw_id}
                        if raw_id is not None
                        else {"id_strategy": "url_sha256"}
                    ),
                )
            )
        return out


class LinkedInSearch(HTTPSourceAdapter):
    source = "linkedin"

    def __init__(self, timeout: int = 20):
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": BROWSER_UA})
        self.timeout = timeout

    def detail(self, vacancy: Vacancy | str) -> str:
        """Описание со страницы вакансии (div.description__text)."""
        url = vacancy.url if isinstance(vacancy, Vacancy) else str(vacancy)
        clean = url.split("?")[0]
        resp = self._get(
            clean,
            operation="detail",
            accepted_statuses={999, 404},
        )
        if resp.status_code == 999:
            return ""
        if resp.status_code == 404 or self._is_removed_page(resp.text):
            raise VacancyUnavailable(
                self.source, "Job posting removed or invalid"
            )
        return self._parse_detail(resp.text)

    @staticmethod
    def _is_removed_page(html: str) -> bool:
        text = re.sub(r"<[^>]+>", " ", html or "").lower()
        return any(
            marker in text
            for marker in LINKEDIN_REMOVED_MARKERS
        )

    def exists(self, vacancy: Vacancy | str) -> bool:
        """False nur wenn die Stelle eindeutig entfernt wurde (404/Marker)."""
        url = vacancy.url if isinstance(vacancy, Vacancy) else str(vacancy)
        clean = url.split("?")[0]
        resp = self._get(
            clean,
            operation="exists",
            accepted_statuses={999, 404},
        )
        if resp.status_code == 404:
            return False
        if resp.status_code == 999:
            return True  # Anti-Bot-Block: Verfuegbarkeit unbekannt
        return not self._is_removed_page(resp.text)

    @staticmethod
    def _parse_detail(html: str) -> str:
        soup = BeautifulSoup(html, "html.parser")
        el = soup.select_one("div.description__text")
        return el.get_text("\n", strip=True) if el else ""

    def search(
        self,
        query: SearchQuery | str,
        location: str = "",
        limit: int = 25,
    ) -> list[Vacancy]:
        if not isinstance(query, SearchQuery):
            query = SearchQuery(str(query), location, 50, limit, label=str(query))
        params = {"keywords": query.keyword, "location": query.location}
        resp = self._get(LINKEDIN_SEARCH, operation="search", params=params)
        vacancies = self._parse(resp.text, query.limit)
        for vacancy in vacancies:
            vacancy.source_metadata["search_keyword"] = query.keyword
        return vacancies

    @staticmethod
    def _parse(html: str, limit: int = 25) -> list[Vacancy]:
        soup = BeautifulSoup(html, "html.parser")
        out = []
        for card in soup.select("div.base-search-card"):
            urn = card.get("data-entity-urn", "")
            m = LINKEDIN_URN_RE.search(urn)
            a = card.select_one("a.base-card__full-link")
            title_el = card.select_one("h3.base-search-card__title")
            company_el = card.select_one("h4.base-search-card__subtitle")
            location_el = card.select_one("span.job-search-card__location")
            time_el = card.select_one("time")
            url = a.get("href", "") if a else ""
            out.append(
                Vacancy(
                    refnr=(
                        f"li-{m.group(1)}"
                        if m
                        else stable_url_ref("li", url)
                    ),
                    title=title_el.get_text(strip=True) if title_el else "",
                    company=company_el.get_text(strip=True) if company_el else "",
                    location=location_el.get_text(strip=True) if location_el else "",
                    url=url,
                    source="linkedin",
                    published=(time_el.get("datetime", "") or "")[:10] if time_el else "",
                    source_metadata=(
                        {"job_posting_id": m.group(1)}
                        if m
                        else {"id_strategy": "url_sha256"}
                    ),
                )
            )
            if len(out) >= limit:
                break
        return out
