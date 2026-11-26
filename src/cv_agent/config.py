import os
import ipaddress
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

import requests
import yaml
from .scoring_config import ScoringConfig, parse_scoring_config

def _resolve_root() -> Path:
    """Каталог данных проекта (этап 7.1): CV_AGENT_DATA_DIR, иначе checkout.

    Вычисление через `__file__.parents[2]` остаётся только как fallback для
    разработки; для установки вне checkout задайте CV_AGENT_DATA_DIR.
    """
    env = os.environ.get("CV_AGENT_DATA_DIR", "").strip()
    if env:
        path = Path(env).expanduser().resolve()
        path.mkdir(parents=True, exist_ok=True, mode=0o700)
        return path
    return Path(__file__).resolve().parents[2]


ROOT = _resolve_root()
DATA_DIR = ROOT / "data"
APPLICATIONS_DIR = ROOT / "applications"
DEFAULT_USER = "Илья"

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "llama-3.3-70b-versatile"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
GEMINI_NATIVE_MODELS_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models"
)

# Hardening (аудит L2): разумный предел длины пользовательского regex
MAX_EXCLUSION_REGEX_LENGTH = 200

# Максимальный лимит контекста документов в конфиге (символы)
MAX_DOCUMENT_LIMIT_CHARS = 200_000


def ollama_host() -> str:
    """Адрес Ollama из env OLLAMA_HOST (формат host:port), по умолчанию localhost:11434."""
    host = os.environ.get("OLLAMA_HOST", "localhost:11434").strip()
    if host.startswith(("http://", "https://")):
        return host.rstrip("/")
    return f"http://{host}"


def ollama_openai_url() -> str:
    return ollama_host() + "/v1"


def ollama_alive(timeout: float = 2.0) -> bool:
    """True, если Ollama отвечает на /api/tags."""
    try:
        from .resource_limits import network_request_limit

        with network_request_limit():
            return (
                requests.get(
                    ollama_host() + "/api/tags", timeout=timeout
                ).status_code
                == 200
            )
    except requests.RequestException:
        return False


def is_ollama_url(base_url: str) -> bool:
    return base_url.rstrip("/") in {
        ollama_openai_url().rstrip("/"),
        "http://localhost:11434/v1",
    }


@dataclass
class LLMConfig:
    provider: str = "deepseek"
    model: str = "deepseek-v4-flash"
    base_url: str = "https://api.deepseek.com"
    api_key_env: str = "DEEPSEEK_API_KEY"
    temperature: float = 0.3
    timeout: float = 120.0


@dataclass(frozen=True)
class RuntimeLimits:
    network_concurrency: int = 4
    llm_backend_concurrency: int = 1


@dataclass
class ExclusionRules:
    words: list[str] = field(default_factory=list)
    regex: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DocumentLimits:
    """Лимиты контекста документов (символы) для одного класса LLM-бэкенда."""

    max_chars_per_file: int
    keywords: int
    scoring: int
    letter: int

    def budget(self, purpose: str) -> int:
        return getattr(self, purpose)


DEFAULT_DOCUMENT_LIMITS = {
    "external": DocumentLimits(
        max_chars_per_file=16000, keywords=24000, scoring=32000, letter=32000
    ),
    "ollama": DocumentLimits(
        max_chars_per_file=6000, keywords=8000, scoring=12000, letter=12000
    ),
}


def _document_limits(document: dict) -> dict[str, DocumentLimits]:
    """Парсит criteria.document_limits; отсутствующие значения — дефолтные."""
    raw = document.get("document_limits", {}) or {}
    if not isinstance(raw, dict) or set(raw) - {"external", "ollama"}:
        raise ValueError(
            "criteria.document_limits must contain only 'external'/'ollama'"
        )
    resolved = {}
    for backend_class in ("external", "ollama"):
        entry = raw.get(backend_class, {}) or {}
        if not isinstance(entry, dict) or set(entry) - {
            "per_file", "keywords", "scoring", "letter"
        }:
            raise ValueError(
                f"criteria.document_limits.{backend_class} is invalid"
            )
        base = DEFAULT_DOCUMENT_LIMITS[backend_class]
        values = {}
        for name, default in (
            ("per_file", base.max_chars_per_file),
            ("keywords", base.keywords),
            ("scoring", base.scoring),
            ("letter", base.letter),
        ):
            configured = entry.get(name, default)
            if (
                isinstance(configured, bool)
                or not isinstance(configured, int)
                or not 1 <= configured <= MAX_DOCUMENT_LIMIT_CHARS
            ):
                raise ValueError(
                    f"criteria.document_limits.{backend_class}.{name} must be "
                    f"between 1 and {MAX_DOCUMENT_LIMIT_CHARS}"
                )
            values[name] = configured
        resolved[backend_class] = DocumentLimits(
            max_chars_per_file=values["per_file"],
            keywords=values["keywords"],
            scoring=values["scoring"],
            letter=values["letter"],
        )
    return resolved


@dataclass
class Criteria:
    region: str
    radius_km: int
    keywords: list[str]
    exclude_keywords: list[str]
    min_score: int
    llm: LLMConfig = field(default_factory=LLMConfig)
    llm_fallback: LLMConfig | None = None
    recruiting_agency_keywords: list[str] = field(default_factory=list)
    recruiting_agencies: list[str] = field(default_factory=list)
    results_min_score: int = 0
    search_max_keywords: int = 12
    search_per_query_limit: int = 25
    search_total_limit: int = 100
    search_similarity_threshold: float = 0.9
    getinit_categories: list[str] = field(default_factory=list)
    description_ttl_hours: int = 168
    exclusions: dict[str, ExclusionRules] = field(default_factory=dict)
    document_limits: dict[str, DocumentLimits] = field(
        default_factory=lambda: dict(DEFAULT_DOCUMENT_LIMITS)
    )
    network_concurrency: int = 4
    llm_backend_concurrency: int = 1
    scoring: ScoringConfig = field(default_factory=ScoringConfig)


def llm_endpoint_allowed(cfg: LLMConfig) -> bool:
    """Проверяет server-configured LLM URL до создания сетевого клиента."""
    try:
        parsed = urlsplit(cfg.base_url)
        host = (parsed.hostname or "").lower()
        _ = parsed.port
    except (TypeError, ValueError):
        return False
    if (
        parsed.scheme not in {"http", "https"}
        or not host
        or parsed.username
        or parsed.password
        or parsed.fragment
    ):
        return False
    provider = cfg.provider.strip().lower()
    if provider == "ollama":
        return True
    if parsed.scheme != "https":
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return False
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        address = None
    if address and (address.is_loopback or address.is_link_local or address.is_unspecified):
        return False
    if provider == "gemini":
        return host == "generativelanguage.googleapis.com"
    if provider == "groq":
        return host == "api.groq.com"
    return True


def current_user() -> str:
    """Текущая учётная запись: env CV_AGENT_USER (ставится web UI), иначе Илья."""
    return os.environ.get("CV_AGENT_USER", DEFAULT_USER).strip() or DEFAULT_USER


def load_yaml(name: str) -> dict:
    with open(ROOT / name, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _runtime_limits(document: dict) -> RuntimeLimits:
    raw = document.get("limits", {}) or {}
    if not isinstance(raw, dict) or set(raw) - {
        "network_concurrency", "llm_backend_concurrency"
    }:
        raise ValueError("criteria.limits is invalid")

    def value(name: str, default: int, maximum: int) -> int:
        configured = raw.get(name, default)
        if (
            isinstance(configured, bool)
            or not isinstance(configured, int)
            or not 1 <= configured <= maximum
        ):
            raise ValueError(
                f"criteria.limits.{name} must be between 1 and {maximum}"
            )
        return configured

    return RuntimeLimits(
        network_concurrency=value("network_concurrency", 4, 32),
        llm_backend_concurrency=value("llm_backend_concurrency", 1, 8),
    )


def load_runtime_limits() -> RuntimeLimits:
    """Leichte Limits-Konfiguration ohne Account-/DB-Auflösung."""
    return _runtime_limits(load_yaml("criteria.yaml"))


def load_criteria(user: str | None = None) -> Criteria:
    d = load_yaml("criteria.yaml")
    runtime_limits = _runtime_limits(d)
    keywords = d.get("keywords") or []
    profile_region = ""
    if user is None:
        user = current_user()
    if user:
        from .store import Store

        s = Store(user=user)
        try:
            user_keywords = s.account_keywords(user)
            profile = s.candidate_profile(user)
        finally:
            s.close()
        if user_keywords is not None:
            keywords = user_keywords
        if profile:
            profile_region = profile["search_region"]
    ra = d.get("recruiting_agencies", {}) or {}
    search = d.get("search", {}) or {}
    if not isinstance(search, dict):
        raise ValueError("criteria.search must be an object")

    def bounded_int(name: str, default: int, maximum: int) -> int:
        value = search.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= maximum:
            raise ValueError(f"criteria.search.{name} must be between 1 and {maximum}")
        return value

    similarity = search.get("similarity_threshold", 0.9)
    if isinstance(similarity, bool) or not isinstance(similarity, (int, float)):
        raise ValueError("criteria.search.similarity_threshold must be a number")
    similarity = float(similarity)
    if not 0 <= similarity <= 1:
        raise ValueError("criteria.search.similarity_threshold must be between 0 and 1")
    categories = search.get("getinit_categories", []) or []
    if not isinstance(categories, list) or not all(
        isinstance(category, str) and category.strip() for category in categories
    ):
        raise ValueError("criteria.search.getinit_categories must be a list of strings")
    raw_exclusions = d.get("exclusions")
    exclusions: dict[str, ExclusionRules] = {}
    if raw_exclusions is None:
        legacy_words = [
            str(value).strip()
            for value in (d.get("exclude_keywords") or [])
            if str(value).strip()
        ]
        exclusions = {
            "title": ExclusionRules(words=legacy_words),
            "company": ExclusionRules(),
            "description": ExclusionRules(words=legacy_words),
        }
    else:
        if not isinstance(raw_exclusions, dict) or set(raw_exclusions) - {
            "title", "company", "description"
        }:
            raise ValueError("criteria.exclusions contains unknown scopes")
        for scope in ("title", "company", "description"):
            raw_rules = raw_exclusions.get(scope, {}) or {}
            if not isinstance(raw_rules, dict) or set(raw_rules) - {"words", "regex"}:
                raise ValueError(f"criteria.exclusions.{scope} is invalid")
            words = raw_rules.get("words", []) or []
            patterns = raw_rules.get("regex", []) or []
            if not isinstance(words, list) or not all(
                isinstance(value, str) and value.strip() for value in words
            ):
                raise ValueError(f"criteria.exclusions.{scope}.words is invalid")
            if not isinstance(patterns, list) or not all(
                isinstance(value, str) and value.strip() for value in patterns
            ):
                raise ValueError(f"criteria.exclusions.{scope}.regex is invalid")
            # Hardening (аудит L2): ограничение длины и валидная компиляция
            # снижают риск ReDoS собственным конфигом; паттерны применяются
            # только к ограниченному по размеру тексту вакансии.
            if any(len(value) > MAX_EXCLUSION_REGEX_LENGTH for value in patterns):
                raise ValueError(
                    f"criteria.exclusions.{scope}.regex: pattern length "
                    f"exceeds {MAX_EXCLUSION_REGEX_LENGTH} characters"
                )
            try:
                for pattern in patterns:
                    re.compile(pattern, re.IGNORECASE)
            except re.error as exc:
                raise ValueError(
                    f"criteria.exclusions.{scope}.regex is invalid"
                ) from exc
            exclusions[scope] = ExclusionRules(
                words=list(dict.fromkeys(value.strip() for value in words)),
                regex=list(dict.fromkeys(value.strip() for value in patterns)),
            )
    # min_score ist der einzige aktuelle Schwellenwert. Das alte Feld bleibt
    # nur als Fallback lesbar, damit bestehende Installationen weiter starten.
    configured_min_score = d.get("min_score", d.get("results_min_score", 60))
    if (
        isinstance(configured_min_score, bool)
        or not isinstance(configured_min_score, (int, float))
        or not 0 <= configured_min_score <= 100
    ):
        raise ValueError("criteria.min_score must be between 0 and 100")
    configured_min_score = int(configured_min_score)
    llm = LLMConfig(**d.get("llm", {}))
    llm.provider = os.environ.get("LLM_PROVIDER", llm.provider)
    llm.model = os.environ.get("LLM_MODEL", llm.model)
    llm.base_url = os.environ.get("LLM_BASE_URL", llm.base_url)
    llm.api_key_env = os.environ.get("LLM_API_KEY_ENV", llm.api_key_env)
    if "LLM_BASE_URL" not in os.environ and "11434" in llm.base_url:
        llm.base_url = ollama_openai_url()
    document_limits = _document_limits(d)
    raw_radius = d.get("radius_km", 50)
    if isinstance(raw_radius, bool) or not isinstance(raw_radius, int) or not 1 <= raw_radius <= 500:
        raise ValueError("criteria.radius_km must be between 1 and 500")
    return Criteria(
        region=profile_region or d.get("region", ""),
        radius_km=raw_radius,
        keywords=keywords,
        exclude_keywords=d.get("exclude_keywords", []),
        min_score=configured_min_score,
        llm=llm,
        llm_fallback=(
            LLMConfig(**d["llm_fallback"]) if d.get("llm_fallback") else None
        ),
        recruiting_agency_keywords=ra.get("keywords", []),
        recruiting_agencies=ra.get("companies", []),
        results_min_score=configured_min_score,
        search_max_keywords=bounded_int("max_keywords", 12, 50),
        search_per_query_limit=bounded_int("per_query_limit", 25, 100),
        search_total_limit=bounded_int("total_limit", 100, 500),
        search_similarity_threshold=similarity,
        getinit_categories=list(dict.fromkeys(category.strip() for category in categories)),
        description_ttl_hours=bounded_int("description_ttl_hours", 168, 8760),
        exclusions=exclusions,
        document_limits=document_limits,
        network_concurrency=runtime_limits.network_concurrency,
        llm_backend_concurrency=runtime_limits.llm_backend_concurrency,
        scoring=parse_scoring_config(d.get("scoring")),
    )


def load_profile() -> dict:
    return load_yaml("profile.yaml")


def is_recruiting_agency(company: str, criteria: Criteria) -> bool:
    """True, если компания — рекрутинговое агентство (по названию)."""
    name = company.lower()
    agencies = getattr(criteria, "recruiting_agencies", None) or []
    keywords = getattr(criteria, "recruiting_agency_keywords", None) or []
    if any(a.lower() in name for a in agencies):
        return True
    return any(k.lower() in name for k in keywords)


def pick_llm(c: Criteria) -> LLMConfig:
    """Основная модель; если это Ollama и она недоступна — фолбэк на llm_fallback."""
    if c.llm_fallback and is_ollama_url(c.llm.base_url) and not ollama_alive():
        return c.llm_fallback
    return c.llm


def llm_backend_class(cfg: LLMConfig) -> str:
    """Класс бэкенда для лимитов контекста: 'ollama' или 'external'.

    Решение принимается по основной конфигурации без сетевого пробника:
    при фолбэке на внешнего провайдера документы собираются с локальными
    лимитами — безопасное направление (меньше контекста, чем может взять
    модель, но никогда больше окна Ollama).
    """
    provider = (cfg.provider or "").strip().lower()
    if provider == "ollama" or is_ollama_url(cfg.base_url):
        return "ollama"
    return "external"


def document_limits_for(
    criteria: Criteria | None = None, cfg: LLMConfig | None = None
) -> DocumentLimits:
    """Действующие лимиты контекста документов для выбранного LLM-бэкенда."""
    if criteria is None:
        criteria = load_criteria()
    cfg = cfg or criteria.llm
    return criteria.document_limits[llm_backend_class(cfg)]
