import json

import pytest

from cv_agent.search import GetInItSearch, LinkedInSearch, StepStoneSearch

STATE = """
<script>
window.__PRELOADED_STATE__ = window.__PRELOADED_STATE__ || {};
window.__PRELOADED_STATE__["app-unifiedResultlist"] = %s;
</script>
"""


def _state_html(items):
    return STATE % json.dumps(
        {"searchResults": {"items": items}}, ensure_ascii=False
    )


def test_stepstone_embedded():
    html = _state_html(
        [
            {
                "title": "AI Engineer (m/w/d)",
                "companyName": "ACME GmbH",
                "location": "Lindenberg im Allgäu",
                "harmonisedId": "ab6d22b9-5d7a-4633-9425-1347fe966972",
                "datePosted": "2026-08-12T01:10:26+02:00",
                "url": "/stellenangebote--AI-Engineer--14332119-inline.html?rltr=1_1",
            }
        ]
    )
    vacs = StepStoneSearch._parse_embedded(html)
    assert len(vacs) == 1
    v = vacs[0]
    assert v.refnr == "ss-ab6d22b9-5d7a-4633-9425-1347fe966972"
    assert v.title == "AI Engineer (m/w/d)"
    assert v.company == "ACME GmbH"
    assert v.location == "Lindenberg im Allgäu"
    assert v.source == "stepstone"
    assert v.source_metadata["harmonised_id"] == "ab6d22b9-5d7a-4633-9425-1347fe966972"
    assert v.published == "2026-08-12"
    assert v.url.startswith("https://www.stepstone.de/stellenangebote--AI-Engineer--14332119-inline.html")
    assert "rltr" not in v.url


def test_stepstone_embedded_empty():
    assert StepStoneSearch._parse_embedded("<html></html>") == []
    assert StepStoneSearch._parse_embedded(_state_html([])) == []


def test_stepstone_detail_ldjson():
    html = """
    <script type="application/ld+json">
    {"@type": "JobPosting", "title": "AI Engineer",
     "description": "<h4>Ihre Aufgaben</h4><p>Sie bauen KI-Modelle.</p>"}
    </script>
    """
    desc = StepStoneSearch._parse_detail(html)
    assert "Ihre Aufgaben" in desc
    assert "KI-Modelle" in desc
    assert "<" not in desc


def test_stepstone_detail_ldjson_list_and_missing():
    html = """
    <script type="application/ld+json">[{"@type": "WebSite", "name": "x"}]</script>
    """
    assert StepStoneSearch._parse_detail(html) == ""
    assert StepStoneSearch._parse_detail("<html></html>") == ""


def test_stepstone_links_fallback():
    html = """
    <a href="/jobs/data-scientist/in-berlin-123456">Data Scientist</a>
    <a href="https://www.stepstone.de/jobs/ai-engineer/in-stuttgart-789">AI Engineer</a>
    <a href="/company-page">not a job</a>
    """
    vacs = StepStoneSearch._parse_links(html, "Berlin")
    assert [v.title for v in vacs] == ["Data Scientist", "AI Engineer"]
    assert vacs[0].refnr == "ss-123456"
    assert vacs[1].url.startswith("https://www.stepstone.de")


def test_linkedin_cards():
    html = """
    <div class="base-card base-search-card job-search-card"
         data-entity-urn="urn:li:jobPosting:4445956025">
      <a class="base-card__full-link" href="https://de.linkedin.com/jobs/view/data-scientist-4445956025"></a>
      <h3 class="base-search-card__title">Data Scientist (m/w/d)</h3>
      <h4 class="base-search-card__subtitle">AltaSigma</h4>
      <span class="job-search-card__location">Ulm, Baden-Württemberg, Germany</span>
      <time datetime="2026-08-10T00:00:00Z"></time>
    </div>
    <div class="base-card base-search-card job-search-card" data-entity-urn="urn:li:jobPosting:111">
      <a class="base-card__full-link" href="https://de.linkedin.com/jobs/view/ml-111"></a>
      <h3 class="base-search-card__title">ML Engineer</h3>
      <h4 class="base-search-card__subtitle">Beta AG</h4>
      <span class="job-search-card__location">München</span>
    </div>
    """
    vacs = LinkedInSearch._parse(html, limit=25)
    assert len(vacs) == 2
    v = vacs[0]
    assert v.refnr == "li-4445956025"
    assert v.title == "Data Scientist (m/w/d)"
    assert v.company == "AltaSigma"
    assert v.location == "Ulm, Baden-Württemberg, Germany"
    assert v.source == "linkedin"
    assert v.source_metadata["job_posting_id"] == "4445956025"
    assert v.url == "https://de.linkedin.com/jobs/view/data-scientist-4445956025"
    assert v.published == "2026-08-10"
    assert vacs[1].published == ""


def test_linkedin_limit():
    html = """
    <div class="base-search-card" data-entity-urn="urn:li:jobPosting:1"><h3 class="base-search-card__title">A</h3></div>
    <div class="base-search-card" data-entity-urn="urn:li:jobPosting:2"><h3 class="base-search-card__title">B</h3></div>
    <div class="base-search-card" data-entity-urn="urn:li:jobPosting:3"><h3 class="base-search-card__title">C</h3></div>
    """
    assert len(LinkedInSearch._parse(html, limit=2)) == 2


def test_linkedin_detail():
    html = """
    <div class="show-more-less-html__markup">
      <div class="description__text">
        <p>Wir suchen einen Python Developer</p>
        <ul><li>Erfahrung mit Flask</li><li>Docker</li></ul>
      </div>
    </div>
    """
    desc = LinkedInSearch._parse_detail(html)
    assert "Python Developer" in desc
    assert "Flask" in desc
    assert LinkedInSearch._parse_detail("<html></html>") == ""


def test_getinit_detail():
    import json as _json

    data = {
        "props": {
            "initialState": {
                "jobJob": {
                    "job": {
                        "content": (
                            "<section><h3>Aufgaben</h3>"
                            "<p>Backend-Entwicklung mit Python</p></section>"
                        )
                    }
                }
            }
        }
    }
    html = (
        '<script id="__NEXT_DATA__" type="application/json">'
        + _json.dumps(data)
        + "</script>"
    )
    desc = GetInItSearch._parse_detail(html)
    assert "Aufgaben" in desc
    assert "Python" in desc
    assert GetInItSearch._parse_detail("<html></html>") == ""


def test_getinit_parse():
    data = {
        "total": 2,
        "items": {
            "results": [
                {
                    "id": 311292,
                    "title": "Software Engineer Fullstack (m/w/d)",
                    "url": "/jobsuche/p311292",
                    "homeOffice": True,
                    "locations": [{"id": 6495, "name": "Aschheim"}],
                    "company": {"id": 612, "title": "Atruvia AG"},
                },
                {
                    "id": 311293,
                    "title": "AI Engineer (m/w/d)",
                    "url": "https://www.get-in-it.de/jobsuche/p311293",
                    "locations": [{"name": "bundesweit"}, {"name": "München"}],
                    "company": {"title": "Beta GmbH"},
                },
            ]
        },
    }
    vacs = GetInItSearch._parse(data)
    assert len(vacs) == 2
    v = vacs[0]
    assert v.refnr == "gi-311292"
    assert v.title == "Software Engineer Fullstack (m/w/d)"
    assert v.company == "Atruvia AG"
    assert v.location == "Aschheim"
    assert v.source == "get-in-it"
    assert v.source_metadata["job_id"] == 311292
    assert v.url == "https://www.get-in-it.de/jobsuche/p311292"
    assert vacs[1].location == "bundesweit; München"
    assert vacs[1].url.startswith("http")


def test_getinit_parse_empty():
    assert GetInItSearch._parse({}) == []
    assert GetInItSearch._parse({"items": {"results": []}}) == []


# ---- Аудит M2: fixture-покрытие непокрытых веток парсеров ----

import requests


class FakeResponse:
    def __init__(
        self,
        status_code=200,
        data=None,
        text="",
        url="",
        history=None,
    ):
        self.status_code = status_code
        self._data = data or {}
        self.text = text
        self.url = url
        self.history = history or []

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            import requests as _r

            raise _r.HTTPError(f"HTTP {self.status_code}", response=self)


def test_stepstone_parse_links_fallback():
    html = """
    <a class="x" href="/jobs/ai-engineer--998877.htm?rltr=5">AI Engineer bei ACME</a>
    <a href="/jobs/uebersicht.htm">listing без id</a>
    <a href="https://www.stepstone.de/jobs/python--5544.htm">Python Dev</a>
    """
    vacs = StepStoneSearch._parse_links(html, "Wangen")
    assert [v.refnr for v in vacs] == ["ss-998877", "ss-5544"]
    assert vacs[0].url.startswith("https://www.stepstone.de/jobs/")
    assert "?rltr" in vacs[0].url  # query сохраняется в ссылке
    assert vacs[0].location == "Wangen"
    assert "AI Engineer" in vacs[0].title
    assert vacs[0].source_metadata["job_id"] == "998877"


def test_stepstone_detail_description_html_and_invalid_json_skipped():
    html = """
    <script type="application/ld+json">{broken json</script>
    <script type="application/ld+json">[
      {"@type": "WebPage", "name": "skip"},
      {"@type": "JobPosting", "descriptionHTML": "<p>Erste Zeile</p><b>Fett</b>"}
    ]</script>
    """
    text = StepStoneSearch._parse_detail(html)
    assert "Erste Zeile" in text and "Fett" in text
    assert "<p>" not in text  # html-теги сняты


def test_ba_search_detail_exists_contract(monkeypatch):
    from cv_agent.models import Vacancy
    from cv_agent.search import BASearch, SearchQuery

    ba = BASearch()
    calls = []

    def fake_get(url, *, operation, accepted_statuses=None, **kwargs):
        calls.append((operation, url))
        if operation == "search":
            return FakeResponse(
                data={
                    "ergebnisliste": [
                        {
                            "referenznummer": "10000-12345",
                            "stellenangebotsTitel": "DevOps Engineer",
                            "firma": "ACME AG",
                            "stellenlokationen": [
                                {"adresse": {"plz": "88239", "ort": "Wangen"}}
                            ],
                            "externeUrl": "https://example.test/ba/1",
                            "eintrittsdatum": "2026-09-01",
                        }
                    ]
                }
            )
        if operation == "detail":
            return FakeResponse(data={"stellenangebotsBeschreibung": "Wir suchen"})
        if operation == "exists":
            return FakeResponse(status_code=404)
        raise AssertionError(operation)

    monkeypatch.setattr(ba, "_get", fake_get)

    vacs = ba.search(SearchQuery("DevOps", "88239 Wangen", 50, 25))
    assert len(vacs) == 1 and vacs[0].refnr == "10000-12345"
    assert vacs[0].location == "88239 Wangen"
    assert vacs[0].source_metadata["search_keyword"] == "DevOps"

    v = Vacancy("10000-12345", "t", "c", "l", "u", "arbeitsagentur")
    assert ba.detail(v) == "Wir suchen"
    assert ba.exists(v) is False  # 404 -> вакансия удалена
    ops = [op for op, _ in calls]
    assert ops == ["search", "detail", "exists"]


def test_linkedin_cards_fixture_full_fields_and_limit():
    from cv_agent.search import LinkedInSearch

    html = """
    <div class="base-search-card" data-entity-urn="urn:li:jobPosting:3099123456">
      <a class="base-card__full-link" href="https://www.linkedin.com/jobs/view/3099123456">x</a>
      <h3 class="base-search-card__title">Backend Engineer</h3>
      <h4 class="base-search-card__subtitle">Muster GmbH</h4>
      <span class="job-search-card__location">München, Bayern</span>
      <time datetime="2026-08-19"></time>
    </div>
    <div class="base-search-card" data-entity-urn="urn:li:jobPosting:111">
      <a class="base-card__full-link" href="https://www.linkedin.com/jobs/view/111"></a>
      <h3 class="base-search-card__title">Second</h3>
    </div>
    """
    vacs = LinkedInSearch._parse(html, limit=1)
    assert len(vacs) == 1  # limit обрезает
    v = vacs[0]
    assert v.refnr == "li-3099123456"
    assert v.title == "Backend Engineer"
    assert v.company == "Muster GmbH"
    assert v.location == "München, Bayern"
    assert v.published == "2026-08-19"
    # карточка без urn -> стабильный hash от URL
    fallback = LinkedInSearch._parse('<div class="base-search-card"><a class="base-card__full-link" href="https://www.linkedin.com/jobs/view/222"></a></div>')
    assert fallback[0].refnr.startswith("li-li_") or len(fallback[0].refnr) > 10
    assert fallback[0].source_metadata.get("id_strategy") == "url_sha256"


def test_linkedin_search_sets_keyword_metadata(monkeypatch):
    from cv_agent.search import LinkedInSearch, SearchQuery

    li = LinkedInSearch()
    captured = {}

    def fake_get(url, *, operation, accepted_statuses=None, **kwargs):
        captured["url"] = url
        captured["params"] = kwargs.get("params")
        return FakeResponse(text="" )  # пустая выдача — достаточно для ветки

    monkeypatch.setattr(li, "_get", fake_get)
    vacs = li.search(SearchQuery("python", "München", 50, 10))
    assert vacs == []
    assert captured["params"]["keywords"] == "python"
    assert captured["params"]["location"] == "München"


def test_getinit_string_query_skips_source_errors(monkeypatch):
    from cv_agent.models import Vacancy
    from cv_agent.search import GetInItSearch, SearchQuery, SourceError

    gi = GetInItSearch()
    items = [
        SearchQuery("", "X", 50, 25, label="IT-Security", category_id=11),
        SearchQuery("", "X", 50, 25, label="Entwicklung", category_id=5),
    ]
    monkeypatch.setattr(gi, "queries", lambda *a, **k: items)
    monkeypatch.setattr(
        gi,
        "_search_category",
        lambda q: (
            (_ for _ in ()).throw(SourceError("get-in-it", "search", "boom"))
            if q.label == "IT-Security"
            else [
                Vacancy("gi-1", "T", "C", "München", "u", "get-in-it"),
                Vacancy("gi-1", "T", "C", "Hamburg", "u", "get-in-it"),  # дубликат
            ]
        ),
    )
    out = gi.search("IT")
    # ошибка первой категории изолирована, дубликат refnr удалён
    assert [v.refnr for v in out] == ["gi-1"]


def test_linkedin_removed_page_raises_unavailable():
    from cv_agent.models import Vacancy
    from cv_agent.search import LinkedInSearch, VacancyUnavailable

    html = (
        "<html><body><h1>Unable to load the page</h1>"
        "<p>Job id provided may not be valid or the job posting has been "
        "removed.</p></body></html>"
    )
    li = LinkedInSearch()
    monkey_response = FakeResponse(status_code=200, text=html)
    li._get = lambda *a, **k: monkey_response
    v = Vacancy("li-4449409532", "t", "c", "l", "https://x.test/1", "linkedin")
    with pytest.raises(VacancyUnavailable):
        li.detail(v)


def test_linkedin_detail_404_raises_unavailable():
    from cv_agent.models import Vacancy
    from cv_agent.search import LinkedInSearch, VacancyUnavailable

    li = LinkedInSearch()
    li._get = lambda *a, **k: FakeResponse(status_code=404, text="")
    v = Vacancy("li-1", "t", "c", "l", "https://x.test/1", "linkedin")
    with pytest.raises(VacancyUnavailable):
        li.detail(v)


def test_linkedin_detail_999_still_returns_empty():
    from cv_agent.models import Vacancy
    from cv_agent.search import LinkedInSearch

    li = LinkedInSearch()
    li._get = lambda *a, **k: FakeResponse(status_code=999, text="blocked")
    v = Vacancy("li-1", "t", "c", "l", "https://x.test/1", "linkedin")
    assert li.detail(v) == ""


def test_ba_detail_404_raises_unavailable():
    from cv_agent.models import Vacancy
    from cv_agent.search import BASearch, VacancyUnavailable

    ba = BASearch()
    ba._get = lambda *a, **k: FakeResponse(status_code=404, data={})
    v = Vacancy("10000-1", "t", "c", "l", "u", "arbeitsagentur")
    with pytest.raises(VacancyUnavailable):
        ba.detail(v)


def test_stepstone_removed_after_warmup_raises(monkeypatch):
    from cv_agent.models import Vacancy
    from cv_agent.search import StepStoneSearch, VacancyUnavailable

    ss = StepStoneSearch()
    responses = [
        FakeResponse(status_code=404, text=""),  # первый detail
        FakeResponse(status_code=200, text="warmup"),
        FakeResponse(status_code=404, text=""),  # повтор после warm-up
    ]
    monkeypatch.setattr(ss, "_get", lambda *a, **k: responses.pop(0))
    v = Vacancy(
        "ss-1",
        "t",
        "c",
        "l",
        "https://www.stepstone.de/jobs/job--1.htm",
        "stepstone",
    )
    with pytest.raises(VacancyUnavailable):
        ss.detail(v)


def test_ensure_description_marks_store_unavailable():
    from cv_agent.models import Vacancy
    from cv_agent.search import VacancyUnavailable, ensure_description

    class FakeStore:
        def __init__(self):
            self.unavailable = []

        def mark_unavailable(self, refnr):
            self.unavailable.append(refnr)

    store = FakeStore()
    v = Vacancy("li-9", "t", "c", "l", "u", "linkedin")

    def fetcher(vacancy):
        raise VacancyUnavailable("linkedin", "Job posting removed or invalid")

    with pytest.raises(VacancyUnavailable):
        ensure_description(store, v, fetcher=fetcher)
    assert store.unavailable == ["li-9"]


_REMOVED_HTML = (
    "<html><body><h1>Unable to load the page</h1>"
    "<p>Job id provided may not be valid or the job posting has been "
    "removed.</p></body></html>"
)


def test_linkedin_exists_removed_page_is_false():
    from cv_agent.models import Vacancy
    from cv_agent.search import LinkedInSearch

    li = LinkedInSearch()
    li._get = lambda *a, **k: FakeResponse(status_code=200, text=_REMOVED_HTML)
    v = Vacancy("li-1", "t", "c", "l", "https://x.test/1?a=b", "linkedin")
    assert li.exists(v) is False


def test_linkedin_exists_999_means_unknown_not_removed():
    from cv_agent.models import Vacancy
    from cv_agent.search import LinkedInSearch

    li = LinkedInSearch()
    li._get = lambda *a, **k: FakeResponse(status_code=999, text="")
    v = Vacancy("li-1", "t", "c", "l", "https://x.test/1", "linkedin")
    assert li.exists(v) is True


def test_linkedin_exists_normal_page_is_true():
    from cv_agent.models import Vacancy
    from cv_agent.search import LinkedInSearch

    li = LinkedInSearch()
    li._get = lambda *a, **k: FakeResponse(
        status_code=200, text="<div class='description__text'>Wir suchen</div>"
    )
    v = Vacancy("li-1", "t", "c", "l", "https://x.test/1", "linkedin")
    assert li.exists(v) is True


def test_stepstone_exists_404_after_warmup_is_false(monkeypatch):
    from cv_agent.models import Vacancy
    from cv_agent.search import StepStoneSearch

    ss = StepStoneSearch()
    responses = [
        FakeResponse(status_code=403, text=""),
        FakeResponse(status_code=200, text="warmup"),
        FakeResponse(status_code=404, text=""),
    ]
    monkeypatch.setattr(ss, "_get", lambda *a, **k: responses.pop(0))
    v = Vacancy(
        "ss-1",
        "t",
        "c",
        "l",
        "https://www.stepstone.de/jobs/job--1.htm",
        "stepstone",
    )
    assert ss.exists(v) is False


def test_getinit_parse_builds_canonical_job_url():
    from cv_agent.search import GetInItSearch

    data = {
        "items": {
            "results": [
                {  # API вернул ссылку на страницу компании
                    "id": 289407,
                    "title": "Werkstudent IoT",
                    "url": "/unternehmen/deloitte-gmbh",
                    "company": {"title": "Deloitte"},
                },
                {  # обычное объявление с относительным URL
                    "id": 311292,
                    "title": "SE Fullstack",
                    "url": "/jobsuche/p311292",
                    "company": {"title": "ACME"},
                },
                {  # без id и без URL объявления — не конкретная вакансия
                    "title": "Company listing",
                    "url": "/unternehmen/acme",
                    "company": {"title": "ACME"},
                },
            ]
        }
    }
    vacs = GetInItSearch._parse(data)
    assert [v.refnr for v in vacs] == ["gi-289407", "gi-311292"]
    assert (
        vacs[0].url == "https://www.get-in-it.de/jobsuche/p289407"
    )
    assert (
        vacs[1].url == "https://www.get-in-it.de/jobsuche/p311292"
    )


def test_getinit_detail_redirect_to_company_raises_unavailable():
    from cv_agent.models import Vacancy
    from cv_agent.search import GetInItSearch, VacancyUnavailable

    gi = GetInItSearch()
    gi._get = lambda *a, **k: FakeResponse(
        status_code=200,
        text="<html>company page</html>",
        url="https://www.get-in-it.de/unternehmen/deloitte-gmbh",
        history=[object()],
    )
    v = Vacancy(
        "gi-289407",
        "t",
        "c",
        "l",
        "https://www.get-in-it.de/jobsuche/p289407",
        "get-in-it",
    )
    with pytest.raises(VacancyUnavailable):
        gi.detail(v)


def test_getinit_detail_same_page_returns_content():
    from cv_agent.models import Vacancy
    from cv_agent.search import GetInItSearch

    gi = GetInItSearch()
    gi._get = lambda *a, **k: FakeResponse(
        status_code=200,
        text=(
            '<script id="__NEXT_DATA__" type="application/json">'
            '{"props":{"initialState":{"jobJob":{"job":{"content":'
            '"<p>Python Backend</p>"}}}}}</script>'
        ),
        url="https://www.get-in-it.de/jobsuche/p311292",
        history=[object()],
    )
    v = Vacancy(
        "gi-311292",
        "t",
        "c",
        "l",
        "https://www.get-in-it.de/jobsuche/p311292",
        "get-in-it",
    )
    assert gi.detail(v) == "Python Backend"


def test_getinit_exists_false_on_company_redirect():
    from cv_agent.models import Vacancy
    from cv_agent.search import GetInItSearch

    gi = GetInItSearch()
    gi._get = lambda *a, **k: FakeResponse(
        status_code=200,
        text="company",
        url="https://www.get-in-it.de/unternehmen/deloitte-gmbh",
        history=[object()],
    )
    v = Vacancy(
        "gi-289407",
        "t",
        "c",
        "l",
        "https://www.get-in-it.de/jobsuche/p289407",
        "get-in-it",
    )
    assert gi.exists(v) is False


def test_getinit_exists_keeps_direct_job_page():
    from cv_agent.models import Vacancy
    from cv_agent.search import GetInItSearch

    gi = GetInItSearch()
    gi._get = lambda *a, **k: FakeResponse(
        status_code=200,
        text="job",
        url="https://www.get-in-it.de/jobsuche/p311292",
    )
    v = Vacancy(
        "gi-311292",
        "t",
        "c",
        "l",
        "https://www.get-in-it.de/jobsuche/p311292",
        "get-in-it",
    )
    assert gi.exists(v) is True
