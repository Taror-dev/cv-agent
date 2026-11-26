from types import SimpleNamespace

import pytest
import requests

from cv_agent.models import Vacancy
from cv_agent.search import (
    BASearch,
    GETINIT_CATEGORIES,
    GetInItSearch,
    HTTPSourceAdapter,
    LinkedInSearch,
    SearchAdapter,
    SearchQuery,
    SourceError,
    StepStoneSearch,
    normalize_url,
    run_adapter,
    stable_url_ref,
    validate_vacancy,
)


class FakeResponse:
    def __init__(self, status_code=200, data=None, text=""):
        self.status_code = status_code
        self._data = data or {}
        self.text = text

    def json(self):
        return self._data

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}", response=self)


def test_all_sources_implement_common_contract():
    for adapter_class in (BASearch, StepStoneSearch, LinkedInSearch, GetInItSearch):
        adapter = adapter_class()
        assert isinstance(adapter, SearchAdapter)
        assert callable(adapter.search)
        assert callable(adapter.detail)
        assert callable(adapter.exists)
        assert callable(adapter.refresh)


def test_get_in_it_exposes_only_profile_relevant_categories():
    queries = GetInItSearch().queries(["Python", "SQL"], "Wangen", 50, 25)
    assert {query.label for query in queries} == {
        "Anwendungsentwicklung",
        "Datenbankentwicklung/BI",
    }
    assert {query.category_id for query in queries} < set(GETINIT_CATEGORIES.values())
    assert all(query.label for query in queries)


@pytest.mark.parametrize(
    ("source", "refnr"),
    [
        ("arbeitsagentur", ""),
        ("stepstone", "ss-"),
        ("get-in-it", "gi-None"),
        ("linkedin", "li-"),
    ],
)
def test_invalid_or_unstable_identifiers_are_skipped(source, refnr):
    vacancy = Vacancy(refnr, "Job", "ACME", "Wangen", "https://example.test/job", source)
    valid, reason = validate_vacancy(vacancy, source)
    assert valid is None
    assert reason in {"invalid_refnr", "unstable_refnr"}


def test_validation_normalizes_fields_and_rejects_bad_url():
    vacancy = Vacancy(
        " ref-1 ",
        "  AI   Engineer ",
        " ACME  GmbH ",
        " Wangen ",
        "HTTPS://EXAMPLE.TEST//jobs/1/?tracking=1#top",
        "linkedin",
    )
    valid, reason = validate_vacancy(vacancy, "linkedin")
    assert reason == ""
    assert valid.title == "AI Engineer"
    assert valid.company == "ACME GmbH"
    assert valid.url == "https://example.test/jobs/1"

    invalid, reason = validate_vacancy(
        Vacancy("1", "Job", "ACME", "x", "javascript:alert(1)", "linkedin"),
        "linkedin",
    )
    assert invalid is None
    assert reason == "invalid_url"


def test_linkedin_fallback_id_is_stable_hash_of_normalized_url():
    html = """
    <div class="base-search-card" data-entity-urn="">
      <a class="base-card__full-link" href="HTTPS://DE.LINKEDIN.COM//jobs/view/test-1/?trk=x#fragment"></a>
      <h3 class="base-search-card__title">AI Engineer</h3>
    </div>
    """
    first = LinkedInSearch._parse(html)[0]
    second = LinkedInSearch._parse(html)[0]
    assert first.refnr == second.refnr
    assert first.refnr == stable_url_ref("li", first.url)
    assert first.refnr.startswith("li-url-")


def test_missing_stepstone_and_getinit_ids_use_stable_url_fallback():
    stepstone_html = """
    <script>
    window.__PRELOADED_STATE__["app-unifiedResultlist"] =
    {"searchResults":{"items":[{"title":"AI","url":"/jobs/ai-role"}]}};
    </script>
    """
    stepstone = StepStoneSearch._parse_embedded(stepstone_html)[0]
    assert stepstone.refnr.startswith("ss-url-")

    # hash-fallback остаётся для редких ссылок на объявление без id;
    # ссылки не на вакансию (например, страница компании) отбрасываются
    getinit = GetInItSearch._parse(
        {"items": {"results": [{"title": "AI", "url": "/jobsuche/p12345"}]}}
    )[0]
    assert getinit.refnr.startswith("gi-url-")
    assert (
        GetInItSearch._parse(
            {
                "items": {
                    "results": [
                        {"title": "AI", "url": "/unternehmen/acme"},
                        {"title": "AI"},
                    ]
                }
            }
        )
        == []
    )


def test_http_retry_handles_network_and_5xx_with_bounded_backoff(monkeypatch):
    adapter = BASearch()
    responses = iter(
        [
            requests.ConnectionError("offline"),
            FakeResponse(503),
            FakeResponse(200, {"ergebnisliste": []}),
        ]
    )
    calls = []
    waits = []

    def fake_get(*args, **kwargs):
        calls.append(1)
        result = next(responses)
        if isinstance(result, Exception):
            raise result
        return result

    adapter.session.get = fake_get
    monkeypatch.setattr("cv_agent.search.time.sleep", waits.append)
    assert adapter.search(SearchQuery("Python", "Wangen", 50, 25)) == []
    assert len(calls) == 3
    assert waits == [0.5, 1.0]


def test_http_attempt_uses_shared_network_pool(monkeypatch):
    adapter = BASearch()
    entered = []

    class Limit:
        def __enter__(self):
            entered.append("enter")

        def __exit__(self, *exc):
            entered.append("exit")

    monkeypatch.setattr("cv_agent.search.network_request_limit", Limit)

    def fake_get(*args, **kwargs):
        assert entered == ["enter"]
        return FakeResponse(200, {"ergebnisliste": []})

    adapter.session.get = fake_get
    assert adapter.search(SearchQuery("Python", "Wangen", 50, 25)) == []
    assert entered == ["enter", "exit"]


def test_http_4xx_is_normalized_and_not_retried():
    adapter = BASearch()
    calls = []

    def fake_get(*args, **kwargs):
        calls.append(1)
        return FakeResponse(400)

    adapter.session.get = fake_get
    with pytest.raises(SourceError) as exc:
        adapter.search(SearchQuery("Python", "Wangen", 50, 25))
    assert exc.value.status_code == 400
    assert exc.value.retryable is False
    assert len(calls) == 1
    assert "HTTP 400" in str(exc.value)


class PartialAdapter(SearchAdapter):
    source = "linkedin"

    def search(self, query):
        if query.keyword == "broken":
            raise SourceError(self.source, "search", "HTTP 503", retryable=True)
        return [
            Vacancy(
                "li-1",
                "AI Engineer",
                "ACME",
                "Wangen",
                "https://example.test/job/1",
                self.source,
            ),
            Vacancy("li-", "Invalid", "ACME", "Wangen", "https://example.test/bad", self.source),
        ]

    def detail(self, vacancy):
        return "description"

    def exists(self, vacancy):
        return True


def test_each_query_is_isolated_and_summary_counts_results():
    result = run_adapter(
        PartialAdapter(),
        ["broken", "Python", "duplicate"],
        "Wangen",
        50,
        25,
    )
    assert result.queries == 3
    assert [vacancy.refnr for vacancy in result.vacancies] == ["li-1"]
    assert result.skipped == 3  # invalid in two successful queries + one duplicate
    assert len(result.errors) == 1
    assert "Query 'broken'" in str(result.errors[0])
    assert "found=1" in result.summary()
    assert "skipped=3" in result.summary()
    assert "errors=1" in result.summary()


def test_refresh_uses_common_detail_contract():
    vacancy = Vacancy("li-1", "AI", "ACME", "x", "https://example.test/1", "linkedin")
    assert PartialAdapter().refresh(vacancy).description == "description"


def test_normalize_url_is_deterministic():
    assert normalize_url("HTTPS://EXAMPLE.TEST//a///b/?x=1#y") == "https://example.test/a/b"


def test_cmd_search_prints_summary_for_every_source(monkeypatch, capsys):
    import cv_agent.cli as cli

    class OneResultAdapter(PartialAdapter):
        def __init__(self, source):
            self.source = source

        def search(self, query):
            return [
                Vacancy(
                    f"{self.source}-1",
                    f"{self.source} Job",
                    self.source,
                    "Wangen",
                    f"https://example.test/{self.source}/1",
                    self.source,
                )
            ]

    adapters = {
        "arbeitsagentur": OneResultAdapter("arbeitsagentur"),
        "stepstone": OneResultAdapter("stepstone"),
        "linkedin": OneResultAdapter("linkedin"),
        "get-in-it": OneResultAdapter("get-in-it"),
    }
    monkeypatch.setattr(cli, "BASearch", lambda: adapters["arbeitsagentur"])
    monkeypatch.setattr(cli, "StepStoneSearch", lambda: adapters["stepstone"])
    monkeypatch.setattr(cli, "LinkedInSearch", lambda: adapters["linkedin"])
    monkeypatch.setattr(cli, "GetInItSearch", lambda: adapters["get-in-it"])
    monkeypatch.setattr(
        cli,
        "load_criteria",
        lambda: SimpleNamespace(
            keywords=["Python"], region="Wangen", radius_km=50
        ),
    )
    stored = []

    class FakeStore:
        def add(self, vacancy):
            stored.append(vacancy.refnr)

        def close(self):
            pass

    monkeypatch.setattr(cli, "Store", FakeStore)
    monkeypatch.setattr(cli, "_prune_stale", lambda store, ba, fresh: 0)
    cli.cmd_search(SimpleNamespace(limit=25))
    output = capsys.readouterr().out
    for source in adapters:
        assert f"Source {source}:" in output
    assert "query errors: 0" in output
    assert len(stored) == 4


def test_run_adapter_isolates_unexpected_exception():
    """Аудит M2: не-SourceError исключение изолируется как ошибка запроса."""

    class ExplodingAdapter(SearchAdapter):
        source = "linkedin"

        def queries(self, keywords, location, radius, limit):
            return [
                SearchQuery(kw, location, radius, limit, label=kw)
                for kw in keywords
            ]

        def search(self, query):
            if query.keyword == "broken":
                raise ValueError("parser exploded")
            return [
                Vacancy(
                    f"li-{query.keyword}",
                    query.keyword,
                    "ACME",
                    "Wangen",
                    f"https://example.test/{query.keyword}",
                    self.source,
                )
            ]

        def detail(self, vacancy):
            return ""

        def exists(self, vacancy):
            return True

    result = run_adapter(ExplodingAdapter(), ["ok", "broken", "ok2"], "W", 50, 25)
    assert [v.refnr for v in result.vacancies] == ["li-ok", "li-ok2"]
    assert len(result.errors) == 1
    assert "unexpected adapter error" in str(result.errors[0])
    assert "ValueError" in str(result.errors[0])
