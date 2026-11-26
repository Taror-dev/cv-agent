from types import SimpleNamespace

import pytest

from cv_agent.models import Vacancy
from cv_agent.search import (
    SearchAdapter,
    SearchQuery,
    dedup,
    location_matches,
    normalize_keywords,
    run_adapter,
    select_getinit_categories,
)


def _vacancy(
    refnr,
    title,
    company="ACME GmbH",
    location="Wangen im Allgäu",
    source="get-in-it",
):
    return Vacancy(
        refnr,
        title,
        company,
        location,
        f"https://example.test/jobs/{refnr}",
        source,
    )


def test_keywords_are_unicode_normalized_deduplicated_and_capped():
    selected = normalize_keywords(
        [" Python ", "PYTHON", "AI   Engineer", "", "ＳＱＬ"],
        max_keywords=3,
    )
    assert selected.keywords == ["Python", "AI Engineer", "SQL"]
    assert selected.input_count == 5
    assert selected.duplicate_count == 1
    assert selected.omitted_count == 0

    capped = normalize_keywords(["A", "B", "C"], max_keywords=2)
    assert capped.keywords == ["A", "B"]
    assert capped.omitted_count == 1


def test_getinit_categories_come_from_profile_or_explicit_settings():
    assert select_getinit_categories(["Python", "PostgreSQL", "Docker"]) == [
        "Anwendungsentwicklung",
        "Datenbankentwicklung/BI",
        "System Engineering / Admin",
    ]
    assert select_getinit_categories(
        ["Python"], ["IT-Security", "Embedded Systems"]
    ) == ["IT-Security", "Embedded Systems"]
    with pytest.raises(ValueError, match="Unbekannte"):
        select_getinit_categories([], ["Alles"])


class ManyAdapter(SearchAdapter):
    source = "get-in-it"

    def __init__(self, *, geo=False):
        self.local_geo_filter = geo
        self.query_limits = []

    def search(self, query):
        self.query_limits.append(query.limit)
        return [
            _vacancy(
                f"gi-{query.keyword}-{index}",
                f"{query.keyword} Engineer {index}",
                company=f"Company {query.keyword} {index}",
                source=self.source,
            )
            for index in range(query.limit)
        ]

    def detail(self, vacancy):
        return "Beschreibung"

    def exists(self, vacancy):
        return True


def test_run_adapter_enforces_true_total_limit_across_queries():
    adapter = ManyAdapter()
    result = run_adapter(
        adapter,
        ["Python", "SQL", "Docker"],
        "Wangen im Allgäu",
        50,
        2,
        total_limit=3,
    )
    assert len(result.vacancies) == 3
    assert result.queries == 2
    assert adapter.query_limits == [2, 1]


def test_local_geo_filter_keeps_requested_city_and_remote_only():
    assert location_matches("88239 Wangen im Allgäu", "88239 Wangen im Allgäu")
    assert location_matches(
        "Wangen im Allgäu, Baden-Württemberg", "88239 Wangen im Allgäu"
    )
    assert location_matches("bundesweit; München", "88239 Wangen im Allgäu")
    assert not location_matches("München, Deutschland", "88239 Wangen im Allgäu")
    assert not location_matches("", "88239 Wangen im Allgäu")

    class GeoAdapter(ManyAdapter):
        def search(self, query):
            return [
                _vacancy("gi-local", "Python", location="Wangen im Allgäu"),
                _vacancy("gi-far", "Java", location="München, Deutschland"),
                _vacancy("gi-remote", "SQL", location="deutschlandweit"),
            ]

    result = run_adapter(
        GeoAdapter(geo=True),
        ["Python"],
        "88239 Wangen im Allgäu",
        50,
        25,
    )
    assert [vacancy.refnr for vacancy in result.vacancies] == [
        "gi-local",
        "gi-remote",
    ]


def test_geo_filter_adapter_fetches_once_for_multiple_locations():
    """Аудит/R4: запросы geo-источника не дублируются по локациям."""

    calls = []

    class CountingGeoAdapter(ManyAdapter):
        def __init__(self):
            super().__init__(geo=True)

        def queries(self, keywords, location, radius, limit):
            calls.append(location)
            return super().queries(keywords, location, radius, limit)

        def search(self, query):
            return [
                _vacancy("gi-muc", "A", location="München"),
                _vacancy("gi-ham", "B", location="Hamburg"),
                _vacancy("gi-far", "C", location="Köln"),
            ]

    result = run_adapter(
        CountingGeoAdapter(),
        ["Python"],
        ["München", "Hamburg"],
        50,
        25,
    )
    assert len(calls) == 1  # один набор запросов на все локации
    assert [v.refnr for v in result.vacancies] == ["gi-muc", "gi-ham"]
    assert result.queries == 1


def test_non_geo_adapter_runs_queries_per_location():
    calls = []

    class PerLocationAdapter(ManyAdapter):  # local_geo_filter=False
        def queries(self, keywords, location, radius, limit):
            calls.append(location)
            return super().queries(keywords, location, radius, limit)

        def search(self, query):
            return [_vacancy(f"x-{query.location}", "A", location=query.location)]

    result = run_adapter(
        PerLocationAdapter(), ["Python"], ["München", "Hamburg"], 50, 25
    )
    assert calls == ["München", "Hamburg"]
    assert sorted(v.refnr for v in result.vacancies) == ["x-Hamburg", "x-München"]
    assert result.queries == 2


def test_dedup_normalizes_fields_and_honors_similarity_threshold():
    first = _vacancy("gi-1", "Python Developer", "ACME GmbH", "88239 Wangen")
    similar = _vacancy("gi-2", "Python Engineer", "acme", "Wangen")
    assert dedup([first, similar], similarity=0.7) == [first]
    assert dedup([first, similar], similarity=0.8) == [first, similar]


def test_dedup_compares_new_vacancies_with_account_stored_rows():
    stored = _vacancy(
        "gi-old",
        "Senior Python Entwickler",
        "ACME",
        "Wangen im Allgäu",
    )
    duplicate = _vacancy(
        "gi-new",
        "Senior Python Entwickler (m/w/d)",
        "Acme GmbH",
        "88239 Wangen im Allgäu",
    )
    other_city = _vacancy(
        "gi-other",
        "Senior Python Entwickler (m/w/d)",
        "Acme GmbH",
        "München",
    )
    assert dedup([duplicate, other_city], existing=[stored]) == [other_city]


def test_cmd_search_applies_total_limit_across_all_sources(monkeypatch, capsys):
    import cv_agent.cli as cli

    adapters = {
        source: ManyAdapter()
        for source in ("arbeitsagentur", "stepstone", "linkedin", "get-in-it")
    }
    for source, adapter in adapters.items():
        adapter.source = source

    monkeypatch.setattr(cli, "BASearch", lambda: adapters["arbeitsagentur"])
    monkeypatch.setattr(cli, "StepStoneSearch", lambda: adapters["stepstone"])
    monkeypatch.setattr(cli, "LinkedInSearch", lambda: adapters["linkedin"])
    monkeypatch.setattr(cli, "GetInItSearch", lambda: adapters["get-in-it"])
    monkeypatch.setattr(
        cli,
        "load_criteria",
        lambda: SimpleNamespace(
            keywords=["Python", "SQL"],
            region="Wangen",
            radius_km=50,
            search_similarity_threshold=0.9,
        ),
    )
    stored = []

    class FakeStore:
        def add(self, vacancy):
            stored.append(vacancy.refnr)

        def vacancies_for_dedup(self):
            return []

        def close(self):
            pass

    monkeypatch.setattr(cli, "Store", FakeStore)
    monkeypatch.setattr(cli, "_prune_stale", lambda store, ba, fresh: 0)
    cli.cmd_search(SimpleNamespace(total_limit=3, per_query_limit=2, max_keywords=2))

    output = capsys.readouterr().out
    assert "max_queries=8" in output
    assert "max_stored_checks=0" in output
    assert "per_query_limit=2, total_limit=3" in output
    assert len(stored) == 3
    assert adapters["arbeitsagentur"].query_limits == [2, 1]
    assert all(
        adapter.query_limits == []
        for source, adapter in adapters.items()
        if source != "arbeitsagentur"
    )
    assert all(f"Source {source}:" in output for source in adapters)


def test_cmd_search_rejects_zero_limits_before_opening_store(monkeypatch):
    import cv_agent.cli as cli

    monkeypatch.setattr(
        cli,
        "load_criteria",
        lambda: SimpleNamespace(
            keywords=["Python"], region="Wangen", radius_km=50
        ),
    )
    monkeypatch.setattr(
        cli,
        "Store",
        lambda: pytest.fail("Store darf bei ungültigen Limits nicht geöffnet werden"),
    )
    with pytest.raises(ValueError, match="total_limit"):
        cli.cmd_search(
            SimpleNamespace(total_limit=0, per_query_limit=25, max_keywords=12)
        )


def _location_capture_adapter(locations):
    class LocationAdapter(ManyAdapter):
        source = "arbeitsagentur"

        def search(self, query):
            locations.append(query.location)
            return super().search(query)

    return LocationAdapter()


def _patch_cmd_search(monkeypatch, adapter, region="Wangen"):
    import cv_agent.cli as cli

    monkeypatch.setattr(cli, "BASearch", lambda: adapter)
    monkeypatch.setattr(cli, "StepStoneSearch", lambda: ManyAdapter())
    monkeypatch.setattr(cli, "LinkedInSearch", lambda: ManyAdapter())
    monkeypatch.setattr(cli, "GetInItSearch", lambda: ManyAdapter())
    monkeypatch.setattr(
        cli,
        "load_criteria",
        lambda: SimpleNamespace(
            keywords=["Python"],
            region=region,
            radius_km=50,
            search_similarity_threshold=0.9,
        ),
    )

    class FakeStore:
        def add(self, vacancy):
            pass

        def vacancies_for_dedup(self):
            return []

        def close(self):
            pass

    monkeypatch.setattr(cli, "Store", FakeStore)
    monkeypatch.setattr(cli, "_prune_stale", lambda store, ba, fresh: 0)


def test_cmd_search_location_override_replaces_profile_region(monkeypatch, capsys):
    import cv_agent.cli as cli

    seen_locations = []
    _patch_cmd_search(monkeypatch, _location_capture_adapter(seen_locations))
    cli.cmd_search(
        SimpleNamespace(
            total_limit=5, per_query_limit=25, max_keywords=12,
            location="München",
        )
    )
    assert seen_locations == ["München"]
    assert "location=München" in capsys.readouterr().out


def test_cmd_search_comma_separated_locations_run_each_once(monkeypatch, capsys):
    import cv_agent.cli as cli

    seen_locations = []
    adapter = _location_capture_adapter(seen_locations)
    _patch_cmd_search(monkeypatch, adapter)
    cli.cmd_search(
        SimpleNamespace(
            total_limit=10, per_query_limit=5, max_keywords=12,
            location="München, Hamburg, München",
        )
    )
    assert seen_locations == ["München", "Hamburg"]
    assert "location=München, Hamburg" in capsys.readouterr().out


def test_cmd_search_empty_location_override_means_no_geo_limit(monkeypatch, capsys):
    import cv_agent.cli as cli

    seen_locations = []
    _patch_cmd_search(monkeypatch, _location_capture_adapter(seen_locations))
    cli.cmd_search(
        SimpleNamespace(
            total_limit=5, per_query_limit=25, max_keywords=12,
            location="",
        )
    )
    assert seen_locations == [""]
    assert "location=*" in capsys.readouterr().out


def test_cmd_search_without_flag_keeps_profile_region(monkeypatch, capsys):
    import cv_agent.cli as cli

    seen_locations = []
    _patch_cmd_search(monkeypatch, _location_capture_adapter(seen_locations))
    cli.cmd_search(
        SimpleNamespace(total_limit=5, per_query_limit=25, max_keywords=12)
    )
    assert seen_locations == ["Wangen"]


def test_cmd_search_does_not_save_excluded_or_agency(monkeypatch, capsys):
    import cv_agent.cli as cli
    from cv_agent.models import Vacancy

    class SaveAdapter(SearchAdapter):
        source = "arbeitsagentur"

        def queries(self, keywords, location, radius, limit):
            return [SearchQuery(k, location, radius, limit, label=k) for k in keywords]

        def search(self, query):
            return [
                Vacancy("ok-1", "Python Dev", "ACME", "Wangen",
                                  "https://example.test/ok-1", "arbeitsagentur"),
                Vacancy("exc-1", "Zeitarbeit Dev", "ACME", "Wangen",
                                  "https://example.test/exc-1", "arbeitsagentur"),
                Vacancy("agency-1", "Dev", "FERCHAU Engineering", "Wangen",
                                  "https://example.test/agency-1", "arbeitsagentur"),
            ]

        def detail(self, vacancy):
            return ""

        def exists(self, vacancy):
            return True

    saved = []

    class FakeStore:
        def add(self, v):
            saved.append(v.refnr)

        def vacancies_for_dedup(self):
            return []

        def close(self):
            pass

    class EmptyAdapter(SearchAdapter):
        source = "stepstone"

        def queries(self, keywords, location, radius, limit):
            return []

        def search(self, query):
            return []

        def detail(self, vacancy):
            return ""

        def exists(self, vacancy):
            return True

    monkeypatch.setattr(cli, "BASearch", lambda: SaveAdapter())
    monkeypatch.setattr(cli, "StepStoneSearch", lambda: EmptyAdapter())
    monkeypatch.setattr(cli, "LinkedInSearch", lambda: EmptyAdapter())
    monkeypatch.setattr(cli, "GetInItSearch", lambda: EmptyAdapter())
    monkeypatch.setattr(
        cli,
        "load_criteria",
        lambda: SimpleNamespace(
            keywords=["Python"],
            region="Wangen",
            radius_km=50,
            search_similarity_threshold=0.9,
            exclusions={},
            exclude_keywords=["Zeitarbeit"],
            recruiting_agencies=["FERCHAU"],
            recruiting_agency_keywords=[],
        ),
    )
    monkeypatch.setattr(cli, "Store", FakeStore)
    monkeypatch.setattr(cli, "_prune_stale", lambda store, ba, fresh: 0)

    cli.cmd_search(SimpleNamespace(total_limit=10, per_query_limit=25, max_keywords=5))

    assert saved == ["ok-1"]
    out = capsys.readouterr().out
    assert "Not saved (filters): 2" in out
