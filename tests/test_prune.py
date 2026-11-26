from unittest.mock import MagicMock

from cv_agent.cli import _prune_stale
from cv_agent.models import Vacancy
from cv_agent.store import Store


def _store_with(tmp_path, refnrs):
    s = Store(tmp_path / "t.db")
    for r in refnrs:
        s.add(Vacancy(r, "AI Engineer", "ACME", "Wangen", "http://x", "arbeitsagentur"))
    return s


def test_prune_archives_stale_without_deleting_history(tmp_path):
    s = _store_with(tmp_path, ["1", "2", "3"])
    ba = MagicMock()
    ba.exists.side_effect = lambda r: r in {"1", "3"}
    archived = _prune_stale(s, ba, fresh_refnrs={"9"})
    assert archived == 1
    assert {r[0] for r in s.list_all()} == {"1", "3"}
    assert s.by_refnr("2") is not None
    assert s.vacancy_lifecycle("2")["source_status"] == "unavailable"
    assert s.vacancy_lifecycle("2")["archived_at"]
    s.close()


def test_prune_keeps_fresh_refnrs(tmp_path):
    s = _store_with(tmp_path, ["1", "2"])
    ba = MagicMock()
    ba.exists.return_value = False
    archived = _prune_stale(s, ba, fresh_refnrs={"1"})
    assert archived == 1
    assert [r[0] for r in s.list_all()] == ["1"]
    assert s.by_refnr("2") is not None
    s.close()


def test_prune_ignores_errors(tmp_path):
    s = _store_with(tmp_path, ["1"])
    ba = MagicMock()
    ba.exists.side_effect = RuntimeError("network down")
    archived = _prune_stale(s, ba, fresh_refnrs=set())
    assert archived == 0
    assert len(s.list_all()) == 1
    assert s.vacancy_lifecycle("1")["source_status"] == "error"
    s.close()


def test_cmd_actualize_archives_removed_and_keeps_active(monkeypatch, tmp_path):
    import cv_agent.cli as cli

    s = Store(tmp_path / "actualize.db")
    s.add(Vacancy("li-1", "AI Engineer", "ACME", "Wangen", "http://li/1", "linkedin"))
    s.add(Vacancy("ba-1", "AI Engineer", "ACME", "Wangen", "http://ba/1", "arbeitsagentur"))
    s.add(Vacancy("ss-1", "AI Engineer", "ACME", "Wangen", "http://ss/1", "stepstone"))

    class FakeLinkedIn:
        def exists(self, vacancy):
            return vacancy.refnr == "li-1"

    class FakeBA:
        def exists(self, vacancy):
            return True

    class FakeStepStone:
        def exists(self, vacancy):
            raise RuntimeError("network down")

    monkeypatch.setitem(cli.ACTUALIZE_ADAPTERS, "linkedin", FakeLinkedIn)
    monkeypatch.setitem(cli.ACTUALIZE_ADAPTERS, "arbeitsagentur", FakeBA)
    monkeypatch.setitem(cli.ACTUALIZE_ADAPTERS, "stepstone", FakeStepStone)
    monkeypatch.setattr(cli, "Store", lambda: s)
    db_path = tmp_path / "actualize.db"

    cli.cmd_actualize(type("Args", (), {})())
    s.close()

    s = Store(db_path)
    # сбой источника обратим: вакансия остаётся активной
    assert [r[0] for r in s.list_all()] == ["li-1", "ba-1", "ss-1"]
    assert s.vacancy_lifecycle("ss-1")["source_status"] == "error"
    assert not s.vacancy_lifecycle("li-1")["archived_at"]
    s.close()


def test_cmd_actualize_marks_unsupported_source(monkeypatch, tmp_path):
    import cv_agent.cli as cli

    db_path = tmp_path / "unsupported.db"
    s = Store(db_path)
    s.add(Vacancy("in-1", "AI Engineer", "ACME", "Wangen", "http://x", "indeed"))
    monkeypatch.setattr(cli, "Store", lambda: s)

    cli.cmd_actualize(type("Args", (), {})())
    s.close()

    s = Store(db_path)
    assert len(s.list_all()) == 1  # unsupported остаётся нетронутым
    s.close()
