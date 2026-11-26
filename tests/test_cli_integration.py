"""Аудит M1: интеграционные тесты CLI-команд с fake LLM (без сети)."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cv_agent import cli
from cv_agent import config as cfg_module
from cv_agent.models import Vacancy
from cv_agent.store import Store


def _root() -> Path:
    return Path(cfg_module.DATA_DIR).parent


@pytest.fixture(autouse=True)
def _redirect_outputs(monkeypatch):
    """CLI пишет письма/досье только во временный проект теста."""
    apps = _root() / "applications"
    import cv_agent.apply as apply_module
    import cv_agent.write as write_module

    monkeypatch.setattr(write_module, "APPLICATIONS_DIR", apps)
    monkeypatch.setattr(apply_module, "APPLICATIONS_DIR", apps)


class FakeCfg:
    provider = "ollama"
    model = "fake-model"


class FakeLLM:
    cfg = FakeCfg()

    def __init__(self, text="Sehr geehrte Damen und Herren,\n\nIlia Rerich"):
        self._text = text

    def complete(self, system, prompt):
        return self._text


@pytest.fixture()
def vacancy_with_description():
    store = Store()
    try:
        store.add(
            Vacancy(
                "int-1",
                "Python Entwickler",
                "ACME GmbH",
                "Wangen",
                "https://example.test/int-1",
                "arbeitsagentur",
            )
        )
        store.update_description(
            "int-1", "Wir suchen Python-Entwickler.", edited=False
        )
        store.set_candidate_profile({"full_name": "Илья Рерих"})
    finally:
        store.close()


def test_cmd_write_generates_letter_and_records_artifact(
    monkeypatch, capsys, vacancy_with_description
):
    monkeypatch.setattr(cli, "_llm_client_for", lambda criteria: FakeLLM())
    monkeypatch.setattr(cli, "_description_for", lambda store, v, criteria: None)

    cli.cmd_write(SimpleNamespace(refnr="int-1"))

    out = capsys.readouterr().out
    assert "Anschreiben" in out
    letters = list(
        (_root() / "applications").glob("Anschreiben_ACME_GmbH_int-1.pdf")
    )
    assert letters and letters[0].read_bytes().startswith(b"%PDF")

    store = Store()
    try:
        n = store.conn.execute(
            "SELECT COUNT(*) FROM artifacts WHERE kind='letter' "
            "AND status='succeeded'"
        ).fetchone()[0]
        assert n == 1
    finally:
        store.close()


def test_cmd_apply_prepared_then_confirm_applied(
    monkeypatch, vacancy_with_description
):
    monkeypatch.setattr(
        cli,
        "_llm_client_for",
        lambda criteria: FakeLLM("Mit freundlichen Grüßen,\nИлья Рерих"),
    )
    monkeypatch.setattr(cli, "_description_for", lambda store, v, criteria: None)

    cli.cmd_apply(SimpleNamespace(refnr="int-1"))

    store = Store()
    try:
        assert store.application_status("int-1") == "prepared"
    finally:
        store.close()

    dossiers = sorted(
        p
        for p in (_root() / "applications").iterdir()
        if p.is_dir() and p.name.startswith("ACME")
    )
    assert len(dossiers) == 1
    assert (dossiers[0] / "Anschreiben.pdf").is_file()
    assert (dossiers[0] / "Lebenslauf.pdf").is_file()
    assert (dossiers[0] / "stelle.txt").read_text(encoding="utf-8").startswith(
        "Python Entwickler"
    )

    cli.cmd_confirm_applied(SimpleNamespace(refnr="int-1"))
    store = Store()
    try:
        assert store.application_status("int-1") == "applied"
    finally:
        store.close()


def test_cmd_write_empty_llm_answer_is_error(monkeypatch, vacancy_with_description):
    monkeypatch.setattr(cli, "_llm_client_for", lambda criteria: FakeLLM("   "))
    monkeypatch.setattr(cli, "_description_for", lambda store, v, criteria: None)

    with pytest.raises(SystemExit, match="LLM error"):
        cli.cmd_write(SimpleNamespace(refnr="int-1"))
    # пустое письмо не сохраняется
    assert not list((_root() / "applications").glob("Anschreiben_*.md"))


def test_migrate_storage_plan_and_apply(capsys):
    args_plan = SimpleNamespace(apply=False)
    cli.cmd_migrate_storage(args_plan)
    report = json.loads(capsys.readouterr().out)
    assert report["already_active"] is False
    assert any("Илья" in a["name"] for a in report["accounts"])
    assert report["documents"], "legacy CV должны попасть в план"
    marker = _root() / "data" / "storage-layout.json"
    assert not marker.exists()

    cli.cmd_migrate_storage(SimpleNamespace(apply=True))
    report = json.loads(capsys.readouterr().out)
    assert report.get("applied_at")
    assert marker.is_file()

    from cv_agent.storage import storage_layout_active

    assert storage_layout_active(_root() / "data")

    cli.cmd_migrate_storage(SimpleNamespace(apply=True))
    report2 = json.loads(capsys.readouterr().out)
    assert report2["already_active"] is True


def test_purge_archived_cli_keeps_applied_deletes_stale(capsys):
    store = Store()
    try:
        store.add(Vacancy("purge-a", "T", "C", "L", "u", "arbeitsagentur"))
        store.add(Vacancy("purge-fresh", "T", "C", "L", "u", "arbeitsagentur"))
        store.mark_unavailable("purge-a")
        store.mark_unavailable("purge-fresh")
        store.conn.execute(
            "UPDATE vacancies SET archived_at=datetime('now','-40 days') "
            "WHERE refnr='purge-a'"
        )  # purge-fresh остаётся свежим
        store.conn.commit()
    finally:
        store.close()

    with pytest.raises(SystemExit):
        cli.cmd_purge_archived(SimpleNamespace(confirm=False, older_than_days=30))

    cli.cmd_purge_archived(SimpleNamespace(confirm=True, older_than_days=30))
    out = capsys.readouterr().out
    assert "deleted: 1" in out

    store = Store()
    try:
        assert store.by_refnr("purge-a") is None
        assert store.by_refnr("purge-fresh") is not None

        # applied-вакансия не удаляется независимо от возраста
        store.add(Vacancy("purge-applied", "T", "C", "L", "u", "arbeitsagentur"))
        store.set_status("purge-applied", "applied")
        store.mark_unavailable("purge-applied")
        store.conn.execute(
            "UPDATE vacancies SET archived_at=datetime('now','-400 days') "
            "WHERE refnr='purge-applied'"
        )
        store.conn.commit()
    finally:
        store.close()

    cli.cmd_purge_archived(SimpleNamespace(confirm=True, older_than_days=30))
    store = Store()
    try:
        assert store.by_refnr("purge-applied") is not None
    finally:
        store.close()
