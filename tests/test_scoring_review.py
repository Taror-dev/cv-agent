import json
import sqlite3

import pytest

from cv_agent import config
from cv_agent.models import Vacancy
from cv_agent.scoring_benchmark import load_dataset
from cv_agent.scoring_config import ScoringConfig
from cv_agent.scoring_inputs import collect_scoring_inputs
from cv_agent.scoring_review import build_packet, main
from cv_agent.store import Store


def setup_account(root, name, count=8):
    store = Store(user=name)
    store.create_account(name)
    folder = root / "cv" / name
    folder.mkdir(parents=True)
    (folder / "Curriculum.txt").write_text("Python\nREST API implementation.\nDeutsch B2", encoding="utf-8")
    collect_scoring_inputs(store)
    assert store.set_document_type("Curriculum.txt", "cv")
    store.set_candidate_profile({"languages": ["English B2"], "scoring_preferences": {"work_mode": "any", "cities": []}})
    for i in range(count):
        store.upsert(Vacancy(f"{name}-{i}", f"Developer {i}", "Example", "Berlin", "https://example.test/job", "ba" if i % 2 else "linkedin",
                             description=f"Aufgaben:\nImplement REST API number {i}.\n" + "Python experience required. " * 4))
    store.close()
    return folder


def snapshot():
    with sqlite3.connect(config.DATA_DIR / "applications.db") as db:
        return "\n".join(db.iterdump())


def test_export_is_readonly_blinded_account_scoped_and_deterministic(isolated_project, monkeypatch):
    folder = setup_account(isolated_project, "One")
    setup_account(isolated_project, "Two")
    (folder / "ignored.txt").symlink_to(isolated_project / "cv" / "Two" / "Curriculum.txt")
    before = snapshot()
    monkeypatch.setattr("cv_agent.store.Store", lambda *a, **kw: pytest.fail("Exporter must not instantiate Store"))
    data, manifest = build_packet(count=6, user="One")
    assert snapshot() == before
    assert data == build_packet(count=6, user="One")[0]
    assert set(data["judgments"]["candidate-1"].values()) == {None}
    assert len(data["vacancies"]) == 6
    assert {job["split"] for job in data["vacancies"].values()} == {"calibration", "holdout"}
    assert all(item["refnr"].startswith("One-") for item in manifest["sources"].values())
    assert {item["source"] for item in manifest["sources"].values()} == {"ba", "linkedin"}
    assert data["profiles"]["candidate-1"]["documents"][0]["kind"] == "cv"
    assert data["profiles"]["candidate-1"]["documents"][-1]["text"] == "English B2"
    assert all(item["filename"] != "ignored.txt" for item in manifest["documents"])
    assert "score" not in json.dumps(data) and "score" not in json.dumps(manifest["sources"])


def test_duplicate_and_related_jobs_cannot_leak_between_splits(isolated_project):
    setup_account(isolated_project, "One")
    store = Store(user="One")
    original = store.by_refnr("One-0")
    store.upsert(Vacancy("duplicate", "Renamed duplicate", "Other", "Berlin", "", "ba", description=original.description.upper()))
    store.upsert(Vacancy("family", original.title, original.company, "Hamburg", "", "ba", description=original.description + " Additional task."))
    store.close()
    data, manifest = build_packet(count=9, user="One")
    assert manifest["selection"]["available_unique"] == 9
    groups = {}
    for row in data["vacancies"].values():
        assert groups.setdefault(row["group"], row["split"]) == row["split"]
    assert len(groups) == 8


def test_budget_and_missing_account_fail_without_writes(isolated_project):
    setup_account(isolated_project, "One")
    data, manifest = build_packet(count=4, user="One", limits=ScoringConfig(per_file=10, total_chars=10))
    assert sum(len(item["text"]) for item in data["profiles"]["candidate-1"]["documents"]) == 10
    assert manifest["warnings"]
    before = snapshot()
    with pytest.raises(ValueError, match="does not exist"):
        build_packet(count=4, user="Missing")
    with pytest.raises(ValueError, match="requested"):
        build_packet(count=100, user="One")
    assert snapshot() == before


def test_cli_writes_private_packet_and_rejects_reuse(isolated_project, monkeypatch):
    setup_account(isolated_project, "One", 60)
    monkeypatch.setenv("CV_AGENT_USER", "One")
    target = isolated_project / "review"
    assert main(["--count", "50", "--output-dir", str(target)]) == 0
    assert target.stat().st_mode & 0o777 == 0o700
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in target.iterdir())
    assert len(load_dataset(target / "dataset.yaml", allow_unreviewed=True)["vacancies"]) == 50
    with pytest.raises(ValueError, match="pending human"):
        load_dataset(target / "dataset.yaml")
    with pytest.raises(SystemExit) as exc:
        main(["--count", "50", "--output-dir", str(target)])
    assert exc.value.code == 2


def test_uuid_layout_uses_only_current_account_documents(isolated_project):
    from cv_agent.storage import apply_storage_migration
    setup_account(isolated_project, "One")
    store = Store(user="One")
    apply_storage_migration(store)
    store.close()
    data, _ = build_packet(count=4, user="One")
    assert data["profiles"]["candidate-1"]["documents"][0]["kind"] == "cv"


def test_export_refuses_production_job_environment(monkeypatch):
    monkeypatch.setenv("CV_AGENT_RUN_ID", "123")
    with pytest.raises(ValueError, match="production jobs"):
        build_packet()


def test_export_honors_configured_algorithmic_document_limits(isolated_project):
    setup_account(isolated_project, "One")
    with (isolated_project / "criteria.yaml").open("a", encoding="utf-8") as handle:
        handle.write("\nscoring:\n  per_file: 12\n  total_chars: 15\n")
    data, manifest = build_packet(count=4, user="One")
    assert manifest["document_limits"] == {"per_file": 12, "total_chars": 15}
    assert sum(len(item["text"]) for item in data["profiles"]["candidate-1"]["documents"]) == 15
