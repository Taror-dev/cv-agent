import json
import sqlite3

from bs4 import BeautifulSoup
import pytest

from cv_agent import config
from cv_agent.models import Vacancy
from cv_agent.scoring_config import ScoringConfig
from cv_agent.store import LATEST_SCHEMA_VERSION, Store
from cv_agent.vacancy_feedback import candidate_revision, capture_inputs, vacancy_revision
from cv_agent.webapp import app


@pytest.fixture
def feedback_client(monkeypatch):
    import cv_agent.webapp as web
    import cv_agent.llm as llm
    monkeypatch.setattr(web, "ensure_description", lambda *a, **kw: None)
    monkeypatch.setattr(llm, "LLMClient", lambda *a, **kw: pytest.fail("Feedback must not call a model"))
    monkeypatch.setattr(web, "LLMClient", lambda *a, **kw: pytest.fail("Feedback must not call a model"))
    store = Store()
    store.add(Vacancy("r1", "Developer", "Example", "Berlin", "https://example.test", "ba", description="Python REST API development. " * 5, score=81))
    store.create_account("Other")
    store.close()
    other = Store(user="Other")
    other.add(Vacancy("r1", "Other vacancy", "Example", "Berlin", "", "ba", description="Private other description." * 5))
    other.close()
    client = app.test_client()
    client.post("/login", data={"account": "Илья"})
    return client


def context(client, refnr="r1"):
    html = BeautifulSoup(client.get(f"/vacancy/{refnr}").data, "html.parser")
    return html.select_one('input[name="review_context"]')["value"]


def test_card_buttons_save_update_clear_and_preserve_automatic_score(feedback_client):
    client = feedback_client
    page = client.get("/vacancy/r1").data.decode()
    assert all(label in page for label in ("Подходит", "Сомнительно", "Не подходит"))
    for grade in ("2", "0", "1"):
        response = client.post("/vacancy/r1/review", data={"review_context": context(client), "grade": grade, "reason": "<script>explanation</script>"})
        assert response.status_code == 303
        store = Store()
        assert store.vacancy_review("r1")["grade"] == int(grade)
        assert store.by_refnr("r1").score == 81
        snapshot = json.loads(store.conn.execute("SELECT inputs_json FROM vacancy_reviews").fetchone()[0])
        assert snapshot["vacancy"]["text"].startswith("Python REST")
        assert snapshot["profile"]["documents"]
        assert store.conn.execute("SELECT count(*) FROM vacancy_reviews").fetchone()[0] == 1
        store.close()
        page = client.get(response.location).data.decode()
        assert "&lt;script&gt;explanation&lt;/script&gt;" in page
        assert "<script>explanation</script>" not in page
        assert f'value="{grade}" class="btn review-choice review-grade-{grade} selected" aria-pressed="true"' in page
    response = client.post("/vacancy/r1/review", data={"review_context": context(client), "action": "clear"})
    assert response.status_code == 303
    store = Store()
    assert store.vacancy_review("r1") is None
    assert store.by_refnr("r1").score == 81
    store.close()


@pytest.mark.parametrize("fields", [{"grade": "3"}, {"grade": "true"}, {"grade": ""}, {"grade": "2", "reason": "x" * 2001}, {"grade": "2", "action": "unknown"}])
def test_invalid_feedback_is_rejected_without_overwriting(feedback_client, fields):
    client = feedback_client
    token = context(client)
    client.post("/vacancy/r1/review", data={"review_context": token, "grade": "2", "reason": "Original"})
    response = client.post("/vacancy/r1/review", data={"review_context": token, **fields})
    assert response.status_code == 400
    store = Store()
    assert store.vacancy_review("r1")["reason"] == "Original"
    store.close()


def test_feedback_requires_login_valid_context_and_current_account(feedback_client):
    client = feedback_client
    token = context(client)
    for bad in ("", token + "tampered"):
        assert client.post("/vacancy/r1/review", data={"review_context": bad, "grade": "2"}).status_code == 403
    assert client.post("/vacancy/different/review", data={"review_context": token, "grade": "2"}).status_code == 403
    client.post("/vacancy/r1/review", data={"review_context": token, "grade": "2", "reason": "First account"})
    client.post("/logout")
    assert client.post("/vacancy/r1/review", data={"review_context": token, "grade": "0"}).status_code == 302
    client.post("/login", data={"account": "Other"})
    assert client.post("/vacancy/r1/review", data={"review_context": token, "action": "clear"}).status_code == 403
    assert "First account" not in client.get("/vacancy/r1").data.decode()
    client.post("/vacancy/r1/review", data={"review_context": context(client), "grade": "0", "reason": "Other account"})
    first, second = Store(), Store(user="Other")
    assert first.vacancy_review("r1")["grade"] == 2
    assert second.vacancy_review("r1")["grade"] == 0
    second.clear_vacancy_review("r1")
    assert first.vacancy_review("r1")["grade"] == 2
    first.close()
    second.close()


@pytest.mark.parametrize("change", ["description", "document", "language", "preferences", "type"])
def test_changes_warn_and_stale_forms_cannot_save(feedback_client, isolated_project, change):
    client = feedback_client
    token = context(client)
    assert client.post("/vacancy/r1/review", data={"review_context": token, "grade": "2"}).status_code == 303
    store = Store()
    if change == "description":
        store.update_description("r1", "An entirely different job description.")
    elif change == "document":
        (isolated_project / "cv" / "Илья" / "Lebenslauf.txt").write_text("New experience with Java.")
    elif change == "language":
        store.set_candidate_profile({"languages": ["Deutsch C1"]})
    elif change == "preferences":
        store.set_candidate_profile({"scoring_preferences": {"work_mode": "remote", "cities": []}})
    else:
        assert store.set_document_type("AI Engineer Program Curriculum.pdf", "cv")
    store.close()
    response = client.post("/vacancy/r1/review", data={"review_context": token, "grade": "0", "reason": "Draft comment"})
    assert response.status_code == 409
    assert "Draft comment" in response.data.decode()
    assert "После вашей оценки изменились" in response.data.decode()
    store = Store()
    assert store.vacancy_review("r1")["grade"] == 2
    store.close()
    fresh_token = context(client)
    assert client.post("/vacancy/r1/review", data={"review_context": fresh_token, "grade": "0"}).status_code == 303
    assert "После вашей оценки изменились" not in client.get("/vacancy/r1").data.decode()


def test_snapshot_survives_refresh_and_score_changes_without_becoming_stale(feedback_client):
    client = feedback_client
    token = context(client)
    client.post("/vacancy/r1/review", data={"review_context": token, "grade": "1"})
    store = Store()
    old_snapshot = store.conn.execute("SELECT inputs_json FROM vacancy_reviews").fetchone()[0]
    store.update_score_details("r1", 35, [], [])
    store.upsert(store.by_refnr("r1"))
    assert store.conn.execute("SELECT inputs_json FROM vacancy_reviews").fetchone()[0] == old_snapshot
    store.close()
    assert "После вашей оценки изменились" not in client.get("/vacancy/r1").data.decode()


def test_v14_migration_backup_idempotence_and_cascade(tmp_path):
    path = tmp_path / "db.sqlite"
    store = Store(path)
    store.add(Vacancy("r", "Job", "Example", "", "", "ba", score=75))
    store.close()
    old = sqlite3.connect(path)
    old.executescript("DROP TABLE vacancy_reviews; DELETE FROM schema_migrations WHERE version=15;")
    old.close()
    store = Store(path)
    assert store.schema_version() == LATEST_SCHEMA_VERSION
    v = store.by_refnr("r")
    store.save_vacancy_review("r", 2, "Fits", candidate_hash="test", vacancy_hash=vacancy_revision(v), inputs={})
    assert store.by_refnr("r").score == 75
    store.delete("r")
    assert store.vacancy_review("r") is None
    store.close()
    backups = list((tmp_path / "backups").glob("*.db"))
    assert len(backups) == 1
    with sqlite3.connect(backups[0]) as db:
        assert db.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 14
        assert db.execute("SELECT score FROM vacancies").fetchone()[0] == 75
    Store(path).close()
    assert len(list((tmp_path / "backups").glob("*.db"))) == 1


def test_capture_rejects_changes_and_store_checks_latest_vacancy(feedback_client):
    store = Store()
    cfg = ScoringConfig()
    v = store.by_refnr("r1")
    revision = candidate_revision(store, cfg)
    with pytest.raises(ValueError, match="changed"):
        capture_inputs(store, v, cfg, "wrong revision")
    snapshot = capture_inputs(store, v, cfg, revision)
    store.update_description("r1", "Changed during extraction")
    with pytest.raises(ValueError, match="changed"):
        store.save_vacancy_review("r1", 2, "Fits", candidate_hash=revision, vacancy_hash=vacancy_revision(v), inputs=snapshot)
    assert store.vacancy_review("r1") is None
    assert not store.conn.in_transaction
    store.close()


def test_feedback_export_uses_rated_snapshots_and_only_current_account(feedback_client, isolated_project):
    from cv_agent.scoring_review import build_feedback_packet, main
    from cv_agent.scoring_benchmark import load_dataset
    client = feedback_client
    store = Store()
    store.add(Vacancy("r2", "Office administrator", "Example", "Hamburg", "", "ba", description="Organize appointments and company documents." * 3))
    store.close()
    for refnr, grade in (("r1", "2"), ("r2", "0")):
        client.post(f"/vacancy/{refnr}/review", data={"review_context": context(client, refnr), "grade": grade, "reason": "Rated earlier"})
    store = Store()
    store.update_description("r1", "A refreshed unrelated description")
    store.close()
    (isolated_project / "cv" / "Илья" / "NewCV.txt").write_text("Updated Java skills")
    client.post("/logout")
    client.post("/login", data={"account": "Other"})
    client.post("/vacancy/r1/review", data={"review_context": context(client), "grade": "1", "reason": "Other private feedback"})
    before = (config.DATA_DIR / "applications.db").read_bytes()
    data, manifest = build_feedback_packet(user="Илья")
    assert {grade for grades in data["judgments"].values() for grade in grades.values()} == {0, 2}
    assert len(data["profiles"]) == 1
    assert any(v["text"].startswith("Python REST") for v in data["vacancies"].values())
    assert "refreshed unrelated" not in json.dumps(data)
    assert "Updated Java skills" not in json.dumps(data)
    assert "Other private" not in json.dumps(data)
    assert {v["refnr"] for v in manifest["sources"].values()} == {"r1", "r2"}
    assert (config.DATA_DIR / "applications.db").read_bytes() == before
    target = isolated_project / "labeled-feedback"
    assert main(["--from-feedback", "--output-dir", str(target)]) == 0
    assert load_dataset(target / "dataset.yaml")["label_source"] == "human_card_feedback_v1"
    assert target.stat().st_mode & 0o777 == 0o700
    assert all(p.stat().st_mode & 0o777 == 0o600 for p in target.iterdir())


def test_empty_feedback_export_has_clear_error_without_creating_directory(feedback_client, isolated_project):
    from cv_agent.scoring_review import main
    target = isolated_project / "no-feedback"
    with pytest.raises(SystemExit) as exc:
        main(["--from-feedback", "--output-dir", str(target)])
    assert exc.value.code == 2
    assert not target.exists()
