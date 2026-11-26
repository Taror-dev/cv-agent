import sqlite3
from types import SimpleNamespace

import pytest

from cv_agent.scoring_inputs import collect_scoring_inputs
from cv_agent.store import LATEST_SCHEMA_VERSION, Store
from cv_agent.webapp import app
from test_algorithmic_score import prepare_cli


def test_preferences_are_account_scoped_hash_inputs_and_survive_contact_updates(isolated_project):
    first = Store()
    first.create_account("Other")
    second = Store(user="Other")
    try:
        before = collect_scoring_inputs(first)
        prefs = {"work_mode": "remote", "cities": ["Wangen im Allgäu"]}
        first.set_candidate_profile({"scoring_preferences": prefs})
        after = collect_scoring_inputs(first)
        assert before.candidate_hash != after.candidate_hash
        assert before.documents == after.documents
        assert after.location_preferences == prefs
        assert second.candidate_profile()["scoring_preferences"] == {"work_mode": "unspecified", "cities": []}
        first.set_candidate_profile({"email": "synthetic@example.test"})
        assert first.candidate_profile()["scoring_preferences"] == prefs
        with pytest.raises(ValueError):
            first.set_candidate_profile({"full_name": "Should not persist", "scoring_preferences": {"work_mode": "invalid"}})
        assert first.candidate_profile()["full_name"] != "Should not persist"
    finally:
        first.close()
        second.close()


def test_v13_migration_preserves_vectors_profile_and_scores_with_backup(tmp_path):
    from test_algorithmic_score import vacancy
    path = tmp_path / "old.db"
    store = Store(path)
    store.set_candidate_profile({"full_name": "Synthetic Person", "search_region": "Wangen"})
    store.add(vacancy())
    store.update_score_details("r1", 78, ["original"], [])
    store.replace_embedding_vectors("vacancy", "r1", "backend", "query", 2, {"hash": [1, 0]})
    store.close()
    old = sqlite3.connect(path)
    old.execute("ALTER TABLE candidate_profiles DROP COLUMN scoring_preferences")
    old.execute("DELETE FROM schema_migrations WHERE version>13")
    old.commit()
    old.close()
    check = Store(path)
    assert check.schema_version() == LATEST_SCHEMA_VERSION
    assert check.candidate_profile()["full_name"] == "Synthetic Person"
    assert check.candidate_profile()["scoring_preferences"] == {"work_mode": "unspecified", "cities": []}
    assert check.score_details("r1")["score"] == 78
    assert check.embedding_vectors("vacancy", "r1", "backend", "query") == {"hash": [1., 0.]}
    check.close()
    backups = list((tmp_path / "backups").glob("*.db"))
    assert len(backups) == 1
    backup = sqlite3.connect(backups[0])
    assert backup.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 13
    assert backup.execute("SELECT COUNT(*) FROM embedding_vectors").fetchone()[0] == 1
    backup.close()
    Store(path).close()
    assert len(list((tmp_path / "backups").glob("*.db"))) == 1


def test_web_preferences_save_validate_and_isolate_accounts():
    store = Store()
    store.create_account("Other")
    store.close()
    client = app.test_client()
    client.post("/login", data={"account": "Илья"})
    response = client.post("/profile/details", data={
        "full_name": "Synthetic Person", "scoring_work_mode": "remote",
        "scoring_cities": "Berlin, 10115 BERLIN, Wangen im Allgäu",
    })
    assert response.status_code == 200
    assert "Полностью удалённо" in response.data.decode()
    check = Store()
    prefs = check.candidate_profile()["scoring_preferences"]
    assert prefs == {"work_mode": "remote", "cities": ["Berlin", "Wangen im Allgäu"]}
    check.close()
    for invalid in ({"scoring_work_mode": "invalid"}, {"scoring_cities": "x" * 121}, {"scoring_cities": ",".join(["Berlin"] * 21)}):
        assert client.post("/profile/details", data=invalid).status_code == 400
    check = Store()
    assert check.candidate_profile()["full_name"] == "Synthetic Person"
    assert check.candidate_profile()["scoring_preferences"] == prefs
    check.close()
    # An older client submitting only the original fields must preserve prefs.
    assert client.post("/profile/details", data={"full_name": "New Synthetic Name"}).status_code == 200
    check = Store()
    assert check.candidate_profile()["scoring_preferences"] == prefs
    check.close()
    client.post("/logout")
    client.post("/login", data={"account": "Other"})
    html = client.get("/profile").data.decode()
    assert 'value="Berlin, Wangen im Allgäu"' not in html


def test_changed_preferences_invalidate_result_without_recomputing_vectors(monkeypatch, tmp_path, isolated_project):
    from cv_agent.scoring_inputs import collect_scoring_inputs as actual_collect
    cli, path, _ = prepare_cli(monkeypatch, tmp_path, isolated_project)
    monkeypatch.setattr("cv_agent.scoring_inputs.collect_scoring_inputs", actual_collect)
    # Supply a CV with a task so that both query/document vectors are cached.
    cv = isolated_project / "cv" / "Илья" / "Lebenslauf.txt"
    cv.write_text("Python\nDeutsch B2\nBerufserfahrung:\nDevelop software services", encoding="utf-8")
    args = SimpleNamespace(engine="algorithmic", preview=False, missing=True, force=False, verbose=False)
    cli.cmd_score(args)
    check = Store(path)
    original = check.score_details("r1")
    check.set_candidate_profile({"scoring_preferences": {"work_mode": "any"}})
    check.close()
    cli.cmd_score(args)
    check = Store(path)
    try:
        updated = check.score_details("r1")
        assert updated["candidate_hash"] != original["candidate_hash"]
        assert updated["score"] == original["score"] + 10
        assert updated["details"]["cache"]["misses"] == 0
        assert updated["details"]["cache"]["hits"] == 2
        assert len(check.artifacts("scoring")) == 2
    finally:
        check.close()
