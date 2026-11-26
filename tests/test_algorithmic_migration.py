import sqlite3

from cv_agent.models import Vacancy
from cv_agent.store import LATEST_SCHEMA_VERSION, Store


def test_v12_migration_preserves_scores_backups_and_is_idempotent(tmp_path):
    path = tmp_path / "schema.db"
    store = Store(path)
    store.add(Vacancy("r", "Developer", "Synthetic", "Wangen", "url", "ba"))
    store.update_score_details("r", 77, ["original"], [])
    store.close()
    old = sqlite3.connect(path)
    old.executescript("DROP TRIGGER vacancy_embedding_cleanup; DROP TABLE embedding_vectors; DELETE FROM schema_migrations WHERE version>=13; ALTER TABLE candidate_profiles DROP COLUMN scoring_preferences;")
    for column in ("score_engine", "score_algorithm_version", "score_settings_hash", "score_details_json"):
        old.execute(f"ALTER TABLE vacancies DROP COLUMN {column}")
    for column in ("embedding_requests", "embedding_wait_seconds"):
        old.execute(f"ALTER TABLE jobs DROP COLUMN {column}")
    old.commit()
    old.close()
    store = Store(path)
    assert store.score_details("r")["score"] == 77
    assert store.score_details("r")["engine"] == "llm"
    assert store.conn.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == LATEST_SCHEMA_VERSION
    store.close()
    backups = list((tmp_path / "backups").glob("*.db"))
    assert len(backups) == 1
    backup = sqlite3.connect(backups[0])
    assert backup.execute("SELECT score FROM vacancies WHERE refnr='r'").fetchone()[0] == 77
    assert backup.execute("SELECT MAX(version) FROM schema_migrations").fetchone()[0] == 12
    backup.close()
    store = Store(path)
    store.close()
    assert len(list((tmp_path / "backups").glob("*.db"))) == 1


def test_purging_vacancy_removes_its_vectors_only(tmp_path):
    store = Store(tmp_path / "schema.db")
    try:
        store.add(Vacancy("gone", "Developer", "Synthetic", "Wangen", "url", "ba"))
        store.add(Vacancy("active", "Developer", "Synthetic", "Wangen", "url2", "ba"))
        store.replace_embedding_vectors("vacancy", "gone", "backend", "query", 2, {"hash": [1, 0]})
        store.replace_embedding_vectors("vacancy", "active", "backend", "query", 2, {"hash": [1, 0]})
        store.conn.execute("UPDATE vacancies SET archived_at='2000-01-01' WHERE refnr='gone'")
        store.conn.commit()
        assert store.purge_archived() == 1
        assert store.embedding_vectors("vacancy", "gone", "backend", "query") == {}
        assert store.embedding_vectors("vacancy", "active", "backend", "query")
    finally:
        store.close()
