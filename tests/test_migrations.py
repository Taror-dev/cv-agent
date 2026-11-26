import json
import hashlib
import sqlite3
from uuid import UUID

from cv_agent.store import LATEST_SCHEMA_VERSION, Store


LEGACY_SCHEMA = """
CREATE TABLE accounts (
    name TEXT PRIMARY KEY,
    keywords TEXT NOT NULL DEFAULT '[]'
);
CREATE TABLE vacancies (
    refnr TEXT PRIMARY KEY,
    title TEXT,
    company TEXT,
    location TEXT,
    url TEXT,
    source TEXT,
    description TEXT,
    score REAL,
    status TEXT DEFAULT 'new',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    applied_at TEXT
);
"""


def test_legacy_database_migrates_without_data_loss(tmp_path):
    path = tmp_path / "legacy.db"
    conn = sqlite3.connect(path)
    conn.executescript(LEGACY_SCHEMA)
    conn.execute(
        "INSERT INTO accounts (name, keywords) VALUES (?, ?)",
        ("default", json.dumps(["Legacy Python"])),
    )
    conn.execute(
        """INSERT INTO vacancies
           (refnr, title, company, location, url, source, description, score)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        ("legacy-1", "Legacy Job", "ACME", "Wangen", "https://example.test", "ba", "Python", 77),
    )
    conn.execute(
        """INSERT INTO vacancies
           (refnr, title, company, location, url, source, description, score)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            "indeed-legacy",
            "Unsupported Job",
            "Old Inc",
            "Berlin",
            "https://indeed.example.test/job",
            "indeed",
            "",
            None,
        ),
    )
    conn.commit()
    conn.close()

    store = Store(path)
    try:
        backups = list((tmp_path / "backups").glob("legacy-pre-migration-*.db"))
        assert len(backups) == 1
        backup_conn = sqlite3.connect(backups[0])
        try:
            backup_columns = {
                row[1] for row in backup_conn.execute("PRAGMA table_info(accounts)")
            }
            assert "account_id" not in backup_columns
            assert backup_conn.execute(
                "SELECT name FROM accounts"
            ).fetchone() == ("default",)
        finally:
            backup_conn.close()

        assert "default" not in store.list_accounts()
        assert "Илья" in store.list_accounts()
        assert store.account_keywords("Илья") == ["Legacy Python"]
        vacancy = store.by_refnr("legacy-1")
        assert vacancy is not None
        assert vacancy.title == "Legacy Job"
        assert vacancy.description == "Python"
        assert vacancy.score == 77
        assert store.account_gemini_key("Илья") == ""
        account_id = store.account_id("Илья")
        assert str(UUID(account_id)) == account_id
        assert store.account_name(account_id) == "Илья"

        columns = {
            row[1]: row for row in store.conn.execute("PRAGMA table_info(vacancies)")
        }
        assert "translation" in columns
        assert columns["account_id"][5] == 1
        assert columns["refnr"][5] == 2
        assert columns["user"][5] == 0
        account_columns = {
            row[1]: row for row in store.conn.execute("PRAGMA table_info(accounts)")
        }
        assert "account_id" in account_columns
        assert [
            row[0]
            for row in store.conn.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            )
        ] == list(range(1, LATEST_SCHEMA_VERSION + 1))
        foreign_keys = store.conn.execute(
            "PRAGMA foreign_key_list(vacancies)"
        ).fetchall()
        assert any(
            row[2] == "accounts"
            and row[3] == "account_id"
            and row[4] == "account_id"
            and row[6] == "CASCADE"
            for row in foreign_keys
        )
        index_names = {
            row[1] for row in store.conn.execute("PRAGMA index_list(vacancies)")
        }
        assert {
            "vacancies_account_status_idx",
            "vacancies_account_score_idx",
            "vacancies_account_source_idx",
        }.issubset(index_names)
        job_columns = {
            row[1] for row in store.conn.execute("PRAGMA table_info(jobs)")
        }
        assert {"run_id", "account_id", "status", "steps_json"}.issubset(
            job_columns
        )
        job_foreign_keys = store.conn.execute(
            "PRAGMA foreign_key_list(jobs)"
        ).fetchall()
        assert any(
            row[2] == "accounts"
            and row[3] == "account_id"
            and row[4] == "account_id"
            and row[6] == "CASCADE"
            for row in job_foreign_keys
        )
        job_indexes = {
            row[1] for row in store.conn.execute("PRAGMA index_list(jobs)")
        }
        assert {
            "jobs_account_created_idx",
            "jobs_one_active_per_account_idx",
        }.issubset(job_indexes)
        profile_columns = {
            row[1] for row in store.conn.execute(
                "PRAGMA table_info(candidate_profiles)"
            )
        }
        assert {
            "account_id",
            "full_name",
            "languages",
            "preferred_letter_language",
            "search_region",
        }.issubset(profile_columns)
        profile_foreign_keys = store.conn.execute(
            "PRAGMA foreign_key_list(candidate_profiles)"
        ).fetchall()
        assert any(
            row[2] == "accounts"
            and row[3] == "account_id"
            and row[4] == "account_id"
            and row[6] == "CASCADE"
            for row in profile_foreign_keys
        )
        assert store.candidate_profile("Илья")["full_name"] == ""
        document_columns = {
            row[1] for row in store.conn.execute("PRAGMA table_info(documents)")
        }
        assert {
            "account_id",
            "filename",
            "document_type",
            "sha256",
            "extracted_chars",
            "extract_error",
        }.issubset(document_columns)
        document_foreign_keys = store.conn.execute(
            "PRAGMA foreign_key_list(documents)"
        ).fetchall()
        assert any(
            row[2] == "accounts"
            and row[3] == "account_id"
            and row[4] == "account_id"
            and row[6] == "CASCADE"
            for row in document_foreign_keys
        )
        artifact_columns = {
            row[1] for row in store.conn.execute("PRAGMA table_info(artifacts)")
        }
        assert {
            "account_id",
            "refnr",
            "kind",
            "provider",
            "model",
            "prompt_version",
            "candidate_hash",
            "vacancy_hash",
            "created_at",
        }.issubset(artifact_columns)
        assert {"score_state", "score_reasons", "score_missing"}.issubset(columns)
        assert {
            "published_at",
            "last_seen_at",
            "description_fetched_at",
            "description_hash",
            "source_status",
            "archived_at",
            "source_metadata",
            "score_provider",
            "score_model",
            "score_prompt_version",
            "score_candidate_hash",
            "score_vacancy_hash",
            "score_description_hash",
            "score_skip_reason",
        }.issubset(columns)
        assert store.vacancy_lifecycle("legacy-1")["description_hash"] == (
            hashlib.sha256(b"Python").hexdigest()
        )
        assert store.vacancy_lifecycle("legacy-1")["description_fetched_at"]
        indeed = store.by_refnr("indeed-legacy")
        assert indeed is not None
        assert indeed.source_status == "unsupported"
        assert indeed.archived_at
        assert indeed.source_metadata == {"legacy_source": "indeed"}
        assert "indeed-legacy" not in {row[0] for row in store.list_all()}
        assert "indeed-legacy" in {
            row[0] for row in store.list_all(include_archived=True)
        }
        vacancy_indexes = {
            row[1] for row in store.conn.execute("PRAGMA index_list(vacancies)")
        }
        assert "vacancies_account_lifecycle_idx" in vacancy_indexes
        assert "vacancies_account_scoring_idx" in vacancy_indexes
        assert store.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='llm_rate_limits'"
        ).fetchone() == (1,)
        job_columns = {
            row[1] for row in store.conn.execute("PRAGMA table_info(jobs)")
        }
        assert {
            "network_requests",
            "network_wait_seconds",
            "llm_requests",
            "llm_wait_seconds",
        }.issubset(job_columns)
        assert store.score_details("legacy-1")["state"] == "scored"
    finally:
        store.close()

    reopened = Store(path)
    try:
        assert reopened.by_refnr("legacy-1") is not None
        assert len(
            list((tmp_path / "backups").glob("legacy-pre-migration-*.db"))
        ) == 1
    finally:
        reopened.close()


def test_resume_city_migration_adds_column(tmp_path):
    from cv_agent.store import Store

    path = tmp_path / "city.db"
    store = Store(path)
    try:
        cols = {
            row[1]
            for row in store.conn.execute(
                "PRAGMA table_info(candidate_profiles)"
            )
        }
        assert "resume_city" in cols
        store.set_candidate_profile(
            {"resume_city": "Wangen im Allgäu"}, "Илья"
        )
        profile = store.candidate_profile("Илья")
        assert profile["resume_city"] == "Wangen im Allgäu"
    finally:
        store.close()


def test_resume_city_survives_upgrade_from_v11(tmp_path):
    """БД версии v11 (без resume_city) обновляется, данные сохраняются."""
    import sqlite3

    from cv_agent.store import Store

    path = tmp_path / "v11.db"
    # создаём актуальную БД и заполняем профиль
    store = Store(path)
    try:
        store.set_candidate_profile({"resume_city": ""}, "Илья")
    finally:
        store.close()

    # откатываем схему до v11: убираем колонку и запись миграции 12
    conn = sqlite3.connect(path)
    try:
        conn.executescript(
            """
            CREATE TABLE candidate_profiles_v11 (
                account_id TEXT PRIMARY KEY,
                full_name TEXT NOT NULL DEFAULT '',
                address TEXT NOT NULL DEFAULT '',
                email TEXT NOT NULL DEFAULT '',
                phone TEXT NOT NULL DEFAULT '',
                languages TEXT NOT NULL DEFAULT '[]',
                preferred_letter_language TEXT NOT NULL DEFAULT 'de',
                search_region TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (account_id) REFERENCES accounts(account_id)
                    ON DELETE CASCADE
            );
            INSERT INTO candidate_profiles_v11
                SELECT account_id, full_name, address, email, phone,
                       languages, preferred_letter_language, search_region,
                       updated_at
                FROM candidate_profiles;
            DROP TABLE candidate_profiles;
            ALTER TABLE candidate_profiles_v11 RENAME TO candidate_profiles;
            DELETE FROM schema_migrations WHERE version = 12;
            """
        )
        conn.commit()
    finally:
        conn.close()

    # повторное открытие выполняет миграцию 12 поверх данных
    store = Store(path)
    try:
        profile = store.candidate_profile("Илья")
        assert profile is not None
        assert profile["resume_city"] == ""
        assert profile["full_name"] == ""
        store.set_candidate_profile({"resume_city": "München"}, "Илья")
        assert store.candidate_profile("Илья")["resume_city"] == "München"
    finally:
        store.close()
