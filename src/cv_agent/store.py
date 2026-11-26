import hashlib
import json
import math
import os
import shutil
import sqlite3
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .config import DATA_DIR, DEFAULT_USER, ROOT
from .models import Vacancy
from .paths import normalize_account_name, valid_account_name

LATEST_SCHEMA_VERSION = 15
JOB_LOG_LIMIT = 200_000
ACTIVE_JOB_STATUSES = frozenset({"queued", "running", "waiting_rate_limit"})
TERMINAL_JOB_STATUSES = frozenset(
    {"succeeded", "failed", "cancelled", "timed_out", "interrupted"}
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _metadata_flag(raw: str, key: str) -> bool:
    try:
        metadata = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return False
    return bool(isinstance(metadata, dict) and metadata.get(key))

SCHEMA = """
CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS accounts (
    name TEXT PRIMARY KEY,
    account_id TEXT NOT NULL UNIQUE,
    keywords TEXT NOT NULL DEFAULT '[]',
    gemini_api_key TEXT,
    groq_api_key TEXT,
    preferred_llm TEXT,
    search_params TEXT NOT NULL DEFAULT ''
);
CREATE TABLE IF NOT EXISTS vacancies (
    account_id TEXT NOT NULL,
    refnr TEXT NOT NULL,
    title TEXT,
    company TEXT,
    location TEXT,
    url TEXT,
    source TEXT,
    published_at TEXT NOT NULL DEFAULT '',
    last_seen_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    source_status TEXT NOT NULL DEFAULT 'active'
        CHECK (source_status IN ('active', 'error', 'unavailable', 'unsupported')),
    archived_at TEXT,
    source_metadata TEXT NOT NULL DEFAULT '{}',
    description TEXT,
    description_fetched_at TEXT,
    description_hash TEXT NOT NULL DEFAULT '',
    score REAL,
    score_state TEXT NOT NULL DEFAULT 'unscored'
        CHECK (score_state IN ('unscored', 'scored', 'score_error')),
    score_reasons TEXT NOT NULL DEFAULT '[]',
    score_missing TEXT NOT NULL DEFAULT '[]',
    score_provider TEXT NOT NULL DEFAULT '',
    score_model TEXT NOT NULL DEFAULT '',
    score_prompt_version TEXT NOT NULL DEFAULT '',
    score_candidate_hash TEXT NOT NULL DEFAULT '',
    score_vacancy_hash TEXT NOT NULL DEFAULT '',
    score_description_hash TEXT NOT NULL DEFAULT '',
    score_skip_reason TEXT NOT NULL DEFAULT '',
    status TEXT DEFAULT 'new',
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    applied_at TEXT,
    translation TEXT,
    user TEXT NOT NULL DEFAULT 'Илья',
    PRIMARY KEY (account_id, refnr),
    FOREIGN KEY (account_id) REFERENCES accounts(account_id) ON DELETE CASCADE
);
"""


def _migrate(conn: sqlite3.Connection):
    """Последовательные idempotent migrations к LATEST_SCHEMA_VERSION."""
    _ensure_legacy_columns(conn)
    _migrate_default_account(conn)
    _ensure_account_ids(conn)
    _record_migration(conn, 1)
    _migrate_vacancy_ownership(conn)
    _record_migration(conn, 2)
    _migrate_jobs(conn)
    _record_migration(conn, 3)
    _migrate_candidate_profiles(conn)
    _record_migration(conn, 4)
    _migrate_documents(conn)
    _record_migration(conn, 5)
    _migrate_artifacts(conn)
    _record_migration(conn, 6)
    _migrate_llm_rate_limits(conn)
    _record_migration(conn, 7)
    _migrate_vacancy_lifecycle(conn)
    _record_migration(conn, 8)
    _migrate_scoring_metadata(conn)
    _record_migration(conn, 9)
    _migrate_job_metrics(conn)
    _record_migration(conn, 10)
    _migrate_search_params(conn)
    _record_migration(conn, 11)
    _migrate_resume_city(conn)
    _record_migration(conn, 12)
    _migrate_algorithmic_scoring(conn)
    _record_migration(conn, 13)
    _migrate_scoring_preferences(conn)
    _record_migration(conn, 14)
    _migrate_vacancy_reviews(conn)
    _record_migration(conn, 15)


def _ensure_legacy_columns(conn: sqlite3.Connection) -> None:
    cols = {r[1] for r in conn.execute("PRAGMA table_info(vacancies)")}
    if "translation" not in cols:
        conn.execute("ALTER TABLE vacancies ADD COLUMN translation TEXT")
    if "user" not in cols:
        conn.execute(
            "ALTER TABLE vacancies ADD COLUMN user TEXT NOT NULL DEFAULT 'Илья'"
        )
    acc_cols = {r[1] for r in conn.execute("PRAGMA table_info(accounts)")}
    if "gemini_api_key" not in acc_cols:
        conn.execute("ALTER TABLE accounts ADD COLUMN gemini_api_key TEXT")
    if "groq_api_key" not in acc_cols:
        conn.execute("ALTER TABLE accounts ADD COLUMN groq_api_key TEXT")
    if "preferred_llm" not in acc_cols:
        conn.execute("ALTER TABLE accounts ADD COLUMN preferred_llm TEXT")
    if "account_id" not in acc_cols:
        conn.execute("ALTER TABLE accounts ADD COLUMN account_id TEXT")
    if "api_keys" not in acc_cols:
        conn.execute("ALTER TABLE accounts ADD COLUMN api_keys TEXT NOT NULL DEFAULT ''")


def _record_migration(conn: sqlite3.Connection, version: int) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO schema_migrations(version) VALUES (?)", (version,)
    )


def _migrate_vacancy_ownership(conn: sqlite3.Connection) -> None:
    """Переносит ownership вакансий с display name на стабильный account_id."""
    users = [
        str(row[0])
        for row in conn.execute("SELECT DISTINCT user FROM vacancies").fetchall()
    ]
    for user in users:
        normalized = normalize_account_name(user)
        if not valid_account_name(normalized):
            raise RuntimeError(
                "Миграция остановлена: вакансия принадлежит профилю "
                "с недопустимым именем"
            )
        if not conn.execute(
            "SELECT 1 FROM accounts WHERE name=?", (normalized,)
        ).fetchone():
            conn.execute(
                "INSERT INTO accounts(name, account_id) VALUES (?, ?)",
                (normalized, _new_account_id()),
            )

    columns = {row[1]: row for row in conn.execute("PRAGMA table_info(vacancies)")}
    pk_columns = [
        name
        for name, row in sorted(columns.items(), key=lambda item: item[1][5])
        if row[5]
    ]
    foreign_keys = conn.execute("PRAGMA foreign_key_list(vacancies)").fetchall()
    has_owner_fk = any(
        row[2] == "accounts" and row[3] == "account_id" and row[4] == "account_id"
        for row in foreign_keys
    )
    needs_rebuild = "account_id" not in columns or pk_columns != [
        "account_id",
        "refnr",
    ] or not has_owner_fk

    if needs_rebuild:
        conn.execute("ALTER TABLE vacancies RENAME TO vacancies_legacy")
        conn.execute(
            """CREATE TABLE vacancies (
                account_id TEXT NOT NULL,
                refnr TEXT NOT NULL,
                title TEXT,
                company TEXT,
                location TEXT,
                url TEXT,
                source TEXT,
                description TEXT,
                score REAL,
                status TEXT DEFAULT 'new',
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                applied_at TEXT,
                translation TEXT,
                user TEXT NOT NULL,
                PRIMARY KEY (account_id, refnr),
                FOREIGN KEY (account_id) REFERENCES accounts(account_id)
                    ON DELETE CASCADE
            )"""
        )
        conn.execute(
            """INSERT INTO vacancies
               (account_id, refnr, title, company, location, url, source,
                description, score, status, created_at, applied_at, translation, user)
               SELECT a.account_id, v.refnr, v.title, v.company, v.location,
                      v.url, v.source, v.description, v.score, v.status,
                      v.created_at, v.applied_at, v.translation, v.user
               FROM vacancies_legacy v
               JOIN accounts a ON a.name=v.user"""
        )
        conn.execute("DROP TABLE vacancies_legacy")

    conn.execute(
        "CREATE INDEX IF NOT EXISTS vacancies_account_status_idx "
        "ON vacancies(account_id, status)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS vacancies_account_score_idx "
        "ON vacancies(account_id, score DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS vacancies_account_source_idx "
        "ON vacancies(account_id, source)"
    )


def _migrate_jobs(conn: sqlite3.Connection) -> None:
    """Создаёт account-scoped хранилище фоновых прогонов."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS jobs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'queued'
                CHECK (status IN ('queued', 'running', 'waiting_rate_limit',
                                  'succeeded', 'failed', 'cancelled',
                                  'timed_out', 'interrupted')),
            steps_json TEXT NOT NULL,
            llm_note TEXT NOT NULL DEFAULT '',
            error TEXT NOT NULL DEFAULT '',
            network_requests INTEGER NOT NULL DEFAULT 0,
            network_wait_seconds REAL NOT NULL DEFAULT 0,
            llm_requests INTEGER NOT NULL DEFAULT 0,
            llm_wait_seconds REAL NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            started_at TEXT,
            finished_at TEXT,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (account_id) REFERENCES accounts(account_id)
                ON DELETE CASCADE
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS jobs_account_created_idx "
        "ON jobs(account_id, created_at DESC)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS jobs_one_active_per_account_idx "
        "ON jobs(account_id) "
        "WHERE status IN ('queued', 'running', 'waiting_rate_limit')"
    )


def _migrate_candidate_profiles(conn: sqlite3.Connection) -> None:
    """Отделяет данные кандидата от технической записи account/session."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS candidate_profiles (
            account_id TEXT PRIMARY KEY,
            full_name TEXT NOT NULL DEFAULT '',
            address TEXT NOT NULL DEFAULT '',
            email TEXT NOT NULL DEFAULT '',
            phone TEXT NOT NULL DEFAULT '',
            languages TEXT NOT NULL DEFAULT '[]',
            preferred_letter_language TEXT NOT NULL DEFAULT 'de',
            search_region TEXT NOT NULL DEFAULT '',
            resume_city TEXT NOT NULL DEFAULT '',
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (account_id) REFERENCES accounts(account_id)
                ON DELETE CASCADE
        )"""
    )
    conn.execute(
        "INSERT OR IGNORE INTO candidate_profiles(account_id) "
        "SELECT account_id FROM accounts"
    )


def _migrate_resume_city(conn: sqlite3.Connection) -> None:
    """Город для футера резюме («Город, ДД.ММ.ГГГГ») в досье (schema v12)."""
    cols = {r[1] for r in conn.execute("PRAGMA table_info(candidate_profiles)")}
    if "resume_city" not in cols:
        conn.execute(
            "ALTER TABLE candidate_profiles "
            "ADD COLUMN resume_city TEXT NOT NULL DEFAULT ''"
        )


def _migrate_documents(conn: sqlite3.Connection) -> None:
    """Metadata извлечения; содержимое CV намеренно остаётся только в файлах."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS documents (
            account_id TEXT NOT NULL,
            filename TEXT NOT NULL,
            document_type TEXT NOT NULL DEFAULT 'other'
                CHECK (document_type IN ('cv', 'curriculum', 'certificate', 'other')),
            type_source TEXT NOT NULL DEFAULT 'auto'
                CHECK (type_source IN ('auto', 'manual')),
            mime_type TEXT NOT NULL DEFAULT 'application/octet-stream',
            size_bytes INTEGER NOT NULL DEFAULT 0,
            sha256 TEXT NOT NULL DEFAULT '',
            extracted_chars INTEGER NOT NULL DEFAULT 0,
            extracted_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            extract_error TEXT NOT NULL DEFAULT '',
            PRIMARY KEY (account_id, filename),
            FOREIGN KEY (account_id) REFERENCES accounts(account_id)
                ON DELETE CASCADE
        )"""
    )


def _migrate_artifacts(conn: sqlite3.Connection) -> None:
    """LLM-Reproduzierbarkeit und expliziter Scoring-Fehlerzustand."""
    vacancy_columns = {r[1] for r in conn.execute("PRAGMA table_info(vacancies)")}
    if "score_state" not in vacancy_columns:
        conn.execute(
            "ALTER TABLE vacancies ADD COLUMN score_state TEXT NOT NULL "
            "DEFAULT 'unscored' CHECK (score_state IN "
            "('unscored', 'scored', 'score_error'))"
        )
    if "score_reasons" not in vacancy_columns:
        conn.execute(
            "ALTER TABLE vacancies ADD COLUMN score_reasons TEXT NOT NULL DEFAULT '[]'"
        )
    if "score_missing" not in vacancy_columns:
        conn.execute(
            "ALTER TABLE vacancies ADD COLUMN score_missing TEXT NOT NULL DEFAULT '[]'"
        )
    conn.execute(
        "UPDATE vacancies SET score_state='scored' "
        "WHERE score IS NOT NULL AND score_state='unscored'"
    )
    conn.execute(
        """CREATE TABLE IF NOT EXISTS artifacts (
            artifact_id INTEGER PRIMARY KEY AUTOINCREMENT,
            account_id TEXT NOT NULL,
            refnr TEXT,
            kind TEXT NOT NULL,
            status TEXT NOT NULL
                CHECK (status IN ('succeeded', 'failed')),
            provider TEXT NOT NULL DEFAULT '',
            model TEXT NOT NULL DEFAULT '',
            prompt_version TEXT NOT NULL DEFAULT '',
            candidate_hash TEXT NOT NULL DEFAULT '',
            vacancy_hash TEXT NOT NULL DEFAULT '',
            output_path TEXT NOT NULL DEFAULT '',
            error TEXT NOT NULL DEFAULT '',
            details_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (account_id) REFERENCES accounts(account_id)
                ON DELETE CASCADE
        )"""
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS artifacts_account_created_idx "
        "ON artifacts(account_id, created_at DESC, artifact_id DESC)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS artifacts_account_refnr_idx "
        "ON artifacts(account_id, refnr, kind)"
    )


def _migrate_llm_rate_limits(conn: sqlite3.Connection) -> None:
    """Account/provider-spezifische, prozessübergreifende 429-Cooldowns."""
    conn.execute(
        """CREATE TABLE IF NOT EXISTS llm_rate_limits (
            account_id TEXT NOT NULL,
            provider_key TEXT NOT NULL,
            retry_after REAL NOT NULL,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (account_id, provider_key),
            FOREIGN KEY (account_id) REFERENCES accounts(account_id)
                ON DELETE CASCADE
        )"""
    )
    cols = {row[1] for row in conn.execute("PRAGMA table_info(llm_rate_limits)")}
    if "quota_headers" not in cols:
        conn.execute(
            "ALTER TABLE llm_rate_limits ADD COLUMN quota_headers TEXT"
        )
    if "quota_updated_at" not in cols:
        conn.execute(
            "ALTER TABLE llm_rate_limits ADD COLUMN quota_updated_at TEXT"
        )


def _migrate_vacancy_lifecycle(conn: sqlite3.Connection) -> None:
    """Source-Lebenszyklus, Description-TTL und Legacy-Indeed-Archivierung."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(vacancies)")}
    additions = {
        "published_at": "TEXT NOT NULL DEFAULT ''",
        "last_seen_at": "TEXT",
        "description_fetched_at": "TEXT",
        "description_hash": "TEXT NOT NULL DEFAULT ''",
        "source_status": (
            "TEXT NOT NULL DEFAULT 'active' CHECK (source_status IN "
            "('active', 'error', 'unavailable', 'unsupported'))"
        ),
        "archived_at": "TEXT",
        "source_metadata": "TEXT NOT NULL DEFAULT '{}'",
    }
    already_applied = conn.execute(
        "SELECT 1 FROM schema_migrations WHERE version=8"
    ).fetchone()
    if already_applied and set(additions).issubset(columns):
        conn.execute(
            "CREATE INDEX IF NOT EXISTS vacancies_account_lifecycle_idx "
            "ON vacancies(account_id, archived_at, source_status, last_seen_at)"
        )
        return
    for name, definition in additions.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE vacancies ADD COLUMN {name} {definition}")

    conn.execute(
        "UPDATE vacancies SET last_seen_at="
        "COALESCE(NULLIF(last_seen_at, ''), created_at, CURRENT_TIMESTAMP)"
    )
    for account_id, refnr, description, created_at, fetched_at, digest in conn.execute(
        "SELECT account_id, refnr, description, created_at, "
        "description_fetched_at, description_hash FROM vacancies"
    ).fetchall():
        text = str(description or "")
        if not text:
            continue
        expected_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if fetched_at and digest == expected_hash:
            continue
        conn.execute(
            "UPDATE vacancies SET description_fetched_at=?, description_hash=? "
            "WHERE account_id=? AND refnr=?",
            (fetched_at or created_at or _utc_now(), expected_hash, account_id, refnr),
        )

    conn.execute(
        "UPDATE vacancies SET source_status='unsupported', "
        "archived_at=COALESCE(archived_at, CURRENT_TIMESTAMP), "
        "source_metadata=? WHERE lower(trim(source))='indeed'",
        (json.dumps({"legacy_source": "indeed"}, sort_keys=True),),
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS vacancies_account_lifecycle_idx "
        "ON vacancies(account_id, archived_at, source_status, last_seen_at)"
    )


def _migrate_scoring_metadata(conn: sqlite3.Connection) -> None:
    """Reproduzierbare Score-Inputs und explizite Skip-Gründe."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(vacancies)")}
    additions = {
        "score_provider": "TEXT NOT NULL DEFAULT ''",
        "score_model": "TEXT NOT NULL DEFAULT ''",
        "score_prompt_version": "TEXT NOT NULL DEFAULT ''",
        "score_candidate_hash": "TEXT NOT NULL DEFAULT ''",
        "score_vacancy_hash": "TEXT NOT NULL DEFAULT ''",
        "score_description_hash": "TEXT NOT NULL DEFAULT ''",
        "score_skip_reason": "TEXT NOT NULL DEFAULT ''",
    }
    already_applied = conn.execute(
        "SELECT 1 FROM schema_migrations WHERE version=9"
    ).fetchone()
    if already_applied and set(additions).issubset(columns):
        conn.execute(
            "CREATE INDEX IF NOT EXISTS vacancies_account_scoring_idx "
            "ON vacancies(account_id, score_state, score_description_hash)"
        )
        return
    for name, definition in additions.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE vacancies ADD COLUMN {name} {definition}")
    conn.execute(
        "CREATE INDEX IF NOT EXISTS vacancies_account_scoring_idx "
        "ON vacancies(account_id, score_state, score_description_hash)"
    )


def _migrate_job_metrics(conn: sqlite3.Connection) -> None:
    """Messbare Request-Anzahl und Queue-Wartezeit pro DB-backed Job."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
    additions = {
        "network_requests": "INTEGER NOT NULL DEFAULT 0",
        "network_wait_seconds": "REAL NOT NULL DEFAULT 0",
        "llm_requests": "INTEGER NOT NULL DEFAULT 0",
        "llm_wait_seconds": "REAL NOT NULL DEFAULT 0",
    }
    for name, definition in additions.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} {definition}")


def _migrate_search_params(conn: sqlite3.Connection) -> None:
    """Сохранённые поля формы поиска для каждого аккаунта."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(accounts)")}
    if "search_params" not in columns:
        conn.execute(
            "ALTER TABLE accounts ADD COLUMN search_params "
            "TEXT NOT NULL DEFAULT ''"
        )


def _migrate_algorithmic_scoring(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(vacancies)")}
    for name, default in {
        "score_engine": "llm", "score_algorithm_version": "",
        "score_settings_hash": "", "score_details_json": "{}",
    }.items():
        if name not in columns:
            conn.execute(f"ALTER TABLE vacancies ADD COLUMN {name} TEXT NOT NULL DEFAULT '{default}'")
    conn.execute("""CREATE TABLE IF NOT EXISTS embedding_vectors (
        account_id TEXT NOT NULL,
        entity_kind TEXT NOT NULL CHECK (entity_kind IN ('document', 'vacancy')),
        entity_id TEXT NOT NULL,
        backend_id TEXT NOT NULL,
        role TEXT NOT NULL CHECK (role IN ('query', 'document')),
        text_hash TEXT NOT NULL,
        dimensions INTEGER NOT NULL CHECK (dimensions > 0),
        vector_json TEXT NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (account_id, entity_kind, entity_id, backend_id, role, text_hash),
        FOREIGN KEY (account_id) REFERENCES accounts(account_id) ON DELETE CASCADE
    )""")
    conn.execute("""CREATE TRIGGER IF NOT EXISTS vacancy_embedding_cleanup
        AFTER DELETE ON vacancies BEGIN
        DELETE FROM embedding_vectors WHERE account_id=OLD.account_id
          AND entity_kind='vacancy' AND entity_id=OLD.refnr;
        END""")
    for name in ("embedding_requests", "embedding_wait_seconds"):
        job_columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
        if name not in job_columns:
            definition = "INTEGER" if name == "embedding_requests" else "REAL"
            conn.execute(f"ALTER TABLE jobs ADD COLUMN {name} {definition} NOT NULL DEFAULT 0")


def _migrate_scoring_preferences(conn: sqlite3.Connection) -> None:
    columns = {row[1] for row in conn.execute("PRAGMA table_info(candidate_profiles)")}
    if "scoring_preferences" not in columns:
        conn.execute("ALTER TABLE candidate_profiles ADD COLUMN scoring_preferences TEXT NOT NULL DEFAULT '{}'")


def _migrate_vacancy_reviews(conn: sqlite3.Connection) -> None:
    conn.execute("""CREATE TABLE IF NOT EXISTS vacancy_reviews (
        account_id TEXT NOT NULL,
        refnr TEXT NOT NULL,
        grade INTEGER NOT NULL CHECK (grade IN (0, 1, 2)),
        reason TEXT NOT NULL DEFAULT '' CHECK (length(reason) <= 2000),
        candidate_hash TEXT NOT NULL,
        vacancy_hash TEXT NOT NULL,
        inputs_json TEXT NOT NULL,
        updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (account_id, refnr),
        FOREIGN KEY (account_id, refnr) REFERENCES vacancies(account_id, refnr) ON DELETE CASCADE
    )""")


def _needs_migration(conn: sqlite3.Connection) -> bool:
    """Read-only проверка перед созданием backup существующей SQLite."""
    tables = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if not tables:
        return False
    if not {
        "accounts",
        "candidate_profiles",
        "documents",
        "artifacts",
        "llm_rate_limits",
        "vacancies",
        "jobs",
        "schema_migrations",
        "embedding_vectors",
        "vacancy_reviews",
    }.issubset(tables):
        return True
    account_columns = {r[1] for r in conn.execute("PRAGMA table_info(accounts)")}
    vacancy_columns = {r[1] for r in conn.execute("PRAGMA table_info(vacancies)")}
    job_columns = {r[1] for r in conn.execute("PRAGMA table_info(jobs)")}
    profile_columns = {r[1] for r in conn.execute("PRAGMA table_info(candidate_profiles)")}
    if not {"resume_city", "scoring_preferences"}.issubset(profile_columns):
        return True
    if not {"account_id", "gemini_api_key", "search_params"}.issubset(
        account_columns
    ):
        return True
    if not {
        "translation",
        "user",
        "account_id",
        "score_state",
        "score_reasons",
        "score_missing",
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
        "score_engine", "score_algorithm_version", "score_settings_hash", "score_details_json",
    }.issubset(vacancy_columns):
        return True
    if not {
        "network_requests",
        "network_wait_seconds",
        "llm_requests",
        "llm_wait_seconds",
        "embedding_requests", "embedding_wait_seconds",
    }.issubset(job_columns):
        return True
    current_version = conn.execute(
        "SELECT COALESCE(MAX(version), 0) FROM schema_migrations"
    ).fetchone()[0]
    versions = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
    if not set(range(1, LATEST_SCHEMA_VERSION + 1)).issubset(versions):
        return True
    return current_version < LATEST_SCHEMA_VERSION or (
        conn.execute("SELECT 1 FROM accounts WHERE name='default'").fetchone()
        is not None
    )


def _backup_before_migration(conn: sqlite3.Connection, source: Path) -> Path:
    """Консистентная локальная копия DB до изменения схемы."""
    backup_dir = source.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    target = backup_dir / f"{source.stem}-pre-migration-{stamp}.db"
    backup_conn = sqlite3.connect(target)
    try:
        conn.backup(backup_conn)
    finally:
        backup_conn.close()
    target.chmod(0o600)
    return target


def _new_account_id() -> str:
    return str(uuid.uuid4())


def _ensure_account_ids(conn: sqlite3.Connection) -> None:
    """Идемпотентно присваивает неизменяемые UUID старым профилям."""
    rows = conn.execute(
        "SELECT name FROM accounts WHERE account_id IS NULL OR account_id=''"
    ).fetchall()
    for (name,) in rows:
        conn.execute(
            "UPDATE accounts SET account_id=? WHERE name=?",
            (_new_account_id(), name),
        )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS accounts_account_id_uq "
        "ON accounts(account_id)"
    )


def _migrate_default_account(conn: sqlite3.Connection):
    """Аккаунт по умолчанию называется DEFAULT_USER; данные из старого 'default' переносятся."""
    if conn.execute("SELECT 1 FROM accounts WHERE name='default'").fetchone():
        if conn.execute("SELECT 1 FROM accounts WHERE name=?", (DEFAULT_USER,)).fetchone():
            conn.execute(
                "UPDATE vacancies SET user=? WHERE user='default'", (DEFAULT_USER,)
            )
            conn.execute("DELETE FROM accounts WHERE name='default'")
        else:
            conn.execute(
                "UPDATE accounts SET name=? WHERE name='default'", (DEFAULT_USER,)
            )
            conn.execute(
                "UPDATE vacancies SET user=? WHERE user='default'", (DEFAULT_USER,)
            )
    if not conn.execute(
        "SELECT 1 FROM accounts WHERE name=?", (DEFAULT_USER,)
    ).fetchone():
        conn.execute(
            "INSERT INTO accounts (name, account_id) VALUES (?, ?)",
            (DEFAULT_USER, _new_account_id()),
        )
        try:
            from .config import load_yaml

            keywords = load_yaml("criteria.yaml").get("keywords") or []
            conn.execute(
                "UPDATE accounts SET keywords=? WHERE name=?",
                (json.dumps(keywords), DEFAULT_USER),
            )
        except Exception:
            pass
    _migrate_cv_files()


def _migrate_cv_files():
    """Перенос файлов из корня cv/ в cv/<DEFAULT_USER>/ (один раз)."""
    cv = ROOT / "cv"
    target = cv / DEFAULT_USER
    if not cv.is_dir() or target.is_dir():
        return
    loose = [p for p in cv.iterdir() if p.is_file()]
    if not loose:
        return
    target.mkdir(parents=True, exist_ok=True)
    for p in sorted(loose):
        shutil.move(str(p), str(target / p.name))


class Store:
    def __init__(self, path=None, user=None):
        self.path = Path(path) if path else DATA_DIR / "applications.db"
        self.path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        raw_user = user or os.environ.get("CV_AGENT_USER") or DEFAULT_USER
        self.user = normalize_account_name(raw_user)
        if not valid_account_name(self.user):
            raise ValueError("Недопустимое имя профиля")
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("PRAGMA foreign_keys=ON")
        try:
            self.path.chmod(0o600)
        except OSError:
            pass
        if _needs_migration(self.conn):
            _backup_before_migration(self.conn, self.path)
        try:
            self.conn.executescript(SCHEMA)
            with self.conn:
                _migrate(self.conn)
        except Exception:
            self.conn.rollback()
            self.conn.close()
            raise

    # ---- учётные записи ----

    def list_accounts(self) -> list[str]:
        return [
            r[0]
            for r in self.conn.execute("SELECT name FROM accounts ORDER BY name")
        ]

    def list_account_records(self) -> list[tuple[str, str]]:
        """Стабильный ID и display name всех профилей."""
        return [
            (str(row[0]), str(row[1]))
            for row in self.conn.execute(
                "SELECT account_id, name FROM accounts ORDER BY name"
            ).fetchall()
        ]

    def account_exists(self, name: str) -> bool:
        name = normalize_account_name(name)
        if not valid_account_name(name):
            return False
        return (
            self.conn.execute("SELECT 1 FROM accounts WHERE name=?", (name,)).fetchone()
            is not None
        )

    def account_id(self, name: str | None = None) -> str | None:
        """Неизменяемый UUID профиля по его отображаемому имени."""
        name = normalize_account_name(name or self.user)
        if not valid_account_name(name):
            return None
        row = self.conn.execute(
            "SELECT account_id FROM accounts WHERE name=?", (name,)
        ).fetchone()
        return str(row[0]) if row and row[0] else None

    def account_name(self, account_id: str) -> str | None:
        """Отображаемое имя по UUID; произвольные session-значения отклоняются."""
        try:
            normalized = str(uuid.UUID(str(account_id)))
        except (TypeError, ValueError, AttributeError):
            return None
        row = self.conn.execute(
            "SELECT name FROM accounts WHERE account_id=?", (normalized,)
        ).fetchone()
        return str(row[0]) if row else None

    def create_account(self, name: str) -> bool:
        name = normalize_account_name(name)
        if not valid_account_name(name):
            return False
        try:
            account_id = _new_account_id()
            self.conn.execute(
                "INSERT INTO accounts (name, account_id) VALUES (?, ?)",
                (name, account_id),
            )
            self.conn.execute(
                "INSERT INTO candidate_profiles(account_id) VALUES (?)",
                (account_id,),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def delete_account(self, name: str) -> bool:
        """Удаляет профиль; FK ON DELETE CASCADE очищает все его данные."""
        name = normalize_account_name(name)
        if not valid_account_name(name):
            return False
        row = self.conn.execute(
            "SELECT account_id FROM accounts WHERE name=?", (name,)
        ).fetchone()
        if row is None:
            return False
        self.conn.execute("DELETE FROM accounts WHERE account_id=?", (row[0],))
        self.conn.commit()
        return True

    def account_keywords(self, name: str | None = None) -> list[str] | None:
        """Ключевые слова аккаунта; None, если аккаунт не существует."""
        name = name or self.user
        row = self.conn.execute(
            "SELECT keywords FROM accounts WHERE name=?", (name,)
        ).fetchone()
        if row is None:
            return None
        try:
            data = json.loads(row[0])
            return [str(x).strip() for x in data if str(x).strip()]
        except (TypeError, ValueError):
            return []

    def set_account_keywords(self, keywords: list[str], name: str | None = None):
        name = name or self.user
        account_id = _new_account_id()
        self.conn.execute(
            "INSERT INTO accounts (name, account_id, keywords) VALUES (?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET keywords=excluded.keywords",
            (name, account_id, json.dumps(keywords)),
        )
        self.conn.execute(
            "INSERT OR IGNORE INTO candidate_profiles(account_id) "
            "SELECT account_id FROM accounts WHERE name=?",
            (name,),
        )
        self.conn.commit()

    ACCOUNT_API_KEY_COLUMNS = {
        "GEMINI_API_KEY": "gemini_api_key",
        "GROQ_API_KEY": "groq_api_key",
    }

    API_KEY_PROVIDER_BY_ENV = {
        "GEMINI_API_KEY": "gemini",
        "GROQ_API_KEY": "groq",
    }

    def _api_keys_env(self, provider: str) -> str:
        for env, prov in self.API_KEY_PROVIDER_BY_ENV.items():
            if prov == provider:
                return env
        return ""

    def _account_api_keys_raw(self, name: str) -> list[dict]:
        row = self.conn.execute(
            "SELECT api_keys FROM accounts WHERE name=?", (name,)
        ).fetchone()
        raw = (row[0] if row else "") or ""
        try:
            items = json.loads(raw)
        except (TypeError, ValueError):
            return []
        return [e for e in items if isinstance(e, dict) and e.get("name")]

    def _save_account_api_keys(self, entries: list[dict], name: str):
        self.conn.execute(
            "UPDATE accounts SET api_keys=? WHERE name=?",
            (json.dumps(entries, ensure_ascii=False), name),
        )
        self.conn.commit()

    def account_api_keys(self, name: str | None = None) -> list[dict]:
        """Именованные API-ключи аккаунта (без секретов)."""
        name = normalize_account_name(name or self.user)
        entries = self._account_api_keys_raw(name)
        if not entries:
            # Ленивая миграция старых одиночных ключей в именованный список.
            legacy_entries = []
            for env, provider in self.API_KEY_PROVIDER_BY_ENV.items():
                legacy = self.account_llm_key(env, name)
                if legacy:
                    legacy_entries.append(
                        {
                            "name": f"{provider}_api_key",
                            "provider": provider,
                            "key": legacy,
                        }
                    )
            if legacy_entries:
                self._save_account_api_keys(legacy_entries, name)
                entries = legacy_entries
        return [
            {"name": e["name"], "provider": e.get("provider", "")} for e in entries
        ]

    def add_account_api_key(
        self,
        key_name: str,
        provider: str,
        key: str,
        name: str | None = None,
    ):
        """Добавляет именованный ключ; последний ключ провайдера становится активным."""
        provider = str(provider or "").strip().lower()
        if provider not in {"gemini", "groq"}:
            raise ValueError("Неизвестный провайдер ключа")
        key_name = str(key_name or "").strip()
        if not key_name:
            raise ValueError("Укажите название ключа")
        if len(key_name) > 100:
            raise ValueError("Название ключа слишком длинное")
        key = str(key or "").strip()
        if not key:
            raise ValueError("Укажите значение ключа")
        name = normalize_account_name(name or self.user)
        entries = self._account_api_keys_raw(name)
        if any(e.get("name") == key_name for e in entries):
            raise ValueError(f"Ключ с названием «{key_name}» уже существует")
        entries.append({"name": key_name, "provider": provider, "key": key})
        self._save_account_api_keys(entries, name)
        env = self._api_keys_env(provider)
        self.set_account_llm_key(env, key, name)

    def delete_account_api_key(self, key_name: str, name: str | None = None) -> bool:
        """Удаляет именованный ключ; при удалении активного активируется предыдущий."""
        key_name = str(key_name or "").strip()
        name = normalize_account_name(name or self.user)
        entries = self._account_api_keys_raw(name)
        removed = next((e for e in entries if e.get("name") == key_name), None)
        if removed is None:
            return False
        entries = [e for e in entries if e.get("name") != key_name]
        self._save_account_api_keys(entries, name)
        provider = removed.get("provider", "")
        env = self._api_keys_env(provider)
        if env and self.account_llm_key(env, name) == removed.get("key", ""):
            replacement = next(
                (e for e in reversed(entries) if e.get("provider") == provider), None
            )
            self.set_account_llm_key(
                env, (replacement or {}).get("key", ""), name
            )
        return True

    def account_llm_key(self, api_key_env: str, name: str | None = None) -> str:
        """Активный API-ключ провайдера аккаунта по имени env-переменной."""
        column = self.ACCOUNT_API_KEY_COLUMNS.get(api_key_env)
        if column is None:
            return ""
        name = name or self.user
        provider = self.API_KEY_PROVIDER_BY_ENV.get(api_key_env)
        if provider:
            entries = [
                e
                for e in self._account_api_keys_raw(name)
                if e.get("provider") == provider
            ]
            if entries:
                return str(entries[-1].get("key", "")).strip()
        row = self.conn.execute(
            f"SELECT {column} FROM accounts WHERE name=?", (name,)
        ).fetchone()
        return (row[0] or "").strip() if row else ""

    def set_account_llm_key(
        self, api_key_env: str, key: str, name: str | None = None
    ):
        column = self.ACCOUNT_API_KEY_COLUMNS.get(api_key_env)
        if column is None:
            raise ValueError(f"Неизвестный провайдер ключа: {api_key_env}")
        name = name or self.user
        self.conn.execute(
            f"UPDATE accounts SET {column}=? WHERE name=?", (key, name)
        )
        provider = self.API_KEY_PROVIDER_BY_ENV.get(api_key_env)
        if provider:
            # Держим именованный список согласованным с одиночным ключом.
            entries = self._account_api_keys_raw(name)
            provider_entries = [
                e for e in entries if e.get("provider") == provider
            ]
            if provider_entries:
                provider_entries[-1]["key"] = str(key or "").strip()
            else:
                entries.append(
                    {
                        "name": f"{provider}_api_key",
                        "provider": provider,
                        "key": str(key or "").strip(),
                    }
                )
            self._save_account_api_keys(entries, name)
        else:
            self.conn.commit()

    def account_gemini_key(self, name: str | None = None) -> str:
        """API-ключ Gemini аккаунта (Ollama общая, внешние ключи — свои)."""
        return self.account_llm_key("GEMINI_API_KEY", name)

    def set_account_gemini_key(self, key: str, name: str | None = None):
        self.set_account_llm_key("GEMINI_API_KEY", key, name)

    def account_groq_key(self, name: str | None = None) -> str:
        """API-ключ Groq аккаунта (console.groq.com/keys)."""
        return self.account_llm_key("GROQ_API_KEY", name)

    def set_account_groq_key(self, key: str, name: str | None = None):
        self.set_account_llm_key("GROQ_API_KEY", key, name)

    def account_preferred_llm(self, name: str | None = None) -> str:
        """Сохранённый в профиле ID модели из серверного allowlist."""
        name = name or self.user
        row = self.conn.execute(
            "SELECT preferred_llm FROM accounts WHERE name=?", (name,)
        ).fetchone()
        return (row[0] or "").strip() if row else ""

    def set_account_preferred_llm(self, option_id: str, name: str | None = None):
        name = name or self.user
        self.conn.execute(
            "UPDATE accounts SET preferred_llm=? WHERE name=?",
            (str(option_id or ""), name),
        )
        self.conn.commit()

    def account_search_params(self, name: str | None = None) -> dict:
        """Сохранённые поля формы поиска аккаунта; {} без данных."""
        name = normalize_account_name(name or self.user)
        row = self.conn.execute(
            "SELECT search_params FROM accounts WHERE name=?", (name,)
        ).fetchone()
        if row is None or not (row[0] or "").strip():
            return {}
        try:
            data = json.loads(row[0])
        except (TypeError, ValueError):
            return {}
        if not isinstance(data, dict):
            return {}
        return {
            str(key): str(value)
            for key, value in data.items()
            if isinstance(key, str) and isinstance(value, str)
        }

    def set_account_search_params(self, params: dict, name: str | None = None):
        """Сохраняет поля формы поиска за аккаунтом (JSON в SQLite)."""
        name = normalize_account_name(name or self.user)
        safe = {
            str(key): str(value)
            for key, value in dict(params or {}).items()
            if isinstance(key, str) and isinstance(value, str)
        }
        self.conn.execute(
            "UPDATE accounts SET search_params=? WHERE name=?",
            (json.dumps(safe, ensure_ascii=False), name),
        )
        self.conn.commit()

    # ---- профиль кандидата ----

    def candidate_profile(self, name: str | None = None) -> dict | None:
        name = normalize_account_name(name or self.user)
        account_id = self.account_id(name)
        if not account_id:
            return None
        row = self.conn.execute(
            "SELECT full_name, address, email, phone, languages, "
            "preferred_letter_language, search_region, resume_city, scoring_preferences "
            "FROM candidate_profiles WHERE account_id=?",
            (account_id,),
        ).fetchone()
        if row is None:
            row = ("", "", "", "", "[]", "de", "", "", "{}")
        try:
            languages = [
                str(value).strip()
                for value in json.loads(row[4] or "[]")
                if str(value).strip()
            ]
        except (TypeError, ValueError):
            languages = []
        from .scoring_location import parse_location_preferences
        try:
            preferences = parse_location_preferences(json.loads(row[8] or "{}"))
        except (TypeError, ValueError):
            # Corrupt data must never grant a positive location score.
            preferences = parse_location_preferences(None)
        return {
            "profile_name": name,
            "full_name": str(row[0] or ""),
            "address": str(row[1] or ""),
            "email": str(row[2] or ""),
            "phone": str(row[3] or ""),
            "languages": languages,
            "preferred_letter_language": str(row[5] or "de"),
            "search_region": str(row[6] or ""),
            "resume_city": str(row[7] or ""),
            "scoring_preferences": preferences,
        }

    def set_candidate_profile(self, profile: dict, name: str | None = None) -> None:
        name = normalize_account_name(name or self.user)
        account_id = self.account_id(name)
        if not account_id:
            raise ValueError(f"Профиль «{name}» не существует")
        languages = profile.get("languages") or []
        if not isinstance(languages, list):
            raise ValueError("languages должен быть списком")
        preferred = str(profile.get("preferred_letter_language") or "de").strip()
        if preferred not in {"de", "en", "ru"}:
            raise ValueError("Неподдерживаемый язык письма")
        from .scoring_location import parse_location_preferences
        # Old callers updating contacts/languages preserve saved preferences.
        raw_preferences = profile.get("scoring_preferences", self.candidate_profile(name)["scoring_preferences"])
        preferences = parse_location_preferences(raw_preferences)
        values = {
            key: str(profile.get(key) or "").strip()
            for key in (
                "full_name",
                "address",
                "email",
                "phone",
                "search_region",
                "resume_city",
            )
        }
        self.conn.execute(
            """INSERT INTO candidate_profiles
               (account_id, full_name, address, email, phone, languages,
                preferred_letter_language, search_region, resume_city, scoring_preferences)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(account_id) DO UPDATE SET
                 full_name=excluded.full_name,
                 address=excluded.address,
                 email=excluded.email,
                 phone=excluded.phone,
                 languages=excluded.languages,
                 preferred_letter_language=excluded.preferred_letter_language,
                 search_region=excluded.search_region,
                 resume_city=excluded.resume_city,
                 scoring_preferences=excluded.scoring_preferences,
                 updated_at=CURRENT_TIMESTAMP""",
            (
                account_id,
                values["full_name"],
                values["address"],
                values["email"],
                values["phone"],
                json.dumps(
                    [str(value).strip() for value in languages if str(value).strip()],
                    ensure_ascii=False,
                ),
                preferred,
                values["search_region"],
                values["resume_city"],
                json.dumps(preferences, ensure_ascii=False),
            ),
        )
        self.conn.commit()

    # ---- metadata документов ----

    def document_metadata(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT filename, document_type, type_source, mime_type, size_bytes, "
            "sha256, extracted_chars, extracted_at, extract_error "
            "FROM documents WHERE account_id=? ORDER BY filename",
            (self._owner_id(),),
        ).fetchall()
        keys = (
            "filename",
            "document_type",
            "type_source",
            "mime_type",
            "size_bytes",
            "sha256",
            "extracted_chars",
            "extracted_at",
            "extract_error",
        )
        return [dict(zip(keys, row)) for row in rows]

    def upsert_document_metadata(self, metadata: dict) -> None:
        document_type = str(metadata.get("document_type") or "other")
        type_source = str(metadata.get("type_source") or "auto")
        if document_type not in {"cv", "curriculum", "certificate", "other"}:
            raise ValueError("Некорректный тип документа")
        if type_source not in {"auto", "manual"}:
            raise ValueError("Некорректный источник типа документа")
        self.conn.execute(
            """INSERT INTO documents
               (account_id, filename, document_type, type_source, mime_type,
                size_bytes, sha256, extracted_chars, extracted_at, extract_error)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP, ?)
               ON CONFLICT(account_id, filename) DO UPDATE SET
                 document_type=CASE WHEN documents.type_source='manual'
                   THEN documents.document_type ELSE excluded.document_type END,
                 type_source=CASE WHEN documents.type_source='manual'
                   THEN 'manual' ELSE excluded.type_source END,
                 mime_type=excluded.mime_type,
                 size_bytes=excluded.size_bytes,
                 sha256=excluded.sha256,
                 extracted_chars=excluded.extracted_chars,
                 extracted_at=CURRENT_TIMESTAMP,
                 extract_error=excluded.extract_error""",
            (
                self._owner_id(),
                str(metadata["filename"]),
                document_type,
                type_source,
                str(metadata.get("mime_type") or "application/octet-stream"),
                int(metadata.get("size_bytes") or 0),
                str(metadata.get("sha256") or ""),
                int(metadata.get("extracted_chars") or 0),
                str(metadata.get("extract_error") or ""),
            ),
        )
        self.conn.commit()

    def set_document_type(self, filename: str, document_type: str) -> bool:
        if document_type not in {"cv", "curriculum", "certificate", "other"}:
            raise ValueError("Некорректный тип документа")
        cursor = self.conn.execute(
            "UPDATE documents SET document_type=?, type_source='manual' "
            "WHERE account_id=? AND filename=?",
            (document_type, self._owner_id(), str(filename)),
        )
        self.conn.commit()
        return cursor.rowcount == 1

    def delete_document_metadata(self, filename: str) -> None:
        document_id = hashlib.sha256(f"{self._owner_id()}:{filename}".encode()).hexdigest()
        self.conn.execute(
            "DELETE FROM embedding_vectors WHERE account_id=? AND entity_kind='document' AND entity_id=?",
            (self._owner_id(), document_id),
        )
        self.conn.execute(
            "DELETE FROM documents WHERE account_id=? AND filename=?",
            (self._owner_id(), str(filename)),
        )
        self.conn.commit()

    # ---- фоновые задания ----

    @staticmethod
    def _validated_job_steps(steps: list[dict]) -> list[dict]:
        """Оставляет только сериализуемые поля, необходимые локальному worker."""
        if not isinstance(steps, list) or not steps:
            raise ValueError("Задание должно содержать хотя бы один шаг")
        clean = []
        for step in steps:
            if not isinstance(step, dict):
                raise ValueError("Некорректный шаг задания")
            cmd = str(step.get("cmd", "")).strip()
            args = step.get("args")
            if not cmd or not isinstance(args, list) or not all(
                isinstance(arg, str) for arg in args
            ):
                raise ValueError("Некорректная команда задания")
            def _is_allowed_search(args: list[str]) -> bool:
                if not args or args[0] != "search":
                    return False
                if len(args) == 1:
                    return True
                # flags должен идти парами: --flag value
                if (len(args) - 1) % 2 != 0:
                    return False
                seen: set[str] = set()
                for i in range(1, len(args), 2):
                    flag, value = args[i], args[i + 1]
                    if flag in seen:
                        return False
                    seen.add(flag)
                    if flag == "--limit":
                        if not value.isdigit() or not 1 <= int(value) <= 500:
                            return False
                    elif flag == "--location":
                        if len(value) > 300:
                            return False
                    elif flag == "--radius":
                        if not value.isdigit() or not 1 <= int(value) <= 500:
                            return False
                    else:
                        return False
                return seen.issubset({"--limit", "--location", "--radius"})

            allowed = (
                args in (
                    ["keywords", "--apply"],
                    ["score"],
                    ["score", "--missing"],
                    ["actualize"],
                )
                or _is_allowed_search(args)
            )
            if not allowed:
                raise ValueError("Команда задания не входит в server allowlist")
            clean.append(
                {
                    "cmd": cmd,
                    "args": list(args),
                    "code": None,
                    "output": "",
                }
            )
        return clean

    @staticmethod
    def _job_from_row(row: tuple | None) -> dict | None:
        if row is None:
            return None
        (
            run_id,
            kind,
            status,
            steps_json,
            llm_note,
            error,
            network_requests,
            network_wait_seconds,
            llm_requests,
            llm_wait_seconds,
            embedding_requests,
            embedding_wait_seconds,
        ) = row
        try:
            steps = json.loads(steps_json)
        except (TypeError, ValueError):
            steps = []
            error = error or "Повреждено сохранённое состояние задания."
            status = "failed"
        return {
            "run_id": int(run_id),
            "kind": str(kind),
            "status": str(status),
            "done": status in TERMINAL_JOB_STATUSES,
            "steps": steps,
            "llm_note": str(llm_note or ""),
            "error": str(error or ""),
            "metrics": {
                "network_requests": int(network_requests or 0),
                "network_wait_seconds": float(network_wait_seconds or 0),
                "llm_requests": int(llm_requests or 0),
                "llm_wait_seconds": float(llm_wait_seconds or 0),
                "embedding_requests": int(embedding_requests or 0),
                "embedding_wait_seconds": float(embedding_wait_seconds or 0),
            },
        }

    def create_job(
        self, kind: str, steps: list[dict], llm_note: str = ""
    ) -> tuple[int, bool]:
        """Создаёт job или возвращает уже активный job этого профиля."""
        if kind not in {"pipeline", "score_missing", "actualize"}:
            raise ValueError("Неизвестный тип задания")
        payload = json.dumps(
            self._validated_job_steps(steps), ensure_ascii=False, separators=(",", ":")
        )
        owner = self._owner_id()
        try:
            cursor = self.conn.execute(
                "INSERT INTO jobs(account_id, kind, steps_json, llm_note) "
                "VALUES (?, ?, ?, ?)",
                (owner, kind, payload, str(llm_note or "")),
            )
            self.conn.commit()
            return int(cursor.lastrowid), True
        except sqlite3.IntegrityError:
            self.conn.rollback()
            active = self.active_job()
            if active is not None:
                return int(active["run_id"]), False
            raise

    def active_job(self) -> dict | None:
        placeholders = ",".join("?" for _ in ACTIVE_JOB_STATUSES)
        row = self.conn.execute(
            f"SELECT run_id, kind, status, steps_json, llm_note, error, "
            "network_requests, network_wait_seconds, llm_requests, llm_wait_seconds, embedding_requests, embedding_wait_seconds "
            f"FROM jobs WHERE account_id=? AND status IN ({placeholders}) "
            "ORDER BY run_id DESC LIMIT 1",
            (self._owner_id(), *sorted(ACTIVE_JOB_STATUSES)),
        ).fetchone()
        return self._job_from_row(row)

    def job(self, run_id: int) -> dict | None:
        row = self.conn.execute(
            "SELECT run_id, kind, status, steps_json, llm_note, error, "
            "network_requests, network_wait_seconds, llm_requests, llm_wait_seconds, embedding_requests, embedding_wait_seconds "
            "FROM jobs WHERE account_id=? AND run_id=?",
            (self._owner_id(), int(run_id)),
        ).fetchone()
        return self._job_from_row(row)

    def claim_job(self, run_id: int) -> bool:
        cursor = self.conn.execute(
            "UPDATE jobs SET status='running', started_at=CURRENT_TIMESTAMP, "
            "updated_at=CURRENT_TIMESTAMP "
            "WHERE account_id=? AND run_id=? AND status='queued'",
            (self._owner_id(), int(run_id)),
        )
        self.conn.commit()
        return cursor.rowcount == 1

    def set_job_rate_limit_wait(self, run_id: int, waiting: bool) -> bool:
        """Macht einen 429-Wartezustand für das Web-Polling sichtbar."""
        source, target = (
            ("running", "waiting_rate_limit")
            if waiting
            else ("waiting_rate_limit", "running")
        )
        cursor = self.conn.execute(
            "UPDATE jobs SET status=?, updated_at=CURRENT_TIMESTAMP "
            "WHERE account_id=? AND run_id=? AND status=?",
            (target, self._owner_id(), int(run_id), source),
        )
        self.conn.commit()
        return cursor.rowcount == 1

    def record_job_resource_usage(
        self,
        run_id: int,
        resource: str,
        wait_seconds: float,
    ) -> bool:
        if resource not in {"network", "llm", "embedding"}:
            raise ValueError("Unbekannte Job-Ressource")
        wait = max(0.0, float(wait_seconds))
        requests_column = f"{resource}_requests"
        wait_column = f"{resource}_wait_seconds"
        cursor = self.conn.execute(
            f"UPDATE jobs SET {requests_column}={requests_column}+1, "
            f"{wait_column}={wait_column}+?, updated_at=CURRENT_TIMESTAMP "
            "WHERE account_id=? AND run_id=?",
            (wait, self._owner_id(), int(run_id)),
        )
        self.conn.commit()
        return cursor.rowcount == 1

    def defer_llm_rate_limit(
        self,
        provider_key: str,
        delay: float,
        *,
        now: float | None = None,
    ) -> float:
        if not provider_key:
            raise ValueError("Provider key must not be empty")
        retry_after = (time.time() if now is None else float(now)) + max(
            0.0, float(delay)
        )
        self.conn.execute(
            """INSERT INTO llm_rate_limits
               (account_id, provider_key, retry_after, updated_at)
               VALUES (?, ?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(account_id, provider_key) DO UPDATE SET
                   retry_after=MAX(llm_rate_limits.retry_after, excluded.retry_after),
                   updated_at=CURRENT_TIMESTAMP""",
            (self._owner_id(), provider_key, retry_after),
        )
        self.conn.commit()
        row = self.conn.execute(
            "SELECT retry_after FROM llm_rate_limits "
            "WHERE account_id=? AND provider_key=?",
            (self._owner_id(), provider_key),
        ).fetchone()
        return float(row[0])

    def llm_rate_limit_delay(
        self,
        provider_key: str,
        *,
        now: float | None = None,
    ) -> float:
        row = self.conn.execute(
            "SELECT retry_after FROM llm_rate_limits "
            "WHERE account_id=? AND provider_key=?",
            (self._owner_id(), provider_key),
        ).fetchone()
        if row is None:
            return 0.0
        return max(0.0, float(row[0]) - (time.time() if now is None else float(now)))

    def clear_llm_rate_limit(self, provider_key: str) -> None:
        self.conn.execute(
            "DELETE FROM llm_rate_limits WHERE account_id=? AND provider_key=?",
            (self._owner_id(), provider_key),
        )
        self.conn.commit()

    def save_llm_quota(self, provider_key: str, headers: dict) -> None:
        """Последний снимок x-ratelimit-заголовков провайдера (с реальных вызовов)."""
        if not provider_key:
            raise ValueError("Provider key must not be empty")
        payload = json.dumps(headers, sort_keys=True)
        self.conn.execute(
            """INSERT INTO llm_rate_limits
               (account_id, provider_key, retry_after, quota_headers,
                quota_updated_at)
               VALUES (?, ?, 0, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(account_id, provider_key) DO UPDATE SET
                   quota_headers=excluded.quota_headers,
                   quota_updated_at=CURRENT_TIMESTAMP""",
            (self._owner_id(), provider_key, payload),
        )
        self.conn.commit()

    def llm_quota(self, provider_key: str) -> tuple[dict, str] | None:
        """Снимок квот провайдера или None; возвращает (заголовки, время)."""
        row = self.conn.execute(
            "SELECT quota_headers, quota_updated_at FROM llm_rate_limits "
            "WHERE account_id=? AND provider_key=?",
            (self._owner_id(), provider_key),
        ).fetchone()
        if row is None or not row[0]:
            return None
        try:
            headers = json.loads(row[0])
        except ValueError:
            return None
        if not isinstance(headers, dict):
            return None
        return headers, row[1]

    def append_job_output(self, run_id: int, step_index: int, text: str) -> None:
        job = self.job(run_id)
        if job is None or job["done"]:
            return
        if not 0 <= step_index < len(job["steps"]):
            raise IndexError("Шаг задания не найден")
        output = job["steps"][step_index].get("output", "") + str(text)
        if len(output) > JOB_LOG_LIMIT:
            marker = "[... начало лога обрезано ...]\n"
            output = marker + output[-(JOB_LOG_LIMIT - len(marker)) :]
        job["steps"][step_index]["output"] = output
        self.conn.execute(
            "UPDATE jobs SET steps_json=?, updated_at=CURRENT_TIMESTAMP "
            "WHERE account_id=? AND run_id=?",
            (
                json.dumps(job["steps"], ensure_ascii=False, separators=(",", ":")),
                self._owner_id(),
                int(run_id),
            ),
        )
        self.conn.commit()

    def set_job_step_code(self, run_id: int, step_index: int, code: int) -> None:
        job = self.job(run_id)
        if job is None or job["done"]:
            return
        if not 0 <= step_index < len(job["steps"]):
            raise IndexError("Шаг задания не найден")
        job["steps"][step_index]["code"] = int(code)
        self.conn.execute(
            "UPDATE jobs SET steps_json=?, updated_at=CURRENT_TIMESTAMP "
            "WHERE account_id=? AND run_id=?",
            (
                json.dumps(job["steps"], ensure_ascii=False, separators=(",", ":")),
                self._owner_id(),
                int(run_id),
            ),
        )
        self.conn.commit()

    def set_job_step_warning(self, run_id: int, step_index: int, warning: str) -> None:
        """Некритичное предупреждение шага (например, частичный успех search)."""
        job = self.job(run_id)
        if job is None or job["done"]:
            return
        if not 0 <= step_index < len(job["steps"]):
            raise IndexError("Шаг задания не найден")
        job["steps"][step_index]["warning"] = str(warning)[:500]
        self.conn.execute(
            "UPDATE jobs SET steps_json=?, updated_at=CURRENT_TIMESTAMP "
            "WHERE account_id=? AND run_id=?",
            (
                json.dumps(job["steps"], ensure_ascii=False, separators=(",", ":")),
                self._owner_id(),
                int(run_id),
            ),
        )
        self.conn.commit()

    def finish_job(self, run_id: int, status: str, error: str = "") -> bool:
        if status not in TERMINAL_JOB_STATUSES:
            raise ValueError("Некорректное конечное состояние задания")
        cursor = self.conn.execute(
            "UPDATE jobs SET status=?, error=?, finished_at=CURRENT_TIMESTAMP, "
            "updated_at=CURRENT_TIMESTAMP "
            "WHERE account_id=? AND run_id=? AND status IN "
            "('queued', 'running', 'waiting_rate_limit')",
            (status, str(error or ""), self._owner_id(), int(run_id)),
        )
        self.conn.commit()
        return cursor.rowcount == 1

    def interrupt_active_jobs(self) -> int:
        """Startup recovery: освобождает профили после остановки web worker."""
        cursor = self.conn.execute(
            "UPDATE jobs SET status='interrupted', "
            "error=CASE WHEN error='' THEN ? ELSE error END, "
            "finished_at=CURRENT_TIMESTAMP, updated_at=CURRENT_TIMESTAMP "
            "WHERE status IN ('queued', 'running', 'waiting_rate_limit')",
            ("Предыдущий web-процесс был остановлен; запустите задание снова.",),
        )
        self.conn.commit()
        return cursor.rowcount

    def prune_finished_jobs(self, older_than_days: int = 30) -> int:
        """Retention: удаляет завершённые задания старше N дней (с их логами)."""
        if older_than_days < 0:
            raise ValueError("older_than_days must be >= 0")
        cursor = self.conn.execute(
            "DELETE FROM jobs WHERE status IN "
            "('succeeded', 'failed', 'cancelled', 'timed_out', 'interrupted') "
            "AND finished_at IS NOT NULL AND finished_at <= datetime('now', ?)",
            (f"-{int(older_than_days)} days",),
        )
        self.conn.commit()
        return cursor.rowcount

    # ---- вакансии ----

    def _owner_id(self) -> str:
        account_id = self.account_id(self.user)
        if not account_id:
            raise ValueError(f"Профиль «{self.user}» не существует")
        return account_id

    def upsert(self, v: Vacancy) -> None:
        """Aktualisiert Source-Felder, ohne Score oder Bewerbungsstatus zu verlieren."""
        score_state = "scored" if v.score is not None else v.score_state
        description = str(v.description or "")
        description_hash = (
            hashlib.sha256(description.encode("utf-8")).hexdigest()
            if description
            else ""
        )
        description_fetched_at = (
            v.description_fetched_at or (_utc_now() if description else None)
        )
        source_metadata = v.source_metadata if isinstance(v.source_metadata, dict) else {}
        if self._description_edited(v.refnr):
            # Ручная правка описания имеет приоритет над свежими данными источника.
            description = ""
            description_hash = ""
            description_fetched_at = None
            source_metadata["description_edited"] = True
        self.conn.execute(
            """INSERT INTO vacancies
               (account_id, refnr, title, company, location, url, source,
                published_at, last_seen_at, source_status, archived_at,
                source_metadata, description, description_fetched_at,
                description_hash, score, score_state, user)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
               ON CONFLICT(account_id, refnr) DO UPDATE SET
                 title=excluded.title,
                 company=excluded.company,
                 location=excluded.location,
                 url=excluded.url,
                 source=excluded.source,
                 published_at=CASE WHEN excluded.published_at<>''
                     THEN excluded.published_at ELSE vacancies.published_at END,
                 last_seen_at=excluded.last_seen_at,
                 source_status='active',
                 archived_at=NULL,
                 source_metadata=CASE WHEN excluded.source_metadata<>'{}'
                     THEN excluded.source_metadata ELSE vacancies.source_metadata END,
                 description=CASE WHEN excluded.description<>''
                     THEN excluded.description ELSE vacancies.description END,
                 description_fetched_at=CASE WHEN excluded.description<>''
                     THEN excluded.description_fetched_at
                     ELSE vacancies.description_fetched_at END,
                 description_hash=CASE WHEN excluded.description<>''
                     THEN excluded.description_hash ELSE vacancies.description_hash END""",
            (
                self._owner_id(),
                v.refnr,
                v.title,
                v.company,
                v.location,
                v.url,
                v.source,
                v.published,
                v.last_seen_at or _utc_now(),
                v.source_status if v.source_status in {
                    "active", "error", "unavailable", "unsupported"
                } else "active",
                v.archived_at or None,
                json.dumps(source_metadata, ensure_ascii=False, sort_keys=True),
                description,
                description_fetched_at,
                description_hash,
                v.score,
                score_state,
                self.user,
            ),
        )
        self.conn.commit()

    def add(self, v: Vacancy) -> None:
        """Kompatibler Name; Suchergebnisse werden seit schema v8 upserted."""
        self.upsert(v)

    def update_score(self, refnr: str, score: float):
        self.conn.execute(
            "UPDATE vacancies SET score=?, score_state='scored', score_skip_reason='' "
            "WHERE account_id=? AND refnr=?",
            (score, self._owner_id(), refnr),
        )
        self.conn.commit()

    def update_score_details(
        self,
        refnr: str,
        score: float,
        reasons: list[str],
        missing: list[str],
        *,
        provider: str = "",
        model: str = "",
        prompt_version: str = "",
        candidate_hash: str = "",
        vacancy_hash: str = "",
        description_hash: str = "",
    ) -> None:
        self.conn.execute(
            "UPDATE vacancies SET score=?, score_state='scored', "
            "score_reasons=?, score_missing=?, score_provider=?, score_model=?, "
            "score_prompt_version=?, score_candidate_hash=?, score_vacancy_hash=?, "
            "score_description_hash=?, score_skip_reason='', score_engine='llm', "
            "score_algorithm_version='', score_settings_hash='', score_details_json='{}' "
            "WHERE account_id=? AND refnr=?",
            (
                score,
                json.dumps(reasons, ensure_ascii=False),
                json.dumps(missing, ensure_ascii=False),
                provider,
                model,
                prompt_version,
                candidate_hash,
                vacancy_hash,
                description_hash,
                self._owner_id(),
                refnr,
            ),
        )
        self.conn.commit()

    def mark_score_error(self, refnr: str, reason: str = "format_error", *, engine: str = "llm") -> None:
        if engine not in {"llm", "algorithmic"}:
            raise ValueError("Invalid scoring engine")
        self.conn.execute(
            "UPDATE vacancies SET score=NULL, score_state='score_error', "
            "score_reasons='[]', score_missing='[]', score_skip_reason=?, "
            "score_engine=?, score_details_json='{}', score_algorithm_version='', score_settings_hash='' "
            "WHERE account_id=? AND refnr=?",
            (reason, engine, self._owner_id(), refnr),
        )
        self.conn.commit()

    def mark_score_skipped(self, refnr: str, reason: str) -> None:
        if not reason.strip() or len(reason) > 200:
            raise ValueError("Invalid scoring skip reason")
        self.conn.execute(
            "UPDATE vacancies SET score=NULL, score_state='unscored', "
            "score_reasons='[]', score_missing='[]', score_skip_reason=? "
            "WHERE account_id=? AND refnr=?",
            (reason.strip(), self._owner_id(), refnr),
        )
        self.conn.commit()

    def needs_scoring(
        self,
        refnr: str,
        *,
        provider: str,
        model: str,
        prompt_version: str,
        candidate_hash: str,
        vacancy_hash: str,
        description_hash: str,
        force: bool = False,
    ) -> bool:
        if force:
            return True
        row = self.conn.execute(
            "SELECT score_state, score_provider, score_model, score_prompt_version, "
            "score_candidate_hash, score_vacancy_hash, score_description_hash "
            "FROM vacancies WHERE account_id=? AND refnr=? AND score_engine='llm'",
            (self._owner_id(), refnr),
        ).fetchone()
        if row is None or row[0] != "scored":
            return True
        return tuple(row[1:]) != (
            provider,
            model,
            prompt_version,
            candidate_hash,
            vacancy_hash,
            description_hash,
        )

    def score_details(self, refnr: str) -> dict | None:
        row = self.conn.execute(
            "SELECT score, score_state, score_reasons, score_missing, "
            "score_provider, score_model, score_prompt_version, "
            "score_candidate_hash, score_vacancy_hash, score_description_hash, "
            "score_skip_reason "
            ", score_engine, score_algorithm_version, score_settings_hash, score_details_json "
            "FROM vacancies WHERE account_id=? AND refnr=?",
            (self._owner_id(), refnr),
        ).fetchone()
        if row is None:
            return None
        return {
            "score": row[0],
            "state": row[1],
            "reasons": json.loads(row[2] or "[]"),
            "missing": json.loads(row[3] or "[]"),
            "provider": row[4] or "",
            "model": row[5] or "",
            "prompt_version": row[6] or "",
            "candidate_hash": row[7] or "",
            "vacancy_hash": row[8] or "",
            "description_hash": row[9] or "",
            "skip_reason": row[10] or "",
            "engine": row[11], "algorithm_version": row[12], "settings_hash": row[13],
            "details": json.loads(row[14] or "{}"),
        }

    def score_summaries(self) -> dict[str, dict]:
        """One account-scoped query for table badges; return no CV excerpts."""
        rows = self.conn.execute(
            "SELECT refnr, score_engine, score_algorithm_version, score_details_json "
            "FROM vacancies WHERE account_id=? AND archived_at IS NULL",
            (self._owner_id(),),
        ).fetchall()
        result = {}
        for refnr, engine, version, raw in rows:
            try:
                details = json.loads(raw or "{}")
                preliminary = details.get("preliminary") if isinstance(details, dict) else None
            except (ValueError, TypeError):
                preliminary = None
            result[refnr] = {"engine": engine, "version": version,
                             "preliminary": preliminary is not False if engine == "algorithmic" else False}
        return result

    def embedding_vectors(self, kind: str, entity_id: str, backend_id: str, role: str) -> dict:
        rows = self.conn.execute(
            "SELECT text_hash, vector_json FROM embedding_vectors WHERE account_id=? AND entity_kind=? AND entity_id=? AND backend_id=? AND role=?",
            (self._owner_id(), kind, entity_id, backend_id, role),
        ).fetchall()
        result = {}
        for digest, vector in rows:
            try:
                result[digest] = json.loads(vector)
            except (ValueError, TypeError):
                continue
        return result

    def replace_embedding_vectors(self, kind: str, entity_id: str, backend_id: str, role: str, dimensions: int, vectors: dict) -> None:
        from .embeddings import unit_vector
        if kind not in {"document", "vacancy"} or role not in {"query", "document"}:
            raise ValueError("Invalid embedding entity or role")
        checked = {key: unit_vector(vector, dimensions) for key, vector in vectors.items()}
        owner = self._owner_id()
        with self.conn:
            self.conn.execute("DELETE FROM embedding_vectors WHERE account_id=? AND entity_kind=? AND entity_id=? AND role=?", (owner, kind, entity_id, role))
            self.conn.executemany(
                "INSERT INTO embedding_vectors(account_id, entity_kind, entity_id, backend_id, role, text_hash, dimensions, vector_json) VALUES (?,?,?,?,?,?,?,?)",
                [(owner, kind, entity_id, backend_id, role, digest, dimensions, json.dumps(vector, allow_nan=False)) for digest, vector in checked.items()],
            )

    def prune_document_vectors(self, document_ids: set[str]) -> None:
        # Called after collecting the *complete* current document inventory.
        owner = self._owner_id()
        rows = self.conn.execute("SELECT DISTINCT entity_id FROM embedding_vectors WHERE account_id=? AND entity_kind='document'", (owner,)).fetchall()
        with self.conn:
            self.conn.executemany("DELETE FROM embedding_vectors WHERE account_id=? AND entity_kind='document' AND entity_id=?", [(owner, row[0]) for row in rows if row[0] not in document_ids])

    def needs_algorithmic_scoring(self, refnr: str, *, version: str, settings_hash: str, model: str, candidate_hash: str, vacancy_hash: str, force: bool = False) -> bool:
        details = self.score_details(refnr)
        return force or not details or details["state"] != "scored" or (
            details["engine"], details["algorithm_version"], details["settings_hash"],
            details["model"], details["candidate_hash"], details["vacancy_hash"],
        ) != ("algorithmic", version, settings_hash, model, candidate_hash, vacancy_hash)

    def update_algorithmic_score(self, refnr: str, result: dict, *, candidate_hash: str, vacancy_hash: str, description_hash: str, settings_hash: str) -> None:
        self.conn.execute(
            "UPDATE vacancies SET score=?, score_state='scored', score_reasons=?, score_missing=?, "
            "score_engine='algorithmic', score_algorithm_version=?, score_settings_hash=?, score_details_json=?, "
            "score_provider='local-embeddings', score_model=?, score_prompt_version='', score_candidate_hash=?, "
            "score_vacancy_hash=?, score_description_hash=?, score_skip_reason='' WHERE account_id=? AND refnr=?",
            (result["score"], json.dumps(result["reasons"], ensure_ascii=False), json.dumps(result["missing"], ensure_ascii=False),
             result["version"], settings_hash, json.dumps(result, ensure_ascii=False, allow_nan=False), result["backend"]["key"],
             candidate_hash, vacancy_hash, description_hash, self._owner_id(), refnr),
        )
        self.conn.commit()

    def record_artifact(
        self,
        kind: str,
        status: str,
        *,
        refnr: str | None = None,
        provider: str = "",
        model: str = "",
        prompt_version: str = "",
        candidate_hash: str = "",
        vacancy_hash: str = "",
        output_path: str = "",
        error: str = "",
        details: dict | None = None,
    ) -> int:
        if not kind.strip():
            raise ValueError("Artifact kind must not be empty")
        if status not in {"succeeded", "failed"}:
            raise ValueError("Invalid artifact status")
        cursor = self.conn.execute(
            """INSERT INTO artifacts
               (account_id, refnr, kind, status, provider, model,
                prompt_version, candidate_hash, vacancy_hash, output_path,
                error, details_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                self._owner_id(),
                refnr,
                kind.strip(),
                status,
                provider,
                model,
                prompt_version,
                candidate_hash,
                vacancy_hash,
                output_path,
                error,
                json.dumps(details or {}, ensure_ascii=False, sort_keys=True),
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def artifacts(self, kind: str | None = None) -> list[dict]:
        sql = (
            "SELECT artifact_id, refnr, kind, status, provider, model, "
            "prompt_version, candidate_hash, vacancy_hash, output_path, error, "
            "details_json, created_at FROM artifacts WHERE account_id=?"
        )
        params: list[object] = [self._owner_id()]
        if kind is not None:
            sql += " AND kind=?"
            params.append(kind)
        sql += " ORDER BY artifact_id DESC"
        rows = self.conn.execute(sql, params).fetchall()
        return [
            {
                "artifact_id": row[0],
                "refnr": row[1],
                "kind": row[2],
                "status": row[3],
                "provider": row[4],
                "model": row[5],
                "prompt_version": row[6],
                "candidate_hash": row[7],
                "vacancy_hash": row[8],
                "output_path": row[9],
                "error": row[10],
                "details": json.loads(row[11] or "{}"),
                "created_at": row[12],
            }
            for row in rows
        ]

    def vacancy_review(self, refnr: str) -> dict | None:
        row = self.conn.execute(
            "SELECT grade,reason,candidate_hash,vacancy_hash,updated_at FROM vacancy_reviews "
            "WHERE account_id=? AND refnr=?", (self._owner_id(), refnr),
        ).fetchone()
        return dict(zip(("grade", "reason", "candidate_hash", "vacancy_hash", "updated_at"), row)) if row else None

    def save_vacancy_review(self, refnr: str, grade: int, reason: str, *, candidate_hash: str, vacancy_hash: str, inputs: dict) -> None:
        if type(grade) is not int or grade not in {0, 1, 2}:
            raise ValueError("Invalid review grade")
        if not isinstance(reason, str) or len(reason) > 2000:
            raise ValueError("Review comment must be at most 2000 characters")
        from .vacancy_feedback import vacancy_revision

        encoded = json.dumps(inputs, ensure_ascii=False, allow_nan=False)
        owner = self._owner_id()
        with self.conn:
            self.conn.execute("BEGIN IMMEDIATE")
            current = self.by_refnr(refnr)
            if current is None:
                raise ValueError("Vacancy not found in current account")
            if vacancy_revision(current) != vacancy_hash:
                raise ValueError("Vacancy changed during review capture")
            self.conn.execute("""INSERT INTO vacancy_reviews
                (account_id,refnr,grade,reason,candidate_hash,vacancy_hash,inputs_json,updated_at)
                VALUES (?,?,?,?,?,?,?,?) ON CONFLICT(account_id,refnr) DO UPDATE SET
                grade=excluded.grade,reason=excluded.reason,candidate_hash=excluded.candidate_hash,
                vacancy_hash=excluded.vacancy_hash,inputs_json=excluded.inputs_json,updated_at=excluded.updated_at""",
                (owner, refnr, grade, reason.strip(), candidate_hash, vacancy_hash, encoded, _utc_now()))

    def clear_vacancy_review(self, refnr: str) -> None:
        self.conn.execute("DELETE FROM vacancy_reviews WHERE account_id=? AND refnr=?", (self._owner_id(), refnr))
        self.conn.commit()

    def scoring_history(self, refnr: str, *, limit: int = 20) -> list[dict]:
        """Bounded successful-score history for one vacancy in this account."""
        if type(limit) is not int or not 1 <= limit <= 100:
            raise ValueError("Invalid scoring history limit")
        rows = self.conn.execute(
            "SELECT provider, model, prompt_version, details_json, created_at "
            "FROM artifacts WHERE account_id=? AND refnr=? AND kind='scoring' AND status='succeeded' "
            "ORDER BY artifact_id DESC LIMIT ?", (self._owner_id(), refnr, limit),
        ).fetchall()
        history = []
        for provider, model, prompt, raw, created_at in rows:
            try:
                details = json.loads(raw)
                score = details.get("score")
                if type(score) not in {int, float} or not 0 <= score <= 100 or not math.isfinite(score):
                    continue
            except (TypeError, ValueError, AttributeError):
                continue
            algorithmic = provider == "local-embeddings"
            backend = details.get("backend")
            history.append({"score": score, "created_at": created_at,
                            "engine": "algorithmic" if algorithmic else "llm",
                            "version": details.get("version", "") if algorithmic else prompt,
                            "model": backend.get("model", model) if algorithmic and isinstance(backend, dict) else model})
        return history

    def _description_edited(self, refnr: str) -> bool:
        row = self.conn.execute(
            "SELECT source_metadata FROM vacancies WHERE account_id=? AND refnr=?",
            (self._owner_id(), refnr),
        ).fetchone()
        return row is not None and _metadata_flag(row[0], "description_edited")

    def _mark_description_edited(self, refnr: str) -> None:
        row = self.conn.execute(
            "SELECT source_metadata FROM vacancies WHERE account_id=? AND refnr=?",
            (self._owner_id(), refnr),
        ).fetchone()
        if row is None:
            return
        try:
            metadata = json.loads(row[0] or "{}")
        except (TypeError, ValueError):
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["description_edited"] = True
        self.conn.execute(
            "UPDATE vacancies SET source_metadata=? WHERE account_id=? AND refnr=?",
            (
                json.dumps(metadata, ensure_ascii=False, sort_keys=True),
                self._owner_id(),
                refnr,
            ),
        )

    def update_description(self, refnr: str, description: str, edited: bool = False):
        """Сохраняет описание; edited=True защищает ручную правку от перезаписи."""
        description = str(description or "")
        digest = hashlib.sha256(description.encode("utf-8")).hexdigest()
        self.conn.execute(
            "UPDATE vacancies SET description=?, description_fetched_at=?, "
            "description_hash=? WHERE account_id=? AND refnr=?",
            (description, _utc_now(), digest, self._owner_id(), refnr),
        )
        if edited:
            self._mark_description_edited(refnr)
        self.conn.commit()

    def description_needs_refresh(self, refnr: str, ttl_hours: int = 168) -> bool:
        if isinstance(ttl_hours, bool) or not 1 <= ttl_hours <= 8760:
            raise ValueError("description TTL must be between 1 and 8760 hours")
        row = self.conn.execute(
            "SELECT description, description_fetched_at, source_status, source_metadata "
            "FROM vacancies WHERE account_id=? AND refnr=?",
            (self._owner_id(), refnr),
        ).fetchone()
        if row is None or row[2] in {"unavailable", "unsupported"}:
            return False
        if _metadata_flag(row[3], "description_edited"):
            return False
        if row[1] is None or row[1] == "":
            return True
        try:
            fetched_at = datetime.fromisoformat(str(row[1]).replace("Z", "+00:00"))
        except ValueError:
            return True
        if fetched_at.tzinfo is None:
            fetched_at = fetched_at.replace(tzinfo=timezone.utc)
        return datetime.now(timezone.utc) - fetched_at >= timedelta(hours=ttl_hours)

    def mark_seen(self, refnr: str) -> None:
        self.conn.execute(
            "UPDATE vacancies SET last_seen_at=?, source_status='active', archived_at=NULL "
            "WHERE account_id=? AND refnr=?",
            (_utc_now(), self._owner_id(), refnr),
        )
        self.conn.commit()

    def mark_source_error(self, refnr: str) -> None:
        self.conn.execute(
            "UPDATE vacancies SET source_status='error' "
            "WHERE account_id=? AND refnr=? AND source_status NOT IN "
            "('unavailable', 'unsupported')",
            (self._owner_id(), refnr),
        )
        self.conn.commit()

    def mark_unavailable(self, refnr: str) -> None:
        self.conn.execute(
            "UPDATE vacancies SET source_status='unavailable', "
            "archived_at=COALESCE(archived_at, ?) WHERE account_id=? AND refnr=?",
            (_utc_now(), self._owner_id(), refnr),
        )
        self.conn.commit()

    def purge_archived(self, older_than_days: int = 30) -> int:
        if isinstance(older_than_days, bool) or not 1 <= older_than_days <= 3650:
            raise ValueError("Retention must be between 1 and 3650 days")
        cursor = self.conn.execute(
            "DELETE FROM vacancies WHERE account_id=? AND archived_at IS NOT NULL "
            "AND status<>'applied' AND archived_at<=datetime('now', ?)",
            (self._owner_id(), f"-{older_than_days} days"),
        )
        self.conn.commit()
        return int(cursor.rowcount)

    def vacancy_lifecycle(self, refnr: str) -> dict | None:
        row = self.conn.execute(
            "SELECT published_at, last_seen_at, description_fetched_at, "
            "description_hash, source_status, archived_at, source_metadata "
            "FROM vacancies WHERE account_id=? AND refnr=?",
            (self._owner_id(), refnr),
        ).fetchone()
        if row is None:
            return None
        try:
            metadata = json.loads(row[6] or "{}")
        except (TypeError, ValueError):
            metadata = {}
        return {
            "published_at": row[0] or "",
            "last_seen_at": row[1] or "",
            "description_fetched_at": row[2] or "",
            "description_hash": row[3] or "",
            "source_status": row[4] or "active",
            "archived_at": row[5] or "",
            "source_metadata": metadata,
        }

    def update_translation(self, refnr: str, translation: str):
        self.conn.execute(
            "UPDATE vacancies SET translation=? WHERE account_id=? AND refnr=?",
            (translation, self._owner_id(), refnr),
        )
        self.conn.commit()

    def set_status(self, refnr: str, status: str):
        self.conn.execute(
            "UPDATE vacancies SET status=?, applied_at=CURRENT_TIMESTAMP "
            "WHERE account_id=? AND refnr=?",
            (status, self._owner_id(), refnr),
        )
        self.conn.commit()

    def applied_at(self, refnr: str) -> str:
        """Дата подтверждения отклика (applied_at) или пустая строка."""
        row = self.conn.execute(
            "SELECT applied_at FROM vacancies WHERE account_id=? AND refnr=?",
            (self._owner_id(), refnr),
        ).fetchone()
        return str(row[0] or "") if row else ""

    def application_status(self, refnr: str) -> str:
        """Текущий application статус вакансии текущего профиля."""
        row = self.conn.execute(
            "SELECT status FROM vacancies WHERE account_id=? AND refnr=?",
            (self._owner_id(), refnr),
        ).fetchone()
        return str(row[0]) if row else ""

    def schema_version(self) -> int:
        """Актуальная версия схемы этой DB (0, если миграций ещё не было)."""
        try:
            row = self.conn.execute(
                "SELECT MAX(version) FROM schema_migrations"
            ).fetchone()
        except sqlite3.OperationalError:
            return 0
        return int(row[0]) if row and row[0] else 0

    def active_job_count(self) -> int:
        """Число активных заданий по всем профилям (без деталей)."""
        placeholders = ",".join("?" for _ in ACTIVE_JOB_STATUSES)
        row = self.conn.execute(
            f"SELECT COUNT(*) FROM jobs WHERE status IN ({placeholders})",
            tuple(sorted(ACTIVE_JOB_STATUSES)),
        ).fetchone()
        return int(row[0]) if row else 0

    def top(self, min_score: float, limit: int = 10) -> list[Vacancy]:
        rows = self.conn.execute(
            "SELECT refnr,title,company,location,url,source,published_at,description,"
            "score,translation,score_state,last_seen_at,description_fetched_at,"
            "description_hash,source_status,archived_at,source_metadata "
            "FROM vacancies WHERE account_id=? AND archived_at IS NULL AND score>=? "
            "ORDER BY score DESC LIMIT ?",
            (self._owner_id(), min_score, limit),
        ).fetchall()
        return [self._row_to_vacancy(r) for r in rows]

    def by_refnr(self, refnr: str) -> Vacancy | None:
        row = self.conn.execute(
            "SELECT refnr,title,company,location,url,source,published_at,description,"
            "score,translation,score_state,last_seen_at,description_fetched_at,"
            "description_hash,source_status,archived_at,source_metadata "
            "FROM vacancies WHERE account_id=? AND refnr=?",
            (self._owner_id(), refnr),
        ).fetchone()
        return self._row_to_vacancy(row) if row else None

    def vacancies_for_dedup(self) -> list[Vacancy]:
        """Account-scoped Vergleichsmenge für source-übergreifendes Such-Dedup."""
        rows = self.conn.execute(
            "SELECT refnr,title,company,location,url,source,published_at,description,"
            "score,translation,score_state,last_seen_at,description_fetched_at,"
            "description_hash,source_status,archived_at,source_metadata "
            "FROM vacancies WHERE account_id=? AND archived_at IS NULL",
            (self._owner_id(),),
        ).fetchall()
        return [self._row_to_vacancy(row) for row in rows]

    @staticmethod
    def _row_to_vacancy(r: tuple) -> Vacancy:
        (
            refnr,
            title,
            company,
            location,
            url,
            source,
            published,
            description,
            score,
            translation,
            score_state,
            last_seen_at,
            description_fetched_at,
            description_hash,
            source_status,
            archived_at,
            source_metadata,
        ) = r
        try:
            metadata = json.loads(source_metadata or "{}")
        except (TypeError, ValueError):
            metadata = {}
        return Vacancy(
            refnr=refnr, title=title, company=company, location=location,
            url=url, source=source, published=published or "",
            description=description or "", score=score,
            translation=translation or "", score_state=score_state,
            last_seen_at=last_seen_at or "",
            description_fetched_at=description_fetched_at or "",
            description_hash=description_hash or "",
            source_status=source_status or "active",
            archived_at=archived_at or "", source_metadata=metadata,
        )

    def list_all(self, include_archived: bool = False) -> list[tuple]:
        sql = (
            "SELECT refnr,title,company,source,score,status,score_state,"
            "score_skip_reason,location,"
            "CASE WHEN description<>'' THEN 1 ELSE 0 END,"
            "CASE WHEN translation<>'' THEN 1 ELSE 0 END "
            "FROM vacancies WHERE account_id=?"
        )
        if not include_archived:
            sql += " AND archived_at IS NULL"
        sql += " ORDER BY created_at DESC"
        return self.conn.execute(sql, (self._owner_id(),)).fetchall()

    def list_archived(self) -> list[tuple]:
        return self.conn.execute(
            "SELECT refnr,title,company,source,status,source_status,archived_at "
            "FROM vacancies WHERE account_id=? AND archived_at IS NOT NULL "
            "ORDER BY archived_at DESC",
            (self._owner_id(),),
        ).fetchall()

    def ba_refnrs(self) -> list[str]:
        return [
            r[0]
            for r in self.conn.execute(
                "SELECT refnr FROM vacancies "
                "WHERE account_id=? AND source='arbeitsagentur' AND archived_at IS NULL",
                (self._owner_id(),),
            ).fetchall()
        ]

    def delete(self, refnr: str):
        self.conn.execute("DELETE FROM embedding_vectors WHERE account_id=? AND entity_kind='vacancy' AND entity_id=?", (self._owner_id(), refnr))
        self.conn.execute(
            "DELETE FROM vacancies WHERE account_id=? AND refnr=?",
            (self._owner_id(), refnr),
        )
        self.conn.commit()

    def close(self):
        self.conn.close()
