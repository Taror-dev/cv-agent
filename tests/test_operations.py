"""Этапы 7–8: структурный лог заданий и restore из backup на пустом каталоге."""

import json
import shutil
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cv_agent import config
from cv_agent.models import Vacancy
from cv_agent.store import LATEST_SCHEMA_VERSION, Store
from cv_agent.webapp import app


def _root() -> Path:
    return Path(config.DATA_DIR).parent


def _login(client):
    r = client.post("/login", data={"account": "Илья"})
    assert r.status_code == 302


def _fake_criteria(monkeypatch):
    from cv_agent.config import Criteria

    c = Criteria(
        region="88239 Wangen im Allgäu",
        radius_km=50,
        keywords=["Python"],
        exclude_keywords=[],
        min_score=60,
    )
    monkeypatch.setattr(
        "cv_agent.webapp.load_criteria", lambda user=None: c
    )


def _fake_run(monkeypatch, out="ok\n"):
    def fake_run(args, extra_env=None, on_line=None):
        if on_line:
            for line in out.splitlines(keepends=True):
                on_line(line)
            return 0, ""
        return 0, out

    monkeypatch.setattr("cv_agent.webapp._run_cli", fake_run)


def _start_pipeline(client, data):
    r = client.post("/pipeline", data=data)
    assert r.status_code == 302
    return int(r.headers["Location"].rstrip("/").rsplit("/", 1)[1])


def _wait(run_id, user="Илья"):
    from cv_agent.webapp import wait_pipeline

    wait_pipeline(run_id, user=user)


def test_pipeline_writes_structured_job_log(monkeypatch):
    _fake_criteria(monkeypatch)
    _fake_run(monkeypatch, out="ok\n")
    client = app.test_client()
    _login(client)
    run_id = _start_pipeline(client, {})
    _wait(run_id)

    log_path = _root() / "data" / "logs" / "jobs.jsonl"
    assert log_path.is_file()
    events = [
        json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ours = [e for e in events if e.get("run_id") == run_id]
    names = [e["event"] for e in ours]
    assert names[0] == "job_started"
    assert "step_finished" in names
    assert names[-1] == "job_finished"
    finished = ours[-1]
    assert finished["status"] == "succeeded"
    assert isinstance(finished["duration_seconds"], (int, float))
    started = ours[0]
    assert started["account_id"]
    assert started["kind"] == "pipeline"
    steps = [e for e in ours if e["event"] == "step_finished"]
    assert len(steps) >= 1
    assert all("code" in e and "cmd" in e for e in steps)


def test_backup_restore_roundtrip():
    """Процедура из docs/OPERATIONS.md: backup → потеря → restore → проверка."""
    data_dir = _root() / "data"
    db_path = data_dir / "applications.db"

    store = Store(db_path, user="Илья")
    try:
        store.add(
            Vacancy(
                "rest-1",
                "Python Entwickler",
                "ACME",
                "Wangen",
                "https://example.test/rest-1",
                "arbeitsagentur",
            )
        )
        backups = data_dir / "backups"
        backups.mkdir(parents=True, exist_ok=True, mode=0o700)
        manual = backups / "manual-test.db"
        conn = sqlite3.connect(manual)
        try:
            store.conn.backup(conn)
        finally:
            conn.close()
        assert (
            store.conn.execute("SELECT COUNT(*) FROM accounts").fetchone()[0] >= 1
        )
    finally:
        store.close()

    # «потеря» живой базы и восстановление из копии
    db_path.unlink()
    check_conn = sqlite3.connect(manual)
    try:
        assert check_conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        check_conn.close()
    shutil.copy2(manual, db_path)

    reopened = Store(db_path, user="Илья")
    try:
        assert reopened.schema_version() == LATEST_SCHEMA_VERSION
        assert reopened.account_exists("Илья")
        rows = reopened.list_all()
        assert any(row[0] == "rest-1" for row in rows)
    finally:
        reopened.close()
