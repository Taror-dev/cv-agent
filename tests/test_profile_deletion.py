"""Политика удаления профиля: archive → export → подтверждённое удаление."""

import json
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cv_agent import config
from cv_agent.models import Vacancy
from cv_agent.paths import ensure_account_path
from cv_agent.store import Store
from cv_agent.webapp import app


def _root() -> Path:
    return Path(config.DATA_DIR).parent


def _store(user: str | None = None) -> Store:
    return Store(_root() / "data" / "applications.db", user=user)


def _login(client, account):
    r = client.post("/login", data={"account": account})
    assert r.status_code == 302


def _add_vacancy(store: Store, refnr: str) -> None:
    store.add(
        Vacancy(
            refnr,
            "Python Entwickler",
            "ACME",
            "Wangen",
            f"https://example.test/{refnr}",
            "arbeitsagentur",
        )
    )


def _delete(client, confirm):
    return client.post("/profile/account/delete", data={"confirm_name": confirm})


def test_delete_requires_exact_name_confirmation():
    client = app.test_client()
    _login(client, "Илья")
    r = _delete(client, "Другой")
    assert r.status_code == 400
    assert "не совпало" in r.data.decode("utf-8")
    assert ">Илья<" in client.get("/login").data.decode("utf-8")


def test_default_profile_cannot_be_deleted():
    client = app.test_client()
    _login(client, "Илья")
    r = _delete(client, "Илья")
    assert r.status_code == 403
    assert "по умолчанию" in r.data.decode("utf-8")
    store = _store()
    try:
        assert store.account_exists("Илья")
    finally:
        store.close()


def test_delete_refuses_while_job_active():
    setup = _store()
    try:
        assert setup.create_account("Anna")
    finally:
        setup.close()
    client = app.test_client()
    _login(client, "Anna")
    store = _store(user="Anna")
    try:
        run_id, created = store.create_job(
            "pipeline", [{"cmd": "score", "args": ["score"]}]
        )
        assert created
    finally:
        store.close()
    r = _delete(client, "Anna")
    assert r.status_code == 409
    assert "активная фоновая задача" in r.data.decode("utf-8")
    store = _store()
    try:
        assert store.account_exists("Anna")
    finally:
        store.close()


def test_delete_archives_then_removes_profile():
    data_dir = _root() / "data"

    setup = _store()
    try:
        assert setup.create_account("Anna")
        anna_id = setup.account_id("Anna")
    finally:
        setup.close()

    client = app.test_client()
    _login(client, "Anna")

    anna_store = _store(user="Anna")
    try:
        _add_vacancy(anna_store, "ba-200")
    finally:
        anna_store.close()
    ensure_account_path(data_dir, anna_id, "cv")
    uuid_cv = data_dir / "accounts" / anna_id / "cv"
    (uuid_cv / "Lebenslauf.pdf").write_bytes(b"%PDF-1.4 test")
    legacy_cv_file = _root() / "cv" / "Anna" / "Alt-CV.pdf"
    legacy_cv_file.parent.mkdir(parents=True, exist_ok=True)
    legacy_cv_file.write_bytes(b"%PDF-1.4 alt")

    r = _delete(client, "Anna")
    assert r.status_code == 302
    assert "/login?deleted=" in r.headers["Location"]
    assert "удалён" in client.get(r.headers["Location"]).data.decode("utf-8")

    exports = sorted((data_dir / "backups").glob("profile-export-*"))
    assert len(exports) == 1
    export_dir = exports[0]
    report = json.loads(
        (export_dir / "profile-export.json").read_text(encoding="utf-8")
    )
    assert report["account_id"] == anna_id
    assert report["counts"]["vacancies"] == 1
    assert {d["filename"] for d in report["documents"]} >= {
        "Lebenslauf.pdf",
        "Alt-CV.pdf",
    }
    conn = sqlite3.connect(export_dir / "applications.db")
    try:
        assert conn.execute(
            "SELECT COUNT(*) FROM accounts WHERE name='Anna'"
        ).fetchone()[0] == 1  # архив сделан до удаления
    finally:
        conn.close()

    live = sqlite3.connect(data_dir / "applications.db")
    try:
        names = [row[0] for row in live.execute("SELECT name FROM accounts")]
        assert names == ["Илья"]
        assert live.execute(
            "SELECT COUNT(*) FROM vacancies WHERE refnr='ba-200'"
        ).fetchone()[0] == 0
        assert live.execute(
            "SELECT COUNT(*) FROM candidate_profiles WHERE account_id=?",
            (anna_id,),
        ).fetchone()[0] == 0
    finally:
        live.close()

    assert not (data_dir / "accounts" / anna_id).exists()
    assert not legacy_cv_file.exists()
    assert (_root() / "cv" / "Илья").is_dir()  # чужие данные не тронуты
    assert ">Anna<" not in client.get("/login").data.decode("utf-8")
