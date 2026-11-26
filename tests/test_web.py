import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cv_agent.webapp import _effective_llm_env, app

OLLAMA_OPTION_ID = "ollama:qwen3.8:latest"
GEMINI_OPTION_ID = "fallback:gemini:gemini-3.5-flash-lite"
GROQ_OPTION_ID = "groq:llama-3.3-70b-versatile"


def _login(client, account="Илья"):
    r = client.post("/login", data={"account": account})
    assert r.status_code == 302
    return r


def _tmp_store(monkeypatch, tmp_path):
    """webapp использует отдельную тестовую БД (вместо data/applications.db)."""
    from cv_agent.store import Store as RealStore

    def factory(user=None):
        return RealStore(tmp_path / "test.db", user=user)

    monkeypatch.setattr("cv_agent.webapp.Store", factory)
    return factory


def test_login_page():
    client = app.test_client()
    r = client.get("/login")
    assert r.status_code == 200
    html = r.data.decode("utf-8")
    assert "Выбрать профиль" in html
    assert "Создать профиль" in html
    assert "Илья" in html


def test_index_requires_login():
    r = app.test_client().get("/")
    assert r.status_code == 302
    assert "/login" in r.headers["Location"]


def test_profile_selection_ignores_proxy_origin_metadata(monkeypatch, tmp_path):
    _tmp_store(monkeypatch, tmp_path)
    r = app.test_client().post(
        "/login",
        data={"account": "Илья"},
        headers={"Origin": "https://evil.example", "Sec-Fetch-Site": "cross-site"},
    )
    assert r.status_code == 302


def test_login_unknown_account():
    client = app.test_client()
    r = client.post("/login", data={"account": "nobody"})
    assert r.status_code == 200
    assert "не найден" in r.data.decode("utf-8")


def test_create_account_flow(monkeypatch, tmp_path):
    _tmp_store(monkeypatch, tmp_path)
    client = app.test_client()
    r = client.post("/accounts/create", data={"name": "Alice"})
    assert r.status_code == 302
    html = client.get("/profile").data.decode("utf-8")
    assert "Alice" in html


def test_create_duplicate_account(monkeypatch, tmp_path):
    _tmp_store(monkeypatch, tmp_path)
    client = app.test_client()
    r = client.post("/accounts/create", data={"name": "Илья"})
    assert r.status_code == 409
    assert "уже существует" in r.data.decode("utf-8")


def test_create_invalid_account_name(monkeypatch, tmp_path):
    _tmp_store(monkeypatch, tmp_path)
    r = app.test_client().post("/accounts/create", data={"name": "x y z!"})
    assert r.status_code == 400


def test_create_dot_account_is_rejected(monkeypatch, tmp_path):
    _tmp_store(monkeypatch, tmp_path)
    r = app.test_client().post("/accounts/create", data={"name": ".."})
    assert r.status_code == 400


@pytest.mark.parametrize(
    "cookie_value",
    ["Илья", "unknown", ".", "..", "%2e%2e", "x%2Fy", "x\\y"],
)
def test_invalid_profile_cookie_is_cleared(monkeypatch, tmp_path, cookie_value):
    _tmp_store(monkeypatch, tmp_path)
    client = app.test_client()
    client.set_cookie("user", cookie_value)
    r = client.get("/")
    assert r.status_code == 302
    assert "/login" in r.headers["Location"]
    assert "user=;" in r.headers.get("Set-Cookie", "")


def test_login_cookie_has_local_security_flags(monkeypatch, tmp_path):
    _tmp_store(monkeypatch, tmp_path)
    r = app.test_client().post("/login", data={"account": "Илья"})
    cookie = next(
        value
        for value in r.headers.getlist("Set-Cookie")
        if value.startswith("cv_agent_session=")
    )
    assert "HttpOnly" in cookie
    assert "SameSite=Strict" in cookie
    assert "Илья" not in cookie


def test_tampered_signed_session_is_rejected(monkeypatch, tmp_path):
    _tmp_store(monkeypatch, tmp_path)
    client = app.test_client()
    client.set_cookie("cv_agent_session", "forged.account-id.signature")
    r = client.get("/")
    assert r.status_code == 302
    assert "/login" in r.headers["Location"]


def test_logout():
    client = app.test_client()
    _login(client)
    assert client.get("/").status_code == 200
    assert client.post("/logout").status_code == 302
    assert client.get("/").status_code == 302


def test_home_page_shows_vacancies():
    client = app.test_client()
    _login(client)
    r = client.get("/")
    assert r.status_code == 200
    html = r.data.decode("utf-8")
    assert "Искать новые вакансии" in html
    assert 'action="/pipeline"' in html
    assert "Главная" not in html
    assert r.headers["X-Frame-Options"] == "DENY"
    assert r.headers["X-Content-Type-Options"] == "nosniff"


def test_results_alias_still_works():
    client = app.test_client()
    _login(client)
    assert client.get("/results").status_code == 200


def test_run_status():
    client = app.test_client()
    _login(client)
    r = client.post("/run", data={"cmd": "status"})
    assert r.status_code == 200
    assert b"\xd0\x9a\xd0\xbe\xd0\xb4 \xd0\xb7\xd0\xb0\xd0\xb2\xd0\xb5\xd1\x80\xd1\x88\xd0\xb5\xd0\xbd\xd0\xb8\xd1\x8f" in r.data


def test_run_search_bad_command():
    client = app.test_client()
    _login(client)
    r = client.post("/run", data={"cmd": "nonexistent"})
    assert r.status_code == 400
    assert "не разрешена" in r.data.decode("utf-8")


def test_run_rejects_invalid_limit_before_cli(monkeypatch):
    calls = _fake_run(monkeypatch)
    client = app.test_client()
    _login(client)
    r = client.post("/run", data={"cmd": "search", "limit": "501"})
    assert r.status_code == 400
    assert "от 1 до 500" in r.data.decode("utf-8")
    assert calls == []


def test_run_timeout_is_reported_without_500(monkeypatch):
    from cv_agent.webapp import CommandTimedOut

    def timed_out(*args, **kwargs):
        raise CommandTimedOut("deadline")

    monkeypatch.setattr("cv_agent.webapp._run_cli", timed_out)
    client = app.test_client()
    _login(client)
    r = client.post("/run", data={"cmd": "status"})
    assert r.status_code == 200
    html = r.data.decode("utf-8")
    assert "124" in html
    assert "deadline" in html


def test_results_page():
    client = app.test_client()
    _login(client)
    r = client.get("/results")
    assert r.status_code == 200
    assert b"\xd0\x92\xd0\xb0\xd0\xba\xd0\xb0\xd0\xbd\xd1\x81\xd0\xb8\xd0\xb8" in r.data


def test_vacancy_not_found():
    client = app.test_client()
    _login(client)
    r = client.get("/vacancy/does-not-exist")
    assert r.status_code == 200
    assert b"\xd0\xbd\xd0\xb5 \xd0\xbd\xd0\xb0\xd0\xb9\xd0\xb4\xd0\xb5\xd0\xbd\xd0\xb0" in r.data


def test_external_vacancy_url_requires_http_scheme():
    from cv_agent.webapp import _safe_external_url

    assert _safe_external_url("https://example.test/job") is True
    assert _safe_external_url("javascript:alert(1)") is False
    assert _safe_external_url("//example.test/job") is False


def test_effective_env_fallback_when_ollama_down(monkeypatch):
    monkeypatch.setattr("cv_agent.webapp._ollama_alive", lambda: False)
    with app.test_request_context(
        "/", headers={"Cookie": "llm=" + OLLAMA_OPTION_ID}
    ):
        env, note = _effective_llm_env()
    assert env["LLM_MODEL"] == "gemini-3.5-flash-lite"
    assert env["LLM_PROVIDER"] == "gemini"
    assert "GEMINI_API_KEY" in env["LLM_API_KEY_ENV"]
    assert "Ollama" in note
    assert "gemini" in note


def test_effective_env_keeps_ollama_when_up(monkeypatch):
    monkeypatch.setattr("cv_agent.webapp._ollama_alive", lambda: True)
    with app.test_request_context(
        "/", headers={"Cookie": "llm=" + OLLAMA_OPTION_ID}
    ):
        env, note = _effective_llm_env()
    assert env["LLM_MODEL"] == "qwen3.8:latest"
    assert note == ""


def test_effective_env_gemini_selected_no_fallback(monkeypatch):
    monkeypatch.setattr("cv_agent.webapp._ollama_alive", lambda: False)
    with app.test_request_context(
        "/", headers={"Cookie": "llm=" + GEMINI_OPTION_ID}
    ):
        env, note = _effective_llm_env()
    assert env["LLM_MODEL"] == "gemini-3.5-flash-lite"
    assert note == ""


def test_effective_env_rejects_unlisted_llm_endpoint(monkeypatch):
    monkeypatch.setattr("cv_agent.webapp._ollama_alive", lambda: True)
    forged = "%7B%22model%22%3A%22evil%22%2C%22base_url%22%3A%22http%3A%2F%2Fevil%22%7D"
    with app.test_request_context("/", headers={"Cookie": "llm=" + forged}):
        env, note = _effective_llm_env()
    assert env == {}
    assert note == ""


def test_ollama_alive_uses_host_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "192.168.1.50:11434")
    import requests

    seen = {}

    def fake_get(url, timeout):
        seen["url"] = url
        return type("R", (), {"status_code": 200})()

    monkeypatch.setattr(requests, "get", fake_get)
    from cv_agent.webapp import _ollama_alive

    assert _ollama_alive() is True
    assert seen["url"] == "http://192.168.1.50:11434/api/tags"


def test_available_llms_uses_host_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "192.168.1.50:11434")
    from cv_agent.webapp import available_llms

    for o in available_llms():
        if o["api_key_env"] == "OLLAMA_API_KEY":
            assert o["base_url"] == "http://192.168.1.50:11434/v1"
            break
    else:
        raise AssertionError("нет ollama-моделей в списке")


def _fake_run(monkeypatch, out="ok\n"):
    calls = []

    def fake_run(args, extra_env=None, on_line=None):
        calls.append(list(args))
        if on_line:
            for line in out.splitlines(keepends=True):
                on_line(line)
        return 0, out

    monkeypatch.setattr("cv_agent.webapp._run_cli", fake_run)
    return calls


def _fake_criteria(monkeypatch, keywords=("Python",)):
    from cv_agent.config import Criteria

    c = Criteria(
        region="88239 Wangen im Allgäu",
        radius_km=50,
        keywords=list(keywords),
        exclude_keywords=[],
        min_score=60,
    )
    monkeypatch.setattr("cv_agent.webapp.load_criteria", lambda user=None: c)


def _start_pipeline(client, data):
    """POST /pipeline (фоновый запуск) → возвращает run_id."""
    r = client.post("/pipeline", data=data)
    assert r.status_code == 302
    return int(r.headers["Location"].rstrip("/").rsplit("/", 1)[1])


def _wait(run_id, user="Илья"):
    from cv_agent.webapp import wait_pipeline

    wait_pipeline(run_id, user=user)


def test_pipeline_no_regenerate(monkeypatch):
    _fake_criteria(monkeypatch)
    calls = _fake_run(monkeypatch)
    client = app.test_client()
    _login(client)
    run_id = _start_pipeline(client, {"limit": "10"})
    _wait(run_id)
    assert calls == [["search", "--limit", "10", "--location", ""], ["score"]]


def test_search_form_fields_persist_per_account(monkeypatch, tmp_path):
    """Введённые поля поиска сохраняются в SQLite и подставляются в форму."""
    from cv_agent.store import Store

    _tmp_store(monkeypatch, tmp_path)
    _fake_criteria(monkeypatch)
    _fake_run(monkeypatch)
    client = app.test_client()
    _login(client)
    run_id = _start_pipeline(
        client,
        {
            "limit": "42",
            "country": "Deutschland",
            "region": "",
            "city": "München",
        },
    )
    _wait(run_id)

    html = client.get("/results").data.decode("utf-8")
    assert 'id="limit" value="42"' in html
    assert 'name="country" id="country" maxlength="300" value="Deutschland"' in html
    assert 'name="city" id="city" maxlength="300" value="München"' in html
    assert 'name="region" id="region" maxlength="300" value=""' in html

    # Значения живут в БД — видны после «перезапуска» (новый Store).
    store = Store(tmp_path / "test.db")
    try:
        assert store.account_search_params("Илья")["limit"] == "42"
    finally:
        store.close()


def test_search_form_fields_are_isolated_between_accounts(
    monkeypatch, tmp_path
):
    from cv_agent.store import Store

    _tmp_store(monkeypatch, tmp_path)
    _fake_criteria(monkeypatch)
    _fake_run(monkeypatch)
    client = app.test_client()
    client.post("/accounts/create", data={"name": "Alice"})
    run_id = _start_pipeline(
        client, {"limit": "7", "country": "", "region": "Hessen", "city": ""}
    )
    _wait(run_id, user="Alice")
    _login(client)  # обратно в Илью
    run_id = _start_pipeline(client, {"limit": "42", "city": "München"})
    _wait(run_id)

    ilia_html = client.get("/results").data.decode("utf-8")
    assert 'id="limit" value="42"' in ilia_html
    assert 'value="München"' in ilia_html
    assert 'value="Hessen"' not in ilia_html

    _login(client, "Alice")
    alice_html = client.get("/results").data.decode("utf-8")
    assert 'id="limit" value="7"' in alice_html
    assert 'value="Hessen"' in alice_html
    assert 'value="München"' not in alice_html

    store = Store(tmp_path / "test.db")
    try:
        assert store.account_search_params("Alice")["region"] == "Hessen"
        assert store.account_search_params("Илья")["city"] == "München"
    finally:
        store.close()


def test_search_form_empty_submission_clears_saved_values(
    monkeypatch, tmp_path
):
    _tmp_store(monkeypatch, tmp_path)
    _fake_criteria(monkeypatch)
    _fake_run(monkeypatch)
    client = app.test_client()
    _login(client)
    run_id = _start_pipeline(client, {"limit": "42", "city": "München"})
    _wait(run_id)
    run_id = _start_pipeline(client, {})
    _wait(run_id)
    html = client.get("/results").data.decode("utf-8")
    # Пустая отправка сбрасывает сохранённые поля; лимит — критерий по умолчанию.
    assert 'id="limit" value="100"' in html
    assert 'value="München"' not in html


def test_pipeline_passes_most_specific_location(monkeypatch):
    _fake_criteria(monkeypatch)
    calls = _fake_run(monkeypatch)
    client = app.test_client()
    _login(client)
    run_id = _start_pipeline(
        client,
        {"country": "Deutschland", "region": "Bayern", "city": "München"},
    )
    _wait(run_id)
    assert calls[0][:3] == ["search", "--location", "München"]
    assert "score" in calls[1:] or calls[1] == ["score"]


def test_pipeline_passes_comma_separated_locations(monkeypatch):
    _fake_criteria(monkeypatch)
    calls = _fake_run(monkeypatch)
    client = app.test_client()
    _login(client)
    run_id = _start_pipeline(
        client,
        {
            "country": "",
            "region": "Bayern, Baden-Württemberg",
            "city": "",
        },
    )
    _wait(run_id)
    assert calls[0][:3] == ["search", "--location", "Bayern, Baden-Württemberg"]


def test_pipeline_location_parts_are_cleaned_individually(monkeypatch):
    _fake_criteria(monkeypatch)
    calls = _fake_run(monkeypatch)
    client = app.test_client()
    _login(client)
    run_id = _start_pipeline(
        client,
        {"country": "", "region": "", "city": " München ,,  Hamburg \x07 ,"},
    )
    _wait(run_id)
    assert calls[0][:3] == ["search", "--location", "München, Hamburg"]


def test_pipeline_location_fields_are_sanitized(monkeypatch):
    _fake_criteria(monkeypatch)
    calls = _fake_run(monkeypatch)
    client = app.test_client()
    _login(client)
    run_id = _start_pipeline(
        client, {"country": "  Deutschland \x07 ", "region": "", "city": ""}
    )
    _wait(run_id)
    assert calls[0][-2:] == ["--location", "Deutschland"]


def test_home_page_shows_search_location_fields():
    client = app.test_client()
    _login(client)
    html = client.get("/").data.decode("utf-8")
    for marker in (
        'name="limit"',
        'name="country"',
        'name="region"',
        'name="city"',
        "Страна",
        "Регион",
        "Город",
    ):
        assert marker in html
    assert '<label for="limit">Общий лимит вакансий за поиск</label>' in html
    assert re.search(
        r'<label for="limit">[^<]*</label>\s*<input type="number"', html
    )


def test_pipeline_regenerate(monkeypatch):
    _fake_criteria(monkeypatch)
    calls = _fake_run(monkeypatch)
    client = app.test_client()
    _login(client)
    run_id = _start_pipeline(client, {"regenerate": "1"})
    _wait(run_id)
    assert calls == [
        ["keywords", "--apply"],
        ["search", "--location", ""],
        ["score"],
    ]
    html = client.get(f"/pipeline/{run_id}").data.decode("utf-8")
    assert "keywords --apply" in html


def test_pipeline_forces_keywords_when_empty(monkeypatch):
    _fake_criteria(monkeypatch, keywords=())
    calls = _fake_run(monkeypatch)
    client = app.test_client()
    _login(client)
    run_id = _start_pipeline(client, {})
    _wait(run_id)
    assert calls == [
        ["keywords", "--apply"],
        ["search", "--location", ""],
        ["score"],
    ]


def test_pipeline_empty_cv_warning(monkeypatch):
    _fake_criteria(monkeypatch, keywords=())
    calls = _fake_run(monkeypatch)
    monkeypatch.setattr("cv_agent.webapp.load_document_texts", lambda user=None: [])
    client = app.test_client()
    _login(client)
    r = client.post("/pipeline", data={})
    html = r.data.decode("utf-8")
    assert "Папка cv пуста" in html
    assert calls == []  # ничего не запускалось


def test_pipeline_status_json(monkeypatch):
    _fake_criteria(monkeypatch)
    _fake_run(monkeypatch, out="Zeile 1\nZeile 2\n")
    client = app.test_client()
    _login(client)
    run_id = _start_pipeline(client, {"limit": "5"})
    _wait(run_id)
    r = client.get(f"/pipeline/{run_id}/status")
    data = r.get_json()
    assert data["done"] is True
    assert [s["cmd"] for s in data["steps"]] == [
        "search --limit 5 --location *",
        "score",
    ]
    assert data["steps"][0]["output"] == "Zeile 1\nZeile 2\n"
    assert data["metrics"] == {
        "network_requests": 0,
        "network_wait_seconds": 0.0,
        "llm_requests": 0,
        "llm_wait_seconds": 0.0,
        "embedding_requests": 0,
        "embedding_wait_seconds": 0.0,
    }


def test_pipeline_status_serves_log_incrementally_by_cursor(monkeypatch):
    _fake_criteria(monkeypatch)
    _fake_run(monkeypatch, out="Zeile 1\nZeile 2\n")
    client = app.test_client()
    _login(client)
    run_id = _start_pipeline(client, {"limit": "5"})
    _wait(run_id)

    full = client.get(f"/pipeline/{run_id}/status").get_json()
    assert full["steps"][0]["output"] == "Zeile 1\nZeile 2\n"
    assert "output_cursor" not in full["steps"][0]

    partial = client.get(
        f"/pipeline/{run_id}/status", query_string={"cursor": "7,"}
    ).get_json()
    assert partial["steps"][0]["output"] == "\nZeile 2\n"
    assert partial["steps"][0]["output_cursor"] == len("Zeile 1\nZeile 2\n")
    assert partial["steps"][0]["output_reset"] is False
    # отсутствующий курсор = «по этому шагу ничего не получено» → полный вывод
    assert partial["steps"][1]["output"] == full["steps"][1]["output"]

    exhausted = client.get(
        f"/pipeline/{run_id}/status",
        query_string={"cursor": f"{len('Zeile 1\nZeile 2\n')},"},
    ).get_json()
    assert exhausted["steps"][0]["output"] == ""
    assert exhausted["steps"][0]["output_reset"] is False

    stale = client.get(
        f"/pipeline/{run_id}/status", query_string={"cursor": "999999,"}
    ).get_json()
    assert stale["steps"][0]["output_reset"] is True
    assert stale["steps"][0]["output"] == "Zeile 1\nZeile 2\n"

    garbage = client.get(
        f"/pipeline/{run_id}/status", query_string={"cursor": "abc,"}
    ).get_json()
    assert garbage["steps"][0]["output_reset"] is True
    assert garbage["steps"][0]["output"] == "Zeile 1\nZeile 2\n"


def test_pipeline_passes_run_id_to_cli_for_rate_limit_status(monkeypatch):
    _fake_criteria(monkeypatch)
    seen = []

    def fake_run(args, extra_env=None, on_line=None):
        seen.append(extra_env.get("CV_AGENT_RUN_ID"))
        return 0, ""

    monkeypatch.setattr("cv_agent.webapp._run_cli", fake_run)
    client = app.test_client()
    _login(client)
    run_id = _start_pipeline(client, {})
    _wait(run_id)
    assert seen == [str(run_id), str(run_id)]


def test_pipeline_timeout_is_persisted(monkeypatch):
    from cv_agent.webapp import CommandTimedOut

    _fake_criteria(monkeypatch)

    def timed_out(*args, **kwargs):
        raise CommandTimedOut("deadline")

    monkeypatch.setattr("cv_agent.webapp._run_cli", timed_out)
    client = app.test_client()
    _login(client)
    run_id = _start_pipeline(client, {})
    _wait(run_id)
    data = client.get(f"/pipeline/{run_id}/status").get_json()
    assert data["done"] is True
    assert data["status"] == "timed_out"
    assert "deadline" in data["error"]


def _register_worker(run_id, proc):
    from cv_agent import webapp

    with webapp._workers_lock:
        webapp._active_workers[run_id] = proc


def test_cancel_process_tree_terminates_and_reaps():
    import subprocess as sp
    import time

    from cv_agent.webapp import _cancel_process_tree

    # bash + дочерний sleep в одной группе процессов
    proc = sp.Popen(
        ["bash", "-c", "sleep 60 & wait"],
        stdin=sp.DEVNULL,
        stdout=sp.DEVNULL,
        stderr=sp.DEVNULL,
        start_new_session=True,
    )
    started = time.monotonic()
    _cancel_process_tree(proc, grace=2.0)
    elapsed = time.monotonic() - started

    assert proc.poll() is not None  # reap подтверждён
    assert elapsed < 5  # SIGTERM сработал быстрее полного grace+kill пути


def test_pipeline_cancel_marks_job_and_kills_process(monkeypatch):
    import subprocess as sp

    client = app.test_client()
    _login(client)

    from cv_agent.store import Store as RealStore
    from cv_agent import config as cfg

    store = RealStore(cfg.DATA_DIR / "applications.db", user="Илья")
    try:
        run_id, created = store.create_job(
            "pipeline", [{"cmd": "score", "args": ["score"]}]
        )
        assert created
        assert store.claim_job(run_id)
    finally:
        store.close()

    proc = sp.Popen(
        ["sleep", "60"], start_new_session=True,
        stdout=sp.DEVNULL, stderr=sp.DEVNULL,
    )
    _register_worker(run_id, proc)

    r = client.post(f"/pipeline/{run_id}/cancel")
    assert r.status_code == 302
    assert f"/pipeline/{run_id}" in r.headers["Location"]
    assert proc.poll() is not None  # дочерний процесс не остался

    data = client.get(f"/pipeline/{run_id}/status").get_json()
    assert data["status"] == "cancelled"
    assert "Отменено" in (data["error"] or "")

    repeat = client.post(f"/pipeline/{run_id}/cancel")
    assert repeat.status_code == 404  # прогон уже завершён


def test_pipeline_rerun_redirects_to_active(monkeypatch):
    import threading

    _fake_criteria(monkeypatch)
    gate = threading.Event()

    def slow_run(args, extra_env=None, on_line=None):
        if args[0] == "search":
            gate.wait(timeout=5)
        return 0, "ok"

    monkeypatch.setattr("cv_agent.webapp._run_cli", slow_run)
    client = app.test_client()
    _login(client)
    run_id = _start_pipeline(client, {})
    r2 = client.post("/pipeline", data={})
    assert r2.status_code == 302
    assert f"/pipeline/{run_id}" in r2.headers["Location"]
    gate.set()
    _wait(run_id)


def test_pipeline_runs_are_private_to_profile_and_can_run_in_parallel(monkeypatch):
    import threading

    from cv_agent.store import Store

    _fake_criteria(monkeypatch)
    store = Store()
    store.create_account("Alice")
    store.close()
    gate = threading.Event()
    started = {"Илья": threading.Event(), "Alice": threading.Event()}

    def slow_run(args, extra_env=None, on_line=None):
        user = extra_env["CV_AGENT_USER"]
        started[user].set()
        gate.wait(timeout=5)
        if on_line:
            on_line(f"{user}\n")
        return 0, ""

    monkeypatch.setattr("cv_agent.webapp._run_cli", slow_run)
    ilia_client = app.test_client()
    alice_client = app.test_client()
    _login(ilia_client)
    _login(alice_client, "Alice")

    ilia_run = _start_pipeline(ilia_client, {})
    alice_run = _start_pipeline(alice_client, {})
    assert started["Илья"].wait(timeout=2)
    assert started["Alice"].wait(timeout=2)
    assert alice_run != ilia_run
    assert alice_client.get(f"/pipeline/{ilia_run}/status").status_code == 404
    assert ilia_client.get(f"/pipeline/{alice_run}/status").status_code == 404

    gate.set()
    _wait(ilia_run)
    _wait(alice_run, user="Alice")
    assert ilia_client.get(f"/pipeline/{ilia_run}/status").get_json()["done"]
    assert alice_client.get(f"/pipeline/{alice_run}/status").get_json()["done"]


def test_results_score_missing_button_removed(monkeypatch, tmp_path):
    """Кнопка «Оценить вакансии без оценки» и её маршрут удалены из UI."""
    _tmp_store(monkeypatch, tmp_path)
    _fake_run(monkeypatch)
    client = app.test_client()
    _login(client)
    html = client.get("/results").data.decode("utf-8")
    assert "Оценить вакансии без оценки" not in html
    assert "score-missing" not in html
    r = client.post("/results/score-missing")
    assert r.status_code == 404


def test_results_explains_score_format_error(monkeypatch, tmp_path):
    from cv_agent.models import Vacancy

    factory = _tmp_store(monkeypatch, tmp_path)
    store = factory(user="Илья")
    store.add(Vacancy("bad-score", "Python", "ACME", "Wangen", "u", "ba"))
    store.mark_score_error("bad-score")
    store.close()
    client = app.test_client()
    _login(client)
    html = client.get("/results").data.decode("utf-8")
    assert "не удалось рассчитать оценку" in html
    assert "доступны для повторной оценки" in html


def test_results_score_missing_redirects_to_active(monkeypatch):
    import threading

    _fake_criteria(monkeypatch)
    gate = threading.Event()

    def slow_run(args, extra_env=None, on_line=None):
        if args[0] == "search":
            gate.wait(timeout=5)
        return 0, "ok"

    monkeypatch.setattr("cv_agent.webapp._run_cli", slow_run)
    client = app.test_client()
    _login(client)
    run_id = _start_pipeline(client, {})
    # Маршрут score-missing удалён: POST отдаёт 404 и не влияет на активный прогон.
    r = client.post("/results/score-missing")
    assert r.status_code == 404
    assert f"/pipeline/{run_id}" not in r.headers.get("Location", "")
    gate.set()
    _wait(run_id)


def test_vacancy_page_has_apply_button(monkeypatch):
    from cv_agent.models import Vacancy
    from cv_agent.store import Store

    store = Store()
    store.add(Vacancy("in-test-apply-btn", "Python Dev", "ACME", "Wangen", "u", "ba"))
    try:
        client = app.test_client()
        _login(client)
        r = client.get("/vacancy/in-test-apply-btn")
        assert b"\xd0\xa1\xd0\xbe\xd0\xb7\xd0\xb4\xd0\xb0\xd1\x82\xd1\x8c Anschreiben" in r.data
    finally:
        store.delete("in-test-apply-btn")
        store.close()


def test_vacancy_page_persists_successful_description_fetch(monkeypatch, tmp_path):
    from cv_agent.models import Vacancy
    from cv_agent.store import Store

    _tmp_store(monkeypatch, tmp_path)
    store = Store(tmp_path / "test.db")
    store.add(
        Vacancy(
            "detail-cache",
            "Python Dev",
            "ACME",
            "Wangen",
            "https://example.test/job",
            "linkedin",
        )
    )
    store.close()
    calls = []
    monkeypatch.setattr(
        "cv_agent.webapp.fetch_description",
        lambda vacancy: calls.append(vacancy.refnr) or "Persistierte Beschreibung",
    )
    client = app.test_client()
    _login(client)

    first = client.get("/vacancy/detail-cache")
    second = client.get("/vacancy/detail-cache")

    assert first.status_code == 200
    assert second.status_code == 200
    assert calls == ["detail-cache"]
    check = Store(tmp_path / "test.db")
    try:
        assert check.by_refnr("detail-cache").description == "Persistierte Beschreibung"
        assert check.vacancy_lifecycle("detail-cache")["description_hash"]
    finally:
        check.close()


def test_apply_vacancy_route(monkeypatch, tmp_path):
    """Успешный apply: PRG-редирект на страницу вакансии с зелёным сообщением."""
    from cv_agent.models import Vacancy
    from cv_agent.store import Store

    _tmp_store(monkeypatch, tmp_path)
    store = Store(tmp_path / "test.db", user="Илья")
    store.create_account("Илья")
    store.add(Vacancy("xyz", "Job", "ACME", "Wangen", "u", "ba"))
    store.close()

    calls = _fake_run(monkeypatch)
    client = app.test_client()
    _login(client)
    r = client.post("/vacancy/xyz/apply")
    assert r.status_code == 302
    assert r.headers["Location"].endswith("/vacancy/xyz?apply_ok=1")
    assert calls == [["apply", "xyz"]]

    page = client.get(r.headers["Location"]).data.decode("utf-8")
    assert "Anschreiben и Lebenslauf успешно созданы." in page


def test_apply_vacancy_route_reports_error_inline(monkeypatch, tmp_path):
    from cv_agent.models import Vacancy
    from cv_agent.store import Store

    _tmp_store(monkeypatch, tmp_path)
    store = Store(tmp_path / "test.db", user="Илья")
    store.create_account("Илья")
    store.add(Vacancy("xyz", "Job", "ACME", "Wangen", "u", "ba"))
    store.close()

    monkeypatch.setattr(
        "cv_agent.webapp._run_cli",
        lambda *a, **k: (1, "LLM error: модель недоступна\n"),
    )
    client = app.test_client()
    _login(client)
    r = client.post("/vacancy/xyz/apply")
    assert r.status_code == 302
    assert "apply_error" in r.headers["Location"]

    page = client.get(r.headers["Location"]).data.decode("utf-8")
    assert "LLM error: модель недоступна" in page
    assert "успешно созданы" not in page


def test_vacancy_edit_saves_description_and_translation(monkeypatch, tmp_path):
    from cv_agent.models import Vacancy
    from cv_agent.store import Store

    _tmp_store(monkeypatch, tmp_path)
    store = Store(tmp_path / "test.db")
    store.add(
        Vacancy(
            "edit-1", "Python Dev", "ACME", "Wangen", "u", "ba",
            description="Оригинальное описание",
        )
    )
    store.close()
    client = app.test_client()
    _login(client)

    r = client.post(
        "/vacancy/edit-1/edit",
        data={"field": "description", "text": "Исправленное описание"},
    )
    assert r.status_code == 200
    assert "Описание сохранено." in r.data.decode("utf-8")
    r = client.post(
        "/vacancy/edit-1/edit",
        data={"field": "translation", "text": "Русский перевод"},
    )
    assert r.status_code == 200
    assert "Перевод сохранён." in r.data.decode("utf-8")

    check = Store(tmp_path / "test.db")
    try:
        v = check.by_refnr("edit-1")
        assert v.description == "Исправленное описание"
        assert v.translation == "Русский перевод"
        # Отредактированное описание не перезагружается по TTL.
        fetches = []
        monkeypatch.setattr(
            "cv_agent.webapp.fetch_description",
            lambda vacancy: fetches.append(vacancy.refnr) or "из источника",
        )
        page = client.get("/vacancy/edit-1")
        assert page.status_code == 200
        assert "Исправленное описание" in page.data.decode("utf-8")
        assert fetches == []
    finally:
        check.close()


def test_vacancy_edit_rejects_bad_field_and_oversized_text(monkeypatch, tmp_path):
    from cv_agent.models import Vacancy
    from cv_agent.store import Store

    _tmp_store(monkeypatch, tmp_path)
    store = Store(tmp_path / "test.db")
    store.add(Vacancy("edit-2", "Python Dev", "ACME", "Wangen", "u", "ba"))
    store.close()
    client = app.test_client()
    _login(client)

    r = client.post(
        "/vacancy/edit-2/edit", data={"field": "title", "text": "x"}
    )
    assert r.status_code == 400
    assert "Некорректное поле" in r.data.decode("utf-8")
    r = client.post(
        "/vacancy/edit-2/edit", data={"field": "description", "text": "x" * 200_001}
    )
    assert r.status_code == 400
    assert "слишком длинный" in r.data.decode("utf-8")
    r = client.post("/vacancy/missing/edit", data={"field": "description", "text": "x"})
    assert r.status_code == 404

    check = Store(tmp_path / "test.db")
    try:
        assert check.by_refnr("edit-2").description == ""
    finally:
        check.close()


def test_accounts_isolate_vacancies(monkeypatch, tmp_path):
    from cv_agent.models import Vacancy

    _tmp_store(monkeypatch, tmp_path)
    from cv_agent.store import Store

    alice = Store(tmp_path / "test.db", user="Alice")
    bob = Store(tmp_path / "test.db", user="Bob")
    alice.create_account("Alice")
    bob.create_account("Bob")
    alice.add(Vacancy("v1", "Alice Job", "ACME", "Wangen", "u", "ba", score=85.0))
    bob.add(Vacancy("v1", "Bob Job", "ACME", "Wangen", "u", "ba", score=85.0))
    alice.close()
    bob.close()

    client = app.test_client()
    _login(client, "Alice")
    assert "Alice Job" in client.get("/results").data.decode("utf-8")
    assert "Bob Job" not in client.get("/results").data.decode("utf-8")

    client.get("/logout")
    _login(client, "Bob")
    html = client.get("/results").data.decode("utf-8")
    assert "Bob Job" in html
    assert "Alice Job" not in html


def test_results_shows_all_vacancies_without_score_threshold(monkeypatch, tmp_path):
    """Порог min_score больше не фильтрует таблицу — видны все вакансии."""
    from cv_agent.models import Vacancy

    _tmp_store(monkeypatch, tmp_path)
    from cv_agent.store import Store

    store = Store(tmp_path / "test.db", user="Илья")
    store.create_account("Илья")
    store.add(
        Vacancy(
            "good",
            "High Score Job",
            "ACME",
            "Wangen im Allgäu, Bayern, Deutschland",
            "u",
            "ba",
            score=85.0,
        )
    )
    store.add(Vacancy("minimum", "Minimum Score Job", "ACME", "Wangen", "u", "ba", score=70.0))
    store.add(Vacancy("bad", "Low Score Job", "ACME", "Wangen", "u", "ba", score=50.0))
    store.add(Vacancy("none", "No Score Job", "ACME", "Wangen", "u", "ba"))
    store.close()

    client = app.test_client()
    _login(client)
    html = client.get("/results").data.decode("utf-8")
    assert "High Score Job" in html
    assert 'data-filter="location"' in html and ">Место<span" in html
    assert "Wangen im Allgäu, Bayern, Deutschland" in html
    assert "Minimum Score Job" in html
    assert "Low Score Job" in html
    assert "No Score Job" in html
    assert "только с оценкой" not in html


def test_results_pagination_shows_50_per_page(monkeypatch, tmp_path):
    from cv_agent.models import Vacancy
    from cv_agent.webapp import RESULTS_PAGE_SIZE

    _tmp_store(monkeypatch, tmp_path)
    from cv_agent.store import Store

    store = Store(tmp_path / "test.db", user="Илья")
    store.create_account("Илья")
    total = RESULTS_PAGE_SIZE + 5
    for i in range(total):
        # Уникальные убывающие оценки делают сортировку детерминированной:
        # страница 1 — Job 0..49, страница 2 — Job 50..54
        store.add(
            Vacancy(
                f"ref-{i}", f"Job {i}", "ACME", "Wangen", "u", "ba", score=float(total - i)
            )
        )
    store.close()

    client = app.test_client()
    _login(client)

    html = client.get("/results").data.decode("utf-8")
    body = html.split("<tbody>")[1].split("</tbody>")[0]
    assert 'class="pagination"' in html
    assert 'data-filter="refnr"' in html and ">Идентификатор<span" in html
    assert ">Номер<" not in html
    assert '<th class="num">№</th>' in html
    # Сквозная нумерация строк: первая страница начинается с 1
    assert ">1</td>" in body
    assert "Job 0</td>" in body
    assert f"Job {RESULTS_PAGE_SIZE - 1}</td>" in body
    assert f"Job {RESULTS_PAGE_SIZE}</td>" not in body

    second = client.get("/results?page=2").data.decode("utf-8")
    second_body = second.split("<tbody>")[1].split("</tbody>")[0]
    assert "Job 0</td>" not in second_body
    assert f"Job {total - 1}</td>" in second_body
    # На второй странице нумерация продолжается с 51
    assert f">{RESULTS_PAGE_SIZE + 1}</td>" in second_body

    # Некорректные и выходящие за границы номера страницы безопасно нормализуются
    for raw in ("0", "-3", "abc", str(total)):
        r = client.get(f"/results?page={raw}")
        assert r.status_code == 200
        assert 'class="pagination"' in r.data.decode("utf-8")

    empty_store = Store(tmp_path / "test.db", user="Alice")
    try:
        empty_store.create_account("Alice")
    finally:
        empty_store.close()
    client.get("/logout")
    _login(client, "Alice")
    empty_html = client.get("/results").data.decode("utf-8")
    assert 'class="pagination"' not in empty_html


def test_results_presence_columns_replace_status(monkeypatch, tmp_path):
    """Столбец «Статус» заменён на признаки наличия описания, перевода и Anschreiben."""
    from cv_agent.models import Vacancy

    _tmp_store(monkeypatch, tmp_path)
    from cv_agent.store import Store

    store = Store(tmp_path / "test.db", user="Илья")
    store.create_account("Илья")
    store.add(
        Vacancy(
            "full",
            "Full Job",
            "ACME",
            "Wangen",
            "u",
            "ba",
            description="Beschreibung",
            score=90.0,
        )
    )
    store.add(Vacancy("bare", "Bare Job", "ACME", "Wangen", "u", "ba", score=80.0))
    store.update_translation("full", "Перевод")
    store.set_status("full", "prepared")
    store.close()

    client = app.test_client()
    _login(client)
    html = client.get("/results").data.decode("utf-8")

    assert 'data-filter="desc"' in html and ">Описание<span" in html
    assert 'data-filter="transl"' in html and ">Перевод<span" in html
    assert 'data-filter="letter"' in html and ">Anschreiben<span" in html
    assert ">Статус<" not in html
    body = html.split("<tbody>")[1].split("</tbody>")[0]
    assert 'class="score-method"' in body
    presence_cells = re.findall(r'<td class="presence">(.*?)</td>', body, re.S)
    assert all('class="badge' not in cell for cell in presence_cells)

    full_row = html.split('href="/vacancy/full"')[1].split("</tr>")[0]
    bare_row = html.split('href="/vacancy/bare"')[1].split("</tr>")[0]
    assert full_row.count('<span class="check"') == 3
    assert bare_row.count('<span class="nocheck"') == 3


def test_results_column_filters(monkeypatch, tmp_path):
    """Excel-подобные фильтры: выбор значений столбца через параметр f_<колонка>."""
    from cv_agent.models import Vacancy
    from cv_agent.webapp import RESULTS_PAGE_SIZE

    _tmp_store(monkeypatch, tmp_path)
    from cv_agent.store import Store

    store = Store(tmp_path / "test.db", user="Илья")
    store.create_account("Илья")
    for i in range(55):
        company = "Alpha" if i % 2 == 0 else "Beta"
        store.add(
            Vacancy(
                f"ref-{i}",
                f"Job {i}",
                company,
                "Wangen",
                "u",
                "ba",
                # Уникальные оценки делают сортировку и состав страниц детерминированными
                score=float(100 - i),
            )
        )
    for i in range(55, 57):
        store.add(Vacancy(f"ref-{i}", f"Job {i}", "Alpha", "Wangen", "u", "stepstone"))
    store.close()

    client = app.test_client()
    _login(client)

    html = client.get("/results").data.decode("utf-8")
    body = html.split("<tbody>")[1].split("</tbody>")[0]
    assert 'id="filter-config"' in html
    assert 'data-filter="source"' in html
    assert 'data-filter="company"' in html
    assert "stepstone" in html
    assert 'class="pagination"' in html

    # Один фильтр: только stepstone-вакансии, пагинация исчезает
    html = client.get("/results?f_source=stepstone").data.decode("utf-8")
    body = html.split("<tbody>")[1].split("</tbody>")[0]
    assert "Job 55</td>" in body and "Job 56</td>" in body
    assert "Job 0</td>" not in body
    assert 'class="pagination"' not in html
    assert "Фильтры активны" in html
    assert 'filterable filtered' in html

    # Несколько значений через разделитель
    html = client.get("/results?f_source=ba||stepstone&page=2").data.decode("utf-8")
    body = html.split("<tbody>")[1].split("</tbody>")[0]
    assert f"Job {RESULTS_PAGE_SIZE + 1}</td>" in body

    # Фильтр сокращает количество страниц
    alpha_count = 28 + 1  # Alpha среди ba-строк + одна stepstone
    assert alpha_count <= RESULTS_PAGE_SIZE
    html = client.get("/results?f_company=Alpha").data.decode("utf-8")
    body = html.split("<tbody>")[1].split("</tbody>")[0]
    assert "Job 56</td>" in body
    assert "Job 1</td>" not in body
    assert 'class="pagination"' not in html

    # Ссылки пагинации сохраняют активный фильтр
    html = client.get("/results?f_source=ba").data.decode("utf-8")
    assert "f_source=ba" in html.split('class="pagination"')[1].split("</nav>")[0]

    # Несуществующее значение — пустой результат с подсказкой
    html = client.get("/results?f_source=nope").data.decode("utf-8")
    assert "Ничего не найдено по заданным фильтрам" in html


def test_profile_page_shows_name(monkeypatch, tmp_path):
    _tmp_store(monkeypatch, tmp_path)
    client = app.test_client()
    _login(client)
    html = client.get("/profile").data.decode("utf-8")
    assert "Илья" in html
    assert "Добавить" in html
    assert "Загрузить" in html


def test_profile_details_are_saved_only_for_current_profile(monkeypatch, tmp_path):
    factory = _tmp_store(monkeypatch, tmp_path)
    store = factory()
    store.create_account("Alice")
    store.close()
    client = app.test_client()
    _login(client, "Alice")
    r = client.post(
        "/profile/details",
        data={
            "full_name": "Alice Example",
            "address": "Example Street 1",
            "email": "alice@example.test",
            "phone": "+49 123",
            "languages": "Deutsch C1, English B2",
            "preferred_letter_language": "en",
            "search_region": "Berlin",
        },
    )
    assert r.status_code == 200
    assert "Данные кандидата сохранены" in r.data.decode("utf-8")
    alice = factory(user="Alice")
    ilia = factory(user="Илья")
    try:
        assert alice.candidate_profile()["full_name"] == "Alice Example"
        assert alice.candidate_profile()["languages"] == ["Deutsch C1", "English B2"]
        assert ilia.candidate_profile()["full_name"] == ""
    finally:
        alice.close()
        ilia.close()


def test_profile_details_validate_email_and_language(monkeypatch, tmp_path):
    _tmp_store(monkeypatch, tmp_path)
    client = app.test_client()
    _login(client)
    invalid_email = client.post(
        "/profile/details",
        data={"email": "not-an-email", "preferred_letter_language": "de"},
    )
    assert invalid_email.status_code == 400
    invalid_language = client.post(
        "/profile/details",
        data={"email": "", "preferred_letter_language": "xx"},
    )
    assert invalid_language.status_code == 400


def test_profile_upload_saves_to_user_folder(monkeypatch, tmp_path):
    import io
    from reportlab.pdfgen import canvas

    _tmp_store(monkeypatch, tmp_path)
    monkeypatch.setattr("cv_agent.documents.CV_DIR", tmp_path / "cv")
    client = app.test_client()
    _login(client)
    pdf = io.BytesIO()
    c = canvas.Canvas(pdf)
    c.drawString(72, 700, "Synthetic CV")
    c.save()
    pdf_bytes = pdf.getvalue()
    data = {"docs": (io.BytesIO(pdf_bytes), "Lebenslauf.pdf")}
    r = client.post(
        "/profile/upload", data=data, content_type="multipart/form-data"
    )
    assert r.status_code == 200
    uploaded = tmp_path / "cv" / "Илья" / "Lebenslauf.pdf"
    assert uploaded.is_file()
    assert uploaded.read_bytes() == pdf_bytes
    html = r.data.decode("utf-8")
    assert "Lebenslauf.pdf" in html
    assert "application/pdf" in html
    assert "извлечено" in html
    assert "в контекст входит" in html


def test_profile_marks_blank_pdf_as_requiring_ocr(monkeypatch, tmp_path):
    import io
    from reportlab.pdfgen import canvas

    _tmp_store(monkeypatch, tmp_path)
    monkeypatch.setattr("cv_agent.documents.CV_DIR", tmp_path / "cv")
    client = app.test_client()
    _login(client)
    pdf = io.BytesIO()
    c = canvas.Canvas(pdf)
    c.showPage()
    c.save()
    r = client.post(
        "/profile/upload",
        data={"docs": (io.BytesIO(pdf.getvalue()), "scan.pdf")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    html = r.data.decode("utf-8")
    assert "scan.pdf" in html
    assert "Требуется OCR" in html


def test_profile_document_type_is_saved_as_manual(monkeypatch, tmp_path):
    factory = _tmp_store(monkeypatch, tmp_path)
    monkeypatch.setattr("cv_agent.documents.CV_DIR", tmp_path / "cv")
    folder = tmp_path / "cv" / "Илья"
    folder.mkdir(parents=True)
    (folder / "notes.md").write_text("Python und Datenanalyse", encoding="utf-8")
    client = app.test_client()
    _login(client)
    assert client.get("/profile").status_code == 200

    r = client.post(
        "/profile/document/notes.md/type", data={"document_type": "cv"}
    )
    assert r.status_code == 302
    store = factory(user="Илья")
    try:
        metadata = store.document_metadata()
        assert metadata[0]["document_type"] == "cv"
        assert metadata[0]["type_source"] == "manual"
    finally:
        store.close()


def test_profile_upload_accepts_utf8_text(monkeypatch, tmp_path):
    import io

    _tmp_store(monkeypatch, tmp_path)
    monkeypatch.setattr("cv_agent.documents.CV_DIR", tmp_path / "cv")
    client = app.test_client()
    _login(client)
    r = client.post(
        "/profile/upload",
        data={"docs": (io.BytesIO("Erfahrung mit Python".encode()), "profil.txt")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    assert (tmp_path / "cv" / "Илья" / "profil.txt").is_file()


def test_profile_upload_uses_uuid_folder_after_storage_migration(isolated_project):
    import io

    from cv_agent.paths import account_path
    from cv_agent.storage import apply_storage_migration
    from cv_agent.store import Store

    root = isolated_project
    data = root / "data"
    store = Store()
    try:
        account_id = store.account_id("Илья")
        apply_storage_migration(
            store,
            data_dir=data,
            legacy_cv_root=root / "cv",
            legacy_applications_root=root / "applications",
        )
    finally:
        store.close()

    client = app.test_client()
    _login(client)
    r = client.post(
        "/profile/upload",
        data={"docs": (io.BytesIO(b"Python und SQL"), "neu.txt")},
        content_type="multipart/form-data",
    )
    assert r.status_code == 200
    assert (account_path(data, account_id, "cv") / "neu.txt").is_file()
    assert not (root / "cv" / "Илья" / "neu.txt").exists()


def test_profile_upload_rejects_broken_pdf(monkeypatch, tmp_path):
    import io

    _tmp_store(monkeypatch, tmp_path)
    monkeypatch.setattr("cv_agent.documents.CV_DIR", tmp_path / "cv")
    client = app.test_client()
    _login(client)
    r = client.post(
        "/profile/upload",
        data={"docs": (io.BytesIO(b"%PDF-1.4 broken"), "broken.pdf")},
        content_type="multipart/form-data",
    )
    assert "повреждённый PDF" in r.data.decode("utf-8")
    assert not (tmp_path / "cv" / "Илья" / "broken.pdf").exists()


def test_profile_upload_rejects_unknown_extension(monkeypatch, tmp_path):
    import io

    _tmp_store(monkeypatch, tmp_path)
    monkeypatch.setattr("cv_agent.documents.CV_DIR", tmp_path / "cv")
    client = app.test_client()
    _login(client)
    data = {"docs": (io.BytesIO(b"x"), "virus.exe")}
    r = client.post(
        "/profile/upload", data=data, content_type="multipart/form-data"
    )
    assert "Не поддерживаются" in r.data.decode("utf-8")
    assert not (tmp_path / "cv" / "Илья" / "virus.exe").exists()


def test_profile_upload_rejects_unsupported_docx(monkeypatch, tmp_path):
    import io

    _tmp_store(monkeypatch, tmp_path)
    monkeypatch.setattr("cv_agent.documents.CV_DIR", tmp_path / "cv")
    client = app.test_client()
    _login(client)
    r = client.post(
        "/profile/upload",
        data={"docs": (io.BytesIO(b"not a docx"), "Lebenslauf.docx")},
        content_type="multipart/form-data",
    )
    assert "Не поддерживаются" in r.data.decode("utf-8")
    assert not (tmp_path / "cv" / "Илья" / "Lebenslauf.docx").exists()


def test_profile_upload_rejects_file_over_limit(monkeypatch, tmp_path):
    import io

    _tmp_store(monkeypatch, tmp_path)
    monkeypatch.setattr("cv_agent.documents.CV_DIR", tmp_path / "cv")
    monkeypatch.setattr("cv_agent.webapp.MAX_DOCUMENT_BYTES", 4)
    client = app.test_client()
    _login(client)
    r = client.post(
        "/profile/upload",
        data={"docs": (io.BytesIO(b"12345"), "notes.txt")},
        content_type="multipart/form-data",
    )
    assert "превышает лимит" in r.data.decode("utf-8")
    assert not (tmp_path / "cv" / "Илья" / "notes.txt").exists()


def test_profile_upload_enforces_profile_quota(monkeypatch, tmp_path):
    import io

    _tmp_store(monkeypatch, tmp_path)
    monkeypatch.setattr("cv_agent.documents.CV_DIR", tmp_path / "cv")
    monkeypatch.setattr("cv_agent.webapp.MAX_ACCOUNT_DOCUMENT_BYTES", 8)
    folder = tmp_path / "cv" / "Илья"
    folder.mkdir(parents=True)
    (folder / "existing.txt").write_bytes(b"12345")
    client = app.test_client()
    _login(client)
    r = client.post(
        "/profile/upload",
        data={"docs": (io.BytesIO(b"6789"), "new.txt")},
        content_type="multipart/form-data",
    )
    assert "квота профиля" in r.data.decode("utf-8")
    assert not (folder / "new.txt").exists()


def test_profile_delete_doc(monkeypatch, tmp_path):
    factory = _tmp_store(monkeypatch, tmp_path)
    monkeypatch.setattr("cv_agent.documents.CV_DIR", tmp_path / "cv")
    doc = tmp_path / "cv" / "Илья" / "alt.pdf"
    doc.parent.mkdir(parents=True)
    doc.write_bytes(b"%PDF")
    client = app.test_client()
    _login(client)
    assert client.get("/profile").status_code == 200
    store = factory(user="Илья")
    try:
        assert len(store.document_metadata()) == 1
    finally:
        store.close()
    r = client.post("/profile/delete/alt.pdf")
    assert r.status_code == 302
    assert not doc.exists()
    store = factory(user="Илья")
    try:
        assert store.document_metadata() == []
    finally:
        store.close()


def test_profile_gemini_key_save(monkeypatch, tmp_path):
    _tmp_store(monkeypatch, tmp_path)
    client = app.test_client()
    _login(client)
    r = client.post(
        "/profile/keys/add",
        data={"name": "gemini_api_key", "provider": "gemini", "api_key": "AIza-TESTKEY"},
    )
    assert "добавлен" in r.data.decode("utf-8")
    assert "AIza-TESTKEY" not in r.data.decode("utf-8")
    from cv_agent.store import Store

    s = Store(tmp_path / "test.db")
    assert s.account_gemini_key("Илья") == "AIza-TESTKEY"
    s.close()


def test_effective_env_uses_account_gemini_key(monkeypatch, tmp_path):
    _tmp_store(monkeypatch, tmp_path)
    from cv_agent.store import Store

    s = Store(tmp_path / "test.db")
    s.set_account_gemini_key("AIza-TEST", "Илья")
    s.close()
    monkeypatch.setattr("cv_agent.webapp._ollama_alive", lambda: True)
    client = app.test_client()
    _login(client)
    client.set_cookie("llm", GEMINI_OPTION_ID)
    with client:
        client.get("/")
        env, _ = _effective_llm_env()
    assert env["GEMINI_API_KEY"] == "AIza-TEST"


def test_effective_env_does_not_reuse_other_profile_key(monkeypatch, tmp_path):
    factory = _tmp_store(monkeypatch, tmp_path)
    store = factory(user="Alice")
    store.create_account("Alice")
    store.create_account("Bob")
    store.set_account_gemini_key("alice-secret", "Alice")
    store.close()
    client = app.test_client()
    _login(client, "Bob")
    client.set_cookie("llm", GEMINI_OPTION_ID)
    with client:
        client.get("/")
        env, _ = _effective_llm_env()
    assert "GEMINI_API_KEY" not in env


def test_subprocess_env_never_contains_gemini_key(monkeypatch, tmp_path):
    _tmp_store(monkeypatch, tmp_path)
    from cv_agent.store import Store

    store = Store(tmp_path / "test.db")
    store.set_account_gemini_key("profile-secret", "Илья")
    store.close()
    client = app.test_client()
    _login(client)
    client.set_cookie("llm", GEMINI_OPTION_ID)
    with client:
        client.get("/")
        from cv_agent.webapp import _subprocess_llm_env

        env, _ = _subprocess_llm_env()
    assert env["LLM_API_KEY_ENV"] == "GEMINI_API_KEY"
    assert "GEMINI_API_KEY" not in env


def test_available_llms_includes_groq_option():
    from cv_agent.webapp import available_llms

    groq = [o for o in available_llms() if o["provider"] == "groq"]
    assert len(groq) == 1
    assert groq[0]["id"] == GROQ_OPTION_ID
    assert groq[0]["base_url"] == "https://api.groq.com/openai/v1"
    assert groq[0]["api_key_env"] == "GROQ_API_KEY"


def test_available_llms_fetches_groq_models_dynamically(monkeypatch, tmp_path):
    from cv_agent import webapp
    from cv_agent.store import Store

    _tmp_store(monkeypatch, tmp_path)
    store = Store(tmp_path / "test.db")
    store.set_account_groq_key("gsk-DYNAMIC", "Илья")
    store.close()
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        assert headers["Authorization"] == "Bearer gsk-DYNAMIC"
        return type(
            "R",
            (),
            {
                "status_code": 200,
                "json": lambda self: {
                    "data": [
                        {"id": "llama-3.3-70b-versatile", "active": True},
                        {"id": "openai/gpt-oss-120b", "active": True},
                        {"id": "retired-model", "active": False},
                        {"id": "", "active": True},
                    ]
                },
            },
        )()

    monkeypatch.setattr(webapp.requests, "get", fake_get)
    monkeypatch.setattr(webapp, "_groq_models_cache", {})
    client = app.test_client()
    _login(client)

    with client:
        client.get("/")
        first = [o["id"] for o in webapp.available_llms() if o["provider"] == "groq"]
        second = [o["id"] for o in webapp.available_llms() if o["provider"] == "groq"]

    assert first == second
    assert "groq:llama-3.3-70b-versatile" in first
    assert "groq:openai/gpt-oss-120b" in first
    assert "groq:retired-model" not in first
    # Список кэшируется — один запрос к Groq на несколько вызовов подряд.
    assert len([c for c in calls if c.endswith("/openai/v1/models")]) == 1


def test_available_llms_groq_fallback_on_api_error(monkeypatch, tmp_path):
    import requests as requests_lib

    from cv_agent import webapp
    from cv_agent.store import Store

    _tmp_store(monkeypatch, tmp_path)
    store = Store(tmp_path / "test.db")
    store.set_account_groq_key("gsk-BROKEN", "Илья")
    store.close()

    def failing_get(url, headers=None, timeout=None):
        raise requests_lib.RequestException("down")

    monkeypatch.setattr(webapp.requests, "get", failing_get)
    monkeypatch.setattr(webapp, "_groq_models_cache", {})
    client = app.test_client()
    _login(client)

    groq = [o for o in webapp.available_llms() if o["provider"] == "groq"]
    assert [o["id"] for o in groq] == [GROQ_OPTION_ID]


def test_available_llms_fetches_gemini_models_dynamically(monkeypatch, tmp_path):
    from cv_agent import webapp
    from cv_agent.store import Store

    _tmp_store(monkeypatch, tmp_path)
    store = Store(tmp_path / "test.db")
    store.set_account_gemini_key("AIza-DYNAMIC", "Илья")
    store.close()
    calls = []

    def fake_get(url, params=None, headers=None, timeout=None):
        calls.append(url)
        assert params["key"] == "AIza-DYNAMIC"
        return type(
            "R",
            (),
            {
                "status_code": 200,
                "json": lambda self: {
                    "models": [
                        {
                            "name": "models/gemini-3.5-flash-lite",
                            "supportedGenerationMethods": ["generateContent"],
                            "inputTokenLimit": 1048576,
                            "outputTokenLimit": 65536,
                        },
                        {
                            "name": "models/gemini-3.5-pro",
                            "supportedGenerationMethods": ["generateContent"],
                        },
                        {
                            "name": "models/text-embedding-004",
                            "supportedGenerationMethods": ["embedContent"],
                        },
                    ]
                },
            },
        )()

    monkeypatch.setattr(webapp.requests, "get", fake_get)
    monkeypatch.setattr(webapp, "_gemini_models_cache", {})
    client = app.test_client()
    _login(client)

    with client:
        client.get("/")
        first = [o["id"] for o in webapp.available_llms() if o["provider"] == "gemini"]
        second = [o["id"] for o in webapp.available_llms() if o["provider"] == "gemini"]
        groups = webapp._llm_groups()

    assert first == second
    # В списке только бесплатные семейства Flash/Flash-Lite.
    assert first == [GEMINI_OPTION_ID, "gemini:gemini-3.5-flash-lite"]
    # Живой список замещает статический фолбэк из criteria.yaml без дубликатов.
    gemini_labels = [o["label"] for o in groups["gemini"]]
    assert len(gemini_labels) == len(set(gemini_labels)) == 1
    assert GEMINI_OPTION_ID not in [o["id"] for o in groups["gemini"]]
    # Карта лимитов полная: Pro виден в «Лимитах», но скрыт из выпадающего списка.
    limits = webapp._gemini_limits_map("AIza-DYNAMIC")
    assert limits["gemini-3.5-flash-lite"]["input"] == 1048576
    assert "gemini-3.5-pro" in limits
    assert "text-embedding-004" not in limits
    # Список кэшируется — один запрос к Gemini на несколько вызовов подряд.
    assert len(calls) == 1


def test_available_llms_gemini_fallback_on_api_error(monkeypatch, tmp_path):
    import requests as requests_lib

    from cv_agent import webapp
    from cv_agent.store import Store

    _tmp_store(monkeypatch, tmp_path)
    store = Store(tmp_path / "test.db")
    store.set_account_gemini_key("AIza-BROKEN", "Илья")
    store.close()

    def failing_get(url, **kwargs):
        raise requests_lib.RequestException("down")

    monkeypatch.setattr(webapp.requests, "get", failing_get)
    monkeypatch.setattr(webapp, "_gemini_models_cache", {})
    client = app.test_client()
    _login(client)

    gemini = [o for o in webapp.available_llms() if o["provider"] == "gemini"]
    assert [o["id"] for o in gemini] == [GEMINI_OPTION_ID]


def test_profile_groq_key_save(monkeypatch, tmp_path):
    _tmp_store(monkeypatch, tmp_path)
    client = app.test_client()
    _login(client)
    r = client.post(
        "/profile/keys/add",
        data={"name": "groq_api_key", "provider": "groq", "api_key": "gsk-TESTKEY"},
    )
    html = r.data.decode("utf-8")
    assert "добавлен" in html
    assert "gsk-TESTKEY" not in html
    from cv_agent.store import Store

    s = Store(tmp_path / "test.db")
    try:
        assert s.account_groq_key("Илья") == "gsk-TESTKEY"
    finally:
        s.close()
    r = client.post("/profile/keys/delete", data={"name": "groq_api_key"})
    assert "удалён" in r.data.decode("utf-8")
    s = Store(tmp_path / "test.db")
    try:
        assert s.account_groq_key("Илья") == ""
    finally:
        s.close()


def test_profile_keys_add_list_delete_flow(monkeypatch, tmp_path):
    """Полный цикл: добавление, отображение списка, удаление активного ключа."""
    from cv_agent.store import Store

    _tmp_store(monkeypatch, tmp_path)
    client = app.test_client()
    _login(client)

    # Два ключа Gemini: последний добавленный активен.
    client.post(
        "/profile/keys/add",
        data={"name": "first", "provider": "gemini", "api_key": "AIza-1"},
    )
    client.post(
        "/profile/keys/add",
        data={"name": "second", "provider": "gemini", "api_key": "AIza-2"},
    )
    html = client.get("/profile?section=keys").data.decode("utf-8")
    assert "first" in html and "second" in html
    assert "******" not in html
    assert "AIza-" not in html

    s = Store(tmp_path / "test.db")
    try:
        assert s.account_gemini_key("Илья") == "AIza-2"
    finally:
        s.close()

    # Удаление активного → активируется предыдущий.
    client.post("/profile/keys/delete", data={"name": "second"})
    s = Store(tmp_path / "test.db")
    try:
        assert s.account_gemini_key("Илья") == "AIza-1"
        assert [k["name"] for k in s.account_api_keys("Илья")] == ["first"]
    finally:
        s.close()

    # Повторное имя отклоняется.
    r = client.post(
        "/profile/keys/add",
        data={"name": "first", "provider": "gemini", "api_key": "AIza-3"},
    )
    assert "уже существует" in r.data.decode("utf-8")


def test_profile_keys_legacy_single_key_migrates_to_list(monkeypatch, tmp_path):
    from cv_agent.store import Store

    _tmp_store(monkeypatch, tmp_path)
    s = Store(tmp_path / "test.db")
    try:
        s.set_account_gemini_key("AIza-OLD", "Илья")
        keys = s.account_api_keys("Илья")
        assert keys == [{"name": "gemini_api_key", "provider": "gemini"}]
    finally:
        s.close()
    client = app.test_client()
    _login(client)
    html = client.get("/profile?section=keys").data.decode("utf-8")
    assert "gemini_api_key" in html


def test_profile_page_shows_model_and_groq_sections(monkeypatch, tmp_path):
    _tmp_store(monkeypatch, tmp_path)
    client = app.test_client()
    _login(client)
    html = client.get("/profile").data.decode("utf-8")
    assert "Добавить" in html
    assert "Провайдер" in html
    assert "Модель" in html
    assert GROQ_OPTION_ID in html
    assert GEMINI_OPTION_ID in html


def test_profile_provider_select_disables_ollama_when_down(monkeypatch, tmp_path):
    """Ollama недоступна — пункт бледный и запрещённый, остальные активны."""
    from cv_agent import webapp

    _tmp_store(monkeypatch, tmp_path)
    monkeypatch.setattr(webapp, "_ollama_alive", lambda timeout=2.0: False)
    client = app.test_client()
    _login(client)
    html = client.get("/profile?section=llm").data.decode("utf-8")
    assert '<option value="ollama" disabled class="provider-disabled">Ollama</option>' in html
    assert '<option value="gemini">Gemini</option>' in html
    assert '<option value="groq">Groq</option>' in html
    # Группы моделей для JS содержат модели провайдеров.
    assert "ollama:" in html and "groq:" in html and "gemini:" in html


def test_profile_provider_select_enables_ollama_when_up(monkeypatch, tmp_path):
    from cv_agent import webapp

    _tmp_store(monkeypatch, tmp_path)
    monkeypatch.setattr(webapp, "_ollama_alive", lambda timeout=2.0: True)
    client = app.test_client()
    _login(client)
    html = client.get("/profile?section=llm").data.decode("utf-8")
    assert '<option value="ollama">Ollama</option>' in html


def test_profile_menu_has_four_sections(monkeypatch, tmp_path):
    _tmp_store(monkeypatch, tmp_path)
    client = app.test_client()
    _login(client)
    html = client.get("/profile").data.decode("utf-8")
    for label in ("Профиль", "Документы", "LLM модель", "API keys"):
        assert f'data-section="{label}"' not in html  # кнопки, не секции
    for section in ("profile", "documents", "llm", "keys"):
        assert f'data-section="{section}"' in html
    # По умолчанию видима только вкладка «Профиль».
    assert '<section data-section="profile" >' in html
    assert '<section data-section="documents" hidden>' in html


def test_profile_section_query_selects_visible_tab(monkeypatch, tmp_path):
    _tmp_store(monkeypatch, tmp_path)
    client = app.test_client()
    _login(client)
    html = client.get("/profile?section=keys").data.decode("utf-8")
    assert '<section data-section="keys" >' in html
    assert '<section data-section="profile" hidden>' in html
    # Неизвестная секция — откат на «Профиль».
    html = client.get("/profile?section=evil").data.decode("utf-8")
    assert '<section data-section="profile" >' in html


def test_profile_llm_tab_shows_saved_model_and_dash_limits(monkeypatch, tmp_path):
    _tmp_store(monkeypatch, tmp_path)
    client = app.test_client()
    _login(client)
    html = client.get("/profile?section=llm").data.decode("utf-8")
    assert "Сохраненная модель:" in html
    assert '<div class="limits-title">Лимиты:</div>' in html
    assert '<div class="limits-stamp">-</div>' in html


def test_profile_llm_tab_shows_groq_limits(monkeypatch, tmp_path):
    from cv_agent import webapp
    from cv_agent.store import Store

    _tmp_store(monkeypatch, tmp_path)
    store = Store(tmp_path / "test.db")
    store.set_account_groq_key("gsk-LIMITS", "Илья")
    store.set_account_preferred_llm(GROQ_OPTION_ID, "Илья")
    store.close()
    calls = []

    def fake_get(url, headers=None, timeout=None):
        calls.append(url)
        return type(
            "R",
            (),
            {
                "status_code": 200,
                "headers": {
                    "x-ratelimit-limit-requests": "1000",
                    "x-ratelimit-remaining-requests": "970",
                    "x-ratelimit-limit-tokens": "12000",
                    "x-ratelimit-remaining-tokens": "11000",
                },
                "json": lambda self: {
                    "data": [
                        {
                            "id": "llama-3.3-70b-versatile",
                            "active": True,
                            "context_window": 131072,
                            "max_completion_tokens": 8192,
                        }
                    ]
                },
            },
        )()

    monkeypatch.setattr(webapp.requests, "get", fake_get)
    monkeypatch.setattr(webapp, "_groq_models_cache", {})
    monkeypatch.setattr(webapp, "_gemini_models_cache", {})
    client = app.test_client()
    _login(client)
    html = client.get("/profile?section=llm").data.decode("utf-8")
    assert "Сохраненная модель:" in html
    assert "Groq — llama-3.3-70b-versatile" in html
    assert "<li><span class=\"limits-label\">Контекст (токенов):</span>" in html
    assert "131 072" in html
    assert (
        "<li><span class=\"limits-label\">Макс. токенов ответ:</span>" in html
    )
    assert "8 192" in html
    assert (
        "<li><span class=\"limits-label\">Запросов в день:</span> "
        "970 из 1 000 (потрачено 30)</li>" in html
    )
    assert (
        "<li><span class=\"limits-label\">Токенов в минуту:</span> "
        "11 000 из 12 000 (потрачено 1 000)</li>" in html
    )
    # Список моделей запрашивается один раз и переиспользуется для лимитов.
    assert len([c for c in calls if c.endswith("/openai/v1/models")]) == 1


def test_profile_llm_tab_groq_limits_without_rate_headers(monkeypatch, tmp_path):
    """Без заголовков квот показываются лимиты модели, но не «-» вместо всего."""
    from cv_agent import webapp
    from cv_agent.store import Store

    _tmp_store(monkeypatch, tmp_path)
    store = Store(tmp_path / "test.db")
    store.set_account_groq_key("gsk-NORATE", "Илья")
    store.set_account_preferred_llm(GROQ_OPTION_ID, "Илья")
    store.close()

    def fake_get(url, headers=None, timeout=None):
        return type(
            "R",
            (),
            {
                "status_code": 200,
                "json": lambda self: {
                    "data": [
                        {
                            "id": "llama-3.3-70b-versatile",
                            "active": True,
                            "context_window": 131072,
                        }
                    ]
                },
            },
        )()

    monkeypatch.setattr(webapp.requests, "get", fake_get)
    monkeypatch.setattr(webapp, "_groq_models_cache", {})
    monkeypatch.setattr(webapp, "_gemini_models_cache", {})
    client = app.test_client()
    _login(client)
    html = client.get("/profile?section=llm").data.decode("utf-8")
    assert "Контекст (токенов):" in html
    assert "131 072" in html
    assert "Запросов в день" not in html
    assert "Токенов в минуту" not in html


def test_profile_llm_tab_shows_quota_from_db_snapshot(monkeypatch, tmp_path):
    """Квоты с реальных вызовов (снимок в SQLite) видны на странице."""
    from cv_agent.llm import provider_key
    from cv_agent import webapp
    from cv_agent.store import Store

    _tmp_store(monkeypatch, tmp_path)
    store = Store(tmp_path / "test.db")
    store.set_account_groq_key("gsk-DBSNAP", "Илья")
    store.set_account_preferred_llm(GROQ_OPTION_ID, "Илья")
    backend = provider_key(
        type(
            "Cfg",
            (),
            {
                "provider": "groq",
                "base_url": webapp.GROQ_BASE_URL,
                "model": "llama-3.3-70b-versatile",
            },
        )()
    )
    store.save_llm_quota(
        backend,
        {
            "x-ratelimit-limit-requests": "1000",
            "x-ratelimit-remaining-requests": "970",
            "x-ratelimit-limit-tokens": "12000",
            "x-ratelimit-remaining-tokens": "11000",
            "_usage": {"prompt_tokens": 5000, "completion_tokens": 700},
        },
    )
    store.close()

    def fake_get(url, headers=None, timeout=None):
        return type(
            "R",
            (),
            {
                "status_code": 200,
                "json": lambda self: {
                    "data": [
                        {
                            "id": "llama-3.3-70b-versatile",
                            "active": True,
                            "context_window": 131072,
                            "max_completion_tokens": 8192,
                        }
                    ]
                },
            },
        )()

    monkeypatch.setattr(webapp.requests, "get", fake_get)
    monkeypatch.setattr(webapp, "_groq_models_cache", {})
    monkeypatch.setattr(webapp, "_gemini_models_cache", {})
    client = app.test_client()
    _login(client)
    html = client.get("/profile?section=llm").data.decode("utf-8")
    assert (
        "<li><span class=\"limits-label\">Запросов в день:</span> "
        "970 из 1 000 (потрачено 30)</li>" in html
    )
    assert (
        "<li><span class=\"limits-label\">Токенов в минуту:</span> "
        "11 000 из 12 000 (потрачено 1 000)</li>" in html
    )
    assert (
        "126 072 из 131 072 (потрачено 5 000)" in html
    )
    assert (
        "7 492 из 8 192 (потрачено 700)" in html
    )
    assert "Данные от" in html


def test_profile_post_returns_to_same_tab(monkeypatch, tmp_path):
    _tmp_store(monkeypatch, tmp_path)
    client = app.test_client()
    _login(client)
    html = client.post(
        "/profile/keys/add",
        data={"name": "k1", "provider": "groq", "api_key": "gsk-X"},
    ).data.decode("utf-8")
    assert '<section data-section="keys" >' in html
    html = client.post("/profile/llm", data={"llm": GROQ_OPTION_ID}).data.decode(
        "utf-8"
    )
    assert '<section data-section="llm" >' in html


def test_profile_tab_shows_readonly_view_and_edit_toggle(monkeypatch, tmp_path):
    _tmp_store(monkeypatch, tmp_path)
    from cv_agent.store import Store

    store = Store(tmp_path / "test.db")
    store.set_candidate_profile(
        {
            "full_name": "Ilia Rerich",
            "address": "Musterstraße 1, 88239 Wangen im Allgäu",
            "email": "ilia@example.de",
            "phone": "+49 123 456789",
            "languages": ["Deutsch", "Русский"],
            "preferred_letter_language": "de",
            "search_region": "88239 Wangen im Allgäu",
        }
    )
    store.close()
    client = app.test_client()
    _login(client)
    html = client.get("/profile").data.decode("utf-8")

    # Режим просмотра: значения видны, кнопка «Редактировать» есть.
    assert "Полное имя:</strong> Ilia Rerich" in html
    assert "Musterstraße 1, 88239 Wangen im Allgäu" in html
    assert "Deutsch, Русский" in html
    assert 'id="edit-profile-btn"' in html

    # Форма редактирования присутствует, но скрыта до нажатия.
    edit_form = html.split('id="profile-edit-form"', 1)[1]
    assert edit_form.startswith(' method="post"')
    assert "hidden>" in edit_form[:200]
    assert ">Сохранить данные</button>" in html


def test_profile_model_selection_saves_and_drives_override(monkeypatch, tmp_path):
    _tmp_store(monkeypatch, tmp_path)
    client = app.test_client()
    _login(client)

    r = client.post("/profile/llm", data={"llm": GROQ_OPTION_ID})
    assert "Модель сохранена" in r.data.decode("utf-8")

    # Некорректный ID не из allowlist отклоняется.
    r = client.post("/profile/llm", data={"llm": "evil:http://attacker"})
    assert r.status_code == 400

    with client:
        client.get("/")
        from cv_agent.webapp import _effective_llm_env, _llm_override

        override = _llm_override()
        assert override["LLM_PROVIDER"] == "groq"
        assert override["LLM_BASE_URL"] == "https://api.groq.com/openai/v1"
        env, note = _effective_llm_env()
    assert env["LLM_MODEL"] == "llama-3.3-70b-versatile"
    assert env["LLM_API_KEY_ENV"] == "GROQ_API_KEY"
    assert note == ""


def test_profile_llm_save_sends_probe_and_updates_limits(monkeypatch, tmp_path):
    from cv_agent import webapp
    from cv_agent.store import Store

    _tmp_store(monkeypatch, tmp_path)
    store = Store(tmp_path / "test.db")
    store.set_account_groq_key("gsk-PROBE", "Илья")
    store.close()

    calls = []

    class FakeProbeClient:
        def __init__(self, cfg, api_key=None, **kwargs):
            calls.append((cfg.provider, cfg.model, api_key))

        def complete(self, system, user, **kwargs):
            return "OK"

    monkeypatch.setattr(webapp, "LLMClient", FakeProbeClient)
    client = app.test_client()
    _login(client)
    html = client.post(
        "/profile/llm", data={"llm": GROQ_OPTION_ID}
    ).data.decode("utf-8")
    assert "Модель сохранена" in html
    assert "Проверочный запрос выполнен" in html
    assert calls == [("groq", "llama-3.3-70b-versatile", "gsk-PROBE")]


def test_profile_llm_probe_failure_keeps_model_saved(monkeypatch, tmp_path):
    from cv_agent import webapp
    from cv_agent.llm import LLMError as LLMErrorClass
    from cv_agent.store import Store

    _tmp_store(monkeypatch, tmp_path)
    store = Store(tmp_path / "test.db")
    store.set_account_groq_key("gsk-FAIL", "Илья")
    store.close()

    class FailingClient:
        def __init__(self, cfg, api_key=None, **kwargs):
            pass

        def complete(self, system, user, **kwargs):
            raise LLMErrorClass("down after 3 попыток")

    monkeypatch.setattr(webapp, "LLMClient", FailingClient)
    client = app.test_client()
    _login(client)
    html = client.post(
        "/profile/llm", data={"llm": GROQ_OPTION_ID}
    ).data.decode("utf-8")
    assert "Модель сохранена" in html
    assert "Проверочный запрос не удался" in html


def test_profile_llm_probe_skipped_without_key(monkeypatch, tmp_path):
    _tmp_store(monkeypatch, tmp_path)
    client = app.test_client()
    _login(client)
    html = client.post(
        "/profile/llm", data={"llm": GROQ_OPTION_ID}
    ).data.decode("utf-8")
    assert "Модель сохранена" in html
    assert "проверочный запрос пропущен" in html


def test_profile_llm_save_overrides_run_cookie(monkeypatch, tmp_path):
    """Сохранение модели сбрасывает разовый /run-cookie: выбор применяется."""
    _tmp_store(monkeypatch, tmp_path)
    client = app.test_client()
    _login(client)
    client.set_cookie("llm", GROQ_OPTION_ID)

    # Cookie имеет приоритет над сохранённым выбором.
    html = client.get("/profile?section=llm").data.decode("utf-8")
    assert "Groq — llama-3.3-70b-versatile" in html

    # Явное сохранение Gemini сбрасывает cookie.
    html = client.post(
        "/profile/llm", data={"llm": GEMINI_OPTION_ID}
    ).data.decode("utf-8")
    assert "Модель сохранена" in html
    html = client.get("/profile?section=llm").data.decode("utf-8")
    m = re.search(r"Сохраненная модель:</strong>\s*([^<]*)", html)
    assert m and "gemini" in m.group(1)
    assert "groq" not in m.group(1).lower()


def test_profile_groq_key_reaches_effective_env(monkeypatch, tmp_path):
    _tmp_store(monkeypatch, tmp_path)
    from cv_agent.store import Store

    s = Store(tmp_path / "test.db")
    s.set_account_groq_key("gsk-TEST", "Илья")
    s.close()
    monkeypatch.setattr("cv_agent.webapp._ollama_alive", lambda: True)
    client = app.test_client()
    _login(client)
    client.set_cookie("llm", GROQ_OPTION_ID)
    with client:
        client.get("/")
        env, _ = _effective_llm_env()
    assert env["GROQ_API_KEY"] == "gsk-TEST"


def test_subprocess_env_never_contains_groq_key(monkeypatch, tmp_path):
    _tmp_store(monkeypatch, tmp_path)
    from cv_agent.store import Store

    store = Store(tmp_path / "test.db")
    store.set_account_groq_key("gsk-profile-secret", "Илья")
    store.close()
    client = app.test_client()
    _login(client)
    client.set_cookie("llm", GROQ_OPTION_ID)
    with client:
        client.get("/")
        from cv_agent.webapp import _subprocess_llm_env

        env, _ = _subprocess_llm_env()
    assert env["LLM_API_KEY_ENV"] == "GROQ_API_KEY"
    assert "GROQ_API_KEY" not in env


def test_pipeline_search_partial_success_gets_warning(monkeypatch):
    _fake_criteria(monkeypatch)

    def fake_run(args, extra_env=None, on_line=None):
        text = (
            "Source arbeitsagentur: queries=3 found=5 skipped=0 "
            "errors=2 duration=0.10s\nSkipped: 0; query errors: 2\n"
            if args and args[0] == "search"
            else "ok\n"
        )
        if on_line:
            for line in text.splitlines(keepends=True):
                on_line(line)
            return 0, ""
        return 0, text

    monkeypatch.setattr("cv_agent.webapp._run_cli", fake_run)
    client = app.test_client()
    _login(client)
    run_id = _start_pipeline(client, {})
    _wait(run_id)
    data = client.get(f"/pipeline/{run_id}/status").get_json()
    assert data["status"] == "succeeded"
    assert "Частичный успех" in data["steps"][0]["warning"]
    assert not data["steps"][1].get("warning")


def test_health_endpoint_reports_status_without_login():
    client = app.test_client()
    r = client.get("/health")
    assert r.status_code == 200
    data = r.get_json()
    assert data["ok"] is True
    assert data["db"] == "ok"
    assert data["schema_version"] >= 1
    assert isinstance(data["active_jobs"], int)
    assert data["ollama"] is False  # сеть отключена в тестах


def test_confirm_applied_route_marks_status(monkeypatch, tmp_path):
    from cv_agent.models import Vacancy
    from cv_agent.store import Store

    _tmp_store(monkeypatch, tmp_path)
    store = Store(tmp_path / "test.db", user="Илья")
    store.add(Vacancy("web-conf", "Job", "ACME", "Wangen", "u", "ba"))
    store.set_status("web-conf", "prepared")
    store.close()

    client = app.test_client()
    _login(client)

    page = client.get("/vacancy/web-conf").data.decode("utf-8")
    assert "Отклик на вакансию отправлен" in page

    r = client.post("/vacancy/web-conf/confirm-applied")
    assert r.status_code == 302
    store = Store(tmp_path / "test.db", user="Илья")
    try:
        assert store.application_status("web-conf") == "applied"
    finally:
        store.close()

    page = client.get("/vacancy/web-conf").data.decode("utf-8")
    # вместо кнопки — надпись с датой подтверждения; строки статуса нет
    assert "applied — отправлено" not in page
    import re as _re

    assert _re.search(r"Отклик на вакансию отправлен \d{2}\.\d{2}\.\d{4}", page)
    assert "Подтвердить: письмо отправлено" not in page


def test_retranslate_reports_success(monkeypatch, tmp_path):
    from cv_agent.models import Vacancy
    from cv_agent.store import Store

    _tmp_store(monkeypatch, tmp_path)
    store = Store(tmp_path / "test.db")
    store.add(Vacancy("v-ok", "Python", "ACME", "Wangen", "u", "ba"))
    store.update_description("v-ok", "Beschreibung", edited=False)
    store.close()

    def fake_translate(v):
        v.translation = "Русский перевод"

    monkeypatch.setattr("cv_agent.webapp._translate_vacancy", fake_translate)
    client = app.test_client()
    _login(client)
    r = client.post("/vacancy/v-ok/retranslate")
    assert r.status_code == 302
    assert "translated=1" in r.headers["Location"]
    html = client.get(r.headers["Location"]).data.decode("utf-8")
    assert "Перевод обновлён." in html
    assert "Русский перевод" in html


def test_retranslate_reports_failure_instead_of_silence(monkeypatch, tmp_path):
    from cv_agent.models import Vacancy
    from cv_agent.store import Store

    _tmp_store(monkeypatch, tmp_path)
    store = Store(tmp_path / "test.db")
    store.add(Vacancy("v-fail", "Python", "ACME", "Wangen", "u", "ba"))
    store.update_description("v-fail", "Beschreibung", edited=False)
    store.close()

    def broken_translate(v):
        raise RuntimeError("LLM error: connection refused")

    monkeypatch.setattr("cv_agent.webapp._translate_vacancy", broken_translate)
    client = app.test_client()
    _login(client)
    r = client.post("/vacancy/v-fail/retranslate")
    assert r.status_code == 302
    assert "llm_error=" in r.headers["Location"]
    html = client.get(r.headers["Location"]).data.decode("utf-8")
    assert "Перевод не удался" in html


def test_retranslate_without_description_is_reported(monkeypatch, tmp_path):
    from cv_agent.models import Vacancy
    from cv_agent.store import Store

    _tmp_store(monkeypatch, tmp_path)
    store = Store(tmp_path / "test.db")
    store.add(Vacancy("v-empty", "Python", "ACME", "Wangen", "u", "ba"))
    store.close()

    called = []
    monkeypatch.setattr(
        "cv_agent.webapp._translate_vacancy",
        lambda v: called.append(v),
    )
    client = app.test_client()
    _login(client)
    r = client.post("/vacancy/v-empty/retranslate")
    assert r.status_code == 302
    assert "llm_error=" in r.headers["Location"]
    import urllib.parse

    assert "описание вакансии недоступно" in urllib.parse.unquote_plus(
        r.headers["Location"]
    )
    assert called == []


def test_profile_requires_login():
    r = app.test_client().get("/profile")
    assert r.status_code == 302
    assert "/login" in r.headers["Location"]


def test_llm_client_accepts_key_only_env_without_keyerror(monkeypatch):
    captured = {}

    def fake_client(cfg, api_key=None):
        captured["model"] = cfg.model
        captured["api_key"] = api_key
        return object()

    monkeypatch.setattr("cv_agent.webapp.LLMClient", fake_client)
    from cv_agent.webapp import _llm_client

    _llm_client({"GEMINI_API_KEY": "not-for-ollama"})
    assert captured == {"model": "qwen3.8:latest", "api_key": None}


def test_llm_client_does_not_fall_back_to_global_key_for_gemini(monkeypatch):
    captured = {}

    def fake_client(cfg, api_key=None):
        captured["model"] = cfg.model
        captured["api_key"] = api_key
        return object()

    monkeypatch.setattr("cv_agent.webapp.LLMClient", fake_client)
    monkeypatch.setenv("GEMINI_API_KEY", "global-secret")
    from cv_agent.webapp import _llm_client

    _llm_client(
        {
            "LLM_MODEL": "gemini-3.5-flash-lite",
            "LLM_BASE_URL": "https://example.test/v1",
            "LLM_API_KEY_ENV": "GEMINI_API_KEY",
        }
    )
    assert captured == {"model": "gemini-3.5-flash-lite", "api_key": ""}


def test_translate_does_not_write_account_key_to_process_env(monkeypatch):
    from cv_agent.models import Vacancy
    from cv_agent.webapp import _translate_vacancy

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(
        "cv_agent.webapp._effective_llm_env",
        lambda: (
            {
                "LLM_MODEL": "gemini-3.5-flash-lite",
                "LLM_BASE_URL": "https://example.test/v1",
                "LLM_API_KEY_ENV": "GEMINI_API_KEY",
                "GEMINI_API_KEY": "profile-secret",
            },
            "",
        ),
    )

    class FakeLLM:
        def complete(self, system, user):
            return "перевод"

    monkeypatch.setattr("cv_agent.webapp._llm_client", lambda env: FakeLLM())
    vacancy = Vacancy("1", "Job", "ACME", "Wangen", "https://example.test", "ba")
    vacancy.description = "Beschreibung"
    _translate_vacancy(vacancy)
    assert vacancy.translation == "перевод"
    assert "GEMINI_API_KEY" not in __import__("os").environ


def test_gemini_limits_rows_match_groq_layout(monkeypatch):
    from cv_agent import webapp as wm

    monkeypatch.setattr(wm, "_account_llm_key", lambda env: "k")
    monkeypatch.setattr(
        wm,
        "_gemini_limits_map",
        lambda key: {"gemini-3.5-flash-lite": {"input": 1048576, "output": 65536}},
    )
    monkeypatch.setattr(wm, "_db_quota", lambda option: ({}, None, None))
    rows, stamp = wm._limits_rows(
        {"provider": "gemini", "model": "gemini-3.5-flash-lite"}
    )
    assert [r["label"] for r in rows] == [
        "Контекст (токенов)",
        "Макс. токенов ответ",
        "Запросов в день",
        "Токенов в минуту",
    ]
    assert rows[0]["value"] == "лимит 1 048 576"
    assert rows[2]["value"] == "—"
    assert rows[3]["value"] == "—"


def _keywords_section_html(client):
    return client.get("/profile?section=keywords").data.decode("utf-8")


def test_keywords_tab_lists_account_keywords(monkeypatch, tmp_path):
    from cv_agent.store import Store

    _tmp_store(monkeypatch, tmp_path)
    s = Store(tmp_path / "test.db")
    try:
        s.set_account_keywords(["Lagerist", "Kommissionierung"], "Илья")
    finally:
        s.close()
    client = app.test_client()
    _login(client)
    html = _keywords_section_html(client)
    assert "Сгенерировать из документов" in html
    assert ">Lagerist<" in html
    assert ">Kommissionierung<" in html
    # На вкладке «Профиль» списка больше нет (старое отображение через запятую).
    profile_html = client.get("/profile").data.decode("utf-8")
    assert "Lagerist, Kommissionierung" not in profile_html


def test_keywords_add_appends_and_rejects_duplicates(monkeypatch, tmp_path):
    from cv_agent.store import Store

    _tmp_store(monkeypatch, tmp_path)
    client = app.test_client()
    _login(client)
    r = client.post(
        "/profile/keywords/add", data={"keyword": "Schichtleitung"}
    )
    assert "добавлено" in r.data.decode("utf-8")
    r = client.post(
        "/profile/keywords/add", data={"keyword": "schichtleitung"}
    )
    assert "уже есть" in r.data.decode("utf-8")
    r = client.post("/profile/keywords/add", data={"keyword": "  "})
    assert "Укажите ключевое слово" in r.data.decode("utf-8")
    s = Store(tmp_path / "test.db")
    try:
        assert s.account_keywords("Илья")[-1] == "Schichtleitung"
        assert s.account_keywords("Илья").count("Schichtleitung") == 1
    finally:
        s.close()


def test_keywords_generate_runs_cli_apply(monkeypatch, tmp_path):
    _tmp_store(monkeypatch, tmp_path)
    calls = []

    def fake_run_cli(args, extra_env=None, timeout=3600, **kwargs):
        calls.append((list(args), dict(extra_env or {})))
        return 0, ""

    monkeypatch.setattr("cv_agent.webapp._run_cli", fake_run_cli)
    client = app.test_client()
    _login(client)
    r = client.post("/profile/keywords/generate")
    assert "сгенерированы" in r.data.decode("utf-8")
    assert calls and calls[0][0] == ["keywords", "--apply"]
    assert calls[0][1].get("CV_AGENT_USER") == "Илья"


def test_keywords_generate_reports_failure(monkeypatch, tmp_path):
    _tmp_store(monkeypatch, tmp_path)
    monkeypatch.setattr(
        "cv_agent.webapp._run_cli",
        lambda *a, **k: (1, "Папка cv пуста"),
    )
    client = app.test_client()
    _login(client)
    r = client.post("/profile/keywords/generate")
    html = r.data.decode("utf-8")
    assert "не удалась" in html
    assert "Папка cv пуста" in html


def test_keywords_delete_removes_word(monkeypatch, tmp_path):
    from cv_agent.store import Store

    _tmp_store(monkeypatch, tmp_path)
    s = Store(tmp_path / "test.db")
    try:
        s.set_account_keywords(["Alpha", "Beta"], "Илья")
    finally:
        s.close()
    client = app.test_client()
    _login(client)
    r = client.post("/profile/keywords/delete", data={"keyword": "Alpha"})
    assert "удалено" in r.data.decode("utf-8")
    r = client.post("/profile/keywords/delete", data={"keyword": "Alpha"})
    assert "не найдено" in r.data.decode("utf-8")
    html = _keywords_section_html(client)
    assert "<li" in html
    assert ">Alpha<" not in html
    assert ">Beta<" in html
    assert "Удалить" in html
    s = Store(tmp_path / "test.db")
    try:
        assert s.account_keywords("Илья") == ["Beta"]
    finally:
        s.close()


def test_vacancy_route_archives_removed_posting(monkeypatch, tmp_path):
    from cv_agent.models import Vacancy
    from cv_agent.search import VacancyUnavailable
    from cv_agent.store import Store as RealStore

    _tmp_store(monkeypatch, tmp_path)
    seed = RealStore(tmp_path / "test.db", user="Илья")
    seed.add(Vacancy("li-4449409532", "Job", "ACME", "Wangen", "u", "linkedin"))
    seed.close()

    def removed_fetch(vacancy):
        raise VacancyUnavailable("linkedin", "Job posting removed or invalid")

    monkeypatch.setattr(
        "cv_agent.webapp.fetch_description",
        removed_fetch,
    )

    client = app.test_client()
    _login(client)
    page = client.get("/vacancy/li-4449409532")
    assert page.status_code == 200

    check = RealStore(tmp_path / "test.db", user="Илья")
    try:
        lifecycle = check.vacancy_lifecycle("li-4449409532")
        assert lifecycle["source_status"] == "unavailable"
        assert lifecycle["archived_at"]
    finally:
        check.close()


def test_results_page_has_actualize_button(monkeypatch, tmp_path):
    _tmp_store(monkeypatch, tmp_path)
    client = app.test_client()
    _login(client)
    html = client.get("/results").data.decode("utf-8")
    assert "Актуализировать текущие вакансии" in html
    assert 'formaction="/actualize"' in html


def test_actualize_runs_background_job(monkeypatch):
    calls = _fake_run(
        monkeypatch,
        out="[1/1] li-1: still available\n"
            "Actualize summary: total 1, active 1, archived 0\n",
    )
    client = app.test_client()
    _login(client)
    r = client.post("/actualize")
    assert r.status_code == 302
    run_id = int(r.headers["Location"].rstrip("/").rsplit("/", 1)[1])
    _wait(run_id)
    assert calls == [["actualize"]]
    store_run = None
    from cv_agent.store import Store

    s = Store()
    try:
        store_run = s.job(run_id)
    finally:
        s.close()
    assert store_run["status"] == "succeeded"
    assert store_run["kind"] == "actualize"


def _score_filter_seed(monkeypatch, tmp_path):
    from cv_agent.models import Vacancy
    from cv_agent.store import Store

    _tmp_store(monkeypatch, tmp_path)
    store = Store(tmp_path / "test.db", user="Илья")
    store.create_account("Илья")
    store.add(Vacancy("low", "Job L", "ACME", "Wangen", "u", "ba", score=50.0))
    store.add(Vacancy("mid", "Job M", "ACME", "Wangen", "u", "ba", score=80.0))
    store.add(Vacancy("high", "Job H", "ACME", "Wangen", "u", "ba", score=95.0))
    store.close()


def test_score_filter_comparison_lt_eq_gt(monkeypatch, tmp_path):
    _score_filter_seed(monkeypatch, tmp_path)
    client = app.test_client()
    _login(client)

    def titles(query):
        html = client.get(f"/results?{query}").data.decode("utf-8")
        body = html.split("<tbody>")[1].split("</tbody>")[0]
        return [line for line in body.splitlines() if "<a href" in line]

    # < 80 — только low
    refs = "".join(titles("f_score=%3C80"))
    assert "/vacancy/low" in refs
    assert "/vacancy/mid" not in refs and "/vacancy/high" not in refs

    # = 80 — только mid
    refs = "".join(titles("f_score=%3D80"))
    assert "/vacancy/mid" in refs
    assert "/vacancy/low" not in refs and "/vacancy/high" not in refs

    # > 80 — только high
    refs = "".join(titles("f_score=%3E80"))
    assert "/vacancy/high" in refs
    assert "/vacancy/low" not in refs and "/vacancy/mid" not in refs


def test_score_filter_combines_comparison_with_exact_value(
    monkeypatch, tmp_path
):
    _score_filter_seed(monkeypatch, tmp_path)
    client = app.test_client()
    _login(client)
    # точное значение и сравнение объединяются через ИЛИ
    html = client.get("/results?f_score=50||%3E90").data.decode("utf-8")
    body = html.split("<tbody>")[1].split("</tbody>")[0]
    assert "/vacancy/low" in body
    assert "/vacancy/high" in body
    assert "/vacancy/mid" not in body


def test_score_filter_invalid_condition_is_exact_match(monkeypatch, tmp_path):
    _score_filter_seed(monkeypatch, tmp_path)
    client = app.test_client()
    _login(client)
    # невалидные условия не ломают страницу и работают как точный фильтр
    for raw in ("%3Cabc", "%3D", "%3E%3C", "100"):
        r = client.get(f"/results?f_score={raw}")
        assert r.status_code == 200
        body = (
            r.data.decode("utf-8")
            .split("<tbody>")[1]
            .split("</tbody>")[0]
        )
        assert 'class="empty-state"' in body or "/vacancy/" in body


def test_score_filter_skips_vacancies_without_score(monkeypatch, tmp_path):
    from cv_agent.models import Vacancy
    from cv_agent.store import Store

    _tmp_store(monkeypatch, tmp_path)
    store = Store(tmp_path / "test.db", user="Илья")
    store.create_account("Илья")
    store.add(Vacancy("noscore", "Job N", "ACME", "Wangen", "u", "ba"))
    store.add(Vacancy("scored", "Job S", "ACME", "Wangen", "u", "ba", score=40.0))
    store.close()
    client = app.test_client()
    _login(client)
    html = client.get("/results?f_score=%3C99").data.decode("utf-8")
    body = html.split("<tbody>")[1].split("</tbody>")[0]
    assert "/vacancy/scored" in body
    assert "/vacancy/noscore" not in body


def _seed_letter_vacancy(monkeypatch, tmp_path):
    from cv_agent.models import Vacancy
    from cv_agent.store import Store

    _tmp_store(monkeypatch, tmp_path)
    store = Store(tmp_path / "test.db", user="Илья")
    store.create_account("Илья")
    store.add(Vacancy("gi-289407", "Werkstudent IoT", "Deloitte", "München", "u", "get-in-it"))
    store.close()


def test_vacancy_without_letter_shows_create_button(monkeypatch, tmp_path):
    _seed_letter_vacancy(monkeypatch, tmp_path)
    client = app.test_client()
    _login(client)
    html = client.get("/vacancy/gi-289407").data.decode("utf-8")
    assert "Создать Anschreiben" in html
    assert "/anschreiben" not in html


def test_vacancy_with_letter_shows_document_instead_of_button(
    monkeypatch, tmp_path, isolated_project
):
    _seed_letter_vacancy(monkeypatch, tmp_path)
    dossier = isolated_project / "applications" / "Deloitte_gi-289407"
    dossier.mkdir(parents=True)
    (dossier / "Anschreiben.md").write_text(
        "Sehr geehrte Damen und Herren", encoding="utf-8"
    )
    client = app.test_client()
    _login(client)
    html = client.get("/vacancy/gi-289407").data.decode("utf-8")
    assert "Создать Anschreiben" not in html
    assert 'href="/vacancy/gi-289407/anschreiben"' in html
    assert 'target="_blank"' in html
    assert 'href="/vacancy/gi-289407/anschreiben/download"' in html
    assert "Скачать" in html


def test_anschreiben_view_serves_inline_in_new_tab(monkeypatch, isolated_project):
    dossier = isolated_project / "applications" / "Deloitte_gi-1"
    dossier.mkdir(parents=True)
    (dossier / "Anschreiben.md").write_text(
        "Sehr geehrte Damen und Herren", encoding="utf-8"
    )
    client = app.test_client()
    _login(client)
    r = client.get("/vacancy/gi-1/anschreiben")
    assert r.status_code == 200
    assert b"Sehr geehrte" in r.data
    assert "text/markdown" in r.headers["Content-Type"]
    assert "attachment" not in (r.headers.get("Content-Disposition") or "")


def test_anschreiben_download_sends_attachment(isolated_project):
    dossier = isolated_project / "applications" / "ACME_li-42"
    dossier.mkdir(parents=True)
    (dossier / "Anschreiben.md").write_text("Hallo", encoding="utf-8")
    client = app.test_client()
    _login(client)
    r = client.get("/vacancy/li-42/anschreiben/download")
    assert r.status_code == 200
    disposition = r.headers.get("Content-Disposition") or ""
    assert "attachment" in disposition
    assert "Anschreiben.md" in disposition


def test_anschreiben_latest_version_is_served(isolated_project):
    base = isolated_project / "applications"
    v1 = base / "ACME_ba-1000"
    v2 = base / "ACME_ba-1000_v2"
    for folder, text in ((v1, "version 1"), (v2, "version 2")):
        folder.mkdir(parents=True)
        (folder / "Anschreiben.md").write_text(text, encoding="utf-8")
    client = app.test_client()
    _login(client)
    r = client.get("/vacancy/ba-1000/anschreiben")
    assert b"version 2" in r.data


def test_anschreiben_routes_404_without_dossier():
    client = app.test_client()
    _login(client)
    assert client.get("/vacancy/gi-999999/anschreiben").status_code == 404
    assert (
        client.get("/vacancy/gi-999999/anschreiben/download").status_code
        == 404
    )


def test_vacancy_with_dossier_shows_lebenslauf_document(
    monkeypatch, tmp_path, isolated_project
):
    from cv_agent.models import Vacancy
    from cv_agent.store import Store

    _tmp_store(monkeypatch, tmp_path)
    store = Store(tmp_path / "test.db", user="Илья")
    store.create_account("Илья")
    store.add(Vacancy("ba-777", "Job", "ACME", "Wangen", "u", "ba"))
    store.close()

    dossier = isolated_project / "applications" / "ACME_ba-777"
    dossier.mkdir(parents=True)
    (dossier / "Anschreiben.md").write_text("Hallo", encoding="utf-8")
    (dossier / "Lebenslauf.pdf").write_bytes(b"%PDF-1.4 fake")

    client = app.test_client()
    _login(client)
    html = client.get("/vacancy/ba-777").data.decode("utf-8")
    assert 'href="/vacancy/ba-777/anschreiben"' in html
    assert 'href="/vacancy/ba-777/lebenslauf"' in html
    assert 'href="/vacancy/ba-777/lebenslauf/download"' in html
    assert "Lebenslauf.pdf" in html


def test_lebenslauf_view_serves_pdf_inline(isolated_project):
    dossier = isolated_project / "applications" / "ACME_li-5"
    dossier.mkdir(parents=True)
    (dossier / "Lebenslauf.pdf").write_bytes(b"%PDF-1.4 fake")
    client = app.test_client()
    _login(client)
    r = client.get("/vacancy/li-5/lebenslauf")
    assert r.status_code == 200
    assert b"%PDF" in r.data
    assert r.headers["Content-Type"].startswith("application/pdf")
    assert "attachment" not in (r.headers.get("Content-Disposition") or "")


def test_lebenslauf_download_sends_attachment(isolated_project):
    dossier = isolated_project / "applications" / "ACME_li-6"
    dossier.mkdir(parents=True)
    (dossier / "Lebenslauf.pdf").write_bytes(b"%PDF-1.4 fake")
    client = app.test_client()
    _login(client)
    r = client.get("/vacancy/li-6/lebenslauf/download")
    assert r.status_code == 200
    disposition = r.headers.get("Content-Disposition") or ""
    assert "attachment" in disposition and "Lebenslauf.pdf" in disposition


def test_lebenslauf_routes_404_without_dossier():
    client = app.test_client()
    _login(client)
    assert client.get("/vacancy/gi-1/lebenslauf").status_code == 404
    assert client.get("/vacancy/gi-1/lebenslauf/download").status_code == 404


def test_profile_resume_city_saved_and_rendered(monkeypatch, tmp_path):
    from cv_agent.store import Store

    _tmp_store(monkeypatch, tmp_path)
    client = app.test_client()
    _login(client)

    r = client.post(
        "/profile/details",
        data={
            "full_name": "Ilia Rerich",
            "email": "",
            "phone": "",
            "languages": "",
            "preferred_letter_language": "de",
            "search_region": "",
            "address": "",
            "resume_city": "Wangen im Allgäu",
        },
    )
    assert r.status_code == 200
    html = r.data.decode("utf-8")
    assert "Город для резюме" in html
    assert 'value="Wangen im Allgäu"' in html

    store = Store(tmp_path / "test.db", user="Илья")
    try:
        assert store.candidate_profile("Илья")["resume_city"] == "Wangen im Allgäu"
    finally:
        store.close()


def _seed_dossier_with_docs(isolated_project, refnr="ba-555", company="ACME"):
    dossier = isolated_project / "applications" / f"{company}_{refnr}"
    dossier.mkdir(parents=True)
    (dossier / "Anschreiben.md").write_text("Hallo", encoding="utf-8")
    (dossier / "Lebenslauf.pdf").write_bytes(b"%PDF-1.4 fake")
    return dossier


def test_dossier_delete_removes_anschreiben_only(monkeypatch, isolated_project):
    from cv_agent.models import Vacancy
    from cv_agent.store import Store

    store = Store()
    try:
        store.add(Vacancy("ba-555", "Job", "ACME", "Wangen", "u", "ba"))
    finally:
        store.close()

    dossier = _seed_dossier_with_docs(isolated_project)
    client = app.test_client()
    _login(client)

    r = client.post("/vacancy/ba-555/anschreiben/delete")
    assert r.status_code == 302
    assert r.headers["Location"].endswith("/vacancy/ba-555")

    # Anschreiben удалён, Lebenslauf и папка целы
    assert not (dossier / "Anschreiben.md").exists()
    assert (dossier / "Lebenslauf.pdf").is_file()

    html = client.get("/vacancy/ba-555").data.decode("utf-8")
    assert "/vacancy/ba-555/lebenslauf" in html
    assert 'href="/vacancy/ba-555/anschreiben"' not in html


def test_dossier_delete_removes_lebenslauf_and_hides_block(
    monkeypatch, isolated_project
):
    from cv_agent.models import Vacancy
    from cv_agent.store import Store

    store = Store()
    try:
        store.add(Vacancy("li-9", "Job", "ACME", "Wangen", "u", "linkedin"))
    finally:
        store.close()

    dossier = _seed_dossier_with_docs(isolated_project, refnr="li-9")
    client = app.test_client()
    _login(client)
    assert (
        client.post("/vacancy/li-9/lebenslauf/delete").status_code == 302
    )
    assert not (dossier / "Lebenslauf.pdf").exists()

    # удаляем и второй документ — блок документов исчезает,
    # снова доступна кнопка создания
    client.post("/vacancy/li-9/anschreiben/delete")
    html = client.get("/vacancy/li-9").data.decode("utf-8")
    assert "Создать Anschreiben" in html
    assert "/vacancy/li-9/anschreiben" not in html


def test_dossier_delete_404_for_missing_document():
    client = app.test_client()
    _login(client)
    assert (
        client.post("/vacancy/gi-424242/anschreiben/delete").status_code == 302
    )  # нет файла — просто возврат на страницу без ошибки


def test_vacancy_page_shows_delete_buttons_for_documents(
    monkeypatch, tmp_path, isolated_project
):
    from cv_agent.models import Vacancy
    from cv_agent.store import Store

    _tmp_store(monkeypatch, tmp_path)
    store = Store(tmp_path / "test.db", user="Илья")
    store.create_account("Илья")
    store.add(Vacancy("ss-1", "Job", "ACME", "Wangen", "u", "stepstone"))
    store.close()
    _seed_dossier_with_docs(isolated_project, refnr="ss-1")

    client = app.test_client()
    _login(client)
    html = client.get("/vacancy/ss-1").data.decode("utf-8")
    assert html.count(">Удалить</button>") >= 2
    assert "/vacancy/ss-1/anschreiben/delete" in html
    assert "/vacancy/ss-1/lebenslauf/delete" in html


def test_anschreiben_view_serves_pdf_for_new_dossiers(isolated_project):
    from cv_agent.models import Vacancy
    from cv_agent.store import Store

    store = Store()
    try:
        store.add(Vacancy("ba-80", "Job", "ACME", "Wangen", "u", "ba"))
    finally:
        store.close()
    dossier = isolated_project / "applications" / "ACME_ba-80"
    dossier.mkdir(parents=True)
    (dossier / "Anschreiben.pdf").write_bytes(b"%PDF-1.4 letter")
    client = app.test_client()
    _login(client)
    r = client.get("/vacancy/ba-80/anschreiben")
    assert r.status_code == 200
    assert r.headers["Content-Type"].startswith("application/pdf")

    page = client.get("/vacancy/ba-80").data.decode("utf-8")
    assert "Anschreiben.pdf" in page


def test_anschreiben_md_legacy_dossier_still_served(isolated_project):
    """Старые досье с Anschreiben.md продолжают открываться."""
    dossier = isolated_project / "applications" / "ACME_ba-81"
    dossier.mkdir(parents=True)
    (dossier / "Anschreiben.md").write_text("alter Stand", encoding="utf-8")
    client = app.test_client()
    _login(client)
    r = client.get("/vacancy/ba-81/anschreiben")
    assert r.status_code == 200
    assert b"alter Stand" in r.data
    assert "text/markdown" in r.headers["Content-Type"]
