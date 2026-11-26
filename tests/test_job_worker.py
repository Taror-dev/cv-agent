import threading
import time

from cv_agent.store import Store
from cv_agent.webapp import _launch_pipeline_worker as real_launch_pipeline_worker
from cv_agent.webapp import _run_cli as real_run_cli
from cv_agent.webapp import _start_pipeline_job as real_start_pipeline_job


def _steps():
    return [{"cmd": "search", "args": ["search"]}]


def test_launcher_uses_separate_module_and_allowlisted_secret_free_env(
    monkeypatch,
):
    import cv_agent.webapp as webapp

    captured = {}

    class Process:
        def wait(self):
            return 0

    def popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return Process()

    monkeypatch.setenv("GEMINI_API_KEY", "global-secret")
    monkeypatch.setattr(webapp.subprocess, "Popen", popen)

    real_launch_pipeline_worker(
        17,
        "00000000-0000-0000-0000-000000000017",
        "Alice",
        {
            "CV_AGENT_USER": "Alice",
            "GEMINI_API_KEY": "request-secret",
            "LLM_MODEL": "test-model",
            "UNSAFE_VARIABLE": "ignored",
        },
    )

    assert captured["cmd"][1:3] == ["-m", "cv_agent.job_worker"]
    assert captured["cmd"][-4:] == [
        "--run-id",
        "17",
        "--account-id",
        "00000000-0000-0000-0000-000000000017",
    ]
    env = captured["kwargs"]["env"]
    assert env["CV_AGENT_USER"] == "Alice"
    assert env["LLM_MODEL"] == "test-model"
    assert "GEMINI_API_KEY" not in env
    assert "UNSAFE_VARIABLE" not in env
    assert captured["kwargs"]["close_fds"] is True


def test_cli_child_env_drops_global_gemini_key(monkeypatch):
    import cv_agent.webapp as webapp

    captured = {}
    monkeypatch.setenv("GEMINI_API_KEY", "global-secret")
    monkeypatch.setattr(webapp, "load_env", lambda: None)
    monkeypatch.setattr(
        webapp,
        "_stream_command",
        lambda cmd, env, emit, **kwargs: captured.setdefault("env", env) or 0,
    )

    real_run_cli(["score"])

    assert "GEMINI_API_KEY" not in captured["env"]


def test_launcher_failure_finishes_created_job(monkeypatch):
    import cv_agent.webapp as webapp

    monkeypatch.setattr(
        webapp,
        "_launch_pipeline_worker",
        lambda *args: (_ for _ in ()).throw(OSError("spawn failed")),
    )

    run_id = real_start_pipeline_job("Илья", "pipeline", _steps(), "", {})

    store = Store()
    try:
        job = store.job(run_id)
    finally:
        store.close()
    assert job["status"] == "failed"
    assert "spawn failed" in job["error"]


def test_worker_crash_marks_the_job_failed(monkeypatch):
    import cv_agent.webapp as webapp

    store = Store()
    run_id, _ = store.create_job("pipeline", _steps())
    account_id = store.account_id("Илья")
    store.close()
    reaped = threading.Event()

    class Process:
        def wait(self):
            reaped.set()
            return 7

    monkeypatch.setattr(webapp.subprocess, "Popen", lambda *a, **kw: Process())
    real_launch_pipeline_worker(run_id, account_id, "Илья", {})
    assert reaped.wait(timeout=2)

    deadline = time.time() + 2
    while time.time() < deadline:
        check = Store()
        try:
            job = check.job(run_id)
        finally:
            check.close()
        if job["done"]:
            break
        time.sleep(0.01)
    assert job["status"] == "failed"
    assert "Code 7" in job["error"]


def test_worker_entry_resolves_stable_account_id(monkeypatch):
    import cv_agent.job_worker as worker

    store = Store()
    store.create_account("Alice")
    account_id = store.account_id("Alice")
    store.close()
    seen = []
    monkeypatch.setattr(
        worker,
        "_pipeline_worker",
        lambda run_id, user, extra_env: seen.append((run_id, user, extra_env)),
    )

    assert worker.main(
        ["--run-id", "23", "--account-id", account_id]
    ) == 0
    # воркер пробрасывает resolved user явно (аудит 22.08: шов worker -> CLI)
    assert seen == [(23, "Alice", {"CV_AGENT_USER": "Alice"})]


def test_worker_entry_rejects_unknown_account(monkeypatch):
    import cv_agent.job_worker as worker

    monkeypatch.setattr(
        worker,
        "_pipeline_worker",
        lambda *args: (_ for _ in ()).throw(AssertionError("must not run")),
    )
    assert worker.main(
        [
            "--run-id",
            "23",
            "--account-id",
            "00000000-0000-0000-0000-000000000099",
        ]
    ) == 2
