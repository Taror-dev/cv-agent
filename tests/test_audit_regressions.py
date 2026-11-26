"""Аудит 22.08: шов web -> job_worker -> CLI не теряет контекст профиля."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cv_agent import config
from cv_agent.store import Store
from cv_agent.webapp import app


def test_worker_forwards_profile_context_to_cli(monkeypatch):
    """CLI-потомок воркера должен получить CV_AGENT_USER реального профиля.

    Регрессия: санитизация _child_base_env убирает request-specific ключи из
    наследуемого os.environ; если воркер не пробрасывает их явно, шаги
    pipeline выполняются под DEFAULT_USER и пишут в чужой аккаунт.
    """
    data_dir = Path(config.DATA_DIR)
    store = Store(data_dir / "applications.db")
    try:
        assert store.create_account("WorkerAnna")
        anna_id = store.account_id("WorkerAnna")
    finally:
        store.close()

    run_store = Store(data_dir / "applications.db", user="WorkerAnna")
    try:
        run_id, created = run_store.create_job(
            "pipeline", [{"cmd": "score", "args": ["score"]}]
        )
        assert created  # job остаётся queued — его claim'нет воркер
    finally:
        run_store.close()

    captured = {}

    def fake_run(args, extra_env=None, on_line=None):
        captured.update(extra_env or {})
        if on_line:
            on_line("ok\n")
            return 0, ""
        return 0, "ok\n"

    monkeypatch.setattr("cv_agent.webapp._run_cli", fake_run)
    # окружение worker-процесса, как его формирует _launch_pipeline_worker
    monkeypatch.setenv("CV_AGENT_USER", "WorkerAnna")
    monkeypatch.setenv("LLM_MODEL", "chosen-model")

    from cv_agent import job_worker

    assert job_worker.main(["--run-id", str(run_id), "--account-id", anna_id]) == 0

    assert captured.get("CV_AGENT_USER") == "WorkerAnna"
    assert captured.get("LLM_MODEL") == "chosen-model"
    assert captured.get("CV_AGENT_RUN_ID") == str(run_id)

    finish = Store(data_dir / "applications.db", user="WorkerAnna")
    try:
        job = finish.job(run_id)
        assert job is not None and job["done"]
        assert job["status"] == "succeeded", job["error"]
    finally:
        finish.close()
