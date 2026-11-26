"""Этап 2.2: request-specific значения не наследуются CLI-потомком из os.environ."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cv_agent import config
from cv_agent.webapp import _child_base_env, _launch_pipeline_worker, _run_cli


def test_child_env_strips_request_specific_and_account_keys(monkeypatch):
    monkeypatch.setenv("CV_AGENT_USER", "Ghost")
    monkeypatch.setenv("CV_AGENT_RUN_ID", "777")
    monkeypatch.setenv("LLM_MODEL", "ghost-model")
    monkeypatch.setenv("GEMINI_API_KEY", "leak")
    monkeypatch.setenv("GROQ_API_KEY", "leak")

    env = _child_base_env()

    for key in (
        "CV_AGENT_USER",
        "CV_AGENT_RUN_ID",
        "LLM_MODEL",
        "GEMINI_API_KEY",
        "GROQ_API_KEY",
    ):
        assert key not in env


def test_run_cli_env_sanitized_dotenv_applies_web_not_polluted(monkeypatch):
    (config.ROOT / ".env").write_text(
        "TEST_DOTENV_MARKER=abc\n", encoding="utf-8"
    )
    captured = {}

    def fake_stream(cmd, env, emit, **kwargs):
        captured.update(env)
        emit("ok\n")
        return 0

    monkeypatch.setattr("cv_agent.webapp._stream_command", fake_stream)
    monkeypatch.setenv("CV_AGENT_USER", "Ghost")
    monkeypatch.setenv("CV_AGENT_RUN_ID", "777")
    monkeypatch.setenv("GEMINI_API_KEY", "leak")

    code, _output = _run_cli(["status"], extra_env={"CV_AGENT_USER": "Илья"})

    assert code == 0
    assert captured["CV_AGENT_USER"] == "Илья"
    assert "CV_AGENT_RUN_ID" not in captured
    assert "GEMINI_API_KEY" not in captured
    assert captured["TEST_DOTENV_MARKER"] == "abc"
    # веб-процесс не загрязнён содержимым .env при обработке запроса
    assert "TEST_DOTENV_MARKER" not in __import__("os").environ


def test_worker_launch_env_uses_same_base(monkeypatch):
    captured = {}

    class FakeProc:
        pid = 4242

        @staticmethod
        def wait():
            return 0

    def fake_popen(cmd, **kwargs):
        captured.update(kwargs["env"])
        return FakeProc()

    monkeypatch.setattr("cv_agent.webapp.subprocess.Popen", fake_popen)
    monkeypatch.setenv("CV_AGENT_USER", "Ghost")
    monkeypatch.setenv("CV_AGENT_RUN_ID", "777")
    monkeypatch.setenv("GEMINI_API_KEY", "leak")

    _launch_pipeline_worker(1, "account-id", "Илья", {"LLM_MODEL": "m"})

    assert captured["CV_AGENT_USER"] == "Илья"
    assert "CV_AGENT_RUN_ID" not in captured
    assert "GEMINI_API_KEY" not in captured
    assert captured["LLM_MODEL"] == "m"
