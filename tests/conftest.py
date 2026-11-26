"""Общие fixtures: каждый тест работает в полностью временном project root."""

from __future__ import annotations

import shutil
import subprocess
import threading
from pathlib import Path

import pytest
import requests
from reportlab.pdfgen import canvas


CRITERIA_YAML = """\
region: 88239 Wangen im Allgäu
radius_km: 50
limits:
  network_concurrency: 4
  llm_backend_concurrency: 1
keywords:
  - Python
  - AI Engineer
exclude_keywords:
  - Zeitarbeit
min_score: 70
recruiting_agencies:
  keywords:
    - Recruiting
  companies:
    - FERCHAU
llm:
  provider: ollama
  model: qwen3.8:latest
  base_url: http://localhost:11434/v1
  api_key_env: OLLAMA_API_KEY
  temperature: 0.3
llm_fallback:
  provider: gemini
  model: gemini-3.5-flash-lite
  base_url: https://generativelanguage.googleapis.com/v1beta/openai/
  api_key_env: GEMINI_API_KEY
  temperature: 0.3
"""

PROFILE_YAML = """\
name: Ilia Rerich
skills:
  - Python
  - SQL
experience:
  - role: Softwareentwickler
    tasks:
      - Systemintegration
"""


def _pdf(path: Path, *lines: str) -> None:
    c = canvas.Canvas(str(path))
    y = 800
    for line in lines:
        c.drawString(72, y, line)
        y -= 20
    c.save()


@pytest.fixture(scope="session")
def _project_template(tmp_path_factory) -> Path:
    """Минимальный синтетический project root без пользовательских данных."""
    root = tmp_path_factory.mktemp("cv-agent-template")
    (root / "criteria.yaml").write_text(CRITERIA_YAML, encoding="utf-8")
    (root / "profile.yaml").write_text(PROFILE_YAML, encoding="utf-8")
    (root / "data").mkdir()
    (root / "applications").mkdir()
    cv = root / "cv" / "Илья"
    cv.mkdir(parents=True)
    _pdf(cv / "AI Engineer Program Curriculum.pdf", "AI Engineer", "RAG Program")
    _pdf(cv / "Lebenslauf-Rerich-Ilia.pdf", "Ilia Rerich", "Python SQL")
    return root


@pytest.fixture(autouse=True)
def isolated_project(monkeypatch, tmp_path, _project_template) -> Path:
    """Перенаправляет DB, документы, dossiers, env и lock каждого теста в tmp."""
    root = tmp_path / "project"
    shutil.copytree(_project_template, root)
    data_dir = root / "data"
    applications_dir = root / "applications"

    for key in (
        "CV_AGENT_USER",
        "GEMINI_API_KEY",
        "OLLAMA_API_KEY",
        "OLLAMA_HOST",
        "LLM_MODEL",
        "LLM_PROVIDER",
        "LLM_BASE_URL",
        "LLM_API_KEY_ENV",
        "CV_AGENT_RUN_ID",
    ):
        monkeypatch.delenv(key, raising=False)

    import cv_agent.apply as apply_module
    import cv_agent.cli as cli_module
    import cv_agent.config as config_module
    import cv_agent.documents as documents_module
    import cv_agent.resource_limits as resource_limits_module
    import cv_agent.store as store_module
    import cv_agent.webapp as webapp_module
    import cv_agent.write as write_module

    monkeypatch.setattr(config_module, "ROOT", root)
    monkeypatch.setattr(config_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(config_module, "APPLICATIONS_DIR", applications_dir)

    monkeypatch.setattr(store_module, "ROOT", root)
    monkeypatch.setattr(store_module, "DATA_DIR", data_dir)
    monkeypatch.setattr(documents_module, "CV_DIR", root / "cv")

    monkeypatch.setattr(apply_module, "ROOT", root)
    monkeypatch.setattr(apply_module, "APPLICATIONS_DIR", applications_dir)
    monkeypatch.setattr(
        apply_module,
        "CV_PDF",
        root / "cv" / "Lebenslauf-Rerich-Ilia.pdf",
    )
    monkeypatch.setattr(write_module, "APPLICATIONS_DIR", applications_dir)
    monkeypatch.setattr(cli_module, "ROOT", root)

    monkeypatch.setattr(resource_limits_module, "DATA_DIR", data_dir)

    monkeypatch.setattr(webapp_module, "APPLICATIONS_DIR", applications_dir)

    real_store = store_module.Store

    def isolated_store(user=None):
        return real_store(data_dir / "applications.db", user=user)

    monkeypatch.setattr(webapp_module, "Store", isolated_store)

    def offline_get(*args, **kwargs):
        raise requests.ConnectionError("network disabled in tests")

    monkeypatch.setattr(requests, "get", offline_get)

    def fake_cli(args, extra_env=None, on_line=None):
        output = "isolated test command\n"
        if on_line:
            on_line(output)
            return 0, ""
        return 0, output

    monkeypatch.setattr(webapp_module, "_run_cli", fake_cli)

    def fake_worker_launcher(run_id, account_id, user, extra_env):
        worker = threading.Thread(
            target=webapp_module._pipeline_worker,
            args=(run_id, user, dict(extra_env)),
            daemon=True,
        )
        worker.start()

    monkeypatch.setattr(
        webapp_module, "_launch_pipeline_worker", fake_worker_launcher
    )

    def fake_subprocess_run(cmd, *args, **kwargs):
        if list(cmd[:2]) == ["ollama", "list"]:
            return subprocess.CompletedProcess(
                cmd,
                0,
                stdout="NAME ID SIZE MODIFIED\nqwen3.8:latest test 1GB now\n",
                stderr="",
            )
        raise AssertionError(f"unerwarteter subprocess im Test: {cmd}")

    monkeypatch.setattr(webapp_module.subprocess, "run", fake_subprocess_run)

    yield root
