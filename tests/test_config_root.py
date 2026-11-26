"""Этап 7.1: каталог данных через CV_AGENT_DATA_DIR вместо checkout-путей."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def test_data_dir_env_overrides_project_root(monkeypatch, tmp_path):
    import importlib

    monkeypatch.setenv("CV_AGENT_DATA_DIR", str(tmp_path / "custom"))
    import cv_agent.config as config

    reloaded = importlib.reload(config)
    try:
        assert reloaded.ROOT == (tmp_path / "custom").resolve()
        assert reloaded.DATA_DIR == (tmp_path / "custom" / "data").resolve()
        assert (tmp_path / "custom").is_dir()  # создан с правами 0700
        mode = (tmp_path / "custom").stat().st_mode & 0o777
        assert mode == 0o700
    finally:
        monkeypatch.delenv("CV_AGENT_DATA_DIR")
        importlib.reload(config)


def test_data_dir_defaults_to_checkout_root(monkeypatch):
    import importlib

    monkeypatch.delenv("CV_AGENT_DATA_DIR", raising=False)
    import cv_agent.config as config

    reloaded = importlib.reload(config)
    try:
        assert reloaded.ROOT == Path(__file__).resolve().parents[1]
        assert reloaded.DATA_DIR == Path(__file__).resolve().parents[1] / "data"
    finally:
        importlib.reload(config)
