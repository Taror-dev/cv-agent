import os

import pytest

from cv_agent.config import (
    DocumentLimits,
    ExclusionRules,
    LLMConfig,
    document_limits_for,
    is_ollama_url,
    llm_backend_class,
    llm_endpoint_allowed,
    load_criteria,
    load_profile,
    ollama_alive,
    ollama_host,
    ollama_openai_url,
    pick_llm,
)


def _criteria_dict(**overrides):
    value = {
        "region": "Wangen",
        "radius_km": 50,
        "keywords": ["Python"],
        "min_score": 70,
    }
    value.update(overrides)
    return value


class _NoAccountStore:
    def __init__(self, user=None):
        pass

    def account_keywords(self, name):
        return None

    def candidate_profile(self, name):
        return None

    def close(self):
        pass


def test_criteria_loads():
    c = load_criteria()
    assert c.region
    assert c.keywords
    assert c.exclude_keywords
    assert c.llm.model
    assert c.search_max_keywords == 12
    assert c.search_per_query_limit == 25
    assert c.search_total_limit == 100
    assert c.search_similarity_threshold == 0.9
    assert c.getinit_categories == []
    assert c.description_ttl_hours == 168
    assert c.network_concurrency == 4
    assert c.llm_backend_concurrency == 1


def test_criteria_loads_scoped_exclusions_and_prefers_min_score(monkeypatch):
    raw = _criteria_dict(
        results_min_score=10,
        exclusions={
            "title": {"words": ["Sales", "Sales"], "regex": []},
            "company": {"words": [], "regex": [r"^Bad Staff$"]},
            "description": {},
        },
    )
    monkeypatch.setattr("cv_agent.config.load_yaml", lambda name: raw)
    monkeypatch.setattr("cv_agent.store.Store", _NoAccountStore)

    criteria = load_criteria()

    assert criteria.min_score == criteria.results_min_score == 70
    assert criteria.exclusions["title"] == ExclusionRules(words=["Sales"])
    assert criteria.exclusions["company"].regex == [r"^Bad Staff$"]


def test_criteria_reads_legacy_results_threshold_only_as_fallback(monkeypatch):
    raw = _criteria_dict(results_min_score=65)
    raw.pop("min_score")
    monkeypatch.setattr("cv_agent.config.load_yaml", lambda name: raw)
    monkeypatch.setattr("cv_agent.store.Store", _NoAccountStore)

    assert load_criteria().min_score == 65


def test_runtime_concurrency_limits_are_validated(monkeypatch):
    raw = _criteria_dict(
        limits={"network_concurrency": 7, "llm_backend_concurrency": 2}
    )
    monkeypatch.setattr("cv_agent.config.load_yaml", lambda name: raw)
    monkeypatch.setattr("cv_agent.store.Store", _NoAccountStore)

    criteria = load_criteria()

    assert criteria.network_concurrency == 7
    assert criteria.llm_backend_concurrency == 2


def test_runtime_concurrency_rejects_unbounded_values(monkeypatch):
    raw = _criteria_dict(limits={"network_concurrency": 33})
    monkeypatch.setattr("cv_agent.config.load_yaml", lambda name: raw)
    monkeypatch.setattr("cv_agent.store.Store", _NoAccountStore)

    with pytest.raises(ValueError, match="network_concurrency"):
        load_criteria()


def test_criteria_rejects_invalid_exclusion_regex(monkeypatch):
    raw = _criteria_dict(
        exclusions={"description": {"regex": ["("]}},
    )
    monkeypatch.setattr("cv_agent.config.load_yaml", lambda name: raw)
    monkeypatch.setattr("cv_agent.store.Store", _NoAccountStore)

    with pytest.raises(ValueError, match="exclusions.description.regex"):
        load_criteria()


def test_llm_fallback_loads():
    c = load_criteria()
    assert c.llm_fallback is not None
    assert c.llm_fallback.model
    assert c.llm_fallback.api_key_env == "GEMINI_API_KEY"


def test_llm_provider_override_tracks_actual_server_selected_provider(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    monkeypatch.setenv("LLM_MODEL", "gemini-test")
    monkeypatch.setenv(
        "LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
    )
    assert load_criteria().llm.provider == "gemini"


def test_ollama_host_defaults():
    assert ollama_host() == "http://localhost:11434"
    assert ollama_openai_url() == "http://localhost:11434/v1"


def test_ollama_host_from_env(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "192.168.1.50:11434")
    assert ollama_host() == "http://192.168.1.50:11434"
    assert ollama_openai_url() == "http://192.168.1.50:11434/v1"


def test_ollama_host_with_scheme(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "https://ollama.lan:11434/")
    assert ollama_host() == "https://ollama.lan:11434"
    assert ollama_openai_url() == "https://ollama.lan:11434/v1"


def test_criteria_uses_ollama_host_env(monkeypatch):
    monkeypatch.delenv("LLM_BASE_URL", raising=False)
    monkeypatch.setenv("OLLAMA_HOST", "192.168.1.50:11434")
    c = load_criteria()
    assert c.llm.base_url == "http://192.168.1.50:11434/v1"


def test_criteria_keeps_explicit_llm_base_url(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://10.0.0.9:11434/v1")
    monkeypatch.setenv("OLLAMA_HOST", "192.168.1.50:11434")
    c = load_criteria()
    assert c.llm.base_url == "http://10.0.0.9:11434/v1"


def test_ollama_alive_uses_host_env(monkeypatch):
    import requests

    seen = {}

    def fake_get(url, timeout):
        seen["url"] = url
        return type("R", (), {"status_code": 200})()

    monkeypatch.setattr(requests, "get", fake_get)
    assert ollama_alive() is True
    assert seen["url"] == "http://localhost:11434/api/tags"


def test_ollama_alive_down(monkeypatch):
    import requests

    def fake_get(url, timeout):
        raise requests.ConnectionError("down")

    monkeypatch.setattr(requests, "get", fake_get)
    assert ollama_alive() is False


def test_is_ollama_url(monkeypatch):
    monkeypatch.setenv("OLLAMA_HOST", "localhost:11434")
    assert is_ollama_url("http://localhost:11434/v1")
    assert not is_ollama_url("http://192.168.1.50:11434/v1")
    assert not is_ollama_url("https://generativelanguage.googleapis.com/v1beta/openai/")
    monkeypatch.setenv("OLLAMA_HOST", "192.168.1.50:11434")
    assert is_ollama_url("http://192.168.1.50:11434/v1")


@pytest.mark.parametrize(
    ("provider", "url", "allowed"),
    [
        ("ollama", "http://localhost:11434/v1", True),
        ("ollama", "http://192.168.1.50:11434/v1", True),
        ("gemini", "https://generativelanguage.googleapis.com/v1beta/openai/", True),
        ("gemini", "http://generativelanguage.googleapis.com/v1beta/openai/", False),
        ("gemini", "https://127.0.0.1/v1", False),
        ("gemini", "https://169.254.169.254/v1", False),
        ("gemini", "https://evil.example/v1", False),
        ("deepseek", "https://api.deepseek.com/v1", True),
        ("deepseek", "https://user:secret@example.test/v1", False),
    ],
)
def test_llm_endpoint_allowlist(provider, url, allowed):
    assert llm_endpoint_allowed(LLMConfig(provider=provider, base_url=url)) is allowed


def test_pick_llm_fallback_when_ollama_down(monkeypatch):
    monkeypatch.setattr("cv_agent.config.ollama_alive", lambda: False)
    c = load_criteria()
    assert pick_llm(c).model == c.llm_fallback.model


def test_pick_llm_keeps_ollama_when_up(monkeypatch):
    monkeypatch.setattr("cv_agent.config.ollama_alive", lambda: True)
    c = load_criteria()
    assert pick_llm(c) is c.llm


def test_pick_llm_keeps_gemini_when_explicit(monkeypatch):
    monkeypatch.setattr("cv_agent.config.ollama_alive", lambda: False)
    monkeypatch.setenv("LLM_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
    monkeypatch.setenv("LLM_MODEL", "gemini-3.5-flash-lite")
    c = load_criteria()
    assert pick_llm(c).model == "gemini-3.5-flash-lite"


def test_backend_class_and_document_limits_resolution():
    ollama_cfg = LLMConfig(
        provider="ollama",
        model="qwen3.8:latest",
        base_url="http://localhost:11434/v1",
        api_key_env="OLLAMA_API_KEY",
    )
    external_cfg = LLMConfig(provider="gemini", base_url="https://generativelanguage.googleapis.com/v1beta/openai/")

    assert llm_backend_class(ollama_cfg) == "ollama"
    assert llm_backend_class(external_cfg) == "external"

    criteria = load_criteria()
    limits = document_limits_for(criteria, ollama_cfg)
    assert isinstance(limits, DocumentLimits)
    assert limits.max_chars_per_file == 6000
    assert limits.budget("scoring") == 12000
    external_limits = document_limits_for(criteria, external_cfg)
    assert external_limits.max_chars_per_file > limits.max_chars_per_file
    assert external_limits.budget("letter") > limits.budget("letter")


def test_document_limits_partial_override_and_defaults(monkeypatch):
    raw = _criteria_dict(
        document_limits={
            "external": {"per_file": 20000},
            "ollama": {"scoring": 9000},
        }
    )
    monkeypatch.setattr("cv_agent.config.load_yaml", lambda name: raw)
    monkeypatch.setattr("cv_agent.store.Store", _NoAccountStore)
    c = load_criteria()
    assert c.document_limits["external"] == DocumentLimits(
        max_chars_per_file=20000, keywords=24000, scoring=32000, letter=32000
    )
    assert c.document_limits["ollama"] == DocumentLimits(
        max_chars_per_file=6000, keywords=8000, scoring=9000, letter=12000
    )


@pytest.mark.parametrize(
    "section",
    [
        {"cloud": {}},  # неизвестный класс бэкенда
        {"external": {"unknown": 1}},  # неизвестное поле
        {"external": {"per_file": True}},  # bool вместо int
        {"external": {"per_file": "много"}},  # не int
        {"external": {"per_file": 0}},  # вне диапазона
        {"external": {"per_file": 10**7}},
        "сломано",
    ],
)
def test_document_limits_rejects_invalid_config(monkeypatch, section):
    raw = _criteria_dict(document_limits=section)
    monkeypatch.setattr("cv_agent.config.load_yaml", lambda name: raw)
    monkeypatch.setattr("cv_agent.store.Store", _NoAccountStore)
    with pytest.raises(ValueError, match="document_limits"):
        load_criteria()


def test_profile_loads():
    p = load_profile()
    assert p["name"] == "Ilia Rerich"
    assert p["skills"]
    assert p["experience"]


def test_criteria_uses_account_keywords(monkeypatch):
    """Ключевые слова берутся из учётной записи в БД, если она существует."""
    from cv_agent.config import load_criteria

    class FakeStore:
        def __init__(self, user=None):
            pass

        def account_keywords(self, name):
            if name == "Alice":
                return ["Alice-KW"]
            return None

        def candidate_profile(self, name):
            return {"search_region": ""} if name == "Alice" else None

        def close(self):
            pass

    monkeypatch.setattr("cv_agent.store.Store", FakeStore)
    c = load_criteria(user="Alice")
    assert c.keywords == ["Alice-KW"]
    c2 = load_criteria(user="nobody")
    assert c2.keywords  # фолбэк на criteria.yaml


def test_criteria_uses_current_candidate_search_region():
    from cv_agent.store import Store

    store = Store()
    store.create_account("Alice")
    store.set_candidate_profile(
        {"search_region": "Berlin", "preferred_letter_language": "de"},
        "Alice",
    )
    store.close()
    assert load_criteria(user="Alice").region == "Berlin"


def test_criteria_unknown_account_falls_back(monkeypatch):
    monkeypatch.delenv("CV_AGENT_USER", raising=False)
    c = load_criteria()
    assert c.keywords


def test_empty_profile_key_is_passed_explicitly_to_llm(monkeypatch):
    from cv_agent.cli import _llm_client_for

    class FakeStore:
        def account_gemini_key(self, name):
            return ""

        def close(self):
            pass

    captured = {}

    def fake_client(cfg, api_key=None):
        captured["model"] = cfg.model
        captured["api_key"] = api_key
        return object()

    monkeypatch.setattr("cv_agent.cli.Store", FakeStore)
    monkeypatch.setattr("cv_agent.cli.LLMClient", fake_client)
    monkeypatch.setattr("cv_agent.cli.pick_llm", lambda criteria: criteria.llm_fallback)
    monkeypatch.setenv("GEMINI_API_KEY", "global-secret")
    _llm_client_for(load_criteria())
    assert captured == {"model": "gemini-3.5-flash-lite", "api_key": ""}
