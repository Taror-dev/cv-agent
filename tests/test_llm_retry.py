import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest

from cv_agent.config import LLMConfig
from cv_agent.llm import LLMError, LLMClient, LLMRateLimiter, MAX_RATE_LIMIT_RETRIES


class FakeRequest:
    def __init__(self, url):
        self.url = url
        self.method = "POST"


class FakeResponse:
    def __init__(self, status_code=429):
        self.status_code = status_code
        self.request = FakeRequest("http://llm")
        self.headers = {}


def _rate_err(text: str):
    from openai import RateLimitError

    return RateLimitError(text, response=FakeResponse(), body=None)


@pytest.fixture(autouse=True)
def _no_lock(monkeypatch):
    """Не блокироваться на flock в тестах."""
    class FakeLock:
        def __init__(self, *args):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

    monkeypatch.setattr("cv_agent.llm.LLMLock", FakeLock)
    monkeypatch.setattr("cv_agent.llm.LLMBackendLock", FakeLock)


def _cfg():
    return LLMConfig(
        provider="gemini",
        model="gemini-3.5-flash-lite",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
        api_key_env="GEMINI_API_KEY",
        timeout=10.0,
    )


def test_rate_limit_waits_and_retries(monkeypatch):
    """429 с "retry in Xs" — ждём указанные секунды и повторяем автоматически."""
    calls = []
    waits = []

    def fake_create(*args, **kwargs):
        calls.append(1)
        if len(calls) < 3:
            raise _rate_err("Error code: 429 - Please retry in 2.5s.")
        return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": "ok"})()})()]})()

    monkeypatch.setattr("cv_agent.llm.time.sleep", lambda s: waits.append(s))
    client = LLMClient(_cfg())
    client.client.chat.completions.create = fake_create
    assert client._complete("s", "u") == "ok"
    assert len(calls) == 3
    assert waits == [2.5, 2.5]


def test_rate_limit_quota_message_retries(monkeypatch):
    """429 с "Quota exceeded ... Please retry in Xs" — тоже ждём и повторяем."""
    calls = []

    def fake_create(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise _rate_err(
                "Error code: 429 - You exceeded your current quota. "
                "Quota exceeded for metric: generate_content_free_tier_requests, "
                "limit: 500. Please retry in 1.5s."
            )
        return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": "ok"})()})()]})()

    monkeypatch.setattr("cv_agent.llm.time.sleep", lambda s: None)
    client = LLMClient(_cfg())
    client.client.chat.completions.create = fake_create
    assert client._complete("s", "u") == "ok"
    assert len(calls) == 2


def test_rate_limit_gives_up_after_max(monkeypatch):
    def fake_create(*args, **kwargs):
        raise _rate_err("Error code: 429 - Please retry in 1.0s.")

    monkeypatch.setattr("cv_agent.llm.time.sleep", lambda s: None)
    client = LLMClient(_cfg())
    client.client.chat.completions.create = fake_create
    with pytest.raises(LLMError):
        client._complete("s", "u")


def test_rate_limit_delay_capped():
    from cv_agent.llm import LLMClient

    client = LLMClient(_cfg())
    assert client._retry_delay(_rate_err("Please retry in 5000.0s.")) == 60.0


def test_retry_after_header_has_priority_and_is_capped():
    client = LLMClient(_cfg())
    error = _rate_err("Please retry in 1s.")
    error.response.headers["Retry-After"] = "75"
    assert client._retry_delay(error) == 60.0


def test_rate_and_regular_retry_counters_are_independent(monkeypatch):
    calls = []
    waits = []

    def fake_create(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise _rate_err("Please retry in 1s.")
        if len(calls) in {2, 3}:
            raise RuntimeError("temporary")
        return type("R", (), {"choices": [type("C", (), {"message": type("M", (), {"content": "ok"})()})()]})()

    monkeypatch.setattr("cv_agent.llm.time.sleep", lambda seconds: waits.append(seconds))
    client = LLMClient(_cfg())
    client.client.chat.completions.create = fake_create
    assert client._complete("s", "u") == "ok"
    assert len(calls) == 4
    assert waits == [1.0, 2, 4]


def test_external_completion_does_not_take_global_ollama_lock(monkeypatch):
    class ForbiddenLock:
        def __enter__(self):
            raise AssertionError("external provider must not take Ollama lock")

        def __exit__(self, *exc):
            pass

    monkeypatch.setattr("cv_agent.llm.LLMLock", ForbiddenLock)
    client = LLMClient(_cfg())
    client.client.chat.completions.create = lambda **kwargs: type(
        "R",
        (),
        {"choices": [type("C", (), {"message": type("M", (), {"content": "ok"})()})()]},
    )()
    assert client.complete("s", "u") == "ok"


def test_external_backend_slot_is_released_before_retry_sleep(monkeypatch):
    active = []
    calls = []

    class TrackingBackendLock:
        def __init__(self, key):
            assert key

        def __enter__(self):
            active.append(True)

        def __exit__(self, *exc):
            active.pop()

    def create(**kwargs):
        assert active == [True]
        calls.append(1)
        if len(calls) == 1:
            raise _rate_err("Please retry in 1s.")
        return type(
            "R",
            (),
            {
                "choices": [
                    type(
                        "C",
                        (),
                        {"message": type("M", (), {"content": "ok"})()},
                    )()
                ]
            },
        )()

    monkeypatch.setattr("cv_agent.llm.LLMBackendLock", TrackingBackendLock)
    monkeypatch.setattr(
        "cv_agent.llm.time.sleep",
        lambda seconds: (active == [] or pytest.fail("slot held during sleep")),
    )
    client = LLMClient(_cfg())
    client.client.chat.completions.create = create
    assert client.complete("s", "u") == "ok"
    assert len(calls) == 2


def test_ollama_completion_uses_shared_lock(monkeypatch):
    entered = []

    class TrackingLock:
        def __enter__(self):
            entered.append(True)

        def __exit__(self, *exc):
            pass

    cfg = _cfg()
    cfg.provider = "ollama"
    cfg.base_url = "http://localhost:11434/v1"
    monkeypatch.setattr("cv_agent.llm.LLMLock", TrackingLock)
    client = LLMClient(cfg)
    client.client.chat.completions.create = lambda **kwargs: type(
        "R",
        (),
        {"choices": [type("C", (), {"message": type("M", (), {"content": "ok"})()})()]},
    )()
    assert client.complete("s", "u") == "ok"
    assert entered == [True]


def test_rate_limiter_exposes_waiting_job_state(monkeypatch, tmp_path):
    from cv_agent.store import Store

    path = tmp_path / "rate.db"
    store = Store(path)
    run_id, _ = store.create_job(
        "pipeline",
        [{"cmd": "score", "args": ["score"], "code": None, "output": ""}],
    )
    assert store.claim_job(run_id)
    cfg = _cfg()
    limiter = LLMRateLimiter(cfg, "Илья", run_id)
    monkeypatch.setattr(
        limiter,
        "_store",
        lambda: Store(path, user="Илья"),
    )
    limiter_store = Store(path)
    limiter_store.defer_llm_rate_limit(limiter.key, 1)
    limiter_store.close()
    observed = []

    def fake_sleep(seconds):
        observer = Store(path)
        observed.append((seconds, observer.job(run_id)["status"]))
        observer.close()

    monkeypatch.setattr("cv_agent.llm.time.sleep", fake_sleep)
    limiter.wait()
    assert observed[0][1] == "waiting_rate_limit"
    assert store.job(run_id)["status"] == "running"
    assert store.llm_rate_limit_delay(limiter.key) == 0
    store.close()


def test_invalid_external_endpoint_is_rejected_before_client_creation():
    cfg = LLMConfig(
        provider="gemini",
        model="gemini-test",
        base_url="https://169.254.169.254/v1",
        api_key_env="GEMINI_API_KEY",
    )
    with pytest.raises(LLMError, match="Недопустимый адрес"):
        LLMClient(cfg, api_key="AIza-SECRETSECRET")


def test_llm_error_redacts_explicit_api_key(monkeypatch):
    secret = "AIza-THIS-IS-A-PRIVATE-KEY"
    client = LLMClient(_cfg(), api_key=secret)

    def fail(*args, **kwargs):
        raise RuntimeError(f"request failed with api_key={secret}")

    monkeypatch.setattr("cv_agent.llm.time.sleep", lambda _seconds: None)
    client.client.chat.completions.create = fail
    with pytest.raises(LLMError) as exc:
        client._complete("s", "u")
    assert secret not in str(exc.value)
    assert "[REDACTED]" in str(exc.value)


def test_complete_json_sends_schema_for_supported_provider():
    captured = {}

    def fake_create(**kwargs):
        captured.update(kwargs)
        return type(
            "R",
            (),
            {
                "choices": [
                    type(
                        "C",
                        (),
                        {"message": type("M", (), {"content": '{"ok":true}'})()},
                    )()
                ]
            },
        )()

    client = LLMClient(_cfg())
    client.client.chat.completions.create = fake_create
    schema = {
        "type": "object",
        "required": ["ok"],
        "properties": {"ok": {"type": "boolean"}},
    }
    assert client.complete_json("s", "u", schema_name="result", schema=schema) == '{"ok":true}'
    assert captured["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "result", "strict": True, "schema": schema},
    }


def test_quota_headers_captured_to_store(monkeypatch):
    """x-ratelimit-заголовки и usage последнего вызова сохраняются в снимок."""
    from cv_agent.llm import provider_key
    from cv_agent.store import Store

    client = LLMClient(_cfg(), api_key="AIza-CAPTURE", user="Илья")

    class QuotaResponse:
        """Мимикрирует под настоящий httpx-ответ: json() требует read()."""

        headers = {
            "X-RateLimit-Limit-Tokens": "12000",
            "x-ratelimit-remaining-tokens": "11000",
            "content-type": "application/json",
        }
        _payload = {
            "usage": {
                "prompt_tokens": 500,
                "completion_tokens": 100,
                "total_tokens": 600,
            }
        }

        def read(self):
            self._read = True
            return b"{}"

        def json(self):
            if not getattr(self, "_read", False):
                raise RuntimeError("ResponseNotRead")
            return self._payload

    client._capture_quota(QuotaResponse())

    store = Store(user="Илья")
    try:
        snapshot = store.llm_quota(provider_key(_cfg()))
    finally:
        store.close()
    assert snapshot is not None
    headers, stamp = snapshot
    assert headers["x-ratelimit-limit-tokens"] == "12000"
    assert headers["x-ratelimit-remaining-tokens"] == "11000"
    assert headers["_usage"] == {
        "prompt_tokens": 500,
        "completion_tokens": 100,
        "total_tokens": 600,
    }
    assert stamp


def test_quota_capture_without_ratelimit_headers_is_noop():
    """Ответ без x-ratelimit-* не создаёт запись и не ломает поток."""
    from cv_agent.llm import provider_key
    from cv_agent.store import Store

    client = LLMClient(_cfg(), api_key="AIza-NOP")

    class PlainResponse:
        headers = {"content-type": "application/json"}

    client._capture_quota(PlainResponse())

    store = Store(user="Илья")
    try:
        assert store.llm_quota(provider_key(_cfg())) is None
    finally:
        store.close()
