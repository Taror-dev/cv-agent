import hashlib
import os
import re
import sys
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from openai import OpenAI, RateLimitError

try:  # openai>=3 зависит от форка httpx2 с тем же API
    import httpx
except ImportError:  # pragma: no cover - зависит от окружения
    import httpx2 as httpx  # type: ignore[no-redef]

from .config import LLMConfig, is_ollama_url, llm_endpoint_allowed
from .resource_limits import llm_backend_limit
from .security import redact_sensitive

MAX_RATE_LIMIT_RETRIES = 15
MAX_REGULAR_ATTEMPTS = 3
MAX_RETRY_DELAY = 60.0
STRUCTURED_OUTPUT_PROVIDERS = frozenset({"openai", "gemini", "ollama", "groq"})


class LLMLock:
    """Kompatibler Name für den gemeinsamen Ollama-Backend-Pool."""

    def __init__(self):
        self._lease = llm_backend_limit("ollama-shared")
        self.measurement = None

    def __enter__(self):
        self.measurement = self._lease.__enter__()
        return self

    def __exit__(self, *exc):
        return self._lease.__exit__(*exc)


class LLMBackendLock:
    """Separater cross-process Pool pro externem provider/base_url/model."""

    def __init__(self, backend_key: str):
        self._lease = llm_backend_limit(backend_key)
        self.measurement = None

    def __enter__(self):
        self.measurement = self._lease.__enter__()
        return self

    def __exit__(self, *exc):
        return self._lease.__exit__(*exc)


class LLMError(RuntimeError):
    """LLM-запрос не удался после всех попыток."""


def provider_key(cfg: LLMConfig) -> str:
    identity = "|".join(
        (cfg.provider.strip().lower(), cfg.base_url.rstrip("/").lower(), cfg.model)
    )
    return hashlib.sha256(identity.encode("utf-8")).hexdigest()


class LLMRateLimiter:
    """Persistenter 429-Cooldown für ein Konto und einen externen Provider."""

    def __init__(
        self,
        cfg: LLMConfig,
        user: str,
        run_id: int | None = None,
    ):
        self.key = provider_key(cfg)
        self.label = f"{cfg.provider} — {cfg.model}"
        self.user = user
        self.run_id = run_id

    def _store(self):
        from .store import Store

        return Store(user=self.user)

    def wait(self) -> None:
        store = self._store()
        try:
            delay = min(store.llm_rate_limit_delay(self.key), MAX_RETRY_DELAY)
            if delay <= 0:
                store.clear_llm_rate_limit(self.key)
                return
            if self.run_id is not None:
                store.set_job_rate_limit_wait(self.run_id, True)
        finally:
            store.close()
        print(
            f"Rate-Limit ({self.label}): warte {delay:.0f}s",
            file=sys.stderr,
            flush=True,
        )
        try:
            time.sleep(delay)
        finally:
            store = self._store()
            try:
                store.clear_llm_rate_limit(self.key)
                if self.run_id is not None:
                    store.set_job_rate_limit_wait(self.run_id, False)
            finally:
                store.close()

    def defer_and_wait(self, delay: float) -> None:
        store = self._store()
        try:
            store.defer_llm_rate_limit(self.key, min(delay, MAX_RETRY_DELAY))
        finally:
            store.close()
        self.wait()


class LLMClient:
    def __init__(
        self,
        cfg: LLMConfig,
        api_key: str | None = None,
        rate_limiter: LLMRateLimiter | None = None,
        user: str | None = None,
    ):
        self.cfg = cfg
        self.rate_limiter = rate_limiter
        self.user = user
        if not llm_endpoint_allowed(cfg):
            raise LLMError("Недопустимый адрес LLM в серверной конфигурации")
        if api_key is None:
            api_key = os.getenv(cfg.api_key_env, "ollama")
        self._secrets = (api_key,) if api_key and api_key != "ollama" else ()
        http_client = None
        if not is_ollama_url(cfg.base_url):
            http_client = httpx.Client(
                timeout=cfg.timeout,
                event_hooks={"response": [self._capture_quota]},
            )
        self.client = OpenAI(
            base_url=cfg.base_url,
            api_key=api_key or "ollama",
            timeout=cfg.timeout,
            max_retries=0,
            http_client=http_client,
        )

    def _capture_quota(self, response) -> None:
        """Сохраняет x-ratelimit-заголовки и usage последнего вызова провайдера.

        Снимок используется на странице профиля для показа лимитов и расхода.
        Ошибки записи никогда не ломают сам LLM-вызов.
        """
        try:
            headers = {
                str(name).lower(): str(value)
                for name, value in response.headers.items()
                if str(name).lower().startswith("x-ratelimit-")
            }
            try:
                response.read()  # хук срабатывает до чтения тела клиентом SDK
                data = response.json()
                raw_usage = data.get("usage") if isinstance(data, dict) else None
                usage = (
                    {
                        key: int(raw_usage[key])
                        for key in (
                            "prompt_tokens",
                            "completion_tokens",
                            "total_tokens",
                        )
                        if isinstance(raw_usage.get(key), int)
                    }
                    if isinstance(raw_usage, dict)
                    else {}
                )
            except Exception:  # noqa: BLE001 — тело ответа опционально
                usage = {}
            if not headers and not usage:
                return
            snapshot = dict(headers)
            if usage:
                snapshot["_usage"] = usage
            from .store import Store

            store = Store(user=self.user)
            try:
                store.save_llm_quota(provider_key(self.cfg), snapshot)
            finally:
                store.close()
        except Exception:  # noqa: BLE001 — телеметрия не должна ронять запрос
            pass

    def complete(
        self,
        system: str,
        user: str,
        *,
        response_format: dict | None = None,
    ) -> str:
        return self._complete(system, user, response_format=response_format)

    def _request_limit(self):
        if is_ollama_url(self.cfg.base_url):
            return LLMLock()
        return LLMBackendLock(provider_key(self.cfg))

    def complete_json(
        self,
        system: str,
        user: str,
        *,
        schema_name: str,
        schema: dict,
    ) -> str:
        """Fordert JSON Schema nur bei bekannten Provider-Capabilities an."""
        response_format = None
        if self.cfg.provider.strip().lower() in STRUCTURED_OUTPUT_PROVIDERS:
            response_format = {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name,
                    "strict": True,
                    "schema": schema,
                },
            }
        return self.complete(system, user, response_format=response_format)

    def _complete(
        self,
        system: str,
        user: str,
        *,
        response_format: dict | None = None,
    ) -> str:
        last_err: Exception | None = None
        rate_retries = 0
        regular_failures = 0
        request_count = 0
        while True:
            if self.rate_limiter is not None:
                self.rate_limiter.wait()
            request_count += 1
            try:
                request = {
                    "model": self.cfg.model,
                    "temperature": self.cfg.temperature,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                }
                if response_format is not None:
                    request["response_format"] = response_format
                with self._request_limit():
                    resp = self.client.chat.completions.create(
                        **request,
                    )
                return resp.choices[0].message.content or ""
            except RateLimitError as e:
                last_err = e
                rate_retries += 1
                if rate_retries > MAX_RATE_LIMIT_RETRIES:
                    break
                delay = self._retry_delay(e)
                if self.rate_limiter is not None:
                    self.rate_limiter.defer_and_wait(delay)
                else:
                    print(
                        f"Rate limit ({self.cfg.model}): waiting {delay:.0f}s, "
                        f"retrying {rate_retries}/{MAX_RATE_LIMIT_RETRIES}",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(delay)
            except Exception as e:  # noqa: BLE001 — сетевые/сервисные ошибки, ещё попытка
                last_err = e
                regular_failures += 1
                if regular_failures >= MAX_REGULAR_ATTEMPTS:
                    break
                time.sleep(2 * regular_failures)
        raise LLMError(
            f"LLM request failed after {request_count} attempts "
            f"(regular errors: {regular_failures}, 429s: {rate_retries}; "
            f"{self.cfg.model}): {redact_sensitive(last_err, self._secrets)}"
        )

    def _retry_delay(self, err: RateLimitError) -> float:
        headers = getattr(getattr(err, "response", None), "headers", {}) or {}
        retry_after = headers.get("Retry-After") or headers.get("retry-after")
        parsed = self._parse_retry_value(retry_after)
        if parsed is not None:
            return parsed
        for name in ("x-ratelimit-reset-after", "x-ratelimit-reset"):
            parsed = self._parse_retry_value(headers.get(name), absolute=name.endswith("reset"))
            if parsed is not None:
                return parsed
        m = re.search(r"retry in ([\d.]+)s", str(err), re.I)
        if m:
            return min(float(m.group(1)), MAX_RETRY_DELAY)
        m = re.search(r"retryDelay.*?([\d.]+)s", str(err), re.S | re.I)
        return min(float(m.group(1)), MAX_RETRY_DELAY) if m else 10.0

    @staticmethod
    def _parse_retry_value(value, *, absolute: bool = False) -> float | None:
        if value is None:
            return None
        raw = str(value).strip()
        try:
            number = float(raw)
            if absolute and number > time.time():
                number -= time.time()
            return min(max(number, 0.0), MAX_RETRY_DELAY)
        except ValueError:
            pass
        try:
            target = parsedate_to_datetime(raw)
            if target.tzinfo is None:
                target = target.replace(tzinfo=timezone.utc)
            seconds = (target - datetime.now(timezone.utc)).total_seconds()
            return min(max(seconds, 0.0), MAX_RETRY_DELAY)
        except (TypeError, ValueError, OverflowError):
            return None
