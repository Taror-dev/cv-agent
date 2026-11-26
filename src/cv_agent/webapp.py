import hashlib
import io
import json
import os
import queue
import re
import secrets
import signal
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
from pathlib import Path

import requests
from itsdangerous import BadSignature, URLSafeTimedSerializer
from flask import (
    Flask,
    abort,
    g,
    has_request_context,
    make_response,
    redirect,
    render_template,
    request,
    send_file,
    session,
    url_for,
)
from pypdf import PdfReader

from . import config as _config
from .cli import dotenv_overrides, load_env
from .config import (
    APPLICATIONS_DIR,
    GEMINI_BASE_URL,
    GEMINI_NATIVE_MODELS_URL,
    GROQ_BASE_URL,
    GROQ_MODEL,
    current_user,
    is_ollama_url,
    is_recruiting_agency,
    llm_endpoint_allowed,
    load_criteria,
    ollama_host,
    ollama_openai_url,
)
from .documents import (
    DOCUMENT_TYPES,
    SUPPORTED_DOCUMENT_EXTENSIONS,
    cv_dir,
    load_document_texts,
    scan_documents,
)
from .llm import LLMClient, LLMRateLimiter, provider_key
from .paths import (
    normalize_account_name,
    safe_filename,
    valid_account_name,
)
from .search import (
    VacancyUnavailable,
    ensure_description,
    fetch_description,
)
from .resource_limits import network_request_limit
from .security import redact_sensitive
from .storage import applications_dir, export_account_archive, remove_account_storage
from .write import safe_slug
from .store import DEFAULT_USER, Store
from .vacancy_feedback import GRADE_LABELS, candidate_revision, capture_inputs, vacancy_revision

app = Flask(__name__)
app.config.update(
    MAX_CONTENT_LENGTH=25 * 1024 * 1024,
    SECRET_KEY=os.environ.get("CV_AGENT_SESSION_KEY") or secrets.token_hex(32),
    SESSION_COOKIE_NAME="cv_agent_session",
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Strict",
    SESSION_COOKIE_SECURE=os.environ.get("CV_AGENT_SESSION_SECURE", "").lower()
    in {"1", "true", "yes"},
    PERMANENT_SESSION_LIFETIME=2592000,
)
AGENCY_LABEL = "рекрутинговое агентство"
LLM_COOKIE = "llm"
LEGACY_USER_COOKIE = "user"
WORKER_ENV_KEYS = frozenset(
    {
        "CV_AGENT_USER",
        "LLM_API_KEY_ENV",
        "LLM_BASE_URL",
        "LLM_MODEL",
        "LLM_PROVIDER",
    }
)


def _select_account(store: Store, account: str) -> bool:
    """Сохраняет в подписанной Flask session только внутренний UUID профиля."""
    account_id = store.account_id(account)
    if not account_id:
        return False
    session.clear()
    session["account_id"] = account_id
    session.permanent = True
    g.current_user = account
    return True


def _allowed_llm_id(value: str) -> str:
    selected = urllib.parse.unquote(value or "")
    return selected if any(o["id"] == selected for o in available_llms()) else ""


def _current_user() -> str:
    """Имя профиля, найденное по UUID из подписанной Flask session."""
    if "current_user" in g:
        return g.current_user
    account_id = session.get("account_id")
    if not account_id:
        return ""
    store = Store()
    try:
        name = store.account_name(account_id) or ""
    finally:
        store.close()
    g.current_user = name
    return name


def _require_account(name: str) -> bool:
    return valid_account_name(name)


def available_llms() -> list[dict]:
    """Модели для выпадающего списка: установленные в Ollama + дефолт из criteria.yaml."""
    criteria = load_criteria()
    default = criteria.llm
    options = []
    try:
        proc = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10
        )
        for line in proc.stdout.splitlines()[1:]:
            parts = line.split()
            if parts:
                embedding_family = getattr(getattr(getattr(criteria, "scoring", None), "embeddings", None), "model", "qwen3-embedding:0.6b").split(":", 1)[0]
                if parts[0].split(":", 1)[0] in {embedding_family, "qwen3-embedding", "bge-m3"}:
                    continue
                options.append(
                    {
                        "id": f"ollama:{parts[0]}",
                        "provider": "ollama",
                        "model": parts[0],
                        "base_url": ollama_openai_url(),
                        "api_key_env": "OLLAMA_API_KEY",
                        "label": f"Ollama — {parts[0]}",
                    }
                )
    except Exception:
        pass
    if llm_endpoint_allowed(default) and not any(
        o["model"] == default.model for o in options
    ):
        default_base = (
            ollama_openai_url() if "11434" in default.base_url else default.base_url
        )
        options.insert(
            0,
            {
                "id": f"configured:{default.provider}:{default.model}",
                "provider": default.provider,
                "model": default.model,
                "base_url": default_base,
                "api_key_env": default.api_key_env,
                "label": f"{default.provider} — {default.model} (из criteria.yaml)",
            },
        )
    fallback = criteria.llm_fallback
    if fallback and llm_endpoint_allowed(fallback) and not any(
        o["model"] == fallback.model and o["base_url"] == fallback.base_url
        for o in options
    ):
        options.append(
            {
                "id": f"fallback:{fallback.provider}:{fallback.model}",
                "provider": fallback.provider,
                "model": fallback.model,
                "base_url": fallback.base_url,
                "api_key_env": fallback.api_key_env,
                "label": f"{fallback.provider} — {fallback.model}",
            }
        )
    for option in _groq_model_options(_account_llm_key("GROQ_API_KEY")):
        if not any(o["id"] == option["id"] for o in options):
            options.append(option)
    for option in _gemini_model_options(_account_llm_key("GEMINI_API_KEY")):
        if not any(o["id"] == option["id"] for o in options):
            options.append(option)
    return options


GROQ_MODELS_TTL_SECONDS = 600
_groq_models_cache: dict[
    str, tuple[float, dict[str, dict], dict[str, dict]]
] = {}
GEMINI_MODELS_TTL_SECONDS = 600
_gemini_models_cache: dict[str, tuple[float, dict[str, dict]]] = {}


def _groq_option(model: str) -> dict:
    return {
        "id": f"groq:{model}",
        "provider": "groq",
        "model": model,
        "base_url": GROQ_BASE_URL,
        "api_key_env": "GROQ_API_KEY",
        "label": f"Groq — {model}",
    }


def _gemini_option(model: str) -> dict:
    return {
        "id": f"gemini:{model}",
        "provider": "gemini",
        "model": model,
        "base_url": GEMINI_BASE_URL,
        "api_key_env": "GEMINI_API_KEY",
        "label": f"Gemini — {model}",
    }


def _parse_groq_rate_headers(headers) -> dict[str, dict]:
    """Квоты из заголовков ответа Groq: RPD (запросы/день) и TPM (токены/мин)."""
    def _int(name: str):
        raw = headers.get(name)
        if raw is None:
            return None
        try:
            return int(str(raw).strip())
        except ValueError:
            return None

    rate: dict[str, dict] = {}
    for key, limit_h, remaining_h in (
        ("requests_per_day", "x-ratelimit-limit-requests", "x-ratelimit-remaining-requests"),
        ("tokens_per_minute", "x-ratelimit-limit-tokens", "x-ratelimit-remaining-tokens"),
    ):
        limit = _int(limit_h)
        remaining = _int(remaining_h)
        if limit is not None or remaining is not None:
            rate[key] = {"limit": limit, "remaining": remaining}
    return rate


def _fetch_groq_models(api_key: str) -> tuple[dict[str, dict], dict[str, dict]]:
    """Актуальные модели Groq с лимитами по ключу аккаунта (/models)."""
    try:
        with network_request_limit():
            resp = requests.get(
                GROQ_BASE_URL.rstrip("/") + "/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=5,
            )
        if resp.status_code != 200:
            return {}, {}
        data = resp.json().get("data", [])
    except (requests.RequestException, ValueError):
        return {}, {}
    models: dict[str, dict] = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id", "")).strip()
        if not model_id or not item.get("active", True):
            continue
        models[model_id] = {
            "context_window": item.get("context_window"),
            "max_completion_tokens": item.get("max_completion_tokens"),
        }
    rate = _parse_groq_rate_headers(getattr(resp, "headers", None) or {})
    return dict(sorted(models.items())), rate


def _groq_cache_entry(api_key: str) -> tuple[dict[str, dict], dict[str, dict]]:
    """(модели, квоты) из кэша или свежего запроса; устаревший кэш лучше пустого."""
    if not api_key:
        return {}, {}
    key_digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    now = time.monotonic()
    cached = _groq_models_cache.get(key_digest)
    if cached and now - cached[0] < GROQ_MODELS_TTL_SECONDS:
        return cached[1], cached[2]
    models, rate = _fetch_groq_models(api_key)
    if models or rate:
        _groq_models_cache[key_digest] = (now, models, rate)
        return models, rate
    if cached:
        return cached[1], cached[2]
    return {}, {}


def _groq_models_map(api_key: str) -> dict[str, dict]:
    return _groq_cache_entry(api_key)[0]


def _groq_rate_map(api_key: str) -> dict[str, dict]:
    return _groq_cache_entry(api_key)[1]


def _groq_model_options(api_key: str) -> list[dict]:
    """Опции Groq: живой список моделей или модель по умолчанию как фолбэк."""
    models = _groq_models_map(api_key)
    if not models:
        return [_groq_option(GROQ_MODEL)]
    return [_groq_option(model) for model in models]


def _fetch_gemini_models(api_key: str) -> dict[str, dict]:
    """Актуальные модели Gemini с лимитами токенов (нативный /v1beta/models).

    Оставляет только модели, поддерживающие generateContent.
    """
    try:
        with network_request_limit():
            resp = requests.get(
                GEMINI_NATIVE_MODELS_URL,
                params={"key": api_key},
                timeout=5,
            )
        if resp.status_code != 200:
            raise ValueError(resp.status_code)
        items = resp.json().get("models", [])
    except (requests.RequestException, ValueError, AttributeError):
        return {}
    models: dict[str, dict] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).removeprefix("models/").strip()
        methods = item.get("supportedGenerationMethods")
        if not name or (methods and "generateContent" not in methods):
            continue
        models[name] = {
            "input": item.get("inputTokenLimit"),
            "output": item.get("outputTokenLimit"),
        }
    return dict(sorted(models.items()))


def _gemini_cache_entry(api_key: str) -> dict[str, dict]:
    """Модели Gemini из кэша или свежего запроса; устаревший кэш лучше пустого."""
    if not api_key:
        return {}
    key_digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()
    now = time.monotonic()
    cached = _gemini_models_cache.get(key_digest)
    if cached and now - cached[0] < GEMINI_MODELS_TTL_SECONDS:
        return cached[1]
    models = _fetch_gemini_models(api_key)
    if models:
        _gemini_models_cache[key_digest] = (now, models)
        return models
    return cached[1] if cached else {}


def _gemini_limits_map(api_key: str) -> dict[str, dict]:
    """Лимиты токенов моделей Gemini (нативный /v1beta/models), кэш 10 минут."""
    return _gemini_cache_entry(api_key)


def _is_free_tier_gemini_model(name: str) -> bool:
    """Семейства Flash/Flash-Lite — единственные с бесплатным тарифом Gemini."""
    return "flash" in name.lower()


def _gemini_model_options(api_key: str) -> list[dict]:
    """Опции Gemini: только бесплатные семейства Flash/Flash-Lite.

    Без ключа остаётся фолбэк из criteria.yaml.
    """
    return [
        _gemini_option(model)
        for model in _gemini_cache_entry(api_key)
        if _is_free_tier_gemini_model(model)
    ]


def _format_token_limit(value) -> str:
    if isinstance(value, bool) or not isinstance(value, int):
        return "?"
    return f"{value:,}".replace(",", " ")


def _quota_values(entry: dict) -> str | None:
    """«X из Y (потрачено Z)» для квоты; None если данных нет."""
    limit = entry.get("limit")
    remaining = entry.get("remaining")
    if limit is None and remaining is None:
        return None
    if remaining is None:
        return f"лимит {_format_token_limit(limit)}"
    if limit is None:
        return _format_token_limit(remaining)
    spent = max(limit - remaining, 0)
    return (
        f"{_format_token_limit(remaining)} из "
        f"{_format_token_limit(limit)} (потрачено {_format_token_limit(spent)})"
    )


def _usage_values(cap, used) -> str | None:
    """Расход последнего запроса против лимита модели; None без лимита."""
    if cap is None:
        return None
    if not isinstance(used, int):
        return f"лимит {_format_token_limit(cap)}"
    used = max(used, 0)
    remaining = max(int(cap) - used, 0)
    return (
        f"{_format_token_limit(remaining)} из {_format_token_limit(cap)} "
        f"(потрачено {_format_token_limit(used)})"
    )


def _quota_header_value(value) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def _db_quota(option: dict) -> tuple[dict[str, dict], dict | None, str | None]:
    """Последний снимок внешнего провайдера с реальных вызовов: (квоты, usage, время)."""
    backend = provider_key(
        type(
            "Cfg",
            (),
            {
                "provider": option["provider"],
                "base_url": option["base_url"],
                "model": option["model"],
            },
        )()
    )
    store = Store()
    try:
        snapshot = store.llm_quota(backend)
    finally:
        store.close()
    if not snapshot:
        return {}, None, None
    headers, updated = snapshot
    rate: dict[str, dict] = {}
    for name, value in headers.items():
        lname = str(name).lower()
        for key, suffix in (
            ("requests_per_day", "-requests"),
            ("tokens_per_minute", "-tokens"),
        ):
            if lname == f"x-ratelimit-limit{suffix}":
                rate.setdefault(key, {})["limit"] = _quota_header_value(value)
            elif lname == f"x-ratelimit-remaining{suffix}":
                rate.setdefault(key, {})["remaining"] = _quota_header_value(value)
    raw_usage = headers.get("_usage")
    usage = raw_usage if isinstance(raw_usage, dict) else None
    stamp = str(updated)[:16] if updated else None
    return rate, usage, stamp


def _limits_rows(option: dict | None) -> tuple[list[dict], str | None]:
    """Строки блока «Лимиты» для текущей модели; [] если показать нечего."""
    if not option or option["provider"] == "ollama":
        return [], None
    if option["provider"] == "groq":
        api_key = _account_llm_key("GROQ_API_KEY")
        info = _groq_models_map(api_key).get(option["model"]) or {}
        db_rate, usage, stamp = _db_quota(option)
        live_rate = _groq_rate_map(api_key)
        rows = []
        context_row = _usage_values(info.get("context_window"), (usage or {}).get("prompt_tokens"))
        if context_row:
            rows.append({"label": "Контекст (токенов)", "value": context_row})
        completion_row = _usage_values(
            info.get("max_completion_tokens"), (usage or {}).get("completion_tokens")
        )
        if completion_row:
            rows.append({"label": "Макс. токенов ответ", "value": completion_row})
        for key, label in (
            ("requests_per_day", "Запросов в день"),
            ("tokens_per_minute", "Токенов в минуту"),
        ):
            entry = db_rate.get(key) or live_rate.get(key)
            value = _quota_values(entry) if entry else None
            if value:
                rows.append({"label": label, "value": value})
        return rows, stamp if rows else None
    if option["provider"] == "gemini":
        info = _gemini_limits_map(_account_llm_key("GEMINI_API_KEY")).get(
            option["model"]
        )
        _rate, usage, stamp = _db_quota(option)
        rows = []
        context_row = _usage_values(
            (info or {}).get("input"), (usage or {}).get("prompt_tokens")
        )
        if context_row:
            rows.append({"label": "Контекст (токенов)", "value": context_row})
        completion_row = _usage_values(
            (info or {}).get("output"), (usage or {}).get("completion_tokens")
        )
        if completion_row:
            rows.append({"label": "Макс. токенов ответ", "value": completion_row})
        # Gemini не отдаёт квоты RPD/TPM ни в заголовках, ни через API.
        rows.append({"label": "Запросов в день", "value": "—"})
        rows.append({"label": "Токенов в минуту", "value": "—"})
        return rows, stamp if rows else None
    return [], None


def _current_llm() -> tuple[str, dict | None]:
    """Метка и опция allowlist для модели, которая сейчас используется."""
    override = _llm_override()
    options = available_llms()
    if override:
        for o in options:
            if (
                o["model"] == override["LLM_MODEL"]
                and o["base_url"] == override["LLM_BASE_URL"]
            ):
                return o["label"], o
        return f"{override.get('LLM_PROVIDER', 'llm')} — {override['LLM_MODEL']}", None
    cfg = load_criteria().llm
    for o in options:
        if o["model"] == cfg.model and o["provider"] == cfg.provider:
            return o["label"], o
    return f"{cfg.provider} — {cfg.model}", None


LLM_PROVIDER_ORDER = ("ollama", "gemini", "groq")


def _llm_groups() -> dict[str, list[dict]]:
    """Опции моделей по провайдерам для связанных выпадающих списков.

    Динамические списки моделей (groq:/gemini:) замещают статические записи
    той же модели, чтобы провайдер не показывал дубликаты.
    """
    groups: dict[str, list[dict]] = {name: [] for name in LLM_PROVIDER_ORDER}
    static: dict[str, dict[str, dict]] = {
        name: {} for name in LLM_PROVIDER_ORDER
    }
    group_models: dict[str, set[str]] = {name: set() for name in LLM_PROVIDER_ORDER}
    seen: set[str] = set()
    for option in available_llms():
        provider = option["provider"]
        if provider not in groups or option["id"] in seen:
            continue
        seen.add(option["id"])
        label = option["model"]
        if "из criteria.yaml" in option["label"]:
            label += " (из criteria.yaml)"
        entry = {"id": option["id"], "label": label}
        if option["id"].startswith(f"{provider}:"):
            static[provider].pop(option["model"], None)
            groups[provider].append(entry)
            group_models[provider].add(option["model"])
        else:
            static[provider][option["model"]] = entry
    for provider, by_model in static.items():
        groups[provider].extend(
            entry
            for model, entry in by_model.items()
            if model not in group_models[provider]
        )
    return groups


def _llm_override() -> dict:
    """Конфигурация выбранной модели только из серверного allowlist.

    Приоритет: cookie запуска → сохранённый в профиле выбор модели.
    """
    raw = request.cookies.get(LLM_COOKIE, "")
    selected_id = urllib.parse.unquote(raw) if raw else _account_preferred_llm()
    for option in available_llms():
        if option["id"] == selected_id:
            return {
                "LLM_MODEL": option["model"],
                "LLM_PROVIDER": option["provider"],
                "LLM_BASE_URL": option["base_url"],
                "LLM_API_KEY_ENV": option["api_key_env"],
            }
    return {}


def _ollama_alive(timeout: float = 2.0) -> bool:
    try:
        with network_request_limit():
            return (
                requests.get(
                    ollama_host() + "/api/tags", timeout=timeout
                ).status_code
                == 200
            )
    except requests.RequestException:
        return False


ACCOUNT_KEY_ENVS = frozenset({"GEMINI_API_KEY", "GROQ_API_KEY"})


def _account_llm_key(api_key_env: str) -> str:
    """Ключ внешнего провайдера текущей учётной записи (Ollama общая)."""
    if not has_request_context() or api_key_env not in ACCOUNT_KEY_ENVS:
        return ""
    user = _current_user()
    if not user:
        return ""
    store = Store(user=user)
    try:
        return store.account_llm_key(api_key_env)
    finally:
        store.close()


def _account_preferred_llm() -> str:
    """Сохранённый в профиле ID модели (серверный allowlist при валидации)."""
    user = _current_user()
    if not user:
        return ""
    store = Store(user=user)
    try:
        return store.account_preferred_llm()
    finally:
        store.close()


def _effective_llm_env() -> tuple[dict, str]:
    """Env-переопределения для запуска + заметка. Если Ollama недоступна — фолбэк на Gemini."""
    criteria = load_criteria()
    override = _llm_override()
    base = override.get("LLM_BASE_URL") or criteria.llm.base_url
    ollama_urls = {ollama_openai_url().rstrip("/"), "http://localhost:11434/v1"}
    note = ""
    if base.rstrip("/") in ollama_urls and not _ollama_alive():
        fb = criteria.llm_fallback
        if fb:
            override = {
                "LLM_PROVIDER": fb.provider,
                "LLM_MODEL": fb.model,
                "LLM_BASE_URL": fb.base_url,
                "LLM_API_KEY_ENV": fb.api_key_env,
            }
            note = f"Ollama недоступна — используется {fb.provider} — {fb.model}"
    env = dict(override)
    effective_key_env = env.get("LLM_API_KEY_ENV", criteria.llm.api_key_env)
    key = _account_llm_key(effective_key_env)
    if key:
        env[effective_key_env] = key
    return env, note


def _subprocess_llm_env() -> tuple[dict, str]:
    """LLM-настройки worker без API key: CLI загружает ключ профиля из SQLite."""
    env, note = _effective_llm_env()
    for key_env in ACCOUNT_KEY_ENVS:
        env.pop(key_env, None)
    return env, note


def _llm_client(override: dict) -> LLMClient:
    cfg = load_criteria().llm
    llm_keys = {"LLM_MODEL", "LLM_BASE_URL", "LLM_API_KEY_ENV"}
    if llm_keys.issubset(override):
        cfg.provider = override.get("LLM_PROVIDER", cfg.provider)
        cfg.model = override["LLM_MODEL"]
        cfg.base_url = override["LLM_BASE_URL"]
        cfg.api_key_env = override["LLM_API_KEY_ENV"]
    api_key = override.get(cfg.api_key_env)
    if cfg.api_key_env in ACCOUNT_KEY_ENVS:
        if api_key is None and has_request_context():
            api_key = _account_llm_key(cfg.api_key_env)
        # Пустая строка запрещает LLMClient брать общий ключ из окружения.
        api_key = api_key or ""
    client = LLMClient(cfg, api_key=api_key)
    if not is_ollama_url(cfg.base_url):
        raw_run_id = override.get("CV_AGENT_RUN_ID", "")
        run_id = int(raw_run_id) if str(raw_run_id).isdigit() else None
        user = (_current_user() if has_request_context() else current_user()) or DEFAULT_USER
        try:
            client.rate_limiter = LLMRateLimiter(cfg, user, run_id)
        except (AttributeError, TypeError):
            pass  # schlanke Test-Doubles ohne beschreibbare Attribute
    return client


def _active_model() -> str:
    override = _llm_override()
    return override["LLM_MODEL"] if override else load_criteria().llm.model


def _translate_vacancy(v) -> None:
    load_env()
    env, _ = _effective_llm_env()
    llm = _llm_client(env)
    v.translation = llm.complete(
        "Du bist ein Übersetzer. Übersetze die Stellenanzeige vollständig ins Russische. "
        "Antworte nur mit der Übersetzung.",
        v.description[:6000],
    )


def _store() -> Store:
    return Store(user=_current_user())


def _user_env(user: str | None = None) -> dict:
    return {"CV_AGENT_USER": user or _current_user() or DEFAULT_USER}


REQUEST_SPECIFIC_ENV_KEYS = WORKER_ENV_KEYS | {"CV_AGENT_RUN_ID"}


def _child_base_env() -> dict:
    """Env для потомка: os.environ без request-specific значений и ключей
    аккаунтов (CLI загружает их по профилю из SQLite) + .env поверх."""
    env = {
        key: value
        for key, value in os.environ.items()
        if key not in REQUEST_SPECIFIC_ENV_KEYS and key not in ACCOUNT_KEY_ENVS
    }
    for key, value in dotenv_overrides().items():
        env.setdefault(key, value)
    return env


@app.before_request
def _require_login():
    if request.endpoint in (
        "login",
        "login_post",
        "accounts_create",
        "static",
        "health",
    ):
        return None
    user = _current_user()
    if not user:
        session.clear()
        resp = redirect(url_for("login"))
        if request.cookies.get(LEGACY_USER_COOKIE):
            resp.delete_cookie(LEGACY_USER_COOKIE, path="/")
        return resp


@app.after_request
def _security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; style-src 'self' 'unsafe-inline'; "
        "script-src 'self' 'unsafe-inline'; connect-src 'self'; "
        "img-src 'self' data:; frame-ancestors 'none'"
    )
    response.headers["Cache-Control"] = "no-store"
    return response


@app.errorhandler(413)
def request_too_large(_error):
    return "Загрузка превышает общий лимит 25 МБ.", 413


@app.route("/login")
def login():
    message = ""
    deleted = normalize_account_name(request.args.get("deleted", ""))
    if deleted and valid_account_name(deleted):
        message = (
            f"Профиль «{deleted}» удалён. Архив его данных сохранён в data/backups/."
        )
    store = Store()
    accounts = store.list_accounts()
    store.close()
    return render_template(
        "login.html", accounts=accounts, error="", message=message
    )


@app.route("/login", methods=["POST"])
def login_post():
    account = request.form.get("account", "").strip()
    store = Store()
    try:
        ok = store.account_exists(account)
        if ok:
            _select_account(store, account)
    finally:
        store.close()
    if not ok:
        fallback = Store()
        try:
            accounts = fallback.list_accounts()
        finally:
            fallback.close()
        return render_template(
            "login.html",
            accounts=accounts,
            error=f"Профиль «{account}» не найден.",
        )
    resp = redirect(url_for("results"))
    resp.delete_cookie(LEGACY_USER_COOKIE, path="/")
    return resp


@app.route("/accounts/create", methods=["POST"])
def accounts_create():
    name = normalize_account_name(request.form.get("name", ""))
    store = Store()
    accounts = store.list_accounts()
    if not _require_account(name):
        store.close()
        return render_template(
            "login.html",
            accounts=accounts,
            error="Недопустимое имя профиля.",
        ), 400
    if not store.create_account(name):
        store.close()
        return render_template(
            "login.html",
            accounts=accounts,
            error=f"Профиль «{name}» уже существует.",
        ), 409
    try:
        _select_account(store, name)
    finally:
        store.close()
    resp = redirect(url_for("results"))
    resp.delete_cookie(LEGACY_USER_COOKIE, path="/")
    return resp


ALLOWED_DOC_EXT = SUPPORTED_DOCUMENT_EXTENSIONS
MAX_DOCUMENTS_PER_UPLOAD = 10
MAX_DOCUMENT_BYTES = 10 * 1024 * 1024
MAX_ACCOUNT_DOCUMENT_BYTES = 50 * 1024 * 1024


def _user_cv_folder() -> Path:
    return cv_dir(_current_user(), legacy_root_fallback=False)


def _validated_upload(f, suffix: str) -> bytes:
    data = f.stream.read(MAX_DOCUMENT_BYTES + 1)
    if len(data) > MAX_DOCUMENT_BYTES:
        raise ValueError("файл превышает лимит 10 МБ")
    if not data:
        raise ValueError("пустой файл")
    if suffix == ".pdf":
        if not data.startswith(b"%PDF-"):
            raise ValueError("некорректная сигнатура PDF")
        try:
            PdfReader(io.BytesIO(data))
        except Exception as exc:
            raise ValueError("повреждённый PDF") from exc
    elif suffix in {".txt", ".md"}:
        try:
            data.decode("utf-8-sig")
        except UnicodeError as exc:
            raise ValueError("текстовый файл должен быть в UTF-8") from exc
    return data


def _atomic_save(folder: Path, name: str, data: bytes) -> None:
    target = folder / name
    if target.is_symlink():
        raise ValueError("нельзя перезаписать символическую ссылку")
    tmp_name = ""
    try:
        with tempfile.NamedTemporaryFile(dir=folder, prefix=".upload-", delete=False) as tmp:
            tmp.write(data)
            tmp_name = tmp.name
        os.chmod(tmp_name, 0o600)
        os.replace(tmp_name, target)
    finally:
        if tmp_name and os.path.exists(tmp_name):
            os.unlink(tmp_name)


PROFILE_SECTIONS = ("profile", "keywords", "documents", "llm", "keys")


def _render_profile(message: str = "", error: str = "", section: str = "profile") -> str:
    if section not in PROFILE_SECTIONS:
        section = "profile"
    user = _current_user()
    store = _store()
    try:
        preferred_llm = store.account_preferred_llm()
        api_keys = store.account_api_keys()
        account_keywords = store.account_keywords() or []
        candidate_profile = store.candidate_profile() or {}
        docs = scan_documents(
            user, purpose="scoring", persist=True, store=store
        )
    finally:
        store.close()
    current_llm, current_option = _current_llm()
    limits_rows, limits_stamp = _limits_rows(current_option)
    return render_template(
        "profile.html",
        name=user,
        api_keys=api_keys,
        preferred_llm=preferred_llm,
        preferred_provider=current_option["provider"] if current_option else "",
        llms=available_llms(),
        llm_groups=_llm_groups(),
        ollama_up=_ollama_alive(),
        candidate_profile=candidate_profile,
        docs=docs,
        account_keywords=account_keywords,
        message=message,
        error=error,
        active_section=section,
        current_llm=current_llm,
        limits_rows=limits_rows,
        limits_stamp=limits_stamp,
    )


@app.route("/profile")
def profile():
    return _render_profile(section=request.args.get("section", "profile"))


@app.route("/profile/details", methods=["POST"])
def profile_details():
    limits = {
        "full_name": 200,
        "address": 1000,
        "email": 320,
        "phone": 100,
        "languages": 1000,
        "search_region": 300,
        "resume_city": 120,
        "scoring_cities": 2500,
    }
    values = {
        key: request.form.get(key, "").strip()
        for key in limits
    }
    too_long = [key for key, limit in limits.items() if len(values[key]) > limit]
    if too_long:
        return _render_profile(
            error="Слишком длинные поля: " + ", ".join(too_long) + ".",
            section="profile",
        ), 400
    email = values["email"]
    if email and ("@" not in email or any(char.isspace() for char in email)):
        return _render_profile(
            error="Некорректный адрес электронной почты.", section="profile"
        ), 400
    preferred = request.form.get("preferred_letter_language", "de").strip()
    if preferred not in {"de", "en", "ru"}:
        return _render_profile(error="Некорректный язык письма.", section="profile"), 400
    languages = [
        value.strip()
        for value in values["languages"].replace("\n", ",").split(",")
        if value.strip()
    ]
    preferences = {}
    if "scoring_work_mode" in request.form or "scoring_cities" in request.form:
        from .scoring_location import parse_location_preferences
        try:
            preferences["scoring_preferences"] = parse_location_preferences({
                "work_mode": request.form.get("scoring_work_mode", "unspecified"),
                "cities": [city.strip() for city in values["scoring_cities"].replace("\n", ",").split(",") if city.strip()],
            })
        except ValueError as exc:
            return _render_profile(error=str(exc), section="profile"), 400
    store = _store()
    try:
        store.set_candidate_profile(
            {
                "full_name": values["full_name"],
                "address": values["address"],
                "email": email,
                "phone": values["phone"],
                "languages": languages,
                "preferred_letter_language": preferred,
                "search_region": values["search_region"],
                "resume_city": values["resume_city"],
                **preferences,
            }
        )
    finally:
        store.close()
    return _render_profile(
        message="Данные кандидата сохранены.", section="profile"
    )


@app.route("/profile/upload", methods=["POST"])
def profile_upload():
    folder = _user_cv_folder()
    folder.mkdir(parents=True, exist_ok=True, mode=0o700)
    saved, rejected = 0, []
    used_bytes = sum(
        p.stat().st_size
        for p in folder.iterdir()
        if p.is_file() and not p.is_symlink()
    )
    files = request.files.getlist("docs")
    if len(files) > MAX_DOCUMENTS_PER_UPLOAD:
        rejected.extend(
            f.filename or "без имени" for f in files[MAX_DOCUMENTS_PER_UPLOAD:]
        )
        files = files[:MAX_DOCUMENTS_PER_UPLOAD]
    for f in files:
        if not f or not f.filename:
            continue
        try:
            name = safe_filename(f.filename)
        except ValueError:
            rejected.append(f.filename)
            continue
        suffix = Path(name).suffix.lower()
        if suffix not in ALLOWED_DOC_EXT:
            rejected.append(name)
            continue
        try:
            data = _validated_upload(f, suffix)
            target = folder / name
            previous_size = (
                target.stat().st_size
                if target.is_file() and not target.is_symlink()
                else 0
            )
            if used_bytes - previous_size + len(data) > MAX_ACCOUNT_DOCUMENT_BYTES:
                raise ValueError("превышена квота профиля 50 МБ")
            _atomic_save(folder, name, data)
            used_bytes = used_bytes - previous_size + len(data)
            saved += 1
        except ValueError as exc:
            rejected.append(f"{name} ({exc})")
    message = f"Загружено документов: {saved}." if saved else ""
    error = f"Не поддерживаются: {', '.join(rejected)}." if rejected else ""
    return _render_profile(message=message, error=error, section="documents")


@app.route("/profile/delete/<path:name>", methods=["POST"])
def profile_delete(name: str):
    folder = _user_cv_folder()
    try:
        safe = safe_filename(name)
    except ValueError:
        return redirect(url_for("profile"))
    target = folder / safe
    if target.is_symlink():
        return redirect(url_for("profile", section="documents"))
    if target.is_file():
        target.unlink()
        store = _store()
        try:
            store.delete_document_metadata(safe)
        finally:
            store.close()
    return redirect(url_for("profile", section="documents"))


@app.route("/profile/document/<path:name>/type", methods=["POST"])
def profile_document_type(name: str):
    try:
        safe = safe_filename(name)
    except ValueError:
        return redirect(url_for("profile", section="documents"))
    document_type = request.form.get("document_type", "")
    if document_type not in DOCUMENT_TYPES:
        return _render_profile(
            error="Некорректный тип документа.", section="documents"
        ), 400
    store = _store()
    try:
        # Сначала обновляем metadata текущих файлов, затем фиксируем manual override.
        scan_documents(
            _current_user(), purpose="scoring", persist=True, store=store
        )
        changed = store.set_document_type(safe, document_type)
    finally:
        store.close()
    if not changed:
        return _render_profile(error="Документ не найден.", section="documents"), 404
    return redirect(url_for("profile", section="documents"))


@app.route("/profile/account/delete", methods=["POST"])
def profile_account_delete():
    """Удаление профиля: экспорт-архив → каскадное удаление → очистка каталогов.

    Требует ввода точного имени профиля; при активной фоновой задаче отказ.
    """
    user = _current_user()
    confirm = normalize_account_name(request.form.get("confirm_name", ""))
    if not user or confirm != user:
        return _render_profile(
            error="Подтверждение не совпало с именем профиля — удаление отменено.",
        ), 400
    if user == DEFAULT_USER:
        return _render_profile(
            error=(
                f"Профиль по умолчанию «{DEFAULT_USER}» удалить нельзя — "
                "он является системным fallback. Любой другой профиль можно."
            ),
        ), 403
    store = _store()
    try:
        if store.active_job() is not None:
            return _render_profile(
                error="У профиля есть активная фоновая задача — сначала дождитесь её завершения.",
            ), 409
        try:
            report = export_account_archive(store, user)
        except ValueError as exc:
            return _render_profile(error=str(exc))
        if not store.delete_account(user):
            return _render_profile(error="Профиль не найден."), 404
    finally:
        store.close()
    remove_account_storage(report["account_id"], user)
    session.clear()
    resp = redirect(url_for("login", deleted=user))
    resp.delete_cookie(LEGACY_USER_COOKIE, path="/")
    return resp


@app.route("/profile/keywords/generate", methods=["POST"])
def profile_keywords_generate():
    """Генерация ключевых слов из документов: CLI `keywords --apply`."""
    try:
        code, output = _run_cli(
            ["keywords", "--apply"], extra_env=_user_env(), timeout=900
        )
    except CommandTimedOut as exc:
        return _render_profile(error=str(exc), section="keywords"), 504
    if code != 0:
        tail = (output or "").strip()[-200:]
        return _render_profile(
            error=f"Генерация не удалась (код {code}). {tail}",
            section="keywords",
        ), 500
    return _render_profile(
        message="Ключевые слова сгенерированы.", section="keywords"
    )


@app.route("/profile/keywords/add", methods=["POST"])
def profile_keywords_add():
    keyword = request.form.get("keyword", "").strip()
    if not keyword or len(keyword) > 100:
        return _render_profile(
            error="Укажите ключевое слово (до 100 символов).",
            section="keywords",
        ), 400
    store = _store()
    try:
        current = store.account_keywords() or []
        if any(k.lower() == keyword.lower() for k in current):
            return _render_profile(
                error="Такое ключевое слово уже есть.", section="keywords"
            ), 400
        store.set_account_keywords(current + [keyword])
    finally:
        store.close()
    return _render_profile(
        message="Ключевое слово добавлено.", section="keywords"
    )


@app.route("/profile/keywords/delete", methods=["POST"])
def profile_keywords_delete():
    keyword = request.form.get("keyword", "").strip()
    store = _store()
    try:
        current = store.account_keywords() or []
        remaining = [k for k in current if k != keyword]
        if len(remaining) == len(current):
            return _render_profile(
                error="Ключевое слово не найдено.", section="keywords"
            ), 404
        store.set_account_keywords(remaining)
    finally:
        store.close()
    return _render_profile(
        message="Ключевое слово удалено.", section="keywords"
    )


@app.route("/profile/keys/add", methods=["POST"])
def profile_keys_add():
    key_name = request.form.get("name", "").strip()
    provider = request.form.get("provider", "").strip().lower()
    key = request.form.get("api_key", "").strip()
    store = _store()
    try:
        store.add_account_api_key(key_name, provider, key)
    except ValueError as exc:
        return _render_profile(error=str(exc), section="keys"), 400
    finally:
        store.close()
    return _render_profile(
        message=f"Ключ «{key_name}» добавлен.", section="keys"
    )


@app.route("/profile/keys/delete", methods=["POST"])
def profile_keys_delete():
    key_name = request.form.get("name", "").strip()
    store = _store()
    try:
        removed = store.delete_account_api_key(key_name)
    finally:
        store.close()
    if not removed:
        return _render_profile(error="Ключ не найден.", section="keys"), 404
    return _render_profile(
        message=f"Ключ «{key_name}» удалён.", section="keys"
    )


def _probe_selected_llm(option: dict) -> str:
    """Короткий проверочный запрос после сохранения модели — освежает лимиты."""
    if option["provider"] == "ollama":
        if not _ollama_alive():
            return "Ollama недоступна — проверочный запрос пропущен."
    elif option["api_key_env"] in ACCOUNT_KEY_ENVS and has_request_context():
        if not _account_llm_key(option["api_key_env"]):
            return (
                f"Ключ {option['api_key_env']} не настроен — "
                "проверочный запрос пропущен."
            )
    cfg = load_criteria().llm
    cfg.provider = option["provider"]
    cfg.model = option["model"]
    cfg.base_url = option["base_url"]
    cfg.api_key_env = option["api_key_env"]
    try:
        cfg.timeout = min(float(cfg.timeout or 30.0), 15.0)
    except (TypeError, ValueError):
        cfg.timeout = 15.0
    api_key = None
    if option["provider"] != "ollama":
        api_key = _account_llm_key(option["api_key_env"]) or ""
    client = LLMClient(cfg, api_key=api_key)
    client.complete("Antworte mit genau einem Wort: OK", "OK")
    return "Проверочный запрос выполнен, лимиты обновлены."


@app.route("/profile/llm", methods=["POST"])
def profile_llm():
    """Выбор модели профиля: сохраняется только ID из серверного allowlist.

    После сохранения уходит минимальный проверочный запрос — его ответ
    (usage и rate-limit заголовки) обновляет блок «Лимиты».
    """
    selected = urllib.parse.unquote(request.form.get("llm", ""))
    option = next((o for o in available_llms() if o["id"] == selected), None)
    if option is None:
        return _render_profile(error="Некорректная модель.", section="llm"), 400
    store = _store()
    try:
        store.set_account_preferred_llm(selected)
    finally:
        store.close()
    try:
        note = _probe_selected_llm(option)
    except Exception as e:  # noqa: BLE001 — сохранение не должно ломаться
        note = f"Проверочный запрос не удался: {redact_sensitive(str(e), ())[:160]}"
    resp = make_response(
        _render_profile(
            message=f"Модель сохранена для этой учётной записи. {note}",
            section="llm",
        )
    )
    # Явное сохранение сбрасывает разовый выбор модели из /run-cookie.
    resp.delete_cookie(LLM_COOKIE, path="/")
    return resp


@app.route("/logout", methods=["POST"])
def logout():
    session.clear()
    resp = redirect(url_for("login"))
    resp.delete_cookie(LEGACY_USER_COOKIE, path="/")
    return resp


@app.route("/health")
def health():
    """Health check без секретов: DB, активные jobs, Ollama, версия схемы."""
    from .store import LATEST_SCHEMA_VERSION

    db_ok = True
    active_jobs = 0
    schema_version = 0
    try:
        store = _store()
        try:
            schema_version = store.schema_version()
            active_jobs = store.active_job_count()
        except Exception:
            db_ok = False
        finally:
            store.close()
    except Exception:
        db_ok = False
    payload = {
        "ok": db_ok,
        "db": "ok" if db_ok else "error",
        "schema_version": schema_version,
        "expected_schema_version": LATEST_SCHEMA_VERSION,
        "active_jobs": active_jobs,
        "ollama": bool(_ollama_alive()),
    }
    return payload, (200 if db_ok else 503)


@app.context_processor
def inject_active_llm():
    return {"active_llm": _active_model(), "current_user": _current_user()}


@app.template_filter("company_name")
def _company_name_filter(company: str) -> str:
    criteria = load_criteria()
    if is_recruiting_agency(company, criteria):
        return f"{company} ({AGENCY_LABEL})"
    return company


@app.template_filter("safe_external_url")
def _safe_external_url(value: str) -> bool:
    try:
        parsed = urllib.parse.urlsplit(value)
    except (TypeError, ValueError):
        return False
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

COMMANDS = {
    "search": {"label": "Поиск вакансий (search)", "args": [("limit", "limit")]},
    "score": {"label": "Оценить вакансии (score)", "args": [("verbose", "flag")]},
    "write": {"label": "Написать сопроводительное (write)", "args": [("refnr", "text")]},
    "apply": {"label": "Подготовить досье (apply)", "args": [("refnr", "text")]},
    "status": {"label": "Показать статусы (status)", "args": []},
    "keywords": {
        "label": "Ключевые слова из CV (keywords)",
        "args": [("apply", "flag")],
    },
}


def _validated_limit(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        return ""
    try:
        number = int(value)
    except ValueError as exc:
        raise ValueError("Лимит должен быть целым числом от 1 до 500.") from exc
    if not 1 <= number <= 500:
        raise ValueError("Лимит должен быть целым числом от 1 до 500.")
    return str(number)


def _validated_radius(raw: str) -> str:
    value = (raw or "").strip()
    if not value:
        return ""
    try:
        number = int(value)
    except ValueError as exc:
        raise ValueError("Радиус должен быть целым числом от 1 до 500.") from exc
    if not 1 <= number <= 500:
        raise ValueError("Радиус должен быть целым числом от 1 до 500.")
    return str(number)


def _clean_location_part(raw: str, max_length: int = 100) -> str:
    value = re.sub(r"[\x00-\x1f\x7f]+", " ", str(raw or ""))
    return " ".join(value.split())[:max_length]


def _split_locations(raw: str, max_total: int = 300) -> list[str]:
    """Поле локации может содержать несколько значений через запятую."""
    parts: list[str] = []
    used = 0
    for chunk in str(raw or "").split(","):
        cleaned = _clean_location_part(chunk)
        if not cleaned:
            continue
        extra = len(cleaned) + (2 if parts else 0)
        if used + extra > max_total:
            break
        parts.append(cleaned)
        used += extra
    return parts


def _effective_search_location(country: str, region: str, city: str) -> str:
    """Страна/регион/город (по несколько через запятую); самое конкретное поле побеждает."""
    for field in (city, region, country):
        locations = _split_locations(field)
        if locations:
            return ", ".join(locations)
    return ""


SEARCH_FORM_FIELDS = ("limit", "country", "region", "city", "radius")

RESULTS_PAGE_SIZE = 50


def _clean_search_form_value(raw: str, max_length: int = 300) -> str:
    value = re.sub(r"[\x00-\x1f\x7f]+", " ", str(raw or ""))
    return " ".join(value.split())[:max_length]


def _save_search_form(user: str) -> None:
    """Запоминает введённые поля формы поиска за аккаунтом (между сессиями)."""
    params = {}
    for field in SEARCH_FORM_FIELDS:
        raw = request.form.get(field, "")
        if field == "radius":
            try:
                params[field] = _validated_radius(raw)
            except ValueError:
                params[field] = _clean_search_form_value(raw)
        else:
            params[field] = _clean_search_form_value(raw)
    store = Store(user=user)
    try:
        store.set_account_search_params(params)
    finally:
        store.close()


def _saved_search_form() -> dict:
    """Сохранённые поля формы поиска текущего аккаунта."""
    user = _current_user()
    if not user:
        return {}
    store = Store(user=user)
    try:
        params = store.account_search_params()
    finally:
        store.close()
    raw_limit = str(params.get("limit", "")).strip()
    raw_radius = str(params.get("radius", "")).strip()
    return {
        "limit": raw_limit if re.fullmatch(r"[0-9]{1,4}", raw_limit) else "",
        "country": _clean_search_form_value(params.get("country", "")),
        "region": _clean_search_form_value(params.get("region", "")),
        "city": _clean_search_form_value(params.get("city", "")),
        "radius": raw_radius if re.fullmatch(r"[0-9]{1,4}", raw_radius) else "",
    }


def _invalid_run(cmd: str, message: str, status: int = 400):
    return (
        render_template(
            "run.html",
            cmd=cmd,
            args=[],
            code=2,
            output=message,
            llm_note="",
        ),
        status,
    )


def _run_cli(
    args: list[str],
    extra_env: dict | None = None,
    on_line=None,
    timeout: float = 3600,
    terminate_grace: float = 5,
) -> tuple[int, str]:
    """Запуск CLI с wall-clock deadline, streaming и обязательным reap."""
    env = _child_base_env()
    env["PYTHONUNBUFFERED"] = "1"
    if extra_env:
        env.update({str(key): str(value) for key, value in extra_env.items()})
    cmd = [sys.executable, "-m", "cv_agent.cli", *args]
    output = ""

    def emit(line: str) -> None:
        nonlocal output
        safe = redact_sensitive(line)
        if on_line is None:
            output = (output + safe)[-200_000:]
        else:
            on_line(safe)

    code = _stream_command(
        cmd,
        env,
        emit,
        timeout=timeout,
        terminate_grace=terminate_grace,
    )
    return code, "" if on_line is not None else output


class CommandTimedOut(RuntimeError):
    pass


def _stream_command(
    cmd: list[str],
    env: dict[str, str],
    on_line,
    *,
    timeout: float,
    terminate_grace: float,
) -> int:
    """Читает stdout без блокировки deadline; timeout завершает и reap-ит child."""
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        bufsize=1,
    )
    assert proc.stdout is not None
    lines: queue.Queue[str | None] = queue.Queue()

    def read_stdout() -> None:
        try:
            for line in proc.stdout:
                lines.put(line)
        finally:
            lines.put(None)

    reader = threading.Thread(target=read_stdout, daemon=True)
    reader.start()
    deadline = time.monotonic() + max(0.01, timeout)
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise CommandTimedOut(
                    f"Команда превысила допустимое время {timeout:g} с."
                )
            try:
                line = lines.get(timeout=min(0.1, remaining))
            except queue.Empty:
                continue
            if line is None:
                return proc.wait(timeout=max(0.01, remaining))
            on_line(line)
    except (CommandTimedOut, subprocess.TimeoutExpired) as exc:
        proc.terminate()
        try:
            proc.wait(timeout=max(0.01, terminate_grace))
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
        raise CommandTimedOut(
            f"Команда превысила допустимое время {timeout:g} с."
        ) from exc
    except Exception:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=max(0.01, terminate_grace))
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        raise
    finally:
        if proc.poll() is not None:
            proc.stdout.close()
        reader.join(timeout=1)


def _active_run(user: str | None = None) -> dict | None:
    """Активный DB-backed прогон только выбранного профиля."""
    store = Store(user=user or _current_user())
    try:
        return store.active_job()
    finally:
        store.close()


def _log_job_event(event: dict) -> None:
    """Структурный лог заданий: `data/logs/jobs.jsonl` (этап 7.3).

    Одна JSON-строка на событие: ts, event, run_id, account_id, kind, коды
    шагов, статусы, длительность, метрики запросов. Ошибки записи лога не
    влияют на выполнение задания; error-поля проходят redact_sensitive.
    """
    try:
        log_dir = Path(_config.DATA_DIR) / "logs"
        log_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        record = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **event,
        }
        if "error" in record:
            record["error"] = redact_sensitive(record["error"] or "")
        path = log_dir / "jobs.jsonl"
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        try:
            path.chmod(0o600)
        except OSError:
            pass
    except Exception:
        pass  # сбой журналирования не должен ломать pipeline


def _pipeline_worker(run_id: int, user: str, extra_env: dict) -> None:
    """Выполняет сохранённый job; состояние и лог остаются в SQLite."""
    store = Store(user=user)
    started = time.monotonic()
    try:
        if not store.claim_job(run_id):
            return
        run = store.job(run_id)
        if run is None:
            return
        account_id = store.account_id(user) or ""
        _log_job_event(
            {
                "event": "job_started",
                "run_id": run_id,
                "account_id": account_id,
                "kind": run["kind"],
                "steps": [s["cmd"] for s in run["steps"]],
            }
        )
        for step_index, step in enumerate(run["steps"]):

            def on_line(line, index=step_index):
                store.append_job_output(run_id, index, line)

            step_env = dict(extra_env)
            step_env["CV_AGENT_RUN_ID"] = str(run_id)
            code, _ = _run_cli(step["args"], extra_env=step_env, on_line=on_line)
            store.set_job_step_code(run_id, step_index, code)
            _log_job_event(
                {
                    "event": "step_finished",
                    "run_id": run_id,
                    "account_id": account_id,
                    "step": step_index,
                    "cmd": step["cmd"],
                    "code": int(code),
                }
            )
            if code != 0:
                reason = (
                    f"Команда {step['cmd']} завершилась с ошибкой (код {code}) — "
                    "процесс остановлен."
                )
                store.finish_job(run_id, "failed", reason)
                _log_job_event(
                    {
                        "event": "job_finished",
                        "run_id": run_id,
                        "account_id": account_id,
                        "kind": run["kind"],
                        "status": "failed",
                        "error": reason,
                        "duration_seconds": round(time.monotonic() - started, 3),
                    }
                )
                return
            if str(step["cmd"]).startswith("search"):
                stored = store.job(run_id)
                output_text = (
                    str(stored["steps"][step_index].get("output", ""))
                    if stored
                    else ""
                )
                match = re.search(r"query errors:\s*(\d+)", output_text)
                errors_seen = int(match.group(1)) if match else 0
                if errors_seen:
                    store.set_job_step_warning(
                        run_id,
                        step_index,
                        f"Частичный успех: {errors_seen} запросов к источникам "
                        "завершились ошибкой — часть результатов могла быть потеряна.",
                    )
        store.finish_job(run_id, "succeeded")
        _log_job_event(
            {
                "event": "job_finished",
                "run_id": run_id,
                "account_id": account_id,
                "kind": run["kind"],
                "status": "succeeded",
                "duration_seconds": round(time.monotonic() - started, 3),
            }
        )
    except CommandTimedOut as e:
        store.finish_job(run_id, "timed_out", redact_sensitive(e))
        _log_job_event(
            {
                "event": "job_finished",
                "run_id": run_id,
                "account_id": store.account_id(user) or "",
                "kind": "",
                "status": "timed_out",
                "error": str(e),
                "duration_seconds": round(time.monotonic() - started, 3),
            }
        )
    except Exception as e:
        reason = f"Ошибка выполнения: {redact_sensitive(e)}"
        store.finish_job(run_id, "failed", reason)
        _log_job_event(
            {
                "event": "job_finished",
                "run_id": run_id,
                "account_id": store.account_id(user) or "",
                "kind": "",
                "status": "failed",
                "error": reason,
                "duration_seconds": round(time.monotonic() - started, 3),
            }
        )
    finally:
        store.close()


_active_workers: dict[int, subprocess.Popen] = {}
_workers_lock = threading.Lock()
CANCEL_GRACE_SECONDS = 5.0


def _cancel_process_tree(proc: subprocess.Popen, grace: float) -> None:
    """terminate → kill → reap для всей группы процессов воркера."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=grace)
        return
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass


def _launch_pipeline_worker(
    run_id: int,
    account_id: str,
    user: str,
    extra_env: dict,
) -> None:
    """Startet den DB-Worker als eigenen Prozess und überwacht seinen Exit."""
    env = _child_base_env()
    env.update(
        {
            key: str(value)
            for key, value in extra_env.items()
            if key in WORKER_ENV_KEYS
        }
    )
    env["CV_AGENT_USER"] = user
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "cv_agent.job_worker",
            "--run-id",
            str(run_id),
            "--account-id",
            account_id,
        ],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        env=env,
        close_fds=True,
        start_new_session=True,
    )
    with _workers_lock:
        _active_workers[run_id] = proc

    def reap_worker() -> None:
        try:
            code = proc.wait()
        except Exception:
            return
        with _workers_lock:
            _active_workers.pop(run_id, None)
        if code == 0:
            return
        worker_store = Store(user=user)
        try:
            worker_store.finish_job(
                run_id,
                "failed",
                f"Worker-Prozess wurde unerwartet beendet (Code {code}).",
            )
        finally:
            worker_store.close()

    threading.Thread(target=reap_worker, daemon=True).start()


def _start_pipeline_job(
    user: str, kind: str, steps: list[dict], llm_note: str, extra_env: dict
) -> int:
    """Атомарно создаёт job профиля и запускает отдельный worker-процесс."""
    store = Store(user=user)
    try:
        run_id, created = store.create_job(kind, steps, llm_note)
        account_id = store.account_id(user)
    finally:
        store.close()
    if created:
        try:
            _launch_pipeline_worker(
                run_id,
                str(account_id),
                user,
                dict(extra_env),
            )
        except Exception as exc:
            failed_store = Store(user=user)
            try:
                failed_store.finish_job(
                    run_id,
                    "failed",
                    "Worker-Prozess konnte nicht gestartet werden: "
                    f"{redact_sensitive(exc)}",
                )
            finally:
                failed_store.close()
    return run_id


def wait_pipeline(
    run_id: int, timeout: float = 30.0, user: str = DEFAULT_USER
) -> None:
    """Блокирует до завершения прогона (для тестов)."""
    deadline = time.time() + timeout
    store = Store(user=user)
    try:
        while time.time() < deadline:
            run = store.job(run_id)
            if run is not None and run["done"]:
                return
            time.sleep(0.02)
    finally:
        store.close()


@app.route("/run", methods=["POST"])
def run():
    cmd = request.form.get("cmd", "status")
    if cmd not in COMMANDS:
        return _invalid_run(cmd, "Команда не разрешена сервером.")
    args = []
    try:
        limit = _validated_limit(request.form.get("limit", ""))
    except ValueError as exc:
        return _invalid_run(cmd, str(exc))
    if cmd == "search" and limit:
        args += ["--limit", limit]
    verbose = request.form.get("verbose")
    if cmd == "score" and verbose:
        args += ["--verbose"]
    apply_flag = request.form.get("apply")
    if cmd == "keywords" and apply_flag:
        args += ["--apply"]
    refnr = request.form.get("refnr", "").strip()
    if cmd in ("write", "apply"):
        if not refnr or len(refnr) > 200 or any(ord(char) < 32 for char in refnr):
            return _invalid_run(cmd, "Некорректный номер вакансии.")
        args += [refnr]
    criteria = load_criteria()
    algorithmic_score = cmd == "score" and getattr(getattr(criteria, "scoring", None), "engine", "llm") == "algorithmic"
    extra_env, llm_note = ({}, "Алгоритмическая оценка с локальными эмбеддингами.") if algorithmic_score else _subprocess_llm_env()
    extra_env.update(_user_env())
    try:
        code, output = _run_cli([cmd, *args], extra_env=extra_env)
    except CommandTimedOut as exc:
        code, output = 124, redact_sensitive(exc)
    resp = make_response(
        render_template(
            "run.html",
            cmd=cmd,
            args=args,
            code=code,
            output=output,
            llm_note=llm_note,
        )
    )
    llm_id = _allowed_llm_id(request.form.get("llm", ""))
    if llm_id:
        resp.set_cookie(
            LLM_COOKIE,
            llm_id,
            path="/",
            max_age=2592000,
            secure=request.is_secure,
            samesite="Strict",
        )
    else:
        resp.delete_cookie(LLM_COOKIE, path="/")
    return resp


@app.route("/pipeline", methods=["POST"])
def pipeline():
    """Поиск вакансий в фоне: (keywords --apply →) search → score, с живым прогрессом.

    Ключевые слова генерируются, если включён чекбокс или их нет в criteria.yaml.
    Если cv/ пуст и нужны ключевые слова — предупреждение и остановка.
    """
    try:
        limit = _validated_limit(request.form.get("limit", ""))
        radius = _validated_radius(request.form.get("radius", ""))
    except ValueError as exc:
        return render_template(
            "pipeline.html", run=None, error=str(exc), llm_note=""
        ), 400
    regenerate = bool(request.form.get("regenerate"))
    user = _current_user()
    active = _active_run(user)
    if active:
        return redirect(url_for("pipeline_view", run_id=active["run_id"]))
    _save_search_form(user)
    c = load_criteria(user=user)
    needs_keywords = regenerate or not c.keywords
    algorithmic = getattr(getattr(c, "scoring", None), "engine", "llm") == "algorithmic"
    extra_env, llm_note = ({}, "Алгоритмическая оценка с локальными эмбеддингами.") if algorithmic and not needs_keywords else _subprocess_llm_env()
    steps = []
    if needs_keywords:
        if not load_document_texts(user=user):
            return render_template(
                "pipeline.html",
                run=None,
                error="Папка cv пуста — невозможно сгенерировать ключевые слова.",
                llm_note=llm_note,
            )
        steps.append(
            {"cmd": "keywords --apply", "args": ["keywords", "--apply"], "code": None, "output": ""}
        )
    extra_env.update(_user_env(user))
    location = _effective_search_location(
        request.form.get("country", ""),
        request.form.get("region", ""),
        request.form.get("city", ""),
    )
    args = (
        ["search"]
        + (["--limit", limit] if limit else [])
        + ["--location", location]
        + (["--radius", radius] if radius else [])
    )
    search_cmd = "search"
    if limit:
        search_cmd += f" --limit {limit}"
    search_cmd += f" --location {location}" if location else " --location *"
    if radius:
        search_cmd += f" --radius {radius}"
    steps.append(
        {
            "cmd": search_cmd,
            "args": args,
            "code": None,
            "output": "",
        }
    )
    steps.append({"cmd": "score", "args": ["score"], "code": None, "output": ""})
    run_id = _start_pipeline_job(user, "pipeline", steps, llm_note, extra_env)
    return redirect(url_for("pipeline_view", run_id=run_id))


@app.route("/actualize", methods=["POST"])
def actualize():
    """Актуализация: проверяет все вакансии профиля у источников в фоне."""
    user = _current_user()
    active = _active_run(user)
    if active:
        return redirect(url_for("pipeline_view", run_id=active["run_id"]))
    steps = [
        {"cmd": "actualize", "args": ["actualize"], "code": None, "output": ""}
    ]
    run_id = _start_pipeline_job(user, "actualize", steps, "", _user_env(user))
    return redirect(url_for("pipeline_view", run_id=run_id))


@app.route("/pipeline/<int:run_id>/cancel", methods=["POST"])
def pipeline_cancel(run_id):
    """Отмена активного прогона: terminate/kill группы процессов воркера."""
    store = _store()
    try:
        run = store.job(run_id)
    finally:
        store.close()
    if run is None or run["done"]:
        return render_template(
            "pipeline.html", run=run,
            error="Активный прогон не найден.", llm_note="",
        ), 404
    with _workers_lock:
        proc = _active_workers.get(run_id)
    if proc is None or proc.poll() is not None:
        return render_template(
            "pipeline.html", run=run,
            error="Процесс воркера не найден — отмена невозможна.",
            llm_note="",
        ), 409
    _cancel_process_tree(proc, CANCEL_GRACE_SECONDS)
    with _workers_lock:
        _active_workers.pop(run_id, None)
    store = _store()
    try:
        store.finish_job(run_id, "cancelled", "Отменено пользователем.")
        _log_job_event(
            {
                "event": "job_finished",
                "run_id": run_id,
                "account_id": store.account_id(_current_user()) or "",
                "kind": run["kind"],
                "status": "cancelled",
                "error": "Отменено пользователем.",
            }
        )
    finally:
        store.close()
    return redirect(url_for("pipeline_view", run_id=run_id))


@app.route("/pipeline/<int:run_id>")
def pipeline_view(run_id):
    store = _store()
    try:
        run = store.job(run_id)
    finally:
        store.close()
    if run is None:
        return render_template(
            "pipeline.html", run=None, error="Прогон не найден.", llm_note=""
        )
    return render_template(
        "pipeline.html", run=run, error="", llm_note=run["llm_note"]
    )


def _apply_log_cursors(run: dict, raw_cursors: str) -> None:
    """Отрезает уже переданный хвост лога; при сдвиге курсора — полный сброс.

    Хвост лога обрезается до JOB_LOG_LIMIT, поэтому длины со временем
    смещаются: невалидный/запредельный курсор помечает output_reset=True и
    отдаёт шаг целиком.
    """
    cursors: list[int] = []
    for part in raw_cursors.split(","):
        part = part.strip()
        cursors.append(int(part) if part.lstrip("-").isdigit() else -1)
    for index, step in enumerate(run["steps"]):
        cursor = cursors[index] if index < len(cursors) else 0
        output = str(step.get("output", ""))
        step["output_cursor"] = len(output)
        if 0 <= cursor <= len(output):
            step["output_reset"] = False
            step["output"] = output[cursor:]
        else:
            step["output_reset"] = True


@app.route("/pipeline/<int:run_id>/status")
def pipeline_status(run_id):
    store = _store()
    try:
        run = store.job(run_id)
    finally:
        store.close()
    if run is None:
        return {"done": True, "error": "Прогон не найден.", "steps": []}, 404
    raw_cursors = request.args.get("cursor", "").strip()
    if raw_cursors:
        _apply_log_cursors(run, raw_cursors)
    return run


def _latest_dossier(refnr: str, user: str | None = None):
    """Актуальная папка досье для refnr текущего профиля.

    Досье лежат в applications/<Компания>_<refnr>/ (легаси-версии
    `<...>_vN` тоже распознаются, пока не удалены); побеждает самый
    свежий по mtime каталог. refnr участвует только в сравнении
    суффиксов имён реальных каталогов, путь из ввода не строится.
    """
    root = applications_dir(user or _current_user(), APPLICATIONS_DIR)
    if not root.is_dir():
        return None
    suffix = f"_{safe_slug(refnr, 40)}"
    candidates = []
    for entry in root.iterdir():
        if not entry.is_dir() or entry.name.startswith("."):
            continue
        # имя досье: <Компания>_<refnr> или легаси <Компания>_<refnr>_v<N>
        version_match = re.search(r"_v(\d+)$", entry.name)
        base_name = (
            entry.name[: version_match.start()]
            if version_match
            else entry.name
        )
        if not base_name.endswith(suffix) or len(base_name) <= len(suffix):
            continue
        try:
            mtime = int(entry.stat().st_mtime)
        except OSError:
            continue
        candidates.append((mtime, entry.name, entry))
    if not candidates:
        return None
    return max(candidates)[2]


def _dossier_file(refnr: str, filename: str, user: str | None = None):
    dossier = _latest_dossier(refnr, user)
    if dossier is None:
        return None
    path = dossier / filename
    return path if path.is_file() else None


def _ansschreiben_file(refnr: str, user: str | None = None):
    """Anschreiben досье: новый формат .pdf, легаси-фолбэк .md."""
    return (
        _dossier_file(refnr, "Anschreiben.pdf", user)
        or _dossier_file(refnr, "Anschreiben.md", user)
    )


@app.route("/vacancy/<refnr>/anschreiben")
def anschreiben_view(refnr):
    """Anschreiben вакансии для просмотра (открывается в новой вкладке)."""
    path = _ansschreiben_file(refnr)
    if path is None:
        abort(404)
    mimetype = (
        "application/pdf" if path.suffix.lower() == ".pdf" else "text/markdown"
    )
    return send_file(path, mimetype=mimetype)


@app.route("/vacancy/<refnr>/anschreiben/download")
def anschreiben_download(refnr):
    """Скачивание Anschreiben вакансии как файла."""
    path = _ansschreiben_file(refnr)
    if path is None:
        abort(404)
    return send_file(path, as_attachment=True, download_name=path.name)


@app.route("/vacancy/<refnr>/lebenslauf")
def lebenslauf_view(refnr):
    """Lebenslauf досье для просмотра (открывается в новой вкладке)."""
    path = _dossier_file(refnr, "Lebenslauf.pdf")
    if path is None:
        abort(404)
    return send_file(path, mimetype="application/pdf")


@app.route("/vacancy/<refnr>/lebenslauf/download")
def lebenslauf_download(refnr):
    """Скачивание Lebenslauf досье как файла."""
    path = _dossier_file(refnr, "Lebenslauf.pdf")
    if path is None:
        abort(404)
    return send_file(path, as_attachment=True, download_name=path.name)


@app.route("/vacancy/<refnr>/<any(anschreiben, lebenslauf):doc>/delete", methods=["POST"])
def dossier_delete(refnr, doc):
    """Удаление документа из досье вакансии (Anschreiben.md/Lebenslauf.pdf)."""
    filename = "Lebenslauf.pdf" if doc == "lebenslauf" else None
    if filename is None:
        path = _ansschreiben_file(refnr)
    else:
        path = _dossier_file(refnr, filename)
    if path is not None:
        try:
            path.unlink()
        except OSError:
            abort(404)
    return redirect(url_for("vacancy", refnr=refnr))


@app.route("/vacancy/<refnr>/apply", methods=["POST"])
def apply_vacancy(refnr):
    """Создаёт досье и возвращается на страницу вакансии (PRG, без run-страницы)."""
    extra_env, _llm_note = _subprocess_llm_env()
    extra_env.update(_user_env())
    try:
        code, output = _run_cli(["apply", refnr], extra_env=extra_env)
    except CommandTimedOut as exc:
        code, output = 124, redact_sensitive(str(exc))
    if code == 0:
        return redirect(url_for("vacancy", refnr=refnr, apply_ok=1))
    lines = [line for line in (output or "").strip().splitlines() if line.strip()]
    detail = lines[-1] if lines else f"CLI apply завершился с кодом {code}"
    return redirect(
        url_for(
            "vacancy",
            refnr=refnr,
            apply_error=redact_sensitive(detail)[:300],
        )
    )


FILTER_PARAM_PREFIX = "f_"
SCORE_FORMAT_LABEL = "ошибка оценки"
CHECK_MARK = "✔"
DASH = "–"
FILTER_VALUE_SEPARATOR = "||"


def _score_filter_value(row) -> str:
    """Отображаемое в столбце «Оценка» значение — как в ячейке таблицы."""
    if row[4] is not None:
        return f"{row[4]:.0f}"
    if row[6] == "score_error":
        return SCORE_FORMAT_LABEL
    return str(row[7] or DASH)


SCORE_CONDITION_RE = re.compile(r"([<>=])\s*(\d{1,3}(?:\.\d)?)")


def _parse_score_condition(value: str) -> tuple[str, float] | None:
    """'<80' / '=80' / '>80' -> ('<'|'='|'>', число); иначе None."""
    match = SCORE_CONDITION_RE.fullmatch(str(value or "").strip())
    if match is None:
        return None
    try:
        number = float(match.group(2))
    except ValueError:
        return None
    return match.group(1), number


def _score_condition_matches(score: float | None, op: str, number: float) -> bool:
    if score is None:
        return False
    rounded = round(score)
    if op == "<":
        return rounded < number
    if op == ">":
        return rounded > number
    return rounded == number


def _filter_column_getters(score_summaries=None):
    criteria = load_criteria()
    summaries = score_summaries or {}

    def company_display(row):
        company = row[2] or ""
        if company and is_recruiting_agency(company, criteria):
            return f"{company} ({AGENCY_LABEL})"
        return company

    return {
        "refnr": lambda r: r[0] or "",
        "title": lambda r: r[1] or "",
        "company": company_display,
        "location": lambda r: r[8] or DASH,
        "source": lambda r: r[3] or "",
        "score": _score_filter_value,
        "engine": lambda r: (
            "Алгоритм" if summaries.get(r[0], {}).get("engine") == "algorithmic" else "ИИ"
        ) if r[4] is not None or r[6] == "score_error" else DASH,
        "desc": lambda r: CHECK_MARK if r[9] else DASH,
        "transl": lambda r: CHECK_MARK if r[10] else DASH,
        "letter": lambda r: CHECK_MARK if r[5] in ("prepared", "applied") else DASH,
    }


def _parse_active_filters(args, getters) -> dict[str, list[str]]:
    active: dict[str, list[str]] = {}
    for key in getters:
        raw = args.get(FILTER_PARAM_PREFIX + key, "")
        values = [v for v in raw.split(FILTER_VALUE_SEPARATOR) if v]
        if values:
            active[key] = values
    return active


def _row_matches_filters(row, getters, active) -> bool:
    for key, allowed in active.items():
        value = getters[key](row)
        exact = [v for v in allowed if _parse_score_condition(v) is None]
        if value in exact:
            continue
        if key == "score":
            conditions = [
                parsed
                for parsed in map(_parse_score_condition, allowed)
                if parsed is not None
            ]
            if any(
                _score_condition_matches(row[4], op, number)
                for op, number in conditions
            ):
                continue
        return False
    return True


def _filter_options(rows, getters) -> dict[str, list[str]]:
    options: dict[str, list[str]] = {}
    for key, getter in getters.items():
        seen: list[str] = []
        for row in rows:
            value = getter(row)
            if value and value not in seen:
                seen.append(value)
        try:
            seen.sort(key=lambda s: (0, float(s)) if s.replace(".", "").isdigit() else (1, s.casefold()))
        except (TypeError, ValueError):
            seen.sort(key=str.casefold)
        options[key] = seen
    return options


@app.route("/")
@app.route("/results")
def results():
    store = _store()
    rows = store.list_all()
    score_summaries = store.score_summaries()
    store.close()
    rows.sort(key=lambda r: (r[4] is None, -(r[4] or -1)))
    score_errors = sum(1 for r in rows if r[6] == "score_error")

    getters = _filter_column_getters(score_summaries)
    active_filters = _parse_active_filters(request.args, getters)
    filtered_rows = [r for r in rows if _row_matches_filters(r, getters, active_filters)]

    total = len(filtered_rows)
    pages = max(1, (total + RESULTS_PAGE_SIZE - 1) // RESULTS_PAGE_SIZE)
    try:
        page = int(request.args.get("page", ""))
    except ValueError:
        page = 1
    page = min(max(page, 1), pages)
    start = (page - 1) * RESULTS_PAGE_SIZE

    filter_config = {
        "active": active_filters,
        "options": _filter_options(rows, getters),
        "separator": FILTER_VALUE_SEPARATOR,
        "paramPrefix": FILTER_PARAM_PREFIX,
    }
    active_filters_param = {
        FILTER_PARAM_PREFIX + key: FILTER_VALUE_SEPARATOR.join(values)
        for key, values in active_filters.items()
    }
    saved_form = _saved_search_form()
    criteria = load_criteria()
    return render_template(
        "results.html",
        rows=filtered_rows[start:start + RESULTS_PAGE_SIZE],
        score_summaries=score_summaries,
        start=start,
        score_errors=score_errors,
        total=total,
        page=page,
        pages=pages,
        active_filters=active_filters,
        active_filters_param=active_filters_param,
        filter_config=filter_config,
        search_form=saved_form,
        search_total_limit=(
            saved_form["limit"]
            or getattr(criteria, "search_total_limit", 100)
        ),
        search_radius=(
            saved_form["radius"]
            or str(getattr(criteria, "radius_km", 50))
        ),
    )


def _review_signer():
    return URLSafeTimedSerializer(app.config["SECRET_KEY"], salt="vacancy-review.v1")


@app.route("/vacancy/<refnr>/review", methods=["POST"])
def review_vacancy(refnr):
    try:
        context = _review_signer().loads(request.form.get("review_context", ""), max_age=86400)
    except BadSignature:
        abort(403, description="Перезагрузите карточку перед сохранением оценки.")
    if not isinstance(context, dict) or context.get("csrf") != session.get("review_csrf") or not session.get("review_csrf") or context.get("refnr") != refnr:
        abort(403)
    store = _store()
    try:
        if context.get("account_id") != store.account_id():
            abort(403)
        v = store.by_refnr(refnr)
        if v is None:
            abort(404)
        action = request.form.get("action", "save")
        if action == "clear":
            store.clear_vacancy_review(refnr)
            return redirect(url_for("vacancy", refnr=refnr, review_cleared=1, _anchor="your-review"), code=303)
        raw_grade = request.form.get("grade", "")
        reason = request.form.get("reason", "")
        if action != "save" or raw_grade not in {"0", "1", "2"} or len(reason) > 2000:
            error = "Выберите оценку и оставьте пояснение не длиннее 2000 символов."
            status = 400
        else:
            cfg = load_criteria().scoring
            revision = candidate_revision(store, cfg)
            if revision != context.get("candidate_hash") or vacancy_revision(v) != context.get("vacancy_hash"):
                error = "Документы профиля или описание вакансии изменились. Прочитайте обновлённые данные перед сохранением оценки."
                status = 409
            else:
                try:
                    inputs = capture_inputs(store, v, cfg, revision)
                    store.save_vacancy_review(refnr, int(raw_grade), reason, candidate_hash=revision,
                                              vacancy_hash=context["vacancy_hash"], inputs=inputs)
                except ValueError:
                    error = "Данные изменились во время сохранения. Проверьте карточку и повторите оценку."
                    status = 409
                else:
                    return redirect(url_for("vacancy", refnr=refnr, review_saved=1, _anchor="your-review"), code=303)
    finally:
        store.close()
    draft = {"grade": int(raw_grade) if raw_grade in {"0", "1", "2"} else None, "reason": reason[:2000]}
    return vacancy(refnr, review_error=error, review_draft=draft, fetch=False), status


@app.route("/vacancy/<refnr>")
def vacancy(refnr, *, review_error="", review_draft=None, fetch=True):
    store = _store()
    v = store.by_refnr(refnr)
    if v is None:
        store.close()
        return render_template("vacancy.html", v=None, refnr=refnr)
    try:
        if fetch:
            ensure_description(
                store,
                v,
                ttl_hours=getattr(load_criteria(), "description_ttl_hours", 168),
                fetcher=fetch_description,
            )
    except VacancyUnavailable:
        pass
    except Exception:
        marker = getattr(store, "mark_source_error", None)
        if callable(marker):
            marker(v.refnr)
    try:
        status = store.application_status(refnr)
    except Exception:
        status = ""
    try:
        applied_raw = store.applied_at(refnr) if status == "applied" else ""
    except Exception:
        applied_raw = ""
    score_details = store.score_details(refnr)
    score_history = store.scoring_history(refnr)
    review = store.vacancy_review(refnr)
    review_context = ""
    review_stale = False
    try:
        cfg = load_criteria().scoring
        current_candidate = candidate_revision(store, cfg)
        current_vacancy = vacancy_revision(v)
        review_stale = bool(review and (review["candidate_hash"] != current_candidate or review["vacancy_hash"] != current_vacancy))
        if not session.get("review_csrf"):
            session["review_csrf"] = secrets.token_urlsafe(32)
        review_context = _review_signer().dumps({"account_id": store.account_id(), "refnr": refnr,
                                                "candidate_hash": current_candidate, "vacancy_hash": current_vacancy,
                                                "csrf": session["review_csrf"]})
    except (OSError, ValueError):
        review_error = review_error or "Не удалось прочитать документы профиля. Повторите попытку после проверки файлов."
    store.close()
    message = ""
    if request.args.get("translated"):
        message = "Перевод обновлён."
    error = redact_sensitive(request.args.get("llm_error", ""))
    apply_message = (
        "Anschreiben и Lebenslauf успешно созданы."
        if request.args.get("apply_ok")
        else ""
    )
    apply_error = redact_sensitive(request.args.get("apply_error", ""))
    applied_date = ""
    if applied_raw[:10]:
        parts = applied_raw[:10].split("-")
        if len(parts) == 3:
            applied_date = ".".join(reversed(parts))
    letter_path = _ansschreiben_file(refnr)
    cv_path = _dossier_file(refnr, "Lebenslauf.pdf")
    dossier_name = (
        letter_path.parent.name if letter_path else
        (cv_path.parent.name if cv_path else "")
    )
    return render_template(
        "vacancy.html",
        v=v,
        refnr=refnr,
        message=message,
        error=error,
        application_status=status,
        score_details=score_details,
        score_history=score_history,
        review=review,
        review_form=review_draft if review_draft is not None else (review or {}),
        review_context=review_context,
        review_stale=review_stale,
        review_error=review_error,
        grade_labels=GRADE_LABELS,
        applied_date=applied_date,
        letter_path=letter_path,
        letter_label=(
            f"{dossier_name} / {letter_path.name}" if letter_path else ""
        ),
        cv_path=cv_path,
        cv_label=f"{dossier_name} / {cv_path.name}" if cv_path else "",
        apply_message=apply_message,
        apply_error=apply_error,
    )


@app.route("/vacancy/<refnr>/confirm-applied", methods=["POST"])
def confirm_applied(refnr):
    """Ручное подтверждение отправки: prepared → applied (этап 6.1)."""
    store = _store()
    try:
        if store.by_refnr(refnr) is None:
            return render_template(
                "vacancy.html", v=None, refnr=refnr
            ), 404
        store.set_status(refnr, "applied")
    finally:
        store.close()
    return redirect(url_for("vacancy", refnr=refnr))


@app.route("/vacancy/<refnr>/retranslate", methods=["POST"])
def retranslate(refnr):
    extra_env, _ = _effective_llm_env()
    store = _store()
    try:
        v = store.by_refnr(refnr)
        if v is None:
            return render_template(
                "vacancy.html", v=None, refnr=refnr
            ), 404
        if not v.description:
            return redirect(
                url_for(
                    "vacancy",
                    refnr=refnr,
                    llm_error="Перевод невозможен: описание вакансии недоступно.",
                )
            )
        try:
            _translate_vacancy(v)
            if not str(v.translation or "").strip():
                raise ValueError("Модель вернула пустой перевод.")
            store.update_translation(v.refnr, v.translation)
        except Exception as exc:
            return redirect(
                url_for(
                    "vacancy",
                    refnr=refnr,
                    llm_error=f"Перевод не удался: {redact_sensitive(exc)}"[:400],
                )
            )
    finally:
        store.close()
    return redirect(url_for("vacancy", refnr=refnr, translated=1))


@app.route("/vacancy/<refnr>/delete-translation", methods=["POST"])
def delete_translation(refnr):
    """Удаление перевода вакансии."""
    store = _store()
    try:
        if store.by_refnr(refnr) is None:
            return render_template("vacancy.html", v=None, refnr=refnr), 404
        store.update_translation(refnr, "")
        v = store.by_refnr(refnr)
    finally:
        store.close()
    return render_template(
        "vacancy.html", v=v, refnr=refnr, message="Перевод удалён."
    )


MAX_VACANCY_TEXT_CHARS = 200_000
VACANCY_EDIT_FIELDS = {
    "description": "Описание сохранено.",
    "translation": "Перевод сохранён.",
}


@app.route("/vacancy/<refnr>/edit", methods=["POST"])
def edit_vacancy_text(refnr):
    """Ручное редактирование описания вакансии и перевода."""
    field = request.form.get("field", "")
    text = request.form.get("text", "")
    store = _store()
    try:
        if store.by_refnr(refnr) is None:
            return render_template("vacancy.html", v=None, refnr=refnr), 404
        error = ""
        if field not in VACANCY_EDIT_FIELDS:
            error = "Некорректное поле для редактирования."
        elif len(text) > MAX_VACANCY_TEXT_CHARS:
            error = (
                "Текст слишком длинный — максимум "
                f"{MAX_VACANCY_TEXT_CHARS} символов."
            )
        elif field == "description":
            store.update_description(refnr, text, edited=True)
        else:
            store.update_translation(refnr, text)
        v = store.by_refnr(refnr)
    finally:
        store.close()
    return render_template(
        "vacancy.html",
        v=v,
        refnr=refnr,
        message="" if error else VACANCY_EDIT_FIELDS.get(field, ""),
        error=error,
    ), (400 if error else 200)


def _ensure_session_secret() -> None:
    """Персистентный секрет сессии без .env (аудит L1).

    Приоритет: CV_AGENT_SESSION_KEY -> data/session-key (0600, создаётся
    один раз) -> случайный на процесс (fallback, сессии живут до рестарта).
    """
    configured = os.environ.get("CV_AGENT_SESSION_KEY", "").strip()
    if configured:
        app.config["SECRET_KEY"] = configured
        return
    try:
        key_file = Path(_config.DATA_DIR) / "session-key"
        if key_file.is_file() and not key_file.is_symlink():
            stored = key_file.read_text(encoding="utf-8").strip()
            if len(stored) >= 32:
                app.config["SECRET_KEY"] = stored
                return
        key_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        new_key = secrets.token_hex(32)
        tmp_name = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=key_file.parent,
                prefix=".session-key-", delete=False,
            ) as tmp:
                tmp.write(new_key)
                tmp_name = tmp.name
            os.chmod(tmp_name, 0o600)
            os.replace(tmp_name, key_file)
        finally:
            if tmp_name and os.path.exists(tmp_name):
                os.unlink(tmp_name)
        app.config["SECRET_KEY"] = new_key
    except Exception as exc:
        print(
            f"Session key persistence failed ({redact_sensitive(exc)}); "
            "using per-process random secret.",
            file=sys.stderr,
        )


def main():
    load_env()
    store = Store()
    try:
        interrupted = store.interrupt_active_jobs()
    finally:
        store.close()
    if interrupted:
        print(
            f"Восстановление очереди: interrupted jobs: {interrupted}",
            file=sys.stderr,
        )
    _ensure_session_secret()
    app.config["SESSION_COOKIE_SECURE"] = os.environ.get(
        "CV_AGENT_SESSION_SECURE", ""
    ).lower() in {"1", "true", "yes"}
    host = os.environ.get("CV_AGENT_HOST", "127.0.0.1")
    port = int(os.environ.get("CV_AGENT_PORT", "8000"))
    if host not in {"127.0.0.1", "localhost", "::1"}:
        print(
            f"WARNUNG: CV-Agent lauscht auf {host}; локальный семейный режим "
            "не предназначен для недоверенной сети.",
            file=sys.stderr,
        )
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
