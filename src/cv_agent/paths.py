"""Проверка имён профилей и построение account-scoped путей."""

from __future__ import annotations

import re
import unicodedata
import uuid
from pathlib import Path


ACCOUNT_NAME_RE = re.compile(r"^[A-Za-zА-Яа-яЁё0-9 _\-.]+$")
MAX_ACCOUNT_NAME_LENGTH = 50
ACCOUNT_STORAGE_KINDS = frozenset({"cv", "applications", "tmp"})


def normalize_account_name(name: str) -> str:
    return unicodedata.normalize("NFC", str(name)).strip()


def valid_account_name(name: str) -> bool:
    normalized = normalize_account_name(name)
    return bool(
        normalized
        and normalized not in {".", ".."}
        and len(normalized) <= MAX_ACCOUNT_NAME_LENGTH
        and ACCOUNT_NAME_RE.fullmatch(normalized)
        and "/" not in normalized
        and "\\" not in normalized
    )


def account_directory(root: Path, account_name: str) -> Path:
    """Возвращает безопасный прямой дочерний каталог профиля внутри root."""
    normalized = normalize_account_name(account_name)
    if not valid_account_name(normalized):
        raise ValueError("Недопустимое имя профиля")
    resolved_root = root.resolve()
    target = (resolved_root / normalized).resolve()
    if target.parent != resolved_root:
        raise ValueError("Путь профиля выходит за разрешённый каталог")
    return target


def safe_filename(name: str) -> str:
    """Имя upload/delete без каталогов и специальных path-компонентов."""
    raw = str(name)
    value = Path(raw).name
    if (
        not value
        or value in {".", ".."}
        or value != raw
        or "/" in raw
        or "\\" in raw
    ):
        raise ValueError("Недопустимое имя файла")
    return value


def account_path(data_dir: Path, account_id: str, kind: str) -> Path:
    """Единственный resolver UUID-каталогов профиля с containment check."""
    if kind not in ACCOUNT_STORAGE_KINDS:
        raise ValueError("Недопустимый тип каталога профиля")
    try:
        normalized_id = str(uuid.UUID(str(account_id)))
    except (TypeError, ValueError, AttributeError) as exc:
        raise ValueError("Недопустимый account_id") from exc
    data_root = Path(data_dir).resolve()
    accounts_entry = data_root / "accounts"
    if accounts_entry.is_symlink():
        raise ValueError("Путь accounts не может быть символической ссылкой")
    accounts_root = accounts_entry.resolve()
    if accounts_root.parent != data_root:
        raise ValueError("Путь accounts выходит за data root")
    profile_root = (accounts_root / normalized_id).resolve()
    if profile_root.parent != accounts_root:
        raise ValueError("Путь профиля выходит за accounts root")
    target = (profile_root / kind).resolve()
    if target.parent != profile_root:
        raise ValueError("Путь данных выходит за профиль")
    return target


def ensure_account_path(data_dir: Path, account_id: str, kind: str) -> Path:
    """Создаёт весь UUID-путь и принудительно задаёт 0700 каждому каталогу."""
    target = account_path(data_dir, account_id, kind)
    accounts_root = target.parents[1]
    profile_root = target.parent
    for folder in (accounts_root, profile_root, target):
        folder.mkdir(parents=True, exist_ok=True, mode=0o700)
        folder.chmod(0o700)
    return target
