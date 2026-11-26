"""Небольшие общие функции защиты вывода и сообщений об ошибках."""

from __future__ import annotations

import re
from collections.abc import Iterable


_SENSITIVE_PATTERNS = (
    re.compile(r"AIza[0-9A-Za-z_-]{10,}"),
    re.compile(r"\bgsk[_-][0-9A-Za-z_-]{10,}\b"),
    re.compile(r"\bsk-[0-9A-Za-z_-]{8,}\b"),
    re.compile(r"(?i)\b(?:api[_-]?key|token|authorization)\s*[=:]\s*[^\s,;]+"),
    re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I),
    # телефоны: >=9 цифр с типичными разделителями; lookahead не трогает
    # даты dd.mm.yyyy (8 цифр), чтобы логи сохраняли footer dates (аудит M3).
    re.compile(
        r"(?<!\w)(?!\d{2}[.\-/]\d{2}[.\-/]\d{4})\+?\d(?:[\s()./-]*\d){9,}(?!\w)"
    ),
)


def redact_sensitive(value: object, secrets: Iterable[str] = ()) -> str:
    """Удаляет известные ключи и типичные credentials/контакты из log text."""
    text = str(value)
    for secret in secrets:
        if secret and len(secret) >= 4:
            text = text.replace(secret, "[REDACTED]")
    for pattern in _SENSITIVE_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text
