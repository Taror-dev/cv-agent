#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PYTHON=.venv/bin/python
STAMP=.venv/.deps-stamp

# Хеш зависимостей: при изменении pyproject.toml или constraints.txt venv
# переустанавливается, а не остаётся устаревшей (этап 7.2 дорожной карты).
CURRENT_HASH="$(cat pyproject.toml constraints.txt 2>/dev/null | sha256sum | cut -d' ' -f1)"

if [ ! -x "$PYTHON" ] || [ ! -f "$STAMP" ] || [ "$(cat "$STAMP" 2>/dev/null)" != "$CURRENT_HASH" ]; then
    if [ ! -x "$PYTHON" ]; then
        echo "Venv nicht gefunden. Setup läuft:"
        python3 -m venv .venv
    else
        echo "Abhängigkeiten haben sich geändert. Aktualisiere venv…"
    fi
    .venv/bin/pip install --upgrade pip >/dev/null
    if [ -f constraints.txt ]; then
        .venv/bin/pip install -e ".[dev]" -c constraints.txt
    else
        .venv/bin/pip install -e ".[dev]"
    fi
    printf '%s' "$CURRENT_HASH" > "$STAMP"
fi

exec "$PYTHON" -m cv_agent.cli "$@"
