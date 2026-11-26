"""Eigenständiger Prozess für DB-backed Web-Jobs."""

from __future__ import annotations

import argparse
import os
import sys

from .store import Store
from .webapp import WORKER_ENV_KEYS, _pipeline_worker


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="CV-Agent DB job worker")
    parser.add_argument("--run-id", required=True, type=int)
    parser.add_argument("--account-id", required=True)
    args = parser.parse_args(argv)

    store = Store()
    try:
        user = store.account_name(args.account_id)
    finally:
        store.close()
    if not user:
        print("Worker profile not found.", file=sys.stderr)
        return 2
    # Web кладёт в наш процесс только allowlisted ключи; пробрасываем их
    # дальше явно (в _run_cli базовый env санитизируется), чтобы CLI-потомок
    # выполнялся в контексте инициировавшего профиля.
    extra_env = {k: v for k, v in os.environ.items() if k in WORKER_ENV_KEYS}
    extra_env["CV_AGENT_USER"] = user
    _pipeline_worker(args.run_id, user, extra_env)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
