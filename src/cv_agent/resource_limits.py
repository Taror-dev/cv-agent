"""Cross-process concurrency pools und messbare Job-Wartezeiten."""

from __future__ import annotations

import fcntl
import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path

from .config import DATA_DIR, load_runtime_limits


@dataclass(frozen=True)
class ResourceMeasurement:
    resource: str
    wait_seconds: float


class ResourceLease:
    def __init__(self, pool: "FileSlotPool"):
        self.pool = pool
        self._file = None
        self.measurement: ResourceMeasurement | None = None

    def __enter__(self) -> ResourceMeasurement:
        started = time.monotonic()
        self._file = self.pool._acquire_file()
        self.measurement = ResourceMeasurement(
            self.pool.resource,
            max(0.0, time.monotonic() - started),
        )
        _record_job_usage(self.measurement)
        return self.measurement

    def __exit__(self, *exc) -> None:
        if self._file is None:
            return
        fcntl.flock(self._file, fcntl.LOCK_UN)
        self._file.close()
        self._file = None


class FileSlotPool:
    """Ein flock-Datei-Slot pro erlaubtem parallelem Request."""

    def __init__(
        self,
        resource: str,
        identity: str,
        slots: int,
        *,
        poll_interval: float = 0.05,
    ):
        if resource not in {"network", "llm", "embedding"}:
            raise ValueError("Unbekannte Ressource")
        if isinstance(slots, bool) or not 1 <= slots <= 32:
            raise ValueError("Concurrency must be between 1 and 32")
        self.resource = resource
        self.slots = slots
        self.poll_interval = max(0.001, float(poll_interval))
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        self.directory = Path(DATA_DIR) / "limits" / f"{resource}-{digest}"

    def lease(self) -> ResourceLease:
        return ResourceLease(self)

    def _acquire_file(self):
        self.directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        while True:
            for index in range(self.slots):
                path = self.directory / f"slot-{index}.lock"
                fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
                handle = os.fdopen(fd, "a+")
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return handle
                except BlockingIOError:
                    handle.close()
            time.sleep(self.poll_interval)


def network_request_limit() -> ResourceLease:
    limits = load_runtime_limits()
    return FileSlotPool(
        "network", "shared-http", limits.network_concurrency
    ).lease()


def llm_backend_limit(backend_key: str) -> ResourceLease:
    limits = load_runtime_limits()
    return FileSlotPool(
        "llm", backend_key, limits.llm_backend_concurrency
    ).lease()


def _record_job_usage(measurement: ResourceMeasurement) -> None:
    raw_run_id = os.environ.get("CV_AGENT_RUN_ID", "")
    if not raw_run_id.isdigit():
        return
    try:
        from .store import Store

        store = Store()
        try:
            store.record_job_resource_usage(
                int(raw_run_id),
                measurement.resource,
                measurement.wait_seconds,
            )
        finally:
            store.close()
    except Exception:
        # Metriken dürfen den eigentlichen Netzwerk-/LLM-Aufruf nie blockieren.
        return
