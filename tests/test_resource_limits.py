import multiprocessing
import threading
import time
from pathlib import Path

from cv_agent.resource_limits import FileSlotPool
from cv_agent.store import Store


def _hold_process_slot(data_dir, acquired, release):
    import cv_agent.resource_limits as limits

    limits.DATA_DIR = Path(data_dir)
    with limits.FileSlotPool(
        "network", "cross-process", 1, poll_interval=0.01
    ).lease():
        acquired.set()
        release.wait(timeout=5)


def _steps():
    return [{"cmd": "search", "args": ["search"]}]


def test_file_slot_pool_limits_different_processes(monkeypatch, tmp_path):
    import cv_agent.resource_limits as limits

    monkeypatch.setattr(limits, "DATA_DIR", tmp_path)
    context = multiprocessing.get_context("fork")
    acquired = context.Event()
    release = context.Event()
    process = context.Process(
        target=_hold_process_slot,
        args=(str(tmp_path), acquired, release),
    )
    process.start()
    assert acquired.wait(timeout=2)
    timer = threading.Timer(0.2, release.set)
    timer.start()
    try:
        with FileSlotPool(
            "network", "cross-process", 1, poll_interval=0.01
        ).lease() as measurement:
            pass
    finally:
        release.set()
        timer.join(timeout=2)
        process.join(timeout=2)
    assert process.exitcode == 0
    assert measurement.wait_seconds >= 0.15


def test_resource_waits_are_persisted_in_current_job(monkeypatch, tmp_path):
    import cv_agent.resource_limits as limits

    monkeypatch.setattr(limits, "DATA_DIR", tmp_path)
    store = Store()
    run_id, _ = store.create_job("pipeline", _steps())
    assert store.claim_job(run_id)
    store.close()
    monkeypatch.setenv("CV_AGENT_RUN_ID", str(run_id))

    with FileSlotPool("network", "metrics", 1).lease():
        pass
    with FileSlotPool("llm", "metrics", 1).lease():
        pass

    check = Store()
    try:
        metrics = check.job(run_id)["metrics"]
    finally:
        check.close()
    assert metrics["network_requests"] == 1
    assert metrics["llm_requests"] == 1
    assert metrics["network_wait_seconds"] >= 0
    assert metrics["llm_wait_seconds"] >= 0


def test_different_backend_pools_do_not_block_each_other(monkeypatch, tmp_path):
    import cv_agent.resource_limits as limits

    monkeypatch.setattr(limits, "DATA_DIR", tmp_path)
    first = FileSlotPool("llm", "backend-a", 1)
    second = FileSlotPool("llm", "backend-b", 1)
    with first.lease():
        started = time.monotonic()
        with second.lease() as measurement:
            pass
    assert time.monotonic() - started < 0.2
    assert measurement.wait_seconds < 0.2
