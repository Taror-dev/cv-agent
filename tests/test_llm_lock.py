import threading
import time

from cv_agent.llm import LLMLock


def test_llm_lock_serializes(tmp_path):
    acquired = threading.Event()
    release = threading.Event()
    errors = []

    def holder():
        try:
            with LLMLock():
                acquired.set()
                release.wait(timeout=5)
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    t = threading.Thread(target=holder)
    t.start()
    assert acquired.wait(timeout=5), "первый поток не получил блокировку"

    got_lock = threading.Event()

    def waiter():
        with LLMLock():
            got_lock.set()

    t2 = threading.Thread(target=waiter)
    t2.start()
    time.sleep(0.3)
    assert not got_lock.is_set(), "второй поток получил блокировку раньше времени"

    release.set()
    t.join(timeout=5)
    t2.join(timeout=5)
    assert got_lock.is_set(), "второй поток не дождался очереди"
    assert not errors