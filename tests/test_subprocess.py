import os
import sys
import time

import pytest

from cv_agent.webapp import CommandTimedOut, _stream_command


def test_stream_command_returns_output_and_exit_code():
    lines = []
    code = _stream_command(
        [sys.executable, "-c", "print('ready', flush=True)"],
        dict(os.environ),
        lines.append,
        timeout=2,
        terminate_grace=0.1,
    )
    assert code == 0
    assert lines == ["ready\n"]


def test_stream_command_timeout_kills_and_reaps_child():
    lines = []
    script = (
        "import os,signal,time; "
        "signal.signal(signal.SIGTERM, lambda *_: None); "
        "print(os.getpid(), flush=True); time.sleep(30)"
    )
    started = time.monotonic()
    with pytest.raises(CommandTimedOut, match="допустимое время"):
        _stream_command(
            [sys.executable, "-c", script],
            dict(os.environ),
            lines.append,
            timeout=0.5,
            terminate_grace=0.1,
        )
    assert time.monotonic() - started < 3
    child_pid = int(lines[0])
    with pytest.raises(ChildProcessError):
        os.waitpid(child_pid, os.WNOHANG)
