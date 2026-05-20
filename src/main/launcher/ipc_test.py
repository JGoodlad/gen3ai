import json
import os
import threading
from unittest.mock import patch

import pytest

import main.launcher.ipc as ipc


@pytest.fixture(autouse=True)
def reset_ipc():
    """Ensure ipc module state is clean before and after every test."""
    ipc._reset_for_testing()
    yield
    ipc._reset_for_testing()


def _open_pipe_with_fd():
    """Return (r_fd, w_fd) and set LAUNCHER_METRICS_FD to the write end."""
    r_fd, w_fd = os.pipe()
    return r_fd, w_fd


def _read_all(r_fd: int) -> list[dict]:
    """Close write side (EOF), read all JSON lines from r_fd."""
    with os.fdopen(r_fd, "r") as f:
        return [json.loads(line) for line in f if line.strip()]


class TestSendEventNoOp:
    def test_no_op_when_env_absent(self):
        env = {k: v for k, v in os.environ.items() if k != "LAUNCHER_METRICS_FD"}
        with patch.dict(os.environ, env, clear=True):
            ipc.send_event("hello")  # must not raise

    def test_no_op_when_fd_negative(self):
        with patch.dict(os.environ, {"LAUNCHER_METRICS_FD": "-1"}):
            ipc.send_event("hello")  # must not raise


class TestSendEvent:
    def test_writes_event_json(self):
        r_fd, w_fd = _open_pipe_with_fd()
        with patch.dict(os.environ, {"LAUNCHER_METRICS_FD": str(w_fd)}):
            ipc.send_event("child started at LR 1.23e-04")
        ipc.close()  # flushes & closes write side → EOF for reader
        lines = _read_all(r_fd)
        assert lines == [{"_event": "child started at LR 1.23e-04"}]

    def test_event_key_present(self):
        r_fd, w_fd = _open_pipe_with_fd()
        with patch.dict(os.environ, {"LAUNCHER_METRICS_FD": str(w_fd)}):
            ipc.send_event("test msg")
        ipc.close()
        lines = _read_all(r_fd)
        assert "_event" in lines[0]
        assert "_step" not in lines[0]


class TestSendMetrics:
    def test_writes_metrics_json(self):
        r_fd, w_fd = _open_pipe_with_fd()
        with patch.dict(os.environ, {"LAUNCHER_METRICS_FD": str(w_fd)}):
            ipc.send_metrics({"time/fps": 1200.0, "_step": 50000})
        ipc.close()
        lines = _read_all(r_fd)
        assert lines == [{"time/fps": 1200.0, "_step": 50000}]

    def test_no_event_key_in_metrics(self):
        r_fd, w_fd = _open_pipe_with_fd()
        with patch.dict(os.environ, {"LAUNCHER_METRICS_FD": str(w_fd)}):
            ipc.send_metrics({"x": 1.0, "_step": 0})
        ipc.close()
        lines = _read_all(r_fd)
        assert "_event" not in lines[0]


class TestMixed:
    def test_event_and_metrics_interleaved(self):
        r_fd, w_fd = _open_pipe_with_fd()
        with patch.dict(os.environ, {"LAUNCHER_METRICS_FD": str(w_fd)}):
            ipc.send_metrics({"time/fps": 900.0, "_step": 1000})
            ipc.send_event("resuming at LR 1e-04")
            ipc.send_metrics({"time/fps": 950.0, "_step": 2000})
        ipc.close()
        lines = _read_all(r_fd)
        assert len(lines) == 3
        assert "_event" not in lines[0]
        assert lines[1] == {"_event": "resuming at LR 1e-04"}
        assert "_event" not in lines[2]


class TestClose:
    def test_send_after_close_is_noop(self):
        r_fd, w_fd = _open_pipe_with_fd()
        with patch.dict(os.environ, {"LAUNCHER_METRICS_FD": str(w_fd)}):
            ipc.send_event("before close")
            ipc.close()
            ipc.send_event("after close")  # must not raise; write end is gone
        lines = _read_all(r_fd)
        assert len(lines) == 1
        assert lines[0] == {"_event": "before close"}

    def test_double_close_is_safe(self):
        r_fd, w_fd = _open_pipe_with_fd()
        with patch.dict(os.environ, {"LAUNCHER_METRICS_FD": str(w_fd)}):
            ipc.send_event("x")
            ipc.close()
            ipc.close()  # must not raise
        os.close(r_fd)


class TestThreadSafety:
    def test_concurrent_writes_do_not_corrupt(self):
        r_fd, w_fd = _open_pipe_with_fd()
        n = 20
        with patch.dict(os.environ, {"LAUNCHER_METRICS_FD": str(w_fd)}):
            threads = [
                threading.Thread(target=ipc.send_event, args=(f"msg-{i}",))
                for i in range(n)
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
        ipc.close()
        lines = _read_all(r_fd)
        assert len(lines) == n
        msgs = {l["_event"] for l in lines}
        assert msgs == {f"msg-{i}" for i in range(n)}
