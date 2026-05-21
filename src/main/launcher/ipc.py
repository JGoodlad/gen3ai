"""Shared pipe writer for child→launcher IPC.

Centralises access to LAUNCHER_METRICS_FD so all callers share one file
object.  Safe to import and call outside the launcher (LAUNCHER_METRICS_FD
absent) — every function is a no-op in that case.
"""

import json
import os
import threading

_lock = threading.Lock()
_pipe = None
_pipe_initialised = False


def _get_pipe():
    global _pipe, _pipe_initialised
    if _pipe_initialised:
        return _pipe
    with _lock:
        if _pipe_initialised:
            return _pipe
        raw = os.environ.get("LAUNCHER_METRICS_FD", "")
        if raw.lstrip("-").isdigit():
            fd = int(raw)
            if fd >= 0:
                try:
                    _pipe = os.fdopen(fd, "w", buffering=1)
                except OSError:
                    pass
        _pipe_initialised = True
    return _pipe


def _write(payload: dict) -> None:
    pipe = _get_pipe()
    if pipe is None:
        return
    with _lock:
        try:
            pipe.write(json.dumps(payload) + "\n")
        except (OSError, BrokenPipeError):
            pass


def init() -> None:
    """Eagerly open the pipe. Call at startup so close() can flush at shutdown."""
    _get_pipe()


def send_event(msg: str) -> None:
    """Post a message to the launcher Events panel."""
    _write({"_event": msg})


def send_metrics(payload: dict) -> None:
    """Send a metrics payload to the launcher dashboard."""
    _write(payload)


def emit(msg: str) -> None:
    """Send msg as a launcher event, or print() when running standalone."""
    if _get_pipe() is not None:
        send_event(msg)
    else:
        print(msg)


def close() -> None:
    global _pipe, _pipe_initialised
    with _lock:
        if _pipe is not None:
            try:
                _pipe.close()
            except OSError:
                pass
            _pipe = None
        _pipe_initialised = False


def _reset_for_testing() -> None:
    """Reset all module-level state. Call from test teardown only."""
    global _pipe, _pipe_initialised
    with _lock:
        if _pipe is not None:
            try:
                _pipe.close()
            except OSError:
                pass
            _pipe = None
        _pipe_initialised = False
