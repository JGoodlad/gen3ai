"""Unit tests for the launcher's headless (no-TTY) mode.

**The bug this exists to prevent.** A detached launch (`nohup … < /dev/null &`, systemd,
cron, an agent's background shell) leaves stdin on /dev/null. Textual's input thread then
busy-loops a whole core *forever*: an fd at EOF is permanently readable, so
`selector.select(0.1)` returns instantly, `os.read` yields b"", and the
`if not unicode_data: break` inside `run_input_thread` breaks only the inner `for` — the
outer `while` spins at full speed.

Measured on a live 15 h production run (2026-08-14): **96% of a core**, 13 h 34 m of CPU
burned by that one thread, and a **982 MB** launcher log of full-screen ANSI repaints
growing at 17 KB/s because the "screen" was a redirected file. A standalone A/B of a
two-line Textual app confirmed the cause was stdin alone — /dev/null 98% of a core, a real
pty 0%, headless 0% and 0 bytes of stdout.

Nothing here can assert on Textual's thread directly, so the gate is the decision:
`headless_mode()` must return True exactly when there is no TTY on stdin. Get that wrong
and the spin returns silently — it costs a core and a gigabyte a day while the run still
*looks* healthy, which is why it went unnoticed for a full run.
"""

import os
import subprocess
import sys

from main.launcher.run import headless_mode
from main.launcher.state import LauncherState

_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class _FakeStream:
    def __init__(self, tty: bool):
        self._tty = tty

    def isatty(self) -> bool:
        return self._tty


class _RaisingStream:
    def isatty(self) -> bool:
        raise ValueError("I/O operation on closed file")


# --- the decision ---

def test_tty_keeps_the_interactive_tui():
    assert headless_mode(_FakeStream(tty=True)) is False


def test_no_tty_goes_headless():
    """The production failure: nohup … < /dev/null."""
    assert headless_mode(_FakeStream(tty=False)) is True


def test_closed_stdin_goes_headless():
    """A raising isatty() is the detached case too — must not propagate."""
    assert headless_mode(_RaisingStream()) is True


def test_devnull_stdin_goes_headless_for_real():
    """Same decision against a real /dev/null handle, not a stub."""
    with open(os.devnull) as fh:
        assert headless_mode(fh) is True


def test_default_reads_stdin():
    """Called with no argument it must consult sys.stdin, not a stale default."""
    saved = sys.stdin
    try:
        sys.stdin = _FakeStream(tty=True)
        assert headless_mode() is False
        sys.stdin = _FakeStream(tty=False)
        assert headless_mode() is True
    finally:
        sys.stdin = saved


# --- the event sink that keeps a headless run followable ---

def test_event_sink_receives_every_event():
    state = LauncherState(interval_hours=3.0)
    seen: list = []
    state.event_sink = seen.append
    state.add_event("child started")
    state.add_event("checkpoint saved")
    assert len(seen) == 2
    assert "child started" in seen[0] and "checkpoint saved" in seen[1]
    assert seen[0].startswith("["), "events must keep their [HH:MM:SS] stamp"


def test_events_still_recorded_without_a_sink():
    """The sink is additive — the in-memory events list must be unaffected."""
    state = LauncherState(interval_hours=3.0)
    state.add_event("no sink here")
    assert any("no sink here" in e for e in state.snapshot().events)


def test_sink_failure_never_breaks_add_event():
    """A broken sink must not take down the supervisor that was only logging."""
    state = LauncherState(interval_hours=3.0)

    def _boom(_line: str) -> None:
        raise OSError("disk full")

    state.event_sink = _boom
    state.add_event("still recorded")  # must not raise
    assert any("still recorded" in e for e in state.snapshot().events)


def test_sink_is_called_outside_the_lock():
    """A sink doing I/O must not hold the state lock — it would stall every reader
    thread and the supervisor behind a slow write."""
    state = LauncherState(interval_hours=3.0)
    reentered: list = []

    def _reenter(_line: str) -> None:
        # If add_event still held the lock, this would deadlock rather than record.
        reentered.append(state.snapshot().pid)

    state.event_sink = _reenter
    state.add_event("reentrant")
    assert len(reentered) == 1


# --- the end-to-end property, measured ---

def test_headless_textual_app_does_not_spin_on_devnull_stdin():
    """The actual regression, end to end: a Textual app with stdin on /dev/null must
    not burn CPU when run headless. Without headless this same app measured ~98% of a
    core; the bound here is deliberately loose (30%) so it fails only on a real spin.
    """
    prog = (
        "import time\n"
        "from textual.app import App, ComposeResult\n"
        "from textual.widgets import Static\n"
        "class T(App):\n"
        "    def compose(self) -> ComposeResult:\n"
        "        yield Static('x')\n"
        "    def on_mount(self):\n"
        "        self.set_timer(3.0, self.exit)\n"
        "def cpu():\n"
        "    p = open('/proc/self/stat').read().split()\n"
        "    return (int(p[13]) + int(p[14])) / 100.0\n"
        "t0, c0 = time.monotonic(), cpu()\n"
        "T().run(headless=True)\n"
        "w, c = time.monotonic() - t0, cpu() - c0\n"
        "print(round(100 * c / w, 1))\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = _SRC + os.pathsep + env.get("PYTHONPATH", "")
    with open(os.devnull) as devnull:
        proc = subprocess.run(
            [sys.executable, "-c", prog],
            stdin=devnull, capture_output=True, text=True, env=env, timeout=120,
        )
    assert proc.returncode == 0, proc.stderr
    pct = float(proc.stdout.strip().splitlines()[-1])
    assert pct < 30.0, f"headless app burned {pct}% of a core — the input thread is spinning again"
