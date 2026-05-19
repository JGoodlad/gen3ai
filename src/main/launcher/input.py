"""Keyboard input handling and command dispatch for the launcher."""

import atexit
import os
import queue
import signal
import sys
import time
from dataclasses import dataclass

from main.launcher.state import LauncherState


@dataclass
class _PollFlags:
    sigterm_sent: bool = False
    quit_requested: bool = False
    restart_requested: bool = False


def _setup_raw_input() -> None:
    """Switch stdin to cbreak (single-keypress, no echo). Restored via atexit.

    Works in tmux — tmux provides a real pty for each pane so isatty() is True.
    Falls back gracefully when stdin is a pipe (tests, CI, redirected input).
    """
    if not sys.stdin.isatty():
        return
    try:
        import termios
        import tty
        fd = sys.stdin.fileno()
        old = termios.tcgetattr(fd)
        tty.setcbreak(fd)
        atexit.register(_restore_tty, fd, old)
    except Exception:
        pass


def _restore_tty(fd: int, old_settings) -> None:
    try:
        import termios
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except Exception:
        pass


def _read_keys(cmd_q: queue.Queue) -> None:
    """Reads single chars from stdin into cmd_q. Tty mode is set by _setup_raw_input()."""
    try:
        if sys.stdin.isatty():
            while True:
                ch = sys.stdin.read(1)
                if ch:
                    cmd_q.put(ch.lower())
        else:
            # Piped stdin: take first char of each line.
            for line in sys.stdin:
                stripped = line.strip().lower()
                if stripped:
                    cmd_q.put(stripped[0])
    except Exception:
        pass


def _dispatch_command(
    ch: str,
    proc,
    state: LauncherState,
    flags: _PollFlags,
    deadline: float,
    interval_hours: float,
) -> None:
    """Handle a single keypress. Mutates flags and state in-place."""
    if ch == "r":
        state.add_event("♻️  Restart requested")
        flags.restart_requested = True
        if not flags.sigterm_sent:
            try:
                os.kill(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            flags.sigterm_sent = True

    elif ch == "c":
        try:
            os.kill(proc.pid, signal.SIGUSR1)
            state.add_event("💾 Checkpoint signal sent")
        except ProcessLookupError:
            state.add_event("💾 Child already exited")

    elif ch == "q":
        if state.view_mode == "dashboard":
            state.view_mode = "confirm_quit"

    elif ch == "l":
        state.view_mode = "logs"

    elif ch == "d":
        state.view_mode = "dashboard"

    elif ch == "s":
        now = time.monotonic()
        elapsed = now - state.run_start
        if interval_hours > 0 and deadline < float("inf"):
            remaining = max(0.0, deadline - now)
            state.add_event(
                f"📊 PID {proc.pid} | elapsed {elapsed / 3600:.2f}h "
                f"| restart in {remaining / 3600:.2f}h"
            )
        else:
            state.add_event(f"📊 PID {proc.pid} | elapsed {elapsed / 3600:.2f}h | no restart")
