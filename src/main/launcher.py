"""Process-level restart loop for train_rl_agent.py.

Periodically SIGTERMs the training child and relaunches from the most recently
saved checkpoint, recovering FPS lost to pymalloc fragmentation over long runs.
Includes a Rich TUI dashboard fed by metrics delivered via a pipe fd from the
child's MetricsExporterCallback.
"""

import glob
import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

from rich.live import Live

from main.launcher_ui import LauncherState, LauncherUI

_PYTHON = "/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3"
_TRAIN_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_rl_agent.py")
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ── Pure utility functions ────────────────────────────────────────────────────

def find_latest_checkpoint(models_root: str) -> "str | None":
    zips = glob.glob(os.path.join(models_root, "**", "*.zip"), recursive=True)
    return max(zips, key=os.path.getmtime) if zips else None


def _insert_or_replace_model_arg(args: list, checkpoint: str) -> list:
    out = []
    i = 0
    replaced = False
    while i < len(args):
        if args[i] == "--model":
            out.extend(["--model", checkpoint])
            replaced = True
            i += 2
        else:
            out.append(args[i])
            i += 1
    if not replaced:
        out.extend(["--model", checkpoint])
    return out


def _strip_launcher_arg(argv: list) -> list:
    out = []
    i = 0
    while i < len(argv):
        if argv[i] == "--restart-interval-hours":
            i += 2
        elif argv[i].startswith("--restart-interval-hours="):
            i += 1
        else:
            out.append(argv[i])
            i += 1
    return out


# ── Git ───────────────────────────────────────────────────────────────────────

def _git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


# ── IPC: pipe readers (run in daemon threads) ─────────────────────────────────

def _read_metrics_pipe(fd_r: int, state: LauncherState) -> None:
    try:
        with os.fdopen(fd_r, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    state.update_metrics(json.loads(line))
                except json.JSONDecodeError:
                    pass
    except Exception:
        pass


def _read_child_stdout(proc: subprocess.Popen, state: LauncherState) -> None:
    try:
        for raw in proc.stdout:
            state.add_log(raw.decode(errors="replace").rstrip())
    except Exception:
        pass


# ── Key input ─────────────────────────────────────────────────────────────────

def _setup_raw_input() -> None:
    """Switch stdin to cbreak (single-keypress, no echo). Restored via atexit.

    Works in tmux — tmux provides a real pty for each pane so isatty() is True.
    Falls back gracefully when stdin is a pipe (tests, CI, redirected input).
    """
    if not sys.stdin.isatty():
        return
    try:
        import atexit
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


# ── Command dispatch (isolated for testability) ───────────────────────────────

@dataclass
class _PollFlags:
    sigterm_sent: bool = False
    quit_requested: bool = False
    restart_requested: bool = False


def _dispatch_command(
    ch: str,
    proc: subprocess.Popen,
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
        state.add_event("👋 Quit requested — waiting for child to save…")
        flags.quit_requested = True
        if not flags.sigterm_sent:
            try:
                os.kill(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            flags.sigterm_sent = True

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


# ── Child lifecycle ───────────────────────────────────────────────────────────

def _build_child_env() -> dict:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (_SRC_DIR + ":" + existing) if existing else _SRC_DIR
    return env


def _launch_child(
    child_args: list, child_env: dict, state: LauncherState
) -> tuple:
    """Create metrics pipe, spawn child, start reader threads.

    Returns (proc, metrics_r_fd). The metrics pipe read-fd is owned by the
    _read_metrics_pipe daemon thread and will be closed when the child exits.
    """
    metrics_r, metrics_w = os.pipe()
    proc = subprocess.Popen(
        [_PYTHON, _TRAIN_SCRIPT] + child_args,
        env={**child_env, "LAUNCHER_METRICS_FD": str(metrics_w)},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        pass_fds=(metrics_w,),
        bufsize=1,
    )
    os.close(metrics_w)  # parent closes write end after fork

    state.pid = proc.pid
    state.run_start = time.monotonic()

    threading.Thread(target=_read_metrics_pipe, args=(metrics_r, state), daemon=True).start()
    threading.Thread(target=_read_child_stdout, args=(proc, state), daemon=True).start()

    return proc


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(child_args: list, interval_hours: float) -> None:
    interval_seconds = interval_hours * 3600
    child_env = _build_child_env()
    state = LauncherState(interval_hours=interval_hours)
    ui = LauncherUI()

    state.initial_git_hash = _git_hash()

    cmd_q: queue.Queue = queue.Queue()
    _setup_raw_input()
    threading.Thread(target=_read_keys, args=(cmd_q,), daemon=True).start()

    if interval_hours > 0:
        state.add_event(f"🚀 Starting — restart every {interval_hours:.1f}h")
    else:
        state.add_event("🚀 Starting — single run (no restart)")

    with Live(refresh_per_second=2, screen=True) as live:
        while True:

            # ── Git check before each restart (skip on first launch) ───────
            if state.restart_count > 0:
                current_hash = _git_hash()
                if current_hash != state.initial_git_hash:
                    state.pending_restart_git_hash = current_hash
                    state.view_mode = "confirm_restart"
                    while state.view_mode == "confirm_restart":
                        live.update(ui.render(state.snapshot()))
                        time.sleep(0.25)
                        while not cmd_q.empty():
                            ch = cmd_q.get_nowait()
                            if ch == "y":
                                state.view_mode = "dashboard"
                                state.pending_restart_git_hash = None
                                state.add_event(
                                    f"✅ Git change acknowledged ({current_hash}) — restarting"
                                )
                            elif ch == "q":
                                sys.exit(0)

            # ── Launch child ───────────────────────────────────────────────
            proc = _launch_child(child_args, child_env, state)
            state.add_event(f"✅ Child PID {proc.pid} started")

            deadline = state.run_start + interval_seconds if interval_hours > 0 else float("inf")
            state.deadline = deadline

            flags = _PollFlags()

            # ── Poll loop ──────────────────────────────────────────────────
            while True:
                live.update(ui.render(state.snapshot()))

                now = time.monotonic()
                remaining = deadline - now

                if not flags.sigterm_sent and remaining <= 0:
                    state.add_event(f"⏰ {interval_hours:.1f}h elapsed — restarting child…")
                    try:
                        os.kill(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    flags.sigterm_sent = True

                poll_timeout = min(0.5, max(0.01, remaining if remaining > 0 else 0.5))
                try:
                    proc.wait(timeout=poll_timeout)
                    break
                except subprocess.TimeoutExpired:
                    pass

                while not cmd_q.empty():
                    _dispatch_command(
                        cmd_q.get_nowait(), proc, state, flags, deadline, interval_hours
                    )

            proc.wait()
            live.update(ui.render(state.snapshot()))

            if proc.returncode != 0:
                state.add_event(f"🛑 Child crashed (exit {proc.returncode}) — not restarting")
                live.update(ui.render(state.snapshot()))
                time.sleep(2)
                sys.exit(proc.returncode)

            if flags.quit_requested:
                state.add_event("👋 Quit complete.")
                live.update(ui.render(state.snapshot()))
                time.sleep(1)
                sys.exit(0)

            if interval_hours <= 0 and not flags.restart_requested:
                sys.exit(0)

            checkpoint = find_latest_checkpoint("models")
            if checkpoint is None:
                state.add_event("🛑 No checkpoint found under models/ — cannot restart")
                live.update(ui.render(state.snapshot()))
                time.sleep(2)
                sys.exit(1)

            state.add_event(f"✅ Restarting from {os.path.basename(checkpoint)}")
            child_args = _insert_or_replace_model_arg(child_args, checkpoint)
            state.restart_count += 1
            state.view_mode = "dashboard"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Launcher: wraps train_rl_agent.py with periodic full-process restart.",
        add_help=False,
    )
    parser.add_argument(
        "--restart-interval-hours",
        type=float,
        default=3.0,
        help="Restart the training process every N hours (0 = run once)",
    )
    parser.add_argument("-h", "--help", action="store_true")

    known, _ = parser.parse_known_args()

    if known.help:
        parser.print_help()
        print("\nAll other arguments are forwarded to train_rl_agent.py.")
        sys.exit(0)

    child_args = _strip_launcher_arg(sys.argv[1:])
    run(child_args, interval_hours=known.restart_interval_hours)


if __name__ == "__main__":
    main()
