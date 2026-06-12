"""Command dispatch for the launcher.

The launcher's only UI is Textual, which feeds keystrokes through its key bindings (see
``app.py``); this module just turns a dispatched command char into the right state mutation
or child signal. ``_dispatch_command`` is shared with the supervisor in ``run.py``.
"""

import os
import signal
import threading
import time
from dataclasses import dataclass

from main.launcher.state import LauncherState


@dataclass
class _PollFlags:
    sigterm_sent: bool = False
    quit_requested: bool = False
    restart_requested: bool = False
    # Set when the LAUNCHER kills the child for stall/deadline (not a user quit or a
    # self-crash) — the run must then restart from checkpoint regardless of the exit
    # code (a hung child ignores SIGTERM and gets SIGKILL'd, so rc won't be INTERRUPTED).
    forced_restart: bool = False
    sigterm_at: float = 0.0   # monotonic time SIGTERM was sent (for SIGKILL escalation)
    sigkill_sent: bool = False


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
            flags.sigterm_at = time.monotonic()  # arms the SIGKILL escalation

    elif ch == "c":
        try:
            os.kill(proc.pid, signal.SIGUSR1)
            state.add_event("💾 Checkpoint signal sent")
        except ProcessLookupError:
            state.add_event("💾 Child already exited")

    elif ch == "f":
        # Force an off-cadence eval cycle. The child decides accept-vs-reject (it owns the
        # authoritative "eval already running" state) and reports the outcome back via the
        # event pipe — so this just sends the signal.
        try:
            os.kill(proc.pid, signal.SIGUSR2)
            state.add_event("⚡ Force-eval signal sent")
        except ProcessLookupError:
            state.add_event("⚡ Child already exited")

    elif ch == "q":
        if state.view_mode == "dashboard":
            state.view_mode = "confirm_quit"

    elif ch == "l":
        state.view_mode = "logs"

    elif ch == "e":
        state.view_mode = "events"

    elif ch == "d":
        state.view_mode = "dashboard"

    elif ch == "p":
        if state.run_dir:
            state.add_event("📊 Generating plots…")
            threading.Thread(
                target=_run_plots, args=(state,), daemon=True
            ).start()
        else:
            state.add_event("⚠️  No run dir yet — can't generate plots")

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


def _run_plots(state: LauncherState) -> None:
    """Background thread: generate plots and report back via events."""
    try:
        from utils.plot_tb import find_tb_dir_for_run, generate_plots

        tb_dir = find_tb_dir_for_run(state.run_dir)
        if tb_dir is None:
            state.add_event("⚠️  TensorBoard dir not found for this run")
            return

        out_dir = os.path.join(state.run_dir, "tb_imgs")
        paths = generate_plots(tb_dir, out_dir)

        if paths:
            state.add_event(f"📊 Plots saved → {out_dir}  ({len(paths)} files)")
        else:
            state.add_event("⚠️  No scalar data found — nothing written")
    except Exception as exc:
        state.add_event(f"❌ Plot error: {exc}")
