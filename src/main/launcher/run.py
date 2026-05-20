"""Main restart loop and session summary for the training launcher."""

import atexit
import os
import queue
import signal
import subprocess
import sys
import threading
import time

from rich.live import Live

from main.exit_codes import TrainExitCode
from main.launcher.checkpoint import (
    find_latest_checkpoint,
    _find_model_arg,
    _insert_or_replace_model_arg,
    _insert_or_replace_run_dir_arg,
)
from main.launcher.child import _build_child_env, _launch_child, _TRAIN_SCRIPT, _SRC_DIR
from main.launcher.input import _PollFlags, _dispatch_command, _read_keys, _setup_raw_input
from main.launcher.state import LauncherState
from main.launcher.ui import LauncherUI
from main.launcher.worktree import (
    _git_hash,
    _prune_stale_launcher_worktrees,
    _read_checkpoint_git_hash,
    _create_run_worktree,
    get_git_hash,
    get_repo_root,
)


def _print_crash_log(log_lines: "list | None") -> None:
    """Dump captured child output to stderr after the Live screen has closed."""
    if not log_lines:
        return
    print("\n── Child output (last 100 lines) ─────────────────────────────", file=sys.stderr)
    for line in log_lines[-100:]:
        print(line, file=sys.stderr)
    print("──────────────────────────────────────────────────────────────", file=sys.stderr)


def _print_exit_summary(run_dir: "str | None", state: LauncherState) -> None:
    snap = state.snapshot()
    lines = ["", "── Training session ended ──────────────────────────────────────"]
    if run_dir:
        lines.append(f"  Run folder : {run_dir}")
    lines.append(f"  Restarts   : {snap.restart_count}")
    if snap.metrics_step:
        lines.append(f"  Last step  : {snap.metrics_step:,}")
    if "rollout/ep_rew_mean" in snap.metrics:
        lines.append(f"  Last reward: {snap.metrics['rollout/ep_rew_mean']:.2f}")
    if "time/fps" in snap.metrics:
        lines.append(f"  FPS        : {int(snap.metrics['time/fps']):,}")
    print("\n".join(lines), file=sys.stderr)


def run(child_args: list, interval_hours: float, pin: bool = True, sync_to_main: bool = False) -> None:
    interval_seconds = interval_hours * 3600
    session_start = time.time()
    child_env = _build_child_env()
    state = LauncherState(interval_hours=interval_hours)
    ui = LauncherUI()

    if pin:
        repo_root = get_repo_root()
        _prune_stale_launcher_worktrees(repo_root)
        model_path = _find_model_arg(child_args)
        if model_path and not sync_to_main:
            checkpoint_hash = _read_checkpoint_git_hash(model_path)
            if not checkpoint_hash:
                sys.exit(
                    f"[launcher] ERROR: --model given but no git_hash found in metadata.json "
                    f"for {model_path!r}.\nUse --no-pin to skip worktree isolation."
                )
            pin_hash = checkpoint_hash
        else:
            pin_hash = get_git_hash()  # full hash for worktree add

        try:
            train_script, src_dir, worktree_cleanup = _create_run_worktree(pin_hash)
        except RuntimeError as e:
            sys.exit(f"[launcher] ERROR: {e}")
        atexit.register(worktree_cleanup)
        state.initial_git_hash = pin_hash[:8]
    else:
        train_script, src_dir = _TRAIN_SCRIPT, _SRC_DIR
        state.initial_git_hash = _git_hash()

    existing_model = _find_model_arg(child_args)
    if not existing_model:
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("models", f"run_{ts}")
        os.makedirs(run_dir, exist_ok=True)
        child_args = _insert_or_replace_run_dir_arg(child_args, run_dir)
    else:
        run_dir = os.path.dirname(os.path.abspath(existing_model))
        child_args = _insert_or_replace_run_dir_arg(child_args, run_dir)

    state.run_dir = run_dir

    cmd_q: queue.Queue = queue.Queue()
    _setup_raw_input()
    threading.Thread(target=_read_keys, args=(cmd_q,), daemon=True).start()

    if interval_hours > 0:
        state.add_event(f"🚀 Starting — restart every {interval_hours:.1f}h")
    else:
        state.add_event("🚀 Starting — single run (no restart)")

    if pin:
        if sync_to_main and _find_model_arg(child_args):
            state.add_event(
                f"🔄 --sync-to-main: pinned to current HEAD {state.initial_git_hash} "
                f"(ignoring checkpoint's original hash)"
            )
        else:
            state.add_event(f"📌 Pinned to {state.initial_git_hash} (isolated worktree)")
    else:
        state.add_event(f"--no-pin: running from current tree ({state.initial_git_hash})")

    _run_dir_box: list = [run_dir]
    atexit.register(lambda: _print_exit_summary(_run_dir_box[0], state))

    with Live(refresh_per_second=2, screen=True) as live:
        while True:

            if state.restart_count > 0:
                current_hash = _git_hash()
                if current_hash != state.initial_git_hash:
                    if pin:
                        state.add_event(
                            f"ℹ️  main diverged to {current_hash} — pinned worktree unaffected"
                        )
                    else:
                        state.add_event(
                            f"ℹ️  main diverged to {current_hash} — next restart uses new code"
                        )

            proc = _launch_child(child_args, child_env, state, train_script, src_dir)
            state.add_event(f"✅ Child PID {proc.pid} started")

            deadline = state.run_start + interval_seconds if interval_hours > 0 else float("inf")
            state.deadline = deadline

            flags = _PollFlags()

            while True:
                live.update(ui.render(state.snapshot(), live.console.height))

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
                    ch = cmd_q.get_nowait()
                    if state.view_mode == "confirm_quit":
                        if ch == "y":
                            flags.quit_requested = True
                            state.add_event("👋 Quit requested — waiting for child to save…")
                            if not flags.sigterm_sent:
                                try:
                                    os.kill(proc.pid, signal.SIGTERM)
                                except ProcessLookupError:
                                    pass
                                flags.sigterm_sent = True
                            state.view_mode = "dashboard"
                        elif ch in ("n", "d"):
                            state.view_mode = "dashboard"
                    else:
                        _dispatch_command(ch, proc, state, flags, deadline, interval_hours)

            proc.wait()
            live.update(ui.render(state.snapshot(), live.console.height))

            rc = proc.returncode

            if rc == TrainExitCode.COMPLETE:
                state.add_event("✅ Training complete — all steps done")
                live.update(ui.render(state.snapshot(), live.console.height))
                time.sleep(1)
                sys.exit(0)

            if rc != TrainExitCode.INTERRUPTED:
                state.add_event(f"🛑 Child crashed (exit {rc}) — not restarting")
                live.update(ui.render(state.snapshot(), live.console.height))
                time.sleep(2)
                atexit.register(_print_crash_log, state.snapshot().log_lines)
                sys.exit(rc)

            if flags.quit_requested:
                state.add_event("👋 Quit complete.")
                live.update(ui.render(state.snapshot(), live.console.height))
                time.sleep(1)
                sys.exit(0)

            if interval_hours <= 0 and not flags.restart_requested:
                sys.exit(0)

            checkpoint = find_latest_checkpoint("models", run_dir=run_dir, min_mtime=session_start)
            if checkpoint is None:
                state.add_event("🛑 No checkpoint found under models/ — cannot restart")
                live.update(ui.render(state.snapshot(), live.console.height))
                time.sleep(2)
                sys.exit(1)

            run_dir = os.path.dirname(os.path.abspath(checkpoint))
            _run_dir_box[0] = run_dir
            state.run_dir = run_dir
            state.add_event(f"✅ Restarting from {os.path.basename(checkpoint)}")
            child_args = _insert_or_replace_model_arg(child_args, checkpoint)
            child_args = _insert_or_replace_run_dir_arg(child_args, run_dir)
            state.restart_count += 1
            state.view_mode = "dashboard"
