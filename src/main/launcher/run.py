"""Main restart loop and session summary for the training launcher."""

import atexit
import os
import queue
import secrets
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
    _peek_arg,
)
from main.launcher.child import _build_child_env, _launch_child, child_log_path, _TRAIN_SCRIPT, _SRC_DIR
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


# A self-crash sooner than this after (re)launch counts toward the consecutive
# circuit-breaker. A child that crashes only after sustained progress is treated
# as a recoverable transient (the counter resets) — the breaker exists to stop a
# *deterministic* startup crash (bad checkpoint, import error) spinning forever.
# Window must comfortably exceed startup cost: bringing up the SubprocVecEnv
# workers + Showdown connections alone takes 3+ minutes, so a deterministic
# startup crash often surfaces well past the 2-minute mark — too short a window
# would misread it as "made progress" and never trip the breaker. 10 minutes.
_FAST_CRASH_SECONDS = 600.0


def _find_ent_coef(args: list) -> "float | None":
    return _peek_arg(args, "--ent-coef", type_=float)


def _print_crash_log(log_lines: "list | None") -> None:
    """Dump captured child output to stderr after the Live screen has closed."""
    if not log_lines:
        return
    print("\n── Child output (last 100 lines) ─────────────────────────────", file=sys.stderr)
    for line in log_lines[-100:]:
        print(line, file=sys.stderr)
    print("──────────────────────────────────────────────────────────────", file=sys.stderr)


def _save_crash_log(run_dir: "str | None", state: LauncherState, rc: int) -> "str | None":
    """Snapshot the crashed child's recent output to a unique
    ``<run_dir>/crashes/restart_err_<token>.txt``.

    ``launcher_child.log`` is reused (appended) across relaunches, so once the
    launcher auto-restarts a crashed child the original traceback is buried under
    the next session's output. This writes a standalone, never-overwritten copy of
    the in-memory scrollback (deep enough to hold the full traceback) per crash so
    it can be debugged later, collected under a ``crashes/`` subfolder of the run
    dir so they don't clutter the checkpoint listing. ``<token>`` is a timestamp
    plus a few random hex chars so back-to-back crashes never collide on a filename.
    Best-effort: returns the path on success, ``None`` if there is no run_dir or the
    write fails.
    """
    if not run_dir:
        return None
    snap = state.snapshot()
    token = time.strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(3)
    crashes_dir = os.path.join(run_dir, "crashes")
    path = os.path.join(crashes_dir, f"restart_err_{token}.txt")
    try:
        os.makedirs(crashes_dir, exist_ok=True)
        with open(path, "w", errors="replace") as f:
            f.write("# Gen3AI launcher crash log\n")
            f.write(f"# time     : {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# exit code: {rc}\n")
            f.write(f"# pid      : {snap.pid}\n")
            f.write(f"# git      : {snap.initial_git_hash}\n")
            f.write(f"# run_dir  : {run_dir}\n")
            f.write(f"# last step: {snap.metrics_step:,}\n")
            f.write(f"# crash #  : {snap.crash_count}\n")
            f.write("# " + "-" * 62 + "\n")
            f.write("\n".join(snap.log_lines) + "\n")
        return path
    except Exception:
        return None


def _dump_logs_on_exit(run_dir: "str | None", state: LauncherState) -> None:
    """Finalize the persisted child log in the model dir and point the user to it.

    The child stdout is streamed to ``<run_dir>/launcher_child.log`` live (child.py),
    so on a normal or crash exit the file is already complete — here we just append a
    session-end footer and print the path. As a safety net, if streaming never wrote
    anything (run_dir resolved late, or the file couldn't be opened), we flush the
    in-memory scrollback to the same file so the logs are never lost on exit.
    """
    path = child_log_path(run_dir)
    if not path:
        return
    try:
        snap = state.snapshot()
        had_stream = os.path.exists(path) and os.path.getsize(path) > 0
        os.makedirs(run_dir, exist_ok=True)
        with open(path, "a", errors="replace") as f:
            if not had_stream and snap.log_lines:
                f.write("\n".join(snap.log_lines) + "\n")
            f.write(f"===== session ended {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
        print(f"\n📄 Full child log: {path}", file=sys.stderr)
    except Exception:
        pass


def _print_exit_summary(run_dir: "str | None", state: LauncherState) -> None:
    snap = state.snapshot()
    lines = ["", "── Training session ended ──────────────────────────────────────"]
    if run_dir:
        lines.append(f"  Run folder : {run_dir}")
    lines.append(f"  Restarts   : {snap.restart_count}")
    if snap.crash_count:
        lines.append(f"  Crashes    : {snap.crash_count} (auto-restarted from checkpoint)")
    if snap.metrics_step:
        lines.append(f"  Last step  : {snap.metrics_step:,}")
    if "rollout/ep_rew_mean" in snap.metrics:
        lines.append(f"  Last reward: {snap.metrics['rollout/ep_rew_mean']:.2f}")
    if "time/fps" in snap.metrics:
        lines.append(f"  FPS        : {int(snap.metrics['time/fps']):,}")
    ckpt = find_latest_checkpoint("models", run_dir=run_dir)
    if ckpt:
        lines.append(f"  Last model : {ckpt}")
    print("\n".join(lines), file=sys.stderr)


def run(child_args: list, interval_hours: float, pin: bool = True, sync_to_main: bool = False, pin_hash_override: "str | None" = None, grace_minutes: float = 20.0, max_crash_restarts: int = 3) -> None:
    interval_seconds = interval_hours * 3600
    grace_seconds = max(0.0, grace_minutes * 60)
    # Fallback SIGKILL window: a hung child can't run its SIGTERM handler (its main
    # thread is blocked in a C-level wait), so it never saves+exits. If it ignores a
    # launcher-sent SIGTERM (deadline overrun / 'r' restart) for this long, SIGKILL it
    # so the run can still recover from the last checkpoint. This is a hard kill grace,
    # NOT a progress guess — "you've had 90 s to respond and didn't."
    KILL_GRACE_SECONDS = 90.0
    session_start = time.time()
    child_env = _build_child_env()
    # Tell the child its restart budget so it can stop cleanly at the next
    # rollout boundary (GracefulRestartCallback). The launcher only hard-kills
    # as a fallback once the child overruns the deadline by grace_seconds.
    if interval_hours > 0:
        child_env["LAUNCHER_RESTART_INTERVAL_SEC"] = str(interval_seconds)
    state = LauncherState(interval_hours=interval_hours)
    ui = LauncherUI()

    if pin:
        repo_root = get_repo_root()
        _prune_stale_launcher_worktrees(repo_root)
        model_path = _find_model_arg(child_args)
        if pin_hash_override:
            pin_hash = pin_hash_override
        elif model_path and not sync_to_main:
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
        child_env["LAUNCHER_GIT_HASH"] = pin_hash
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
    state.ent_coef = _find_ent_coef(child_args)

    cmd_q: queue.Queue = queue.Queue()
    _setup_raw_input()
    threading.Thread(target=_read_keys, args=(cmd_q,), daemon=True).start()

    if interval_hours > 0:
        state.add_event(f"🚀 Starting — restart every {interval_hours:.1f}h")
    else:
        state.add_event("🚀 Starting — single run (no restart)")

    if "--use-showdown-bridge" in child_args:
        # In-process BattleStream transport for training AND eval — no server, the port is unused.
        state.add_event("🌉 Transport: in-process bridge (no Showdown server)")
    else:
        showdown_port = _peek_arg(child_args, "--showdown-port", type_=int)
        if showdown_port is not None:
            state.add_event(f"🔌 Showdown server :{showdown_port}")

    if pin:
        if pin_hash_override:
            state.add_event(
                f"📌 --pin-to-hash: pinned to {state.initial_git_hash} (explicit override)"
            )
        elif sync_to_main and _find_model_arg(child_args):
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
    # Persist the full child log to the model dir on every exit (crash, complete,
    # quit). atexit runs LIFO, so this fires before the exit-summary print above.
    atexit.register(lambda: _dump_logs_on_exit(_run_dir_box[0], state))

    # Number of consecutive *rapid* self-crashes since the last healthy run. Reset
    # on any non-crash exit or a crash that came after sustained progress; drives the
    # circuit-breaker that stops a deterministic crash loop (see _FAST_CRASH_SECONDS).
    consecutive_fast_crashes = 0

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
            deadline_announced = False

            while True:
                live.update(ui.render(state.snapshot(), live.console.height))

                now = time.monotonic()
                remaining = deadline - now

                # The child stops itself at the next rollout boundary
                # (GracefulRestartCallback) once the interval elapses, so the
                # launcher no longer SIGTERMs at the exact deadline. It only
                # hard-kills as a fallback if the child overruns by grace_seconds
                # (hung, or a pathologically long rollout/eval).
                if interval_hours > 0 and not deadline_announced and remaining <= 0:
                    state.add_event(
                        f"⏳ {interval_hours:.1f}h elapsed — waiting for child to reach rollout boundary…"
                    )
                    deadline_announced = True

                if not flags.sigterm_sent and remaining <= -grace_seconds and interval_hours > 0:
                    state.add_event(
                        f"⏰ Child overran by {grace_minutes:.0f}m past deadline — forcing restart"
                    )
                    try:
                        os.kill(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    flags.sigterm_sent = True
                    flags.forced_restart = True
                    flags.sigterm_at = now

                # SIGKILL escalation: a hung child can't run its SIGTERM handler, so it
                # never saves+exits. After KILL_GRACE_SECONDS, SIGKILL it so the run can
                # actually recover (the forced_restart flag makes the loop restart from
                # the last checkpoint despite the non-clean exit).
                if (flags.sigterm_sent and not flags.sigkill_sent
                        and now - flags.sigterm_at >= KILL_GRACE_SECONDS):
                    state.add_event(
                        f"💀 Child ignored SIGTERM for {KILL_GRACE_SECONDS:.0f}s — SIGKILL"
                    )
                    try:
                        os.kill(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    flags.sigkill_sent = True

                poll_timeout = 0.5
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
                                flags.sigterm_at = time.monotonic()
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

            if flags.quit_requested:
                state.add_event("👋 Quit complete.")
                live.update(ui.render(state.snapshot(), live.console.height))
                time.sleep(1)
                sys.exit(0)

            # Classify the exit. An INTENDED restart — the user pressed 'r'
            # (restart_requested) or the launcher force-killed a stalled/overran child
            # (forced_restart) — recovers regardless of the exit code: a hung child
            # ignores SIGTERM and is SIGKILL'd, so rc won't be the clean INTERRUPTED(15).
            # A *self*-crash (unintended, non-INTERRUPTED exit) used to stop the
            # launcher; it now auto-restarts from the last checkpoint after saving the
            # traceback, guarded by a consecutive-rapid-crash circuit-breaker so a
            # deterministic startup crash can't spin forever.
            intended_restart = flags.forced_restart or flags.restart_requested
            crashed = not intended_restart and rc != TrainExitCode.INTERRUPTED

            if crashed:
                state.crash_count += 1
                err_path = _save_crash_log(run_dir, state, rc)
                ran_for = time.monotonic() - state.run_start
                if ran_for < _FAST_CRASH_SECONDS:
                    consecutive_fast_crashes += 1
                else:
                    consecutive_fast_crashes = 0  # made progress → treat as transient
                saved = (
                    f" — saved {os.path.join('crashes', os.path.basename(err_path))}" if err_path
                    else " — crash-log capture failed"
                )
                state.add_event(f"🛑 Child crashed (exit {rc}){saved} · crash #{state.crash_count}")
                if max_crash_restarts > 0 and consecutive_fast_crashes >= max_crash_restarts:
                    state.add_event(
                        f"🛑 {consecutive_fast_crashes} rapid crashes in a row "
                        f"(< {int(_FAST_CRASH_SECONDS / 60)}m each) — giving up"
                    )
                    live.update(ui.render(state.snapshot(), live.console.height))
                    time.sleep(2)
                    atexit.register(_print_crash_log, state.snapshot().log_lines)
                    sys.exit(rc)
            else:
                consecutive_fast_crashes = 0

            if interval_hours <= 0 and not intended_restart and not crashed:
                sys.exit(0)

            checkpoint = find_latest_checkpoint("models", run_dir=run_dir, min_mtime=session_start)
            if checkpoint is None:
                state.add_event("🛑 No checkpoint found under models/ — cannot restart")
                live.update(ui.render(state.snapshot(), live.console.height))
                time.sleep(2)
                # A crash with nothing to resume from is genuinely fatal — propagate the
                # child's exit code and dump its output, rather than masking it as exit 1.
                if crashed:
                    atexit.register(_print_crash_log, state.snapshot().log_lines)
                    sys.exit(rc)
                sys.exit(1)

            run_dir = os.path.dirname(os.path.abspath(checkpoint))
            _run_dir_box[0] = run_dir
            state.run_dir = run_dir
            if crashed:
                state.add_event(
                    f"♻️  Auto-restart #{state.restart_count + 1} after crash "
                    f"from {os.path.basename(checkpoint)}"
                )
            else:
                state.add_event(f"✅ Restarting from {os.path.basename(checkpoint)}")
            child_args = _insert_or_replace_model_arg(child_args, checkpoint)
            child_args = _insert_or_replace_run_dir_arg(child_args, run_dir)
            state.restart_count += 1
            state.view_mode = "dashboard"
