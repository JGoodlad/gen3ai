"""Launcher driver + child supervisor (Textual).

The launcher's run loop: ``run()`` sets up the session (worktree pin, run dir, at-exit
handlers), then drives a ``LauncherApp`` (``app.py``) whose ``@work(thread=True)`` worker
runs ``_supervise`` — the restart/child-supervision loop that drives ``LauncherState`` and
returns an exit code (the app renders from ``state.snapshot()`` on its own timer and never
blocks). On exit, ``_reap`` makes sure the child is stopped (no orphan).

Framework-agnostic pieces are reused from sibling modules: ``child.py`` (spawn + reader
threads), ``checkpoint.py``/``worktree.py`` helpers, ``input.py``'s ``_PollFlags`` +
``_dispatch_command``. The pure crash/log/exit helpers live here.
"""

import atexit
import os
import queue
import secrets
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

from main.exit_codes import TrainExitCode
from main.launcher.checkpoint import (
    find_latest_checkpoint,
    run_dir_for_checkpoint,
    child_uses_bridge,
    _apply_default_showdown_port,
    _find_model_arg,
    _insert_or_replace_model_arg,
    _insert_or_replace_run_dir_arg,
    _peek_arg,
    resolve_launch_run_dir,
    _strip_launcher_args,
)
from main.launcher.child import (
    _build_child_env,
    _launch_child,
    child_log_path,
    _TRAIN_SCRIPT,
    _SRC_DIR,
)
from main.launcher.input import _PollFlags, _dispatch_command
from main.launcher.state import LauncherState
from main.launcher.worktree import (
    _git_hash,
    _prune_stale_launcher_worktrees,
    _read_checkpoint_git_hash,
    _create_run_worktree,
    get_git_hash,
    get_repo_root,
)


# A self-crash sooner than this after (re)launch counts toward the consecutive
# circuit-breaker (a deterministic startup crash can't spin forever); a crash after
# sustained progress resets the counter. 10 minutes — comfortably past the 3+ min it
# takes to bring up the SubprocVecEnv workers + Showdown connections.
_FAST_CRASH_SECONDS = 600.0


# Substrings in the child's output that mark a deterministic, non-recoverable startup
# failure — restarting would hit the exact same error every time. The dedicated
# FATAL_CONFIG exit code is the primary signal; matching one of these ALSO surfaces the
# specific reason line(s) in the Events panel (vs the generic exit-code-only fallback).
_FATAL_CONFIG_SIGNATURES = ("[ModelVersion] FATAL", "[StableOpponent] FATAL")


def _fatal_config_reason(rc: int, log_lines: "list | None") -> "list | None":
    """If this exit is a non-recoverable config/architecture error, return a short list of
    human-readable reason lines (so the launcher gives up instead of auto-restarting into
    the identical failure); otherwise return None.

    Two signals, either sufficient: the dedicated ``FATAL_CONFIG`` exit code the child
    raises for a ``ModelVersionError`` (arch-family / vf_coef / reward-config mismatch), or
    a known FATAL signature in the captured output (defensive — catches a FATAL that came
    out as a generic exit 1)."""
    lines = list(log_lines or [])
    for i, line in enumerate(lines):
        if any(sig in line for sig in _FATAL_CONFIG_SIGNATURES):
            # The FATAL line + a couple of following explanatory lines (the
            # ModelVersionError message spans 2-3 lines: what mismatched + the fix).
            return [s for s in (l.strip() for l in lines[i : i + 3]) if s]
    if rc == int(TrainExitCode.FATAL_CONFIG):
        return ["non-recoverable configuration error (see crash log)"]
    return None


def _print_crash_log(log_lines: "list | None") -> None:
    """Dump captured child output to stderr after the screen has closed."""
    if not log_lines:
        return
    print("\n── Child output (last 100 lines) ─────────────────────────────", file=sys.stderr)
    for line in log_lines[-100:]:
        print(line, file=sys.stderr)
    print("──────────────────────────────────────────────────────────────", file=sys.stderr)


def _save_crash_log(run_dir: "str | None", state: LauncherState, rc: int) -> "str | None":
    """Snapshot the crashed child's recent output to a unique
    ``<run_dir>/crashes/restart_err_<token>.txt`` (timestamp + random hex so back-to-back
    crashes never collide; never overwritten, unlike the reused ``launcher_child.log``).
    Best-effort: returns the path, or None if there's no run_dir or the write fails."""
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
    """Finalize the persisted child log in the model dir and point the user to it. The
    child stdout is streamed to ``<run_dir>/launcher_child.log`` live (child.py), so the
    file is already complete — here we just append a session-end footer and print the path
    (flushing the in-memory scrollback as a fallback if streaming never wrote anything)."""
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


@dataclass
class _SessionCtx:
    """Resolved, frontend-agnostic session config produced by :func:`_prepare_session`.

    Mirrors the locals the Rich ``run()`` computes before its ``with Live(...)`` block."""

    child_args: list
    child_env: dict
    train_script: str
    src_dir: str
    run_dir: str
    run_dir_box: list      # 1-element [run_dir]; updated as restarts re-derive run_dir
    session_start: float   # time.time() — floor for checkpoint discovery (min_mtime)
    interval_seconds: float
    grace_seconds: float
    pin: bool


def _prepare_session(
    state: LauncherState,
    child_args: list,
    *,
    interval_hours: float,
    pin: bool,
    sync_to_main: bool,
    pin_hash_override: "str | None",
    grace_minutes: float,
    max_crash_restarts: int,
) -> _SessionCtx:
    """Worktree pin + run-dir resolution + initial events + at-exit handlers.

    Runs on the main thread *before* the Textual screen opens, so a pin failure can
    ``sys.exit`` with a clean error message instead of being swallowed by the TUI or
    raised uselessly inside a worker thread."""
    interval_seconds = interval_hours * 3600
    grace_seconds = max(0.0, grace_minutes * 60)
    child_env = _build_child_env()
    # Record the launcher's own invocation so the child can persist it into metadata.json
    # (launcher-managed runs write no command.txt). Carried across restarts via ctx.child_env.
    child_env["LAUNCHER_COMMAND"] = " ".join(sys.argv)
    if interval_hours > 0:
        child_env["LAUNCHER_RESTART_INTERVAL_SEC"] = str(interval_seconds)

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

    # Run dir: fresh → minted/named; a --run-name (or --exploiter) resume → FORK into a new dir
    # (init from --model, write here, never clobber the checkpoint's source dir); plain resume →
    # continue the checkpoint's own dir. (resolve_launch_run_dir documents all three.)
    try:
        run_dir = resolve_launch_run_dir(child_args, time.strftime("%Y%m%d_%H%M%S"))
    except ValueError as e:
        print(f"[launcher] ERROR: {e}")
        sys.exit(1)
    os.makedirs(run_dir, exist_ok=True)
    child_args = _insert_or_replace_run_dir_arg(child_args, run_dir)

    state.run_dir = run_dir

    if interval_hours > 0:
        state.add_event(f"🚀 Starting — restart every {interval_hours:.1f}h")
    else:
        state.add_event("🚀 Starting — single run (no restart)")

    if child_uses_bridge(child_args):
        # In-process BattleStream transport for training AND eval — no server, the port is unused.
        _impl = _peek_arg(child_args, "--use-bridge") or "node"
        state.add_event(f"🌉 Transport: in-process bridge [{_impl}] (no Showdown server)")
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

    run_dir_box: list = [run_dir]
    atexit.register(lambda: _print_exit_summary(run_dir_box[0], state))
    # atexit runs LIFO, so this fires before the exit-summary print above.
    atexit.register(lambda: _dump_logs_on_exit(run_dir_box[0], state))

    return _SessionCtx(
        child_args=child_args,
        child_env=child_env,
        train_script=train_script,
        src_dir=src_dir,
        run_dir=run_dir,
        run_dir_box=run_dir_box,
        session_start=time.time(),
        interval_seconds=interval_seconds,
        grace_seconds=grace_seconds,
        pin=pin,
    )


def _supervise(
    state: LauncherState,
    cmd_q: queue.Queue,
    ctx: _SessionCtx,
    *,
    interval_hours: float,
    grace_minutes: float,
    max_crash_restarts: int,
    on_tick=None,
    shutdown: "threading.Event | None" = None,
    proc_box: "list | None" = None,
) -> int:
    """The restart/child-supervision loop — mirror of ``run.py`` lines 244-419.

    Returns the process exit code instead of calling ``sys.exit`` (it runs in a worker
    thread). ``on_tick(snapshot)`` marks where Rich would repaint (Textual passes ``None``
    and renders on its own timer). ``shutdown``/``proc_box`` support frontend-driven
    teardown so an abnormal app exit never orphans the child."""
    KILL_GRACE_SECONDS = 90.0

    child_args = ctx.child_args
    child_env = ctx.child_env
    train_script = ctx.train_script
    src_dir = ctx.src_dir
    run_dir = ctx.run_dir
    run_dir_box = ctx.run_dir_box
    session_start = ctx.session_start
    interval_seconds = ctx.interval_seconds
    grace_seconds = ctx.grace_seconds

    def _tick() -> None:
        if on_tick is not None:
            on_tick(state.snapshot())

    consecutive_fast_crashes = 0

    while True:
        if shutdown is not None and shutdown.is_set():
            return 0  # don't launch a fresh child while tearing down

        if state.restart_count > 0:
            current_hash = _git_hash()
            if current_hash != state.initial_git_hash:
                if ctx.pin:
                    state.add_event(
                        f"ℹ️  main diverged to {current_hash} — pinned worktree unaffected"
                    )
                else:
                    state.add_event(
                        f"ℹ️  main diverged to {current_hash} — next restart uses new code"
                    )

        proc = _launch_child(child_args, child_env, state, train_script, src_dir)
        if proc_box is not None:
            proc_box[:] = [proc]
        state.add_event(f"✅ Child PID {proc.pid} started")

        deadline = state.run_start + interval_seconds if interval_hours > 0 else float("inf")
        state.deadline = deadline

        flags = _PollFlags()
        deadline_announced = False

        while True:
            _tick()

            now = time.monotonic()
            remaining = deadline - now

            # Frontend-requested teardown (abnormal app exit / ctrl-c): treat as a quit.
            if shutdown is not None and shutdown.is_set() and not flags.quit_requested:
                flags.quit_requested = True
                state.add_event("👋 Shutting down — waiting for child to save…")
                if not flags.sigterm_sent:
                    try:
                        os.kill(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    flags.sigterm_sent = True
                    flags.sigterm_at = now

            # The child stops itself at the next rollout boundary once the interval
            # elapses; the launcher only hard-kills as a fallback after a grace overrun.
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

            # The Textual app handles view navigation + the confirm overlays itself (instant,
            # event-loop); it only sends us child-control chars (r/c/p/s/f via _dispatch_command,
            # `f` arriving only after its own confirm) and the "__quit__" sentinel once the user
            # has confirmed quit.
            while not cmd_q.empty():
                ch = cmd_q.get_nowait()
                if ch == "__quit__":
                    flags.quit_requested = True
                    state.add_event("👋 Quit requested — waiting for child to save…")
                    if not flags.sigterm_sent:
                        try:
                            os.kill(proc.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        flags.sigterm_sent = True
                        flags.sigterm_at = time.monotonic()
                else:
                    _dispatch_command(ch, proc, state, flags, deadline, interval_hours)

        proc.wait()
        _tick()

        rc = proc.returncode

        if rc == TrainExitCode.COMPLETE:
            state.add_event("✅ Training complete — all steps done")
            _tick()
            time.sleep(1)
            return 0

        if flags.quit_requested:
            state.add_event("👋 Quit complete.")
            _tick()
            time.sleep(1)
            return 0

        # A non-recoverable config/architecture error (checkpoint arch-family mismatch,
        # vf_coef / reward-config drift): restarting would deterministically hit the SAME
        # error, so give up immediately with the reason on-screen rather than burning the
        # crash circuit-breaker and forcing the user to open the logs to find out why.
        fatal_reason = _fatal_config_reason(rc, state.snapshot().log_lines)
        if fatal_reason:
            state.crash_count += 1
            err_path = _save_crash_log(run_dir, state, rc)
            saved = (
                f" — saved {os.path.join('crashes', os.path.basename(err_path))}" if err_path
                else ""
            )
            state.add_event(f"🛑 Fatal config error — will NOT restart{saved}")
            for line in fatal_reason:
                state.add_event(f"   {line}")
            _tick()
            time.sleep(2)
            atexit.register(_print_crash_log, state.snapshot().log_lines)
            return rc

        # An INTENDED restart (user 'r' or launcher force-kill) recovers regardless of the
        # exit code; a self-crash (unintended non-INTERRUPTED exit) auto-restarts from the
        # last checkpoint, guarded by the consecutive-rapid-crash circuit-breaker.
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
                _tick()
                time.sleep(2)
                atexit.register(_print_crash_log, state.snapshot().log_lines)
                return rc
        else:
            consecutive_fast_crashes = 0

        if interval_hours <= 0 and not intended_restart and not crashed:
            return 0

        checkpoint = find_latest_checkpoint("models", run_dir=run_dir, min_mtime=session_start)
        if checkpoint is None:
            state.add_event("🛑 No checkpoint found under models/ — cannot restart")
            _tick()
            time.sleep(2)
            # A crash with nothing to resume from is genuinely fatal — propagate the
            # child's exit code and dump its output, rather than masking it as exit 1.
            if crashed:
                atexit.register(_print_crash_log, state.snapshot().log_lines)
                return rc
            return 1

        # checkpoint may be <run>/checkpoints/checkpoint_*.zip → strip checkpoints/ so
        # run_dir is the run root (not the subdir) for the TUI badge + child --run-dir.
        run_dir = run_dir_for_checkpoint(checkpoint)
        run_dir_box[0] = run_dir
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


def _reap(proc_box: "list | None") -> None:
    """Teardown safety net: ensure the child is dead before the process exits.

    Normal quit (q→y) already SIGTERMs + reaps inside ``_supervise``; this covers an
    abnormal app exit (ctrl-c handled by Textual, terminal closed) where the supervisor
    worker may still be blocked in ``proc.wait``. SIGTERM (give the child time to save its
    checkpoint), then SIGKILL. Reaping the child also unblocks the worker's ``proc.wait``
    so the interpreter can shut its non-daemon worker thread down cleanly."""
    if not proc_box:
        return
    proc = proc_box[0]
    try:
        if proc.poll() is not None:
            return   # already exited (clean q→y path reaped it) — nothing to wait on
        # The screen is down by now, so narrate the wait on stderr — otherwise it looks
        # like a silent freeze while the child saves its checkpoint.
        print("\n⏳ Stopping — waiting for the training child to save its checkpoint…",
              file=sys.stderr, flush=True)
        try:
            os.kill(proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        start = time.monotonic()
        for _ in range(10):   # up to a 10s graceful-save grace
            try:
                proc.wait(timeout=1)
                print(f"✅ Child stopped & checkpoint saved ({time.monotonic() - start:.0f}s).",
                      file=sys.stderr, flush=True)
                return
            except subprocess.TimeoutExpired:
                print(f"   …still saving ({time.monotonic() - start:.0f}s)",
                      file=sys.stderr, flush=True)
        print("⚠ Child didn't stop within 10s — forcing kill (checkpoint may be partial).",
              file=sys.stderr, flush=True)
        try:
            os.kill(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
    except Exception:
        pass


def run(
    child_args: list,
    interval_hours: float,
    pin: bool = True,
    sync_to_main: bool = False,
    pin_hash_override: "str | None" = None,
    grace_minutes: float = 20.0,
    max_crash_restarts: int = 3,
) -> None:
    """Set up the session, run the app, reap, exit.

    The supervisor runs in a Textual worker thread (started by ``LauncherApp.on_mount``);
    when it returns, it records the exit code and asks the app to exit. After the screen
    closes we set ``shutdown`` + reap any survivor, then propagate the exit code."""
    from main.launcher.app import LauncherApp  # local import: keep textual off the import path until launch

    state = LauncherState(interval_hours=interval_hours)
    ctx = _prepare_session(
        state,
        child_args,
        interval_hours=interval_hours,
        pin=pin,
        sync_to_main=sync_to_main,
        pin_hash_override=pin_hash_override,
        grace_minutes=grace_minutes,
        max_crash_restarts=max_crash_restarts,
    )

    cmd_q: queue.Queue = queue.Queue()
    shutdown = threading.Event()
    proc_box: list = []

    app = LauncherApp(state, cmd_q)

    def _supervisor() -> None:
        try:
            code = _supervise(
                state,
                cmd_q,
                ctx,
                interval_hours=interval_hours,
                grace_minutes=grace_minutes,
                max_crash_restarts=max_crash_restarts,
                on_tick=None,
                shutdown=shutdown,
                proc_box=proc_box,
            )
        except Exception as e:  # noqa: BLE001 — never let a supervisor fault hang the app
            state.add_event(f"❌ launcher supervisor error: {e}")
            code = 1
        app.exit_code = code
        try:
            app.call_from_thread(app.exit)
        except Exception:
            pass

    app.supervisor = _supervisor
    try:
        app.run()
    finally:
        # Teardown safety net — never orphan the child, even if the app errors out
        # (a Textual startup/runtime fault after the supervisor worker spawned a child).
        shutdown.set()
        _reap(proc_box)
    sys.exit(getattr(app, "exit_code", 0))


def main() -> None:
    """Launcher entry point (``python -m main.launcher`` / ``-m main.launcher.tui``).

    Parses launcher-owned flags, strips them, defaults the Showdown port, and forwards the
    rest to ``train_rl_agent.py``."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m main.launcher",
        description="Launcher: wraps train_rl_agent.py with periodic full-process restart (Textual UI).",
        add_help=False,
    )
    parser.add_argument(
        "--restart-interval-hours",
        type=float,
        default=3.0,
        help="Restart the training process every N hours (0 = run once)",
    )
    parser.add_argument(
        "--restart-grace-minutes",
        type=float,
        default=20.0,
        help=(
            "Fallback window after a scheduled restart's deadline. The child normally "
            "stops itself at the next rollout boundary; the launcher only force-kills if "
            "the child overruns the deadline by this many minutes (hung / very long rollout)."
        ),
    )
    parser.add_argument(
        "--max-crash-restarts",
        type=int,
        default=3,
        help=(
            "Auto-restart from the last checkpoint when the child self-crashes, up to "
            "this many consecutive rapid crashes (each < 10 min of launch) before giving "
            "up. 0 = unlimited. A crash after sustained progress resets the counter."
        ),
    )
    parser.add_argument(
        "--no-pin",
        action="store_true",
        default=False,
        help="Skip worktree pinning — child runs from the current working tree (old behaviour)",
    )
    pin_group = parser.add_mutually_exclusive_group()
    pin_group.add_argument(
        "--sync-to-main",
        action="store_true",
        default=False,
        help=(
            "When resuming from a checkpoint, pin the isolated worktree to the current HEAD "
            "instead of the checkpoint's original git hash. Useful for picking up UI or tooling "
            "fixes without discarding the checkpoint."
        ),
    )
    pin_group.add_argument(
        "--pin-to-hash",
        metavar="HASH",
        default=None,
        help=(
            "Pin the isolated worktree to a specific git hash instead of the checkpoint's "
            "recorded hash or current HEAD. Useful for resuming a run against a known-good commit."
        ),
    )
    parser.add_argument("-h", "--help", action="store_true")

    known, _ = parser.parse_known_args()

    if known.help:
        parser.print_help()
        print("\nAll other arguments are forwarded to train_rl_agent.py.")
        sys.exit(0)

    if known.pin_to_hash and known.no_pin:
        parser.error("--pin-to-hash and --no-pin are mutually exclusive")

    child_args = _strip_launcher_args(sys.argv[1:])
    # Launcher sessions are long-lived: default to the dedicated training server (8001)
    # so dev-server churn on 8000 can't drop every worker's connection mid-run. An
    # explicit --showdown-port still wins.
    child_args = _apply_default_showdown_port(child_args)
    try:
        run(
            child_args,
            interval_hours=known.restart_interval_hours,
            pin=not known.no_pin,
            sync_to_main=known.sync_to_main,
            pin_hash_override=known.pin_to_hash,
            grace_minutes=known.restart_grace_minutes,
            max_crash_restarts=known.max_crash_restarts,
        )
    except KeyboardInterrupt:
        sys.exit(0)
