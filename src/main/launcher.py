"""Process-level restart loop for train_rl_agent.py.

Periodically SIGTERMs the training child and relaunches from the most recently
saved checkpoint, recovering FPS lost to pymalloc fragmentation over long runs.
Includes a Rich TUI dashboard fed by metrics delivered via a pipe fd from the
child's MetricsExporterCallback.
"""

import atexit
import glob
import json
import re
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

from rich.live import Live

from main.exit_codes import TrainExitCode
from main.launcher_ui import LauncherState, LauncherUI
from utils.git import get_git_hash, get_repo_root

_PYTHON = "/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3"
_TRAIN_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_rl_agent.py")
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKTREE_PREFIX = "launcher-"


# ── Pure utility functions ────────────────────────────────────────────────────

def find_latest_checkpoint(
    models_root: str,
    run_dir: "str | None" = None,
    min_mtime: float = 0.0,
) -> "str | None":
    if run_dir:
        latest_txt = os.path.join(run_dir, "latest.txt")
        if os.path.exists(latest_txt):
            with open(latest_txt) as f:
                name = f.read().strip()
            candidate = os.path.join(run_dir, name)
            if os.path.exists(candidate):
                return candidate

    zips = glob.glob(os.path.join(models_root, "**", "*.zip"), recursive=True)
    if min_mtime:
        zips = [p for p in zips if os.path.getmtime(p) >= min_mtime]
    if not zips:
        return None

    def _step_key(path: str) -> int:
        n = os.path.basename(path)
        m = re.search(r"(\d+)_steps\.zip$", n)
        if m:
            return int(m.group(1))
        m = re.search(r"forced_(\d+)_", n)
        if m:
            return int(m.group(1))
        return 0

    return max(zips, key=lambda p: (_step_key(p), os.path.getmtime(p)))


def _find_model_arg(args: list) -> "str | None":
    for i, a in enumerate(args):
        if a == "--model" and i + 1 < len(args):
            return args[i + 1]
    return None


def _read_checkpoint_git_hash(model_path: str) -> "str | None":
    """Read git_hash from the metadata.json saved alongside a checkpoint."""
    candidates = [
        os.path.dirname(os.path.abspath(model_path)),
        os.path.abspath(model_path),
    ]
    for d in candidates:
        meta = os.path.join(d, "metadata.json")
        if os.path.exists(meta):
            try:
                with open(meta) as f:
                    return json.load(f).get("git_hash")
            except Exception:
                return None
    return None


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


def _insert_or_replace_run_dir_arg(args: list, run_dir: str) -> list:
    out = []
    i = 0
    replaced = False
    while i < len(args):
        if args[i] == "--run-dir":
            out.extend(["--run-dir", run_dir])
            replaced = True
            i += 2
        else:
            out.append(args[i])
            i += 1
    if not replaced:
        out.extend(["--run-dir", run_dir])
    return out


def _strip_launcher_args(argv: list) -> list:
    """Strip launcher-only flags so they are not forwarded to train_rl_agent.py."""
    out = []
    i = 0
    while i < len(argv):
        if argv[i] == "--restart-interval-hours":
            i += 2
        elif argv[i].startswith("--restart-interval-hours="):
            i += 1
        elif argv[i] == "--no-pin":
            i += 1
        else:
            out.append(argv[i])
            i += 1
    return out


# ── Git ───────────────────────────────────────────────────────────────────────

def _git_hash() -> str:
    return get_git_hash(short=True)


# ── Worktree management ───────────────────────────────────────────────────────

def _prune_stale_launcher_worktrees(repo_root: str) -> None:
    """Remove any stale launcher-* worktrees left by crashed sessions."""
    result = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        capture_output=True, text=True, cwd=repo_root,
    )
    if result.returncode != 0:
        return
    for line in result.stdout.splitlines():
        if line.startswith("worktree "):
            path = line[len("worktree "):]
            if os.path.basename(path).startswith(_WORKTREE_PREFIX):
                subprocess.run(
                    ["git", "worktree", "remove", "--force", path],
                    capture_output=True, cwd=repo_root,
                )


def _create_run_worktree(git_hash: str) -> "tuple[str, str, callable]":
    """Create a detached git worktree pinned to git_hash.

    Returns (train_script_path, src_dir_path, cleanup_fn).
    Raises RuntimeError if git worktree add fails.
    """
    import tempfile

    repo_root = get_repo_root()
    tmp = tempfile.mkdtemp(prefix=f"{_WORKTREE_PREFIX}{git_hash[:8]}-")
    os.rmdir(tmp)  # git worktree add requires the target not to exist
    result = subprocess.run(
        ["git", "worktree", "add", "--detach", tmp, git_hash],
        capture_output=True, text=True, cwd=repo_root,
    )
    if result.returncode != 0:
        raise RuntimeError(f"git worktree add failed:\n{result.stderr.strip()}")

    # Replace the empty submodule placeholder with a symlink to the main repo's
    # fully-initialized pokemon-showdown checkout.  Node's require() needs the
    # whole directory (including package.json), not just dist/ + node_modules/.
    # Git-status correctness doesn't matter for this temporary /tmp worktree.
    ps_main = os.path.join(repo_root, "deps", "pokemon-showdown")
    ps_wt = os.path.join(tmp, "deps", "pokemon-showdown")
    if os.path.isdir(ps_wt) and not os.path.islink(ps_wt):
        os.rmdir(ps_wt)  # git leaves an empty placeholder directory
    if not os.path.lexists(ps_wt):
        os.symlink(ps_main, ps_wt)

    train_script = os.path.join(tmp, "src", "main", "train_rl_agent.py")
    src_dir = os.path.join(tmp, "src")

    def cleanup() -> None:
        subprocess.run(
            ["git", "worktree", "remove", "--force", tmp],
            capture_output=True, cwd=repo_root,
        )

    return train_script, src_dir, cleanup


# ── Exit summary / crash log ──────────────────────────────────────────────────

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


# ── Child lifecycle ───────────────────────────────────────────────────────────

def _build_child_env() -> dict:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _launch_child(
    child_args: list,
    child_env: dict,
    state: LauncherState,
    train_script: str,
    src_dir: str,
) -> subprocess.Popen:
    """Create metrics pipe, spawn child, start reader threads."""
    metrics_r, metrics_w = os.pipe()
    existing = child_env.get("PYTHONPATH", "")
    pythonpath = (src_dir + ":" + existing) if existing else src_dir
    proc = subprocess.Popen(
        [_PYTHON, train_script] + child_args,
        env={**child_env, "LAUNCHER_METRICS_FD": str(metrics_w), "PYTHONPATH": pythonpath},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        pass_fds=(metrics_w,),
    )
    os.close(metrics_w)  # parent closes write end after fork

    state.pid = proc.pid
    state.run_start = time.monotonic()

    threading.Thread(target=_read_metrics_pipe, args=(metrics_r, state), daemon=True).start()
    threading.Thread(target=_read_child_stdout, args=(proc, state), daemon=True).start()

    return proc


# ── Main loop ─────────────────────────────────────────────────────────────────

def run(child_args: list, interval_hours: float, pin: bool = True) -> None:
    interval_seconds = interval_hours * 3600
    session_start = time.time()
    child_env = _build_child_env()
    state = LauncherState(interval_hours=interval_hours)
    ui = LauncherUI()

    # ── Worktree pinning ──────────────────────────────────────────────────────
    if pin:
        repo_root = get_repo_root()
        _prune_stale_launcher_worktrees(repo_root)
        model_path = _find_model_arg(child_args)
        if model_path:
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

    # ── Fresh run: own the run directory from the start ───────────────────────
    if not _find_model_arg(child_args):
        ts = time.strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join("models", f"run_{ts}")
        os.makedirs(run_dir, exist_ok=True)
        child_args = _insert_or_replace_run_dir_arg(child_args, run_dir)
    else:
        run_dir = None  # resolved from checkpoint path after first child exits

    cmd_q: queue.Queue = queue.Queue()
    _setup_raw_input()
    threading.Thread(target=_read_keys, args=(cmd_q,), daemon=True).start()

    if interval_hours > 0:
        state.add_event(f"🚀 Starting — restart every {interval_hours:.1f}h")
    else:
        state.add_event("🚀 Starting — single run (no restart)")

    if pin:
        state.add_event(f"📌 Pinned to {state.initial_git_hash} (isolated worktree)")
    else:
        state.add_event(f"--no-pin: running from current tree ({state.initial_git_hash})")

    _run_dir_box: list = [run_dir]
    atexit.register(lambda: _print_exit_summary(_run_dir_box[0], state))

    with Live(refresh_per_second=2, screen=True) as live:
        while True:

            # ── Git drift check (informational only — worktree is immune) ──
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

            # ── Launch child ───────────────────────────────────────────────
            proc = _launch_child(child_args, child_env, state, train_script, src_dir)
            state.add_event(f"✅ Child PID {proc.pid} started")

            deadline = state.run_start + interval_seconds if interval_hours > 0 else float("inf")
            state.deadline = deadline

            flags = _PollFlags()

            # ── Poll loop ──────────────────────────────────────────────────
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

            # ── Decode exit code ───────────────────────────────────────────
            rc = proc.returncode

            if rc == TrainExitCode.COMPLETE:
                state.add_event("✅ Training complete — all steps done")
                live.update(ui.render(state.snapshot(), live.console.height))
                time.sleep(1)
                sys.exit(0)

            if rc != TrainExitCode.INTERRUPTED:
                # Crash or unknown exit
                state.add_event(f"🛑 Child crashed (exit {rc}) — not restarting")
                live.update(ui.render(state.snapshot(), live.console.height))
                time.sleep(2)
                atexit.register(_print_crash_log, state.snapshot().log_lines)
                sys.exit(rc)

            # rc == TrainExitCode.INTERRUPTED (graceful save + restart)
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
            state.add_event(f"✅ Restarting from {os.path.basename(checkpoint)}")
            child_args = _insert_or_replace_model_arg(child_args, checkpoint)
            child_args = _insert_or_replace_run_dir_arg(child_args, run_dir)
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
    parser.add_argument(
        "--no-pin",
        action="store_true",
        default=False,
        help="Skip worktree pinning — child runs from the current working tree (old behaviour)",
    )
    parser.add_argument("-h", "--help", action="store_true")

    known, _ = parser.parse_known_args()

    if known.help:
        parser.print_help()
        print("\nAll other arguments are forwarded to train_rl_agent.py.")
        sys.exit(0)

    child_args = _strip_launcher_args(sys.argv[1:])
    try:
        run(child_args, interval_hours=known.restart_interval_hours, pin=not known.no_pin)
    except KeyboardInterrupt:
        sys.exit(0)


if __name__ == "__main__":
    main()
