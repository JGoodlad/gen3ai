"""Process-level restart loop for train_rl_agent.py.

Periodically SIGTERMs the training child and relaunches from the most recently
saved checkpoint, recovering FPS lost to pymalloc fragmentation over long runs.
"""

import glob
import os
import queue
import signal
import subprocess
import sys
import threading
import time

_PYTHON = "/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3"
_TRAIN_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "train_rl_agent.py")
_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def find_latest_checkpoint(models_root: str) -> "str | None":
    zips = glob.glob(os.path.join(models_root, "**", "*.zip"), recursive=True)
    return max(zips, key=os.path.getmtime) if zips else None


def _insert_or_replace_model_arg(args: list[str], checkpoint: str) -> list[str]:
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


def _strip_launcher_arg(argv: list[str]) -> list[str]:
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


def _read_stdin(cmd_q: queue.Queue) -> None:
    try:
        for line in sys.stdin:
            cmd_q.put(line.strip().lower())
    except Exception:
        pass


def run(child_args: list[str], interval_hours: float) -> None:
    interval_seconds = interval_hours * 3600

    child_env = os.environ.copy()
    existing_pythonpath = child_env.get("PYTHONPATH", "")
    child_env["PYTHONPATH"] = (_SRC_DIR + ":" + existing_pythonpath) if existing_pythonpath else _SRC_DIR

    cmd_q: queue.Queue = queue.Queue()
    threading.Thread(target=_read_stdin, args=(cmd_q,), daemon=True).start()

    if interval_hours > 0:
        print(f"🚀 [Launcher] Starting — restart every {interval_hours:.1f}h")
    else:
        print("🚀 [Launcher] Starting — single run (no restart)")

    while True:
        proc = subprocess.Popen([_PYTHON, _TRAIN_SCRIPT] + child_args, env=child_env)
        print(f"✅ [Launcher] Child PID {proc.pid} started")

        start_time = time.monotonic()
        deadline = start_time + interval_seconds if interval_hours > 0 else float("inf")
        sigterm_sent = False
        quit_requested = False
        restart_requested = False

        # Poll until child exits, handling the deadline and stdin commands.
        while True:
            now = time.monotonic()
            remaining = deadline - now

            if not sigterm_sent and remaining <= 0:
                print(f"⏰ [Launcher] {interval_hours:.1f}h elapsed — restarting child...")
                try:
                    os.kill(proc.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                sigterm_sent = True

            poll_timeout = min(1.0, max(0.01, remaining if remaining > 0 else 1.0))
            try:
                proc.wait(timeout=poll_timeout)
                break
            except subprocess.TimeoutExpired:
                pass

            while not cmd_q.empty():
                cmd = cmd_q.get_nowait()
                if cmd in ("c", "checkpoint"):
                    try:
                        os.kill(proc.pid, signal.SIGUSR1)
                        print("💾 [Launcher] Checkpoint signal sent to child.")
                    except ProcessLookupError:
                        print("💾 [Launcher] Child already exited.")
                elif cmd in ("r", "restart"):
                    print("♻️  [Launcher] Restart requested — sending SIGTERM to child...")
                    restart_requested = True
                    if not sigterm_sent:
                        try:
                            os.kill(proc.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        sigterm_sent = True
                elif cmd in ("q", "quit"):
                    print("👋 [Launcher] Quit requested — waiting for child to save...")
                    quit_requested = True
                    if not sigterm_sent:
                        try:
                            os.kill(proc.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        sigterm_sent = True
                elif cmd in ("s", "status"):
                    elapsed = time.monotonic() - start_time
                    if interval_hours > 0:
                        time_left = max(0.0, deadline - time.monotonic())
                        print(
                            f"📊 [Launcher] PID {proc.pid} | "
                            f"elapsed {elapsed / 3600:.2f}h | "
                            f"restart in {time_left / 3600:.2f}h"
                        )
                    else:
                        print(f"📊 [Launcher] PID {proc.pid} | elapsed {elapsed / 3600:.2f}h | no restart")
                elif cmd:
                    print(f"❓ [Launcher] Unknown command '{cmd}'. "
                          "Commands: c/checkpoint, r/restart, q/quit, s/status")

        proc.wait()

        if proc.returncode != 0:
            print(f"🛑 [Launcher] Child crashed (exit {proc.returncode}) — not restarting")
            sys.exit(proc.returncode)

        if quit_requested:
            print("👋 [Launcher] Quit complete.")
            sys.exit(0)

        if interval_hours <= 0 and not restart_requested:
            sys.exit(0)

        checkpoint = find_latest_checkpoint("models")
        if checkpoint is None:
            print("🛑 [Launcher] No checkpoint found under models/ — cannot restart")
            sys.exit(1)

        print(f"✅ [Launcher] Restarting from {checkpoint}")
        child_args = _insert_or_replace_model_arg(child_args, checkpoint)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Launcher: wraps train_rl_agent.py with periodic full-process restart.",
        add_help=False,
    )
    parser.add_argument("--restart-interval-hours", type=float, default=3.0,
                        help="Restart the training process every N hours (0 = run once)")
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
