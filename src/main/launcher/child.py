"""Child process lifecycle: environment setup, spawning, and IPC pipe readers."""

import json
import os
import subprocess
import threading
import time

from main.launcher.state import LauncherState

_PYTHON = "/home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3"
_MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAIN_SCRIPT = os.path.join(_MAIN_DIR, "train_rl_agent.py")
_SRC_DIR = os.path.dirname(_MAIN_DIR)


def _build_child_env() -> dict:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    # Prevent PyTorch from spawning extra threads inside each SubprocVecEnv worker.
    # With 64 workers, the default (1 thread per core) creates hundreds of competing
    # threads and kills throughput.
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    # Use the CUDA caching allocator's expandable-segments mode so freed activation
    # blocks (e.g. from gradient checkpointing, or between the rollout/update phases) are
    # reusable rather than stranded by fragmentation — reclaims headroom on the 12GB card
    # at no compute cost. setdefault so an explicit override always wins; no-op on CPU.
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    return env


def _read_metrics_pipe(fd_r: int, state: LauncherState) -> None:
    try:
        with os.fdopen(fd_r, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if "_event" in data:
                        state.add_event(data["_event"])
                    else:
                        state.update_metrics(data)
                except json.JSONDecodeError:
                    pass
    except Exception:
        pass


def child_log_path(run_dir: "str | None") -> "str | None":
    """Path of the persisted child-output log for a run, or None if no run_dir."""
    if not run_dir:
        return None
    return os.path.join(run_dir, "launcher_child.log")


def _open_child_log(state: LauncherState):
    """Open (append) the persistent child-output log in the run directory.

    Streaming every line to disk as it arrives means a crash — even a hard
    ``os._exit`` in the child that bypasses Python cleanup — still leaves a
    complete log behind in the model directory. Returns None if there is no
    run_dir yet or the file can't be opened (logging is best-effort, never fatal).
    """
    path = child_log_path(state.run_dir)
    if not path:
        return None
    try:
        os.makedirs(state.run_dir, exist_ok=True)
        f = open(path, "a", buffering=1, errors="replace")  # line-buffered
        f.write(f"\n===== child attached {time.strftime('%Y-%m-%d %H:%M:%S')} (pid {state.pid}) =====\n")
        return f
    except Exception:
        return None


def _read_child_stdout(proc: subprocess.Popen, state: LauncherState, log_file=None) -> None:
    try:
        for raw in proc.stdout:
            line = raw.decode(errors="replace").rstrip()
            state.add_log(line)
            if log_file is not None:
                try:
                    log_file.write(line + "\n")
                except Exception:
                    pass
            if "[CHECKPOINT]" in line:
                # Surface the save confirmation in the events panel too.
                fname = line.split("→")[-1].strip() if "→" in line else line
                state.add_event(f"💾 Checkpoint saved → {os.path.basename(fname)}")
    except Exception:
        pass
    finally:
        if log_file is not None:
            try:
                log_file.flush()
                log_file.close()
            except Exception:
                pass


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
    state.mark_activity()  # fresh stall-watchdog clock for the new child

    log_file = _open_child_log(state)
    threading.Thread(target=_read_metrics_pipe, args=(metrics_r, state), daemon=True).start()
    threading.Thread(target=_read_child_stdout, args=(proc, state, log_file), daemon=True).start()

    return proc
