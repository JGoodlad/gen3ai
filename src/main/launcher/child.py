"""Child process lifecycle: environment setup, spawning, and IPC pipe readers."""

import json
import os
import subprocess
import sys
import threading
import time

from main.launcher.state import LauncherState

_MAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TRAIN_SCRIPT = os.path.join(_MAIN_DIR, "train_rl_agent.py")
_SRC_DIR = os.path.dirname(_MAIN_DIR)

#: Env var that pins the training child's interpreter explicitly.
PYTHON_ENV_VAR = "GEN3AI_PYTHON"


def resolve_child_python() -> str:
    """The interpreter the training child is spawned with.

    Precedence: ``$GEN3AI_PYTHON`` (explicit override) → ``sys.executable`` (the
    launcher's OWN interpreter). The default is right on every machine, because the
    child then runs in whatever environment the launcher was started from — there is no
    env name, conda prefix or absolute path to keep in sync, and a fresh clone needs no
    edit. Set ``$GEN3AI_PYTHON`` only to run the child under a *different* interpreter
    than the parent deliberately.

    Resolved at spawn time rather than import time, so a periodic/crash restart picks up
    a changed override.
    """
    override = os.environ.get(PYTHON_ENV_VAR, "").strip()
    return override or sys.executable


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
    """Path of the persisted child-output log for a run, or None if no run_dir.

    This is the RING BUFFER — the recent tail, the file the TUI and every crash dump read."""
    if not run_dir:
        return None
    return os.path.join(run_dir, "launcher_child.log")


def child_full_log_path(run_dir: "str | None") -> "str | None":
    """Path of the ROTATING full copy of the child's output, or None if no run_dir.

    ``launcher_child.full.log``, with older generations at ``.1`` … ``._FULL_LOG_BACKUPS``."""
    if not run_dir:
        return None
    return os.path.join(run_dir, "launcher_child.full.log")


# Cap the on-disk child log to a disk ring buffer. The in-memory scrollback (state.py's
# deque) was already bounded, but the file was append-only and grew without limit across a
# long multi-restart run (observed 1 GiB+). Keep the recent tail only.
_CHILD_LOG_MAX_BYTES = 1024 * 1024   # ~1 MiB high-water mark


class _CappedChildLog:
    """Best-effort, size-capped append log for the child's stdout — a disk ring buffer.

    Streams every line line-buffered, so a hard child ``os._exit`` (bypassing Python
    cleanup) still leaves the recent output on disk. When the file passes ``max_bytes`` it
    is rewritten keeping only the most recent ~half (aligned to a line boundary), so it
    never grows unbounded. An already-oversized file (e.g. a legacy multi-GB log) is trimmed
    on open. Used by a single reader thread, so no locking is needed."""

    def __init__(self, path: str, max_bytes: int = _CHILD_LOG_MAX_BYTES) -> None:
        self.path = path
        self.max_bytes = max_bytes
        self._trim()  # shrink a pre-existing oversized file before we start appending
        self._f = open(path, "a", buffering=1, errors="replace")
        self._size = self._disk_size()

    def _disk_size(self) -> int:
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0

    def write(self, s: str) -> None:
        self._f.write(s)
        self._size += len(s.encode("utf-8", "replace"))
        if self._size >= self.max_bytes:
            self._rotate_tail()

    def _rotate_tail(self) -> None:
        try:
            self._f.flush()
            self._f.close()
        except Exception:
            pass
        self._trim()
        try:
            self._f = open(self.path, "a", buffering=1, errors="replace")
        except Exception:
            self._f = open(os.devnull, "a")
        self._size = self._disk_size()

    def _trim(self) -> None:
        """Rewrite the file to keep only its last ~max_bytes/2 (drop a partial first line)."""
        size = self._disk_size()
        if size <= self.max_bytes:
            return
        keep = self.max_bytes // 2
        try:
            with open(self.path, "rb") as r:
                r.seek(size - keep)
                tail = r.read()
            nl = tail.find(b"\n")
            if nl != -1:
                tail = tail[nl + 1:]
            header = (
                f"[ … older launcher_child.log lines trimmed — ring buffer ~"
                f"{self.max_bytes // 1024} KiB … ]\n"
            ).encode("utf-8", "replace")
            with open(self.path, "wb") as w:
                w.write(header)
                w.write(tail)
        except Exception:
            pass

    def flush(self) -> None:
        try:
            self._f.flush()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._f.flush()
            self._f.close()
        except Exception:
            pass


# ── The ROTATING full copy (`launcher_child.full.log`) ───────────────────────────────────────
#
# WHY IT EXISTS. The ring buffer above keeps ~1 MiB — deliberately, after a repaint storm wrote a
# 982 MB `launcher_child.log`. But a ring that trims SILENTLY also destroys evidence: on
# 2026-09-06 the Training Run session counted per-worker compile lines across a restart and the
# number became unrecoverable the moment the ring wrapped, so the count had to be settled from
# source instead. **A read that cannot be redone is a read that cannot be checked.**
#
# THE CAPS, and why these numbers. 64 MiB per file × (1 live + 7 rotations) = **512 MiB maximum on
# disk, per run, forever**. Chosen against the incident that produced the ring, not against a
# typical run:
#
#   * the 982 MB storm would fill this rotation and STOP at 512 MiB — bounded, roughly half the
#     incident, and the storm's own tail is still the file you open first;
#   * a normal 3-hour restart cycle writes single-digit MiB, so the rotation never engages at all
#     and the full log simply IS the whole run;
#   * 64 MiB is a file a reader can actually grep, `less` and copy off the box, which a single
#     512 MiB file is not — the point is a redoable READ, not an archive;
#   * 8 files is the smallest count that keeps a whole storm plus the normal output that preceded
#     it. Fewer, and a storm evicts exactly the context that explains it.
#
# 🚨 UNBOUNDED IS NOT AN OPTION HERE. The disk this writes to also holds `models/`; the 982 MB log
# is the proof that a training child can emit at a rate no human notices until the volume is full.
_FULL_LOG_MAX_BYTES = 64 * 1024 * 1024   # 64 MiB per generation
_FULL_LOG_BACKUPS = 7                    # + the live file = 8 files = 512 MiB hard ceiling


class _RotatingChildLog:
    """Best-effort size-capped ROTATING log — the full child output beside the ring buffer.

    Same streaming discipline as `_CappedChildLog` (line-buffered, so a hard ``os._exit`` still
    leaves everything written on disk) and the same never-raise contract. The difference is what
    happens at the cap: this one ROTATES (``.log`` → ``.log.1`` → … → ``.log.7``, oldest deleted)
    instead of throwing the old bytes away, so a count taken across a restart can be REDONE.

    Used by a single reader thread, so no locking is needed.
    """

    def __init__(self, path: str, max_bytes: int = _FULL_LOG_MAX_BYTES,
                 backups: int = _FULL_LOG_BACKUPS) -> None:
        self.path = path
        self.max_bytes = max_bytes
        self.backups = backups
        # A pre-existing file already at/over the cap is rotated BEFORE we append, mirroring the
        # ring's trim-on-open: a restart must not push one generation past the per-file bound.
        if self._disk_size() >= self.max_bytes:
            self._rotate()
        self._f = open(path, "a", buffering=1, errors="replace")
        self._size = self._disk_size()

    def _disk_size(self) -> int:
        try:
            return os.path.getsize(self.path)
        except OSError:
            return 0

    def write(self, s: str) -> None:
        try:
            self._f.write(s)
        except Exception:
            return
        self._size += len(s.encode("utf-8", "replace"))
        if self._size >= self.max_bytes:
            self._roll()

    def _roll(self) -> None:
        """Close, shift the generations, reopen at zero."""
        try:
            self._f.flush()
            self._f.close()
        except Exception:
            pass
        self._rotate()
        try:
            self._f = open(self.path, "a", buffering=1, errors="replace")
        except Exception:
            # Losing the full log must never cost us the ring or the run.
            self._f = open(os.devnull, "a")
        self._size = self._disk_size()

    def _rotate(self) -> None:
        """``.6``→``.7``, … , ``.1``→``.2``, base→``.1``; whatever was at ``.7`` is DELETED.

        Deleting the oldest generation is what makes the ceiling hard. Walked newest-last so no
        rename can clobber a generation that has not moved yet.
        """
        try:
            oldest = f"{self.path}.{self.backups}"
            if os.path.exists(oldest):
                os.remove(oldest)
            for n in range(self.backups - 1, 0, -1):
                src, dst = f"{self.path}.{n}", f"{self.path}.{n + 1}"
                if os.path.exists(src):
                    os.replace(src, dst)
            if os.path.exists(self.path):
                os.replace(self.path, f"{self.path}.1")
        except Exception:
            pass

    def flush(self) -> None:
        try:
            self._f.flush()
        except Exception:
            pass

    def close(self) -> None:
        try:
            self._f.flush()
            self._f.close()
        except Exception:
            pass


class _ChildLogFanout:
    """Writes every line to BOTH sinks — the ring buffer and the rotating full copy.

    🚨 THE RING IS UNTOUCHED. It is what the TUI reads, what the crash dump tails and what every
    "the full log path is printed on exit" line names; the full copy is strictly ADDITIVE. The
    ring is written FIRST on every call, so a failure in the (larger, rotating) full log can never
    cost a byte of the surface anybody currently depends on.

    ``path`` is the ring's path, so callers that print "log written to …" are unchanged.
    """

    def __init__(self, ring, full=None) -> None:
        self._ring = ring
        self._full = full
        self.path = ring.path

    @property
    def full_path(self) -> "str | None":
        return self._full.path if self._full is not None else None

    def write(self, s: str) -> None:
        self._ring.write(s)
        if self._full is not None:
            try:
                self._full.write(s)
            except Exception:
                pass

    def flush(self) -> None:
        self._ring.flush()
        if self._full is not None:
            try:
                self._full.flush()
            except Exception:
                pass

    def close(self) -> None:
        self._ring.close()
        if self._full is not None:
            try:
                self._full.close()
            except Exception:
                pass


def _open_child_log(state: LauncherState):
    """Open the size-capped persistent child-output log in the run directory.

    Streaming every line to disk as it arrives means a crash — even a hard ``os._exit`` in
    the child that bypasses Python cleanup — still leaves the recent output behind. The log
    is ring-buffered to ``_CHILD_LOG_MAX_BYTES`` (a pre-existing oversized file is trimmed on
    open). Returns None if there is no run_dir yet or the file can't be opened (logging is
    best-effort, never fatal).

    TWO SINKS, one stream. The ring (``launcher_child.log``) is unchanged in every respect — same
    path, same cap, same trim — and a second, ROTATING copy (``launcher_child.full.log``, 64 MiB ×
    8) rides beside it so a count taken across a restart survives the ring wrapping. The full copy
    is strictly optional: if it cannot be opened, the ring alone is returned and nothing else
    changes."""
    path = child_log_path(state.run_dir)
    if not path:
        return None
    try:
        os.makedirs(state.run_dir, exist_ok=True)
        ring = _CappedChildLog(path)
    except Exception:
        return None
    full = None
    try:
        full_path = child_full_log_path(state.run_dir)
        if full_path:
            full = _RotatingChildLog(full_path)
    except Exception:
        full = None
    log = _ChildLogFanout(ring, full)
    try:
        log.write(f"\n===== child attached {time.strftime('%Y-%m-%d %H:%M:%S')} (pid {state.pid}) =====\n")
    except Exception:
        pass
    return log


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
    # 🚨 THIS PYTHONPATH IS WORKTREE-ISOLATION MACHINERY — do not "clean it up".
    # src_dir is the PINNED worktree's src/, and the Popen below passes no cwd=, so this
    # export is the ONLY thing making a resumed run import the code its checkpoint was
    # saved on. Measured (2026-08-22 scope survey, Finding B): with an editable install
    # present and no PYTHONPATH, a pinned old-commit child imports `agents` from the MAIN
    # checkout — i.e. an old checkpoint silently resumes on current HEAD, the arch-drift
    # disaster class. PYTHONPATH entries land in sys.path BEFORE a .pth's, so the pin and
    # an editable install coexist correctly exactly as long as this line stays.
    # The child stays in the launcher's session (no start_new_session): a closed
    # tmux/SSH terminal SIGHUPs the whole group, and train_rl_agent now handles SIGHUP
    # itself (checkpoints, like SIGTERM), so it saves before exiting. The launcher also
    # installs its own SIGHUP/SIGTERM handler (app.py) so it tears down cleanly rather
    # than dying abruptly — the two are complementary backstops.
    proc = subprocess.Popen(
        [resolve_child_python(), train_script] + child_args,
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
