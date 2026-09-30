"""The box's GPU lock — the ONE way to take it, and it is RE-ENTRANT across processes.

USAGE
    # Python, in-process (a nested ``with`` in a child of a holder is a no-op, not a deadlock):
    from utils.gpu_lock import gpu_lock
    with gpu_lock():
        ...                                   # children inherit GEN3AI_GPU_LOCK_HELD=<this pid>

    # a shell, around one command:
    scripts/ops/gpu_lock.sh python -m main.rust_core_m5 gates --tier milestone --gpu
    python -m utils.gpu_lock -- <cmd> [args...]          # the same, without the wrapper
    python -m utils.gpu_lock --status                     # who holds it now

WHY. On 2026-09-30 an agent wrapped ``python -m main.rust_core_m5 gates … --gpu`` in
``flock ~/.claude/jobs/gpu.lock``. ``gates.py`` takes that same lock itself around its GPU pytest, and a
``flock`` on a NEW open of the file is a different lock owner, so the inner flock waited 15 min at 0%
CPU on a lock its own ancestor held. Nothing could ever release it. The class fix:

* **Holding the lock exports** ``GEN3AI_GPU_LOCK_HELD=<holder pid>`` to every child.
* **A taker that sees the marker does NOT re-acquire — after VERIFYING it**: the marker's pid must be
  this process or one of its ANCESTORS, and ``/proc/locks`` must list that pid HOLDING this lock file.
  A stale marker (inherited into a process that outlived its holder, or copied into an unrelated
  environment) is ignored with a line on stderr, and the lock is taken normally.
* **A taker that finds the lock held checks the HOLDER** at first contention and again every
  ``report_every_s`` (declared: 60 s): a holder that is this process's ancestor — e.g. a bare
  ``flock`` around this command — raises :class:`GpuLockSelfDeadlock` naming both pids at once;
  any other holder is waited for, with its pid and command line printed each interval.

The lock path is ``~/.claude/jobs/gpu.lock``; ``$GEN3AI_GPU_LOCK`` overrides (tests point it at a
temp file — never at the real lock). A waiter BLOCKS in the kernel exactly as a bare ``flock`` does
(so ``/proc/locks`` and ``scripts/ops/idle_waiter_watchdog.py`` see it); the holder is first checked
``poll_s`` (0.5 s) into the wait. Linux only (``/proc``).
"""
from __future__ import annotations

import contextlib
import fcntl
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Callable, Iterator, List, Mapping, Optional, Sequence, Union

from utils import procfs

GPU_LOCK_ENV = "GEN3AI_GPU_LOCK"
HELD_ENV = "GEN3AI_GPU_LOCK_HELD"
REPORT_EVERY_S = 60.0
POLL_S = 0.5

PathLike = Union[str, "os.PathLike[str]"]


def lock_path() -> Path:
    """The box's GPU lock file (``$GEN3AI_GPU_LOCK`` overrides)."""
    return Path(os.environ.get(GPU_LOCK_ENV) or Path.home() / ".claude" / "jobs" / "gpu.lock")


class GpuLockSelfDeadlock(RuntimeError):
    """The lock is held by this process's own ANCESTOR, which is waiting (directly or not) on this
    process — the wait can never end. Take the outer lock through ``utils.gpu_lock`` /
    ``scripts/ops/gpu_lock.sh`` instead of a bare ``flock``, and the inner take becomes a no-op."""

    def __init__(self, path: PathLike, holder_pid: int, waiter_pid: int, holder_cmd: str) -> None:
        self.path, self.holder_pid, self.waiter_pid, self.holder_cmd = str(path), holder_pid, waiter_pid, holder_cmd
        super().__init__(
            f"GPU lock SELF-DEADLOCK: pid {waiter_pid} wants {path}, which is held by pid {holder_pid} — "
            f"its own ANCESTOR ({holder_cmd[:160]!r}), so it can never be released. The holder took it "
            f"without exporting {HELD_ENV} (a bare `flock`?); take the outer lock with "
            f"scripts/ops/gpu_lock.sh / utils.gpu_lock and the inner take is a no-op.")


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _self_and_ancestors() -> List[int]:
    me = os.getpid()
    return [me, *procfs.ancestors(me)]


def verified_holder(path: Optional[PathLike] = None, env: Optional[Mapping[str, str]] = None) -> Optional[int]:
    """The pid in ``$GEN3AI_GPU_LOCK_HELD`` iff it is this process or an ancestor AND ``/proc/locks``
    lists it holding ``path``; otherwise ``None`` (no marker, or a stale / forged one)."""
    raw = (os.environ if env is None else env).get(HELD_ENV, "")
    try:
        pid = int(raw)
    except ValueError:
        return None
    key = procfs.file_key(str(path or lock_path()))
    if key is None or pid not in _self_and_ancestors():
        return None
    return pid if pid in procfs.holders(key) else None


def holder_report(path: Optional[PathLike] = None) -> str:
    """One line naming who holds the lock now (pid + command line), for a waiter's progress print."""
    p = str(path or lock_path())
    key = procfs.file_key(p)
    hs = procfs.holders(key) if key else []
    if not hs:
        return f"{p}: not held"
    return f"{p}: held by " + ", ".join(f"pid {h} ({procfs.cmdline(h)[:160] or 'gone'!r})" for h in hs)


def _acquire(fd: int, path: Path, *, report_every_s: float, poll_s: float, timeout_s: Optional[float],
             log: Callable[[str], None], what: str) -> None:
    """Take an exclusive ``flock`` on ``fd``. The wait is a BLOCKING flock in a helper thread (so the
    waiter is listed in ``/proc/locks`` as a ``->`` waiter — which ``idle_waiter_watchdog.py`` reads —
    and is woken by the kernel), while this thread checks the holder at first contention and every
    ``report_every_s``. The helper blocks on a DUP of ``fd`` (same open file description, so the lock
    it takes is ``fd``'s); if the wait is abandoned it releases whatever it later gets."""
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return
    except BlockingIOError:
        pass
    t0 = time.monotonic()
    st = os.fstat(fd)
    key = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}"
    got = threading.Event()
    mu = threading.Lock()
    box: dict = {"abandoned": False, "err": None}
    tfd = os.dup(fd)

    def blocker() -> None:
        try:
            fcntl.flock(tfd, fcntl.LOCK_EX)
            with mu:
                if box["abandoned"]:
                    fcntl.flock(tfd, fcntl.LOCK_UN)
                else:
                    got.set()
        except OSError as e:           # pragma: no cover — a flock on an open fd does not fail
            box["err"] = e
            got.set()
        finally:
            os.close(tfd)

    threading.Thread(target=blocker, name="gpu_lock-wait", daemon=True).start()

    def give_up(exc: BaseException) -> None:
        with mu:
            if got.is_set() and box["err"] is None:
                return                 # it arrived while we decided — keep it
            box["abandoned"] = True
        raise exc

    next_check = t0 + min(poll_s, report_every_s)
    while True:
        deadline = next_check if timeout_s is None else min(next_check, t0 + timeout_s)
        if got.wait(timeout=max(0.0, deadline - time.monotonic())):
            if box["err"] is not None:
                raise box["err"]
            if time.monotonic() - t0 > report_every_s:
                log(f"[gpu_lock] acquired {path}{what} after {time.monotonic() - t0:.0f} s")
            return
        now = time.monotonic()
        if now >= next_check:
            chain = _self_and_ancestors()
            hs = procfs.holders(key)
            mine = [h for h in hs if h in chain]
            if mine:
                give_up(GpuLockSelfDeadlock(path, mine[0], os.getpid(), procfs.cmdline(mine[0])))
                return
            who = ", ".join(f"pid {h} ({procfs.cmdline(h)[:160] or 'gone'!r})" for h in hs) or "an unlisted holder"
            log(f"[gpu_lock] waiting {now - t0:.0f} s for {path}{what}: held by {who}")
            next_check = now + report_every_s
        if timeout_s is not None and now - t0 >= timeout_s:
            give_up(TimeoutError(f"[gpu_lock] gave up on {path} after {now - t0:.0f} s ({holder_report(path)})"))
            return


@contextlib.contextmanager
def gpu_lock(path: Optional[PathLike] = None, *, report_every_s: float = REPORT_EVERY_S, poll_s: float = POLL_S,
             timeout_s: Optional[float] = None, log: Callable[[str], None] = _stderr,
             what: str = "") -> Iterator[int]:
    """Hold the GPU lock for the ``with`` body; yields the HOLDER's pid (this process, or the verified
    ancestor that already holds it). While held, ``os.environ[HELD_ENV]`` names the holder, so every
    subprocess started in the body inherits it; the previous value is restored on exit."""
    p = Path(path) if path is not None else lock_path()
    what = f" ({what})" if what else ""
    held = verified_holder(p)
    if held is not None:
        yield held             # re-entrant: an ancestor (or this process) already holds it
        return
    if os.environ.get(HELD_ENV):
        log(f"[gpu_lock] ignoring a STALE {HELD_ENV}={os.environ[HELD_ENV]} (not a live ancestor holding {p})")
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(p), os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o666)
    try:
        _acquire(fd, p, report_every_s=report_every_s, poll_s=poll_s, timeout_s=timeout_s, log=log, what=what)
        before = os.environ.get(HELD_ENV)
        os.environ[HELD_ENV] = str(os.getpid())
        try:
            yield os.getpid()
        finally:
            if before is None:
                os.environ.pop(HELD_ENV, None)
            else:
                os.environ[HELD_ENV] = before
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


def run_under_lock(cmd: Sequence[str], **kw) -> int:
    """Run ``cmd`` as a CHILD while this process holds the lock (so the lock dies with this process,
    never with a daemonised grandchild); SIGINT/SIGTERM/SIGHUP are forwarded. Returns the child's exit
    code, or 128+N when a signal killed it."""
    with gpu_lock(**kw):
        child = subprocess.Popen(list(cmd))
        fwd = {s: signal.signal(s, lambda n, _f: child.send_signal(n)) for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        try:
            rc = child.wait()
        finally:
            for s, h in fwd.items():
                signal.signal(s, h)
    return 128 - rc if rc < 0 else rc


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in ("-h", "--help"):
        print(__doc__)
        return 0 if args else 2
    if args[0] == "--status":
        print(holder_report())
        return 0
    kw: dict = {}
    while args and args[0] != "--":
        flag = args.pop(0)
        if flag in ("--report-every-s", "--timeout-s", "--poll-s") and args:
            kw[flag[2:].replace("-", "_")] = float(args.pop(0))
        else:
            print(f"[gpu_lock] unknown option {flag!r} (put the command after --)", file=sys.stderr)
            return 2
    cmd = args[1:]
    if not cmd:
        print("[gpu_lock] REFUSING: no command after --", file=sys.stderr)
        return 2
    try:
        return run_under_lock(cmd, **kw)
    except GpuLockSelfDeadlock as e:
        print(str(e), file=sys.stderr)
        return 3
    except TimeoutError as e:
        print(str(e), file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
