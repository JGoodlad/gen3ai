"""The box's TEST-GATE SEMAPHORE — at most N routine gates run at once; RE-ENTRANT across processes.

USAGE
    scripts/ops/gate_lock.sh python3 -m pytest src/ -m "not slow and not e2e" -q -n 6
    python -m utils.gate_lock [--timeout-s S] -- <cmd> [args...]    # the same, without the wrapper
    python -m utils.gate_lock --status                               # who holds each slot now

    from utils.gate_lock import gate_slot
    with gate_slot():                          # children inherit GEN3AI_GATE_LOCK_HELD=<pid>:<slot>
        ...

WHY. On 2026-09-30 seven agents each ran a routine gate (``pytest -n 2``) at once, beside a compile
inventory and a GPU benchmark: load ~36 on 8 cores / 16 threads. Every gate crawled, tier budgets
tripped and benchmarks were contaminated — and none of it was a property of the code. ONE gate at a
time, at ``-n 6``, is the measured policy (2026-10-01, quiet box, full routine gate): two gates at
``-n 4`` at once took 691 s and 694 s EACH and burned 2.2x their solo CPU, while one at ``-n 6`` takes
259 s — so a queued second gate is done in ~8.6 min instead of 11.5. ``designs/ops/testing.md`` has the
table. When a training run is LIVE its env workers share the 8 cores: every take prints a one-line
warning naming the run and recommending ``-n 4`` (:func:`live_run_warning`) — a warning only, it
never changes the command.

HOW. ``N`` slot files (``slot0.lock`` … ``slot{N-1}.lock`` under ``~/.claude/jobs/gate_slots/``;
``$GEN3AI_GATE_LOCK_DIR`` overrides, and tests point it at a temp dir — never at the real one). A taker
tries every slot NON-BLOCKING; when all are taken it BLOCKS on one (``pid % N``) in the kernel, so it is
listed in ``/proc/locks`` as a waiter exactly as a bare ``flock`` is, while re-trying the others every
``poll_s``. A slot is an ``flock``, so a crashed holder's slot frees the instant it dies.

``N`` is DECLARED: :data:`GATE_SLOTS` (1); ``$GEN3AI_GATE_SLOTS`` overrides for a box that is yours.

RE-ENTRANT, the ``utils.gpu_lock`` way: holding a slot exports ``GEN3AI_GATE_LOCK_HELD=<pid>:<slot>``;
a taker that sees it does NOT take a second slot — after VERIFYING that the pid is itself or an
ANCESTOR and that ``/proc/locks`` lists it holding that slot. A stale marker is ignored with a line on
stderr. A taker all of whose slots are held by its own ancestors (a bare ``flock`` around it) raises
:class:`GateLockSelfDeadlock` at once.

Every take LOGS its wait and when it acquired; every release logs how long it held. TIMEOUTS, as for
the GPU lock: ``--timeout-s`` bounds the WAIT for a slot (exit 4); a WALL timeout on the gate goes
INSIDE — ``gate_lock.sh timeout 3000 <cmd>`` — never around it (``timeout 3000 gate_lock.sh <cmd>``
counts queue time and kills a gate that never ran).
Linux only (``/proc``).
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
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

from utils import procfs

GATE_SLOTS = 1
SLOTS_ENV = "GEN3AI_GATE_SLOTS"
DIR_ENV = "GEN3AI_GATE_LOCK_DIR"
HELD_ENV = "GEN3AI_GATE_LOCK_HELD"
REPORT_EVERY_S = 60.0
POLL_S = 0.5


def slot_count() -> int:
    """The declared N (``$GEN3AI_GATE_SLOTS`` overrides; a malformed or < 1 value is refused)."""
    raw = os.environ.get(SLOTS_ENV)
    if not raw:
        return GATE_SLOTS
    n = int(raw)
    if n < 1:
        raise ValueError(f"{SLOTS_ENV}={raw!r}: need at least one slot")
    return n


def lock_dir() -> Path:
    return Path(os.environ.get(DIR_ENV) or Path.home() / ".claude" / "jobs" / "gate_slots")


def slot_path(i: int, d: Optional[Path] = None) -> Path:
    return (d or lock_dir()) / f"slot{i}.lock"


class GateLockSelfDeadlock(RuntimeError):
    """Every slot is held by this process's own ancestors — the wait can never end."""


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _chain() -> List[int]:
    me = os.getpid()
    return [me, *procfs.ancestors(me)]


def _holders(path: Path) -> List[int]:
    key = procfs.file_key(str(path))
    return procfs.holders(key) if key else []


def verified_holder(d: Optional[Path] = None) -> Optional[Tuple[int, int]]:
    """``(pid, slot)`` from ``$GEN3AI_GATE_LOCK_HELD`` iff the pid is this process or an ancestor AND
    holds that slot's file in ``/proc/locks``; else ``None``."""
    raw = os.environ.get(HELD_ENV, "")
    try:
        pid_s, slot_s = raw.split(":")
        pid, slot = int(pid_s), int(slot_s)
    except ValueError:
        return None
    if pid not in _chain():
        return None
    return (pid, slot) if pid in _holders(slot_path(slot, d)) else None


def status(d: Optional[Path] = None) -> str:
    lines = []
    for i in range(slot_count()):
        hs = _holders(slot_path(i, d))
        who = ", ".join(f"pid {h} ({procfs.cmdline(h)[:120] or 'gone'!r})" for h in hs) or "free"
        lines.append(f"slot {i}: {who}")
    return "\n".join(lines)


def _try(fd: int) -> bool:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def _acquire(fds: List[int], paths: List[Path], *, report_every_s: float, poll_s: float,
             timeout_s: Optional[float], log: Callable[[str], None]) -> Tuple[int, float]:
    """Take one slot; returns ``(slot index, seconds waited)``."""
    t0 = time.monotonic()
    for i, fd in enumerate(fds):
        if _try(fd):
            return i, 0.0
    n = len(fds)
    home = os.getpid() % n
    got = threading.Event()
    mu = threading.Lock()
    box = {"abandoned": False}
    tfd = os.dup(fds[home])

    def blocker() -> None:                 # a kernel-blocking wait on the home slot
        try:
            fcntl.flock(tfd, fcntl.LOCK_EX)
            with mu:
                if box["abandoned"]:
                    fcntl.flock(tfd, fcntl.LOCK_UN)
                else:
                    got.set()
        finally:
            os.close(tfd)

    threading.Thread(target=blocker, name="gate_lock-wait", daemon=True).start()

    def abandon() -> bool:
        """Stop the blocker; False when it already won (then the home slot is ours)."""
        with mu:
            if got.is_set():
                return False
            box["abandoned"] = True
            return True

    next_report = t0 + report_every_s
    checked_self = False
    while True:
        if got.wait(timeout=poll_s):
            return home, time.monotonic() - t0
        for i, fd in enumerate(fds):
            if i != home and _try(fd):
                if abandon():
                    return i, time.monotonic() - t0
                fcntl.flock(fd, fcntl.LOCK_UN)        # the home slot won the race: keep that one
                return home, time.monotonic() - t0
        now = time.monotonic()
        if not checked_self or now >= next_report:
            chain = set(_chain())
            held = [_holders(p) for p in paths]
            if not checked_self and all(hs and set(hs) <= chain for hs in held):
                if abandon():
                    raise GateLockSelfDeadlock(
                        f"gate lock SELF-DEADLOCK: every one of the {n} slots is held by this "
                        f"process's own ancestors ({sorted({h for hs in held for h in hs})}) — take the "
                        f"outer slot with scripts/ops/gate_lock.sh / utils.gate_lock, not a bare flock")
                return home, time.monotonic() - t0
            checked_self = True
            if now >= next_report:
                log(f"[gate_lock] waiting {now - t0:.0f} s for one of {n} gate slots: "
                    + "; ".join(status(paths[0].parent).splitlines()))
                next_report = now + report_every_s
        if timeout_s is not None and now - t0 >= timeout_s:
            if abandon():
                raise TimeoutError(f"[gate_lock] gave up after {now - t0:.0f} s: "
                                   + "; ".join(status(paths[0].parent).splitlines()))
            return home, time.monotonic() - t0


@contextlib.contextmanager
def gate_slot(*, report_every_s: float = REPORT_EVERY_S, poll_s: float = POLL_S,
              timeout_s: Optional[float] = None, log: Callable[[str], None] = _stderr,
              what: str = "") -> Iterator[int]:
    """Hold one gate slot for the ``with`` body; yields the slot index (or the verified ancestor's)."""
    d = lock_dir()
    held = verified_holder(d)
    if held is not None:
        yield held[1]                 # re-entrant: an ancestor already holds a slot for this gate
        return
    if os.environ.get(HELD_ENV):
        log(f"[gate_lock] ignoring a STALE {HELD_ENV}={os.environ[HELD_ENV]} "
            f"(not a live ancestor holding a slot in {d})")
    n = slot_count()
    d.mkdir(parents=True, exist_ok=True)
    paths = [slot_path(i, d) for i in range(n)]
    fds = [os.open(str(p), os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o666) for p in paths]
    what = f" ({what})" if what else ""
    try:
        slot, waited = _acquire(fds, paths, report_every_s=report_every_s, poll_s=poll_s,
                                timeout_s=timeout_s, log=log)
        t_acq = time.time()
        log(f"[gate_lock] acquired slot {slot} of {n}{what} at "
            f"{time.strftime('%H:%M:%S', time.localtime(t_acq))} after waiting {waited:.1f} s")
        before = os.environ.get(HELD_ENV)
        os.environ[HELD_ENV] = f"{os.getpid()}:{slot}"
        try:
            yield slot
        finally:
            if before is None:
                os.environ.pop(HELD_ENV, None)
            else:
                os.environ[HELD_ENV] = before
            fcntl.flock(fds[slot], fcntl.LOCK_UN)
            log(f"[gate_lock] released slot {slot} of {n}{what} after holding "
                f"{time.time() - t_acq:.0f} s")
    finally:
        for fd in fds:
            os.close(fd)


def live_run_warning(runs: Optional[dict] = None) -> Optional[str]:
    """One line naming the live training run(s), recommending ``-n 4`` — or None when none is live.
    A WARNING only: the gate's command is never changed."""
    if runs is None:
        try:
            runs = procfs.live_training_runs()
        except Exception:                    # noqa: BLE001 — a warning must never stop a gate
            return None
    if not runs:
        return None
    named = ", ".join(f"{os.path.basename(rd.rstrip('/'))} (pid {pid})" for rd, pid in sorted(runs.items()))
    return (f"[gate_lock] ⚠️  a TRAINING RUN is live: {named}. Its env workers share this box's 8 cores — "
            "run the routine gate at -n 4, not -n 6 (designs/ops/testing.md, the -n x gate_lock policy).")


def run_under_slot(cmd: Sequence[str], **kw) -> int:
    """Run ``cmd`` as a CHILD while this process holds a slot; SIGINT/SIGTERM/SIGHUP are forwarded.
    Returns the child's exit code, or 128+N when signal N killed it."""
    warning = live_run_warning()
    if warning:
        _stderr(warning)
    with gate_slot(what=" ".join(cmd)[:80], **kw):
        child = subprocess.Popen(list(cmd))
        fwd = {s: signal.signal(s, lambda n, _f: child.send_signal(n))
               for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
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
        print(status())
        return 0
    kw: dict = {}
    while args and args[0] != "--":
        flag = args.pop(0)
        if flag in ("--report-every-s", "--timeout-s", "--poll-s") and args:
            kw[flag[2:].replace("-", "_")] = float(args.pop(0))
        else:
            print(f"[gate_lock] unknown option {flag!r} (put the command after --)", file=sys.stderr)
            return 2
    cmd = args[1:]
    if not cmd:
        print("[gate_lock] REFUSING: no command after --", file=sys.stderr)
        return 2
    try:
        return run_under_slot(cmd, **kw)
    except GateLockSelfDeadlock as e:
        print(str(e), file=sys.stderr)
        return 3
    except TimeoutError as e:
        print(str(e), file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
