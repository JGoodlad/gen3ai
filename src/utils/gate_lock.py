"""The box's TEST-GATE SEMAPHORE — at most N routine gates run at once; RE-ENTRANT across processes.

USAGE
    scripts/ops/gate_lock.sh python3 -m pytest src/ -m "not slow and not e2e" -q -n 6
    python -m utils.gate_lock [--timeout-s S] -- <cmd> [args...]    # the same, without the wrapper
    python -m utils.gate_lock --status [--json] [--sample-s S]      # who holds it, since when, ETA, the queue

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
HEAVY SLOTS (2026-10-19, ``utils.heavy_slots``): while a training run is live a gate also holds ONE of
the box's heavy-job slots for its whole run, taken AFTER its gate slot (so a queued gate holds none);
its children inherit the holder marker, so the ``mem_cap.sh`` inside it takes no second slot. LOCK
ORDER is gate before heavy: a take that would have to WAIT for the gate while an ancestor already
holds a heavy slot (``mem_cap.sh … gate_lock.sh …``) is refused at once, :class:`GateLockOrderError`
(exit 5) — two of those plus a gate holder waiting for a heavy slot is a deadlock.
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

from utils import gate_lock_status as GS
from utils import heavy_slots as HS
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


class GateLockOrderError(RuntimeError):
    """The gate would have to be WAITED for while an ancestor holds a heavy slot (lock order: gate first)."""


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


def status(d: Optional[Path] = None, *, sample_s: float = 0.0, compact: bool = False) -> str:
    """Who holds each slot, since when, the ETA and the queue (``utils.gate_lock_status``)."""
    state = GS.collect(d or lock_dir(), slot_count(), sample_s=sample_s)
    return GS.format_compact(state) if compact else GS.format_human(state)


def _try(fd: int) -> bool:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def _acquire(fds: List[int], paths: List[Path], *, report_every_s: float, poll_s: float,
             timeout_s: Optional[float], log: Callable[[str], None],
             may_wait: bool = True) -> Tuple[int, float]:
    """Take one slot; returns ``(slot index, seconds waited)``. ``may_wait=False`` raises
    :class:`GateLockOrderError` instead of waiting."""
    t0 = time.monotonic()
    for i, fd in enumerate(fds):
        if _try(fd):
            return i, 0.0
    n = len(fds)
    if not may_wait:
        raise GateLockOrderError(
            f"gate lock REFUSED (lock order): every gate slot is held and this process's ancestor holds a "
            f"HEAVY slot ({HS.HELD_ENV}={os.environ.get(HS.HELD_ENV)}) — waiting here could deadlock. Put "
            "gate_lock.sh OUTSIDE mem_cap.sh (`gate_lock.sh mem_cap.sh 24 …`, the ship gate's order). "
            + status(paths[0].parent, compact=True))
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
                    + status(paths[0].parent, compact=True))
                next_report = now + report_every_s
        if timeout_s is not None and now - t0 >= timeout_s:
            if abandon():
                raise TimeoutError(f"[gate_lock] gave up after {now - t0:.0f} s: "
                                   + status(paths[0].parent, compact=True))
            return home, time.monotonic() - t0


@contextlib.contextmanager
def gate_slot(*, report_every_s: float = REPORT_EVERY_S, poll_s: float = POLL_S,
              timeout_s: Optional[float] = None, log: Callable[[str], None] = _stderr,
              what: str = "", cmd: Sequence[str] = (), outcome: Optional[dict] = None) -> Iterator[int]:
    """Hold one gate slot for the ``with`` body; yields the slot index (or the verified ancestor's).

    A FRESH take writes the slot's record (``utils.gate_lock_status``: pid, worktree, ``cmd``, cpus, nice,
    acquired-at) and removes it on release; with ``cmd`` given, the release also appends the gate's wall
    and ``outcome["rc"]`` to the history the ETA is read from. Records only report — the flock is the lock."""
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
                                timeout_s=timeout_s, log=log, may_wait=HS.verified_holder() is None)
        t_acq = time.time()
        log(f"[gate_lock] acquired slot {slot} of {n}{what} at "
            f"{time.strftime('%H:%M:%S', time.localtime(t_acq))} after waiting {waited:.1f} s")
        rec = GS.SlotRecord.mine(slot, cmd)
        try:
            GS.write_record(d, rec)
        except OSError as exc:            # a record only reports; never stop a gate over it
            log(f"[gate_lock] slot record NOT written ({exc!r}); --status falls back to /proc")
        before = os.environ.get(HELD_ENV)
        os.environ[HELD_ENV] = f"{os.getpid()}:{slot}"
        try:
            # rule 5b: the gate holds one heavy slot itself, inside what remains of ``timeout_s``
            with HS.heavy_slot(cmd, log=log, timeout_s=None if timeout_s is None else max(0.0, timeout_s - waited)):
                yield slot
        finally:
            if before is None:
                os.environ.pop(HELD_ENV, None)
            else:
                os.environ[HELD_ENV] = before
            held = time.time() - t_acq
            if cmd:
                try:
                    GS.append_history(d, rec, held, (outcome or {}).get("rc"))
                except OSError as exc:
                    log(f"[gate_lock] history NOT appended ({exc!r})")
            GS.clear_record(d, slot, rec.pid)
            fcntl.flock(fds[slot], fcntl.LOCK_UN)
            log(f"[gate_lock] released slot {slot} of {n}{what} after holding {held:.0f} s")
    finally:
        for fd in fds:
            os.close(fd)


#: The routine gate's xdist workers per LOGICAL cpu of its affinity while a training run is live
#: (rule 5b: the agent cores, nice 19). MEASURED, designs/ops/testing.md "Routine-gate wall time".
LEASED_WORKERS_PER_CPU = 0.5


def recommended_workers(ncpus: int) -> int:
    return max(1, int(ncpus * LEASED_WORKERS_PER_CPU))


def live_run_warning(runs: Optional[dict] = None, *, my_cpus: Optional[List[int]] = None,
                     my_nice: Optional[int] = None, trainer_cpus: Optional[List[int]] = None) -> Optional[str]:
    """One line naming the live training run(s) and the rule-5b shape (nice 19, the agent cores, the
    measured ``-n``), plus what is WRONG with this gate's own shape — or None when no run is live.
    A WARNING only: the gate's command is never changed. The ``my_*`` / ``trainer_cpus`` arguments
    are for tests; by default they are read from this process and the first live trainer."""
    if runs is None:
        try:
            runs = procfs.live_training_runs()
        except Exception:                    # noqa: BLE001 — a warning must never stop a gate
            return None
    if not runs:
        return None
    named = ", ".join(f"{os.path.basename(rd.rstrip('/'))} (pid {pid})" for rd, pid in sorted(runs.items()))
    mine = my_cpus if my_cpus is not None else sorted(os.sched_getaffinity(0))
    nice = my_nice if my_nice is not None else os.getpriority(os.PRIO_PROCESS, 0)
    if trainer_cpus is None:
        trainer_cpus = []
        for pid in runs.values():
            trainer_cpus.extend(GS.affinity_of(pid) or [])
    online = range(os.cpu_count() or 0)
    agent = [c for c in online if c not in set(trainer_cpus)] if trainer_cpus else []
    problems = []
    if nice < 19:
        problems.append(f"this gate runs at nice {nice}, not 19")
    overlap = sorted(set(mine) & set(trainer_cpus))
    if overlap:
        problems.append(f"its cpus {GS.cpu_list(mine)} OVERLAP the trainer's ({GS.cpu_list(overlap)})")
    where = f"taskset -c {GS.cpu_list(agent)}" if agent else "taskset -c <the agent cores>"
    n = recommended_workers(len(agent) if agent else len(mine))
    line = (f"[gate_lock] ⚠️  a TRAINING RUN is live: {named}. Rule 5b: the gate runs at LOW PRIORITY on the "
            f"agent cores — `nice -n 19 {where} scripts/ops/gate_lock.sh … -n {n}` (designs/ops/testing.md, "
            "the -n x gate_lock policy).")
    if problems:
        line += " THIS gate: " + "; ".join(problems) + "."
    return line


def run_under_slot(cmd: Sequence[str], **kw) -> int:
    """Run ``cmd`` as a CHILD while this process holds a slot; SIGINT/SIGTERM/SIGHUP are forwarded.
    Returns the child's exit code, or 128+N when signal N killed it."""
    warning = live_run_warning()
    if warning:
        _stderr(warning)
    outcome: dict = {}
    with gate_slot(what=" ".join(cmd)[:80], cmd=cmd, outcome=outcome, **kw):
        child = subprocess.Popen(list(cmd))
        fwd = {s: signal.signal(s, lambda n, _f: child.send_signal(n))
               for s in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
        try:
            rc = child.wait()
            outcome["rc"] = rc
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
        rest = args[1:]
        as_json = "--json" in rest
        sample_s = float(rest[rest.index("--sample-s") + 1]) if "--sample-s" in rest else 1.0
        state = GS.collect(lock_dir(), slot_count(), sample_s=sample_s)
        print(GS.as_json(state) if as_json else GS.format_human(state))
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
    except GateLockOrderError as e:
        print(str(e), file=sys.stderr)
        return 5
    except TimeoutError as e:
        print(str(e), file=sys.stderr)
        return 4


if __name__ == "__main__":
    sys.exit(main())
