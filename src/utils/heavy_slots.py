"""The box's HEAVY-JOB SLOTS — at most N heavy agent jobs at once while a training run is live (rule 5b).

USAGE
    scripts/ops/mem_cap.sh 24 <cmd>                 # takes a slot by itself while a run is live
    scripts/ops/mem_cap.sh --slot 24 <cmd>          # take one even with no run live
    scripts/ops/mem_cap.sh --no-wait 24 <cmd>       # refuse at once (exit 87) when every slot is held
    GEN3AI_BLOCKING="X34 plateau tick" scripts/ops/mem_cap.sh 24 <cmd>   # orchestrator-marked: bypass
    scripts/ops/mem_cap.sh --status                 # the memory slice, then holders / queue / bypassers
    python -m utils.heavy_slots --status [--json]

WHY (2026-10-19). Rule 5b says "<= 2 heavy jobs at once" beside a live run, and nothing enforced it:
separate agents each ran 1-6 heavy jobs on the 3 agent cores, load hit 29, the X34 plateau check (which
BLOCKS the GPU's next work) was starved, and a 14-min ship gate took over an hour. Every heavy job
already goes through ``scripts/ops/mem_cap.sh``, so that is where the slot is taken.

POLICY (:func:`wanted`). A job takes a slot when a training run is live
(``utils.procfs.live_training_runs``), while the GPU is LEASED (:func:`lease_reason`: rule 5b is keyed to the lease
state too, so a leased window - a measurement battery under ``mem_cap`` in the same 64 GiB slice - is not flooded
by the jobs that queued behind a run the moment it ends), or when it asks (``--slot``). A job the orchestrator marks
BLOCKING (``$GEN3AI_BLOCKING=<reason>``, or ``mem_cap.sh --blocking <reason>``) BYPASSES the slots: it
runs at once, says so on stderr and in ``--status``. A job that already holds the GPU
(it runs under ``gpu_lock`` - ``utils.gpu_lock.gpu_job``: the flock holder or the lease owner's pass-through)
bypasses them too, whatever the wrapper order: ``gpu_lock.sh mem_cap.sh …`` would otherwise hold the GPU idle
while it queued for a CPU slot (D-3). The routine gate (``utils.gate_lock``) holds ONE
slot itself for its whole run, taken AFTER its gate slot.

HOW. ``N`` (:data:`HEAVY_SLOTS` = 2; ``$GEN3AI_HEAVY_SLOTS``) ``flock`` slot files
``slot<i>.lock`` under ``~/.claude/jobs/heavy_slots/`` (``$GEN3AI_HEAVY_SLOT_DIR``; tests point it at a
temp dir). A crashed holder's slot frees the instant it dies (the kernel drops its flock). The wait is visible:
one line at once, one every :data:`REPORT_EVERY_S` naming the holders; ``--no-wait`` refuses instead
(:class:`SlotsFull`, exit 87).

THE QUEUE IS FIFO (2026-10-20: waiters polled the slots at random and a gate starved 3.4 h behind later arrivals).
Every waiter takes a TICKET (``ticket-<seq>-<pid>.tk``, ``seq`` = the highest live one + 1, allocated under a lock held
for that instant only) and may take a free slot only when fewer live tickets than there are free slots are older than
its own: the oldest gets the next free slot. A ticket of a dead pid is skipped and removed; one whose owner has not
polled for :data:`TICKET_STALE_S` (a gate blocked on the gate lock between two slot attempts) is skipped without
losing its place. The ticket is dropped when the slot is taken or the wait ends; the flock alone is still the lock.

THE SLOT LIVES AS LONG AS THE JOB (D-2, core review 2026-10-20). The flock is on an open file description, and
``mem_cap`` hands the locked fd down to its supervisor and the job (:func:`slot_fd`, ``pass_fds``), so a SIGKILL /
OOM kill of the ``mem_cap`` wrapper does NOT free the slot while the capped job's process tree still runs (the
supervisor stops the job when its wrapper dies, but a job that defers SIGTERM runs on for the grace): the slot
frees when the last process of that tree exits. A NORMAL end still unlocks at once (``LOCK_UN``): a daemon the
finished job leaked does not hold it.

HOLD A SLOT PER BOUNDED CHUNK (<= 45 min), NEVER PER RUN: a multi-hour job holding a slot froze every other agent for hours
(2026-10-19); split it into resumable chunks, one slot each. `--status` flags a holder past 60 min as
`LONG HOLD (policy: <= 45 min per chunk)` and the holder gets one stderr warning; nothing is killed.

NEVER TWO SLOTS FOR ONE JOB. Holding a slot exports ``GEN3AI_HEAVY_SLOT_HELD=<pid>:<slot>``; a taker
that sees it takes NO slot after VERIFYING that the pid is itself or an ancestor and that
``/proc/locks`` lists it holding that slot (the ``utils.gate_lock`` re-entrance) — so a gate run under
mem_cap inside gate_lock holds ONE slot, and a mem_cap.sh a test spawns inside it takes none.

LOCK ORDER: the gate slot BEFORE a heavy slot. ``gate_lock.sh mem_cap.sh …`` (the ship gate's order)
is safe. ``mem_cap.sh gate_lock.sh …`` holds a heavy slot while it queues for the gate, and two of
those plus a gate holder waiting for a heavy slot would deadlock — so ``utils.gate_lock`` REFUSES to
WAIT for the gate while an ancestor holds a heavy slot (:class:`utils.gate_lock.GateLockOrderError`).

The records (``slot<i>.json``, ``wait-<pid>.json``, ``bypass-<pid>.json``) only REPORT; the flock alone
is the lock. Linux only (``/proc``).
"""
from __future__ import annotations

import contextlib
import fcntl
import json
import os
import random
import shlex
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

from utils import gpu_lock as GL
from utils import procfs

HEAVY_SLOTS = 2
SLOTS_ENV = "GEN3AI_HEAVY_SLOTS"
DIR_ENV = "GEN3AI_HEAVY_SLOT_DIR"
HELD_ENV = "GEN3AI_HEAVY_SLOT_HELD"
BLOCKING_ENV = "GEN3AI_BLOCKING"
POLL_S = 0.5
#: a ticket whose owner has not touched it for this long is not waiting right now (e.g. a gate blocked on the gate
#: lock): the queue skips it, and it keeps its place.
TICKET_STALE_S = 20.0
TICKET_LOCK = "ticket.seq.lock"
#: POLICY (orchestrator 2026-10-20): hold a slot per BOUNDED CHUNK (<= CHUNK_MIN), never per run. --status flags a
#: holder past LONG_HOLD_S and the holder is warned once; nothing is killed.
CHUNK_MIN = 45
LONG_HOLD_S = 3600.0
LONG_HOLD_TAG = f"LONG HOLD (policy: <= {CHUNK_MIN} min per chunk)"
REPORT_EVERY_S = 60.0
EXIT_SLOTS_FULL = 87
RECORD_SCHEMA = "gen3_heavy_slot_record_v1"


#: slot index -> the locked fd THIS process holds for it (set only while this process is the holder).
_HELD_FDS: dict = {}


class SlotsFull(RuntimeError):
    """``--no-wait`` and every slot is held."""


def slot_count() -> int:
    raw = os.environ.get(SLOTS_ENV)
    if not raw:
        return HEAVY_SLOTS
    n = int(raw)
    if n < 1:
        raise ValueError(f"{SLOTS_ENV}={raw!r}: need at least one slot")
    return n


def lock_dir() -> Path:
    return Path(os.environ.get(DIR_ENV) or Path.home() / ".claude" / "jobs" / "heavy_slots")


def slot_path(i: int, d: Optional[Path] = None) -> Path:
    return (d or lock_dir()) / f"slot{i}.lock"


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def _chain() -> List[int]:
    me = os.getpid()
    return [me, *procfs.ancestors(me)]


def _holders(path: Path) -> List[int]:
    key = procfs.file_key(str(path))
    return procfs.holders(key) if key else []


def verified_holder(d: Optional[Path] = None) -> Optional[Tuple[int, int]]:
    """``(pid, slot)`` from ``$GEN3AI_HEAVY_SLOT_HELD`` iff that pid is this process or an ancestor AND
    ``/proc/locks`` lists it holding that slot; else ``None`` (absent, malformed or stale)."""
    raw = os.environ.get(HELD_ENV, "")
    try:
        pid_s, slot_s = raw.split(":")
        pid, slot = int(pid_s), int(slot_s)
    except ValueError:
        return None
    if pid not in _chain():
        return None
    return (pid, slot) if pid in _holders(slot_path(slot, d)) else None


def slot_fd(slot: Optional[int]) -> Optional[int]:
    """The locked fd this process holds for ``slot``, for a caller that hands it to the job's process tree
    (:mod:`utils.mem_cap` does: D-2). ``None`` for a re-entrant / bypassed / unwanted slot (nothing to hand on)."""
    return None if slot is None else _HELD_FDS.get(slot)


def blocking_reason() -> Optional[str]:
    """The orchestrator's BLOCKING mark (``$GEN3AI_BLOCKING``, any non-empty text = the reason)."""
    raw = (os.environ.get(BLOCKING_ENV) or "").strip()
    return raw or None


def lease_reason() -> Optional[str]:
    """Why a GPU LEASE being held wants the slots (``None``: no valid lease, or any read error - a job is never
    failed over this). Rule 5b is keyed to the LEASE state as well as to a live run: while the GPU is leased its
    window (a measurement battery under ``mem_cap`` in the SAME memory slice) must not be flooded by the agent
    jobs that queued behind a live run (they all start the moment the run ends and a cell can be OOM-killed)."""
    try:
        lease = GL.L.read_lease(GL.lock_path())
        return None if lease is None else f"the GPU is LEASED ({lease.describe()}): rule 5b"
    except Exception:            # noqa: BLE001 — a /proc or file read must never stop a job
        return None


def wanted(*, force: bool = False, runs: Optional[dict] = None, lease: bool = True) -> Optional[str]:
    """Why this job must take a slot, or ``None`` when it need not (no run live, no GPU lease, not forced).
    ``lease=False`` ignores the lease (tests that pin the run-only policy)."""
    if force:
        return "asked for (--slot)"
    try:
        runs = procfs.live_training_runs() if runs is None else runs
    except Exception:            # noqa: BLE001 — a /proc read must never stop a job; it then takes no slot
        runs = {}
    if runs:
        names = ", ".join(os.path.basename(rd.rstrip("/")) for rd in sorted(runs))
        return f"a training run is live ({names}): rule 5b"
    return lease_reason() if lease else None


@dataclass
class Record:
    pid: int
    kind: str          # "holder" | "wait" | "bypass"
    since: float
    cwd: str
    cmd: List[str]
    note: str = ""
    slot: int = -1
    schema: str = RECORD_SCHEMA

    @classmethod
    def mine(cls, kind: str, cmd: Sequence[str], note: str = "", slot: int = -1) -> "Record":
        return cls(pid=os.getpid(), kind=kind, since=time.time(), cwd=os.getcwd(), cmd=list(cmd)[:48],
                   note=note, slot=slot)


def _rec_path(d: Path, rec_kind: str, key: int) -> Path:
    return d / (f"slot{key}.json" if rec_kind == "holder" else f"{rec_kind}-{key}.json")


def _write(d: Path, rec: Record) -> Optional[Path]:
    p = _rec_path(d, rec.kind, rec.slot if rec.kind == "holder" else rec.pid)
    try:
        tmp = p.with_name(f".{p.name}.{os.getpid()}")
        tmp.write_text(json.dumps(asdict(rec)))
        os.replace(tmp, p)
        return p
    except OSError:
        return None                # a record only reports


def _read(p: Path) -> Optional[Record]:
    try:
        raw = json.loads(p.read_text())
        return Record(**raw) if raw.get("schema") == RECORD_SCHEMA else None
    except (OSError, ValueError, TypeError):
        return None


def _drop(p: Optional[Path], pid: int) -> None:
    if p is None:
        return
    rec = _read(p)
    if rec is not None and rec.pid == pid:
        with contextlib.suppress(OSError):
            p.unlink()


def _try(fd: int) -> bool:
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def _tickets(d: Path) -> List[Tuple[int, int, Path]]:
    """``(seq, pid, path)`` of every ticket, oldest first (dead owners' tickets removed on the way)."""
    out: List[Tuple[int, int, Path]] = []
    for p in d.glob("ticket-*.tk") if d.is_dir() else ():
        try:
            _, seq_s, pid_s = p.stem.split("-")
            seq, pid = int(seq_s), int(pid_s)
        except ValueError:
            continue
        if not _alive(pid):
            with contextlib.suppress(OSError):
                p.unlink()
            continue
        out.append((seq, pid, p))
    return sorted(out)


def take_ticket(d: Path, pid: Optional[int] = None) -> Path:
    """This pid's place in the queue (idempotent: a second take returns the SAME ticket, keeping its age)."""
    pid = os.getpid() if pid is None else pid
    d.mkdir(parents=True, exist_ok=True)
    with open(d / TICKET_LOCK, "a") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)          # held for the instant of the allocation, never while waiting
        live = _tickets(d)
        for _, tpid, p in live:
            if tpid == pid:
                return p
        mine = d / f"ticket-{(live[-1][0] + 1) if live else 0:012d}-{pid}.tk"
        mine.touch()
        return mine


def drop_ticket(d: Optional[Path] = None, pid: Optional[int] = None) -> None:
    """Leave the queue (a no-op without a ticket)."""
    d = d or lock_dir()
    pid = os.getpid() if pid is None else pid
    for _, tpid, p in _tickets(d):
        if tpid == pid:
            with contextlib.suppress(OSError):
                p.unlink()


def has_ticket(d: Path, pid: Optional[int] = None) -> bool:
    pid = os.getpid() if pid is None else pid
    return any(tpid == pid for _, tpid, _p in _tickets(d))


def _beat(ticket: Path) -> None:
    with contextlib.suppress(OSError):
        os.utime(ticket)


def queue_ahead(d: Path, pid: Optional[int] = None, *, now: Optional[float] = None) -> int:
    """How many LIVE, polling tickets are older than ``pid``'s (a stale ticket does not count; ``pid``'s own always does)."""
    pid = os.getpid() if pid is None else pid
    now = time.time() if now is None else now
    ahead = 0
    for _, tpid, p in _tickets(d):
        if tpid == pid:
            return ahead
        try:
            if now - p.stat().st_mtime <= TICKET_STALE_S:
                ahead += 1
        except OSError:
            continue
    return ahead


def free_count(d: Path, n: int) -> int:
    locks = procfs.read_locks()
    keys = [procfs.file_key(str(slot_path(i, d))) for i in range(n)]
    return sum(1 for k in keys if not (procfs.holders(k, locks) if k else []))


def entitled(d: Path, n: int, pid: Optional[int] = None, *, now: Optional[float] = None) -> bool:
    """FIFO: this pid may take a free slot iff fewer polling tickets than there are free slots are older than its own."""
    ahead = queue_ahead(d, pid, now=now)
    return ahead == 0 or ahead < free_count(d, n)


@contextlib.contextmanager
def heavy_slot(cmd: Sequence[str] = (), *, force: bool = False, wait: bool = True,
               timeout_s: Optional[float] = None, poll_s: float = POLL_S,
               report_every_s: float = REPORT_EVERY_S, log: Callable[[str], None] = _stderr,
               runs: Optional[dict] = None, keep_ticket: bool = False) -> Iterator[Optional[int]]:
    """Hold one heavy slot for the ``with`` body when the policy wants one; yields the slot index, or
    ``None`` when no slot was taken (bypass / not wanted). Raises :class:`SlotsFull` (``wait=False``)
    or ``TimeoutError`` (``timeout_s``) without running the body. FIFO: a slot is taken only when this job is
    entitled to it (:func:`entitled`); ``keep_ticket`` leaves the ticket in the queue when ``wait=False`` refuses
    (the caller will wait with :func:`await_free_slot` and keep its place)."""
    d = lock_dir()
    held = verified_holder(d)
    if held is not None:
        yield held[1]              # re-entrant: an ancestor holds this job's slot
        return
    if os.environ.get(HELD_ENV):
        log(f"[heavy_slots] ignoring a STALE {HELD_ENV}={os.environ[HELD_ENV]} (not a live ancestor holding it)")
    why = wanted(force=force, runs=runs)
    if why is None:
        yield None
        return
    reason = blocking_reason()
    gpu = GL.gpu_job() if reason is None else None
    if reason is not None or gpu is not None:
        d.mkdir(parents=True, exist_ok=True)
        note = reason if reason is not None else f"GPU job (gpu_lock pid {gpu}): the GPU would idle behind a slot wait"
        p = _write(d, Record.mine("bypass", cmd, note=note))
        if reason is not None:
            log(f"[heavy_slots] BLOCKING ({BLOCKING_ENV}={reason!r}): bypassing the {slot_count()} heavy slots")
        else:
            log(f"[heavy_slots] a GPU job (under gpu_lock, pid {gpu}) takes no CPU heavy slot: waiting for one "
                f"would hold the GPU idle (D-3, core review 2026-10-20)")
        try:
            yield None
        finally:
            _drop(p, os.getpid())
        return
    n = slot_count()
    d.mkdir(parents=True, exist_ok=True)
    paths = [slot_path(i, d) for i in range(n)]
    fds = [os.open(str(p), os.O_RDWR | os.O_CREAT | os.O_CLOEXEC, 0o666) for p in paths]
    wait_rec: Optional[Path] = None
    ticket = take_ticket(d)
    kept = False
    try:
        t0 = time.monotonic()
        slot: Optional[int] = None
        next_report = t0
        while slot is None:
            _beat(ticket)
            if entitled(d, n):
                slot = next((i for i, fd in enumerate(fds) if _try(fd)), None)
            if slot is not None:
                break
            now = time.monotonic()
            if not wait:
                kept = keep_ticket
                raise SlotsFull(f"[heavy_slots] REFUSING (--no-wait): all {n} heavy slots are held or spoken for "
                                f"({queue_ahead(d)} queued ahead, FIFO) — " + compact_status(d, n))
            if wait_rec is None:
                wait_rec = _write(d, Record.mine("wait", cmd, note=why))
            if now >= next_report:
                log(f"[heavy_slots] waiting {now - t0:.0f} s for one of {n} heavy slots ({why}; "
                    f"{queue_ahead(d)} queued ahead, FIFO): "
                    + compact_status(d, n) + f" — `scripts/ops/mem_cap.sh --status`; the orchestrator "
                    f"marks a GPU-blocking job with {BLOCKING_ENV}=<reason>")
                next_report = now + report_every_s
            if timeout_s is not None and now - t0 >= timeout_s:
                raise TimeoutError(f"[heavy_slots] gave up after {now - t0:.0f} s: " + compact_status(d, n))
            time.sleep(poll_s)
        drop_ticket(d)
        waited = time.monotonic() - t0
        _drop(wait_rec, os.getpid())
        wait_rec = None
        rec_p = _write(d, Record.mine("holder", cmd, note=why, slot=slot))
        log(f"[heavy_slots] took heavy slot {slot} of {n} after waiting {waited:.1f} s ({why})")
        before = os.environ.get(HELD_ENV)
        os.environ[HELD_ENV] = f"{os.getpid()}:{slot}"
        _HELD_FDS[slot] = fds[slot]
        t_acq = time.monotonic()
        warn = threading.Timer(LONG_HOLD_S, lambda: log(
            f"[heavy_slots] {LONG_HOLD_TAG}: slot {slot} held for {LONG_HOLD_S / 60:.0f} min by pid {os.getpid()} "
            f"({shlex.join(list(cmd)[:6])}) - release it and re-take one per bounded chunk"))
        warn.daemon = True
        warn.start()
        try:
            yield slot
        finally:
            warn.cancel()
            _HELD_FDS.pop(slot, None)
            if before is None:
                os.environ.pop(HELD_ENV, None)
            else:
                os.environ[HELD_ENV] = before
            _drop(rec_p, os.getpid())
            fcntl.flock(fds[slot], fcntl.LOCK_UN)
            log(f"[heavy_slots] released heavy slot {slot} after {time.monotonic() - t_acq:.0f} s")
    finally:
        _drop(wait_rec, os.getpid())
        if not kept:
            drop_ticket(d)
        for fd in fds:
            os.close(fd)


def await_free_slot(cmd: Sequence[str] = (), *, force: bool = False, timeout_s: Optional[float] = None,
                    poll_s: float = POLL_S, report_every_s: float = REPORT_EVERY_S,
                    log: Callable[[str], None] = _stderr, runs: Optional[dict] = None) -> None:
    """Return once a heavy slot is free AND this job is next in the FIFO queue, holding NO lock while it waits (a
    visible ``wait`` record and a queue ticket, polled with jitter). The ticket STAYS on a normal return, so the
    caller's ``heavy_slot(wait=False, keep_ticket=True)`` keeps the place and, if a sibling got there first, waits
    again without losing its age; it is dropped on a timeout or when the slot is no longer wanted.
    Returns at once when this job needs no slot (re-entrant, not wanted, BLOCKING / GPU-job bypass).

    WHY (the gate convoy, 2026-10-19/20). ``utils.gate_lock`` took its heavy slot AFTER the gate lock and waited
    for it while holding the gate lock: one gate idle on a 28 h research job's slot blocked every other gate for
    over an hour. No process may hold the gate lock while waiting for a heavy slot - it waits HERE, holding none.
    Raises ``TimeoutError`` (``timeout_s``)."""
    d = lock_dir()
    if verified_holder(d) is not None or wanted(force=force, runs=runs) is None:
        return
    if blocking_reason() is not None or GL.gpu_job() is not None:
        return
    n = slot_count()
    ticket = take_ticket(d)
    wait_rec: Optional[Path] = None
    done = False

    def ready() -> bool:
        _beat(ticket)
        return free_count(d, n) > 0 and entitled(d, n)

    try:
        if ready():
            done = True
            return
        wait_rec = _write(d, Record.mine("wait", cmd, note="queued for a heavy slot holding NO lock"))
        t0 = time.monotonic()
        next_report = next_check = t0
        while not ready():
            now = time.monotonic()
            if now >= next_check:                          # the run ended / the lease was released while it waited
                if wanted(force=force, runs=runs) is None:
                    return
                next_check = now + 5.0
            if now >= next_report:
                log(f"[heavy_slots] waiting {now - t0:.0f} s for one of {n} heavy slots (holding no other lock; "
                    f"{queue_ahead(d)} queued ahead, FIFO): " + compact_status(d, n))
                next_report = now + report_every_s
            if timeout_s is not None and now - t0 >= timeout_s:
                raise TimeoutError(f"[heavy_slots] gave up after {now - t0:.0f} s: " + compact_status(d, n))
            time.sleep(poll_s * (0.5 + random.random()))
        done = True
    finally:
        _drop(wait_rec, os.getpid())
        if not done:
            drop_ticket(d)


def _alive(pid: int) -> bool:
    return os.path.exists(f"/proc/{pid}")


def collect(d: Optional[Path] = None, n: Optional[int] = None, *, now: Optional[float] = None) -> dict:
    """Holders (from ``/proc/locks``, with their records when the pid matches), waiters and bypassers
    (records whose pid is alive)."""
    d = d or lock_dir()
    n = slot_count() if n is None else n
    now = time.time() if now is None else now
    locks = procfs.read_locks()
    slots = []
    for i in range(n):
        key = procfs.file_key(str(slot_path(i, d)))
        pids = procfs.holders(key, locks) if key else []
        rec = _read(_rec_path(d, "holder", i))
        row: dict = {"slot": i, "pids": pids, "held_s": None, "cmd": "", "cwd": "", "why": ""}
        if pids:
            if rec is not None and rec.pid in pids:
                row.update(held_s=now - rec.since, cmd=shlex.join(rec.cmd), cwd=rec.cwd, why=rec.note)
                row["long_hold"] = (now - rec.since) > LONG_HOLD_S
            else:
                row.update(cmd=procfs.cmdline(pids[0]), why="(no record)")
            if not _alive(pids[0]):
                # D-2: the locking process (a mem_cap wrapper) was killed; its job tree still holds the fd
                row["why"] += " [owner gone: held by the job's surviving process tree]"
        slots.append(row)
    side: dict = {"wait": [], "bypass": []}
    if d.is_dir():
        for p in sorted(d.glob("*-*.json")):
            rec = _read(p)
            if rec is None or rec.kind not in side:
                continue
            if not _alive(rec.pid):
                with contextlib.suppress(OSError):
                    p.unlink()     # a dead waiter's / bypasser's record
                continue
            side[rec.kind].append({"pid": rec.pid, "for_s": now - rec.since, "cmd": shlex.join(rec.cmd),
                                   "cwd": rec.cwd, "why": rec.note})
    order = {pid: seq for seq, pid, _p in _tickets(d)}          # the FIFO queue order (oldest ticket first)
    for k in side:
        side[k].sort(key=lambda r: (order.get(r["pid"], 1 << 60), -r["for_s"]))
    return {"dir": str(d), "n_slots": n, "slots": slots, "waiting": side["wait"], "bypassing": side["bypass"]}


def _dur(s: Optional[float]) -> str:
    if s is None:
        return "?"
    s = int(s)
    return f"{s // 3600}h{s % 3600 // 60:02d}m" if s >= 3600 else f"{s // 60}m{s % 60:02d}s"


def format_human(state: dict) -> str:
    out = [f"heavy slots ({state['n_slots']}, {state['dir']}):"]
    for r in state["slots"]:
        if not r["pids"]:
            out.append(f"  slot {r['slot']}: free")
        else:
            flag = f"  <-- {LONG_HOLD_TAG}" if r.get("long_hold") else ""
            out.append(f"  slot {r['slot']}: pid {','.join(map(str, r['pids']))} held {_dur(r['held_s'])} "
                       f"[{r['why']}] {r['cmd'][:140]}  (cwd {r['cwd'] or '?'}){flag}")
    out.append(f"  queue ({len(state['waiting'])}, FIFO: the head gets the next free slot):")
    out += [f"    pid {w['pid']} waiting {_dur(w['for_s'])}: {w['cmd'][:140]}" for w in state["waiting"]]
    out.append(f"  BLOCKING bypassers ({len(state['bypassing'])}):")
    out += [f"    pid {b['pid']} for {_dur(b['for_s'])} [{b['why']}]: {b['cmd'][:140]}" for b in state["bypassing"]]
    return "\n".join(out)


def compact_status(d: Optional[Path] = None, n: Optional[int] = None) -> str:
    st = collect(d, n)
    held = [f"slot {r['slot']} pid {r['pids'][0]} {_dur(r['held_s'])} {r['cmd'][:60]!r}"
            for r in st["slots"] if r["pids"]]
    return f"held: {'; '.join(held) or 'none'}; {len(st['waiting'])} waiting"


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] != ["--status"]:
        print(__doc__)
        return 0 if args[:1] in (["-h"], ["--help"]) else 2
    st = collect()
    print(json.dumps(st, indent=1) if "--json" in args else format_human(st))
    return 0


if __name__ == "__main__":
    sys.exit(main())
