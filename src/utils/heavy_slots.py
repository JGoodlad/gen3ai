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
(``utils.procfs.live_training_runs``) or when it asks (``--slot``). A job the orchestrator marks
BLOCKING (``$GEN3AI_BLOCKING=<reason>``, or ``mem_cap.sh --blocking <reason>``) BYPASSES the slots: it
runs at once, says so on stderr and in ``--status``. The routine gate (``utils.gate_lock``) holds ONE
slot itself for its whole run, taken AFTER its gate slot.

HOW. ``N`` (:data:`HEAVY_SLOTS` = 2; ``$GEN3AI_HEAVY_SLOTS``) ``flock`` slot files
``slot<i>.lock`` under ``~/.claude/jobs/heavy_slots/`` (``$GEN3AI_HEAVY_SLOT_DIR``; tests point it at a
temp dir). A taker tries every slot non-blocking every :data:`POLL_S`; a crashed holder's slot frees
the instant it dies (the kernel drops its flock). The wait is visible: one line at once, one every
:data:`REPORT_EVERY_S` naming the holders; ``--no-wait`` refuses instead (:class:`SlotsFull`, exit 87).

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
import shlex
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterator, List, Optional, Sequence, Tuple

from utils import procfs

HEAVY_SLOTS = 2
SLOTS_ENV = "GEN3AI_HEAVY_SLOTS"
DIR_ENV = "GEN3AI_HEAVY_SLOT_DIR"
HELD_ENV = "GEN3AI_HEAVY_SLOT_HELD"
BLOCKING_ENV = "GEN3AI_BLOCKING"
POLL_S = 0.5
REPORT_EVERY_S = 60.0
EXIT_SLOTS_FULL = 87
RECORD_SCHEMA = "gen3_heavy_slot_record_v1"


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


def blocking_reason() -> Optional[str]:
    """The orchestrator's BLOCKING mark (``$GEN3AI_BLOCKING``, any non-empty text = the reason)."""
    raw = (os.environ.get(BLOCKING_ENV) or "").strip()
    return raw or None


def wanted(*, force: bool = False, runs: Optional[dict] = None) -> Optional[str]:
    """Why this job must take a slot, or ``None`` when it need not (no run live and not forced)."""
    if force:
        return "asked for (--slot)"
    try:
        runs = procfs.live_training_runs() if runs is None else runs
    except Exception:            # noqa: BLE001 — a /proc read must never stop a job; it then takes no slot
        return None
    if runs:
        names = ", ".join(os.path.basename(rd.rstrip("/")) for rd in sorted(runs))
        return f"a training run is live ({names}): rule 5b"
    return None


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


@contextlib.contextmanager
def heavy_slot(cmd: Sequence[str] = (), *, force: bool = False, wait: bool = True,
               timeout_s: Optional[float] = None, poll_s: float = POLL_S,
               report_every_s: float = REPORT_EVERY_S, log: Callable[[str], None] = _stderr,
               runs: Optional[dict] = None) -> Iterator[Optional[int]]:
    """Hold one heavy slot for the ``with`` body when the policy wants one; yields the slot index, or
    ``None`` when no slot was taken (bypass / not wanted). Raises :class:`SlotsFull` (``wait=False``)
    or ``TimeoutError`` (``timeout_s``) without running the body."""
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
    if reason is not None:
        d.mkdir(parents=True, exist_ok=True)
        p = _write(d, Record.mine("bypass", cmd, note=reason))
        log(f"[heavy_slots] BLOCKING ({BLOCKING_ENV}={reason!r}): bypassing the {slot_count()} heavy slots")
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
    try:
        t0 = time.monotonic()
        slot: Optional[int] = None
        next_report = t0
        while slot is None:
            slot = next((i for i, fd in enumerate(fds) if _try(fd)), None)
            if slot is not None:
                break
            now = time.monotonic()
            if not wait:
                raise SlotsFull(f"[heavy_slots] REFUSING (--no-wait): all {n} heavy slots are held — "
                                + compact_status(d, n))
            if wait_rec is None:
                wait_rec = _write(d, Record.mine("wait", cmd, note=why))
            if now >= next_report:
                log(f"[heavy_slots] waiting {now - t0:.0f} s for one of {n} heavy slots ({why}): "
                    + compact_status(d, n) + f" — `scripts/ops/mem_cap.sh --status`; the orchestrator "
                    f"marks a GPU-blocking job with {BLOCKING_ENV}=<reason>")
                next_report = now + report_every_s
            if timeout_s is not None and now - t0 >= timeout_s:
                raise TimeoutError(f"[heavy_slots] gave up after {now - t0:.0f} s: " + compact_status(d, n))
            time.sleep(poll_s)
        waited = time.monotonic() - t0
        _drop(wait_rec, os.getpid())
        wait_rec = None
        rec_p = _write(d, Record.mine("holder", cmd, note=why, slot=slot))
        log(f"[heavy_slots] took heavy slot {slot} of {n} after waiting {waited:.1f} s ({why})")
        before = os.environ.get(HELD_ENV)
        os.environ[HELD_ENV] = f"{os.getpid()}:{slot}"
        t_acq = time.monotonic()
        try:
            yield slot
        finally:
            if before is None:
                os.environ.pop(HELD_ENV, None)
            else:
                os.environ[HELD_ENV] = before
            _drop(rec_p, os.getpid())
            fcntl.flock(fds[slot], fcntl.LOCK_UN)
            log(f"[heavy_slots] released heavy slot {slot} after {time.monotonic() - t_acq:.0f} s")
    finally:
        _drop(wait_rec, os.getpid())
        for fd in fds:
            os.close(fd)


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
            else:
                row.update(cmd=procfs.cmdline(pids[0]), why="(no record)")
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
    for k in side:
        side[k].sort(key=lambda r: -r["for_s"])
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
            out.append(f"  slot {r['slot']}: pid {','.join(map(str, r['pids']))} held {_dur(r['held_s'])} "
                       f"[{r['why']}] {r['cmd'][:140]}  (cwd {r['cwd'] or '?'})")
    out.append(f"  queue ({len(state['waiting'])}, oldest first; polled, so not a promise of order):")
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
