"""MEMORY-AWARE ADMISSION for the box's heavy slots (2026-11-09): who may take a slot is decided by the SUM OF THE DECLARED
CAPS, not by a fixed count of 2. ``utils.heavy_slots`` owns the slots, the queue and the records; this module owns the
arithmetic (pure: :func:`decide`) and the two reads it needs (``MemTotal``, the heavy slice's cgroup).

WHY. The fixed count was a memory rule in disguise ("memory cannot be shared by priority"), and it cost more queue time than
it bought: on 2026-11-09 the two slots had served 17.5 h of work against 23.9 h of queue wait by 12:17, while the heavy slice
held 4-9 GiB of its 64. Measured over 25,177 capped jobs (the ``[mem_cap] done:`` lines of two weeks): the median job peaks
at 2.2 GiB and the p99 at 8.1 GiB, no job passed its cap, and the kernel log since boot shows no OOM outside the mem_cap
tests' own slices. Every heavy job already DECLARES a cap (``mem_cap.sh <GB>``) that the kernel enforces on its scope
(``MemoryMax``, swap 0), so the sum of the caps of the running jobs is a bound on what they can use together.

THE RULE (:func:`decide`). A job that declares a cap of C is admitted when ALL hold:

* it is its turn (FIFO per lane, ``heavy_slots.grant_order``): every job queued ahead of it would be admitted with it;
* fewer than :func:`max_slots` (4; ``$GEN3AI_HEAVY_MAX_SLOTS``) slots are held - the bound is CPU: the agent cores are 6
  hardware threads and a heavy job uses 1-4 of them (a Kakuna unit 1.0-1.8, a routine gate 3);
* ``committed + C <= budget``, or no slot is held at all (a job alone always runs, so one whose cap exceeds the budget is
  never starved - the fixed count would have run it too).

``budget = min(the heavy slice's MemoryMax, MemTotal - reserve)`` (:func:`read_box`). The RESERVE (:data:`RESERVE_GB` = 28
GiB) is what is NOT the heavy jobs': the training run's process tree at its worst (18.4 GiB of summed RSS over 21 runs'
5-min samples, 2026-10-20..11-09; the median run peaks at 14.5), everything else that is not reclaimable (7.5 GiB measured
beside a live run: sessions, shared memory, kernel), and 2 GiB of margin. On this box: min(64, 89.7 - 28) = 61.7 GiB.
``committed`` = for each slot holder the larger of its declared cap and the ``memory.max`` of the capped scopes it owns
(a scope ``gen3ai-capped-<name>-<wrapper pid>.scope`` belongs to the holder that is its wrapper or an ancestor of it: a
gate's own ``mem_cap`` child), plus the ``memory.max`` of every other live capped scope in the slice (a BLOCKING or GPU
job, one started before the run went live, one whose wrapper died). The kernel is the source of truth for a running job;
the declared cap covers the second between admission and the scope's creation.

FAIL-SAFE: ANYTHING UNKNOWN IS THE FIXED COUNT (:data:`BASE_SLOTS` = 2, counted over every slot). A job that declares no
cap, an unreadable ``MemTotal`` or cgroup tree, a holder with neither a declared cap nor a scope yet, a scope with no
``memory.max``, a job with no declared cap queued ahead: each falls back to "admit while fewer than 2 slots
are held", which is the rule this replaces. A job on the OLD code queued ahead while both base slots are held does not hold
a capped job back (it cannot see the slots above the base); a job on this code that declares no cap does (FIFO, or a
stream of capped jobs would starve it).

THE SWITCH BACK. :func:`mode` is read at every poll, so it binds jobs already queued:

* ``python -m utils.heavy_slots --admission count`` writes ``ADMISSION`` in the slot directory: every job on this code is
  back on the fixed count of 2 at its next poll (``--admission memory`` removes it);
* ``$GEN3AI_HEAVY_ADMISSION=count|memory`` overrides the file for one process tree;
* an explicit ``$GEN3AI_HEAVY_SLOTS=N`` is the fixed-count rule with N slots for that process tree, as it always was.

Anything that is not exactly ``memory`` reads as ``count``.

MIXED VERSIONS. A read pin runs the ``heavy_slots`` of its commit for days: it sees ``slot0`` / ``slot1`` only and checks
no memory. So a capped job takes the HIGHEST free slot (the base slots stay free for the old code, whose jobs then show up
in ``committed`` through their scopes), and the slice's ``MemoryMax`` stays what protects the box: an unchecked old-code
job can push the sum of caps past the budget, never the slice's use past its cap.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

from utils import procfs

GiB = 1 << 30
MODE_MEMORY, MODE_COUNT = "memory", "count"
MODE_ENV = "GEN3AI_HEAVY_ADMISSION"
#: the box-wide switch: a file in the slot directory holding ``count`` (absent = memory)
MODE_FILE = "ADMISSION"
#: the fixed count this replaces, and what every unknown falls back to
BASE_SLOTS = 2
#: the hard maximum on the count (module docs: CPU, not memory)
MAX_SLOTS = 4
MAX_SLOTS_ENV = "GEN3AI_HEAVY_MAX_SLOTS"
#: MEASURED (module docs): what the heavy jobs must leave - the training tree's worst + everything else + margin
RESERVE_GB = 28.0
RESERVE_ENV = "GEN3AI_HEAVY_RESERVE_GB"
#: tests point these at a fake ``/proc/meminfo`` and a fake slice directory
MEMINFO_ENV = "GEN3AI_HEAVY_MEMINFO"
CGROUP_ENV = "GEN3AI_HEAVY_CGROUP_DIR"
#: the heavy slice (``utils.mem_cap`` creates the scopes; the names live here so that the slots can read them)
SLICE_ENV = "GEN3AI_HEAVY_SLICE"
SLICE_GB_ENV = "GEN3AI_HEAVY_SLICE_GB"
DEFAULT_SLICE = "gen3ai-heavy.slice"
DEFAULT_SLICE_GB = 64.0
UNIT_PREFIX = "gen3ai-capped-"
RULE_MEMORY, RULE_COUNT = "memory", "count-fallback"


def slice_name() -> str:
    return os.environ.get(SLICE_ENV) or DEFAULT_SLICE


def slice_gb() -> float:
    raw = os.environ.get(SLICE_GB_ENV)
    return float(raw) if raw else DEFAULT_SLICE_GB


def max_slots() -> int:
    raw = os.environ.get(MAX_SLOTS_ENV)
    n = int(raw) if raw else MAX_SLOTS
    if n < BASE_SLOTS:
        raise ValueError(f"{MAX_SLOTS_ENV}={raw!r}: need at least the base count of {BASE_SLOTS}")
    return n


def reserve_bytes() -> int:
    raw = os.environ.get(RESERVE_ENV)
    gb = float(raw) if raw else RESERVE_GB
    if not gb >= 0:
        raise ValueError(f"{RESERVE_ENV}={raw!r}: need a number of GiB >= 0")
    return int(gb * GiB)


def mode(d: Path) -> Tuple[str, str]:
    """``(mode, where it was set)``. The environment wins, then the ``ADMISSION`` file in the slot directory ``d``, then
    the default (memory). Fail-safe: a value that is not exactly ``memory``, or a file that cannot be read, is ``count``."""
    raw = (os.environ.get(MODE_ENV) or "").strip().lower()
    if raw:
        return (MODE_MEMORY if raw == MODE_MEMORY else MODE_COUNT), f"${MODE_ENV}={raw}"
    p = d / MODE_FILE
    try:
        raw = p.read_text().strip().lower()
    except FileNotFoundError:
        return MODE_MEMORY, "default"
    except OSError as e:
        return MODE_COUNT, f"{p} is unreadable ({e.__class__.__name__})"
    return (MODE_MEMORY if raw == MODE_MEMORY else MODE_COUNT), f"{p}"


def set_mode(d: Path, value: str) -> str:
    """Write the box-wide switch (``count``), or remove it (``memory``: the default). Returns the line to print."""
    if value not in (MODE_MEMORY, MODE_COUNT):
        raise ValueError(f"--admission takes {MODE_MEMORY!r} or {MODE_COUNT!r}, got {value!r}")
    p = d / MODE_FILE
    if value == MODE_MEMORY:
        try:
            p.unlink()
        except FileNotFoundError:
            pass
        return f"admission: memory (the default; removed {p}) - jobs take it at their next poll"
    d.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(f".{p.name}.{os.getpid()}")
    tmp.write_text(MODE_COUNT + "\n")
    os.replace(tmp, p)
    return (f"admission: count (wrote {p}) - every job on this code is back on the fixed count of {BASE_SLOTS} at its "
            f"next poll; `--admission {MODE_MEMORY}` undoes it")


def slice_dir() -> Optional[Path]:
    """The heavy slice's cgroup directory, which need not exist (no capped job has run yet); ``None`` when this user's
    systemd manager has no cgroup-v2 directory to look under (nothing can then be known about the running caps)."""
    raw = os.environ.get(CGROUP_ENV)
    if raw:
        return Path(raw)
    uid = os.getuid()
    manager = Path(f"/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service")
    if not manager.is_dir():
        return None
    parts = slice_name()[:-len(".slice")].split("-")
    return manager.joinpath(*("-".join(parts[:i + 1]) + ".slice" for i in range(len(parts))))


@dataclass(frozen=True)
class Scope:
    unit: str
    wrapper: int               # the ``mem_cap`` wrapper's pid, from the unit's name (0: not parseable)
    cap: Optional[int]         # ``memory.max`` in bytes; None: no limit in force, or unreadable


@dataclass(frozen=True)
class Box:
    budget: Optional[int]      # bytes the heavy jobs' caps may sum to; None: unknown (``why`` says what)
    scopes: Tuple[Scope, ...] = ()
    total: int = 0
    reserve: int = 0
    slice_cap: int = 0
    why: str = ""


def _read(p: Path) -> Optional[str]:
    try:
        return p.read_text().strip()
    except OSError:
        return None


def read_box() -> Box:
    """The memory the heavy jobs may commit, and every live capped scope in the heavy slice."""
    meminfo = os.environ.get(MEMINFO_ENV)
    total = procfs.mem_total_bytes(meminfo) if meminfo else procfs.mem_total_bytes()
    if total <= 0:
        return Box(None, why="MemTotal is unreadable")
    sd = slice_dir()
    if sd is None:
        return Box(None, total=total, why="this user's cgroup-v2 tree is not there to read the running caps from")
    reserve = reserve_bytes()
    slice_cap = int(slice_gb() * GiB)
    scopes: List[Scope] = []
    if sd.is_dir():
        raw = _read(sd / "memory.max")
        if raw is not None and raw.isdigit():
            slice_cap = int(raw)               # the kernel's value; "max" keeps the declared one (the next launch sets it)
        for p in sorted(sd.glob(UNIT_PREFIX + "*.scope")):
            events = _read(p / "cgroup.events")
            if events is None or "populated 1" not in events:
                continue                        # the scope is gone, or empty and about to be
            raw = _read(p / "memory.max")
            tail = p.name[:-len(".scope")].rsplit("-", 1)[-1]
            scopes.append(Scope(p.name, int(tail) if tail.isdigit() else 0,
                                int(raw) if raw is not None and raw.isdigit() else None))
    box = Box(min(slice_cap, total - reserve), tuple(scopes), total, reserve, slice_cap)
    uncapped = [s.unit for s in scopes if s.cap is None]
    if uncapped:
        return Box(None, box.scopes, total, reserve, slice_cap, why=f"{uncapped[0]} has no memory.max in force")
    if box.budget is None or box.budget <= 0:
        return Box(None, box.scopes, total, reserve, slice_cap,
                   why=f"MemTotal {total / GiB:.1f} GiB leaves nothing above the {reserve / GiB:.1f} GiB reserve")
    return box


@dataclass(frozen=True)
class Holder:
    slot: int
    pids: Tuple[int, ...]      # the flock holders of the slot (``/proc/locks``)
    declared: Optional[int]    # the cap in its holder record, bytes; None: no record of this holder, or no cap in it


def charges(holders: Sequence[Holder], scopes: Sequence[Scope],
            ancestors: Callable[[int], List[int]] = procfs.ancestors) -> Tuple[List[Optional[int]], int]:
    """``(one charge per holder, the bytes of the scopes no holder owns)``. A holder is charged the larger of its declared
    cap and the caps of the scopes it owns; ``None`` when it has neither (unknown). A scope is owned by the holder that
    is its wrapper or an ancestor of it; a scope whose wrapper is dead or is nobody's child is charged on its own."""
    owned = [0] * len(holders)
    other = 0
    for s in scopes:
        chain = [s.wrapper, *ancestors(s.wrapper)] if s.wrapper > 0 else []
        idx = next((i for i, h in enumerate(holders) if any(p in chain for p in h.pids)), None)
        if idx is None:
            other += s.cap or 0
        else:
            owned[idx] += s.cap or 0
    out: List[Optional[int]] = []
    for h, o in zip(holders, owned):
        c = max(h.declared or 0, o)
        out.append(c if c > 0 else None)
    return out, other


@dataclass(frozen=True)
class Decision:
    slot: Optional[int]        # the slot to take now, or None: wait
    rule: str                  # RULE_MEMORY | RULE_COUNT
    why: str


def decide(cap: Optional[int], *, free: Sequence[int], held: Sequence[Optional[int]], other: int,
           budget: Optional[int], ahead: Sequence[Optional[int]], base: int = BASE_SLOTS,
           unknown: str = "") -> Decision:
    """May a job with a declared cap of ``cap`` bytes (``None``: it declared none) take a slot NOW, and which?

    ``free``: the free slot indices, ascending. ``held``: one charge per held slot (:func:`charges`; ``None`` = unknown).
    ``other``: bytes of capped scopes that hold no slot. ``budget``: :attr:`Box.budget` (``None`` = unknown, ``unknown``
    says why). ``ahead``: the polling waiters ahead of this job in the grant order, each its declared cap in bytes, ``0``
    (it declared none: it is admitted by the fixed count, so nobody passes it) or ``None`` (it runs the code from before
    this rule, or its record is not written yet: it takes a base slot only). All integers: the comparison is exact."""
    free = sorted(free)

    def count(why: str) -> Decision:
        room = base - len(held)
        ok = bool(free) and room >= 1 and len(ahead) < room
        return Decision(free[0] if ok else None, RULE_COUNT,
                        f"{why}: the fixed count of {base} ({len(held)} held, {len(ahead)} queued ahead)")

    if cap is None:
        return count("this job declares no cap")
    if budget is None:
        return count(unknown or "the memory budget is unknown")
    if any(c is None for c in held):
        return count("a slot holder's cap is not known (it declared none and has no capped scope yet)")
    left = budget - other - sum(c for c in held if c is not None)
    avail = list(free)
    running = len(held)
    for w in ahead:
        if w is None:
            if any(i < base for i in avail):
                return count("a job on the code from before this rule is queued ahead and a base slot is free")
            continue                           # it has no claim above the base slots, and those are held
        if w <= 0:
            return count("a job that declares no cap is queued ahead")
        if not avail:
            return Decision(None, RULE_MEMORY, "every slot is held or spoken for by the queue ahead")
        if w > left and running > 0:
            return Decision(None, RULE_MEMORY, f"queued behind a job whose cap ({w / GiB:.1f} GiB) does not fit yet "
                                               f"({left / GiB:.1f} GiB of the budget is uncommitted)")
        avail.pop()
        running += 1
        left -= w
    if not avail:
        return Decision(None, RULE_MEMORY, "every slot is held or spoken for by the queue ahead")
    if cap > left and running > 0:
        return Decision(None, RULE_MEMORY, f"its cap ({cap / GiB:.1f} GiB) does not fit: {left / GiB:.1f} GiB of the "
                                           f"{budget / GiB:.1f} GiB budget is uncommitted")
    return Decision(avail[-1], RULE_MEMORY, f"cap {cap / GiB:.1f} GiB fits ({left / GiB:.1f} GiB of the "
                                            f"{budget / GiB:.1f} GiB budget was uncommitted)"
                    if running > 0 else f"cap {cap / GiB:.1f} GiB, alone (a job alone always runs)")
