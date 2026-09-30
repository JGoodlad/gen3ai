"""A WINDOWED CPU-contention meter — ``gen3_contention_meter_v2``.

``utils.contention.cpu_contention_factor`` answers "how busy is the box right now" from the
1-minute load average. That is good enough to stretch a timeout, and it is the WRONG instrument for
the question the test tier asks — "was this test's duration a measurement of the test?" — for three
reasons, each of which cost a day of RED routine gates with 0 tests failed (2026-09-30):

1. **It is sampled once, at the end.** A burst that came and went during a 40 s test is invisible.
2. **load1 is a 1-minute EMA of the run queue**, so short-lived children and bursts are smoothed away.
3. **It cannot see SMT.** This box is 8 cores / 16 hardware threads. Ten runnable threads is load
   10 on 16 cpus — "0.63, quiet" — while two of them share each of two cores. A CPU-bound thread
   whose sibling is busy runs **1.70-1.83x slower** (pinned probe, 5 clean pairs). The Linux PSI
   ``cpu some`` stall share is blind to it too: there is no run-queue wait at all, just a slower
   core. MEASURED (calibration, below): 4-9 busy workers beside the day's background read
   PSI some 0.07-6 % while the probe ran 1.4-2.1x slower.

So this meter diffs three KERNEL-INTEGRATED counters across a window — no sampling holes — and
combines them with a declared, calibrated mapping:

* ``/proc/stat`` per-CPU busy ticks -> ``U``, the mean number of busy logical CPUs. Past the
  physical-core count ``P`` threads start sharing cores; the expected slowdown of a busy thread is
  ``g(U) = 1 + (s - 1) * min(1, 2 (U - P) / U)``, with ``s = SMT_SIBLING_SLOWDOWN``.
* the run-queue STRETCH ``1 + wait/run`` — how long a runnable thread waited per unit it ran — read
  from the SELF subtree's OWN tasks (``/proc/<pid>/task/<tid>/schedstat``) whenever there is a self.
  The box-wide ``/proc/schedstat`` (per-CPU ``run_delay`` / ``rq_cpu_time``) is the fallback, and
  it is NOT what our tests feel: MEASURED 2026-09-30, it read 1.06-1.23 with only 5-8 of 16 cpus
  busy, all of it one other agent's process queueing on itself, and one 29 s window read x11.1 at
  U 8.0 / PSI 1.7 %. Either form beats PSI here, which saturates: PSI some -> 1.0 at ~2 threads per
  cpu, so ``1 / (1 - some)`` read 8.5 where the probe measured 5.65 (N = 32 busy workers).
* PSI ``cpu some`` total -> reported always; the run-queue term falls back to it
  (``1 / (1 - some)``, conservative) when ``/proc/schedstat`` is unavailable. Where neither
  ``/proc/stat`` exists, the load average is the last resort, and the SOURCE says so.

**SELF IS NOT CONTENTION.** The pytest session's own process tree (and, per test, that test's own
worker subtree) is subtracted: a test that saturates the box by itself is exactly what the budget
must catch, and a meter that called it "contended" would excuse it. The occupancy term becomes
``g(U_total) / g(U_self)``, and the run-queue excess is scaled by the external share of demand.

CALIBRATION (2026-09-30, AMD 9800X3D 8C/16T shared with other agents; a pure-Python probe, quiet
0.396 s, beside K busy ``while 1: pass`` workers, every column read over the SAME 6 s window; the
full table is in ``designs/ops/testing.md``): K=8 probe 1.92x — load1 meter 1.00, PSI-only 1.00, this
meter 1.52; K=12 2.01x — 1.00 / 1.19 / 2.05; K=24 3.51x — 1.00 / 2.51 / 2.94. It under-reads the
probe by up to ~25 % in the 8-core band and never calls a slowed window quiet, which is the
direction the enforcement decision needs.

Pure stdlib, Linux ``/proc`` only; every reader returns ``None`` when its file is missing, and the
reading names which sources it used.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

__all__ = [
    "SMT_SIBLING_SLOWDOWN",
    "CpuSample",
    "ContentionReading",
    "take_sample",
    "reading_between",
    "override_reading",
    "loadavg_reading",
    "Topology",
    "topology",
]

#: Per-thread slowdown of a CPU-bound thread whose SMT sibling is busy, relative to having the core
#: alone. MEASURED 2026-09-30 (pinned probe, sibling busy vs idle, 5 clean pairs on cores 5-7):
#: 1.70 / 1.71 / 1.74 / 1.82 / 1.83. It folds in the all-core clock drop, which is real slowdown.
SMT_SIBLING_SLOWDOWN = 1.75

_MAX_FACTOR = 12.0
_CLK_TCK = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


@dataclass(frozen=True)
class Topology:
    logical: Tuple[int, ...]      # the CPUs this process may run on
    physical: int                 # distinct cores among them


def topology() -> Topology:
    try:
        cpus = tuple(sorted(os.sched_getaffinity(0)))
    except (AttributeError, OSError):          # pragma: no cover - non-Linux
        cpus = tuple(range(os.cpu_count() or 1))
    cores = set()
    for c in cpus:
        try:
            with open(f"/sys/devices/system/cpu/cpu{c}/topology/thread_siblings_list") as f:
                cores.add(f.read().strip())
        except OSError:
            cores.add(str(c))                  # unknown topology => assume no SMT
    return Topology(cpus, max(1, len(cores)))


@dataclass(frozen=True)
class CpuSample:
    """Cumulative counters at one instant. Diff two with :func:`reading_between`."""
    mono: float
    wall: float                                  # time.time(), to match pytest's report.start/stop
    busy_ticks: Optional[int]                    # /proc/stat, summed over the affinity CPUs
    total_ticks: Optional[int]
    run_ns: Optional[int]                        # /proc/schedstat rq_cpu_time
    wait_ns: Optional[int]                       # /proc/schedstat run_delay
    psi_some_us: Optional[int]                   # /proc/pressure/cpu some total
    load1: float
    self_cpu_s: Dict[int, float] = field(default_factory=dict)   # root pid -> its subtree's CPU
    # root pid -> {tid: (run_ns, wait_ns)} for every live task of its subtree (/proc/*/task/*/schedstat)
    self_tasks: Dict[int, Dict[int, Tuple[int, int]]] = field(default_factory=dict)


def _read_stat(cpus: Iterable[int]) -> Tuple[Optional[int], Optional[int]]:
    want = {f"cpu{c}" for c in cpus}
    busy = total = 0
    try:
        with open("/proc/stat") as f:
            for ln in f:
                if not ln.startswith("cpu"):
                    break
                p = ln.split()
                if p[0] not in want:
                    continue
                v = [int(x) for x in p[1:9]]
                b = v[0] + v[1] + v[2] + v[5] + v[6] + v[7]   # user nice system irq softirq steal
                busy += b
                total += b + v[3] + v[4]                    # + idle iowait
    except (OSError, ValueError, IndexError):
        return None, None
    return (busy, total) if total else (None, None)


def _read_schedstat(cpus: Iterable[int]) -> Tuple[Optional[int], Optional[int]]:
    want = {f"cpu{c}" for c in cpus}
    run = wait = 0
    try:
        with open("/proc/schedstat") as f:
            first = f.readline().split()
            if len(first) != 2 or first[0] != "version" or int(first[1]) not in (15, 16, 17):
                return None, None                           # unknown layout / jiffies units
            for ln in f:
                p = ln.split()
                if p and p[0] in want:
                    run += int(p[7])
                    wait += int(p[8])
    except (OSError, ValueError, IndexError):
        return None, None
    return (run, wait) if run else (None, None)


def _read_psi() -> Optional[int]:
    try:
        with open("/proc/pressure/cpu") as f:
            for tok in f.readline().split():
                if tok.startswith("total="):
                    return int(tok[6:])
    except (OSError, ValueError):
        pass
    return None


def _subtree_cpu(roots: Iterable[int]
                 ) -> Tuple[Dict[int, float], Dict[int, Dict[int, Tuple[int, int]]]]:
    """CPU seconds (utime + stime + reaped children) of each root's live process subtree, and of
    each root's direct children's subtrees.

    One pass over ``/proc/*/stat`` (~2.6 ms for ~550 processes). A descendant reaped inside the
    subtree moves into its parent's ``cutime`` and leaves the walk, so nothing is counted twice; an
    ORPHAN reparented out of the tree takes its CPU with it (the delta clamps at zero)."""
    roots = list(roots)
    if not roots:
        return {}, {}
    ppid: Dict[int, int] = {}
    cpu: Dict[int, float] = {}
    try:
        names = os.listdir("/proc")
    except OSError:
        return {}, {}
    for name in names:
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/stat", "rb") as f:
                raw = f.read()
        except OSError:
            continue
        rest = raw[raw.rfind(b")") + 2:].split()
        try:
            pid = int(name)
            ppid[pid] = int(rest[1])
            cpu[pid] = sum(int(x) for x in rest[11:15]) / _CLK_TCK
        except (ValueError, IndexError):
            continue
    children: Dict[int, List[int]] = {}
    for pid, parent in ppid.items():
        children.setdefault(parent, []).append(pid)
    def members(top: int) -> List[int]:
        out, stack = [], [top]
        while stack:
            pid = stack.pop()
            out.append(pid)
            stack.extend(children.get(pid, ()))
        return out

    task_cache: Dict[int, Dict[int, Tuple[int, int]]] = {}

    def tasks_of(pid: int) -> Dict[int, Tuple[int, int]]:
        if pid not in task_cache:
            found: Dict[int, Tuple[int, int]] = {}
            try:
                for tid in os.listdir(f"/proc/{pid}/task"):
                    try:
                        with open(f"/proc/{pid}/task/{tid}/schedstat") as f:
                            run, wait = f.read().split()[:2]
                        found[int(tid)] = (int(run), int(wait))
                    except (OSError, ValueError):
                        continue
            except OSError:
                pass
            task_cache[pid] = found
        return task_cache[pid]

    cpu_out: Dict[int, float] = {}
    tasks_out: Dict[int, Dict[int, Tuple[int, int]]] = {}

    def record(top: int) -> None:
        pids = members(top)
        cpu_out[top] = sum(cpu.get(p, 0.0) for p in pids)
        merged: Dict[int, Tuple[int, int]] = {}
        for p in pids:
            merged.update(tasks_of(p))
        tasks_out[top] = merged

    for root in roots:
        if root not in cpu:
            continue
        record(root)
        # Each DIRECT child too: under xdist those are the workers, so a test's own worker subtree
        # is on record from the moment the worker exists, before any of its reports has arrived.
        for child in children.get(root, ()):
            if child not in cpu_out:
                record(child)
    return cpu_out, tasks_out


def take_sample(roots: Iterable[int] = (), topo: Optional[Topology] = None) -> CpuSample:
    """Read every counter now. ``roots`` are the pids whose subtrees count as SELF."""
    topo = topo or topology()
    busy, total = _read_stat(topo.logical)
    run, wait = _read_schedstat(topo.logical)
    try:
        load1 = os.getloadavg()[0]
    except OSError:                                  # pragma: no cover
        load1 = 0.0
    self_cpu, self_tasks = _subtree_cpu(roots)
    return CpuSample(mono=time.monotonic(), wall=time.time(), busy_ticks=busy, total_ticks=total,
                     run_ns=run, wait_ns=wait, psi_some_us=_read_psi(), load1=load1,
                     self_cpu_s=self_cpu, self_tasks=self_tasks)


def smt_slowdown(busy: float, topo: Topology, s: float = SMT_SIBLING_SLOWDOWN) -> float:
    """Expected slowdown of a busy thread when ``busy`` logical CPUs are busy on average."""
    logical, physical = len(topo.logical), topo.physical
    if logical <= physical or busy <= physical:
        return 1.0
    shared = min(1.0, (logical / physical) * (busy - physical) / busy)
    return 1.0 + (s - 1.0) * shared


@dataclass(frozen=True)
class ContentionReading:
    factor: float
    source: str
    window_s: float = 0.0
    busy_cpus: Optional[float] = None
    self_cpus: Optional[float] = None
    logical: int = 0
    physical: int = 0
    smt: float = 1.0
    queue: float = 1.0
    psi_some: Optional[float] = None
    load1_factor: Optional[float] = None       # what the OLD meter (load1 / cpus) read

    def describe(self) -> str:
        bits = [f"contention factor {self.factor:.2f} [{self.source}"]
        if self.window_s:
            bits.append(f", {self.window_s:.0f}s window")
        if self.busy_cpus is not None:
            bits.append(f": {self.busy_cpus:.1f} of {self.logical} logical cpus busy "
                        f"({self.physical} cores), {self.self_cpus or 0.0:.1f} ours"
                        f" -> SMT x{self.smt:.2f}, run-queue x{self.queue:.2f}")
        if self.psi_some is not None:
            bits.append(f"; PSI cpu some {100 * self.psi_some:.1f}%")
        if self.load1_factor is not None:
            bits.append(f"; the old load1/cpus meter reads {self.load1_factor:.2f}")
        return "".join(bits) + "]"


def override_reading() -> Optional[ContentionReading]:
    """``GEN3AI_TIMEOUT_SCALE`` forces the factor, exactly as it does for ``cpu_contention_factor``."""
    raw = os.environ.get("GEN3AI_TIMEOUT_SCALE")
    if not raw:
        return None
    try:
        return ContentionReading(max(1.0, min(_MAX_FACTOR, float(raw))),
                                 "GEN3AI_TIMEOUT_SCALE override")
    except ValueError:
        return None


def loadavg_reading() -> ContentionReading:
    """The last resort when /proc/stat is unreadable: the old load1 / cpus meter, labelled as such."""
    ncpu = len(topology().logical)
    try:
        f = max(1.0, min(_MAX_FACTOR, os.getloadavg()[0] / ncpu))
    except OSError:                                  # pragma: no cover
        f = 1.0
    return ContentionReading(f, "load1/cpus FALLBACK (no /proc/stat)", load1_factor=f)


#: Below this much CPU run by the self subtree in a window, its own wait/run ratio is noise, and the
#: box-wide counter answers instead.
_MIN_OWN_RUN_NS = 200_000_000


def _own_stretch(a: CpuSample, b: CpuSample, self_root: Optional[int]) -> Optional[float]:
    """``1 + wait/run`` over the self subtree's OWN tasks — how long OUR threads waited for a cpu.

    Per task, so a task born inside the window counts from zero and one that exited drops out
    whole (its in-window share is lost, never double-counted)."""
    if self_root is None or self_root not in b.self_tasks:
        return None
    before = a.self_tasks.get(self_root, {})
    run = wait = 0
    for tid, (r1, w1) in b.self_tasks[self_root].items():
        r0, w0 = before.get(tid, (0, 0))
        if r1 >= r0 and w1 >= w0:                 # a recycled tid reads backwards: skip it
            run += r1 - r0
            wait += w1 - w0
    if run < _MIN_OWN_RUN_NS:
        return None
    return 1.0 + wait / run


def reading_between(a: CpuSample, b: CpuSample, *, self_root: Optional[int] = None,
                    topo: Optional[Topology] = None) -> ContentionReading:
    """The contention a thread in ``self_root``'s subtree saw between two samples.

    ``self_root=None`` treats ALL load as external."""
    override = override_reading()
    if override is not None:
        return override
    topo = topo or topology()
    logical = len(topo.logical)
    dt = max(1e-9, b.mono - a.mono)
    load1_factor = max(1.0, min(_MAX_FACTOR, b.load1 / logical))
    psi = None
    if a.psi_some_us is not None and b.psi_some_us is not None:
        psi = min(1.0, max(0.0, (b.psi_some_us - a.psi_some_us) / (dt * 1e6)))
    if (a.busy_ticks is None or b.busy_ticks is None or a.total_ticks is None
            or b.total_ticks is None or b.total_ticks <= a.total_ticks):
        fb = loadavg_reading()
        return ContentionReading(fb.factor, fb.source, window_s=dt, psi_some=psi,
                                 load1_factor=load1_factor)
    busy = logical * (b.busy_ticks - a.busy_ticks) / (b.total_ticks - a.total_ticks)
    mine = 0.0
    if self_root is not None and self_root in a.self_cpu_s and self_root in b.self_cpu_s:
        mine = (b.self_cpu_s[self_root] - a.self_cpu_s[self_root]) / dt
    elif self_root is not None and self_root in b.self_cpu_s:
        mine = b.self_cpu_s[self_root] / dt          # the subtree started inside the window
    mine = min(max(0.0, mine), busy)
    external_share = (busy - mine) / busy if busy > 0 else 0.0
    smt = smt_slowdown(busy, topo) / smt_slowdown(mine, topo)
    own = _own_stretch(a, b, self_root)
    if own is not None:
        stretch = own
        source = "/proc/stat occupancy + OWN tasks' run-queue delay"
    elif (a.run_ns is not None and b.run_ns is not None and a.wait_ns is not None
            and b.wait_ns is not None and b.run_ns > a.run_ns):
        stretch = 1.0 + (b.wait_ns - a.wait_ns) / (b.run_ns - a.run_ns)
        source = "/proc/stat occupancy + /proc/schedstat run-queue"
    elif psi is not None:
        stretch = 1.0 / max(0.05, 1.0 - psi)
        source = "/proc/stat occupancy + PSI cpu some (no /proc/schedstat)"
    else:
        stretch = 1.0
        source = "/proc/stat occupancy only (no run-queue counter)"
    queue = 1.0 + max(0.0, stretch - 1.0) * external_share
    factor = max(1.0, min(_MAX_FACTOR, smt * queue))
    return ContentionReading(factor, source, window_s=dt, busy_cpus=busy, self_cpus=mine,
                             logical=logical, physical=topo.physical, smt=smt, queue=queue,
                             psi_some=psi, load1_factor=load1_factor)


class WindowMeter:
    """Session-long sample log. :meth:`maybe_sample` is cheap to call often (rate-limited);
    :meth:`window` answers "what did the interval [start, stop] (wall clock) look like"."""

    def __init__(self, roots_fn, *, min_interval_s: float = 2.0) -> None:
        self._roots_fn = roots_fn
        self._min = min_interval_s
        self.topo = topology()
        self.samples: List[CpuSample] = [self._take()]

    def _take(self) -> CpuSample:
        return take_sample(self._roots_fn(), self.topo)

    def maybe_sample(self) -> None:
        if time.monotonic() - self.samples[-1].mono >= self._min:
            self.samples.append(self._take())

    def window(self, start_wall: float, stop_wall: float, self_root: Optional[int]
               ) -> ContentionReading:
        if self.samples[-1].wall < stop_wall:
            self.samples.append(self._take())
        before = [s for s in self.samples if s.wall <= start_wall]
        a = before[-1] if before else self.samples[0]
        b = next(s for s in self.samples if s.wall >= stop_wall)
        if b is a:                                   # a zero-length window: nothing to diff
            b = self._take()
            self.samples.append(b)
        return reading_between(a, b, self_root=self_root, topo=self.topo)

    def session(self, self_root: Optional[int]) -> ContentionReading:
        self.samples.append(self._take())
        return reading_between(self.samples[0], self.samples[-1], self_root=self_root,
                               topo=self.topo)
