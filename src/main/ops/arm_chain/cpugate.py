"""The GO's CPU gate (orchestrator ruling 2026-10-20 ~15:25), measured on the TRAINER cores, never ``load1``.

THE CLASS BUG: ``load1 <= 12`` held GOs while agent jobs pinned to the agent cores (5-7,13-15) inflated the box-wide
load without touching a single trainer core. The gate is now, over one sample window (default 60 s):

(a) the trainer CPUs are at least ``idle_bar`` idle (idle + iowait jiffies over all jiffies, summed over them);
(b) no process whose affinity reaches OUTSIDE the agent cores used more than ``proc_bar`` % of one CPU (named) —
    one born inside the window counts too (a lower bound on its rate); one that exited inside it cannot be seen.

Read-only: it never signals anything. ``proc`` and ``sleep`` are seams for the tests.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Set, Tuple

HZ = os.sysconf("SC_CLK_TCK") if hasattr(os, "sysconf") else 100


def parse_cpu_list(s: str) -> Set[int]:
    out: Set[int] = set()
    for part in s.strip().split(","):
        if not part:
            continue
        a, _, b = part.partition("-")
        out.update(range(int(a), int(b or a) + 1))
    return out


def _cpu_jiffies(proc: str) -> Dict[int, Tuple[int, int]]:
    out: Dict[int, Tuple[int, int]] = {}
    with open(os.path.join(proc, "stat")) as f:
        for line in f:
            p = line.split()
            if p and p[0].startswith("cpu") and p[0] != "cpu":
                v = [int(x) for x in p[1:9]]
                out[int(p[0][3:])] = (v[3] + v[4], sum(v))
    return out


def _procs(proc: str) -> Dict[int, Tuple[int, str, str]]:
    out: Dict[int, Tuple[int, str, str]] = {}
    for d in os.listdir(proc):
        if not d.isdigit():
            continue
        try:
            with open(os.path.join(proc, d, "stat")) as f:
                st = f.read()
            rest = st[st.rindex(")") + 2:].split()
            comm = st[st.index("(") + 1:st.rindex(")")]
            with open(os.path.join(proc, d, "status")) as f:
                aff = next(line.split(":", 1)[1] for line in f if line.startswith("Cpus_allowed_list"))
            out[int(d)] = (int(rest[11]) + int(rest[12]), comm, aff.strip())
        except (OSError, ValueError, StopIteration, IndexError):
            continue
    return out


@dataclass(frozen=True)
class GateResult:
    ok: bool
    measured: bool
    idle_frac: float
    hogs: Tuple[str, ...]
    line: str


def sample(trainer_cpus: str, agent_cpus: str, sample_s: float = 60.0, idle_bar: float = 0.70,
           proc_bar: float = 50.0, proc: str = "/proc", sleep: Callable[[float], None] = time.sleep,
           monotonic: Callable[[], float] = time.monotonic, me: int = -1) -> GateResult:
    trainer, agent = parse_cpu_list(trainer_cpus), parse_cpu_list(agent_cpus)
    c0, p0, t0 = _cpu_jiffies(proc), _procs(proc), monotonic()
    sleep(sample_s)
    c1, p1 = _cpu_jiffies(proc), _procs(proc)
    dt = max(monotonic() - t0, 1e-9)
    if not trainer <= set(c0) or not trainer <= set(c1):
        return GateResult(False, False, 0.0, (), f"CANNOT MEASURE: /proc/stat lacks CPUs {sorted(trainer - set(c1))}")
    idle = sum(c1[c][0] - c0[c][0] for c in trainer)
    tot = sum(c1[c][1] - c0[c][1] for c in trainer)
    frac = idle / tot if tot else 0.0
    hogs: List[str] = []
    for pid, (j1, comm, aff) in sorted(p1.items()):
        if pid == me:
            continue
        try:
            cpus = parse_cpu_list(aff)
        except ValueError:
            continue
        if cpus <= agent:
            continue
        # a process BORN inside the window spent all its CPU inside it: its whole count over the window is a lower
        # bound on its rate (it used to be skipped, so a hog started mid-sample was invisible); one that EXITED
        # inside the window is invisible either way (its time moves to a parent's cutime)
        pct = 100.0 * (j1 - (p0[pid][0] if pid in p0 else 0)) / HZ / dt
        if pct > proc_bar:
            hogs.append(f"pid {pid} {pct:.0f}% affinity {aff} [{comm}]")
    head = f"CPUs {trainer_cpus} {100 * frac:.1f}% idle over {dt:.0f} s (bar {100 * idle_bar:.0f}%)"
    ok = frac >= idle_bar and not hogs
    if ok:
        return GateResult(True, True, frac, (), f"PASS: {head}; no process outside {agent_cpus} above {proc_bar:.0f}% CPU")
    why = ([] if frac >= idle_bar else [f"{head} — BELOW the bar"]) + (
        [f"{len(hogs)} process(es) outside {agent_cpus} above {proc_bar:.0f}% CPU: " + "; ".join(hogs[:4])] if hogs else [])
    return GateResult(False, True, frac, tuple(hogs), "FAIL: " + " | ".join(why))
