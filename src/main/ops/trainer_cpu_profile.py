"""``python -m main.ops.trainer_cpu_profile``: how much CPU a LIVE training run uses, per thread and per phase.

The question it answers is the one the box's core split rests on (``designs/ops/core_split.md``): the trainer is
pinned to a set of hardware threads and agent work to the rest, so "how many of its threads does the trainer keep
busy, in which phase, and does it ever wait for a core" decides how many it needs.

It READS ``/proc`` ONLY. It never signals, re-pins or renices anything, and it is safe beside a live run::

    CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c <the agent cores of rule 5b> \\
        python -m main.ops.trainer_cpu_profile sample --launcher-pid <pid> --seconds 900 --out rows.jsonl
    python -m main.ops.trainer_cpu_profile summarize rows.jsonl [--json]

``sample`` writes one JSON line per interval (default 1 s; a PPO cycle of this trainer is ~70 s, so 15 min is ~13
cycles). Each line holds, for the interval:

* ``groups``: busy threads per THREAD GROUP of the launcher's process tree (CPU ticks / elapsed ticks, so 1.0 is one
  hardware thread kept busy). The groups are :data:`GROUPS`: ``env_workers`` (the Rust env process's ``rust_env_w*``
  battle workers), ``env_main`` (its main thread), ``learner`` (the trainer's Python main thread: batched inference
  during a rollout, the PPO update otherwise), ``autograd`` (``pt_autograd_*``), ``cuda`` (the driver's threads),
  ``compile`` (torch's inductor compile-worker pool), ``launcher`` (the supervising parent) and ``other``.
* ``wait``: seconds the tree's threads spent RUNNABLE BUT NOT RUNNING (``schedstat`` run-delay), per group: the
  direct reading of "the trainer wanted a core and did not get one".
* ``cpu``: the busy fraction of every logical CPU from ``/proc/stat`` (every tenant, not only the trainer).
* ``on``: the trainer tree's busy threads attributed to the logical CPU each thread last ran on.

``summarize`` splits the intervals into PHASES by the env workers alone, which is deterministic and needs no log
timestamps: ``rollout`` when the env workers kept >= :data:`ROLLOUT_MIN` threads busy, ``update`` when they kept
<= :data:`UPDATE_MAX` busy, and ``transition`` in between (an interval that straddles the boundary; EXCLUDED from
both phases rather than assigned by chance). It reports per phase the busy threads (mean, p95, max), the per-group
means, the run-queue wait per second, and the peak of the busiest PHYSICAL core among the tree's own CPUs.

The fps of the same window comes from the run's log (:func:`log_progress`: the last ``total_timesteps`` /
``time_elapsed`` rows at the first and the last sample).

``burst`` answers what a 1 s interval cannot: HOW MANY THREADS WANT A CORE AT THE SAME INSTANT. A 1 s mean of 2 busy
env workers is 8 workers each a quarter busy, or 2 always busy, and only the first needs 8 hardware threads. It reads
the run state (``R`` = running or waiting for a CPU) of the env workers, the learner and the autograd threads every
``--period`` seconds (default 4 ms) and ``summarize`` reports, per phase (the same rule, on 1 s bins of the mean
runnable workers): the share of time with k workers runnable, the peak number of runnable threads, the bursts per
second and the mean burst length (runnable share / burst rate, so the sampling period does not bias it)::

    python -m main.ops.trainer_cpu_profile burst --launcher-pid <pid> --seconds 150 --out burst.jsonl

Stdlib only. Tests: ``src/main/ops/trainer_cpu_profile_test.py`` (a faked ``/proc``).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from utils import procfs

#: The thread groups, in report order.
GROUPS = ("env_workers", "env_main", "learner", "autograd", "cuda", "compile", "launcher", "other")
#: An interval is ROLLOUT when the env workers kept at least this many threads busy ...
ROLLOUT_MIN = 1.0
#: ... UPDATE when they kept at most this many busy, and TRANSITION (excluded from both) in between.
UPDATE_MAX = 0.10
PHASES = ("rollout", "update", "transition")
DEFAULT_INTERVAL_S = 1.0
#: How often the process tree is re-walked (a restart or a new worker shows up within this many intervals).
TREE_EVERY = 10
_TAIL_BYTES = 262_144
_ROW = re.compile(r"\|\s*(total_timesteps|time_elapsed)\s*\|\s*([0-9.e+]+)\s*\|")


def classify(comm: str, tid: int, pid: int, launcher: int, trainer: int, cmdline: str) -> str:
    """The :data:`GROUPS` member a thread belongs to. ``comm`` is the THREAD's name, ``cmdline`` its process's."""
    if comm.startswith("rust_env_w"):
        return "env_workers"
    if comm.startswith("rust_env"):
        return "env_main"
    if "compile_worker" in cmdline:
        return "compile"
    if pid == launcher:
        return "launcher"
    if comm.startswith("pt_autograd"):
        return "autograd"
    if comm.startswith("cuda"):
        return "cuda"
    if pid == trainer and tid == pid:
        return "learner"
    return "other"


@dataclass(frozen=True)
class Thread:
    """One ``/proc/<pid>/task/<tid>`` reading: CPU ticks, run-queue wait (ns), and the CPU it last ran on."""

    pid: int
    tid: int
    group: str
    ticks: int
    wait_ns: int
    cpu: int


def _read(path: str) -> str:
    try:
        with open(path, "rb") as f:
            return f.read().decode("utf-8", "replace")
    except OSError:
        return ""


def read_thread(pid: int, tid: int, proc: str) -> Optional[Tuple[str, int, int, int]]:
    """``(comm, ticks, wait_ns, last_cpu)`` of one thread, or ``None`` if it is gone."""
    raw = _read(f"{proc}/{pid}/task/{tid}/stat")
    lp, rp = raw.find("("), raw.rfind(")")
    if lp < 0 or rp < 0:
        return None
    rest = raw[rp + 2:].split()
    try:
        ticks, cpu = int(rest[11]) + int(rest[12]), int(rest[36])
    except (IndexError, ValueError):
        return None
    sched = _read(f"{proc}/{pid}/task/{tid}/schedstat").split()
    wait_ns = int(sched[1]) if len(sched) >= 2 and sched[1].isdigit() else 0
    return raw[lp + 1:rp], ticks, wait_ns, cpu


def tree_pids(launcher: int, proc: str) -> List[int]:
    """The launcher and every descendant, parents first."""
    procs = procfs.snapshot(proc)
    if launcher not in procs:
        return []
    return [launcher] + procfs.descendants(launcher, procfs.children_map(procs))


def trainer_pid(pids: Sequence[int], launcher: int, proc: str) -> int:
    """The trainer child: the launcher's descendant with the most accumulated CPU that is not an env process or a
    compile worker (it is the only one that runs for the whole life of the run). 0 when there is none."""
    best, best_cpu = 0, -1
    for pid in pids:
        if pid == launcher:
            continue
        st = procfs.read_stat(pid, proc)
        if st is None or st.comm.startswith("rust_env") or "compile_worker" in procfs.cmdline(pid, proc):
            continue
        if st.utime + st.stime > best_cpu:
            best, best_cpu = pid, st.utime + st.stime
    return best


def read_threads(pids: Sequence[int], launcher: int, trainer: int, proc: str) -> Dict[int, Thread]:
    out: Dict[int, Thread] = {}
    for pid in pids:
        try:
            tids = [int(t) for t in os.listdir(f"{proc}/{pid}/task") if t.isdigit()]
        except OSError:
            continue
        cmd = procfs.cmdline(pid, proc)
        for tid in tids:
            r = read_thread(pid, tid, proc)
            if r is not None:
                out[tid] = Thread(pid, tid, classify(r[0], tid, pid, launcher, trainer, cmd), r[1], r[2], r[3])
    return out


def read_cpus(proc: str) -> Dict[int, Tuple[int, int]]:
    """``{cpu: (busy_ticks, total_ticks)}`` from ``/proc/stat`` (idle and iowait are not busy)."""
    out: Dict[int, Tuple[int, int]] = {}
    for line in _read(f"{proc}/stat").splitlines():
        m = re.match(r"cpu(\d+)\s+(.*)", line)
        if m:
            v = [int(x) for x in m.group(2).split()[:8]]
            total = sum(v)
            out[int(m.group(1))] = (total - v[3] - (v[4] if len(v) > 4 else 0), total)
    return out


def interval_row(t: float, dt: float, a: Dict[int, Thread], b: Dict[int, Thread],
                 ca: Dict[int, Tuple[int, int]], cb: Dict[int, Tuple[int, int]], hz: int) -> Dict[str, Any]:
    """One output row from two consecutive readings ``dt`` seconds apart. Only a thread present in BOTH readings
    counts: one first seen in ``b`` may carry ticks older than the interval (a process found by a later tree walk), and
    one that vanished took its last ticks with it. Either loses at most one interval of one thread."""
    groups = {g: 0.0 for g in GROUPS}
    wait = {g: 0.0 for g in GROUPS}
    on: Dict[int, float] = {}
    for tid, th in b.items():
        prev = a.get(tid)
        if prev is None:
            continue
        busy = max(0, th.ticks - prev.ticks) / (hz * dt)
        groups[th.group] += busy
        wait[th.group] += max(0, th.wait_ns - prev.wait_ns) / 1e9
        if busy > 0:
            on[th.cpu] = on.get(th.cpu, 0.0) + busy
    cpu = {}
    for c, (busy_b, tot_b) in cb.items():
        busy_a, tot_a = ca.get(c, (busy_b, tot_b))
        cpu[str(c)] = round((busy_b - busy_a) / (tot_b - tot_a), 4) if tot_b > tot_a else 0.0
    return {"t": round(t, 3), "dt": round(dt, 4), "groups": {g: round(v, 4) for g, v in groups.items()},
            "wait": {g: round(v, 4) for g, v in wait.items()}, "cpu": cpu,
            "on": {str(c): round(v, 4) for c, v in sorted(on.items())}}


def log_progress(log: Path) -> Optional[Tuple[int, float]]:
    """``(total_timesteps, time_elapsed)`` of the LAST logger table in the run's log, or ``None``."""
    try:
        with open(log, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - _TAIL_BYTES))
            tail = f.read().decode("utf-8", "replace")
    except OSError:
        return None
    last: Dict[str, float] = {}
    for m in _ROW.finditer(tail):
        last[m.group(1)] = float(m.group(2))
    if "total_timesteps" not in last or "time_elapsed" not in last:
        return None
    return int(last["total_timesteps"]), last["time_elapsed"]


def affinity(pid: int, proc: str) -> List[int]:
    """The CPUs ``pid`` may run on (``Cpus_allowed_list``), ``[]`` when unreadable."""
    for line in _read(f"{proc}/{pid}/status").splitlines():
        if line.startswith("Cpus_allowed_list:"):
            out: List[int] = []
            for part in line.split(":", 1)[1].strip().split(","):
                lo, _, hi = part.partition("-")
                if lo.isdigit():
                    out.extend(range(int(lo), int(hi or lo) + 1))
            return out
    return []


def sample(launcher: int, seconds: float, out: Path, *, interval: float = DEFAULT_INTERVAL_S, log: Optional[Path] = None,
           proc: str = "/proc", clock: Callable[[], float] = time.monotonic,
           sleep: Callable[[float], None] = time.sleep, hz: int = procfs.HZ) -> int:
    """Sample for ``seconds`` and write the rows to ``out``. Returns the number of interval rows written."""
    pids = tree_pids(launcher, proc)
    if not pids:
        raise SystemExit(f"trainer_cpu_profile: launcher pid {launcher} is not running")
    trainer = trainer_pid(pids, launcher, proc)
    head = {"header": 1, "kind": "sample", "launcher": launcher, "trainer": trainer, "interval": interval, "hz": hz,
            "affinity": affinity(trainer or launcher, proc), "log": str(log) if log else None,
            "log_start": log_progress(log) if log else None, "wall_start": time.time()}
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        f.write(json.dumps(head) + "\n")
        t0 = clock()
        prev_t, prev, prev_c = t0, read_threads(pids, launcher, trainer, proc), read_cpus(proc)
        while prev_t - t0 < seconds:
            sleep(max(0.0, interval - (clock() - prev_t)))
            if n % TREE_EVERY == TREE_EVERY - 1:
                pids = tree_pids(launcher, proc)
                if not pids:
                    break
            now, cur, cur_c = clock(), read_threads(pids, launcher, trainer, proc), read_cpus(proc)
            if now <= prev_t:
                continue
            f.write(json.dumps(interval_row(now - t0, now - prev_t, prev, cur, prev_c, cur_c, hz)) + "\n")
            f.flush()
            n += 1
            prev_t, prev, prev_c = now, cur, cur_c
        f.write(json.dumps({"footer": 1, "log_end": log_progress(log) if log else None, "wall_end": time.time()}) + "\n")
    return n


# ---------------------------------------------------------------------------------------------------------------------
# burst: the instantaneous demand

#: The groups whose run state ``burst`` reads (the only ones that keep a CPU busy in a rollout or an update).
BURST_GROUPS = ("env_workers", "learner", "autograd")
DEFAULT_PERIOD_S = 0.004


def _state(fd: int) -> str:
    """The run-state letter of an open ``/proc/<pid>/task/<tid>/stat`` ('' when the thread is gone)."""
    try:
        raw = os.pread(fd, 512, 0)
    except OSError:
        return ""
    rp = raw.rfind(b")")
    return raw[rp + 2:rp + 3].decode("ascii", "replace") if rp >= 0 else ""


def burst(launcher: int, seconds: float, out: Path, *, period: float = DEFAULT_PERIOD_S, proc: str = "/proc",
          clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep) -> int:
    """Read the run state of the tree's hot threads every ``period`` seconds for ``seconds``; one row
    ``[t, runnable env workers, runnable learner, runnable autograd]`` per reading. Returns the row count."""
    pids = tree_pids(launcher, proc)
    if not pids:
        raise SystemExit(f"trainer_cpu_profile: launcher pid {launcher} is not running")
    trainer = trainer_pid(pids, launcher, proc)
    fds: Dict[str, List[int]] = {g: [] for g in BURST_GROUPS}
    for tid, th in read_threads(pids, launcher, trainer, proc).items():
        if th.group in fds:
            try:
                fds[th.group].append(os.open(f"{proc}/{th.pid}/task/{tid}/stat", os.O_RDONLY))
            except OSError:
                pass
    n = 0
    try:
        with open(out, "w", encoding="utf-8") as f:
            f.write(json.dumps({"header": 1, "kind": "burst", "launcher": launcher, "trainer": trainer, "period": period,
                                "threads": {g: len(v) for g, v in fds.items()},
                                "affinity": affinity(trainer or launcher, proc), "wall_start": time.time()}) + "\n")
            t0 = clock()
            due = t0
            while True:
                now = clock()
                if now - t0 >= seconds:
                    break
                row = [round(now - t0, 4)] + [sum(1 for fd in fds[g] if _state(fd) == "R") for g in BURST_GROUPS]
                f.write(json.dumps(row) + "\n")
                n += 1
                due = max(due + period, clock())      # a late reading never causes a catch-up flurry
                sleep(max(0.0, due - clock()))
    finally:
        for v in fds.values():
            for fd in v:
                os.close(fd)
    return n


def summarize_burst(rows: Sequence[Any]) -> Dict[str, Any]:
    """The per-phase instantaneous demand of a ``burst`` file's rows."""
    head = next((r for r in rows if isinstance(r, dict) and r.get("header")), {})
    data = [r for r in rows if isinstance(r, list)]
    bins: Dict[int, List[List[float]]] = {}
    for r in data:
        bins.setdefault(int(r[0]), []).append(r)
    phase: Dict[int, str] = {}
    for sec, v in bins.items():
        g = {k: 0.0 for k in GROUPS}
        g["env_workers"] = sum(r[1] for r in v) / len(v)
        phase[sec] = phase_of({"groups": g})
    out: Dict[str, Any] = {"kind": "burst", "readings": len(data), "period": head.get("period"),
                           "threads": head.get("threads"), "affinity": head.get("affinity"), "phases": {}}
    for ph in PHASES:
        secs = sorted(s for s, p in phase.items() if p == ph)
        sel = [r for s in secs for r in bins[s]]
        if not sel:
            out["phases"][ph] = {"seconds": 0, "readings": 0}
            continue
        n = len(sel)
        hist: Dict[int, int] = {}
        for r in sel:
            hist[int(r[1])] = hist.get(int(r[1]), 0) + 1
        starts = 0
        for s in secs:                       # a burst START: 0 runnable workers, then >= 1, inside one second's bin
            v = bins[s]
            starts += sum(1 for a, b in zip(v, v[1:]) if a[1] == 0 and b[1] >= 1)
        active = sum(1 for r in sel if r[1] >= 1) / n
        rate = starts / len(secs)
        out["phases"][ph] = {
            "seconds": len(secs), "readings": n,
            "workers_runnable_share": {str(k): round(c / n, 4) for k, c in sorted(hist.items())},
            "workers_runnable_mean": round(sum(r[1] for r in sel) / n, 3),
            "threads_runnable_peak": int(max(r[1] + r[2] + r[3] for r in sel)),
            "threads_runnable_mean": round(sum(r[1] + r[2] + r[3] for r in sel) / n, 3),
            "learner_or_autograd_runnable_share": round(sum(1 for r in sel if r[2] + r[3] >= 1) / n, 4),
            "workers_and_learner_together_share": round(sum(1 for r in sel if r[1] >= 1 and r[2] + r[3] >= 1) / n, 4),
            "bursts_per_s": round(rate, 2), "workers_active_share": round(active, 4),
            "burst_ms_mean": round(1000.0 * active / rate, 2) if rate else 0.0,
        }
    return out


def render_burst(s: Dict[str, Any]) -> str:
    lines = [f"trainer burst profile: {s['readings']} readings every {s['period']} s, threads {s['threads']}, "
             f"trainer CPUs {s['affinity']}"]
    for ph in PHASES:
        p = s["phases"][ph]
        if not p["readings"]:
            continue
        lines.append(f"\n[{ph}] {p['seconds']} s: runnable threads peak {p['threads_runnable_peak']}, mean "
                     f"{p['threads_runnable_mean']:.2f}; {p['bursts_per_s']:.1f} worker bursts per s of "
                     f"{p['burst_ms_mean']:.1f} ms (workers active {100 * p['workers_active_share']:.1f} % of the time; "
                     f"together with the learner {100 * p['workers_and_learner_together_share']:.1f} %)")
        lines.append("  share of time with k env workers runnable: " + ", ".join(
            f"{k}: {100 * v:.1f} %" for k, v in p["workers_runnable_share"].items()))
    return "\n".join(lines)


# ---------------------------------------------------------------------------------------------------------------------
# summarize


def phase_of(row: Dict[str, Any]) -> str:
    w = row["groups"]["env_workers"]
    if w >= ROLLOUT_MIN:
        return "rollout"
    if w <= UPDATE_MAX:
        return "update"
    return "transition"


def percentile(xs: Sequence[float], q: float) -> float:
    """The nearest-rank percentile (``q`` in 0-100): a value that occurs in the data, no interpolation."""
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, max(0, math.ceil(len(s) * q / 100 - 1e-9) - 1))]


def _stats(xs: Sequence[float]) -> Dict[str, float]:
    if not xs:
        return {"mean": 0.0, "p95": 0.0, "max": 0.0}
    return {"mean": round(sum(xs) / len(xs), 3), "p95": round(percentile(xs, 95), 3), "max": round(max(xs), 3)}


def _physical(cpu: int, n_cpus: int) -> int:
    """The physical core of a logical CPU on this box's topology: CPU n and n + n_cpus/2 are SMT siblings."""
    return cpu % (n_cpus // 2) if n_cpus >= 2 else cpu


def summarize(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """The per-phase profile of a ``sample`` file's rows (header, intervals, footer)."""
    head: Dict[str, Any] = {}
    foot: Dict[str, Any] = {}
    ivs: List[Dict[str, Any]] = []
    for r in rows:
        if r.get("header"):
            head = r
        elif r.get("footer"):
            foot = r
        else:
            ivs.append(r)
    n_cpus = max((int(c) for r in ivs for c in r["cpu"]), default=-1) + 1
    own = set(head.get("affinity") or range(n_cpus))
    out: Dict[str, Any] = {"intervals": len(ivs), "seconds": round(sum(r["dt"] for r in ivs), 1),
                           "affinity": sorted(own), "phases": {}}
    for ph in PHASES + ("all",):
        sel = [r for r in ivs if ph == "all" or phase_of(r) == ph]
        secs = sum(r["dt"] for r in sel)
        total = [sum(r["groups"].values()) for r in sel]
        cores: List[float] = []      # the busiest physical core of the tree's own CPUs, every tenant (/proc/stat)
        logical: List[float] = []    # the busiest logical CPU of the tree's own CPUs
        own_busy: List[float] = []   # busy logical CPUs among the tree's own, every tenant
        for r in sel:
            per_core: Dict[int, float] = {}
            for c, v in r["cpu"].items():
                if int(c) in own:
                    per_core[_physical(int(c), n_cpus)] = per_core.get(_physical(int(c), n_cpus), 0.0) + v
            if per_core:
                cores.append(max(per_core.values()))
                logical.append(max(v for c, v in r["cpu"].items() if int(c) in own))
                own_busy.append(sum(v for c, v in r["cpu"].items() if int(c) in own))
        out["phases"][ph] = {
            "intervals": len(sel), "seconds": round(secs, 1), "busy_threads": _stats(total),
            "groups": {g: _stats([r["groups"][g] for r in sel]) for g in GROUPS},
            "wait_s_per_s": round(sum(sum(r["wait"].values()) for r in sel) / secs, 4) if secs else 0.0,
            "wait_s_per_s_by_group": {g: round(sum(r["wait"][g] for r in sel) / secs, 4) if secs else 0.0
                                      for g in GROUPS},
            "own_cpus_busy_all_tenants": _stats(own_busy),
            "busiest_logical_cpu": _stats(logical), "busiest_physical_core": _stats(cores),
        }
    a, b = head.get("log_start"), foot.get("log_end")
    if a and b and b[1] > a[1]:
        out["fps"] = {"steps": b[0] - a[0], "seconds": round(b[1] - a[1], 1), "fps": round((b[0] - a[0]) / (b[1] - a[1]), 1)}
    return out


def render(s: Dict[str, Any]) -> str:
    lines = [f"trainer CPU profile: {s['intervals']} intervals, {s['seconds']} s, trainer CPUs {s['affinity']}"]
    if "fps" in s:
        lines.append(f"fps over the window (the run's log): {s['fps']['fps']:.0f} "
                     f"({s['fps']['steps']} steps / {s['fps']['seconds']} s)")
    for ph in PHASES + ("all",):
        p = s["phases"][ph]
        if not p["intervals"]:
            continue
        bt = p["busy_threads"]
        lines.append(f"\n[{ph}] {p['seconds']} s ({p['intervals']} intervals): busy threads mean {bt['mean']:.2f}, "
                     f"p95 {bt['p95']:.2f}, max {bt['max']:.2f}; run-queue wait {p['wait_s_per_s']:.3f} s per s")
        lines.append("  " + ", ".join(f"{g} {p['groups'][g]['mean']:.2f} (p95 {p['groups'][g]['p95']:.2f})"
                                      for g in GROUPS if p["groups"][g]["max"] >= 0.005))
        lines.append(f"  own CPUs, every tenant: busy {p['own_cpus_busy_all_tenants']['mean']:.2f} "
                     f"(p95 {p['own_cpus_busy_all_tenants']['p95']:.2f}); busiest logical CPU p95 "
                     f"{p['busiest_logical_cpu']['p95']:.2f}, max {p['busiest_logical_cpu']['max']:.2f}; busiest "
                     f"physical core (2 threads) p95 {p['busiest_physical_core']['p95']:.2f}, max "
                     f"{p['busiest_physical_core']['max']:.2f}")
    return "\n".join(lines)


def load(path: Path) -> List[Any]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m main.ops.trainer_cpu_profile", description=(__doc__ or "").split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample", help="sample a live run's process tree (reads /proc only)")
    s.add_argument("--launcher-pid", type=int, required=True)
    s.add_argument("--seconds", type=float, required=True)
    s.add_argument("--out", type=Path, required=True)
    s.add_argument("--interval", type=float, default=DEFAULT_INTERVAL_S)
    s.add_argument("--log", type=Path, default=None, help="the run's launcher_child.full.log, for the window's fps")
    b = sub.add_parser("burst", help="sample the run state of the hot threads at a high rate (reads /proc only)")
    b.add_argument("--launcher-pid", type=int, required=True)
    b.add_argument("--seconds", type=float, required=True)
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--period", type=float, default=DEFAULT_PERIOD_S)
    z = sub.add_parser("summarize", help="the per-phase profile of a sample or a burst file")
    z.add_argument("rows", type=Path)
    z.add_argument("--json", action="store_true")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    a = _parser().parse_args(argv)
    if a.cmd == "sample":
        n = sample(a.launcher_pid, a.seconds, a.out, interval=a.interval, log=a.log)
        print(f"trainer_cpu_profile: {n} intervals -> {a.out}")
        return 0
    if a.cmd == "burst":
        n = burst(a.launcher_pid, a.seconds, a.out, period=a.period)
        print(f"trainer_cpu_profile: {n} readings -> {a.out}")
        return 0
    rows = load(a.rows)
    if rows and isinstance(rows[0], dict) and rows[0].get("kind") == "burst":
        s = summarize_burst(rows)
        print(json.dumps(s, indent=2) if a.json else render_burst(s))
        return 0
    s = summarize(rows)
    print(json.dumps(s, indent=2) if a.json else render(s))
    return 0


if __name__ == "__main__":
    sys.exit(main())
