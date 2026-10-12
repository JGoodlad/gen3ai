"""WHO holds the routine-gate slot, SINCE WHEN, how long it should take, and who is QUEUED behind it.

USAGE
    scripts/ops/gate_lock.sh --status            # human: one block per slot, then the queue
    scripts/ops/gate_lock.sh --status --json     # the same as JSON (an agent deciding whether to wait)

WHY (owner, 2026-10-12, "all subagents are all blocked on it"). With a training run live, the gate runs
on the 3 agent cores and takes over an hour; agents queued behind it had only a pid and a truncated
command line, and had to GUESS how long the wait would be. This module answers four questions from
durable records plus ``/proc`` — never from a guess:

* **who**: every holder writes ``slot<i>.json`` beside its slot file (pid, worktree, command, cpus,
  nice, ``-n``, acquired-at). A record is TRUSTED only while ``/proc/locks`` lists its pid holding that
  slot; a holder with no record (an older gate_lock, a bare ``flock``) falls back to ``/proc`` and says so;
* **since when / how long**: the acquisition time, and the gate's ETA = the MEDIAN wall of the last
  :data:`ETA_SAMPLES` gates of the SAME SHAPE (the same ``-n`` on the same number of cpus, a routine-gate
  command, exit 0 or 1 = it ran to the end) from ``history.jsonl``, which every release appends to.
  No comparable history ⇒ "no ETA", never a number from another shape;
* **is it alive**: the holder's whole process tree's CPU, averaged since acquisition and (CLI only)
  over a short sample window. A tree using under :data:`IDLE_CORES` of a core over the sample is
  flagged IDLE — a possible deadlock to ``py-spy dump``; never a licence to kill a gate you did not
  start (standing rule 1). A holder past :data:`OVERDUE_FACTOR` x its ETA is flagged OVERDUE;
* **who waits**: every blocked taker is a ``/proc/locks`` waiter on one slot file; its wait is its
  process age (a gate_lock process starts waiting at once). flock does not wake waiters in FIFO
  order, so the queue is listed oldest first but is NOT a promise of order.

The records only REPORT. The lock is the flock alone: a missing, stale or corrupt record changes what
``--status`` prints, never who holds a slot.
"""
from __future__ import annotations

import json
import os
import shlex
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from utils import procfs

RECORD_SCHEMA = "gen3_gate_slot_record_v1"
HISTORY_NAME = "history.jsonl"
HISTORY_KEEP = 400            # lines kept when the history is rewritten
ETA_SAMPLES = 7               # the median of at most this many comparable past gates
OVERDUE_FACTOR = 1.5
IDLE_CORES = 0.05
ROUTINE_MARK = "not slow and not e2e"


def record_path(d: Path, slot: int) -> Path:
    return d / f"slot{slot}.json"


def history_path(d: Path) -> Path:
    return d / HISTORY_NAME


def workers_of(cmd: Sequence[str]) -> Optional[int]:
    """The xdist ``-n`` of a pytest command line (``-n 3`` / ``-n3`` / ``--numprocesses 3``), else None.
    ``-n auto`` and friends are None: they are not a shape this module can compare."""
    args = list(cmd)
    for i, a in enumerate(args):
        val: Optional[str] = None
        if a in ("-n", "--numprocesses") and i + 1 < len(args):
            val = args[i + 1]
        elif a.startswith("--numprocesses="):
            val = a.split("=", 1)[1]
        elif a.startswith("-n") and len(a) > 2 and a[2:].isdigit():
            val = a[2:]
        if val is not None:
            return int(val) if val.isdigit() else None
    return None


def is_routine_gate(cmd: Sequence[str]) -> bool:
    return ROUTINE_MARK in " ".join(cmd) and "pytest" in " ".join(cmd)


@dataclass
class SlotRecord:
    """What a holder says about itself (``slot<i>.json``) — written after the flock is taken."""

    pid: int
    slot: int
    acquired: float
    cwd: str
    cmd: List[str]
    cpus: int
    nice: int
    workers: Optional[int]
    schema: str = RECORD_SCHEMA

    @classmethod
    def mine(cls, slot: int, cmd: Sequence[str]) -> "SlotRecord":
        return cls(pid=os.getpid(), slot=slot, acquired=time.time(), cwd=os.getcwd(), cmd=list(cmd)[:64],
                   cpus=len(os.sched_getaffinity(0)), nice=os.getpriority(os.PRIO_PROCESS, 0),
                   workers=workers_of(cmd))


def write_record(d: Path, rec: SlotRecord) -> None:
    """Atomic replace; a failure to write is reported by the caller and never stops a gate."""
    p = record_path(d, rec.slot)
    tmp = p.with_name(f".{p.name}.{os.getpid()}")
    tmp.write_text(json.dumps(asdict(rec)))
    os.replace(tmp, p)


def read_record(d: Path, slot: int) -> Optional[SlotRecord]:
    try:
        raw = json.loads(record_path(d, slot).read_text())
        if raw.get("schema") != RECORD_SCHEMA:
            return None
        return SlotRecord(**raw)
    except (OSError, ValueError, TypeError):
        return None


def clear_record(d: Path, slot: int, pid: int) -> None:
    """Remove ``slot<i>.json`` only if it is still ``pid``'s (a later holder's record is left alone)."""
    rec = read_record(d, slot)
    if rec is not None and rec.pid == pid:
        try:
            record_path(d, slot).unlink()
        except OSError:
            pass


def append_history(d: Path, rec: SlotRecord, held_s: float, rc: Optional[int]) -> None:
    """One line per finished gate. O_APPEND of one short line is atomic, so concurrent releases
    (N > 1) cannot interleave; the file is trimmed to :data:`HISTORY_KEEP` lines by the holder that
    finds it past twice that."""
    p = history_path(d)
    line = json.dumps({"acquired": round(rec.acquired, 1), "held_s": round(held_s, 1), "rc": rc,
                       "cpus": rec.cpus, "workers": rec.workers, "nice": rec.nice, "cwd": rec.cwd,
                       "routine": is_routine_gate(rec.cmd)}) + "\n"
    with open(p, "a") as f:
        f.write(line)
    try:
        lines = p.read_text().splitlines(keepends=True)
        if len(lines) > 2 * HISTORY_KEEP:
            tmp = p.with_name(f".{p.name}.{os.getpid()}")
            tmp.write_text("".join(lines[-HISTORY_KEEP:]))
            os.replace(tmp, p)
    except OSError:
        pass


def read_history(d: Path) -> List[dict]:
    out: List[dict] = []
    try:
        text = history_path(d).read_text()
    except OSError:
        return out
    for line in text.splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue                       # a torn or foreign line is skipped, never fatal
    return out


def eta_s(history: Sequence[dict], *, cpus: int, workers: Optional[int]) -> Optional[tuple]:
    """``(median wall s, n samples)`` of the last :data:`ETA_SAMPLES` finished ROUTINE gates of the
    same shape (``cpus`` and ``workers`` equal, exit 0 or 1), or None when there is none."""
    if workers is None:
        return None
    same = [h["held_s"] for h in history
            if h.get("routine") and h.get("cpus") == cpus and h.get("workers") == workers
            and h.get("rc") in (0, 1) and isinstance(h.get("held_s"), (int, float))]
    if not same:
        return None
    last = same[-ETA_SAMPLES:]
    return float(statistics.median(last)), len(last)


@dataclass
class TreeCpu:
    avg_cores: Optional[float]           # the tree's CPU since acquisition / wall since acquisition
    now_cores: Optional[float]           # over the sample window (None when not sampled)


def _tree_ticks(pid: int) -> Optional[int]:
    procs = procfs.snapshot()
    if pid not in procs:
        return None
    return procfs.subtree_cpu(procs, procfs.children_map(procs)).get(pid)


def tree_cpu(pid: int, since: float, sample_s: float = 0.0) -> TreeCpu:
    t0 = _tree_ticks(pid)
    if t0 is None:
        return TreeCpu(None, None)
    elapsed = max(1e-6, time.time() - since)
    avg = t0 / procfs.HZ / elapsed
    now = None
    if sample_s > 0:
        start = time.monotonic()
        time.sleep(sample_s)
        t1 = _tree_ticks(pid)
        if t1 is not None:
            now = max(0, t1 - t0) / procfs.HZ / (time.monotonic() - start)
    return TreeCpu(round(avg, 2), None if now is None else round(now, 2))


@dataclass
class Holder:
    slot: int
    pid: int
    since: Optional[float]               # acquisition (record) or process start (no record)
    since_source: str                    # "record", or the process start (no record: includes its queue wait)
    cwd: str
    cmd: str
    cpus: Optional[int]
    nice: Optional[int]
    workers: Optional[int]
    elapsed_s: Optional[float]
    eta_s: Optional[float]
    eta_n: int
    cpu: TreeCpu
    flags: List[str] = field(default_factory=list)


@dataclass
class Waiter:
    pid: int
    slot: int
    waiting_s: Optional[float]
    cwd: str
    cmd: str


def _proc_start_epoch(pid: int) -> Optional[float]:
    p = procfs.read_stat(pid)
    if p is None:
        return None
    try:
        return time.time() - procfs.age_s(p, procfs.uptime_s())
    except OSError:
        return None


def _gated_argv(pid: int) -> List[str]:
    """The command a ``gate_lock`` process runs (what follows its ``--``), else its whole argv. Read raw
    from ``/proc/<pid>/cmdline`` so an argument with spaces (``-m "not slow and not e2e"``) stays one."""
    try:
        with open(f"/proc/{pid}/cmdline", "rb") as f:
            argv = [a.decode(errors="replace") for a in f.read().split(b"\0") if a]
    except OSError:
        return []
    return argv[argv.index("--") + 1:] if "--" in argv else argv


def _nice_of(pid: int) -> Optional[int]:
    try:
        return os.getpriority(os.PRIO_PROCESS, pid)
    except OSError:
        return None


def _cwd(pid: int) -> str:
    try:
        return os.readlink(f"/proc/{pid}/cwd")
    except OSError:
        return ""


def _holders_of(d: Path, slot: int, locks) -> List[int]:
    key = procfs.file_key(str(d / f"slot{slot}.lock"))
    return procfs.holders(key, locks) if key else []


def _waiters_of(d: Path, slot: int, locks) -> List[int]:
    key = procfs.file_key(str(d / f"slot{slot}.lock"))
    return procfs.waiters(key, locks) if key else []


def collect(d: Path, n_slots: int, *, sample_s: float = 0.0, now: Optional[float] = None) -> dict:
    """``{"slots": [Holder | None per slot], "waiters": [Waiter, oldest first]}`` — read-only."""
    locks = procfs.read_locks()
    history = read_history(d)
    now = time.time() if now is None else now
    slots: List[Optional[Holder]] = []
    for i in range(n_slots):
        pids = _holders_of(d, i, locks)
        if not pids:
            slots.append(None)
            continue
        pid = pids[0]
        rec = read_record(d, i)
        if rec is not None and rec.pid == pid:
            since, src = rec.acquired, "record"
            cwd, cmd, cpus, nice, workers = rec.cwd, shlex.join(rec.cmd), rec.cpus, rec.nice, rec.workers
        else:
            since, src = _proc_start_epoch(pid), "process start, no slot record: includes its queue wait"
            argv = _gated_argv(pid)
            aff = affinity_of(pid)
            cwd, cmd, workers = _cwd(pid), shlex.join(argv), workers_of(argv)
            cpus, nice = (len(aff) if aff else None), _nice_of(pid)
        elapsed = None if since is None else max(0.0, now - since)
        est = eta_s(history, cpus=cpus, workers=workers) if cpus is not None else None
        cpu = tree_cpu(pid, since if since is not None else now, sample_s)
        h = Holder(slot=i, pid=pid, since=since, since_source=src, cwd=cwd, cmd=cmd, cpus=cpus, nice=nice,
                   workers=workers, elapsed_s=elapsed, eta_s=None if est is None else est[0],
                   eta_n=0 if est is None else est[1], cpu=cpu)
        if est is not None and elapsed is not None and elapsed > OVERDUE_FACTOR * est[0]:
            h.flags.append(f"OVERDUE: {_dur(elapsed)} against a typical {_dur(est[0])}")
        if cpu.now_cores is not None and cpu.now_cores < IDLE_CORES:
            h.flags.append(f"IDLE: its process tree used {cpu.now_cores:.2f} cores over {sample_s:.0f} s — "
                           f"a possible deadlock: `py-spy dump --pid {pid}` and its children; do NOT kill a gate "
                           "you did not start (standing rule 1), tell its owner")
        slots.append(h)
    waiters: List[Waiter] = []
    seen = set()
    for i in range(n_slots):
        for pid in _waiters_of(d, i, locks):
            if pid in seen:
                continue
            seen.add(pid)
            start = _proc_start_epoch(pid)
            waiters.append(Waiter(pid=pid, slot=i, waiting_s=None if start is None else max(0.0, now - start),
                                  cwd=_cwd(pid), cmd=shlex.join(_gated_argv(pid))))
    waiters.sort(key=lambda w: -(w.waiting_s or 0.0))
    return {"slots": slots, "waiters": waiters}


def _dur(s: Optional[float]) -> str:
    if s is None:
        return "?"
    s = int(round(s))
    if s < 90:
        return f"{s} s"
    if s < 3600:
        return f"{s // 60} min"
    return f"{s // 3600} h {(s % 3600) // 60:02d} min"


def _short(cmd: str, width: int = 140) -> str:
    return cmd if len(cmd) <= width else cmd[: width - 1] + "…"


def _hhmm(t: Optional[float]) -> str:
    return "?" if t is None else time.strftime("%H:%M:%S", time.localtime(t))


def format_human(state: dict) -> str:
    lines: List[str] = []
    for i, h in enumerate(state["slots"]):
        if h is None:
            lines.append(f"slot {i}: FREE")
            continue
        shape = []
        if h.workers is not None:
            shape.append(f"-n {h.workers}")
        if h.cpus is not None:
            shape.append(f"on {h.cpus} cpus")
        if h.nice is not None:
            shape.append(f"nice {h.nice}")
        lines.append(f"slot {i}: HELD by pid {h.pid} for {_dur(h.elapsed_s)} (since {_hhmm(h.since)}, "
                     f"{h.since_source}) — {' '.join(shape) or 'shape unknown'}")
        lines.append(f"    worktree: {h.cwd or '?'}")
        lines.append(f"    command:  {_short(h.cmd)}")
        if h.eta_s is not None:
            left = h.eta_s - (h.elapsed_s or 0.0)
            when = f"~{_dur(left)} left" if left > 0 else f"past it by {_dur(-left)}"
            lines.append(f"    ETA:      a gate of this shape typically takes {_dur(h.eta_s)} (median of the last "
                         f"{h.eta_n}) — {when}")
        else:
            lines.append("    ETA:      none (no finished routine gate of this -n on this many cpus in the history)")
        cpu = f"{h.cpu.avg_cores} cores average since it started" if h.cpu.avg_cores is not None else "tree gone"
        if h.cpu.now_cores is not None:
            cpu += f", {h.cpu.now_cores} cores now"
        lines.append(f"    CPU:      its process tree used {cpu}")
        for fl in h.flags:
            lines.append(f"    ⚠️  {fl}")
    ws = state["waiters"]
    if not ws:
        lines.append("queue: nobody waiting")
    else:
        lines.append(f"queue: {len(ws)} waiting (oldest first; flock wakes waiters in NO guaranteed order):")
        for w in ws:
            lines.append(f"    pid {w.pid} waiting {_dur(w.waiting_s)} — {w.cwd or '?'} — {_short(w.cmd, 100)}")
    return "\n".join(lines)


def format_compact(state: dict) -> str:
    """One line for the waiting log."""
    parts = []
    for i, h in enumerate(state["slots"]):
        if h is None:
            parts.append(f"slot {i} free")
            continue
        eta = ""
        if h.eta_s is not None:
            left = h.eta_s - (h.elapsed_s or 0.0)
            eta = f", ETA ~{_dur(left)}" if left > 0 else f", OVERDUE by {_dur(-left)}"
        parts.append(f"slot {i}: pid {h.pid} ({os.path.basename(h.cwd) or '?'}) held {_dur(h.elapsed_s)}{eta}")
    parts.append(f"{len(state['waiters'])} waiting")
    return "; ".join(parts)


def as_json(state: dict) -> str:
    def enc(o):
        return asdict(o) if o is not None else None
    return json.dumps({"slots": [enc(h) for h in state["slots"]],
                       "waiters": [asdict(w) for w in state["waiters"]]}, indent=1)


def cpu_list(cpus: Sequence[int]) -> str:
    """``[5, 6, 7, 13, 14, 15]`` → ``"5-7,13-15"``."""
    out: List[str] = []
    xs = sorted(set(cpus))
    i = 0
    while i < len(xs):
        j = i
        while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
            j += 1
        out.append(str(xs[i]) if i == j else f"{xs[i]}-{xs[j]}")
        i = j + 1
    return ",".join(out)


def affinity_of(pid: int) -> Optional[List[int]]:
    try:
        return sorted(os.sched_getaffinity(pid))
    except OSError:
        return None

