"""``python -m main.ops.core_split_ab``: the core split's 10-minute A/B on a live training run.

The design and the rule are ``designs/ops/core_split.md`` §5; the lists are ``utils.core_split``'s. The question: how
much slower does a rollout get when the trainer keeps 4 or 3 of its 5 physical cores? The answer decides the split.

**What it does to the run.** It changes ONE thing, the CPU affinity of the threads of ONE launcher's process tree, and
only to a SUBSET of the CPUs that tree already has. It never signals, renices or widens anything, and it touches no
process outside that tree. It puts the original list back when it ends, on any error, and on every catchable signal
that would end or stop it (:data:`_HANDLED`; a SIGHUP the caller ignores, as under ``nohup``, stays ignored). The
restore is VERIFIED by a walk after the last set, and it reaches every process a narrowing touched even when that
process has left the tree (the launcher died, a child was re-parented): each is named in the marker
``utils.core_split.ab_marker_path()`` BEFORE it is narrowed. The marker is removed only after a verified restore, so
``python -m utils.core_split show`` reports a narrowing that is still in place. A SIGKILL cannot be caught:
``restore`` reads the marker and undoes it (it refuses while the driver that wrote the marker is alive).

The run takes ~9 minutes: start it detached and wait on its PID, never in a foreground call that a tool may cut and
never under a job killer (``mem_cap.sh --max-s``, ``timeout -s KILL``); the watcher on its PID ends with ``restore``,
which does nothing after a clean exit and undoes the narrowing after a kill::

    A="$(python -m utils.core_split agent-cpus)"
    nice -n 19 taskset -c "$A" python -m main.ops.core_split_ab run --launcher-pid <pid> --out-dir <dir> --check
    setsid nohup nice -n 19 taskset -c "$A" python -m main.ops.core_split_ab run --launcher-pid <pid> \\
        --out-dir <dir> > <dir>.log 2>&1 &                                             # the training session
    python -m main.ops.core_split_ab verdict <dir>/record.json                         # re-read a record
    python -m main.ops.core_split_ab restore                                           # no marker: does nothing

**The measurement.** Arm A is the tree's own list (5 cores), B and C drop its highest one and two physical cores
(``core_split.narrowed``). A rollout (~24 s) holds three 8-second segments, one per arm, in :data:`ROTATION` order
over six rollouts; the update between rollouts runs on arm A. The driver reads the run state of the TRAIN env's
workers every 4 ms (the env process with the most CPU time; the eval env is a second env process and its bursts are
not a rollout's; ``trainer_cpu_profile``'s reader and file format, so ``trainer_cpu_profile summarize`` reads the
recording too) and finds a rollout's start as the first runnable worker after :attr:`Timing.idle_s` with none,
confirmed by :attr:`Timing.confirm_bursts` bursts. A rollout ends with its third segment, or at once when no worker
was runnable for :attr:`Timing.end_idle_s`; the open segment then ends THERE, never at the time planned for it. Per
segment it drops the first second after the switch and takes the mean time from one burst's start to the next. Per
rollout: period(B) / period(A), period(C) / period(A). Projected loss of steps per second = :data:`COLLECT_SHARE` x
(mean ratio - 1).

**A segment counts only when its CPU list was verified.** Every switch reports whether the last walk found every
thread on the arm's list (``settled``), and every segment's end reads each thread's list back before the next switch
(``on_arm``). A segment where either is not True is never a measurement: its arm is INCONCLUSIVE.

**The rule (the orchestrator's, 2026-11-09, bar 3 %).** An arm PASSES at a projected loss <= 2.5 % and FAILS at
>= 3.5 %; between the two it is INCONCLUSIVE. It is also INCONCLUSIVE when the standard error of its loss exceeds
0.5 %, when an eval landed in the measured window (from rollout 1's start to the control's last logger table: an eval
prints its line when it ends), when the control (``update_wall_s`` of the six updates against the three before) moved
more than 10 %, when a segment is unverified, missing or too thin to read (below), or when the A/B was cut short.
INCONCLUSIVE counts as a fail. Adopt the narrowest arm that passes.

**A sampling gap is excluded, never tolerated.** The driver runs beside busy agent jobs, so a reading can be late.
A burst-to-burst interval that contains a gap longer than :attr:`Timing.gap_s` (shorter than a burst, so no burst can
hide in an accepted gap) is dropped; a segment with more than :data:`MAX_DROPPED` of its intervals dropped, or fewer
than :data:`MIN_INTERVALS` left, makes the arm INCONCLUSIVE.

Tests: ``src/main/ops/core_split_ab_test.py`` (a fake tree and a fake clock; real syscalls only on a ``sleep`` tree
the test spawns). Never run it on a run you did not launch.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import re
import signal
import sys
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, Iterator, List, Optional, Sequence, Set, Tuple

from main.ops import trainer_cpu_profile as T
from utils import core_split, procfs

SCHEMA = "core_split_ab_v1"
#: arm -> physical cores dropped from the tree's own list
ARMS: Tuple[Tuple[str, int], ...] = (("A", 0), ("B", 1), ("C", 2))
#: the arm of each of a rollout's three segments, per rollout: every arm sits in every position twice
ROTATION: Tuple[str, ...] = ("ABC", "BCA", "CAB", "CBA", "ACB", "BAC")
#: the collect share of a PPO cycle (core_split.md §2): what a slower rollout costs in steps per second
COLLECT_SHARE = 0.353
PASS_MAX, FAIL_MIN = 0.025, 0.035
SE_MAX = 0.005
CONTROL_MAX = 0.10
MAX_DROPPED = 0.25
MIN_INTERVALS = 40
BEFORE_TABLES = 3
_APPLY_PASSES = 5
_RESTORE_TRIES = 3
_TAIL_BYTES = 4_000_000
_TABLE_ROW = re.compile(rb"\|\s*(update_wall_s|iterations)\s*\|\s*([0-9.eE+-]+)\s*\|")
_EVAL_LINE = re.compile(rb"\[(?:SELFPLAY )?EVAL(?: LEDGER)?\]")

EXIT_OK, EXIT_USAGE, EXIT_REFUSED, EXIT_UNRESTORED = 0, 2, 3, 4


@dataclass(frozen=True)
class Timing:
    period_s: float = T.DEFAULT_PERIOD_S    # between readings of the workers' run state
    segment_s: float = 8.0
    settle_s: float = 1.0                   # dropped after a switch
    idle_s: float = 2.0                     # this long with no runnable worker = between rollouts
    confirm_s: float = 0.5                  # a rollout's start is confirmed by ...
    confirm_bursts: int = 4                 # ... this many bursts in confirm_s (8.6 expected), else it was a blip
    end_idle_s: float = 0.3                 # this long with no runnable worker inside a rollout = it is over
    gap_s: float = 0.012                    # a late reading: shorter than a burst (16 ms)
    max_s: float = 1500.0                   # give up (six cycles are ~7 min)
    tables_wait_s: float = 240.0            # wait for the control's last logger table


class Refusal(Exception):
    """The A/B may not start (nothing was changed)."""


class _Stop(BaseException):
    """A signal asked the driver to stop: unwind through the restore."""

    def __init__(self, signum: int) -> None:
        super().__init__(signum)
        self.signum = signum


def _is_launcher(argv: Sequence[str]) -> bool:
    return core_split.is_trainer_argv(argv) == "launcher"


class Tree:
    """ONE launcher's process tree: the only thing this module changes, and only its threads' CPU affinity.

    ``touched`` names every process (``{pid: "pid:starttime"}``) a NARROWING ever walked: a restore owes each of
    them its list back even after it LEFT the tree (the launcher died, or the process was re-parented), and
    :attr:`on_touch` is called BEFORE the first set on a new one, so the marker on disk names it first."""

    def __init__(self, launcher: int, *, proc: str = "/proc",
                 setaff: Callable[[int, Set[int]], None] = os.sched_setaffinity,
                 getaff: Callable[[int], Set[int]] = os.sched_getaffinity,
                 is_launcher: Callable[[Sequence[str]], bool] = _is_launcher,
                 key: Optional[str] = None, touched: Optional[Dict[int, str]] = None) -> None:
        self.launcher, self.proc, self._set, self._get = launcher, proc, setaff, getaff
        if key is None:
            st = procfs.read_stat(launcher, proc)
            if st is None or st.state in ("Z", "X"):
                raise Refusal(f"pid {launcher} is not running")
            if not is_launcher([a for a in procfs.read_text(launcher, "cmdline", proc).split("\0") if a]):
                raise Refusal(f"pid {launcher} is not a training launcher (`-m main.launcher`)")
            key = st.key
        self.key = key                                # a given key (a marker's) is a RESTORE: the launcher may be gone
        self.touched: Dict[int, str] = dict(touched or {})
        self.on_touch: Optional[Callable[[], None]] = None
        self._fds: List[int] = []
        self.n_workers = 0
        self.env_pid = 0

    def alive(self) -> bool:
        """Still the SAME process (pid and start time), and not a zombie."""
        st = procfs.read_stat(self.launcher, self.proc)
        return st is not None and st.key == self.key and st.state not in ("Z", "X")

    def procs(self, strays: bool = False) -> Dict[int, str]:
        """``{pid: key}`` of the launcher and its descendants, walked NOW (nothing when the launcher is gone or its
        pid is another process). ``strays``: also every process in :attr:`touched` that is still the same process,
        and its descendants (a child inherits its creator's list)."""
        snap = procfs.snapshot(self.proc)
        kids = procfs.children_map(snap)
        roots: List[int] = []
        st = snap.get(self.launcher)
        if st is not None and st.key == self.key and st.state not in ("Z", "X"):
            roots.append(self.launcher)
        if strays:
            roots += [pid for pid, key in sorted(self.touched.items()) if pid in snap and snap[pid].key == key]
        out: Dict[int, str] = {}
        for root in roots:
            for pid in [root] + procfs.descendants(root, kids):
                if snap[pid].state not in ("Z", "X"):
                    out[pid] = snap[pid].key
        return out

    def _threads(self, procs: Dict[int, str]) -> Dict[int, int]:
        out: Dict[int, int] = {}
        for pid in procs:
            try:
                out.update({int(t): pid for t in os.listdir(f"{self.proc}/{pid}/task") if t.isdigit()})
            except OSError:
                continue
        return out

    def tids(self, strays: bool = False) -> Dict[int, int]:
        """``{tid: pid}`` of every thread of :meth:`procs`, walked NOW."""
        return self._threads(self.procs(strays))

    def note(self, procs: Dict[int, str]) -> None:
        """Name ``procs`` in :attr:`touched` (and so in the marker) as holding, or about to hold, a narrowed list."""
        new = {pid: key for pid, key in procs.items() if self.touched.get(pid) != key}
        if new:
            self.touched.update(new)
            if self.on_touch is not None:
                self.on_touch()

    def masks(self, strays: bool = False, *, procs: Optional[Dict[int, str]] = None) -> Dict[int, Optional[FrozenSet[int]]]:
        out: Dict[int, Optional[FrozenSet[int]]] = {}
        for tid in self._threads(self.procs(strays) if procs is None else procs):
            try:
                out[tid] = frozenset(self._get(tid))
            except OSError:
                out[tid] = None                       # it exited between the walk and the read
        return out

    def apply(self, cpus: Sequence[int], within: Sequence[int], *, strays: bool = False) -> bool:
        """Give every thread of the tree the list ``cpus``, which must lie inside ``within`` (the tree's ORIGINAL
        list: this never widens). The tree is re-walked until a pass changes nothing (a thread born during a pass
        inherits its creator's list). True ONLY when the last pass found at least one thread, found every thread on
        ``cpus``, and no set on a thread that still exists failed: anything else is an unverified list."""
        target = set(cpus)
        if not target or not target <= set(within):
            raise Refusal(f"refusing to set {core_split.format_cpus(cpus)}: it is not inside the tree's own "
                          f"{core_split.format_cpus(within)}")
        for _ in range(_APPLY_PASSES):
            procs = self.procs(strays)
            if target != set(within):                 # write-ahead: the marker names a process before it is narrowed
                self.note(procs)
            threads = self._threads(procs)
            changed = failed = 0
            for tid, pid in threads.items():
                try:
                    if set(self._get(tid)) == target:
                        continue
                except OSError:
                    continue                          # it exited
                if not os.path.isdir(f"{self.proc}/{pid}/task/{tid}"):          # still that process's thread
                    continue
                try:
                    self._set(tid, set(target))
                    changed += 1
                except OSError:
                    failed += 1 if os.path.isdir(f"{self.proc}/{pid}/task/{tid}") else 0
            if threads and not changed and not failed:
                return True
        return False

    def open_workers(self) -> int:
        """Open the run-state files of the TRAIN env's worker threads and return how many. The eval env is a second
        env process with workers of its own, and its bursts are not a rollout's: the train env is the env process
        with the most CPU time, and a tree where that is not clear-cut is refused."""
        self.close()
        pids = T.tree_pids(self.launcher, self.proc)
        trainer = T.trainer_pid(pids, self.launcher, self.proc)
        by_pid: Dict[int, List[int]] = {}
        for tid, th in sorted(T.read_threads(pids, self.launcher, trainer, self.proc).items()):
            if th.group == "env_workers":
                by_pid.setdefault(th.pid, []).append(tid)
        self.n_workers = self.env_pid = 0
        if not by_pid:
            return 0
        cpu: Dict[int, int] = {}
        for pid in by_pid:
            st = procfs.read_stat(pid, self.proc)
            cpu[pid] = st.utime + st.stime if st is not None else 0
        ranked = sorted(by_pid, key=lambda pid: (-cpu[pid], pid))
        if len(ranked) > 1 and 2 * cpu[ranked[1]] >= cpu[ranked[0]]:
            raise Refusal(f"cannot tell the train env from the eval env: env processes {ranked[0]} and {ranked[1]} "
                          f"have {cpu[ranked[0]]} and {cpu[ranked[1]]} CPU ticks (the train env should have more than "
                          "twice the other's; start after the third logged iteration)")
        self.env_pid = ranked[0]
        for tid in by_pid[self.env_pid]:
            try:
                self._fds.append(os.open(f"{self.proc}/{self.env_pid}/task/{tid}/stat", os.O_RDONLY))
            except OSError:
                pass
        self.n_workers = len(self._fds)
        return self.n_workers

    def runnable(self) -> int:
        """How many train-env workers are running or waiting for a CPU right now; -1 when ANY of them is gone (the env
        process restarted: what is left is not the rollout that was being measured)."""
        states = [T._state(fd) for fd in self._fds]
        return sum(1 for s in states if s == "R") if states and all(states) else -1

    def run_dir(self) -> Optional[Path]:
        pids = T.tree_pids(self.launcher, self.proc)
        argv = [a for a in procfs.read_text(T.trainer_pid(pids, self.launcher, self.proc) or self.launcher,
                                            "cmdline", self.proc).split("\0") if a]
        for i, a in enumerate(argv):
            if a == "--run-dir" and i + 1 < len(argv):
                return Path(argv[i + 1])
            if a.startswith("--run-dir="):
                return Path(a.split("=", 1)[1])
        return None

    def close(self) -> None:
        for fd in self._fds:
            with contextlib.suppress(OSError):
                os.close(fd)
        self._fds = []


# ---------------------------------------------------------------------------------------------------------------------
# the run's log: the control and the evals


def read_tables(log: Path, offset: int = 0, end: Optional[int] = None) -> List[Dict[str, Any]]:
    """The logger tables of ``log`` in bytes ``[offset, end)``, in order: ``{"iterations", "update_wall_s"}``. A table
    is closed by its ``iterations`` row; its ``update_wall_s`` is the one printed since the table before."""
    try:
        with open(log, "rb") as f:
            f.seek(offset)
            raw = f.read() if end is None else f.read(max(0, end - offset))
    except OSError:
        return []
    out: List[Dict[str, Any]] = []
    wall: Optional[float] = None
    for m in _TABLE_ROW.finditer(raw):
        try:
            v = float(m.group(2))
        except ValueError:
            continue
        if not math.isfinite(v):                      # a logged `inf` / `nan` is not a reading
            continue
        if m.group(1) == b"update_wall_s":
            wall = v
        else:
            out.append({"iterations": int(v), "update_wall_s": wall})
            wall = None
    return out


def eval_lines(log: Path, start: int, end: int) -> int:
    try:
        with open(log, "rb") as f:
            f.seek(start)
            return len(_EVAL_LINE.findall(f.read(max(0, end - start))))
    except OSError:
        return 0


def _size(p: Optional[Path]) -> int:
    try:
        return p.stat().st_size if p is not None else 0
    except OSError:
        return 0


def _rows(p: Optional[Path]) -> int:
    try:
        return sum(1 for line in open(p, "rb") if line.strip()) if p is not None else 0
    except OSError:
        return 0


# ---------------------------------------------------------------------------------------------------------------------
# the arithmetic (pure: re-run by `verdict`)


def segment_period(rows: Sequence[Sequence[float]], t_from: float, t_to: float, gap_s: float) -> Dict[str, Any]:
    """The mean burst-start-to-burst-start time of the readings ``[t, runnable workers]`` in ``[t_from, t_to)``.
    A burst START is a reading with a runnable worker after one with none (``trainer_cpu_profile``'s rule). An
    interval that contains two readings more than ``gap_s`` apart is dropped."""
    kept: List[float] = []
    dropped = 0
    prev_t: Optional[float] = None
    prev_w = 1.0
    last_start: Optional[float] = None
    dirty = False
    for r in rows:
        t, w = r[0], r[1]
        if t < t_from:
            continue
        if t >= t_to:
            break
        late = prev_t is not None and t - prev_t > gap_s
        dirty = dirty or late
        if prev_t is not None and prev_w == 0 and w >= 1:
            if last_start is not None:
                if dirty:
                    dropped += 1
                else:
                    kept.append(t - last_start)
            last_start, dirty = t, late
        prev_t, prev_w = t, w
    n = len(kept) + dropped
    return {"intervals": len(kept), "dropped": dropped, "dropped_share": round(dropped / n, 4) if n else 1.0,
            "period_s": sum(kept) / len(kept) if kept else None}


def _mean_se(xs: Sequence[float]) -> Tuple[float, float]:
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, float("inf")
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))


def control_of(before: Sequence[Optional[float]], after: Sequence[Optional[float]], n_after: int) -> Dict[str, Any]:
    """``update_wall_s`` of the updates during the A/B against the :data:`BEFORE_TABLES` before it."""
    b = [x for x in before if x is not None]
    a = [x for x in after if x is not None]
    out: Dict[str, Any] = {"before": list(before), "after": list(after), "moved": None, "ok": False}
    if len(b) < BEFORE_TABLES or len(a) < n_after:
        out["why"] = f"{len(b)} of {BEFORE_TABLES} updates before and {len(a)} of {n_after} during were logged"
        return out
    moved = (sum(a) / len(a)) / (sum(b) / len(b)) - 1.0
    out.update(moved=round(moved, 5), ok=abs(moved) <= CONTROL_MAX)
    if not out["ok"]:
        out["why"] = f"update_wall_s moved {100 * moved:+.1f} % (limit {100 * CONTROL_MAX:.0f} %)"
    return out


def decide(segments: Sequence[Dict[str, Any]], *, rollouts: int, control: Dict[str, Any], evals: int,
           aborted: Optional[str] = None) -> Dict[str, Any]:
    """The verdict per arm and the split to adopt, from the per-segment periods. ``segments``: dicts with
    ``rollout``, ``arm``, ``intervals``, ``dropped_share``, ``period_s``, and ``settled`` / ``on_arm`` (the arm's
    list was verified on every thread after the switch / at the segment's end; anything but ``True`` is unverified)."""
    by: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for s in segments:
        by.setdefault(int(s["rollout"]), {})[str(s["arm"])] = s
    shared: List[str] = []
    if aborted:
        shared.append(f"the A/B did not finish: {aborted}")
    if evals:
        shared.append(f"an eval landed in the measured window ({evals} eval line(s) / row(s))")
    if not control.get("ok"):
        shared.append("the control: " + str(control.get("why", "not read")))
    arms: Dict[str, Any] = {}
    for arm, drop in ARMS[1:]:
        why = list(shared)
        ratios: List[float] = []
        for k in range(rollouts):
            pair = [by.get(k, {}).get(a) for a in ("A", arm)]
            thin = [f"rollout {k + 1} arm {a}" for a, s in zip(("A", arm), pair)
                    if s is None or s.get("period_s") is None or s["intervals"] < MIN_INTERVALS
                    or s["dropped_share"] > MAX_DROPPED]
            loose = [f"rollout {k + 1} arm {a}" for a, s in zip(("A", arm), pair)
                     if s is not None and not (s.get("settled") is True and s.get("on_arm") is True)]
            if loose:
                why.append("the CPU list was not verified on every thread (at the switch or at the segment's end): "
                           + ", ".join(loose))
            if thin:
                why.append("too thin to read (< %d intervals or > %d %% dropped, or the segment is missing): %s"
                           % (MIN_INTERVALS, round(100 * MAX_DROPPED), ", ".join(thin)))
            if not thin and not loose:
                ratios.append(pair[1]["period_s"] / pair[0]["period_s"])        # type: ignore[index]
        row: Dict[str, Any] = {"dropped_cores": drop, "ratios": [round(r, 5) for r in ratios]}
        if len(ratios) == rollouts and rollouts >= 2:
            mean, se = _mean_se(ratios)
            loss, loss_se = COLLECT_SHARE * (mean - 1.0), COLLECT_SHARE * se
            row.update(ratio_mean=round(mean, 5), ratio_se=round(se, 5), loss=round(loss, 5), loss_se=round(loss_se, 5))
            if loss_se > SE_MAX:
                why.append(f"the standard error of the projected loss is {100 * loss_se:.2f} % (limit {100 * SE_MAX:.1f} %)")
            if not why:
                if loss <= PASS_MAX:
                    row["verdict"] = "PASS"
                elif loss >= FAIL_MIN:
                    row["verdict"] = "FAIL"
                else:
                    why.append(f"the projected loss {100 * loss:.2f} % is between the pass and the fail line")
        row.setdefault("verdict", "INCONCLUSIVE")
        row["why"] = why
        arms[arm] = row
    passing = [a for a, _ in ARMS[1:] if arms[a]["verdict"] == "PASS"]
    return {"verdicts": arms, "adopt_arm": passing[-1] if passing else "A", "passing": passing}


# ---------------------------------------------------------------------------------------------------------------------
# the marker and the restore


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _create_marker(path: Path, obj: Any) -> None:
    """Create the marker, or refuse when one exists. It is written beside its place and LINKED into it: a link
    fails when the name exists, so two drivers cannot both pass, and no reader ever sees half a marker."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".new{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    try:
        os.link(tmp, path)
    except FileExistsError:
        raise Refusal(f"an A/B marker exists ({path}): another A/B is running, or one died. "
                      "`python -m main.ops.core_split_ab restore` clears it") from None
    finally:
        tmp.unlink()


def _own_key(proc: str = "/proc") -> str:
    st = procfs.read_stat(os.getpid(), proc)
    return st.key if st is not None else ""


def restore_tree(tree: Tree, original: Sequence[int]) -> Dict[str, Any]:
    """Every thread of the tree, and of every process this driver narrowed that has left the tree, back on
    ``original``. ``verified`` is True only when a walk AFTER the last set found each of them on ``original``. A
    launcher that is gone (a new process with its pid is never touched) leaves nothing to walk: that is verified only
    because every process named in ``tree.touched`` was looked for as well."""
    settled = tree.apply(original, original, strays=True)
    masks = tree.masks(strays=True)
    off = sorted(t for t, m in masks.items() if m is not None and m != frozenset(original))
    alive = tree.alive()
    return {"verified": not off and (settled if masks else not alive), "threads": len(masks), "off": off[:20],
            "launcher_alive": alive}


#: Every catchable signal whose default action would end or stop the driver with the tree still narrowed (the
#: synchronous faults and SIGKILL / SIGSTOP excepted: nothing in the process can answer those).
_HANDLED = ("SIGTERM", "SIGINT", "SIGHUP", "SIGQUIT", "SIGUSR1", "SIGUSR2", "SIGTSTP", "SIGTTIN", "SIGTTOU", "SIGALRM",
            "SIGVTALRM", "SIGPROF", "SIGXCPU", "SIGXFSZ", "SIGIO", "SIGPWR", "SIGPIPE")


class _Signals:
    """``raising``: the next signal raises :class:`_Stop` (and clears the flag itself, so only ONE signal ever
    raises); otherwise a signal is only written down in ``seen``."""

    def __init__(self) -> None:
        self.raising = False
        self.seen: Optional[int] = None

    def handle(self, signum: int, _frame: Any) -> None:
        self.seen = signum
        if self.raising:
            self.raising = False
            raise _Stop(signum)


@contextlib.contextmanager
def _signals(enabled: bool) -> Iterator[_Signals]:
    """:data:`_HANDLED` go to a :class:`_Signals` while the block runs (main thread only). A signal the caller
    IGNORES stays ignored (SIGHUP under ``nohup``)."""
    sig = _Signals()
    if not enabled or threading.current_thread() is not threading.main_thread():
        yield sig
        return
    old: Dict[int, Any] = {}
    for name in _HANDLED:
        s = getattr(signal, name, None)
        if s is not None and signal.getsignal(s) is not signal.SIG_IGN:
            old[s] = signal.signal(s, sig.handle)
    try:
        yield sig
    finally:
        for s, h in old.items():
            signal.signal(s, h)


# ---------------------------------------------------------------------------------------------------------------------
# the run


def preflight(tree: Tree, arms: Dict[str, List[int]], log: Optional[Path], marker: Path) -> List[int]:
    """Every refusal a run can find without changing anything (``run --check`` runs exactly these); arm A's list."""
    original = sorted(arms["A"])
    masks = tree.masks()
    odd = sorted(t for t, m in masks.items() if m is not None and m != frozenset(original))
    if not masks or odd:
        raise Refusal(f"the tree's threads are not all on arm A ({core_split.format_cpus(original)}): "
                      f"{len(odd)} of {len(masks)} differ (first tid {odd[:1]}). Is a narrowing still in place "
                      "(`python -m utils.core_split show`)?")
    for arm, cpus in arms.items():
        if not cpus or not set(cpus) <= set(original):
            raise Refusal(f"arm {arm} ({core_split.format_cpus(cpus)}) is not inside the tree's own list")
    if log is None or len(read_tables(log, max(0, _size(log) - _TAIL_BYTES))) < BEFORE_TABLES:
        raise Refusal(f"the run's log ({log}) holds fewer than {BEFORE_TABLES} logger tables: start after the third "
                      "logged iteration")
    if marker.exists():
        raise Refusal(f"an A/B marker exists ({marker}): another A/B is running, or one died. "
                      "`python -m main.ops.core_split_ab restore` clears it")
    there = next(d for d in marker.parents if d.exists())       # the marker's directory, or what it would be made in
    if not os.access(there, os.W_OK | os.X_OK):
        raise Refusal(f"the marker's directory ({there}) is not writable: a narrowing could not be written down")
    if tree.open_workers() == 0:
        raise Refusal("the tree has no env worker threads (is the run still starting?)")
    return original


def run_ab(tree: Tree, arms: Dict[str, List[int]], out_dir: Path, *, log: Optional[Path], evals: Optional[Path] = None,
           timing: Timing = Timing(), rotation: Sequence[str] = ROTATION, marker: Optional[Path] = None,
           clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
           handle_signals: bool = True, say: Callable[[str], None] = lambda s: None) -> Dict[str, Any]:
    """Run the A/B on ``tree`` and return the record (also written to ``out_dir/record.json``). ``arms``: the CPU
    list per arm, ``arms["A"]`` being the tree's own. Raises :class:`Refusal` before anything is changed."""
    marker = core_split.ab_marker_path() if marker is None else marker
    original = preflight(tree, arms, log, marker)
    rec: Dict[str, Any] = {"schema": SCHEMA, "launcher": tree.launcher, "launcher_key": tree.key, "arms": arms,
                           "rotation": list(rotation), "timing": asdict(timing), "collect_share": COLLECT_SHARE,
                           "log": str(log), "out_dir": str(out_dir), "started_wall": time.time(),
                           "env_pid": tree.env_pid, "env_workers": tree.n_workers, "switches": [], "segments": [],
                           "rollouts": [], "aborted": None, "restored": None}
    body = {"schema": SCHEMA, "launcher": tree.launcher, "launcher_key": tree.key,
            "original": core_split.format_cpus(original), "driver_pid": os.getpid(), "driver_key": _own_key(),
            "out_dir": str(out_dir), "started_wall": rec["started_wall"]}

    def marker_body() -> Dict[str, Any]:
        return {**body, "touched": {str(pid): key for pid, key in sorted(tree.touched.items())}}

    aborted: Optional[str] = None
    stop: Optional[_Stop] = None
    error: Optional[Exception] = None
    rows: List[List[float]] = []
    window: Optional[Tuple[int, int]] = None        # the log's size and the eval file's rows at rollout 1's start
    held = "A"
    t0 = clock()

    def switch(arm: str, now: float) -> Tuple[float, bool]:
        """Put the tree on ``arm``; ``(when it was done, verified on every thread)``."""
        nonlocal held
        if arm == held:
            return now, True
        held = arm                                  # before the call: a signal inside it still finds a restore due
        ok = tree.apply(arms[arm], original, strays=arm == "A")
        done = clock() - t0
        rec["switches"].append({"t": round(now, 4), "done": round(done, 4), "arm": arm, "settled": ok})
        return done, ok

    def close_segment(now: float) -> None:
        """The open segment ends NOW at the latest (never at a time planned before a rollout ended early), its
        arm's list is read back from every thread before anything is switched, and the marker names every process
        found on a narrow list."""
        s = rec["segments"][-1]
        procs = tree.procs()
        if s["arm"] != "A":                         # a child born on the narrow list since the switch inherited it
            tree.note(procs)
        found = tree.masks(procs=procs)
        s["t_to"] = min(s["t_to"], round(now, 4))
        s["on_arm"] = bool(found) and all(m is None or m == frozenset(arms[s["arm"]]) for m in found.values())

    with _signals(handle_signals) as sig:
        try:
            _create_marker(marker, marker_body())
        except BaseException:
            tree.close()
            raise
        tree.on_touch = lambda: _write_json(marker, marker_body())
        try:
            try:
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_dir / "burst.jsonl", "w", encoding="utf-8") as f:
                    sig.raising = True
                    if sig.seen is not None:            # a signal that came before this line was only written down
                        sig.raising = False
                        raise _Stop(sig.seen)
                    f.write(json.dumps({"header": 1, "kind": "burst", "launcher": tree.launcher, "trainer": 0,
                                        "period": timing.period_s, "threads": {"env_workers": tree.n_workers},
                                        "affinity": original, "wall_start": rec["started_wall"]}) + "\n")
                    k = 0                               # rollouts measured so far
                    phase = "between"                   # between -> confirm -> rollout
                    last_r = 0.0                        # the last reading with a runnable worker (the start counts)
                    quiet = prev_t = 0.0                # no runnable worker in any reading since `quiet`, none late
                    start = 0.0
                    bursts = pos = 0
                    prev_w = 1
                    due = t0
                    check_at = 0.0
                    while k < len(rotation):
                        now = clock() - t0
                        if now >= timing.max_s:
                            aborted = f"{k} of {len(rotation)} rollouts after {timing.max_s:.0f} s"
                            break
                        if now >= check_at:
                            if not tree.alive():
                                aborted = "the launcher is gone"
                                break
                            check_at = now + 1.0
                        w = tree.runnable()
                        if w < 0:
                            aborted = "an env worker is gone (the env process restarted?)"
                            break
                        rows.append([round(now, 4), w])
                        f.write(json.dumps([round(now, 4), w, 0, 0]) + "\n")
                        rising = w >= 1 and prev_w == 0
                        if w >= 1 or now - prev_t > timing.end_idle_s / 3:
                            quiet = now                 # a late reading proves nothing about the time it skipped
                        if phase == "between":
                            if w >= 1 and now - last_r >= timing.idle_s:
                                phase, start, bursts = "confirm", now, 1
                        elif phase == "confirm":
                            bursts += 1 if rising else 0
                            if now - start >= timing.confirm_s:
                                if bursts >= timing.confirm_bursts:
                                    if window is None:
                                        window = (_size(log), _rows(evals))
                                    phase, pos = "rollout", 0
                                    done, ok = switch(rotation[k][0], now)
                                    rec["segments"].append({"rollout": k, "pos": 0, "arm": rotation[k][0], "settled": ok,
                                                            "t_from": round(done + timing.settle_s, 4),
                                                            "t_to": round(start + timing.segment_s, 4)})
                                else:
                                    phase = "between"   # a blip inside an update
                        elif phase == "rollout":
                            over = w == 0 and now - quiet >= timing.end_idle_s
                            if over or now >= start + (pos + 1) * timing.segment_s:
                                close_segment(now)
                                pos += 1
                                if over or pos >= len(rotation[k]):
                                    _, ok = switch("A", now)
                                    rec["rollouts"].append({"rollout": k, "order": rotation[k], "start": round(start, 4),
                                                            "end": round(now, 4), "segments": pos,
                                                            "ended_by": "idle" if over else "segments"})
                                    k, phase = k + 1, "between"
                                    if not ok:
                                        aborted = "arm A could not be verified between two rollouts"
                                        break
                                else:
                                    done, ok = switch(rotation[k][pos], now)
                                    rec["segments"].append({"rollout": k, "pos": pos, "arm": rotation[k][pos],
                                                            "settled": ok, "t_from": round(done + timing.settle_s, 4),
                                                            "t_to": round(start + (pos + 1) * timing.segment_s, 4)})
                        if w >= 1:
                            last_r = now
                        prev_w, prev_t = w, now
                        due = max(due + timing.period_s, clock())
                        sleep(max(0.0, due - clock()))
                    sig.raising = False                 # from here a signal is written down; the restore is next
            except Exception as e:                      # restore first (the finally), then let it propagate
                sig.raising = False
                error, aborted = e, f"{type(e).__name__}: {e}"
        except _Stop as e:
            stop, aborted = e, f"signal {e.signum}"
        finally:
            detail: Dict[str, Any] = {"verified": False}
            tries = 0
            while tries < _RESTORE_TRIES:           # a signal's handler clears `raising` before it raises, so the
                try:                                # pass after one cannot be interrupted
                    sig.raising = False
                    detail = restore_tree(tree, original)
                except _Stop as e:
                    stop, aborted = stop or e, aborted or f"signal {e.signum}"
                    continue
                except Exception as e:              # a failed restore is reported, never raised past the marker
                    detail = {"verified": False, "error": f"{type(e).__name__}: {e}"}
                tries += 1
                if detail["verified"]:
                    break
            tree.close()
            restored = bool(detail["verified"])
            rec["restored"], rec["restore"], rec["aborted"] = restored, detail, aborted
            if restored:
                with contextlib.suppress(OSError):
                    marker.unlink()
            else:
                with contextlib.suppress(Exception):
                    _write_json(marker, marker_body())
            with contextlib.suppress(OSError):
                _write_json(out_dir / "record.json", rec)
            with contextlib.suppress(OSError, ValueError):      # a dead terminal must not lose what follows
                say(f"restored {core_split.format_cpus(original)} on the tree of launcher {tree.launcher}: "
                    + (f"verified on {detail.get('threads', 0)} threads"
                       + ("" if detail.get("launcher_alive") else " (the launcher is gone)") if restored
                       else f"NOT VERIFIED ({detail}); run `python -m main.ops.core_split_ab restore` ({marker})"))
        # the affinity is back; what is left only reads. Signals stay caught: one ends the wait below, not the record.
        for s in rec["segments"]:
            s.update(segment_period(rows, s["t_from"], s["t_to"], timing.gap_s))
        n = len(rotation)
        control: Dict[str, Any] = {"ok": False, "why": "not read"}
        n_evals = 0
        if window is not None and log is not None:
            off0, rows0 = window
            if aborted is None:
                waited = clock()
                while True:                         # table 1 holds the update BEFORE rollout 1; tables 2..n+1 the A/B's
                    tables = read_tables(log, off0)
                    if len(tables) >= n + 1 or clock() - waited >= timing.tables_wait_s or sig.seen is not None:
                        break
                    sleep(1.0)
                before = read_tables(log, max(0, off0 - _TAIL_BYTES), off0)[-(BEFORE_TABLES - 1):] + tables[:1]
                control = control_of([t["update_wall_s"] for t in before],
                                     [t["update_wall_s"] for t in tables[1:n + 1]], n)
            # an eval prints its line when it ENDS: the window closes with the control's last table, not the last rollout
            off1, rows1 = _size(log), _rows(evals)
            n_evals = eval_lines(log, off0, off1) + max(0, rows1 - rows0)
            rec["window"] = {"log_from": off0, "log_to": off1, "eval_rows_from": rows0, "eval_rows_to": rows1}
        rec.update(control=control, evals_in_window=n_evals,
                   **decide(rec["segments"], rollouts=n, control=control, evals=n_evals, aborted=aborted))
        with contextlib.suppress(OSError):
            _write_json(out_dir / "record.json", rec)
        if stop is None and sig.seen is not None:
            stop = _Stop(sig.seen)
    if stop is not None:
        raise SystemExit(128 + stop.signum)
    if error is not None:
        raise error
    return rec


def render(rec: Dict[str, Any]) -> str:
    arms = rec.get("arms", {})
    lines = [f"core split A/B on launcher {rec.get('launcher')}: bar 3 % (PASS <= {100 * PASS_MAX:.1f} %, FAIL >= "
             f"{100 * FAIL_MIN:.1f} %, INCONCLUSIVE counts as a fail)"]
    verdicts = rec.get("verdicts") or {}
    for arm, drop in ARMS[1:]:
        v = verdicts.get(arm, {})
        cores = len(arms.get(arm, [])) // 2
        head = f"  arm {arm} ({cores} trainer cores, {core_split.format_cpus(arms.get(arm, []))}): {v.get('verdict', '?')}"
        if "loss" in v:
            head += (f"  projected loss {100 * v['loss']:+.2f} % +- {100 * v['loss_se']:.2f} % "
                     f"(period ratio {v['ratio_mean']:.4f} over {len(v['ratios'])} rollouts)")
        lines.append(head)
        lines.extend(f"      {w}" for w in v.get("why", []))
    c = rec.get("control", {})
    lines.append("  control (update_wall_s, during vs before): "
                 + (f"{100 * c['moved']:+.1f} %" if c.get("moved") is not None else str(c.get("why", "not read"))))
    adopt = rec.get("adopt_arm", "A")
    cores = len(arms.get(adopt, [])) // 2
    name = f"{cores}+{core_split.PHYSICAL_CORES - cores}"
    if adopt == "A":
        lines.append(f"  RESULT: no narrower arm passed. Keep the split as it is ({name}).")
    else:
        lines.append(f"  RESULT: adopt {name} (arm {adopt}, the narrowest that passed). Record this, then: "
                     f"`python -m utils.core_split set {name} --next-launch --note \"A/B {rec.get('out_dir', '')}\"`")
        wider = [a for a, _ in ARMS[1:] if a < adopt and verdicts.get(a, {}).get("verdict") != "PASS"]
        if wider:
            lines.append(f"  NOTE: the wider arm {', '.join(wider)} did not pass while {adopt} did: read the record "
                         "before adopting.")
    lines.append(f"  restored: {rec.get('restored')}" + (f"   ABORTED: {rec['aborted']}" if rec.get("aborted") else ""))
    if not rec.get("restored"):
        lines.append("  THE TREE MAY STILL BE NARROWED: run `python -m main.ops.core_split_ab restore`, then "
                     "`python -m utils.core_split show`")
    return "\n".join(lines)


def _marker_tree(launcher: int, key: str, touched: Dict[int, str], proc: str) -> Tree:
    return Tree(launcher, proc=proc, key=key, touched=touched)


def restore_from_marker(marker: Optional[Path] = None, *, proc: str = "/proc",
                        make_tree: Callable[[int, str, Dict[int, str], str], Tree] = _marker_tree) -> Tuple[int, str]:
    """Undo a narrowing whose driver died without restoring (SIGKILL, OOM). ``(exit code, message)``. The marker
    names the launcher AND every process the driver narrowed, so a tree whose launcher died is restored too."""
    marker = core_split.ab_marker_path() if marker is None else marker
    try:
        m = json.loads(marker.read_text(encoding="utf-8"))
    except (FileNotFoundError, NotADirectoryError):
        return EXIT_OK, f"no A/B marker ({marker}): nothing to restore"
    except (OSError, ValueError) as e:
        m = {"unreadable": str(e)}
    try:
        if not isinstance(m, dict) or "unreadable" in m:
            raise ValueError(str(m))
        launcher, key = int(m["launcher"]), str(m["launcher_key"])
        original = core_split.parse_cpus(str(m["original"]))
        touched = {int(pid): str(k) for pid, k in (m.get("touched") or {}).items()}
        driver_key = str(m.get("driver_key") or "")
        driver = int(driver_key.split(":")[0]) if driver_key else 0
    except (KeyError, ValueError, TypeError, AttributeError, core_split.CoreSplitError) as e:
        return EXIT_REFUSED, f"the marker {marker} is unreadable ({e}): restore by hand, then delete it"
    st = procfs.read_stat(driver, proc) if driver else None
    if driver != os.getpid() and st is not None and st.key == driver_key and st.state not in ("Z", "X"):
        return EXIT_REFUSED, (f"the A/B driver (pid {driver}) is still running and restores on its own exit: send it "
                              "SIGTERM and wait. If it does not exit, SIGKILL it, then run this again (the marker "
                              "is never cleared under a live driver)")
    detail = restore_tree(make_tree(launcher, key, touched, proc), original)
    if not detail["verified"]:
        return EXIT_UNRESTORED, (f"could NOT verify {m['original']} on every thread of launcher {launcher}'s tree "
                                 f"({detail}); marker kept")
    marker.unlink()
    if not detail["threads"]:
        return EXIT_OK, (f"launcher {launcher} is gone and no process the driver narrowed is left: nothing to "
                         "restore; marker removed")
    return EXIT_OK, (f"restored {m['original']} on {detail['threads']} threads of launcher {launcher}'s tree"
                     + ("" if detail["launcher_alive"] else " (the launcher itself is gone)") + "; marker removed")


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m main.ops.core_split_ab", description=(__doc__ or "").split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run the A/B on a launcher's tree (narrows it inside its own CPUs; restores)")
    r.add_argument("--launcher-pid", type=int, required=True)
    r.add_argument("--out-dir", type=Path, required=True, help="burst.jsonl and record.json land here")
    r.add_argument("--log", type=Path, default=None, help="the run's log (default: <run dir>/launcher_child.full.log)")
    r.add_argument("--max-s", type=float, default=Timing.max_s)
    r.add_argument("--check", action="store_true", help="run every refusal check and print the plan; change nothing")
    sub.add_parser("restore", help="undo a narrowing whose driver was killed (reads the marker)")
    v = sub.add_parser("verdict", help="re-derive and print the verdict of a record.json")
    v.add_argument("record", type=Path)
    return ap


def main(argv: Optional[Sequence[str]] = None, *, make_tree: Callable[[int], Tree] = Tree,
         marker: Optional[Path] = None) -> int:
    args = _parser().parse_args(argv)
    if args.cmd == "restore":
        code, msg = restore_from_marker(marker)
        print(msg)
        return code
    if args.cmd == "verdict":
        rec = json.loads(args.record.read_text(encoding="utf-8"))
        if not isinstance(rec, dict) or rec.get("schema") != SCHEMA:
            print(f"core_split_ab: REFUSED: {args.record} is not a record of schema {SCHEMA!r}", file=sys.stderr)
            return EXIT_REFUSED
        rec.update(decide(rec["segments"], rollouts=len(rec["rotation"]), control=rec.get("control") or {},
                          evals=rec.get("evals_in_window", 0), aborted=rec.get("aborted")))
        print(render(rec))
        return EXIT_OK
    try:
        tree = make_tree(args.launcher_pid)
        control = sorted({c for m in tree.masks().values() if m for c in m})
        if control != list(core_split.DEFAULT.trainer_ids):
            raise Refusal(f"the tree runs on {core_split.format_cpus(control)}; the A/B is defined for a trainer on the "
                          f"{core_split.DEFAULT.name} split's list ({core_split.DEFAULT.trainer_list})")
        arms = {arm: core_split.narrowed(control, drop) for arm, drop in ARMS}
        run_dir = tree.run_dir()
        log = args.log or (run_dir / "launcher_child.full.log" if run_dir else None)
        evals = run_dir / "eval_results.jsonl" if run_dir else None
        if args.check:
            try:
                preflight(tree, arms, log, core_split.ab_marker_path() if marker is None else marker)
                n = len(read_tables(log, max(0, _size(log) - _TAIL_BYTES))) if log else 0
                print(f"launcher {args.launcher_pid}: {len(tree.masks())} threads, {tree.n_workers} train-env workers "
                      f"(env process {tree.env_pid}), on {core_split.format_cpus(control)}; log {log} ({n} tables in "
                      "its tail); arms " + ", ".join(f"{a}={core_split.format_cpus(c)}" for a, c in arms.items())
                      + ". Nothing was changed.")
            finally:
                tree.close()
            return EXIT_OK
        rec = run_ab(tree, arms, args.out_dir, log=log, evals=evals, timing=Timing(max_s=args.max_s), marker=marker,
                     say=lambda s: print(s, flush=True))
    except Refusal as e:
        print(f"core_split_ab: REFUSED: {e}", file=sys.stderr)
        return EXIT_REFUSED
    print(render(rec))
    return EXIT_OK if rec["restored"] else EXIT_UNRESTORED


if __name__ == "__main__":
    sys.exit(main())
