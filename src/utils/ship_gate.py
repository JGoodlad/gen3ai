"""THE SHIP GATE — the routine gate, in the shape the box can afford right now, as ONE command.

``scripts/ops/ship_gate.sh`` (this module) is what an agent runs before ``/gen3ai-ship``. It RUNS
EVERYTHING the routine gate runs (``-m "not slow and not e2e"`` over all of ``src/``, the static gates
included) — no test is chosen by the change (owner, 2026-09-30: no change-aware selection framework).
What it decides is only the SHAPE, from the box:

* a training run is LIVE (``utils.procfs.live_training_runs``) ⇒ rule 5b: ``nice -n 19``, pinned to
  the agent cores (every online cpu the trainer is NOT pinned to), ``-n`` = ``LIVE_WORKERS_PER_CPU`` x
  those cpus. A live run that is AGENT WORK is not a training run here (:func:`split_runs`: its whole tree sits
  inside the agent list, the core-split owner's ``is_agent_work``, or it was started with CUDA hidden): an agent's
  CPU smoke on the agent cores once read as a trainer "pinned to EVERY cpu" and refused the gate for 25 minutes;
* a GPU LEASE is held (no run live: a measurement battery owns the card and its CPU timing) ⇒ the SAME rule-5b shape
  on the active split's agent cores (``utils.core_split``). The predicate is ``utils.heavy_slots.wanted`` - the ONE definition of "rule 5b applies"
  the heavy slots use too, so the two cannot drift;
* no run is live and no lease is held ⇒ the quiet policy: ``-n`` = :data:`QUIET_WORKERS`, normal priority, no pin.

The session WARMS the Rust binaries the tests resolve lazily before its first test (``$GEN3AI_SHIP_WARM=1``,
the root conftest's ``pytest_sessionstart`` -> ``utils.ship_gate_warm``: a cargo build paid inside a test body
is that test's cost to the bar; one gate-lock slot covers both).

Always through ``scripts/ops/gate_lock.sh`` (one gate at a time) and ``scripts/ops/mem_cap.sh``, and always with ``CUDA_VISIBLE_DEVICES=""`` (standing rule 5: a gate is never a GPU lease). Always with THE
8 s BAR on (``$GEN3AI_SHIP_BAR=1``, ``utils.ship_bar``): a routine test over the bar the re-tier set is
named, and one far over it (>= 2x the contention-stretched bar) FAILS the gate; every test at or over
half the bar is written to :data:`BAR_REPORT` for calibration.

Why it can be ~15 min beside a live run: the expensive tests are tiered ``slow`` by their MEASURED
cost (designs/ops/testing.md "The SHIP GATE"), and the slow tier banks its verdicts in
``designs/ops/slow_tier_status.json`` — a recorded red there FAILS this gate. A slow test that broke
since its last banked run still reads green here; only running the slow tier closes that.

USAGE
    scripts/ops/ship_gate.sh run [--max-run-s S] [-- <extra pytest args>]     # run the gate (queues on the gate lock)
    scripts/ops/ship_gate.sh --dry-run [--max-run-s S] [-- <extra pytest args>]  # print the resolved command, run nothing
    scripts/ops/ship_gate.sh --help

NEVER wrap ``ship_gate.sh run`` in ``timeout``: the gate QUEUES (the gate lock, then a heavy slot), an outer timeout counts
that wait, and firing there it kills the gate and loses its place (2026-11-09). To bound the gate, pass ``--max-run-s S``:
it goes to ``mem_cap.sh`` (``utils.mem_cap``, THE RUN-TIME BOUND), whose clock starts when pytest starts, after every
queue; past it the gate is stopped and exits 89.

No argument REFUSES (exit 2), like every scripts/ops/ entry point: a gate is never started by accident.
Exit status: the gate's own; 89 = ``--max-run-s`` stopped it; 2 = usage.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from utils import core_split
from utils import heavy_slots as HS
from utils import mem_cap
from utils import procfs
from utils.gate_lock_status import affinity_of, cpu_list
from utils.paths import repo_path

#: The marker expression — THE routine gate's, verbatim. The ship gate never narrows it.
MARKERS = "not slow and not e2e"
#: xdist workers with no live run (designs/ops/testing.md "Routine-gate wall time": -n 6).
QUIET_WORKERS = 6
#: xdist workers per LOGICAL agent cpu beside a live run — MEASURED, designs/ops/testing.md "The SHIP GATE".
LIVE_WORKERS_PER_CPU = 0.5
#: The memory cap (GB) the gate runs under (scripts/ops/mem_cap.sh).
MEM_CAP_GB = 24
#: Where the last ship gate's 8 s-bar rows land (``utils.ship_bar``; per user, like the test-cost table).
BAR_REPORT = os.path.join(os.path.expanduser("~"), ".cache", "gen3ai", "ship_bar_last.json")


def _names(runs: Dict[str, int]) -> str:
    return ", ".join(os.path.basename(rd.rstrip("/")) for rd in sorted(runs))


@dataclass(frozen=True)
class Shape:
    live_runs: Dict[str, int]
    cpus: Optional[List[int]]   # None = no pin
    nice: int
    workers: int
    leased: str = ""            # why rule 5b applies with no run live (a held GPU lease), else ""
    agent_runs: str = ""        # the live runs that are agent work (:func:`split_runs`), named; they shape nothing

    def describe(self) -> str:
        aside = f" (agent work, not a training run: {self.agent_runs})" if self.agent_runs else ""
        if not self.live_runs and self.leased:
            return (f"{self.leased}: rule 5b — nice {self.nice}, cpus {cpu_list(self.cpus or [])}, "
                    f"-n {self.workers}{aside}")
        if not self.live_runs:
            return (f"no training run is live and no GPU lease is held: -n {self.workers}, normal priority, "
                    f"no cpu pin{aside}")
        return (f"a training run is LIVE ({_names(self.live_runs)}): rule 5b — nice {self.nice}, cpus "
                f"{cpu_list(self.cpus or [])}, -n {self.workers}{aside}")


#: ``CUDA_VISIBLE_DEVICES`` values that hide every GPU from a process.
_CUDA_HIDDEN = ("", "-1")


def _cuda_hidden(pid: int, proc: str) -> bool:
    """Was ``pid`` STARTED with every GPU hidden (root ``CLAUDE.md`` rule 5: an agent without a lease)? Read from the
    environment it was started with; unset or unreadable is not hidden."""
    for kv in procfs.read_text(pid, "environ", proc).split("\0"):
        name, eq, value = kv.partition("=")
        if eq and name == "CUDA_VISIBLE_DEVICES":
            return value in _CUDA_HIDDEN
    return False


def _tasks(pid: int, proc: str) -> List[int]:
    """The thread ids of ``pid`` (an affinity is per thread); the pid alone when they cannot be listed."""
    try:
        return sorted(int(t) for t in os.listdir(f"{proc}/{pid}/task") if t.isdigit()) or [pid]
    except OSError:
        return [pid]


def _gone(pid: int, task: int, proc: str) -> bool:
    """A thread or process that exited after it was listed (its affinity then cannot be read)."""
    return not procfs.pid_alive(pid, proc) if task == pid else not os.path.exists(f"{proc}/{pid}/task/{task}")


def split_runs(runs: Dict[str, int], *, proc: str = "/proc",
               affinity: Callable[[int], Optional[List[int]]] = affinity_of) -> Tuple[Dict[str, int], Dict[str, int]]:
    """``(training, agent_work)``: the live runs that shape the gate, and the ones that do not. A run is AGENT WORK
    when its trainer was started with CUDA hidden, or when its WHOLE TREE (the launcher above the trainer, the
    trainer, every descendant, every thread of each) sits inside the active split's agent list. The rule is the
    core-split owner's (``core_split.is_agent_work``); only the tree walk is here. A process whose affinity cannot be
    read keeps the run a training run, unless it has just exited (never the trainer itself)."""
    procs = procfs.snapshot(proc)
    kids = procfs.children_map(procs)
    training: Dict[str, int] = {}
    agent: Dict[str, int] = {}
    for rd, pid in runs.items():
        if _cuda_hidden(pid, proc):
            agent[rd] = pid
            continue
        above = [a for a in procfs.ancestors(pid, procs)
                 if core_split.is_trainer_argv([x for x in procfs.read_text(a, "cmdline", proc).split("\0") if x])]
        inside = True
        for p in [*above, pid, *procfs.descendants(pid, kids)]:
            for task in _tasks(p, proc):
                aff = affinity(task)
                if aff is None and task != pid and _gone(p, task, proc):
                    continue
                inside = inside and core_split.is_agent_work(aff)
        (agent if inside else training)[rd] = pid
    return training, agent


def resolve_shape(runs: Optional[Dict[str, int]] = None, *, online: Optional[Sequence[int]] = None,
                  trainer_cpus: Optional[Sequence[int]] = None, proc: str = "/proc",
                  affinity: Callable[[int], Optional[List[int]]] = affinity_of) -> Shape:
    """The gate's shape for the box as it is now. The arguments are for tests (``runs`` given: taken as training
    runs, unclassified)."""
    aside = ""
    if runs is None:                          # the box as it is: agent work is not a training run (split_runs)
        runs, agent_runs = split_runs(procfs.live_training_runs(proc), proc=proc, affinity=affinity)
        aside = _names(agent_runs)
    why = HS.wanted(runs=runs)               # THE predicate of rule 5b (a live run OR a held GPU lease)
    if why is None:
        return Shape({}, None, 0, QUIET_WORKERS, agent_runs=aside)
    online = list(range(os.cpu_count() or 1)) if online is None else list(online)
    if not runs:                              # a lease and no run: no trainer pin to take the complement of
        agent = [c for c in core_split.agent_cpu_ids() if c in online] or online
        return Shape({}, agent, 19, max(1, int(len(agent) * LIVE_WORKERS_PER_CPU)), leased=why, agent_runs=aside)
    if trainer_cpus is None:
        trainer_cpus = [c for pid in runs.values() for c in (affinity(pid) or [])]
    agent = core_split.agent_ids_beside(trainer_cpus, online)
    if not agent or set(trainer_cpus) >= set(online):
        raise SystemExit("REFUSING: a training run is live but pinned to EVERY cpu, so there are no agent "
                         "cores for rule 5b (designs/ops/TRAINING_RUN_SOP.md). Ask the orchestrator.")
    return Shape(dict(runs), agent, 19, max(1, int(len(agent) * LIVE_WORKERS_PER_CPU)), agent_runs=aside)


def build_argv(shape: Shape, python: str, extra: Sequence[str] = (), *, max_run_s: Optional[float] = None) -> List[str]:
    """``max_run_s``: mem_cap's run-time bound, on the INNERMOST wrapper so that neither queue (gate lock, heavy slot) counts."""
    bound = [] if max_run_s is None else ["--max-run-s", f"{max_run_s:g}"]
    pre: List[str] = []
    if shape.nice:
        pre += ["nice", "-n", str(shape.nice)]
    if shape.cpus is not None:
        pre += ["taskset", "-c", cpu_list(shape.cpus)]
    return pre + [str(repo_path("scripts", "ops", "gate_lock.sh")),
                  str(repo_path("scripts", "ops", "mem_cap.sh")), *bound, str(MEM_CAP_GB),
                  python, "-m", "pytest", "src/", "-m", MARKERS, "-q", "-n", str(shape.workers), *extra]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] in (["-h"], ["--help"]):
        print("ship_gate.sh — " + (__doc__ or ""))
        return 0
    head = args[: args.index("--")] if "--" in args else args
    extra = args[args.index("--") + 1:] if "--" in args else []
    max_run_s: Optional[float] = None
    if len(head) == 3 and head[1] == "--max-run-s":
        try:
            max_run_s = mem_cap.parse_max_run_s(head[2])
        except ValueError as e:
            print(f"REFUSING: --max-run-s: {e}", file=sys.stderr)
            return 2
        head = head[:1]
    if head not in (["run"], ["--dry-run"]):
        print(f"REFUSING: ship_gate.sh needs `run` or `--dry-run`, then optionally `--max-run-s S` (got {head}); pytest "
              "options go after --. See --help.", file=sys.stderr)
        return 2
    dry = head == ["--dry-run"]
    shape = resolve_shape()
    cmd = build_argv(shape, os.environ.get("GEN3AI_PYTHON") or sys.executable, extra, max_run_s=max_run_s)
    print(f"[ship_gate] {shape.describe()}", file=sys.stderr)
    print(f"[ship_gate] CUDA_VISIBLE_DEVICES='' {shlex.join(cmd)}", file=sys.stderr)
    if dry:
        return 0
    os.makedirs(os.path.dirname(BAR_REPORT), exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", GEN3AI_SHIP_BAR="1", GEN3AI_SHIP_BAR_REPORT=BAR_REPORT,
               GEN3AI_SHIP_WARM="1")
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(repo_path()), env=env)
    if rc == mem_cap.EXIT_RUN_TIMEOUT and max_run_s is not None:
        print(f"[ship_gate] the gate was STOPPED by --max-run-s {max_run_s:g} (counted from pytest's start, no queue wait): "
              "its verdict is INCOMPLETE, never a red", file=sys.stderr)
    print(f"[ship_gate] exit {rc} after {(time.time() - t0) / 60:.1f} min wall (queue wait included; "
          "gate_lock's own line gives the held time)", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
