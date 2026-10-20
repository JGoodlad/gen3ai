"""THE SHIP GATE — the routine gate, in the shape the box can afford right now, as ONE command.

``scripts/ops/ship_gate.sh`` (this module) is what an agent runs before ``/gen3ai-ship``. It RUNS
EVERYTHING the routine gate runs (``-m "not slow and not e2e"`` over all of ``src/``, the static gates
included) — no test is chosen by the change (owner, 2026-09-30: no change-aware selection framework).
What it decides is only the SHAPE, from the box:

* a training run is LIVE (``utils.procfs.live_training_runs``) ⇒ rule 5b: ``nice -n 19``, pinned to
  the agent cores (every online cpu the trainer is NOT pinned to), ``-n`` = ``LIVE_WORKERS_PER_CPU`` x
  those cpus;
* a GPU LEASE is held (no run live: a measurement battery owns the card and its CPU timing) ⇒ the SAME rule-5b shape
  on :data:`AGENT_CPUS`. The predicate is ``utils.heavy_slots.wanted`` - the ONE definition of "rule 5b applies"
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
    scripts/ops/ship_gate.sh run [-- <extra pytest args>]     # run the gate (queues on the gate lock)
    scripts/ops/ship_gate.sh --dry-run [-- <extra pytest args>]  # print the resolved command, run nothing
    scripts/ops/ship_gate.sh --help

No argument REFUSES (exit 2), like every scripts/ops/ entry point: a gate is never started by accident.
Exit status: the gate's own; 2 = usage.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from utils import heavy_slots as HS
from utils import procfs
from utils.gate_lock_status import affinity_of, cpu_list
from utils.paths import repo_path

#: The marker expression — THE routine gate's, verbatim. The ship gate never narrows it.
MARKERS = "not slow and not e2e"
#: xdist workers with no live run (designs/ops/testing.md "Routine-gate wall time": -n 6).
QUIET_WORKERS = 6
#: xdist workers per LOGICAL agent cpu beside a live run — MEASURED, designs/ops/testing.md "The SHIP GATE".
LIVE_WORKERS_PER_CPU = 0.5
#: The agent cores when rule 5b applies with NO trainer to read the complement from (a held GPU lease, no live run):
#: this box's, CPU n and n+8 being SMT siblings (CLAUDE.md rule 5b). Filtered to the online cpus.
AGENT_CPUS = (5, 6, 7, 13, 14, 15)
#: The memory cap (GB) the gate runs under (scripts/ops/mem_cap.sh).
MEM_CAP_GB = 24
#: Where the last ship gate's 8 s-bar rows land (``utils.ship_bar``; per user, like the test-cost table).
BAR_REPORT = os.path.join(os.path.expanduser("~"), ".cache", "gen3ai", "ship_bar_last.json")


@dataclass(frozen=True)
class Shape:
    live_runs: Dict[str, int]
    cpus: Optional[List[int]]   # None = no pin
    nice: int
    workers: int
    leased: str = ""            # why rule 5b applies with no run live (a held GPU lease), else ""

    def describe(self) -> str:
        if not self.live_runs and self.leased:
            return (f"{self.leased}: rule 5b — nice {self.nice}, cpus {cpu_list(self.cpus or [])}, "
                    f"-n {self.workers}")
        if not self.live_runs:
            return f"no training run is live and no GPU lease is held: -n {self.workers}, normal priority, no cpu pin"
        names = ", ".join(os.path.basename(rd.rstrip("/")) for rd in sorted(self.live_runs))
        return (f"a training run is LIVE ({names}): rule 5b — nice {self.nice}, cpus "
                f"{cpu_list(self.cpus or [])}, -n {self.workers}")


def resolve_shape(runs: Optional[Dict[str, int]] = None, *, online: Optional[Sequence[int]] = None,
                  trainer_cpus: Optional[Sequence[int]] = None) -> Shape:
    """The gate's shape for the box as it is now. The keyword arguments are for tests."""
    runs = procfs.live_training_runs() if runs is None else runs
    why = HS.wanted(runs=runs)               # THE predicate of rule 5b (a live run OR a held GPU lease)
    if why is None:
        return Shape({}, None, 0, QUIET_WORKERS)
    online = list(range(os.cpu_count() or 1)) if online is None else list(online)
    if not runs:                              # a lease and no run: no trainer pin to take the complement of
        agent = [c for c in AGENT_CPUS if c in online] or online
        return Shape({}, agent, 19, max(1, int(len(agent) * LIVE_WORKERS_PER_CPU)), leased=why)
    if trainer_cpus is None:
        trainer_cpus = [c for pid in runs.values() for c in (affinity_of(pid) or [])]
    agent = [c for c in online if c not in set(trainer_cpus)]
    if not agent or set(trainer_cpus) >= set(online):
        raise SystemExit("REFUSING: a training run is live but pinned to EVERY cpu, so there are no agent "
                         "cores for rule 5b (designs/ops/TRAINING_RUN_SOP.md). Ask the orchestrator.")
    return Shape(dict(runs), agent, 19, max(1, int(len(agent) * LIVE_WORKERS_PER_CPU)))


def build_argv(shape: Shape, python: str, extra: Sequence[str] = ()) -> List[str]:
    pre: List[str] = []
    if shape.nice:
        pre += ["nice", "-n", str(shape.nice)]
    if shape.cpus is not None:
        pre += ["taskset", "-c", cpu_list(shape.cpus)]
    return pre + [str(repo_path("scripts", "ops", "gate_lock.sh")),
                  str(repo_path("scripts", "ops", "mem_cap.sh")), str(MEM_CAP_GB),
                  python, "-m", "pytest", "src/", "-m", MARKERS, "-q", "-n", str(shape.workers), *extra]


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args[:1] in (["-h"], ["--help"]):
        print("ship_gate.sh — " + (__doc__ or ""))
        return 0
    head = args[: args.index("--")] if "--" in args else args
    extra = args[args.index("--") + 1:] if "--" in args else []
    if head not in (["run"], ["--dry-run"]):
        print(f"REFUSING: ship_gate.sh needs `run` or `--dry-run` (got {head}); pytest options go after --. "
              "See --help.", file=sys.stderr)
        return 2
    dry = head == ["--dry-run"]
    shape = resolve_shape()
    cmd = build_argv(shape, os.environ.get("GEN3AI_PYTHON") or sys.executable, extra)
    print(f"[ship_gate] {shape.describe()}", file=sys.stderr)
    print(f"[ship_gate] CUDA_VISIBLE_DEVICES='' {shlex.join(cmd)}", file=sys.stderr)
    if dry:
        return 0
    os.makedirs(os.path.dirname(BAR_REPORT), exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="", GEN3AI_SHIP_BAR="1", GEN3AI_SHIP_BAR_REPORT=BAR_REPORT,
               GEN3AI_SHIP_WARM="1")
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(repo_path()), env=env)
    print(f"[ship_gate] exit {rc} after {(time.time() - t0) / 60:.1f} min wall (queue wait included; "
          "gate_lock's own line gives the held time)", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
