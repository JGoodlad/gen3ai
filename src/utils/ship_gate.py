"""THE SHIP GATE — the routine gate, in the shape the box can afford right now, as ONE command.

``scripts/ops/ship_gate.sh`` (this module) is what an agent runs before ``/gen3ai-ship``. It RUNS
EVERYTHING the routine gate runs (``-m "not slow and not e2e"`` over every collected root, ``pytest.ini``'s ``testpaths``:
``src/``, ``tools/`` and the measurement records' own tests; the static gates included) — no test is chosen by the change (owner, 2026-09-30: no change-aware selection framework).
What it decides is only the SHAPE, from the box:

* a training run is LIVE (``utils.procfs.live_training_runs``) ⇒ rule 5b: ``nice -n 19``, pinned to
  the agent cores (every online cpu the trainer is NOT pinned to), ``-n`` = ``LIVE_WORKERS_PER_CPU`` x
  those cpus. A live run that is AGENT WORK is not a training run here (``core_split.split_runs``, the one rule the
  heavy slots and the gate lock's warning read too: its whole tree sits inside the agent list, or it was started with
  CUDA hidden): an agent's CPU smoke on the agent cores once read as a trainer "pinned to EVERY cpu" and refused the
  gate for 25 minutes;
* a GPU LEASE is held (no run live: a measurement battery owns the card and its CPU timing) ⇒ the SAME rule-5b shape
  on the active split's agent cores (``utils.core_split``). The predicate is ``utils.heavy_slots.wanted`` - the ONE definition of "rule 5b applies"
  the heavy slots use too, so the two cannot drift;
* no run is live and no lease is held ⇒ the quiet policy: ``-n`` = :data:`QUIET_WORKERS`, normal priority, no pin.

The session WARMS the Rust binaries the tests resolve lazily before its first test (``$GEN3AI_SHIP_WARM=1``,
the root conftest's ``pytest_sessionstart`` -> ``utils.ship_gate_warm``: a cargo build paid inside a test body
is that test's cost to the bar; one gate-lock slot covers both).

Always through ``scripts/ops/gate_lock.sh`` (one gate at a time) and ``scripts/ops/mem_cap.sh`` (a cap measured from the gate's peaks and
scaled by its ``-n``: :func:`mem_cap_gb`), and always with ``CUDA_VISIBLE_DEVICES=""`` (standing rule 5: a gate is never a GPU lease). Always with THE
8 s BAR on (``$GEN3AI_SHIP_BAR=1``, ``utils.ship_bar``): a routine test over the bar the re-tier set is
named, and one far over it (>= 2x the contention-stretched bar) FAILS the gate; every test at or over
half the bar is written to :data:`BAR_REPORT` for calibration.

Why it can be ~15 min beside a live run: the expensive tests are tiered ``slow`` by their MEASURED
cost (designs/ops/testing.md "The SHIP GATE"), and a slow-tier session told to bank (``GEN3AI_SLOW_STATUS_BANK=1``)
records its verdicts in ``designs/ops/slow_tier_status.json`` — a recorded red there FAILS this gate. A slow test that broke
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
An UN-BOOTSTRAPPED checkout (no Showdown ``dist/`` / ``node_modules``: a fresh worktree) is refused BEFORE any queue, exit 4, naming
the command (``scripts/bootstrap.sh --skip-env --no-rust --no-check`` in a worktree): ``utils.checkout_deps``, also the pytest
session's guard. ``--dry-run`` does not check.
Exit status: the gate's own; 89 = ``--max-run-s`` stopped it; 2 = usage; 4 = not bootstrapped.
"""
from __future__ import annotations

import configparser
import math
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

from utils import checkout_deps
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
#: THE GATE'S MEMORY CAP (``scripts/ops/mem_cap.sh <GB>``) is MEASURED, not a habit (2026-11-17). The cap is what the heavy slots
#: ADMIT by (``utils.heavy_admission``: the caps of the running jobs sum to <= 61.7 GiB), so a cap far above the gate's use
#: keeps a slot from the next job for nothing: the gate declared 24 and peaked at 6.6. The peaks are the ``capped`` rows
#: of ``~/.claude/jobs/heavy_slots/events.jsonl`` (``kind: capped``, ``peak_gb``), joined to the gate's own slot rows:
#:   -n 3 (rule 5b, the shape beside a live run): 64 gates, 2026-11-09..17, peak min 5.1, median 7.0, p90 8.4, MAX 10.18;
#:   -n 6 (the quiet shape): MAX 13.57 over 66 ``pytest`` jobs capped at 24 that took no slot (a quiet gate takes none) -
#:   UNVERIFIED that every one of them was a gate, so it is taken as the bound it is: none was higher.
#: The cap is ``CAP_MARGIN`` x the line through those two maxima at the gate's ``-n`` (xdist workers are what the memory
#: scales with: each imports and collects the whole suite), rounded UP to a whole GB: 16 at -n 3, 21 at -n 6. The cap still
#: kills an overrun (exit 86, "peak X of cap Y"), so a gate that outgrows it fails LOUDLY: re-derive from the ``capped`` rows
#: when the peak passes two thirds of it (``mem_cap.sh --status`` shows each job's peak).
PEAK_AT_3_GB = 10.2
PEAK_AT_6_GB = 13.6
CAP_MARGIN = 1.5
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
    agent_runs: str = ""        # the live runs that are agent work (``core_split.split_runs``), named; they shape nothing

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


def resolve_shape(runs: Optional[Dict[str, int]] = None, *, online: Optional[Sequence[int]] = None,
                  trainer_cpus: Optional[Sequence[int]] = None, proc: str = "/proc",
                  affinity: Callable[[int], Optional[List[int]]] = affinity_of) -> Shape:
    """The gate's shape for the box as it is now. The arguments are for tests (``runs`` given: taken as training
    runs, unclassified)."""
    aside = ""
    if runs is None:                          # the box as it is: agent work is not a training run (core_split.split_runs)
        runs, agent_runs = core_split.split_runs(procfs.live_training_runs(proc), proc=proc, affinity=affinity)
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


def mem_cap_gb(workers: int) -> int:
    """The gate's declared memory cap in whole GB for ``workers`` xdist workers (see :data:`PEAK_AT_3_GB`): ``CAP_MARGIN`` x
    the measured peak at that many workers, rounded up."""
    slope = (PEAK_AT_6_GB - PEAK_AT_3_GB) / 3.0
    return max(1, math.ceil(CAP_MARGIN * (PEAK_AT_3_GB + slope * (max(1, workers) - 3))))


def effective_workers(shape_workers: int, extra: Sequence[str] = ()) -> int:
    """The ``-n`` the gate really runs: the LAST one in ``extra`` (pytest takes the last; ``-- -n 6`` overrides the shape's),
    else the shape's. ``-n auto`` / ``logical`` / a count that is not a number read as every logical cpu this process may use."""
    args = list(extra)
    found: Optional[str] = None
    for i, a in enumerate(args):
        if a in ("-n", "--numprocesses") and i + 1 < len(args):
            found = args[i + 1]
        elif a.startswith("--numprocesses="):
            found = a.split("=", 1)[1]
        elif a.startswith("-n") and len(a) > 2 and not a.startswith("--"):
            found = a[2:]
    if found is None:
        return shape_workers
    return int(found) if found.isdigit() else max(1, len(os.sched_getaffinity(0)))


def collected_roots() -> List[str]:
    """The roots the gate collects: ``pytest.ini``'s ``testpaths``, the ONE list (``src/uncollected_tests_gate_test.py``
    fails on a test file outside it, so a root added there is gated here without an edit)."""
    cp = configparser.ConfigParser(interpolation=None)
    cp.read(repo_path("pytest.ini"))
    roots = cp.get("pytest", "testpaths", fallback="src").split()   # a checkout with no pytest.ini: src/ alone
    return [r.rstrip("/") + "/" for r in roots]


def build_argv(shape: Shape, python: str, extra: Sequence[str] = (), *, max_run_s: Optional[float] = None) -> List[str]:
    """``max_run_s``: mem_cap's run-time bound, on the INNERMOST wrapper so that neither queue (gate lock, heavy slot) counts."""
    bound = [] if max_run_s is None else ["--max-run-s", f"{max_run_s:g}"]
    pre: List[str] = []
    if shape.nice:
        pre += ["nice", "-n", str(shape.nice)]
    if shape.cpus is not None:
        pre += ["taskset", "-c", cpu_list(shape.cpus)]
    return pre + [str(repo_path("scripts", "ops", "gate_lock.sh")),
                  str(repo_path("scripts", "ops", "mem_cap.sh")), *bound,
                  str(mem_cap_gb(effective_workers(shape.workers, extra))),
                  python, "-m", "pytest", *collected_roots(), "-m", MARKERS, "-q", "-n", str(shape.workers), *extra]


#: pytest's UsageError code, which the same refusal returns when pytest itself starts (``conftest.py``).
EXIT_UNBOOTSTRAPPED = 4


def preflight() -> str:
    """``""`` when this checkout is bootstrapped, else the refusal naming the exact bootstrap command. Runs BEFORE the gate
    lock and the heavy slot (a never-bootstrapped worktree used to queue 20-40 min and only then fail at pytest's start):
    the SAME definition as the pytest session's guard (``utils.checkout_deps``), file-existence only, milliseconds.
    It never runs bootstrap (that can build Rust)."""
    refused = checkout_deps.refusal(str(repo_path()))
    return f"[ship_gate] REFUSING before queueing: this checkout is not bootstrapped.\n{refused}" if refused else ""


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
    if not dry:
        refused = preflight()
        if refused:
            print(refused, file=sys.stderr)
            return EXIT_UNBOOTSTRAPPED
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
