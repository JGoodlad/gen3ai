"""The routine gate's xdist SCHEDULE: whole test FILES, the most EXPENSIVE first (``gen3_cost_ordered_loadfile_v1``).

WHY (measured 2026-09-30, the routine gate at ``-n 8``, xdist 3.8.0's default ``--dist load``): the
wall was **437 s for 1,782 s of test time** — one worker was busy 426 s while the other seven sat idle
after ~200 s. ``load`` hands each worker a CONTIGUOUS first chunk of a quarter of the collection split
n ways (~390 tests at ``-n 8``) and never takes it back, and the collection's ``agents/training/rust_*``
stretch (the Lane E / H / rollout parity gates, ~400 s together) landed in ONE worker's chunk. More
workers only made the chunks smaller around the same pole.

THE SCHEDULE. A work unit is one test FILE (``--dist loadfile``: module- and class-scoped fixtures —
the Lane H gate's three ~40-70 s eval runs — are built once, never once per worker that happens to
draw one of their tests), and the units are handed out longest-first (LPT, a 4/3-approximation of the
best makespan), pulled one at a time by whichever worker runs dry. A unit's cost is the sum of its
tests' last recorded durations (:func:`load_costs`); a test with no record costs
:data:`DEFAULT_COST_S`, so a NEW file sorts by its size and is learned on its first run.

THE COST TABLE is per USER, not per checkout — ``~/.cache/gen3ai/test_costs.json``
(``$GEN3AI_TEST_COSTS`` overrides) — keyed by the repo-relative node id, so every worktree on the box
shares it and a fresh worktree is ordered from its first gate. Every session merges what it measured
(:func:`record_costs`: flock + atomic replace; entries unseen for :data:`PRUNE_DAYS` drop out). It
only ever ORDERS work: a stale, missing or corrupt table changes the wall time, never which tests run
or what they assert — the collection is untouched, so every xdist worker still collects the identical
list.

Opt out with ``GEN3AI_COST_SCHEDULE=0`` (plain ``loadfile``), or pick another mode with ``--dist``.
"""
from __future__ import annotations

import fcntl
import json
import os
import tempfile
import time
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

COSTS_ENV = "GEN3AI_TEST_COSTS"
SCHEDULE_ENV = "GEN3AI_COST_SCHEDULE"
DEFAULT_COST_S = 0.05            # a test with no record: ~the routine gate's median test
PRUNE_DAYS = 60
SCHEMA = "gen3_test_costs_v1"


def costs_path() -> Path:
    raw = os.environ.get(COSTS_ENV)
    if raw:
        return Path(raw)
    return Path.home() / ".cache" / "gen3ai" / "test_costs.json"


#: What :func:`load_costs` could not read, for the session summary (the root conftest prints these).
NOTES: List[str] = []


def load_costs(path: Optional[Path] = None) -> Dict[str, float]:
    """``{nodeid: seconds}`` — EMPTY on a missing or unreadable table, with a note in :data:`NOTES`.

    The table is never load-bearing: it decides the ORDER files are handed out in, so a missing one
    costs wall time (plain collection order) and nothing else."""
    p = path or costs_path()
    try:
        doc = json.loads(p.read_text())
        if doc.get("schema") != SCHEMA:
            raise ValueError(f"schema {doc.get('schema')!r} is not {SCHEMA!r}")
        return {k: float(v[0]) for k, v in doc["costs"].items()}
    except FileNotFoundError:
        NOTES.append(f"test-cost table {p} not found: files scheduled in collection order (speed only); "
                     "this session's durations will create it")
    except (OSError, ValueError, KeyError, TypeError, IndexError, AttributeError) as exc:
        NOTES.append(f"test-cost table {p} UNREADABLE ({exc!r}): IGNORED, files scheduled in collection "
                     "order (speed only); this session's durations rewrite it")
    return {}


def record_costs(measured: Mapping[str, float], path: Optional[Path] = None,
                 now: Optional[float] = None) -> Path:
    """Merge ``measured`` (``{nodeid: seconds}``) into the table under an exclusive lock, atomically."""
    p = path or costs_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    day = int((time.time() if now is None else now) // 86400)
    with open(str(p) + ".lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            doc = json.loads(p.read_text())
            costs = doc["costs"] if doc.get("schema") == SCHEMA else {}
        except (OSError, ValueError, KeyError, TypeError):
            costs = {}
        costs = {k: v for k, v in costs.items()
                 if isinstance(v, list) and len(v) == 2 and day - int(v[1]) <= PRUNE_DAYS}
        for k, s in measured.items():
            costs[k] = [round(float(s), 4), day]
        fd, tmp = tempfile.mkstemp(dir=str(p.parent), prefix=p.name + ".")
        with os.fdopen(fd, "w") as f:
            json.dump({"schema": SCHEMA, "costs": costs}, f, separators=(",", ":"), sort_keys=True)
        os.replace(tmp, p)
    return p


def unit_cost(nodeids: Iterable[str], costs: Mapping[str, float]) -> float:
    return sum(costs.get(n, DEFAULT_COST_S) for n in nodeids)


def order_units(workqueue: "OrderedDict[str, dict]", costs: Mapping[str, float]) -> "OrderedDict[str, dict]":
    """The work units most-expensive first; ties keep collection order (``sorted`` is stable)."""
    ranked = sorted(workqueue.items(), key=lambda kv: -unit_cost(kv[1], costs))
    return OrderedDict(ranked)


def make_scheduler(config, log):
    """The cost-ordered ``loadfile`` scheduler (imported lazily: xdist is only needed under ``-n``)."""
    from xdist.scheduler import LoadFileScheduling

    class CostOrderedLoadFileScheduling(LoadFileScheduling):
        """``LoadFileScheduling`` whose work queue is re-ordered, ONCE, by recorded cost before the
        first unit is handed out. Everything else — the unit = file grouping, the pull-when-dry
        reschedule, crash handling — is xdist's own."""

        _ordered = False

        def _assign_work_unit(self, node):
            if not self._ordered:
                self._ordered = True
                self.workqueue = order_units(self.workqueue, load_costs())
            return super()._assign_work_unit(node)

        def remove_node(self, node):
            """xdist's ``LoadScopeScheduling.remove_node`` with ONE change: the test that crashed its
            worker is NOT handed out again. Upstream re-queues the crashed work unit with the
            crashing test still marked incomplete, so a restarted worker runs it AGAIN — a test that
            kills its worker (a hard crash, a SIGTERM'd worker) then loops until the restart budget
            is spent. ``--dist load`` reports it once and moves on; so does this. The rest of its
            file is re-queued as upstream does."""
            workload = self.assigned_work.pop(node)
            if not self._pending_of(workload):
                return None
            crashitem = next(nodeid for unit in workload.values()
                             for nodeid, done in unit.items() if not done)
            for unit in workload.values():
                if crashitem in unit:
                    unit[crashitem] = True          # reported as crashed by DSession; never re-run
            self.workqueue.update({scope: unit for scope, unit in workload.items()
                                   if self._pending_of({scope: unit})})
            for other in self.assigned_work:
                self._reschedule(other)
            return crashitem

    return CostOrderedLoadFileScheduling(config, log)
