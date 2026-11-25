"""X72's entropy floor, read from a run's TensorBoard at one MARK (PREREG §7). Read-only: safe on a live run.

    python -m main.x72.floor read --round K --mark {5M,10M,15M,end} [--seed 7341 ...] [--models-dir D] [--out F]

**The statistic** [§7]: the mean of a run's last 20 points of -``train/entropy_loss`` at or before the mark, read
through ``main.ops.run_ref.scalar_events`` (the run's OWN points: a fork's inherited prefix is cut at the lineage
record's fork step), an abandoned attempt's points dropped as CE-3d's ``diag9_ce3d.py`` drops them (:func:`clean`,
restated from that frozen file and pinned to it by test). It is CE-3d's end-entropy definition evaluated at a step,
and :func:`statistic` at a banked swap fork's last step reproduces ``diag9_ce3d.json``'s number.

**The marks** [§7]: round start + 5M, + 10M, + 15M (round start = the previous round's target, 300,000,170 in round
1) and the round's END. The END's step is the run's own: the first point at or past the round's target, which is
where the launch stops (the loop ends at the first rollout that reaches the step budget).

**The floor** [§7]: 0.5 x the lowest banked swap end entropy, READ from CE-3d's ``record/diag9_ce3d.json`` and never
retyped. It FIRES iff a statistic is below it by more than 1e-9 nats; a value within 1e-9 fires nothing and is
recorded. The comparison is ``x72_rule.py floor``'s, restated and pinned to the frozen file by test; the output
also carries the exact ``x72_rule.py floor <e1> [<e2> <e3>]`` command, so the registered CLI can be run on the same
numbers.

**Which of two rows at one step is kept, and why.** A launch's LAST step carries TWO rows of every ``train/*`` tag
by design (``designs/training/telemetry_scalars.md``, ``gen3_final_update_dump_v1``): update k-1's, then update
k's, one update's wall time apart. :func:`clean` keeps the LATER row in wall time and drops the earlier one (a
point is dropped when a later-written point has a step at or below its own), because that is what the registered
``diag9_ce3d.py`` does and the floor's own reference numbers were computed that way. So at a launch's last step
the statistic holds update k's entropy and not update k-1's; every earlier step holds one row and is unaffected.

**When a mark is READY** (a statistic is a function of the mark, never of when it is read):

* a +5M / +10M / +15M mark: once a point STRICTLY past the mark exists. Rows are written in step order, so every
  row at or before the mark is then on disk.
* the END: once the round's last step holds its two rows (the final update's row is on disk). Read it after the
  launch has exited; the two-row check only catches a reader run early on a normal end.

Before that the lineage is NOT-READY and nothing is compared. A run resumed from a checkpoint BELOW a mark after
the mark was read can change that mark's statistic (the abandoned points are replaced): the output counts the
dropped points (``superseded``) so that case is visible.

Exit codes: 0 every requested lineage read, none below the floor · 10 FIRED · 5 NOT-READY (and none fired) ·
3 refused (fewer than 20 points, an unreadable run, an unknown fork boundary) · 2 usage.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, NamedTuple, Optional, Sequence, Tuple

from main.x72 import finals as F
from utils.paths import repo_path

TAG = "train/entropy_loss"
#: the statistic's points [§7: "the mean of a run's last 20 points"]
POINTS = 20
#: mark -> steps past the round's start; the END has no offset (its step is the run's own) [§7]
OFFSETS = {"5M": 5_000_000, "10M": 10_000_000, "15M": 15_000_000}
END = "end"
MARKS = (*OFFSETS, END)
#: the floor is this fraction of the lowest banked swap end entropy [§7; x72_rule.FLOOR_FRACTION]
FLOOR_FRACTION = 0.5
#: rule 8's band around the floor [§7; x72_rule.ENT_EPS_NATS]
ENT_EPS_NATS = 1e-9
#: CE-3d's banked record, the only source of the swap end entropies [§7]
CE3D_RECORD = "designs/research_state/measurements/ce3d_magnet_screen_2026-11-11/record/diag9_ce3d.json"
RULE_FILE = "designs/research_state/measurements/x72_swap_plateau_2026-11-25/x72_rule.py"
#: a normal end's last step holds this many rows of a ``train/*`` tag (gen3_final_update_dump_v1)
END_ROWS = 2

READ, NOT_READY = "READ", "NOT-READY"
FIRED, CLEAR = "FIRED", "NOT-FIRED"
EXIT_OK, EXIT_USAGE, EXIT_REFUSED, EXIT_NOT_READY, EXIT_FIRED = 0, 2, 3, 5, 10


class FloorError(ValueError):
    """The read is refused: nothing is compared."""


class Point(NamedTuple):
    step: int
    wall_time: float
    value: float


def floor_nats() -> float:
    """The floor, from CE-3d's banked record (never retyped)."""
    path = repo_path(CE3D_RECORD)
    try:
        ends = json.loads(path.read_text(encoding="utf-8"))["summary"]["swap_end_entropy"]
        vals = [float(v) for v in ends.values()]
    except (OSError, ValueError, KeyError, AttributeError, TypeError) as exc:
        raise FloorError(f"{path}: no summary.swap_end_entropy to take the floor from ({exc!r})") from exc
    if len(vals) != len(F.SEEDS) or not all(math.isfinite(v) and v > 0.0 for v in vals):
        raise FloorError(f"{path}: summary.swap_end_entropy is not three positive numbers ({ends})")
    return FLOOR_FRACTION * min(vals)


def mark_step(k: int, mark: str) -> Optional[int]:
    """The step of a +5M / +10M / +15M mark of round ``k``; None for the END (its step is read from the run)."""
    F.check_round(k, lo=1)
    if mark not in MARKS:
        raise FloorError(f"mark {mark!r} is not one of {list(MARKS)}")
    return None if mark == END else F.target(k - 1) + OFFSETS[mark]


def clean(evs: Iterable[Any]) -> Tuple[List[Any], int]:
    """``diag9_ce3d.py``'s ``clean``: by wall time, a point drops every earlier-written point whose step is at or
    past its own. Returns (points, number dropped). Of two rows at one step the LATER-written survives."""
    out: List[Any] = []
    dropped = 0
    for e in sorted(evs, key=lambda e: e.wall_time):
        while out and out[-1].step >= e.step:
            out.pop()
            dropped += 1
        out.append(e)
    return out, dropped


def raw_points(run_dir: Path) -> List[Point]:
    """The run's own ``train/entropy_loss`` rows, as written (nothing dropped)."""
    from main.ops.run_ref import scalar_events

    try:
        got = scalar_events(Path(run_dir), [TAG])
    except SystemExit as exc:           # an unknown fork boundary: scalar_events refuses rather than guess one
        raise FloorError(f"{run_dir}: the run's own points cannot be told from its parent's (the fork boundary "
                         f"is not confirmed by the lineage record)") from exc
    return [Point(int(e.step), float(e.wall_time), float(e.value)) for e in got.get(TAG, [])]


def statistic(pts: Sequence[Any], step: int) -> Dict[str, Any]:
    """The mean of the last 20 points of -entropy_loss at or before ``step`` (``pts`` cleaned, in step order)."""
    at = [p for p in pts if p.step <= step]
    if len(at) < POINTS:
        raise FloorError(f"{len(at)} own point(s) at or before step {step:,}: the statistic is the mean of the last "
                         f"{POINTS}")
    last = [-float(p.value) for p in at[-POINTS:]]
    mean = statistics.mean(last)
    if not math.isfinite(mean) or mean < 0.0:
        raise FloorError(f"the last {POINTS} points at or before step {step:,} do not average to a finite "
                         f"entropy >= 0 ({mean!r})")
    return {"stat_nats": mean, "points": POINTS, "first_step": at[-POINTS].step, "last_step": at[-1].step,
            "min_nats": min(last), "max_nats": max(last)}


def read_run(run_dir: Path, k: int, mark: str) -> Dict[str, Any]:
    """One run at one mark: ``{"status": READ, "stat_nats", ...}`` or ``{"status": NOT-READY, "reason"}``."""
    want = mark_step(k, mark)
    if not Path(run_dir).is_dir():
        return {"status": NOT_READY, "reason": f"no run dir {run_dir}"}
    raw = raw_points(Path(run_dir))
    pts, dropped = clean(raw)
    base: Dict[str, Any] = {"own_points": len(pts), "superseded": dropped,
                            "last_own_step": pts[-1].step if pts else None}
    if want is None:
        lo = F.target(k)
        ends = [p.step for p in pts if p.step >= lo]
        if not ends:
            return {"status": NOT_READY, **base,
                    "reason": f"no point at or past the round's target {lo:,}: round {k} has not ended"}
        step = ends[0]
        rows = sum(1 for p in raw if p.step == step)
        if rows < END_ROWS:
            return {"status": NOT_READY, **base, "mark_step": step,
                    "reason": f"step {step:,} holds {rows} row(s); a launch's last step holds {END_ROWS} "
                              f"(the final update's row is not on disk, or the launch did not end here)"}
        base["rows_at_mark_step"] = rows
    else:
        step = want
        if not pts or pts[-1].step <= step:
            return {"status": NOT_READY, **base, "mark_step": step,
                    "reason": f"no point past step {step:,} yet: the run has not passed the mark"}
    return {"status": READ, "mark_step": step, **statistic(pts, step), **base}


def compare(stats: Sequence[Tuple[int, float]]) -> Dict[str, Any]:
    """``x72_rule.floor`` on (lineage seed, statistic) pairs: which are below the floor by more than 1e-9, which
    are within 1e-9 of it (those fire nothing and are recorded)."""
    bar = floor_nats()
    near = [seed for seed, x in stats if abs(x - bar) <= ENT_EPS_NATS]
    below = [seed for seed, x in stats if x < bar - ENT_EPS_NATS]
    return {"fired": bool(below), "below_floor": below, "within_1e-9_of_the_floor": near, "floor_nats": bar}


def read(k: int, mark: str, *, seeds: Optional[Sequence[int]] = None, models_dir: Optional[Path] = None,
         now: Optional[_dt.datetime] = None) -> Dict[str, Any]:
    """The floor at one mark of round ``k`` for the named lineages (default: all three), in lineage order."""
    mark_step(k, mark)
    picked = [s for s in F.SEEDS if s in set(seeds)] if seeds else list(F.SEEDS)
    unknown = sorted(set(seeds or ()) - set(F.SEEDS))
    if unknown:
        raise FloorError(f"seed(s) {unknown} are not X72 lineages ({list(F.SEEDS)})")
    models = F._models(models_dir)
    lineages = []
    for seed in picked:
        run = F.run_name(seed)
        lineages.append({"seed": seed, "run": run, **read_run(models / run, k, mark)})
    stats = [(x["seed"], x["stat_nats"]) for x in lineages if x["status"] == READ]
    cmp_ = compare(stats)
    waiting = [x["seed"] for x in lineages if x["status"] != READ]
    outcome = FIRED if cmp_["fired"] else NOT_READY if waiting else CLEAR
    stamp = (now or _dt.datetime.now(_dt.timezone.utc)).isoformat(timespec="seconds")
    return {"registration": F.PREREG, "round": k, "mark": mark, "outcome": outcome, **cmp_,
            "read": [s for s, _ in stats], "not_ready": waiting, "lineages": lineages,
            "rule_command": (f"python {RULE_FILE} floor " + " ".join(repr(x) for _, x in stats)) if stats else None,
            "read_at": stamp}


def exit_code(rec: Dict[str, Any]) -> int:
    return {FIRED: EXIT_FIRED, NOT_READY: EXIT_NOT_READY, CLEAR: EXIT_OK}[rec["outcome"]]


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m main.x72.floor", description=(__doc__ or "").split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("read", help="the floor statistic of each lineage at one mark, compared against the floor")
    p.add_argument("--round", type=int, required=True)
    p.add_argument("--mark", required=True, choices=MARKS)
    p.add_argument("--seed", type=int, action="append", default=None,
                   help="a lineage to read (repeatable; default: all three)")
    p.add_argument("--models-dir", default=None, help="default: the run archive (models/)")
    p.add_argument("--out", default=None, help="also write the JSON record here (replaced atomically)")
    a = ap.parse_args(argv)
    try:
        rec = read(a.round, a.mark, seeds=a.seed, models_dir=Path(a.models_dir) if a.models_dir else None)
    except (FloorError, F.FinalsError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return EXIT_REFUSED
    text = json.dumps(rec, indent=1)
    if a.out:
        out = Path(a.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + f".tmp{os.getpid()}")
        tmp.write_text(text + "\n", encoding="utf-8")
        os.replace(tmp, out)
    print(text)
    return exit_code(rec)


if __name__ == "__main__":
    sys.exit(main())
