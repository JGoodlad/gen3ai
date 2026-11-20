"""The card check's step-time meter (Q6-N PREREG §7, Q5-N PREREG §7): ONE formula for the pilot and for its reference.

    seconds per 1M steps = (median lifecycle/update_wall_s + median rollout/collect_ms / 1000) / median d(steps) * 1e6

over the run's TensorBoard rollout rows with LO <= step < HI (HI is before the pool seeds, so the rows are one regime).
rho = the pilot's figure / the reference's figure on the SAME window. The reference is BOUNDARY-1's cell-C pilot
(`rb_b1pinpilot_c_s9201`, B1 PREREG §13 item 4), the same argv as the banked cell at the same pin on the same card.

B1's own `rho.json` (527.67 s per 1M for C) came from a script that is not in the repo and this formula does not
reproduce it to the digit (518.27 on B1's rows), which is why rho is never formed against that number: both sides
of the ratio go through this file.

    PYTHONPATH=<checkout>/src CUDA_VISIBLE_DEVICES="" python pilot_rho.py <run_dir> [--lo 300000] [--hi 4000000]
    ... --reference <reference run_dir>      # also prints rho
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

UPDATE = "lifecycle/update_wall_s"
COLLECT = "rollout/collect_ms"


def step_time(run: Path, lo: int, hi: int) -> dict:
    acc = EventAccumulator(str(run / "tb"), size_guidance={"scalars": 0})
    acc.Reload()
    upd = {e.step: e.value for e in acc.Scalars(UPDATE)}
    col = {e.step: e.value for e in acc.Scalars(COLLECT)}
    every = sorted(set(upd) & set(col))
    steps = [s for s in every if lo <= s < hi]
    if len(steps) < 10:
        raise SystemExit(f"{run}: {len(steps)} rollout rows in [{lo}, {hi}); the meter needs at least 10")
    dsteps = [b - a for a, b in zip(every, every[1:]) if lo <= b < hi]
    u = statistics.median(upd[s] for s in steps)
    c = statistics.median(col[s] for s in steps) / 1000.0
    d = statistics.median(dsteps)
    return {"run": run.name, "rows": len(steps), "first_step": steps[0], "last_step": steps[-1],
            "median_update_s": round(u, 3), "median_collect_s": round(c, 3), "median_dsteps": d,
            "update_share": round(u / (u + c), 4), "sec_per_1M_steps": round((u + c) / d * 1e6, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run", type=Path)
    ap.add_argument("--lo", type=int, default=300_000)
    ap.add_argument("--hi", type=int, default=4_000_000)
    ap.add_argument("--reference", type=Path)
    a = ap.parse_args()
    out = {"window": [a.lo, a.hi], "pilot": step_time(a.run, a.lo, a.hi)}
    if a.reference:
        out["reference"] = step_time(a.reference, a.lo, a.hi)
        out["rho"] = round(out["pilot"]["sec_per_1M_steps"] / out["reference"]["sec_per_1M_steps"], 4)
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
