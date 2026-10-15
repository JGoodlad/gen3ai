"""The bf16 A/B's SPEED read (REPORTED, never decided; registration §1) and the run-health read, from the 16 runs' own files.

    python train_ms_read.py <models-dir> <out.json>

* ``train_ms`` is the trainer's per-update wall time (TensorBoard scalar ``train_ms``, one point per PPO update; 153
  updates per run at 15M steps / 98,304 rows). Per arm: the median of the eight seeds' medians (the headline), the mean of
  the eight seeds' medians, the pooled median and mean, and the per-seed medians.
* the run-health read counts, per run, the ``Training complete`` lines (one per launcher-child leg: seeds 2001-2004 trained
  a 5M leg and then a same-pin resume to 15M) and the ``Traceback`` lines in ``launcher_child.full.log``.

Read-only on the models dir (standing rule 6). Needs ``tensorboard`` (the gen3ai_torch28 env has it).
"""
from __future__ import annotations

import glob
import json
import statistics
import sys
from pathlib import Path
from typing import Dict, List

SEEDS = range(2001, 2009)
ARMS = {"E": "rb_bf_e_s", "EB": "rb_bf_eb_s"}


def _train_ms(run: Path) -> List[float]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    vals: List[float] = []
    for f in sorted(glob.glob(str(run / "tb" / "events.out.tfevents.*"))):
        ea = EventAccumulator(f, size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags().get("scalars", []):
            if tag == "train_ms" or tag.endswith("/train_ms"):
                vals += [e.value for e in ea.Scalars(tag)]
    return vals


def main(models: str, out: str) -> int:
    root = Path(models)
    res: Dict[str, object] = {"unit": "ms per PPO update (TensorBoard train_ms)", "arms": {}, "health": {}}
    for arm, prefix in ARMS.items():
        per_seed = {}
        for s in SEEDS:
            run = root / f"{prefix}{s}"
            v = _train_ms(run)
            full = (run / "launcher_child.full.log").read_text(errors="replace")
            per_seed[str(s)] = {"n_updates": len(v), "median_ms": statistics.median(v), "mean_ms": statistics.mean(v),
                                "legs_complete": full.count("Training complete"), "tracebacks": full.count("Traceback")}
            per_seed[str(s)]["_values_ms"] = v
        meds = [per_seed[str(s)]["median_ms"] for s in SEEDS]
        pooled = [x for s in SEEDS for x in per_seed[str(s)]["_values_ms"]]
        res["arms"][arm] = {"median_of_seed_medians_ms": statistics.median(meds),
                            "mean_of_seed_medians_ms": statistics.mean(meds),
                            "pooled_median_ms": statistics.median(pooled), "pooled_mean_ms": statistics.mean(pooled),
                            "seed_median_range_ms": [min(meds), max(meds)], "per_seed": per_seed}
    e, eb = res["arms"]["E"], res["arms"]["EB"]
    res["ratio_EB_over_E_median_of_seed_medians"] = eb["median_of_seed_medians_ms"] / e["median_of_seed_medians_ms"]
    legs = sum(r["legs_complete"] for a in res["arms"].values() for r in a["per_seed"].values())
    tb = sum(r["tracebacks"] for a in res["arms"].values() for r in a["per_seed"].values())
    res["health"] = {"runs": 2 * len(SEEDS), "launcher_child_legs_complete": legs, "tracebacks": tb}
    Path(out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("ratio_EB_over_E_median_of_seed_medians", "health")}))
    print({a: (res["arms"][a]["median_of_seed_medians_ms"], res["arms"][a]["pooled_mean_ms"]) for a in ARMS})
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1], sys.argv[2]))
