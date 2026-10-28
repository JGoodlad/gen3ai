"""LR-F look 1, AM-LRF-L1 item 8: the REPORTED mechanism instruments (STAGED_PLAN §4.1.2), never a decision.

Per seed, for the LR-F finals and (for scale) the same seeds' C0 finals, from each run's OWN TensorBoard rows
(``main.x35.tbrows``: the inherited parent file excluded, steps <= the fork dropped) and its ``snapshots/``:

* the LR path: the share of ``train/learning_rate`` rows at or above the floor 1.7361e-4 (LR-F's predicted >= 95 %),
  the min / median / max rate;
* the mean ``train/approx_kl`` and ``train/clip_fraction`` over the fork's rows (predicted [0.025, 0.045] and
  [0.20, 0.30]);
* N_prom and the final pool's median age, from ``snapshots/`` as SPRT look 1 read them (predicted: N_prom within
  +-2 of C0's, pooled).

A step logged twice (a restart's replay) keeps the later segment's value, as ``batch2_void`` does. Read-only on
``models/``. Prints one JSON document.

    python lrf_read/mechanism.py > lrf_read/mechanism.json
"""
from __future__ import annotations

import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

from main.x35 import tbrows as T

MODELS = Path("/home/goodlad/dev/gen3ai/models")
FORK = 280_000_170
FLOOR = 1.7361e-4
SEEDS = ("3501", "3502", "3503")
RUNS = {"lrfloor": "rb_x35_lrfloor_s{}", "c0": "rb_x35_c0_s{}"}
TAGS = ("train/learning_rate", "train/approx_kl", "train/clip_fraction")
PREDICTED = {"lr_at_floor_share_min": 0.95, "approx_kl_mean": [0.025, 0.045], "clip_fraction_mean": [0.20, 0.30],
             "n_prom_pooled_within": 2}


def _rows(run_dir: str) -> Dict[str, List[Tuple[int, float]]]:
    out: Dict[str, Dict[int, float]] = {t: {} for t in TAGS}
    for seg in T.load_segments(run_dir, TAGS, after_step=FORK):   # ordered by first wall time: later wins
        for t in TAGS:
            for step, v in seg.rows.get(t, ()):
                out[t][int(step)] = float(v)
    return {t: sorted(d.items()) for t, d in out.items()}


def _final_step(run_dir: Path) -> int:
    from main.x35.kit import _final_step as kit_final_step   # the kit's reader (the final's recorded step)

    return kit_final_step(str(run_dir / "final_model.zip"))


def _snapshots(run_dir: Path, final_step: int) -> Dict[str, Any]:
    steps = sorted(int(re.search(r"(\d+)", p.name).group(1)) for p in (run_dir / "snapshots").glob("snapshot_*.zip"))
    return {"n_snapshots": len(steps), "n_prom": sum(s > FORK for s in steps),
            "median_age_m": statistics.median((final_step - s) / 1e6 for s in steps)}


def seed_row(run: str) -> Dict[str, Any]:
    rd = MODELS / run
    r = _rows(str(rd))
    lr = [v for _, v in r["train/learning_rate"]]
    kl = [v for _, v in r["train/approx_kl"]]
    cf = [v for _, v in r["train/clip_fraction"]]
    fs = _final_step(rd)
    return {"run": run, "final_step": fs, "updates": len(lr),
            "lr_at_floor_share": (sum(v >= FLOOR * (1 - 1e-6) for v in lr) / len(lr)) if lr else None,
            "lr_min": min(lr) if lr else None, "lr_median": statistics.median(lr) if lr else None,
            "lr_max": max(lr) if lr else None,
            "approx_kl_mean": statistics.fmean(kl) if kl else None,
            "clip_fraction_mean": statistics.fmean(cf) if cf else None,
            **_snapshots(rd, fs)}


def main() -> int:
    out: Dict[str, Any] = {"schema": "x35_lrf_mechanism_v1", "reported_only": True, "fork_step": FORK,
                           "floor": FLOOR, "predicted": PREDICTED, "arms": {}}
    for arm, pat in RUNS.items():
        out["arms"][arm] = {s: seed_row(pat.format(s)) for s in SEEDS}
    for arm in RUNS:
        rows = out["arms"][arm].values()
        out["arms"][arm]["pooled"] = {
            "n_prom": sum(r["n_prom"] for r in rows),
            "approx_kl_mean_of_seeds": statistics.fmean(r["approx_kl_mean"] for r in rows),
            "clip_fraction_mean_of_seeds": statistics.fmean(r["clip_fraction_mean"] for r in rows),
            "lr_at_floor_share_min": min(r["lr_at_floor_share"] for r in rows)}
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
