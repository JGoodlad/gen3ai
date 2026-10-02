#!/usr/bin/env python3
"""D-19: an arm's `train/train_ms` and update-cycle wall over updates whose TB wall time is AFTER a cutoff
(contention windows excluded), C-1's eval-step drop and updates 1-2 dropped, as read_arm.py does; plus the
O9 eval wall delta (cycles straddling an eval step minus the median cycle). Bootstrap 95 % CI of the median.

    train_ms_window.py models/<run> [--after EPOCH_S] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from read_arm import scalars  # same TB reader


def boot_median(x, n=20000, seed=20260915):
    x = np.asarray(x, float)
    r = np.random.default_rng(seed)
    m = np.median(x[r.integers(0, len(x), (n, len(x)))], axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--after", type=float, default=0.0)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    s = scalars(Path(a.run))
    ev = sorted({st for st, _v, _w in s.get("eval/wall_sec", [])} | {st for st, _v, _w in s.get("eval/duration_sec", [])})
    tm = s["train/train_ms"]
    keep = [(st, v, w) for i, (st, v, w) in enumerate(tm) if i >= 2 and st not in ev and w > a.after]
    walls = [w for _st, _v, w in tm]
    cyc = np.diff(walls)
    ends = [tm[i + 1][0] for i in range(len(cyc))]
    # an update cycle "straddles" an eval step when the eval step lies in (prev update step, this step]
    straddle = [float(c) for c, (p, e) in zip(cyc, zip([t[0] for t in tm[:-1]], ends)) if any(p < x <= e for x in ev)]
    plain = [float(c) for c, (p, e), w in zip(cyc, zip([t[0] for t in tm[:-1]], ends), walls[1:])
             if not any(p < x <= e for x in ev) and w > a.after]
    xs = [v for _s, v, _w in keep]
    out = {"run": a.run, "after": a.after, "n": len(xs), "train_ms_median": float(np.median(xs)),
           "train_ms_median_ci95": boot_median(xs), "train_ms_q25_75": [float(np.percentile(xs, 25)), float(np.percentile(xs, 75))],
           "cycle_s_median_plain": float(np.median(plain)), "n_plain_cycles": len(plain),
           "eval_steps": ev, "eval_straddle_cycles_s": straddle,
           "eval_wall_delta_s": [c - float(np.median(plain)) for c in straddle],
           "arm_wall_s": float(walls[-1] - walls[0])}
    out["eval_share_of_wall"] = sum(out["eval_wall_delta_s"]) / out["arm_wall_s"]
    txt = json.dumps(out, indent=1)
    if a.json:
        Path(a.json).write_text(txt)
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
