"""Per-pair ratios vs arm A (B/A = the production build's cost of this change; C/A = the self-check
build's), their median and a paired bootstrap 95% CI on that median (20,000 resamples, fixed
seed) — the M1/M2 records' statistic.

    python analyze.py replay_rows.jsonl
"""

from __future__ import annotations

import json
import random
import statistics
import sys


def boot_median_ci(xs, n=20000, seed=12345):
    rng = random.Random(seed)
    meds = sorted(statistics.median(rng.choices(xs, k=len(xs))) for _ in range(n))
    return meds[int(0.025 * n)], meds[int(0.975 * n) - 1]


def main(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    pairs = {}
    for r in rows:
        pairs.setdefault(r["pair"], {})[r["arm"]] = r
    full = {k: v for k, v in sorted(pairs.items()) if {"A", "B", "C"} <= v.keys()}
    loads = [r[k] for r in rows for k in ("load1_start", "load1_end")]
    print(f"{path}: {len(full)} complete triples, load1 {min(loads):.1f}–{max(loads):.1f}")
    for metric in ("replay_cpu_s", "replay_wall_s"):
        for arm in ("B", "C"):
            ratios = [v[arm][metric] / v["A"][metric] for v in full.values()]
            lo, hi = boot_median_ci(ratios)
            slower = sum(x > 1 for x in ratios)
            med_a = statistics.median(v["A"][metric] for v in full.values())
            med_x = statistics.median(v[arm][metric] for v in full.values())
            print(f"  {metric:14s} {arm}/A median {statistics.median(ratios):.4f}  95% CI [{lo:.4f}, {hi:.4f}]"
                  f"  {arm} slower in {slower}/{len(ratios)}  (median A {med_a*1000:.1f} ms, {arm} {med_x*1000:.1f} ms)")


if __name__ == "__main__":
    main(sys.argv[1])
