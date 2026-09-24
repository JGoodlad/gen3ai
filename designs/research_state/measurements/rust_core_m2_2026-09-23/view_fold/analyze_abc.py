"""Per-pair ratios of the three-arm replay bench: median + paired bootstrap 95 % CI on the median
(20,000 resamples, fixed seed) — the M1 record's `analyze.py` statistic, for C/A, C/B and B/A.

    python analyze_abc.py replay_abc_rows.jsonl
"""

import json
import random
import statistics
import sys


def boot(ratios, n=20000, seed=0):
    rng = random.Random(seed)
    meds = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(n))
    return meds[int(0.025 * n)], meds[int(0.975 * n)]


def main(path):
    rows = [json.loads(line) for line in open(path)]
    by = {}
    for r in rows:
        by.setdefault(r["pair"], {})[r["arm"]] = r
    pairs = sorted(p for p, d in by.items() if len(d) == 3)
    loads = [r["load1_start"] for r in rows] + [r["load1_end"] for r in rows]
    print(f"{len(pairs)} complete triples; load1 {min(loads):.1f}-{max(loads):.1f}")
    for arm in "ABC":
        v = [by[p][arm]["replay_cpu_s"] * 1000 for p in pairs]
        print(f"  {arm}: median replay CPU {statistics.median(v):.1f} ms")
    for num, den in (("C", "A"), ("C", "B"), ("B", "A")):
        for key in ("replay_cpu_s", "replay_wall_s"):
            rs = [by[p][num][key] / by[p][den][key] for p in pairs]
            lo, hi = boot(rs)
            above = sum(r > 1 for r in rs)
            print(f"  {num}/{den} {key:14s} median {statistics.median(rs):.4f}  95% CI [{lo:.4f}, {hi:.4f}]"
                  f"  {num} slower in {above}/{len(rs)}")


if __name__ == "__main__":
    main(sys.argv[1])
