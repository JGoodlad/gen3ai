"""Paired ratios for the fork-clone A/B (`run_pairs.sh` rows): per pair, AFTER / BEFORE of the
per-boundary MEDIAN clone cost, the median over pairs and a paired bootstrap 95 % CI on that median
(20,000 resamples, seed 0) — the statistic of the M2 record.

    python analyze.py rows.jsonl
"""
import json
import random
import statistics
import sys


def boot(xs, n=20000, seed=0):
    rng = random.Random(seed)
    meds = sorted(statistics.median(rng.choices(xs, k=len(xs))) for _ in range(n))
    return meds[int(0.025 * n)], meds[int(0.975 * n)]


def main(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    by = {}
    for r in rows:
        by.setdefault(r["pair"], {})[r["arm"]] = r
    pairs = sorted(p for p, a in by.items() if "before" in a and "after" in a)
    loads = [r["load1_start"] for r in rows] + [r["load1_end"] for r in rows]
    print(f"{len(pairs)} complete pairs; load1 {min(loads):.1f}-{max(loads):.1f}")

    def med(r, key):
        return r["result"][key]["median"]

    arms = {
        "before snapshot (whole compacted session)": [med(by[p]["before"], "snapshot_ns") for p in pairs],
        "before script+seeds alone": [med(by[p]["before"], "script_seeds_ns") for p in pairs],
        "after engine clone": [med(by[p]["after"], "engine_ns") for p in pairs],
        "after resume (the fork a version drives)": [med(by[p]["after"], "resume_ns") for p in pairs],
        "after snapshot (whole session, same code)": [med(by[p]["after"], "snapshot_ns") for p in pairs],
    }
    for k, v in arms.items():
        print(f"  {k:45s} median of per-boundary medians: {statistics.median(v):8.0f} ns")
    b_snap = arms["before snapshot (whole compacted session)"]
    for name, num in (("resume / before-snapshot", arms["after resume (the fork a version drives)"]),
                      ("engine / before-snapshot", arms["after engine clone"])):
        ratios = [a / b for a, b in zip(num, b_snap)]
        lo, hi = boot(ratios)
        print(f"  {name:30s} {statistics.median(ratios):.3f} [{lo:.3f}, {hi:.3f}]  "
              f"(> 1 in {sum(r > 1 for r in ratios)}/{len(ratios)})")
    share = [s / t for s, t in zip(arms["before script+seeds alone"], b_snap)]
    lo, hi = boot(share)
    print(f"  script+seeds share of the before fork       {statistics.median(share):.3f} [{lo:.3f}, {hi:.3f}]")
    r0 = by[pairs[0]]["before"]["result"]
    print(f"  boundaries per run {r0['boundaries']}, mean script length {r0['mean_script_len']}, "
          f"mean seed-anchor length {r0['mean_seed_len']}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "rows.jsonl")
