"""Per-pair B/A ratios, their median, and a paired bootstrap 95% CI on that median.

    python analyze.py env_rows.jsonl thru_rows.jsonl

For a TIME metric (wall, CPU, ms) a ratio > 1 means B is SLOWER; for a RATE metric (fps) the
ratio is inverted to A/B so that > 1 ALSO means B is slower — every printed ratio reads the same
way: "cost of B relative to A".
"""

from __future__ import annotations

import json
import random
import statistics
import sys

TIME_METRICS = ("wall_s", "child_cpu_s", "self_cpu_s", "cycle_med_ms", "our_cpu_med_ms",
                "wait_med_ms", "child_per_self", "replay_cpu_s", "replay_wall_s", "cpu_per_spawn_ms", "wall_per_spawn_ms")
RATE_METRICS = ("fps",)


def _boot_median_ci(xs, n=20000, seed=12345):
    rng = random.Random(seed)
    meds = sorted(statistics.median(rng.choices(xs, k=len(xs))) for _ in range(n))
    return meds[int(0.025 * n)], meds[int(0.975 * n) - 1]


def analyze(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    for r in rows:
        # CONTENTION CONTROL: in the env arm both sides run byte-identical Python over the SAME
        # pinned battles, so the parent's own CPU is a per-run speedometer for the box; the Rust
        # children's CPU divided by it cancels the machine-speed swing shared within one run.
        if r.get("child_cpu_s") and r.get("self_cpu_s"):
            r["child_per_self"] = r["child_cpu_s"] / r["self_cpu_s"]
    pairs = {}
    for r in rows:
        pairs.setdefault(r["pair"], {})[r["arm"]] = r
    full = {k: v for k, v in sorted(pairs.items()) if "A" in v and "B" in v}
    loads = [r[k] for r in rows for k in ("load1_start", "load1_end")]
    print(f"\n== {path}: {len(full)} complete pairs, load1 range "
          f"{min(loads):.2f}-{max(loads):.2f}")
    # identical-work check: PINNED runs (the env arm) must play the same battles on both sides
    # (the throughput arm is a fixed time window, so its battle count is an output, not work)
    for k, v in full.items():
        if "measured" not in v["A"]:
            continue
        for fld in ("measured", "battles"):
            if v["A"][fld] != v["B"][fld]:
                print(f"  !! pair {k}: {fld} differs A={v['A'][fld]} B={v['B'][fld]}")
    out = {}
    for m in TIME_METRICS + RATE_METRICS:
        if m not in rows[0] or rows[0][m] is None:
            continue
        if m in RATE_METRICS:
            ratios = [v["A"][m] / v["B"][m] for v in full.values()]
        else:
            ratios = [v["B"][m] / v["A"][m] for v in full.values()]
        med = statistics.median(ratios)
        lo, hi = _boot_median_ci(ratios)
        b_slower = sum(r > 1 for r in ratios)
        out[m] = (med, lo, hi)
        label = f"{m} ({'A/B' if m in RATE_METRICS else 'B/A'})"
        print(f"  {label:<24} median {med:.4f}  95% CI [{lo:.4f}, {hi:.4f}]  "
              f"B slower in {b_slower}/{len(ratios)}  per-pair: "
              + " ".join(f"{r:.3f}" for r in ratios))
    return out


if __name__ == "__main__":
    for p in sys.argv[1:]:
        analyze(p)
