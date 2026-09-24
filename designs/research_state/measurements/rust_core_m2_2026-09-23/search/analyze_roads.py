"""Per-pair ratios of `run_roads.py`'s rows: median + paired bootstrap 95 % CI on the median (20,000
resamples, fixed seed), per label (B setting).

* per DECISION: `ms_per_decision_median` — core/view (the fork + succession cost of the core road
  against the view road) and core/core-text (what the TYPED shortcut saves over the full text
  path);
* per SUCCESSOR: the Rust driver's own fold time per arm (`rust_timing_ms["core"]`, the version's
  stream fold) typed vs text, and the whole `expand_many` Rust time per arm (`total`);
* the shortcut's SHARE of the decision wall: (core-text − core) fold ms per decision ÷ the core
  road's decision wall.

    python analyze_roads.py rows.jsonl
"""

import collections
import json
import random
import statistics
import sys


def boot(ratios, n=20000, seed=0):
    rng = random.Random(seed)
    meds = sorted(statistics.median(rng.choices(ratios, k=len(ratios))) for _ in range(n))
    return meds[int(0.025 * n)], meds[int(0.975 * n)]


def line(name, rs):
    lo, hi = boot(rs)
    return f"    {name:44s} median {statistics.median(rs):.3f}  95% CI [{lo:.3f}, {hi:.3f}]  n={len(rs)}"


def main(path):
    rows = [json.loads(x) for x in open(path)]
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["label"], r["pair"])][r["road"]] = r
    for label in sorted({k[0] for k in by}):
        pairs = [v for k, v in sorted(by.items()) if k[0] == label and len(v) == 3]
        loads = [r[x] for v in pairs for r in v.values() for x in ("load1_start", "load1_end")]
        arms = {road: pairs[0][road]["arms"] for road in ("view", "core", "core-text")}
        print(f"[{label}] {len(pairs)} complete triples, arms per run {arms}, "
              f"load1 {min(loads):.1f}-{max(loads):.1f}, argv {pairs[0]['core']['argv']}")
        for road in ("view", "core", "core-text"):
            ms = [v[road]["ms_per_decision_median"] for v in pairs]
            fold = [v[road]["rust_timing_ms"].get("core", 0.0) / v[road]["arms"] for v in pairs]
            tot = [v[road]["rust_timing_ms"].get("total", 0.0) / v[road]["arms"] for v in pairs]
            print(f"    {road:10s} ms/decision {statistics.median(ms):8.1f}   rust fold ms/arm "
                  f"{statistics.median(fold):.4f}   rust expand_many ms/arm {statistics.median(tot):.4f}")
        dec = lambda a, b: [v[a]["ms_per_decision_median"] / v[b]["ms_per_decision_median"] for v in pairs]  # noqa: E731
        fold = lambda a, b: [(v[a]["rust_timing_ms"]["core"] / v[a]["arms"])  # noqa: E731
                             / (v[b]["rust_timing_ms"]["core"] / v[b]["arms"]) for v in pairs]
        tot = lambda a, b: [(v[a]["rust_timing_ms"]["total"] / v[a]["arms"])  # noqa: E731
                            / (v[b]["rust_timing_ms"]["total"] / v[b]["arms"]) for v in pairs]
        print(line("per decision  core / view", dec("core", "view")))
        print(line("per decision  core / core-text", dec("core", "core-text")))
        print(line("per successor fold  core / core-text", fold("core", "core-text")))
        print(line("per successor expand_many  core / core-text", tot("core", "core-text")))
        print(line("per successor expand_many  core / view", tot("core", "view")))
        share = []
        for v in pairs:
            dfold = (v["core-text"]["rust_timing_ms"]["core"] - v["core"]["rust_timing_ms"]["core"])
            wall = v["core"]["ms_total"]
            share.append(dfold / wall)
        print(line("shortcut saving / core decision wall", share))


if __name__ == "__main__":
    main(sys.argv[1])
