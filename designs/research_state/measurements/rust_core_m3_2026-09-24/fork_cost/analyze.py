"""The M3 tracker fork-cost A/B (`run_roads.py` rows, roads `core` and `core-trk`), per label.

* BEFORE — the Python tracker fork a searched decision pays per successor on today's core road: the
  pinned-pickle THAW (`tracker fork (thaw)`) plus the Python trackers' per-successor fold
  (`python trackers (record_context)` + `(advance_window)`), as ms per arm and as a share of the
  decision wall (the `core` rows' exclusive phase table).
* AFTER — the Rust core's tracker fork: `core-trk`'s Rust fold + render time (`core` +
  `core_render`: with the trackers on, the decision's `present()` view moves from the render into the
  fold) minus `core`'s, per arm, paired by pair; as a share of `core`'s wall. Both
  `phases_ms` and `rust_timing_ms` are WHOLE-RUN sums (every decision's arms), like `ms_total`.

Statistic: per-pair values, median over pairs, paired bootstrap 95 % CI on the median (20,000
resamples, seed 0).

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


def fmt(xs, unit=""):
    lo, hi = boot(xs)
    return f"{statistics.median(xs):.4g}{unit} [{lo:.4g}, {hi:.4g}]"


def main(path):
    rows = [json.loads(line) for line in open(path) if line.strip()]
    for label in sorted({r["label"] for r in rows}):
        by = {}
        for r in rows:
            if r["label"] == label:
                by.setdefault(r["pair"], {})[r["road"]] = r
        pairs = sorted(p for p, a in by.items() if "core" in a and "core-trk" in a)
        if not pairs:
            continue
        loads = [r[k] for p in pairs for r in by[p].values() for k in ("load1_start", "load1_end")]
        c0 = by[pairs[0]]["core"]
        n_dec = c0.get("decisions") or len(c0.get("ms_per_decision", [])) or 1
        arms_per_dec = c0["arms"] / n_dec
        print(f"== {label}: {len(pairs)} pairs, {c0['arms']} arms over {n_dec} decisions "
              f"({arms_per_dec:.1f} arms/decision), load1 {min(loads):.1f}-{max(loads):.1f}")
        thaw, py_trk, share_before, rust_arm, share_after, wall, thaw_share = [], [], [], [], [], [], []
        for p in pairs:
            core, trk = by[p]["core"], by[p]["core-trk"]
            ph = core["phases_ms"]
            t = ph.get("tracker fork (thaw)", 0.0)
            f = ph.get("python trackers (record_context)", 0.0) + ph.get("python trackers (advance_window)", 0.0)
            arms = core["arms"]
            thaw.append(t / arms)
            py_trk.append((t + f) / arms)
            dec_wall = core["ms_per_decision_median"]
            wall.append(dec_wall)
            share_before.append((t + f) / core["ms_total"])
            thaw_share.append(t / core["ms_total"])
            # rust_timing_ms is the WHOLE run's (every decision's arms summed), like ms_total
            # fold + render: with the trackers on, a decision's present() runs inside the fold and is
            # handed to the version's memo, so the view cost MOVES from `core_render` into `core`
            rt = lambda r: r["rust_timing_ms"]["core"] + r["rust_timing_ms"]["core_render"]  # noqa: E731
            d = rt(trk) - rt(core)
            rust_arm.append(d / arms)
            share_after.append(d / core["ms_total"])
        print(f"  decision wall (core road)                 {fmt(wall, ' ms')}")
        print(f"  BEFORE  python thaw / arm                  {fmt(thaw, ' ms')}")
        print(f"  BEFORE  thaw + python trackers / arm       {fmt(py_trk, ' ms')}")
        print(f"  BEFORE  share of the searched decision     {fmt([100 * x for x in share_before], ' %')}"
              "   (the python-trackers phase includes the ROOT prefix replay's own record/advance calls)")
        print(f"  BEFORE  THAW-ONLY share (per-successor)    {fmt([100 * x for x in thaw_share], ' %')}")
        print(f"  AFTER   rust tracker fold / arm            {fmt(rust_arm, ' ms')}")
        print(f"  AFTER   share of the searched decision     {fmt([100 * x for x in share_after], ' %')}")
        print(f"  AFTER / BEFORE per arm                     {fmt([a / b for a, b in zip(rust_arm, py_trk)])}")
        print(f"  AFTER / THAW-ONLY per arm                  {fmt([a / b for a, b in zip(rust_arm, thaw)])}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "rows.jsonl")
