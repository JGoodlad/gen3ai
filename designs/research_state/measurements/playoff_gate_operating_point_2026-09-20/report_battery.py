"""THE BATTERY'S READ — the playoff cell at the CHOSEN GATE OPERATING POINT and its two
contemporaneous controls.

REUSED VERBATIM from `rollout_leaf_kcurve_2026-09-19/report_battery.py` except for ONE added
derived rate -- `playoff_resolve_rate_of_run`, the fraction of playoffs that ACTUALLY RAN which the
gate then resolved. That is the number the offline operating curve predicts, and it is the
registered live-vs-offline comparison (PREDICTION.md B1); computing it here rather than by hand in
the README is the point. Everything else -- the overlap refusal, the three intervals, the mechanism
fold, the 25 % unfinished bar -- is the proven instrument and is not re-derived.


    python3 report_battery.py <data_dir> --out battery.json

THE PAIRING IS THE 2026-09-11 BATTERY'S, unchanged: `<cell>__<head>__s<lo>.jsonl`, a REFUSAL on
overlapping shard windows (the per-game seed and team draw are functions of the index alone, so an
overlap double-counts one battle and silently breaks the pairing), the side-swap PAIR score in
{0, 0.5, 1}, and `mirror_report`'s null of 0.50 BY CONSTRUCTION. `main.search_dividend.summary`'s
own folds are imported rather than re-derived.

THREE INTERVALS ON L2, and the deviation is declared. The instruction asked for a WILSON interval;
a side-swap PAIR is not a Bernoulli trial (it scores 1 / 0.5 / 0), so Wilson does not apply to it.
Published together:
  * `paired_normal` — the mean of the pair scores with a normal interval. THE HEADLINE, because it
    is the instrument every previous cell was read on and the team draw cancels inside a pair;
  * `paired_boot` — a bootstrap over PAIRS, which makes no normality assumption at small n;
  * `unpaired_wilson` — a Wilson interval on the UNPAIRED decisive-game win rate, where the trial
    really is Bernoulli. It ignores the team draw and is the weakest of the three at these n.

THE MECHANISM ROW IS THE PRIMARY READ (AMENDMENT.md §2): a cell whose override rate is zero has an
L2 that is the harness's own null by construction, so `n_changed`, the playoff's stage split and
the realized rollouts/wall per decision are what the cell is read on, with L2 beside them at
whatever width the clock bought.

HYGIENE: > 25 % of attempted battles unfinished ⇒ INCONCLUSIVE, never a win rate.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import re
from collections import defaultdict

FNAME = re.compile(r"(?P<cell>[a-zA-Z0-9]+)__(?P<head>[a-zA-Z0-9_]+)__s(?P<lo>\d+)\.jsonl$")
INCONCLUSIVE_FRAC = 0.25


def load(data_dir):
    groups, windows = defaultdict(list), defaultdict(list)
    for path in sorted(glob.glob(os.path.join(data_dir, "*.jsonl"))):
        m = FNAME.search(os.path.basename(path))
        if not m:
            continue
        key = m["cell"]
        rows = [json.loads(line) for line in open(path) if line.strip()]
        idx = {(int(r["game"]), int(r.get("orientation", 0) or 0)) for r in rows}
        for other, prev in windows[key]:
            if idx & prev:
                raise SystemExit(f"OVERLAPPING SHARDS for {key}: {other} and {path} share "
                                 f"{len(idx & prev)} (game, orientation) keys — refusing to pool.")
        windows[key].append((path, idx))
        groups[key].extend(rows)
    return groups


def pair_scores(rows):
    by = defaultdict(dict)
    for r in rows:
        if not int(r.get("finished", 0)):
            continue
        s = 0.5 if int(r.get("tied", 0) or 0) else float(int(r.get("won", 0)))
        by[int(r["game"])][int(r.get("orientation", 0) or 0)] = s
    return {g: sum(o.values()) / 2.0 for g, o in by.items() if len(o) == 2}


def mean_ci(xs, boot=4000, seed=7):
    import numpy as np
    n = len(xs)
    if n < 2:
        return {"n": n, "mean": (sum(xs) / n if n else None), "ci_normal": None, "ci_boot": None}
    a = np.asarray(xs, float)
    m = float(a.mean())
    half = 1.96 * math.sqrt(float(a.var(ddof=1)) / n)
    rng = np.random.default_rng(seed)
    bs = [float(a[rng.integers(0, n, n)].mean()) for _ in range(boot)]
    return {"n": n, "mean": m, "ci_normal": [m - half, m + half],
            "ci_boot": [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]}


def wilson(k, n, z=1.96):
    if n <= 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return [(c - h) / d, (c + h) / d]


def paired_delta(a_rows, b_rows, boot=4000, seed=11):
    import numpy as np
    a, b = pair_scores(a_rows), pair_scores(b_rows)
    shared = sorted(set(a) & set(b))
    if len(shared) < 2:
        return {"n_shared_pairs": len(shared), "delta": None}
    d = np.asarray([a[g] - b[g] for g in shared], float)
    m = float(d.mean())
    half = 1.96 * math.sqrt(float(d.var(ddof=1)) / len(d))
    rng = np.random.default_rng(seed)
    bs = [float(d[rng.integers(0, len(d), len(d))].mean()) for _ in range(boot)]
    ci = [float(np.quantile(bs, 0.025)), float(np.quantile(bs, 0.975))]
    return {"n_shared_pairs": len(shared), "delta": m, "ci_normal": [m - half, m + half],
            "ci_boot": ci, "detected": bool(ci[0] > 0 or ci[1] < 0)}


SUMS = ("n_decisions", "n_searched", "n_changed", "n_screen_decisive", "n_playoff",
        "n_playoff_inconclusive", "n_playoff_no_budget", "n_playoff_capped", "n_playoff_failed",
        "n_playoff_ran", "playoff_r_total", "n_defensive", "n_defensive_forced",
        "n_defensive_raced", "n_defensive_separated", "n_defensive_overruled")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir")
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    groups = load(args.data_dir)
    out = {"cells": {}}
    for cell, rows in sorted(groups.items()):
        fin = [r for r in rows if int(r.get("finished", 0))]
        unfinished = len(rows) - len(fin)
        ps = pair_scores(rows)
        l2 = mean_ci(list(ps.values()))
        dec = [r for r in fin if not int(r.get("tied", 0) or 0)]
        won = sum(int(r.get("won", 0)) for r in dec)
        tot = {k: sum(int(r.get(k, 0) or 0) for r in rows) for k in SUMS}
        wall = sum(float(r.get("playoff_wall_s", 0.0) or 0.0) for r in rows)
        c = {
            "n_battles": len(rows), "n_unfinished": unfinished,
            "inconclusive": bool(len(rows) and unfinished / len(rows) > INCONCLUSIVE_FRAC),
            "L2_paired_normal": l2,
            "L2_unpaired_wilson": {"won": won, "decisive": len(dec),
                                   "rate": (won / len(dec) if dec else None),
                                   "ci": wilson(won, len(dec))},
            "mechanism": tot,
            "rates": {
                "changed_per_decision": tot["n_changed"] / max(1, tot["n_decisions"]),
                "screen_decisive_per_decision":
                    tot["n_screen_decisive"] / max(1, tot["n_decisions"]),
                "playoff_played_per_decision": tot["n_playoff"] / max(1, tot["n_decisions"]),
                # THE LIVE RESOLVE RATE -- of the playoffs that actually ran, how many the gate
                # concluded on. This is the offline curve's `resolve_rate`, measured live.
                "playoff_resolve_rate_of_run":
                    tot["n_playoff"]
                    / max(1, tot["n_playoff"] + tot["n_playoff_inconclusive"]),
                "playoff_inconclusive_of_run":
                    tot["n_playoff_inconclusive"]
                    / max(1, tot["n_playoff"] + tot["n_playoff_inconclusive"]),
                "mean_realized_R": tot["playoff_r_total"] / max(1, tot["n_playoff_ran"]),
                "playoff_wall_s_per_ran_decision": wall / max(1, tot["n_playoff_ran"]),
                "defensive_separated_of_raced":
                    tot["n_defensive_separated"] / max(1, tot["n_defensive_raced"]),
                "defensive_overruled_per_decision":
                    tot["n_defensive_overruled"] / max(1, tot["n_decisions"]),
            },
            "mean_wall_s_per_battle": (sum(float(r.get("wall_s", 0) or 0) for r in rows)
                                       / max(1, len(rows))),
            "realized_mean": (rows[0].get("realized_mean") if rows else None),
        }
        out["cells"][cell] = c
        L = c["L2_paired_normal"]
        print(f"[{cell:6s}] battles={c['n_battles']:4d} unfin={unfinished:3d} pairs={L['n']:4d} "
              f"L2={L['mean'] if L['mean'] is None else round(L['mean'],4)} "
              f"ci_norm={L['ci_normal'] and [round(x,4) for x in L['ci_normal']]} "
              f"changed/dec={c['rates']['changed_per_decision']:.4f} "
              f"resolve={c['rates']['playoff_resolve_rate_of_run']:.3f} "
              f"R={c['rates']['mean_realized_R']:.2f} "
              f"s/battle={c['mean_wall_s_per_battle']:.0f}", flush=True)
    out["paired_deltas"] = {}
    names = sorted(groups)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            out["paired_deltas"][f"{names[i]} - {names[j]}"] = paired_delta(
                groups[names[i]], groups[names[j]])
    if args.out:
        json.dump(out, open(args.out, "w"), indent=1)
        print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
