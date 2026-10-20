"""X55 / LG-0, part 3: READ P*'s measured row against the registered predictions and the existing-games basis.

    python -m main.h2h read ~/.cache/gen3ai/x55_league_headroom/ledger --json > row.json
    python read_row.py row.json headroom_existing.json > row_read.json

Pure: reads the two JSON files only. Prints, per cell: the measured score of P* (pair-clustered SE from main.h2h's
edge summary), the P1 prediction (PREDICTIONS.md, registered before the games), the spine's extrapolation (part 1) and
the in-run sentinel proxy; then the derived quantities:
  - S(P*) over R's 7 members (the X-block cells are the frozen R file's 500-pair band scores);
  - stable PFSP over A1's 8 nodes (max(0.05, 1 - p), mastery at p >= 0.80): each node's share, the max / min ratio;
  - the NEAR-EQUAL set: cells whose 95 % interval contains 0.5;
  - P*'s score against the archive's posterior Nash mixture, restricted to the measured support snapshots;
  - P1's pass mark: >= 15 of 18 cells within prediction +- 2 SE.
"""
from __future__ import annotations

import json
import math
import sys

PRED = {30000045: 0.68, 60000135: 0.63, 100000037: 0.60, 80000111: 0.61, 130000065: 0.55, 140000062: 0.55,
        160000085: 0.55, 190000228: 0.51, 210000149: 0.52, 230000074: 0.52, 260000235: 0.52, 270000128: 0.51,
        268000171: 0.51, 244000047: 0.54, 194000195: 0.49, 254000018: 0.51, 224000053: 0.53, 246000132: 0.51}
A1 = [30000045, 60000135, 100000037, 130000065, 160000085, 190000228, 230000074, 260000235]
R_A = [270000128, 80000111, 140000062, 210000149]
X_BLOCK = {"production": 0.6935, "tstar": 0.5905}            # reference_population_production.json band cells
NASH = [268000171, 244000047, 194000195, 254000018, 224000053, 246000132]


def step_of(label: str) -> int:
    return int(str(label).rsplit("@", 1)[-1])


def main() -> int:
    edges = json.load(open(sys.argv[1]))
    base = json.load(open(sys.argv[2]))
    cells = {}
    for e in edges:
        if step_of(e["player"]) != 280000170:
            continue
        cells[step_of(e["opponent"])] = e
    spine = {int(k): v for k, v in base["nash"]["archive_to_268M"]["spine_score_newest_vs"].items()}
    prox = {r["sentinel_step"]: r for r in base["pstar_neighbourhood_row"]["trainee_240_300M"]["rows"]}
    post = {n["step"]: n["post_mean_w"] for n in base["nash"]["archive_to_268M"]["nodes"]}
    rows, hits, n_pred = [], 0, 0
    for st in sorted(cells):
        e = cells[st]
        p, se = float(e["score"]), float(e["se_pp"]) / 100.0
        pr = PRED.get(st)
        inside = None
        if pr is not None:
            n_pred += 1
            inside = abs(p - pr) <= 2 * se + 1e-12
            hits += bool(inside)
        near_sp = min(spine, key=lambda s: abs(s - st))
        rows.append({"opponent_step": st, "pairs": e["pairs"], "score": round(p, 4), "se": round(se, 4),
                     "ci95": [round(p - 1.96 * se, 4), round(p + 1.96 * se, 4)], "predicted": pr,
                     "within_2se_of_prediction": inside, "spine_268M_vs_nearest_ladder_node": spine[near_sp],
                     "spine_node": near_sp, "sentinel_proxy": (prox.get(st) or {}).get("p"),
                     "contains_half": p - 1.96 * se <= 0.5 <= p + 1.96 * se})
    by = {r["opponent_step"]: r for r in rows}
    out = {"cells": rows, "n_cells": len(rows), "p1_pass_mark": {"inside": hits, "of": n_pred,
                                                                 "pass": hits >= 15 and n_pred == 18}}
    if all(s in by for s in R_A):
        s_vals = [0.5] + [by[s]["score"] for s in R_A] + list(X_BLOCK.values())
        se_vals = [0.0] + [by[s]["se"] for s in R_A] + [0.01383, 0.0153]
        out["S_pstar"] = {"value": round(sum(s_vals) / 7, 4),
                          "se": round(math.sqrt(sum(x * x for x in se_vals)) / 7, 4),
                          "predicted": 0.567, "members": ["pstar (0.5 by symmetry)"] + [str(s) for s in R_A] +
                          list(X_BLOCK)}
    if all(s in by for s in A1):
        live = [s for s in A1 if by[s]["score"] < 0.8]
        w = {s: max(0.05, 1.0 - by[s]["score"]) for s in live}
        tw = sum(w.values())
        out["stable_pfsp_over_a1"] = {
            "mastered_at_start": [s for s in A1 if by[s]["score"] >= 0.8],
            "nodes": [{"step": s, "p": by[s]["score"], "share": round(w[s] / tw, 4),
                       "ratio_vs_uniform": round(len(live) * w[s] / tw, 3)} for s in live],
            "max_over_min": round(max(w.values()) / min(w.values()), 3),
            "tv_from_uniform": round(0.5 * sum(abs(w[s] / tw - 1 / len(live)) for s in live), 4)}
        out["a1_row"] = {"mean": round(sum(by[s]["score"] for s in A1) / 8, 4),
                         "early4_mean": round(sum(by[s]["score"] for s in A1[:4]) / 4, 4),
                         "late4_mean": round(sum(by[s]["score"] for s in A1[4:]) / 4, 4)}
    meas = [s for s in NASH if s in by]
    if meas:
        wsum = sum(post[s] for s in meas)
        out["pstar_vs_nash_mixture"] = {
            "support_measured": meas, "posterior_mass_covered": round(wsum, 4),
            "score": round(sum(post[s] * by[s]["score"] for s in meas) / wsum, 4),
            "se": round(math.sqrt(sum((post[s] / wsum) ** 2 * by[s]["se"] ** 2 for s in meas)), 4),
            "worst_cell": min(((by[s]["score"], s) for s in meas))}
    out["near_equal_set"] = sorted(r["opponent_step"] for r in rows if r["contains_half"])
    out["below_half"] = sorted((r["opponent_step"], r["score"]) for r in rows if r["score"] < 0.5)
    json.dump(out, sys.stdout, indent=1)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
