"""The SIZE of forum216b, from X54's power (README.md "Why 216"). Reads X37a's banked Kakuna rows; plays nothing.

    python <this> > power.json

The planning model of a SYMMETRIC panel cell (both sides draw from the panel's T teams, G games):
    Var(mean score) ~= tau2 / T + p (1 - p) / G
where tau2 = Var(a_t + b_t), a_t = the score shift when WE pilot team t and b_t = when KAKUNA pilots it. Team strength
enters a_t and b_t with opposite signs, so it cancels; tau2 is the piloting-by-team interaction a team-clustered
interval charges. It is estimated by method of moments on X37a's two symmetric unfamiliar cells (forum72 and P72,
X26 @161M vs Kakuna, 400 games each): the variance of the per-team d_t minus its binomial noise. The forum cell's
point estimate is the one that applies to a forum panel; its one-sided 95 % upper bound uses the chi-square-free
normal approximation SE(var) ~= var * sqrt(2 / (T - 1)).

Power: P(LB > 0.50) = Phi((p - 0.5 - z se) / se), se at the truth p (a normal interval at level 95 %, two-sided).
"""
from __future__ import annotations

import collections
import json
from math import sqrt
from pathlib import Path
from statistics import NormalDist

HERE = Path(__file__).resolve().parent
ROWS = HERE.parent / "x37a_eval_matrix_2026-10-18"
Z = 1.959964
PLAN_P = (0.53, 0.55)
PLAN_G = (1200, 1600, 2400)
PLAN_T = (72, 108, 144, 216, 288)
TARGET_POWER = 0.80
MAX_G = 2400


def tau2(cell: str) -> dict:
    rows = [json.loads(ln) for f in ("rows_stage1.jsonl", "rows_stage2.jsonl") for ln in (ROWS / f).open()
            if ln.strip()]
    rs = [r for r in rows if r.get("read_cell") == cell]
    y = [1.0 if r["won"] else 0.0 for r in rs]
    p = sum(y) / len(y)
    ours, theirs = collections.defaultdict(list), collections.defaultdict(list)
    for r, v in zip(rs, y):
        ours[r["our_team_stem"]].append(v)
        theirs[r["their_team"]].append(v)
    d, noise = [], []
    for t in sorted(set(ours) & set(theirs)):
        d.append((sum(ours[t]) / len(ours[t]) - p) + (sum(theirs[t]) / len(theirs[t]) - p))
        noise.append(p * (1 - p) * (1 / len(ours[t]) + 1 / len(theirs[t])))
    m = sum(d) / len(d)
    var_d = sum((x - m) ** 2 for x in d) / (len(d) - 1)
    est = var_d - sum(noise) / len(noise)
    upper = max(est, 0.0) + 1.645 * var_d * sqrt(2 / (len(d) - 1))
    return {"cell": cell, "games": len(y), "teams": len(d), "p": round(p, 4), "var_d_hat": round(var_d, 5),
            "binomial_noise": round(sum(noise) / len(noise), 5), "tau2_mom": round(est, 5),
            "tau2_upper95": round(upper, 5)}


def power(p: float, T: int, G: int, t2: float) -> float:
    se = sqrt(t2 / T + p * (1 - p) / G)
    return NormalDist().cdf((p - 0.5 - Z * se) / se)


def main() -> None:
    cells = {c: tau2(c) for c in ("forum72", "p72")}
    plan = {"optimistic (forum point)": max(cells["forum72"]["tau2_mom"], 0.0),
            "pessimistic (forum upper 95 %)": cells["forum72"]["tau2_upper95"]}
    table = {name: {str(T): {f"G{G}": {f"p{p}": round(power(p, T, G, t2), 3) for p in PLAN_P} for G in PLAN_G}
                    for T in PLAN_T} for name, t2 in plan.items()}
    t2 = plan["pessimistic (forum upper 95 %)"]
    need = next(T for T in PLAN_T if power(0.55, T, MAX_G, t2) >= TARGET_POWER)
    print(json.dumps({
        "schema": "x54_panel_power_v1",
        "model": "Var(mean) = tau2 / T + p (1 - p) / G; tau2 = Var(a_t + b_t) by method of moments",
        "cells": cells,
        "tau2_plan": plan,
        "power_LB_gt_0.50": table,
        "rule": f"the smallest T in {list(PLAN_T)} with power >= {TARGET_POWER} at the planning truth p = 0.55 "
                f"and G <= {MAX_G} under the PESSIMISTIC tau2 (games can be added at X54; teams cannot, once "
                "registered)",
        "chosen_T": need,
    }, indent=1))


if __name__ == "__main__":
    main()
