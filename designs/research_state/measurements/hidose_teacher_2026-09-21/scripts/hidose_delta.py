#!/usr/bin/env python3
"""THE HIGH-DOSE TEACHER's TWO CELLS — the admission gate and the collateral untaught-8 read.

Adapted from the 2026-09-21 gate's ``admission_delta.py`` (``~/.claude/jobs/1046b1d6/tmp/admission/``),
which copied ``plateau_delta.py``, which copied ``split_untaught_delta.py``.  **The bootstrap is
VERBATIM** — 20,000 draws, ONE shared index set, seed 20260915, and rule 10's unit (the TEAM, not
the battle).  What changed is only which cells are read: one teacher instead of three archetypes,
plus the off-slice cell that the gate did not have.

Both cells were measured with both refs in ONE ``main.untaught_meter`` invocation (registry opponent
``untaught_meter_opponent`` = ``ai_v9_29_rev1_0823@24,000,000``, ``--seed 0``, concurrency 1), so
teacher and parent saw identical teams and identical dice (CRN).

THE RULE, registered in ``../PREDICTION.md`` §3.3 before the first battle:
**OUTSIDE iff |delta| > floor AND the 95 % CI excludes the floor POINT.**
**ADMITTED iff OUTSIDE *and* ABOVE the parent**, at BOTH admission floors.  Not admitted ⇒
**report, do not fold.**

🚨 The training-side vs-target curve plays the TARGET ITSELF; the admission cell plays a fixed THIRD
PARTY.  They are different measurements and neither predicts the other.

Usage:  hidose_delta.py <out_dir> [dest.json]
        <out_dir> holds admission_hidose.json and (optionally) untaught_hidose.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

#: the registered admission floors (pp) — the 2026-09-20 read's two slice cells, unchanged
ADMISSION_FLOORS_PP = {"low": 4.75, "high": 8.50}
#: the registered off-slice floor (pp) — every read in this series has used it
UNTAUGHT_FLOOR_PP = 3.69


def per_team(d: dict, refs: tuple):
    res = d["result"]
    teams, lv = res["teams"], res["levels"]
    for r in refs:
        if r not in lv:
            raise SystemExit(f"ref {r!r} not in {sorted(lv)} — wrong artifact?")
    per = {r: np.array([lv[r]["per_team"][t]["win_rate"] for t in teams], float) for r in refs}
    win = {r: np.array([lv[r]["per_team"][t]["wins"] for t in teams], float) for r in refs}
    fin = {r: np.array([lv[r]["per_team"][t]["finished"] for t in teams], float) for r in refs}
    return teams, per, win, fin


def paired(a: np.ndarray, b: np.ndarray, seed: int = 20260915):
    """VERBATIM from admission_delta.py — same seed, 20,000 draws, ONE shared index set."""
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(20000, n))
    dd = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return (float(a.mean() - b.mean()),
            float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5)))


def verdict(delta_pp: float, ci_pp, floor_pp: float) -> dict:
    """VERBATIM from admission_delta.py."""
    lo, hi = ci_pp
    d = abs(delta_pp)
    inside = (lo <= floor_pp <= hi) or (lo <= -floor_pp <= hi)
    outside = bool(d > floor_pp and not inside)
    return {"delta_pp": round(delta_pp, 2), "ci95_pp": [round(lo, 2), round(hi, 2)],
            "floor_pp": floor_pp,
            "clause_a_abs_delta_gt_floor": bool(d > floor_pp),
            "clause_b_ci_excludes_floor_point": bool(not inside),
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "outside_the_floor": outside,
            "ABOVE_the_parent": bool(delta_pp > 0),
            "ADMITTED": bool(outside and delta_pp > 0)}


def _cell(path: Path, refs: tuple, floors: dict) -> dict:
    teams, per, win, fin = per_team(json.loads(path.read_text()), refs)
    d, lo, hi = paired(per[refs[0]], per[refs[1]])
    return {
        "teams": teams,
        "games_per_team": int(fin[refs[0]][0]) if len(fin[refs[0]]) else None,
        "levels_pp": {r: round(100.0 * float(win[r].sum() / fin[r].sum()), 2) for r in refs},
        "per_team": {t: {refs[0]: round(float(per[refs[0]][i]), 4),
                         refs[1]: round(float(per[refs[1]][i]), 4),
                         "delta": round(float(per[refs[0]][i] - per[refs[1]][i]), 4)}
                     for i, t in enumerate(teams)},
        "at_floor": {k: verdict(100 * d, (100 * lo, 100 * hi), v) for k, v in floors.items()},
    }


def main() -> int:
    src = Path(sys.argv[1])
    dest = Path(sys.argv[2]) if len(sys.argv) > 2 else src / "hidose_delta.json"
    out = {"what": ("ai_v13_18_teach5_offense_hidose vs THE PLATEAU PARENT "
                    "(models/ai_v13_12_plateau/final_model.zip @95,158,272): the ADMISSION cell on "
                    "its own five offense teams, and the COLLATERAL cell on the untaught 8."),
           "rule": ("OUTSIDE iff |delta| > floor AND the 95% CI excludes the floor point; "
                    "ADMITTED iff OUTSIDE *and* ABOVE the parent, at BOTH admission floors. "
                    "Not admitted => report, do not fold."),
           "bootstrap": "20,000 draws, ONE shared team index set, seed 20260915; the TEAM is the unit",
           "opponent": "registry name `untaught_meter_opponent` = ai_v9_29_rev1_0823@24,000,000",
           "admission_floors_pp": ADMISSION_FLOORS_PP,
           "untaught_floor_pp": UNTAUGHT_FLOOR_PP,
           "cells": {}, "missing_cells": []}

    adm = src / "admission_hidose.json"
    if adm.exists():
        c = _cell(adm, ("teacher", "plateau_parent"), ADMISSION_FLOORS_PP)
        c["ADMITTED_at_both_floors"] = all(v["ADMITTED"] for v in c["at_floor"].values())
        out["cells"]["admission"] = c
    else:
        out["missing_cells"].append("admission")

    unt = src / "untaught_hidose.json"
    if unt.exists():
        c = _cell(unt, ("hidose", "plateau_parent"), {"offslice": UNTAUGHT_FLOOR_PP})
        v = c["at_floor"]["offslice"]
        c["COLLATERAL_COST"] = bool(v["outside_the_floor"] and v["delta_pp"] < 0)
        out["cells"]["untaught"] = c
    else:
        out["missing_cells"].append("untaught")

    if not out["missing_cells"]:
        admitted = out["cells"]["admission"]["ADMITTED_at_both_floors"]
        collateral = out["cells"]["untaught"]["COLLATERAL_COST"]
        out["branch"] = (
            "(c) ADMITTED WITH COLLATERAL => report BOTH numbers; branch (a)'s consequences are "
            "SUSPENDED and the trade goes back to the orchestrator" if admitted and collateral else
            "(a) ADMITTED => DOSE was the account; teachers exist on this parent at 1.78x, and "
            "balance + stall get the same treatment before any K ladder" if admitted else
            "(d) UNCOVERED: NOT ADMITTED *and* collateral below the floor => the dose is actively "
            "harmful on this parent. Report; launch nothing" if collateral else
            "(b) NOT ADMITTED => the TEAM-LEVEL CEILING is the surviving account and the teacher "
            "line is CLOSED for this parent (scoped: specialising THIS parent on a pinned team set "
            "at <= +8M; this closes neither distillation nor the K ladder)")
    dest.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: v.get("at_floor") for k, v in out["cells"].items()}, indent=2))
    print(out.get("branch", f"INCOMPLETE — missing {out['missing_cells']}"))
    return 0 if not out["missing_cells"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
