#!/usr/bin/env python3
"""ROW 1 — THE UNTAUGHT METER on the era-1 fold, paired over TEAMS at THREE depths.

Same tool, same registry OPPONENT (``untaught_meter_opponent`` = ``ai_v9_29_rev1_0823@24,000,000``),
same registry config, same ``--seed 0``, same concurrency 1 and the same untaught-8 manifest in its
canonical order as arm S's 2026-09-14 read, the pair read's 2026-09-16 one and the floor read's
2026-09-18 one. **All five refs were measured in ONE invocation**, so every ref saw the identical
8 teams and the identical 200 games each and every contrast below is PAIRED on the same games.

``main.untaught_meter`` publishes a level and a CI per ref and a delta only against a ``--baseline``
(which none of these is), so every paired contrast is taken here off the meter's own per-team rows
on ONE resampled team index set — rule 10, the team is the unit. The procedure, the 20,000 draws and
the seed are copied **VERBATIM** from ``flywheel_wb_floor_read_2026-09-18/scripts/floor_untaught_delta.py``,
which copied them from ``pair_untaught_delta.py``.

**TWO reproduction checks ride in this file, and they are the reason to trust the levels.**
arm W and W_b are BOTH refs here, so (i) each must reproduce its banked 2026-09-18 level to the last
win, and (ii) the 3.69 pp floor itself is re-measured on the identical games.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

MAIN = Path("/home/goodlad/dev/gen3ai")
FLOOR_DIR = MAIN / "designs/research_state/measurements/flywheel_wb_floor_read_2026-09-18/out/untaught"
BANKED_LEVELS_PP = {"armW": 46.19, "armWb": 49.88, "armS": 54.50}
IMPORTED_FLOOR_PP = 3.69          # |arm W - W_b|, the 75M FRESH-ARM seed pair
FOLD_DEPTH_FLOORS_PP = {"TC_replicate_pairs_END_depth": 1.19, "frozen_dose_folds_six_draws": 1.66,
                        "K6_cell_three_draws": 2.46, "controller_live_folds": 4.27,
                        "G5_three_continuation_arms": 1.00, "TC_replicate_pairs_p1M_depth": 4.00}


def per_team(d: dict):
    res = d["result"]
    teams = res["teams"]
    lv = res["levels"]
    per = {r: np.array([lv[r]["per_team"][t]["win_rate"] for t in teams], float) for r in lv}
    fin = {r: np.array([lv[r]["per_team"][t]["finished"] for t in teams], float) for r in lv}
    win = {r: np.array([lv[r]["per_team"][t]["wins"] for t in teams], float) for r in lv}
    return teams, per, fin, win, res


def paired(a: np.ndarray, b: np.ndarray, seed: int = 20260915):
    """VERBATIM from floor_untaught_delta.py — same seed, same 20,000 draws, ONE shared index set."""
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(20000, n))
    dd = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return (float(a.mean() - b.mean()),
            float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5)))


def verdict(delta_pp: float, ci_pp, floor_pp: float) -> dict:
    lo, hi = ci_pp
    d = abs(delta_pp)
    inside = (lo <= floor_pp <= hi) or (lo <= -floor_pp <= hi)
    return {"abs_delta_pp": round(d, 2), "ci95_pp": [round(lo, 2), round(hi, 2)],
            "floor_pp": floor_pp,
            "clause_a_abs_delta_gt_floor": bool(d > floor_pp),
            "clause_b_ci_excludes_floor_point": bool(not inside),
            "direction": ("fold ABOVE parent" if delta_pp > 0 else
                          "fold BELOW parent" if delta_pp < 0 else "exactly level"),
            "verdict": ("OUTSIDE THE FLOOR" if (d > floor_pp and not inside)
                        else "WITHIN FLOOR at n = 2")}


def main() -> int:
    src = Path(sys.argv[1])
    dest = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("fold_untaught_delta.json")
    teams, per, fin, win, res = per_team(json.loads(src.read_text()))

    out: dict = {
        "what": ("the untaught meter -- the OFF-SLICE / collateral read of registered row 1 -- on "
                 "ai_v13_07_fold1 at +1M / +3M / +6M against its parent arm W and the seed arm W_b"),
        "source": str(src),
        "teams_manifest": res.get("teams_manifest"),
        "opponent": "registry name `untaught_meter_opponent` = ai_v9_29_rev1_0823@24,000,000",
        "config_resolution": "the REGISTRY `untaught_meter_config` (see README hazard D-M)",
        "n_teams": len(teams), "timeouts": res.get("timeouts"),
        "levels_pp": {r: round(100 * float(per[r].mean()), 2) for r in per},
        "wins_over_finished": {r: f"{int(win[r].sum())} / {int(fin[r].sum())}" for r in per},
        "per_team_pp": {r: {t: round(100 * v, 2) for t, v in zip(teams, per[r])} for r in per},
        "imported_floor_pp": IMPORTED_FLOOR_PP,
        "imported_floor_caveat": (
            "3.69 pp is |arm W - W_b|, a 75M FRESH-ARM seed pair. This arm is a FOLD. A floor is a "
            "property of the DEPTH and the REGIME (sec 3.3); the meter's own FOLD floors are "
            f"{FOLD_DEPTH_FLOORS_PP}. 3.69 sits between the frozen-dose 1.66 and the "
            "controller-live 4.27, which is why the GO chose it -- it is still an IMPORT."),
        "fold_depth_floors_pp_for_reference": FOLD_DEPTH_FLOORS_PP,
        "contrasts": {}, "reproduction": {}, "verdicts": {},
    }

    # --- reproduction of the two banked levels ------------------------------------------------
    for r, banked in BANKED_LEVELS_PP.items():
        if r in per:
            here = round(100 * float(per[r].mean()), 2)
            out["reproduction"][r] = {
                "banked_2026-09-18_pp": banked, "here_pp": here,
                "delta_pp": round(here - banked, 4),
                "reproduces_exactly": abs(here - banked) < 0.005}
    fp = FLOOR_DIR / "untaught_registry.json"
    if fp.exists():
        fteams, fper, _ffin, fwin, _ = per_team(json.loads(fp.read_text()))
        out["reproduction"]["same_team_set_as_floor_read"] = bool(fteams == teams)
        out["reproduction"]["per_team_identical"] = {
            r: bool(np.allclose(per[r], fper[r])) for r in ("armW", "armWb") if r in per and r in fper}

    # --- every registered contrast -------------------------------------------------------------
    pairs = [("fold_p1M", "armW", "THE FINDING at +1M -- fold minus PARENT"),
             ("fold_p3M", "armW", "THE FINDING at +3M -- fold minus PARENT"),
             ("fold_p6M", "armW", "THE FINDING at +6M -- fold minus PARENT"),
             ("fold_p1M", "armWb", "+1M against the OTHER SEED"),
             ("fold_p3M", "armWb", "+3M against the OTHER SEED"),
             ("fold_p6M", "armWb", "+6M against the OTHER SEED"),
             ("armW", "armWb", "🚨 THE FLOOR, re-measured on the identical games in THIS invocation"),
             ("fold_p6M", "fold_p1M", "the fold's own trajectory, +6M minus +1M (within-arm)")]
    for a, b, label in pairs:
        if a not in per or b not in per:
            continue
        pt, lo, hi = paired(per[a], per[b])
        rec = {"label": label, "delta_pp": round(100 * pt, 2),
               "ci95_pp": [round(100 * lo, 2), round(100 * hi, 2)],
               "teams_favouring_first": int((per[a] > per[b]).sum()), "n_teams": len(teams)}
        out["contrasts"][f"{a}_minus_{b}"] = rec
        if b == "armW" and a.startswith("fold"):
            out["verdicts"][a] = verdict(rec["delta_pp"], rec["ci95_pp"], IMPORTED_FLOOR_PP)

    # --- the registered branch clause: >= 2 of 3 checkpoints, UPWARD --------------------------
    v = out["verdicts"]
    outside = {k: d for k, d in v.items() if d["verdict"] == "OUTSIDE THE FLOOR"}
    up = sorted(k for k, d in outside.items() if d["direction"] == "fold ABOVE parent")
    dn = sorted(k for k, d in outside.items() if d["direction"] == "fold BELOW parent")
    out["row1_branch_clauses"] = {
        "n_checkpoints_reported": len(v),
        "checkpoints_OUTSIDE_upward": up, "checkpoints_OUTSIDE_downward": dn,
        "branch_a_first_clause_untaught_clears_UPWARD_at_ge_2": len(up) >= 2,
        "branch_c_untaught_DOWN_past_the_floor": len(dn) >= 1,
        "note": ("read at all three checkpoints by registration: a fold's off-slice track is a "
                 "HOLE-then-recovery shape (sec 2.3) and the endpoint alone reports whichever "
                 "phase the budget happened to stop in.")}

    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: val for k, val in out.items() if k != "per_team_pp"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
