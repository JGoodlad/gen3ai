#!/usr/bin/env python3
"""ROW 1 — THE UNTAUGHT METER: the CONTINUATION CONTROL beside the FOLD PATH, paired over TEAMS.

Same tool, same registry OPPONENT (``untaught_meter_opponent`` = ``ai_v9_29_rev1_0823@24,000,000``),
same registry config, same ``--seed 0``, same concurrency 1 and the same untaught-8 manifest in its
canonical order as arm S's 2026-09-14 read, the pair read's 2026-09-16 one, the floor read's
2026-09-18 one, the fold's +6M read and the convergence read. **The four MEASURED refs
(``wcont_p3M``, ``wcont_p6M``, ``wcont_p12M``, ``armW``) were measured in ONE invocation**, so every
one saw the identical 8 teams and the identical 200 games each.

🚨 **THE WHOLE FOLD PATH AND W_b ARE DECLARED IMPORTS** from
``fold1_cont_read_2026-09-19/out/untaught/untaught_registry.json`` — the same games — and the
**arm-W reproduction check in this file is their WARRANT**, registered in ``PREDICTION.md`` sec 1.2
before the first battle: arm W must return 46.19 pp (739/1600) AND all eight per-team rows
identical. If it does not, every import is VOID and this row is reported INVALID, not patched.

``main.untaught_meter`` publishes a level and a CI per ref and a delta only against a ``--baseline``
(which none of these is), so every paired contrast is taken here off the meter's own per-team rows
on ONE resampled team index set — rule 10, the team is the unit. The procedure, the 20,000 draws
and the seed are copied **VERBATIM** from ``cont_untaught_delta.py``, which copied them from
``fold_untaught_delta.py``, from ``floor_untaught_delta.py``, from ``pair_untaught_delta.py``.

**TWO sign conventions, both labelled at every use:**
  ``delta_path``    = <path> - armW        POSITIVE = ahead of the frozen parent
  ``delta_extract`` = fold path - control  POSITIVE = the fold bought what the continuation did not
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

MAIN = Path("/home/goodlad/dev/gen3ai")
BANKED = MAIN / ("designs/research_state/measurements/fold1_cont_read_2026-09-19/"
                 "out/untaught/untaught_registry.json")
# banked 2026-09-18 (floor read) and 2026-09-19 (both fold reads)
BANKED_LEVELS_PP = {"armW": 46.19}
BANKED_WINS = {"armW": (739, 1600)}
IMPORT_REFS = ("fold_p1M", "fold_p3M", "fold_p6M", "cont_p7_5M", "cont_p9M", "cont_p12M", "armWb")
IMPORTED_FLOOR_PP = 3.69          # |arm W - W_b|, the 75M FRESH-ARM seed pair
FOLD_DEPTH_FLOORS_PP = {"TC_replicate_pairs_END_depth": 1.19, "frozen_dose_folds_six_draws": 1.66,
                        "K6_cell_three_draws": 2.46, "controller_live_folds": 4.27,
                        "G5_three_continuation_arms": 1.00, "TC_replicate_pairs_p1M_depth": 4.00}
# THE MATCHED GRID: (label, control ref, fold-path ref, the step offset between them)
GRID = [("+3M", "wcont_p3M", "fold_p3M", 0),
        ("+6M", "wcont_p6M", "fold_p6M", 42480),
        ("+12M", "wcont_p12M", "cont_p12M", 0)]
CONTROL_REFS = ("wcont_p3M", "wcont_p6M", "wcont_p12M")


def per_team(d: dict):
    res = d["result"]
    teams = res["teams"]
    lv = res["levels"]
    per = {r: np.array([lv[r]["per_team"][t]["win_rate"] for t in teams], float) for r in lv}
    fin = {r: np.array([lv[r]["per_team"][t]["finished"] for t in teams], float) for r in lv}
    win = {r: np.array([lv[r]["per_team"][t]["wins"] for t in teams], float) for r in lv}
    return teams, per, fin, win, res


def paired(a: np.ndarray, b: np.ndarray, seed: int = 20260915):
    """VERBATIM from cont_untaught_delta.py -- same seed, 20,000 draws, ONE shared index set."""
    rng = np.random.default_rng(seed)
    n = len(a)
    idx = rng.integers(0, n, size=(20000, n))
    dd = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return (float(a.mean() - b.mean()),
            float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5)))


def verdict(delta_pp: float, ci_pp, floor_pp: float, pos: str, neg: str) -> dict:
    lo, hi = ci_pp
    d = abs(delta_pp)
    inside = (lo <= floor_pp <= hi) or (lo <= -floor_pp <= hi)
    return {"delta_pp": round(delta_pp, 2), "abs_delta_pp": round(d, 2),
            "ci95_pp": [round(lo, 2), round(hi, 2)], "floor_pp": floor_pp,
            "clause_a_abs_delta_gt_floor": bool(d > floor_pp),
            "clause_b_ci_excludes_floor_point": bool(not inside),
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "direction": (pos if delta_pp > 0 else neg if delta_pp < 0 else "exactly level"),
            "verdict": ("OUTSIDE THE FLOOR" if (d > floor_pp and not inside)
                        else "WITHIN FLOOR at n = 2")}


def main() -> int:
    src = Path(sys.argv[1])
    dest = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("wcont_untaught_delta.json")
    teams, per, fin, win, res = per_team(json.loads(src.read_text()))

    out: dict = {
        "what": ("registered row 1 -- the untaught meter (the OFF-SLICE / collateral read) on the "
                 "CONTINUATION CONTROL ai_v13_09_wcont at +3M/+6M/+12M, beside the era-1 FOLD PATH "
                 "at the same three depths, both against the frozen parent arm W. The G5 cell of "
                 "the win-prob era: it decides whether the fold's off-slice lean was TEACHING or "
                 "CONTINUATION."),
        "source_measured": str(src), "source_imported": str(BANKED),
        "teams_manifest": res.get("teams_manifest"),
        "opponent": "registry name `untaught_meter_opponent` = ai_v9_29_rev1_0823@24,000,000",
        "config_resolution": "the REGISTRY `untaught_meter_config`",
        "n_teams": len(teams), "timeouts_measured": res.get("timeouts"),
        "grid_note": ("the matched grid. +3M and +12M are the SAME STEP NUMBER on both paths "
                      "(78,006,048 and 87,097,344); the +6M points differ by 42,480 steps and the "
                      "CONTROL is the DEEPER of the two -- hazard W-D, declared, not corrected."),
        "imported_floor_pp": IMPORTED_FLOOR_PP,
        "imported_floor_caveat": (
            "3.69 pp is |arm W - W_b|, a 75M FRESH-ARM seed pair. Both paths here are "
            "CONTINUATIONS of that parent. A floor is a property of the DEPTH and the REGIME "
            f"(rule 3); the meter's own fold/continuation floors are {FOLD_DEPTH_FLOORS_PP}. The "
            "nearest in KIND is the gen-era G5 three-continuation-arm floor of 1.00 pp, which is "
            "at a different parent, era and depth. 3.69 is an IMPORT and is labelled as one in "
            "every row it bars."),
        "fold_depth_floors_pp_for_reference": FOLD_DEPTH_FLOORS_PP,
        "levels_pp": {}, "wins_over_finished": {}, "per_team_pp": {}, "imported": {},
        "reproduction": {}, "delta_path": {}, "delta_extract": {}, "contrasts": {},
    }

    # --- THE REGISTERED REPRODUCTION CHECK, and it GATES every import -------------------------
    banked = json.loads(BANKED.read_text())
    bteams, bper, bfin, bwin, _ = per_team(banked)
    for r, lvl in BANKED_LEVELS_PP.items():
        here = round(100 * float(per[r].mean()), 2)
        w, n = int(win[r].sum()), int(fin[r].sum())
        bw, bn = BANKED_WINS[r]
        out["reproduction"][r] = {
            "banked_pp": lvl, "here_pp": here, "delta_pp": round(here - lvl, 4),
            "banked_wins": f"{bw} / {bn}", "here_wins": f"{w} / {n}",
            "per_team_identical_to_banked": bool(np.allclose(per[r], bper[r])),
            "reproduces_exactly": bool(abs(here - lvl) < 0.005 and w == bw and n == bn
                                       and np.allclose(per[r], bper[r]))}
    out["reproduction"]["same_team_set_as_the_banked_reads"] = bool(bteams == teams)
    repro_ok = (out["reproduction"]["armW"]["reproduces_exactly"]
                and out["reproduction"]["same_team_set_as_the_banked_reads"])
    out["reproduction"]["WARRANT_HOLDS"] = repro_ok
    out["reproduction"]["role"] = (
        "the WARRANT for every imported cell, registered in PREDICTION.md sec 1.2 before the first "
        "battle: if arm W does not return 46.19 pp = 739/1600 on all eight per-team rows, every "
        "import is VOID and this row is reported INVALID, not patched.")

    if repro_ok:
        for r in IMPORT_REFS:
            if r in bper:
                per[r], fin[r], win[r] = bper[r], bfin[r], bwin[r]
                out["imported"][r] = {"wins": f"{int(bwin[r].sum())} / {int(bfin[r].sum())}",
                                      "level_pp": round(100 * float(bper[r].mean()), 2)}
    else:
        out["VERDICT"] = "INVALID -- the arm-W reproduction check FAILED; no import applied."

    for r in per:
        out["levels_pp"][r] = round(100 * float(per[r].mean()), 2)
        out["wins_over_finished"][r] = f"{int(win[r].sum())} / {int(fin[r].sum())}"
        out["per_team_pp"][r] = {t: round(100 * v, 2) for t, v in zip(teams, per[r])}

    base = per["armW"]

    # --- delta_path: each path at each depth against the FROZEN PARENT -------------------------
    for label, cref, fref, off in GRID:
        row = {"depth": label, "step_offset_between_paths": off}
        for tag, ref in (("control", cref), ("fold_path", fref)):
            if ref not in per:
                continue
            pt, lo, hi = paired(per[ref], base)
            row[tag] = {"ref": ref, "level_pp": round(100 * float(per[ref].mean()), 2),
                        "wins": f"{int(win[ref].sum())} / {int(fin[ref].sum())}",
                        "imported": ref in out["imported"],
                        **verdict(100 * pt, [100 * lo, 100 * hi], IMPORTED_FLOOR_PP,
                                  "ABOVE the frozen parent", "BELOW the frozen parent"),
                        "teams_favouring_over_armW": int((per[ref] > base).sum())}
        out["delta_path"][label] = row

    # --- 🚨 delta_extract: THE NEW QUANTITY -- fold path minus control, paired -----------------
    for label, cref, fref, off in GRID:
        if cref not in per or fref not in per:
            continue
        pt, lo, hi = paired(per[fref], per[cref])
        out["delta_extract"][label] = {
            "label": f"THE EXTRACTION ROW at {label}: fold path ({fref}) minus the CONTROL ({cref})",
            "step_offset_between_paths": off,
            **verdict(100 * pt, [100 * lo, 100 * hi], IMPORTED_FLOOR_PP,
                      "the FOLD PATH is ahead of the plain continuation",
                      "the CONTROL is ahead of the fold path"),
            "teams_favouring_fold_path": int((per[fref] > per[cref]).sum()), "n_teams": len(teams)}

    # --- every other registered contrast ------------------------------------------------------
    extra = [("armW", "armWb", "🚨 THE FLOOR, re-measured on the identical games in THIS invocation"),
             ("wcont_p12M", "wcont_p3M",
              "the control's own last 75 % of the budget (within-arm)"),
             ("wcont_p12M", "wcont_p6M",
              "the control's own last 50 % of the budget (within-arm), beside the fold path's +0.62"),
             ("wcont_p6M", "wcont_p3M", "the control's own +3M -> +6M leg"),
             ("wcont_p12M", "armWb", "the control's endpoint against the OTHER PARENT SEED"),
             ("wcont_p3M", "armWb", "the control at +3M against the other parent seed"),
             ("wcont_p6M", "armWb", "the control at +6M against the other parent seed")]
    for a, b, lab in extra:
        if a not in per or b not in per:
            continue
        pt, lo, hi = paired(per[a], per[b])
        out["contrasts"][f"{a}_minus_{b}"] = {
            "label": lab, "delta_pp": round(100 * pt, 2),
            "ci95_pp": [round(100 * lo, 2), round(100 * hi, 2)],
            "teams_favouring_first": int((per[a] > per[b]).sum()), "n_teams": len(teams)}

    # --- THE REGISTERED BRANCH CLAUSES, mechanically ------------------------------------------
    dp = {k: v for k, v in out["delta_path"].items() if "control" in v}
    dx = out["delta_extract"]
    a_clause = [k for k, v in dx.items() if abs(v["delta_pp"]) <= IMPORTED_FLOOR_PP]
    a_pos = [k for k in a_clause if dp.get(k, {}).get("control", {}).get("delta_pp", 0) > 0]
    b_small = [k for k, v in dp.items() if abs(v["control"]["delta_pp"]) < IMPORTED_FLOOR_PP]
    b_clear = [k for k, v in dx.items() if v["verdict"] == "OUTSIDE THE FLOOR"]
    ctrl_gt_floor = [k for k, v in dp.items()
                     if v["control"]["delta_pp"] > IMPORTED_FLOOR_PP]
    ctrl_outside = [k for k, v in dp.items() if v["control"]["verdict"] == "OUTSIDE THE FLOOR"]
    out["row1_branch_clauses"] = {
        "branch_a_SENIORITY_CONTINUATION": {
            "clause": ("|delta_extract| <= 3.69 pp at >= 2 of 3 depths, with delta_path(control) "
                       "POSITIVE there"),
            "depths_within_floor_on_delta_extract": a_clause,
            "of_those_with_control_positive": a_pos, "MET": bool(len(a_pos) >= 2)},
        "branch_b_EXTRACTION": {
            "clause": ("|delta_path(control)| < 3.69 pp at >= 2 of 3 AND delta_extract clears the "
                       "floor by BOTH clauses at >= 1 depth"),
            "depths_control_below_floor": b_small,
            "depths_delta_extract_OUTSIDE": b_clear,
            "MET": bool(len(b_small) >= 2 and len(b_clear) >= 1)},
        "control_point_estimate_above_the_floor_at": ctrl_gt_floor,
        "control_OUTSIDE_by_both_clauses_at": ctrl_outside,
        "cis_excluding_zero_control": sorted(
            k for k, v in dp.items() if v["control"]["ci_excludes_zero"]),
        "cis_excluding_zero_delta_extract": sorted(
            k for k, v in dx.items() if v["ci_excludes_zero"]),
        "clause_b_note": (
            "clause (b) is NEAR-UNSATISFIABLE on this row at this n and it was registered as such "
            "BEFORE the games: the team-clustered half-width ran ~3.8-4.0 pp on the fold's own "
            "points and ~4.9-6.2 on the continuation's, against a 3.69 pp floor, so the rule needs "
            "|delta| >~ 7.5 pp. A CI clear of ZERO is reported separately and is NOT a verdict."),
    }
    bc = out["row1_branch_clauses"]
    if bc["branch_a_SENIORITY_CONTINUATION"]["MET"] and not bc["branch_b_EXTRACTION"]["MET"]:
        bc["BRANCH"] = "(a) SENIORITY / CONTINUATION"
    elif bc["branch_b_EXTRACTION"]["MET"] and not bc["branch_a_SENIORITY_CONTINUATION"]["MET"]:
        bc["BRANCH"] = "(b) EXTRACTION"
    elif bc["branch_a_SENIORITY_CONTINUATION"]["MET"] and bc["branch_b_EXTRACTION"]["MET"]:
        bc["BRANCH"] = ("UNCOVERED -- BOTH clauses hold, which the registration did not "
                        "anticipate; reported as uncovered, neither branch declared met")
    else:
        # 🚨 say WHERE the control landed. The registration anticipated a MIDDLE (the control
        # between "~0" and "~+5"); it did not anticipate an OVERSHOOT, and the diagnostic must
        # not call one the other. No branch is rewritten either way.
        ctrl = [v["control"]["delta_pp"] for v in dp.values()]
        fold = [v["fold_path"]["delta_pp"] for v in dp.values() if "fold_path" in v]
        if ctrl and fold and sum(c > f for c, f in zip(ctrl, fold)) >= 2:
            where = ("the CONTROL OVERSHOT the fold path -- it is ABOVE it at >= 2 of 3 depths, "
                     "which is OUTSIDE the span the two registered clauses cover (they bracket a "
                     "control at ~0 and a control at ~+5). Branch (a) fails because "
                     "|delta_extract| is far LARGER than the floor, not smaller; branch (b) fails "
                     "because delta_extract clears the floor in the NEGATIVE direction.")
        elif ctrl and all(abs(c) >= IMPORTED_FLOOR_PP for c in ctrl):
            where = "the control is above the floor but delta_extract does not behave as either clause specifies"
        else:
            where = "the control landed in the MIDDLE the registration named as uncovered"
        bc["BRANCH"] = ("UNCOVERED -- neither registered clause is met; reported as such, with "
                        "every clause printed and no branch rewritten (PREDICTION.md sec 4). "
                        + where)
        bc["where_the_control_landed"] = where

    # --- was the control's own row flat, like the fold path's? --------------------------------
    lv = {r: 100 * float(per[r].mean()) for r in CONTROL_REFS if r in per}
    if lv:
        mx = max(lv, key=lv.get)
        out["row1_control_shape"] = {
            "levels_pp_in_order": {r: round(lv[r], 2) for r in CONTROL_REFS if r in per},
            "argmax": mx, "max_pp": round(lv[mx], 2),
            "note": ("three points on ONE arm is a DESCRIPTION of this arm, never a trend (the "
                     "standing short-window refusal, 7-for-7 on this campaign).")}

    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_team_pp"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
