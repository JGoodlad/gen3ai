#!/usr/bin/env python3
"""ROW 1 — THE UNTAUGHT METER: the SPLIT arm beside the CONTROL and the FOLD PATH, paired over TEAMS.

Same tool, same registry OPPONENT (``untaught_meter_opponent`` = ``ai_v9_29_rev1_0823@24,000,000``),
same registry config, same ``--seed 0``, same concurrency 1 and the same untaught-8 manifest in its
canonical order as arm S's 2026-09-14 read, the pair read's 2026-09-16 one, the floor read's
2026-09-18 one, the fold's +6M read, the convergence read and the control read. **The four MEASURED
refs (``split_p3M``, ``split_p6M``, ``split_p12M``, ``armW``) were measured in ONE invocation**, so
every one saw the identical 8 teams and the identical 200 games each.

🚨 **THE WHOLE CONTROL PATH, THE WHOLE FOLD PATH AND W_b ARE DECLARED IMPORTS** — the control's
three depths from ``wcont_control_read_2026-09-20``, the fold path's three and W_b from
``fold1_cont_read_2026-09-19``, the same games in both cases — and the **arm-W reproduction check
in this file is their WARRANT**, registered in ``PREDICTION.md`` sec 1.2/3.5 before the first
battle: arm W must return 46.19 pp (739/1600) AND all eight per-team rows identical, **in BOTH
banked sources**. If it does not, every import is VOID, this row is reported INVALID rather than
patched, and the job STOPS.

``main.untaught_meter`` publishes a level and a CI per ref and a delta only against a ``--baseline``
(which none of these is), so every paired contrast is taken here off the meter's own per-team rows
on ONE resampled team index set — rule 10, the team is the unit. The procedure, the 20,000 draws
and the seed are copied **VERBATIM** from ``wcont_untaught_delta.py``, which copied them from
``cont_untaught_delta.py``, from ``fold_untaught_delta.py``, from ``floor_untaught_delta.py``.

**THREE sign conventions, all labelled at every use:**
  ``delta_path``    = <path> - armW          POSITIVE = ahead of the frozen parent
  ``delta_ecology`` = split - control        THREE levers (bias + pool + team-block-64)
  ``delta_loss``    = split - fold path      ONE lever: the distillation LOSS
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

MAIN = Path("/home/goodlad/dev/gen3ai")
MEAS = MAIN / "designs/research_state/measurements"
BANKED_CONTROL = MEAS / "wcont_control_read_2026-09-20/out/untaught/untaught_registry.json"
BANKED_FOLD = MEAS / "fold1_cont_read_2026-09-19/out/untaught/untaught_registry.json"
BANKED_LEVELS_PP = {"armW": 46.19}
BANKED_WINS = {"armW": (739, 1600)}
CONTROL_IMPORTS = ("wcont_p3M", "wcont_p6M", "wcont_p12M")
FOLD_IMPORTS = ("fold_p1M", "fold_p3M", "fold_p6M", "cont_p7_5M", "cont_p9M", "cont_p12M", "armWb")
IMPORTED_FLOOR_PP = 3.69          # |arm W - W_b|, the 75M FRESH-ARM seed pair
FOLD_DEPTH_FLOORS_PP = {"TC_replicate_pairs_END_depth": 1.19, "frozen_dose_folds_six_draws": 1.66,
                        "K6_cell_three_draws": 2.46, "controller_live_folds": 4.27,
                        "G5_three_continuation_arms": 1.00, "TC_replicate_pairs_p1M_depth": 4.00}
# THE MATCHED GRID: (label, split ref, control ref, fold ref, split-control offset, split-fold offset)
GRID = [("+3M", "split_p3M", "wcont_p3M", "fold_p3M", 0, 0),
        ("+6M", "split_p6M", "wcont_p6M", "fold_p6M", 260928, 303408),
        ("+12M", "split_p12M", "wcont_p12M", "cont_p12M", 0, 0)]
SPLIT_REFS = ("split_p3M", "split_p6M", "split_p12M")
BRANCH_DEPTH, ROBUST_DEPTH = "+12M", "+3M"


def per_team(d: dict):
    res = d["result"]
    teams = res["teams"]
    lv = res["levels"]
    per = {r: np.array([lv[r]["per_team"][t]["win_rate"] for t in teams], float) for r in lv}
    fin = {r: np.array([lv[r]["per_team"][t]["finished"] for t in teams], float) for r in lv}
    win = {r: np.array([lv[r]["per_team"][t]["wins"] for t in teams], float) for r in lv}
    return teams, per, fin, win, res


def paired(a: np.ndarray, b: np.ndarray, seed: int = 20260915):
    """VERBATIM from wcont_untaught_delta.py -- same seed, 20,000 draws, ONE shared index set."""
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
    dest = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("split_untaught_delta.json")
    teams, per, fin, win, res = per_team(json.loads(src.read_text()))

    out: dict = {
        "what": ("registered row 1 -- the untaught meter (the OFF-SLICE / collateral read) on the "
                 "SPLIT arm ai_v13_11_split_lossoff (fold-1's argv with the distillation LOSS OFF) "
                 "at +3M/+6M/+12M, beside the CONTINUATION CONTROL ai_v13_09_wcont and the era-1 "
                 "FOLD PATH at the same three depths, all three against the frozen parent arm W. "
                 "It asks WHICH LEVER carried the era-1 fold's cost: the LOSS (split - fold, ONE "
                 "lever) or the ECOLOGY bundle (split - control, THREE: 40% team bias + two "
                 "specialists in the stable pool + --team-block-episodes 64)."),
        "source_measured": str(src),
        "source_imported_control": str(BANKED_CONTROL),
        "source_imported_fold": str(BANKED_FOLD),
        "teams_manifest": res.get("teams_manifest"),
        "opponent": "registry name `untaught_meter_opponent` = ai_v9_29_rev1_0823@24,000,000",
        "config_resolution": "the REGISTRY `untaught_meter_config`",
        "n_teams": len(teams), "timeouts_measured": res.get("timeouts"),
        "grid_note": ("the matched grid. +3M and +12M are the SAME STEP NUMBER on ALL THREE paths "
                      "(78,006,048 and 87,097,344); at +6M the split's nearest checkpoint is "
                      "81,404,208, which is 260,928 steps DEEPER than the control's 81,143,280 and "
                      "303,408 deeper than the fold's 81,100,800 -- hazard S-D, declared in "
                      "PREDICTION.md sec 1.1, not corrected. The BRANCH CLAUSES are evaluated at "
                      "+12M; +3M is the registered ROBUSTNESS depth; +6M carries no clause."),
        "branch_depth": BRANCH_DEPTH, "robustness_depth": ROBUST_DEPTH,
        "imported_floor_pp": IMPORTED_FLOOR_PP,
        "imported_floor_caveat": (
            "3.69 pp is |arm W - W_b|, a 75M FRESH-ARM seed pair. All three paths here are "
            "CONTINUATIONS of that parent. A floor is a property of the DEPTH and the REGIME "
            f"(rule 3); the meter's own fold/continuation floors are {FOLD_DEPTH_FLOORS_PP}. The "
            "nearest in KIND is the gen-era G5 three-continuation-arm floor of 1.00 pp, which is "
            "at a different parent, era and depth. 3.69 is an IMPORT and is labelled as one in "
            "every row it bars."),
        "fold_depth_floors_pp_for_reference": FOLD_DEPTH_FLOORS_PP,
        "levels_pp": {}, "wins_over_finished": {}, "per_team_pp": {}, "imported": {},
        "reproduction": {}, "delta_path": {}, "delta_ecology": {}, "delta_loss": {},
        "contrasts": {},
    }

    # --- THE REGISTERED REPRODUCTION CHECK, and it GATES every import -------------------------
    bc_teams, bc_per, _bc_fin, bc_win = per_team(json.loads(BANKED_CONTROL.read_text()))[:4]
    bf_teams, bf_per, bf_fin, bf_win = per_team(json.loads(BANKED_FOLD.read_text()))[:4]
    for r, lvl in BANKED_LEVELS_PP.items():
        here = round(100 * float(per[r].mean()), 2)
        w, n = int(win[r].sum()), int(fin[r].sum())
        bw, bn = BANKED_WINS[r]
        out["reproduction"][r] = {
            "banked_pp": lvl, "here_pp": here, "delta_pp": round(here - lvl, 4),
            "banked_wins": f"{bw} / {bn}", "here_wins": f"{w} / {n}",
            "per_team_identical_to_the_control_reads": bool(np.allclose(per[r], bc_per[r])),
            "per_team_identical_to_the_fold_reads": bool(np.allclose(per[r], bf_per[r])),
            "reproduces_exactly": bool(abs(here - lvl) < 0.005 and w == bw and n == bn
                                       and np.allclose(per[r], bc_per[r])
                                       and np.allclose(per[r], bf_per[r]))}
    out["reproduction"]["same_team_set_as_the_banked_reads"] = bool(
        bc_teams == teams and bf_teams == teams)
    repro_ok = (out["reproduction"]["armW"]["reproduces_exactly"]
                and out["reproduction"]["same_team_set_as_the_banked_reads"])
    out["reproduction"]["WARRANT_HOLDS"] = repro_ok
    out["reproduction"]["role"] = (
        "the WARRANT for every imported cell -- the ONE re-derived verification cell the GO names. "
        "Registered in PREDICTION.md sec 1.2/3.5 before the first battle: if arm W does not return "
        "46.19 pp = 739/1600 on all eight per-team rows in BOTH banked sources, every import is "
        "VOID, this row is reported INVALID rather than patched, and the job STOPS and reports.")

    if repro_ok:
        for r in CONTROL_IMPORTS:
            if r in bc_per:
                per[r], fin[r], win[r] = bc_per[r], _bc_fin[r], bc_win[r]
                out["imported"][r] = {"source": "control read",
                                      "wins": f"{int(bc_win[r].sum())} / {int(_bc_fin[r].sum())}",
                                      "level_pp": round(100 * float(bc_per[r].mean()), 2)}
        for r in FOLD_IMPORTS:
            if r in bf_per:
                per[r], fin[r], win[r] = bf_per[r], bf_fin[r], bf_win[r]
                out["imported"][r] = {"source": "fold convergence read",
                                      "wins": f"{int(bf_win[r].sum())} / {int(bf_fin[r].sum())}",
                                      "level_pp": round(100 * float(bf_per[r].mean()), 2)}
    else:
        out["VERDICT"] = "INVALID -- the arm-W reproduction check FAILED; no import applied."

    for r in per:
        out["levels_pp"][r] = round(100 * float(per[r].mean()), 2)
        out["wins_over_finished"][r] = f"{int(win[r].sum())} / {int(fin[r].sum())}"
        out["per_team_pp"][r] = {t: round(100 * v, 2) for t, v in zip(teams, per[r])}

    base = per["armW"]

    # --- delta_path: each path at each depth against the FROZEN PARENT -------------------------
    for label, sref, cref, fref, offc, offf in GRID:
        row = {"depth": label, "step_offset_split_minus_control": offc,
               "step_offset_split_minus_fold": offf}
        for tag, ref in (("split", sref), ("control", cref), ("fold_path", fref)):
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

    # --- 🚨 THE TWO CONTRAST ROWS THIS READ EXISTS TO PRODUCE ---------------------------------
    for key, other_idx, lab, pos, neg, nlev in (
            ("delta_ecology", 2,
             "split minus the CONTROL -- THREE levers (40% team bias + two specialists in the "
             "stable pool + --team-block-episodes 64 vs 1)",
             "the ECOLOGY bundle is ahead of plain continuation",
             "plain continuation is ahead of the ecology bundle", 3),
            ("delta_loss", 3,
             "split minus the FOLD PATH -- ONE lever: the distillation LOSS "
             "(--distill-coef 0.1761 -> 0.0)",
             "turning the LOSS OFF is ahead, i.e. the loss COST this much",
             "the LOSS was worth this much", 1)):
        for label, sref, cref, fref, offc, offf in GRID:
            oref = (cref, fref)[0 if key == "delta_ecology" else 1]
            if sref not in per or oref not in per:
                continue
            pt, lo, hi = paired(per[sref], per[oref])
            out[key][label] = {
                "label": f"{label}: {lab}", "n_levers": nlev,
                "step_offset": offc if key == "delta_ecology" else offf,
                **verdict(100 * pt, [100 * lo, 100 * hi], IMPORTED_FLOOR_PP, pos, neg),
                "teams_favouring_split": int((per[sref] > per[oref]).sum()), "n_teams": len(teams)}

    # --- every other registered contrast ------------------------------------------------------
    extra = [("armW", "armWb", "🚨 THE FLOOR, re-measured on the identical games in THIS invocation"),
             ("split_p12M", "split_p3M", "the split's own last 75 % of the budget (within-arm)"),
             ("split_p12M", "split_p6M", "the split's own last 50 % of the budget (within-arm)"),
             ("split_p6M", "split_p3M", "the split's own +3M -> +6M leg"),
             ("split_p12M", "armWb", "the split's endpoint against the OTHER PARENT SEED"),
             ("split_p3M", "armWb", "the split at +3M against the other parent seed"),
             ("split_p6M", "armWb", "the split at +6M against the other parent seed")]
    for a, b, lab in extra:
        if a not in per or b not in per:
            continue
        pt, lo, hi = paired(per[a], per[b])
        out["contrasts"][f"{a}_minus_{b}"] = {
            "label": lab, "delta_pp": round(100 * pt, 2),
            "ci95_pp": [round(100 * lo, 2), round(100 * hi, 2)],
            "teams_favouring_first": int((per[a] > per[b]).sum()), "n_teams": len(teams)}

    # --- the shape of the split's own row ------------------------------------------------------
    lv = {r: 100 * float(per[r].mean()) for r in SPLIT_REFS if r in per}
    if lv:
        mx = max(lv, key=lv.get)
        out["row1_split_shape"] = {
            "levels_pp_in_order": {r: round(lv[r], 2) for r in SPLIT_REFS if r in per},
            "argmax": mx, "max_pp": round(lv[mx], 2),
            "note": ("three points on ONE arm is a DESCRIPTION of this arm, never a trend (the "
                     "standing short-window refusal). The control and the fold path were both "
                     "FRONT-LOADED: the control had +12.31 of its +15.50 by +3M, the fold path "
                     "moved +0.62 pp over its last 50 %.")}

    out["row1_verdict_rows"] = {
        d: {"delta_ecology": out["delta_ecology"].get(d, {}).get("verdict"),
            "delta_ecology_pp": out["delta_ecology"].get(d, {}).get("delta_pp"),
            "delta_loss": out["delta_loss"].get(d, {}).get("verdict"),
            "delta_loss_pp": out["delta_loss"].get(d, {}).get("delta_pp")}
        for d in (x[0] for x in GRID)}

    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_team_pp"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
