#!/usr/bin/env python3
"""ROW 1 — THE UNTAUGHT METER across the WHOLE FOLD PATH, paired over TEAMS at SIX depths.

Same tool, same registry OPPONENT (``untaught_meter_opponent`` = ``ai_v9_29_rev1_0823@24,000,000``),
same registry config, same ``--seed 0``, same concurrency 1 and the same untaught-8 manifest in its
canonical order as arm S's 2026-09-14 read, the pair read's 2026-09-16 one, the floor read's
2026-09-18 one and the fold's +6M read. **All EIGHT refs were measured in ONE invocation**, so every
ref saw the identical 8 teams and the identical 200 games each: the six-point trajectory
+1M / +3M / +6M / +7.59M / +9.09M / +12.09M is PAIRED on ONE index set, and so is every contrast
against the parent arm W and the seed arm W_b.

``main.untaught_meter`` publishes a level and a CI per ref and a delta only against a ``--baseline``
(which none of these is), so every paired contrast is taken here off the meter's own per-team rows
on ONE resampled team index set — rule 10, the team is the unit. The procedure, the 20,000 draws and
the seed are copied **VERBATIM** from ``fold1_read_2026-09-19/scripts/fold_untaught_delta.py``,
which copied them from ``flywheel_wb_floor_read_2026-09-18/scripts/floor_untaught_delta.py``, which
copied them from ``pair_untaught_delta.py``.

🚨 **FIVE REPRODUCTION CHECKS RIDE IN THIS FILE, REGISTERED AS PASS/FAIL BEFORE THE GAMES.**
arm W, W_b and the fold's three depths are ALL refs here, so each must reproduce its banked level to
the last win AND on the per-team rows; the 3.69 pp floor is re-measured on the identical games. A
failure on any one of them makes every level in this directory suspect and is the headline.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

MAIN = Path("/home/goodlad/dev/gen3ai")
PRIOR = MAIN / "designs/research_state/measurements/fold1_read_2026-09-19/out/untaught"
# banked 2026-09-18 (floor read) and 2026-09-19 (the fold's +6M read) -- see each file's sec 1.1
BANKED_LEVELS_PP = {"armW": 46.19, "armWb": 49.88,
                    "fold_p1M": 47.62, "fold_p3M": 51.56, "fold_p6M": 51.31}
BANKED_WINS = {"armW": (739, 1600), "armWb": (798, 1600),
               "fold_p1M": (762, 1600), "fold_p3M": (825, 1600), "fold_p6M": (821, 1600)}
IMPORTED_FLOOR_PP = 3.69          # |arm W - W_b|, the 75M FRESH-ARM seed pair
FOLD_DEPTH_FLOORS_PP = {"TC_replicate_pairs_END_depth": 1.19, "frozen_dose_folds_six_draws": 1.66,
                        "K6_cell_three_draws": 2.46, "controller_live_folds": 4.27,
                        "G5_three_continuation_arms": 1.00, "TC_replicate_pairs_p1M_depth": 4.00}
# the trajectory, in order, with its cumulative post-fork depth
TRAJ = [("fold_p1M", "+1.00M"), ("fold_p3M", "+3.00M"), ("fold_p6M", "+6.09M"),
        ("cont_p7_5M", "+7.59M"), ("cont_p9M", "+9.09M"), ("cont_p12M", "+12.09M")]
NEW_POINTS = ("cont_p7_5M", "cont_p9M", "cont_p12M")   # the three the GO registered
CROSSING_AFTER = "fold_p6M"     # the fork crossing sits between this point and the next


def per_team(d: dict):
    res = d["result"]
    teams = res["teams"]
    lv = res["levels"]
    per = {r: np.array([lv[r]["per_team"][t]["win_rate"] for t in teams], float) for r in lv}
    fin = {r: np.array([lv[r]["per_team"][t]["finished"] for t in teams], float) for r in lv}
    win = {r: np.array([lv[r]["per_team"][t]["wins"] for t in teams], float) for r in lv}
    return teams, per, fin, win, res


def paired(a: np.ndarray, b: np.ndarray, seed: int = 20260915):
    """VERBATIM from fold_untaught_delta.py -- same seed, same 20,000 draws, ONE shared index set."""
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
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
            "direction": ("fold path ABOVE parent" if delta_pp > 0 else
                          "fold path BELOW parent" if delta_pp < 0 else "exactly level"),
            "verdict": ("OUTSIDE THE FLOOR" if (d > floor_pp and not inside)
                        else "WITHIN FLOOR at n = 2")}


def main() -> int:
    src = Path(sys.argv[1])
    dest = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("cont_untaught_delta.json")
    teams, per, fin, win, res = per_team(json.loads(src.read_text()))

    out: dict = {
        "what": ("the untaught meter -- the OFF-SLICE / collateral read of registered row 1 -- "
                 "across the whole era-1 fold path at SIX depths (+1M/+3M/+6.09M on "
                 "ai_v13_07_fold1, +7.59M/+9.09M/+12.09M on ai_v13_08_fold1_cont), against the "
                 "frozen parent arm W and the seed arm W_b, all eight refs in ONE invocation"),
        "source": str(src),
        "teams_manifest": res.get("teams_manifest"),
        "opponent": "registry name `untaught_meter_opponent` = ai_v9_29_rev1_0823@24,000,000",
        "config_resolution": "the REGISTRY `untaught_meter_config` (see README hazard C-K)",
        "n_teams": len(teams), "timeouts": res.get("timeouts"),
        "levels_pp": {r: round(100 * float(per[r].mean()), 2) for r in per},
        "wins_over_finished": {r: f"{int(win[r].sum())} / {int(fin[r].sum())}" for r in per},
        "per_team_pp": {r: {t: round(100 * v, 2) for t, v in zip(teams, per[r])} for r in per},
        "imported_floor_pp": IMPORTED_FLOOR_PP,
        "imported_floor_caveat": (
            "3.69 pp is |arm W - W_b|, a 75M FRESH-ARM seed pair. This path is a FOLD, twice "
            "forked. A floor is a property of the DEPTH and the REGIME (sec 3.3); the meter's own "
            f"FOLD floors are {FOLD_DEPTH_FLOORS_PP}. 3.69 sits between the frozen-dose 1.66 and "
            "the controller-live 4.27, which is why the GO chose it -- it is still an IMPORT."),
        "fold_depth_floors_pp_for_reference": FOLD_DEPTH_FLOORS_PP,
        "fork_crossing_note": (
            "the +6.09M -> +7.59M leg CROSSES A FORK BOUNDARY (ai_v13_07_fold1 ends at 81,100,800; "
            "ai_v13_08_fold1_cont begins there). Same frozen LR, same teachers, same pool by name, "
            "same matchup hash 0a7b730a4d -- not a rule-15 opponent-regime boundary -- but a "
            "discontinuity in the run, and it is printed on every trajectory row."),
        "trajectory": {}, "contrasts": {}, "reproduction": {}, "verdicts": {},
    }

    # --- the FIVE registered reproduction checks ----------------------------------------------
    for r, banked in BANKED_LEVELS_PP.items():
        if r in per:
            here = round(100 * float(per[r].mean()), 2)
            w, n = int(win[r].sum()), int(fin[r].sum())
            bw, bn = BANKED_WINS[r]
            out["reproduction"][r] = {
                "banked_pp": banked, "here_pp": here, "delta_pp": round(here - banked, 4),
                "banked_wins": f"{bw} / {bn}", "here_wins": f"{w} / {n}",
                "reproduces_exactly": bool(abs(here - banked) < 0.005 and w == bw and n == bn)}
    fp = PRIOR / "untaught_registry.json"
    if fp.exists():
        fteams, fper, _ffin, _fwin, _ = per_team(json.loads(fp.read_text()))
        out["reproduction"]["same_team_set_as_the_p6M_read"] = bool(fteams == teams)
        out["reproduction"]["per_team_identical"] = {
            r: bool(np.allclose(per[r], fper[r])) for r in BANKED_LEVELS_PP if r in per and r in fper}
    out["reproduction"]["ALL_FIVE_PASS"] = all(
        v["reproduces_exactly"] for k, v in out["reproduction"].items()
        if isinstance(v, dict) and "reproduces_exactly" in v)

    # --- THE WITHIN-FOLD TRAJECTORY, the deliverable ------------------------------------------
    base = per.get("armW")
    for i, (r, depth) in enumerate(TRAJ):
        if r not in per:
            continue
        row = {"depth_cumulative_post_fork": depth,
               "run": ("ai_v13_07_fold1" if r.startswith("fold") else "ai_v13_08_fold1_cont"),
               "level_pp": round(100 * float(per[r].mean()), 2),
               "wins": f"{int(win[r].sum())} / {int(fin[r].sum())}",
               "registered_new_point": r in NEW_POINTS,
               "fork_crossing_immediately_before": bool(i and TRAJ[i - 1][0] == CROSSING_AFTER)}
        if base is not None:
            pt, lo, hi = paired(per[r], base)
            row["delta_vs_armW_pp"] = round(100 * pt, 2)
            row["ci95_pp"] = [round(100 * lo, 2), round(100 * hi, 2)]
            row["teams_favouring_fold"] = int((per[r] > base).sum())
            row["verdict"] = verdict(row["delta_vs_armW_pp"], row["ci95_pp"],
                                     IMPORTED_FLOOR_PP)["verdict"]
        if "armWb" in per:
            pt2, lo2, hi2 = paired(per[r], per["armWb"])
            row["delta_vs_armWb_pp"] = round(100 * pt2, 2)
            row["ci95_vs_armWb_pp"] = [round(100 * lo2, 2), round(100 * hi2, 2)]
        out["trajectory"][r] = row

    # --- every registered contrast -------------------------------------------------------------
    pairs = [(r, "armW", f"the fold path at {d} minus the FROZEN PARENT") for r, d in TRAJ]
    pairs += [(r, "armWb", f"{d} against the OTHER SEED") for r, d in TRAJ]
    pairs += [("armW", "armWb",
               "🚨 THE FLOOR, re-measured on the identical games in THIS invocation"),
              ("cont_p12M", "fold_p6M",
               "🚨 THE LAST 50 % OF THE BUDGET -- the endpoint minus the +6M point the earlier "
               "read stopped at (within-path, CROSSES THE FORK)"),
              ("cont_p12M", "fold_p3M",
               "the endpoint minus the +3M peak (within-path, crosses the fork)"),
              ("cont_p7_5M", "fold_p6M",
               "the STOP POINT minus +6M -- the leg the stop rule says completed the teaching"),
              ("cont_p12M", "cont_p7_5M",
               "the endpoint minus the STOP POINT -- 4.5M steps AFTER the recipe's own endpoint"),
              ("fold_p6M", "fold_p1M", "the fold's own early leg, for the record")]
    for a, b, label in pairs:
        if a not in per or b not in per:
            continue
        pt, lo, hi = paired(per[a], per[b])
        rec = {"label": label, "delta_pp": round(100 * pt, 2),
               "ci95_pp": [round(100 * lo, 2), round(100 * hi, 2)],
               "teams_favouring_first": int((per[a] > per[b]).sum()), "n_teams": len(teams)}
        out["contrasts"][f"{a}_minus_{b}"] = rec
        if b == "armW" and a in NEW_POINTS:
            out["verdicts"][a] = verdict(rec["delta_pp"], rec["ci95_pp"], IMPORTED_FLOOR_PP)

    # --- the registered branch clause: >= 2 of the 3 NEW checkpoints --------------------------
    v = out["verdicts"]
    outside = {k: d for k, d in v.items() if d["verdict"] == "OUTSIDE THE FLOOR"}
    up = sorted(k for k, d in outside.items() if d["direction"] == "fold path ABOVE parent")
    dn = sorted(k for k, d in outside.items() if d["direction"] == "fold path BELOW parent")
    out["row1_branch_clauses"] = {
        "n_new_checkpoints_reported": len(v),
        "checkpoints_OUTSIDE_upward": up, "checkpoints_OUTSIDE_downward": dn,
        "branch_a_clause_clears_at_ge_2_of_3_new_points": len(up) >= 2 or len(dn) >= 2,
        "branch_a_clause_UPWARD_at_ge_2": len(up) >= 2,
        "cis_excluding_zero": sorted(k for k, d in v.items() if d["ci_excludes_zero"]),
        "note": ("clause (b) is NEAR-UNSATISFIABLE on this row at this n and it was registered as "
                 "such BEFORE the games: the team-clustered half-width is ~3.8-4.0 pp against a "
                 "3.69 pp floor, so the rule needs |delta| >~ 7.5 pp. A CI clear of ZERO is "
                 "reported separately and is NOT a verdict.")}

    # --- was the +3M peak the maximum, off-slice? ---------------------------------------------
    lv = {r: 100 * float(per[r].mean()) for r, _ in TRAJ if r in per}
    if lv:
        mx = max(lv, key=lv.get)
        out["row1_peak"] = {
            "levels_pp_in_order": {r: round(lv[r], 2) for r, _ in TRAJ if r in per},
            "argmax": mx, "max_pp": round(lv[mx], 2),
            "the_p3M_peak_stands_off_slice": bool(mx == "fold_p3M"),
            "note": ("the off-slice ROW's own peak. The GO's registered peak question is about the "
                     "SLICES (row 2); this is its off-slice companion and is reported beside it.")}

    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: val for k, val in out.items() if k != "per_team_pp"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
