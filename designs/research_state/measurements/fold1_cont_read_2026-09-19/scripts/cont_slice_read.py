#!/usr/bin/env python3
"""ROW 2 — PER-SLICE PILOTING ACROSS THE WHOLE FOLD PATH, the MATCHED-EXTRACTION ROW.

Every ref pilots the SAME pinned taught team against the SAME fixed opponent
(``untaught_meter_opponent`` = ``ai_v9_29_rev1_0823@24,000,000``) drawing the SAME 800-long
pool-team sequence under the SAME dice, so a ref-vs-ref difference is a PILOTING difference on the
same games and not a head-to-head. Instrument, opponent, manifest ORDER, seed, cell size and shard
layout are REUSED VERBATIM from ``fold1_read_2026-09-19/scripts/fold_slice_read.py``.

🚨 **THREE CELLS PER TEAM ARE DECLARED IMPORTS, AND THE ARM-W CELL IS THE WARRANT.**
``fold_p3M``, ``fold_p6M`` and ``armWb`` (the floor arm) are NOT re-run; their counts are taken from
the +6M read's own JSON. ``armW`` IS re-run, and it must return 394/800 on Big-5 and 376/800 on
DDTar. The meter is deterministic at seed 0 / concurrency 1 (demonstrated three times before this
read and five more times inside it, on the untaught row's per-team rows), the opponent / manifest /
order / seed / cell size are identical, and the shard layout cannot move a cell because a cell is a
pure function of (ref, team index, battle index). **If arm W does not reproduce, every imported
cell is VOID and this row is reported as INVALID rather than patched.** Registered in
``PREDICTION.md`` sec 3.2 before the first battle.

**MATCHED-EXTRACTION DISCIPLINE** (`feedback-matched-extraction-row`, the era standard):

* both arms on ONE harness — never a run's own eval win rate against an h2h number;
* the baseline is MATCHED and ZERO-HEAD-START — **arm W itself**, same role, same opponent, same pin;
* the draws are **PAIRED** by construction (CRN on the battle index);
* **per-cell first, with Wilson intervals**; a pooled number is secondary and labelled;
* 🚨 **seniority is a SEPARATE term from extraction, and at +12.09M it is TWICE what it was at +6M.**
  ``delta = seniority + extraction`` and this row cannot split them. The nearest bound in hand is
  the untaught row; the arm that actually splits them is ``ai_v13_09_wcont``, LIVE and NOT READ.

⚠️ **THE INTERVAL ON A DIFFERENCE IS NEWCOMBE's, WHICH IS CONSERVATIVE HERE.** The games are paired
under CRN but ``main.untaught_meter`` retains only per-cell counts (no per-battle outcome vector),
so the pairing cannot be exploited. A cluster bootstrap over TWO teams is not a CI and none is
printed.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

Z = 1.959963984540054
MAIN = Path("/home/goodlad/dev/gen3ai")
PRIOR = MAIN / "designs/research_state/measurements/fold1_read_2026-09-19/out/fold_slice_read.json"
IMPORTED_REFS = ("fold_p3M", "fold_p6M", "armWb")
REPRO_REF = "armW"
REPRO_EXPECT = {"U_f6229d2c": (394, 800), "U_9eb3abdc": (376, 800)}

TEAM_LABEL = {
    "U_f6229d2c": "Big-5 (BALANCE) — t1 `ai_v13_05_exploit_big5starmie`'s pinned team "
                  "data/teams/sample/f6229d2c867e21d6.txt",
    "U_9eb3abdc": "DDTar/Spikes (OFFENSE) — t2 `ai_v13_06_exploit_ddtar_spikes`'s pinned team "
                  "data/teams/sample/9eb3abdc52876a63.txt",
}
TEACHER_OF = {"U_f6229d2c": "t1_big5", "U_9eb3abdc": "t2_ddtar"}
TEACHER_LEVEL = {"U_f6229d2c": ("t1_big5", 461, 800), "U_9eb3abdc": ("t2_ddtar", 382, 800)}
TRAJ = [("fold_p1M", "+1.00M", "ai_v13_07_fold1"), ("fold_p3M", "+3.00M", "ai_v13_07_fold1"),
        ("fold_p6M", "+6.09M", "ai_v13_07_fold1"), ("cont_p7_5M", "+7.59M", "ai_v13_08_fold1_cont"),
        ("cont_p9M", "+9.09M", "ai_v13_08_fold1_cont"),
        ("cont_p12M", "+12.09M", "ai_v13_08_fold1_cont")]
NEW_POINTS = ("cont_p7_5M", "cont_p12M")   # the two the GO registered as bar-carrying
CROSSING_AFTER = "fold_p6M"


def wilson(k: int, n: int):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + Z * Z / n
    c = p + Z * Z / (2 * n)
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return p, (c - h) / d, (c + h) / d


def newcombe(k1: int, n1: int, k2: int, n2: int):
    """Newcombe's method 10 for a difference of two independent proportions (CONSERVATIVE here)."""
    p1, l1, u1 = wilson(k1, n1)
    p2, l2, u2 = wilson(k2, n2)
    d = p1 - p2
    lo = d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return d, lo, hi


def main() -> int:
    shards = [Path(p) for p in sys.argv[1:-1]]
    dest = Path(sys.argv[-1])
    cells: dict = {}
    meta: dict = {}
    for p in shards:
        raw = json.loads(p.read_text())
        d = raw["result"]
        meta.setdefault("teams", d["teams"])
        meta.setdefault("teams_manifest", raw["_meta"].get("teams_manifest"))
        meta.setdefault("games_per_team", raw["_meta"].get("games_per_team"))
        meta.setdefault("team_pins", {t["key"]: t["pin_sha"] for t in raw["_meta"]["teams"]})
        meta.setdefault("opponent_resolved", raw["_meta"]["opponent"]["resolved_file"])
        for ref, lv in d["levels"].items():
            for t, c in lv["per_team"].items():
                cells[(ref, t)] = c
        meta.setdefault("timeouts", []).append({str(p.name): d.get("timeouts")})
    teams = meta["teams"]

    # --- THE REPRODUCTION CHECK, and it GATES every import -------------------------------------
    repro = {}
    for t, (bw, bn) in REPRO_EXPECT.items():
        c = cells.get((REPRO_REF, t))
        repro[t] = {"banked": f"{bw} / {bn}",
                    "here": (f"{c['wins']} / {c['finished']}" if c else None),
                    "reproduces_exactly": bool(c and c["wins"] == bw and c["finished"] == bn)}
    repro_ok = all(v["reproduces_exactly"] for v in repro.values())

    # --- the DECLARED IMPORTS, pulled only if the warrant holds -------------------------------
    imported = {}
    prior = json.loads(PRIOR.read_text())
    for t in teams:
        for ref in IMPORTED_REFS:
            c = prior["levels"].get(t, {}).get(ref)
            if c:
                imported[f"{ref}|{t}"] = {"wins": c["wins"], "finished": c["finished"]}
                if repro_ok:
                    cells.setdefault((ref, t), {"wins": c["wins"], "finished": c["finished"],
                                                "attempted": c["finished"], "timeouts": 0,
                                                "IMPORTED": True})
        tn, tw, tnn = TEACHER_LEVEL[t]
        pc = prior["levels"].get(t, {}).get(tn)
        if pc and repro_ok:
            cells.setdefault((tn, t), {"wins": pc["wins"], "finished": pc["finished"],
                                       "attempted": pc["finished"], "timeouts": 0,
                                       "IMPORTED": True})

    out: dict = {
        "what": ("registered row 2 -- per-slice piloting on the two TAUGHT teams across the whole "
                 "fold path (matched extraction)"),
        "opponent": ("registry name `untaught_meter_opponent` = ai_v9_29_rev1_0823@24,000,000 -- ONE "
                     "frozen era-external checkpoint, the same for every ref. NOT arm W's pool "
                     "sentinels: a sentinel is the trainee's OWN snapshot, so arm W's and W_b's "
                     "differ (floor read hazard F-G) and the seed floor would then face different "
                     "opponents from the treatment contrast."),
        "games_per_cell": meta["games_per_team"], "teams": teams,
        "team_pin_sha": meta["team_pins"], "opponent_resolved": meta["opponent_resolved"],
        "team_labels": TEAM_LABEL, "shard_timeouts": meta["timeouts"],
        "reproduction_check": {
            "ref": REPRO_REF, "per_team": repro, "PASSES": repro_ok,
            "role": ("the WARRANT for every imported cell. Registered in PREDICTION.md sec 3.2 "
                     "before the first battle: if arm W does not return 394/800 and 376/800, every "
                     "import is VOID and this row is INVALID, not patched.")},
        "declared_imports": {
            "refs": list(IMPORTED_REFS) + ["t1_big5", "t2_ddtar"],
            "source": str(PRIOR), "cells": imported,
            "applied": repro_ok,
            "why": ("these are the SAME GAMES -- same opponent, manifest, order, seed 0, "
                    "concurrency 1, 800 games/cell -- and the meter is deterministic. Re-deriving "
                    "them would cost 6,400 battles and return the same counts.")},
        "interval_note": ("Wilson per cell; Newcombe on every difference. CONSERVATIVE under CRN -- "
                          "the games are paired and the tool retains no per-battle outcomes."),
        "seniority_caveat": ("delta = seniority + extraction and this row cannot split them: at "
                             "+12.09M the fold path carries TWELVE MILLION steps arm W does not. "
                             "The arm that splits them is ai_v13_09_wcont -- LIVE, and NOT READ."),
        "fork_crossing_note": ("the +6.09M -> +7.59M leg crosses the fold1 -> fold1_cont fork at "
                              "81,100,800. Same frozen LR, same teachers, same pool by name, same "
                              "matchup hash -- a discontinuity in the run, not a regime boundary."),
        "levels": {}, "per_team": {}, "trajectory": {},
    }
    if not repro_ok:
        out["VERDICT"] = "INVALID -- the arm-W reproduction check FAILED; no import applied."

    for t in teams:
        rows = {}
        for ref in sorted({r for (r, tt) in cells if tt == t}):
            c = cells[(ref, t)]
            p, lo, hi = wilson(c["wins"], c["finished"])
            rows[ref] = {"wins": c["wins"], "finished": c["finished"],
                         "attempted": c.get("attempted"), "timeouts": c.get("timeouts"),
                         "imported": bool(c.get("IMPORTED")),
                         "win_rate": round(p, 4), "wilson95": [round(lo, 4), round(hi, 4)]}
        out["levels"][t] = rows

        rec: dict = {"label": TEAM_LABEL.get(t, t), "contrasts": {}}

        def diff(a, b, label):
            if (a, t) not in cells or (b, t) not in cells:
                return None
            ca, cb = cells[(a, t)], cells[(b, t)]
            d, lo, hi = newcombe(ca["wins"], ca["finished"], cb["wins"], cb["finished"])
            return {"label": label, "delta": round(d, 4),
                    "newcombe95": [round(lo, 4), round(hi, 4)]}

        for r, depth, run in TRAJ:
            rec["contrasts"][f"{r}_minus_armW"] = diff(
                r, "armW", f"the fold path at {depth} ({run}) minus the zero-head-start PARENT")
        rec["contrasts"]["THE_FLOOR_armW_minus_armWb"] = diff(
            "armW", "armWb", "🚨 THE SEED FLOOR ON THIS CELL -- arm W minus W_b, same team, same "
                             "opponent, same games (arm W MEASURED here, W_b IMPORTED)")
        teach = TEACHER_OF[t]
        rec["contrasts"]["teacher_minus_cont_p12M"] = diff(
            teach, "cont_p12M", f"THE TEACHER'S CEILING at the endpoint -- {teach} on its OWN team "
                                f"minus the converged fold path")
        rec["contrasts"]["teacher_minus_armW"] = diff(
            teach, "armW", f"{teach} minus the parent it was forked from -- the whole gap the fold "
                           f"was asked to close")
        rec["contrasts"]["cont_p12M_minus_fold_p6M"] = diff(
            "cont_p12M", "fold_p6M", "🚨 THE LAST 50 % OF THE BUDGET -- the endpoint minus the "
                                     "point the earlier read stopped at (CROSSES THE FORK)")
        rec["contrasts"]["cont_p12M_minus_fold_p3M"] = diff(
            "cont_p12M", "fold_p3M", "the endpoint minus the +3M PEAK")
        rec["contrasts"]["cont_p7_5M_minus_fold_p6M"] = diff(
            "cont_p7_5M", "fold_p6M", "the STOP POINT minus +6M -- the leg the stop rule says "
                                      "completed the teaching")

        fl = rec["contrasts"]["THE_FLOOR_armW_minus_armWb"]
        for r, _d, _run in TRAJ:
            key = f"{r}_minus_armW"
            f, g = fl, rec["contrasts"].get(key)
            if not f or not g:
                continue
            floor = abs(f["delta"])
            lo, hi = g["newcombe95"]
            inside = (lo <= floor <= hi) or (lo <= -floor <= hi)
            rec.setdefault("verdicts", {})[key] = {
                "registered_bar_carrying_point": r in NEW_POINTS,
                "delta": g["delta"], "abs_delta": round(abs(g["delta"]), 4),
                "newcombe95": [lo, hi], "floor_on_this_cell": round(floor, 4),
                "clause_a_abs_delta_gt_floor": bool(abs(g["delta"]) > floor),
                "clause_b_ci_excludes_floor_point": bool(not inside),
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
                "verdict": ("OUTSIDE THE FLOOR" if (abs(g["delta"]) > floor and not inside)
                            else "WITHIN FLOOR at n = 2"),
                "direction": "fold path ABOVE parent" if g["delta"] > 0 else "fold path BELOW parent"}

        # --- THE REGISTERED PEAK QUESTION: was +3M the maximum on this slice? ------------------
        lv = {r: out["levels"][t][r]["win_rate"] for r, _d, _run in TRAJ if r in out["levels"][t]}
        if lv:
            mx = max(lv, key=lv.get)
            rec["peak"] = {
                "levels_in_order": lv, "argmax": mx, "max": lv[mx],
                "the_p3M_peak_STANDS": bool(mx == "fold_p3M"),
                "endpoint_minus_peak": round(lv.get("cont_p12M", float("nan")) - lv[mx], 4),
                "note": ("the GO's registered question: 'was the +3M peak the maximum?' A peak on "
                         "a six-point single-arm series is a DESCRIPTION of this arm, never a "
                         "trend (the standing short-window refusal, 5-for-5 on this campaign).")}
        out["per_team"][t] = rec

    # --- the trajectory table, both teams side by side ----------------------------------------
    for r, depth, run in TRAJ:
        row = {"depth_cumulative_post_fork": depth, "run": run,
               "registered_bar_carrying": r in NEW_POINTS,
               "fork_crossing_immediately_before": bool(
                   TRAJ[[x[0] for x in TRAJ].index(r) - 1][0] == CROSSING_AFTER
                   and r != TRAJ[0][0])}
        for t in teams:
            c = out["levels"].get(t, {}).get(r)
            g = out["per_team"][t]["contrasts"].get(f"{r}_minus_armW")
            row[t] = {"win_rate": c["win_rate"] if c else None,
                      "imported": c["imported"] if c else None,
                      "delta_vs_armW": g["delta"] if g else None,
                      "newcombe95": g["newcombe95"] if g else None}
        out["trajectory"][r] = row

    # --- THE DIVERGENCE, re-asked at the endpoint ---------------------------------------------
    a, b = "U_f6229d2c", "U_9eb3abdc"
    ga = out["per_team"][a]["contrasts"].get("cont_p12M_minus_armW")
    gb = out["per_team"][b]["contrasts"].get("cont_p12M_minus_armW")
    if ga and gb:
        out["the_divergence_at_the_endpoint"] = {
            "question": ("at +6M the two slices split in SIGN (-0.0612 Big-5 / +0.0675 DDTar) and "
                         "the teachers' gated shares ANTI-predicted it. Does the split survive to "
                         "convergence? The ledger records that the gated-share divergence itself "
                         "did NOT continue (2.5x at the fold's end -> ~1.2x at +12M)."),
            "big5_gain": ga["delta"], "ddtar_gain": gb["delta"],
            "big5_minus_ddtar": round(ga["delta"] - gb["delta"], 4),
            "big5_gained_more": bool(ga["delta"] > gb["delta"]),
            "at_p6M_for_comparison": {"big5": -0.0612, "ddtar": 0.0675, "difference": -0.1287},
            "note": ("a difference of two independent differences on 800 games each; no interval "
                     "is printed for it because the four cells are not a factorial design and an "
                     "interval here would invite a verdict the design cannot carry. DESCRIPTOR.")}

    # --- pooled, SECONDARY and labelled -- the trap, pre-declared -----------------------------
    pooled = {}
    for ref in sorted({r for (r, _t) in cells}):
        w = sum(cells[(ref, t)]["wins"] for t in teams if (ref, t) in cells)
        n = sum(cells[(ref, t)]["finished"] for t in teams if (ref, t) in cells)
        if n:
            p, lo, hi = wilson(w, n)
            pooled[ref] = {"wins": w, "finished": n, "win_rate": round(p, 4),
                           "wilson95": [round(lo, 4), round(hi, 4)]}
    out["pooled_SECONDARY"] = {
        "levels": pooled,
        "warning": ("🚨 POOLED ACROSS TEAMS -- secondary by the matched-extraction rule and by rule "
                    "10, and PRE-DECLARED as a trap in PREDICTION.md sec 2. At +6M the pooled "
                    "number was +0.003 over the parent while the per-team rows were -0.061 and "
                    "+0.068, each CI clear of zero: an exact null manufactured out of a real "
                    "split. No verdict is taken from this table. Read the per-team rows.")}

    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
