#!/usr/bin/env python3
"""ROW 2 — PER-SLICE PILOTING: 🚨 THE TEAM-DIFFERENTIAL CONTINUATION READ.

Every ref pilots the SAME pinned taught team against the SAME fixed opponent
(``untaught_meter_opponent`` = ``ai_v9_29_rev1_0823@24,000,000``) drawing the SAME 800-long
pool-team sequence under the SAME dice, so a ref-vs-ref difference is a PILOTING difference on the
same games and not a head-to-head. Instrument, opponent, manifest ORDER, seed, cell size and shard
layout are REUSED VERBATIM from ``fold1_cont_read_2026-09-19/scripts/cont_slice_read.py``.

🚨 **THE WHOLE FOLD PATH, W_b AND BOTH TEACHERS ARE DECLARED IMPORTS, AND THE ARM-W CELL IS THE
WARRANT.** ``armW`` IS re-run and must return 394/800 on Big-5 and 376/800 on DDTar. The meter is
deterministic at seed 0 / concurrency 1 (demonstrated across three different ref lists in three
earlier reads), and a cell is a pure function of (ref, team index, battle index). **If arm W does
not reproduce, every imported cell is VOID and this row is reported as INVALID rather than
patched.** Registered in ``PREDICTION.md`` sec 1.2 before the first battle.

⚠️ **THE CONTROL NEVER TRAINED ON THESE TWO TEAMS WITH ANY BIAS** — no ``--distill-team-bias``, no
teacher piloting them in its stable pool; it played the ordinary 719-team pool for 12.09M steps.
That is what makes it the control for the SPLIT, and it is also why this row cannot separate "a
continuation gets better at DDTar-like teams" from "a continuation gets better at everything and
DDTar is where a 0.47 baseline had room".

⚠️ **THE INTERVAL ON A DIFFERENCE IS NEWCOMBE's, WHICH IS CONSERVATIVE HERE.** The games are paired
under CRN but ``main.untaught_meter`` retains only per-cell counts, so the pairing cannot be
exploited. A cluster bootstrap over TWO teams is not a CI and none is printed.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

Z = 1.959963984540054
MAIN = Path("/home/goodlad/dev/gen3ai")
BANKED = MAIN / ("designs/research_state/measurements/fold1_cont_read_2026-09-19/"
                 "out/cont_slice_read.json")
IMPORTED_REFS = ("fold_p1M", "fold_p3M", "fold_p6M", "cont_p7_5M", "cont_p9M", "cont_p12M",
                 "armWb", "t1_big5", "t2_ddtar")
REPRO_REF = "armW"
REPRO_EXPECT = {"U_f6229d2c": (394, 800), "U_9eb3abdc": (376, 800)}

TEAM_LABEL = {
    "U_f6229d2c": "Big-5 (BALANCE) — t1 `ai_v13_05_exploit_big5starmie`'s pinned team "
                  "data/teams/sample/f6229d2c867e21d6.txt",
    "U_9eb3abdc": "DDTar/Spikes (OFFENSE) — t2 `ai_v13_06_exploit_ddtar_spikes`'s pinned team "
                  "data/teams/sample/9eb3abdc52876a63.txt",
}
TEACHER_OF = {"U_f6229d2c": "t1_big5", "U_9eb3abdc": "t2_ddtar"}
# the matched grid: (label, control ref, fold-path ref, step offset)
GRID = [("+3M", "wcont_p3M", "fold_p3M", 0),
        ("+6M", "wcont_p6M", "fold_p6M", 42480),
        ("+12M", "wcont_p12M", "cont_p12M", 0)]
BAR_CARRYING = ("wcont_p6M", "wcont_p12M")
BIG5, DDTAR = "U_f6229d2c", "U_9eb3abdc"
FOLD_SPLIT = {"+6M": -0.1287, "+12M": -0.2063}   # banked Big-5 minus DDTar on the fold path


def wilson(k: int, n: int):
    if n == 0:
        return (0.0, 0.0, 0.0)
    p = k / n
    d = 1 + Z * Z / n
    c = p + Z * Z / (2 * n)
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return p, (c - h) / d, (c + h) / d


def newcombe(k1: int, n1: int, k2: int, n2: int):
    """Newcombe's method 10 for a difference of two proportions (CONSERVATIVE under CRN)."""
    p1, l1, u1 = wilson(k1, n1)
    p2, l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


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

    imported = {}
    prior = json.loads(BANKED.read_text())
    for t in teams:
        for ref in IMPORTED_REFS:
            c = prior["levels"].get(t, {}).get(ref)
            if not c:
                continue
            imported[f"{ref}|{t}"] = {"wins": c["wins"], "finished": c["finished"]}
            if repro_ok:
                cells.setdefault((ref, t), {"wins": c["wins"], "finished": c["finished"],
                                            "attempted": c["finished"], "timeouts": 0,
                                            "IMPORTED": True})

    out: dict = {
        "what": ("registered row 2 -- per-slice piloting on the two TAUGHT teams for the "
                 "CONTINUATION CONTROL at +3M/+6M/+12M, beside the era-1 fold path at the same "
                 "depths. THE TEAM-DIFFERENTIAL CONTINUATION READ: if the control ALSO rises on "
                 "DDTar and falls on Big-5, the fold's sign split is continuation, not teaching."),
        "opponent": ("registry name `untaught_meter_opponent` = ai_v9_29_rev1_0823@24,000,000 -- ONE "
                     "frozen era-external checkpoint, the same for every ref. NOT arm W's pool "
                     "sentinels: a sentinel is the trainee's OWN snapshot, so arm W's and W_b's "
                     "differ (floor read hazard F-G) and the seed floor would then face different "
                     "opponents from the treatment contrast."),
        "games_per_cell": meta["games_per_team"], "teams": teams,
        "team_pin_sha": meta["team_pins"], "opponent_resolved": meta["opponent_resolved"],
        "team_labels": TEAM_LABEL, "shard_timeouts": meta["timeouts"],
        "control_never_saw_these_teams_with_bias": (
            "the control has no --distill-team-bias and no teacher piloting these teams in its "
            "stable pool; it played the ordinary 719-team pool for 12.09M steps. That is what "
            "makes it the control for the split, and it is also the limit of this row."),
        "reproduction_check": {
            "ref": REPRO_REF, "per_team": repro, "PASSES": repro_ok,
            "role": ("the WARRANT for every imported cell. Registered in PREDICTION.md sec 1.2 "
                     "before the first battle: if arm W does not return 394/800 and 376/800, every "
                     "import is VOID and this row is INVALID, not patched.")},
        "declared_imports": {"refs": list(IMPORTED_REFS), "source": str(BANKED),
                             "cells": imported, "applied": repro_ok,
                             "why": ("the SAME GAMES -- same opponent, manifest, order, seed 0, "
                                     "concurrency 1, 800 games/cell -- and the meter is "
                                     "deterministic. Re-deriving them would cost 9,600 battles "
                                     "and return the same counts.")},
        "interval_note": ("Wilson per cell; Newcombe on every difference. CONSERVATIVE under CRN -- "
                          "the games are paired and the tool retains no per-battle outcomes."),
        "grid_note": ("+3M and +12M are the SAME STEP NUMBER on both paths; the +6M points differ "
                      "by 42,480 steps with the CONTROL the DEEPER (hazard W-D)."),
        "levels": {}, "per_team": {}, "trajectory": {},
    }
    if not repro_ok:
        out["VERDICT"] = "INVALID -- the arm-W reproduction check FAILED; no import applied."

    def diff(a, b, t, label):
        if (a, t) not in cells or (b, t) not in cells:
            return None
        ca, cb = cells[(a, t)], cells[(b, t)]
        d, lo, hi = newcombe(ca["wins"], ca["finished"], cb["wins"], cb["finished"])
        return {"label": label, "delta": round(d, 4), "newcombe95": [round(lo, 4), round(hi, 4)]}

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

        rec: dict = {"label": TEAM_LABEL.get(t, t), "contrasts": {}, "verdicts": {}}
        for label, cref, fref, off in GRID:
            rec["contrasts"][f"{cref}_minus_armW"] = diff(
                cref, "armW", t, f"THE CONTROL at {label} minus the zero-head-start PARENT")
            rec["contrasts"][f"{fref}_minus_armW"] = diff(
                fref, "armW", t, f"the FOLD PATH at {label} minus the parent (IMPORTED cells)")
            rec["contrasts"][f"EXTRACT_{label}"] = diff(
                fref, cref, t, f"🚨 THE EXTRACTION ROW at {label}: fold path minus the CONTROL")
        rec["contrasts"]["THE_FLOOR_armW_minus_armWb"] = diff(
            "armW", "armWb", t, "🚨 THE SEED FLOOR ON THIS CELL -- arm W minus W_b, same team, same "
                                "opponent, same games (arm W MEASURED here, W_b IMPORTED)")
        teach = TEACHER_OF[t]
        rec["contrasts"]["teacher_minus_control_p12M"] = diff(
            teach, "wcont_p12M", t, f"{teach} on its OWN team minus the CONTROL at its endpoint")
        rec["contrasts"]["teacher_minus_armW"] = diff(
            teach, "armW", t, f"{teach} minus the parent it was forked from")
        rec["contrasts"]["control_p12M_minus_p6M"] = diff(
            "wcont_p12M", "wcont_p6M", t, "the CONTROL's own last 50 % of the budget")
        rec["contrasts"]["control_p12M_minus_p3M"] = diff(
            "wcont_p12M", "wcont_p3M", t, "the CONTROL's own last 75 % of the budget")

        fl = rec["contrasts"]["THE_FLOOR_armW_minus_armWb"]
        floor = abs(fl["delta"]) if fl else None
        for key, g in list(rec["contrasts"].items()):
            if not g or floor is None or not (key.endswith("_minus_armW") or key.startswith("EXTRACT_")):
                continue
            lo, hi = g["newcombe95"]
            inside = (lo <= floor <= hi) or (lo <= -floor <= hi)
            ref = key.split("_minus_")[0]
            rec["verdicts"][key] = {
                "registered_bar_carrying_point": ref in BAR_CARRYING,
                "is_extraction_row": key.startswith("EXTRACT_"),
                "delta": g["delta"], "abs_delta": round(abs(g["delta"]), 4),
                "newcombe95": [lo, hi], "floor_on_this_cell": round(floor, 4),
                "clause_a_abs_delta_gt_floor": bool(abs(g["delta"]) > floor),
                "clause_b_ci_excludes_floor_point": bool(not inside),
                "ci_excludes_zero": bool(lo > 0 or hi < 0),
                "verdict": ("OUTSIDE THE FLOOR" if (abs(g["delta"]) > floor and not inside)
                            else "WITHIN FLOOR at n = 2"),
                "direction": "ABOVE" if g["delta"] > 0 else "BELOW"}
        out["per_team"][t] = rec

    # --- the trajectory table, both paths, both teams -----------------------------------------
    for label, cref, fref, off in GRID:
        row = {"depth": label, "step_offset_between_paths": off,
               "registered_bar_carrying": cref in BAR_CARRYING}
        for tag, ref in (("control", cref), ("fold_path", fref)):
            row[tag] = {}
            for t in teams:
                c = out["levels"].get(t, {}).get(ref)
                g = out["per_team"][t]["contrasts"].get(f"{ref}_minus_armW")
                row[tag][t] = {"win_rate": c["win_rate"] if c else None,
                               "imported": c["imported"] if c else None,
                               "delta_vs_armW": g["delta"] if g else None,
                               "newcombe95": g["newcombe95"] if g else None}
        out["trajectory"][label] = row

    # --- 🚨 THE REGISTERED SPLIT STATISTIC -----------------------------------------------------
    split = {}
    for label, cref, fref, off in GRID:
        ent = {}
        for tag, ref in (("control", cref), ("fold_path", fref)):
            ga = out["per_team"][BIG5]["contrasts"].get(f"{ref}_minus_armW")
            gb = out["per_team"][DDTAR]["contrasts"].get(f"{ref}_minus_armW")
            if ga and gb:
                ent[tag] = {"big5_delta": ga["delta"], "ddtar_delta": gb["delta"],
                            "big5_minus_ddtar": round(ga["delta"] - gb["delta"], 4),
                            "sign_pattern_matches_the_fold": bool(gb["delta"] > 0 > ga["delta"])}
        split[label] = ent
    out["THE_SPLIT"] = {
        "question": ("the fold path split in SIGN on its two taught slices -- Big-5 NEGATIVE at all "
                     "six depths, DDTar POSITIVE and clearing its floor at the endpoint -- and a "
                     "SCALAR seniority term cannot produce a sign split. A TEAM-DIFFERENTIAL "
                     "continuation effect can. Does the control split the same way?"),
        "by_depth": split, "fold_path_banked_split": FOLD_SPLIT,
        "note": ("a difference of two independent differences on 800 games each. NO interval is "
                 "printed for it -- that was registered in advance: a bar here would invite a "
                 "verdict this design cannot carry. DESCRIPTOR.")}

    # --- the registered branch clauses (c) / (d) ----------------------------------------------
    end = split.get("+12M", {}).get("control")
    if end:
        matches = end["sign_pattern_matches_the_fold"]
        half = FOLD_SPLIT["+12M"] / 2.0     # -0.10315
        c_met = bool(matches and end["big5_minus_ddtar"] <= half)
        d_met = bool(end["big5_minus_ddtar"] >= 0 or not matches)
        out["row2_branch_clauses"] = {
            "branch_c_TEAM_DIFFERENTIAL_CONTINUATION": {
                "clause": ("the control's +12M signs match the fold path's (DDTar +, Big-5 -) AND "
                           f"its Big-5 minus DDTar <= {round(half, 4)} (half the fold path's "
                           f"{FOLD_SPLIT['+12M']})"),
                "control_big5_minus_ddtar": end["big5_minus_ddtar"],
                "sign_pattern_matches": matches, "MET": c_met},
            "branch_d_THE_SPLIT_IS_THE_FOLDs": {
                "clause": "the control's Big-5 minus DDTar >= 0, or the sign pattern is absent",
                "MET": d_met},
        }
        bc = out["row2_branch_clauses"]
        if c_met and not d_met:
            bc["BRANCH"] = "(c) TEAM-DIFFERENTIAL CONTINUATION"
        elif d_met and not c_met:
            bc["BRANCH"] = "(d) THE SPLIT IS THE FOLD's"
        else:
            bc["BRANCH"] = ("UNCOVERED MIDDLE -- neither registered clause is met as written; "
                            "reported as such, with every clause printed and no branch rewritten "
                            "(PREDICTION.md sec 4)")

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
                    "10, and PRE-DECLARED as a trap in PREDICTION.md sec 2. At the fold's +6M the "
                    "pooled number was +0.003 over the parent while the per-team rows were -0.061 "
                    "and +0.068, each CI clear of zero: an exact null manufactured out of a real "
                    "split. No verdict is taken from this table. Read the per-team rows.")}

    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
