#!/usr/bin/env python3
"""THE BRANCH DECISION — joins row 1 (untaught) and row 2 (the two slice cells) and evaluates the
five registered branches MECHANICALLY, exactly as ``PREDICTION.md`` sec 4 fixed them before the
first battle.

THE THREE ROWS: the untaught 8 (floor 3.69 pp), Big-5 (floor 0.0475), DDTar (floor 0.0850).
THE TWO CONTRASTS: ``Δ_ecology`` = split − control (THREE levers) and ``Δ_loss`` = split − fold
path (ONE lever, the distillation loss).

🚨 The clauses are evaluated at **+12M**, the depth at which all three paths sit on the IDENTICAL
step 87,097,344. **+3M** (also identical, 78,006,048) is the registered ROBUSTNESS depth and is
evaluated too; if the two depths disagree the outcome is **SPLIT BY DEPTH** and no branch is
declared met. **+6M carries the 260,928-step offset and carries no clause.**

Nothing here is hand-entered: every verdict is read out of ``out/split_untaught_delta.json`` and
``out/split_slice_read.json``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

BIG5, DDTAR = "U_f6229d2c", "U_9eb3abdc"
ROWS = ("untaught_8", "big5_slice", "ddtar_slice")


def rows_at(unt: dict, sl: dict, depth: str) -> dict:
    """the three registered ROWS at one depth, each with both contrasts."""
    out = {}
    u_e, u_l = unt["delta_ecology"].get(depth), unt["delta_loss"].get(depth)
    out["untaught_8"] = {
        "floor": unt["imported_floor_pp"], "units": "pp",
        "ecology": {"delta": u_e["delta_pp"], "ci": u_e["ci95_pp"], "verdict": u_e["verdict"],
                    "clause_a": u_e["clause_a_abs_delta_gt_floor"],
                    "clause_b": u_e["clause_b_ci_excludes_floor_point"]},
        "loss": {"delta": u_l["delta_pp"], "ci": u_l["ci95_pp"], "verdict": u_l["verdict"],
                 "clause_a": u_l["clause_a_abs_delta_gt_floor"],
                 "clause_b": u_l["clause_b_ci_excludes_floor_point"]}}
    for name, team in (("big5_slice", BIG5), ("ddtar_slice", DDTAR)):
        v = sl["per_team"][team]["verdicts"]
        e, l_ = v[f"ECOLOGY_{depth}"], v[f"LOSS_{depth}"]
        out[name] = {"floor": e["floor_on_this_cell"], "units": "win-rate",
                     "ecology": {"delta": e["delta"], "ci": e["newcombe95"],
                                 "verdict": e["verdict"], "clause_a": e["clause_a_abs_delta_gt_floor"],
                                 "clause_b": e["clause_b_ci_excludes_floor_point"]},
                     "loss": {"delta": l_["delta"], "ci": l_["newcombe95"],
                              "verdict": l_["verdict"], "clause_a": l_["clause_a_abs_delta_gt_floor"],
                              "clause_b": l_["clause_b_ci_excludes_floor_point"]}}
    return out


def decide(rows: dict) -> dict:
    out_e = [r for r in ROWS if rows[r]["ecology"]["verdict"] == "OUTSIDE THE FLOOR"]
    in_e = [r for r in ROWS if rows[r]["ecology"]["verdict"] != "OUTSIDE THE FLOOR"]
    out_l = [r for r in ROWS if rows[r]["loss"]["verdict"] == "OUTSIDE THE FLOOR"]
    in_l = [r for r in ROWS if rows[r]["loss"]["verdict"] != "OUTSIDE THE FLOOR"]
    e_pos_out = [r for r in out_e if rows[r]["ecology"]["delta"] > 0]
    e_neg_out = [r for r in out_e if rows[r]["ecology"]["delta"] < 0]

    a = len(in_e) == 3 and len(out_l) >= 2
    b = len(in_l) == 3 and len(out_e) >= 2
    c = len(out_e) >= 2 and len(out_l) >= 2 and len(e_neg_out) >= 2
    d = len(e_pos_out) >= 2

    cl = {
        "branch_a_THE_LOSS_IS_THE_CARRIER": {
            "clause": ("|Δ_ecology| WITHIN FLOOR on ALL THREE rows AND |Δ_loss| OUTSIDE THE FLOOR "
                       "on >= 2 rows"),
            "rows_ecology_WITHIN": in_e, "rows_loss_OUTSIDE": out_l, "MET": bool(a),
            "reading": ("the split tracks the CONTROL => the distillation LOSS carried the era-1 "
                        "fold's cost; the ecology bundle is exonerated at this dose")},
        "branch_b_THE_ECOLOGY_IS_THE_CARRIER": {
            "clause": ("|Δ_loss| WITHIN FLOOR on ALL THREE rows AND |Δ_ecology| OUTSIDE THE FLOOR "
                       "on >= 2 rows"),
            "rows_loss_WITHIN": in_l, "rows_ecology_OUTSIDE": out_e, "MET": bool(b),
            "reading": ("the split tracks the FOLD => the bias + pool + team-block-64 bundle "
                        "carried it; the loss is exonerated")},
        "branch_c_BOTH_CARRY": {
            "clause": ("|Δ_ecology| OUTSIDE on >= 2 AND |Δ_loss| OUTSIDE on >= 2, with Δ_ecology "
                       "NEGATIVE (the split BETWEEN) on >= 2 of those"),
            "rows_ecology_OUTSIDE_and_NEGATIVE": e_neg_out, "MET": bool(c),
            "reading": "both levers carry; the two sizes are stated per row"},
        "branch_d_THE_ECOLOGY_HELPED": {
            "clause": "Δ_ecology POSITIVE and OUTSIDE THE FLOOR on >= 2 rows",
            "rows_ecology_OUTSIDE_and_POSITIVE": e_pos_out, "MET": bool(d),
            "reading": ("the split is ABOVE the control => the ecology helped and the loss cost "
                        "MORE than the whole control-to-fold gap")},
    }
    met = [k for k, v in cl.items() if v["MET"]]
    if d and c:
        branch, why = ("(d) THE ECOLOGY HELPED",
                       "registered TIE-BREAK: where (c) and (d) could both be argued the SIGN of "
                       "Δ_ecology decides, and it is POSITIVE on >= 2 rows")
    elif len(met) == 1:
        key = met[0]
        branch = {"branch_a_THE_LOSS_IS_THE_CARRIER": "(a) THE LOSS IS THE CARRIER",
                  "branch_b_THE_ECOLOGY_IS_THE_CARRIER": "(b) THE ECOLOGY IS THE CARRIER",
                  "branch_c_BOTH_CARRY": "(c) BOTH CARRY",
                  "branch_d_THE_ECOLOGY_HELPED": "(d) THE ECOLOGY HELPED"}[key]
        why = cl[key]["clause"]
    elif len(met) > 1:
        branch = "(e) UNCOVERED"
        why = ("more than one registered clause holds and the (c)/(d) tie-break does not apply: "
               + ", ".join(met) + ". Reported as UNCOVERED; no branch rewritten.")
    else:
        branch = "(e) UNCOVERED"
        why = ("no registered clause is met. Ecology OUTSIDE on " + str(out_e) + ", WITHIN on "
               + str(in_e) + "; loss OUTSIDE on " + str(out_l) + ", WITHIN on " + str(in_l)
               + ". PREDICTION.md sec 4's standing instruction is followed literally: every clause "
                 "printed, the nearest description given in the registration's own vocabulary, and "
                 "NO branch rewritten.")
    return {"clauses": cl, "BRANCH": branch, "why": why,
            "counts": {"ecology_OUTSIDE": len(out_e), "loss_OUTSIDE": len(out_l),
                       "ecology_OUTSIDE_positive": len(e_pos_out),
                       "ecology_OUTSIDE_negative": len(e_neg_out)}}


def main() -> int:
    unt = json.loads(Path(sys.argv[1]).read_text())
    sl = json.loads(Path(sys.argv[2]).read_text())
    dest = Path(sys.argv[3])
    out: dict = {
        "what": ("the registered branch decision of PREDICTION.md sec 4, evaluated mechanically "
                 "over THREE rows (untaught 8, Big-5 slice, DDTar slice) and TWO contrasts "
                 "(Δ_ecology = split - control, THREE levers; Δ_loss = split - fold path, ONE "
                 "lever -- the distillation loss)."),
        "rule": ("OUTSIDE THE FLOOR iff |Δ| > floor AND the Δ's 95 % CI excludes the floor POINT; "
                 "else WITHIN FLOOR at n = 2 -- never 'equivalent' (rule 6). One pair BOUNDS a "
                 "floor, it does not estimate one (rules 19/22)."),
        "warrants": {
            "untaught_reproduction": unt["reproduction"]["WARRANT_HOLDS"],
            "slice_reproduction": sl["reproduction_check"]["PASSES"]},
        "by_depth": {}, "PRIMARY": None,
    }
    if not (out["warrants"]["untaught_reproduction"] and out["warrants"]["slice_reproduction"]):
        out["BRANCH"] = ("INVALID -- an arm-W reproduction check FAILED; every import is VOID and "
                         "no branch is declared. The job STOPS and reports.")
        dest.write_text(json.dumps(out, indent=1))
        print(json.dumps(out, indent=1))
        return 1

    for depth, role in (("+12M", "PRIMARY -- all three paths on the identical step 87,097,344"),
                        ("+3M", "ROBUSTNESS -- all three paths on the identical step 78,006,048"),
                        ("+6M", "NO CLAUSE -- the split is 260,928 steps deeper than the control "
                                "and 303,408 deeper than the fold (hazard S-D); printed only")):
        rows = rows_at(unt, sl, depth)
        ent = {"role": role, "rows": rows}
        if depth != "+6M":
            ent.update(decide(rows))
        out["by_depth"][depth] = ent

    prim, rob = out["by_depth"]["+12M"]["BRANCH"], out["by_depth"]["+3M"]["BRANCH"]
    out["PRIMARY"] = prim
    out["ROBUSTNESS_+3M"] = rob
    out["depths_agree"] = bool(prim == rob)
    out["THE_BRANCH"] = prim if prim == rob else (
        f"SPLIT BY DEPTH -- +12M reaches {prim} and +3M reaches {rob}. PREDICTION.md sec 4: both "
        "are printed and NO branch is declared met.")
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
