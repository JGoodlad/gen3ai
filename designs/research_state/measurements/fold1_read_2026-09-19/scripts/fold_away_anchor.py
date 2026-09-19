#!/usr/bin/env python3
"""ROW 4 — the Metamon ``SmallRL`` greedy AWAY cell on the fold, against arm W and W_b.

The cell came through the tool of record, ``python -m main.anchors``, which starts and stops its own
Showdown server (:9450 here), verifies the greedy regime PER DECISION on both sides, and writes the
Wilson interval itself. Nothing here re-derives a win rate; this script joins the three arms,
computes the Newcombe interval on each difference, and applies the registered floor.

🚨 **THE FLOOR IS 0.090** = ``|arm W − W_b|`` on this exact cell, from
``flywheel_wb_floor_read_2026-09-18`` §6.1 — itself INSIDE the anchors SOP's imported three-seed
run-level floor of 0.110, and 4.5x the SOP's 0.020 eval-draw floor (rule 19: the variance on this
row is RUN-level, so more games buy nothing).
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

Z = 1.959963984540054
BANKED = {"armS": {"win_rate": 0.520, "n": 100, "wins": 52},
          "armW": {"win_rate": 0.500, "n": 100, "wins": 50},
          "armWb": {"win_rate": 0.590, "n": 100, "wins": 59}}
FLOOR = 0.090
SOP_THREE_SEED_FLOOR = 0.110
SOP_EVAL_DRAW_FLOOR = 0.020


def wilson(k, n):
    p = k / n
    d = 1 + Z * Z / n
    c = p + Z * Z / (2 * n)
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n))
    return p, (c - h) / d, (c + h) / d


def newcombe(k1, n1, k2, n2):
    p1, l1, u1 = wilson(k1, n1)
    p2, l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def main() -> int:
    s = json.loads(Path(sys.argv[1]).read_text())
    dest = Path(sys.argv[2])
    out = {
        "what": "registered row 4 — Metamon SmallRL, greedy-vs-greedy, AWAY team set, 100 games",
        "cell": s["cell"], "status": s["status"],
        "regime_verified_per_decision": s["provenance"]["peer_report"]["regime_verified"],
        "their_argmax_match_rate": s["their_argmax_match_rates"],
        "our_stochastic_kwargs": s["provenance"]["peer_report"]["our_stochastic_kwargs"],
        "n_decisions": s["provenance"]["peer_report"]["n_decisions"],
        "fold_p6M": {"win_rate": s["win_rate"], "wins": s["wins"], "n": s["n"],
                     "wilson95": [round(x, 3) for x in s["wilson95"]],
                     "ties": s["ties"], "hit_forfeit_limit": s["hit_forfeit_limit"],
                     "distinct_our_teams": s["distinct_our_teams"],
                     "by_half": {k: {"win_rate": v["win_rate"], "n": v["n"]}
                                 for k, v in s["by_half"].items()},
                     "mean_turns": s["mean_turns"]},
        "banked_comparators": BANKED,
        "floor": {"value": FLOOR, "source": "|arm W - W_b| on this cell, floor read sec 6.1",
                  "sop_three_seed_run_floor": SOP_THREE_SEED_FLOOR,
                  "sop_eval_draw_floor": SOP_EVAL_DRAW_FLOOR},
        "contrasts": {},
    }
    for ref in ("armW", "armWb"):
        b = BANKED[ref]
        d, lo, hi = newcombe(s["wins"], s["n"], b["wins"], b["n"])
        inside = (lo <= FLOOR <= hi) or (lo <= -FLOOR <= hi)
        out["contrasts"][f"fold_p6M_minus_{ref}"] = {
            "delta": round(d, 3), "newcombe95": [round(lo, 3), round(hi, 3)],
            "clause_a_abs_delta_gt_floor": bool(abs(d) > FLOOR),
            "clause_b_ci_excludes_floor_point": bool(not inside),
            "verdict": ("OUTSIDE THE FLOOR" if (abs(d) > FLOOR and not inside)
                        else "WITHIN FLOOR at n = 2")}
    rh, rp = s["by_half"]["ours_challenge"]["win_rate"], s["by_half"]["peer_challenge"]["win_rate"]
    out["role_split"] = {"ours_challenge": rh, "peer_challenge": rp, "split": round(rh - rp, 3),
                         "note": ("role balance inside the cell is what stops a split becoming a "
                                  "level error; at n = 50 per half nothing else may be taken from "
                                  "it. W_b's 26 pp split on the HOME cell excluded zero.")}
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
