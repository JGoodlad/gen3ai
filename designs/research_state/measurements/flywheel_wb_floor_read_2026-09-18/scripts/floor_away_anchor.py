#!/usr/bin/env python3
"""THE 75M RUN-LEVEL FLOOR — the AWAY-set external anchor, through the TOOL OF RECORD `main.anchors`.

`designs/ops/EXTERNAL_ANCHORS_SOP.md` sec 1: **report BOTH team sets, always** — they disagreed in
SIGN for `SmallRL`, and the away set (Metamon's own 20-team `competitive` gen3ou export) is the one
no dial of ours is set to. W_b's cell is taken with the SAME command the pair read used on arm W and arm S:
`python -m main.anchors --opponent metamon:SmallRL --regime greedy --teamset away --games 100`,
which verifies the regime PER DECISION on both sides and stamps every row of `games.jsonl` with it.
Arm W's and arm S's cells are joined from the pair read's committed `out/away/` summaries.

|arm W - W_b| on this cell is the 75M RUN-LEVEL FLOOR for it, and the SOP already carries a
THREE-SEED run-level floor of **0.110** for this exact cell (ctrl10M / _b / _c read 0.440 / 0.380 /
0.330 at 10M) plus an eval-draw floor of 0.020 [EXTERNAL_ANCHORS_SOP amendment 2026-09-16]. Both
are printed. 🚨 The SOP also says the budget effect on this cell (~0.38 at 10M -> ~0.64 at 75M)
dwarfs every lever the ladder measured, so a 10M floor imported to 75M is an import, not a match.

Sign convention: arm S MINUS arm W on the finding row (the pair read's orientation); the floor row
is arm W MINUS W_b and is reported unsigned as well.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

Z = 1.959963985
NEW = Path("/home/goodlad/.claude/jobs/9ab51de6/tmp/wb_read/anchors/away")
BANKED = Path("/home/goodlad/dev/gen3ai/designs/research_state/measurements/"
              "flywheel_pair_read_2026-09-15/out/away")
ARMS = {
    "armWb": (NEW / "armWb_smallrl_greedy_away" / "summary.json",
              "ai_v13_04_flywheel_winprob_b", "W_b — arm W's TOKEN-EXACT SEED REPLICATE (seed 1002)"),
    "armW": (BANKED / "armW" / "summary.json",
             "ai_v13_02_flywheel_winprob", "arm W — the pair's WIN-PROB arm (seed 1001)"),
    "armS": (BANKED / "armS" / "summary.json",
             "ai_v13_01_flywheel_shaped", "arm S — the pair's SHAPED arm (seed 1001)"),
}
SOP_RUN_FLOOR = 0.110      # three seeds on THIS cell, measured at 10M [SOP amendment 2026-09-16]
SOP_DRAW_FLOOR = 0.020     # four team-seed draws of one 10M snapshot, same amendment


def wilson(k: int, n: int):
    p = k / n
    d = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / d
    h = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def newcombe(k1, n1, k2, n2):
    p1, l1, u1 = wilson(k1, n1)
    p2, l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return (d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2),
            d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2))


def main() -> int:
    out: dict = {"cell": "metamon:SmallRL, regime greedy (BOTH sides, verified per decision), "
                         "teamset away (Metamon's own 20-team competitive gen3ou export), 100 games "
                         "split evenly across the two challenge roles",
                 "tool": "python -m main.anchors", "arms": {}}
    ks = {}
    for tag, (path, run, human) in ARMS.items():
        if not path.exists():
            out["arms"][tag] = {"error": f"no summary at {path}"}
            continue
        s = json.loads(path.read_text())
        k, n = int(s["wins"]), int(s["n"])
        ks[tag] = (k, n)
        p, lo, hi = wilson(k, n)
        pr = s["provenance"]["peer_report"]
        out["arms"][tag] = {
            "run": run, "human_description": human,
            "model_step": s["cell"]["model_step"], "model_zip": s["cell"]["model_zip"],
            "n": n, "wins": k, "losses": s["losses"], "ties": s.get("ties"),
            "win_rate": round(p, 4), "wilson95": [round(lo, 4), round(hi, 4)],
            "by_half": {h: {"n": v["n"], "wins": v["wins"], "win_rate": v["win_rate"]}
                        for h, v in s["by_half"].items()},
            "mean_turns": s["mean_turns"], "max_turns": s["max_turns"],
            "hit_forfeit_limit": s["hit_forfeit_limit"],
            "regime_verified": {"ours": s["cell"]["our_regime"],
                                "theirs": s["cell"]["their_regime"],
                                "regime_matched": s["cell"]["regime_matched"],
                                "peer_argmax_match_rate": pr.get("argmax_match_rate")},
            "opponent": {"version": s["cell"]["opponent_version"],
                         "commit": s["cell"]["opponent_commit"]},
            "showdown_pin": s["cell"]["showdown_pin"],
        }
    def contrast(a, b, label):
        d, lo, hi = newcombe(*ks[a], *ks[b])
        return {"label": label, "delta": round(d, 4), "abs_delta": round(abs(d), 4),
                "newcombe95": [round(lo, 4), round(hi, 4)],
                "verdict": ("NOT DETECTED — the difference interval covers zero. Rule 6: never "
                            "'equal'; equivalence needs the delta's own CI inside a bar"
                            if lo <= 0 <= hi else "DETECTED at 95% — the interval excludes zero")}

    if "armW" in ks and "armWb" in ks:
        out["THE_FLOOR_W_minus_Wb"] = contrast("armW", "armWb",
            "THE 75M RUN-LEVEL FLOOR on this cell — arm W (seed 1001) minus W_b (seed 1002)")
        out["THE_FLOOR_W_minus_Wb"]["sop_three_seed_run_floor_at_10M"] = SOP_RUN_FLOOR
        out["THE_FLOOR_W_minus_Wb"]["sop_eval_draw_floor_at_10M"] = SOP_DRAW_FLOOR
        out["THE_FLOOR_W_minus_Wb"]["inside_sop_run_floor"] = bool(
            abs(out["THE_FLOOR_W_minus_Wb"]["delta"]) <= SOP_RUN_FLOOR)
    if "armS" in ks and "armW" in ks:
        out["THE_FINDING_S_minus_W"] = contrast("armS", "armW",
            "the pair's row — arm S minus arm W (the pair read reported +0.020 [-0.117, +0.155])")
    if "armS" in ks and "armWb" in ks:
        out["S_minus_Wb"] = contrast("armS", "armWb",
            "arm S minus W_b — the treatment contrast against the OTHER seed")
    f, g = out.get("THE_FLOOR_W_minus_Wb"), out.get("THE_FINDING_S_minus_W")
    if f and g:
        fl = f["abs_delta"]; dd = abs(g["delta"]); lo, hi = g["newcombe95"]
        inside = (lo <= fl <= hi) or (lo <= -fl <= hi)
        out["floor_verdict"] = {
            "finding_abs": dd, "finding_ci95": [lo, hi], "floor_abs": fl,
            "clause_a_abs_delta_gt_floor": bool(dd > fl),
            "clause_b_ci_excludes_floor_point": bool(not inside),
            "verdict": ("OUTSIDE THE 75M FLOOR" if (dd > fl and not inside)
                        else "WITHIN FLOOR at n = 2"),
            "caveats": ["one replicate pair BOUNDS a floor; no CI attaches to it (rules 19/22)",
                        "WITHIN FLOOR is NEVER 'equivalent' (rule 6)",
                        "100 games resolves ~+/-10 pp at best; the SOP says every cell near even "
                        "stays NOT DETECTED however the point estimate reads"]}
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("floor_away_anchor.json")
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "arms"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
