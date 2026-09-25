#!/usr/bin/env python3
"""Round 2 §4.1 MANIPULATION CHECK — did B2 (the round-2 loop generalist) absorb what it was shown?

M2 = [B2's pooled greedy rate over its LAST TWO eval cycles against all THREE specialists]
   − [C2's (the round-2 no-exploiter control's) same], n = 600 per arm, Newcombe 95 %.
ABSORBED iff M2's CI lower bound > 0. Read from each run's own ``eval_results.jsonl`` ``externals``
(100 greedy-vs-greedy games per specialist per cycle). Specialists: A = ai_v13_18_teach5_offense_hidose,
A13 = ai_v13_13_exploit5_offense, RB = ai_v13_24_popr1_read_loop (the new one, the one that matters most).
Round 1's script (../../../population_loop_r1_2026-09-23/read/scripts/manipulation_check.py) with the
arms and the third specialist changed.

PYTHONPATH must contain src (the Newcombe helper is the meter's own).  Usage: manipulation_check.py <out.json>
"""
import json
import sys
from pathlib import Path

from agents.training.best_response_gap import newcombe_diff_ci

MODELS = Path("/home/goodlad/dev/gen3ai/models")
ARMS = {"B2": "ai_v13_27_popr2_loop", "C2": "ai_v13_28_popr2_ctrl"}
SPECS = ["ext_ai_v13_18_teach5_offense_hidose", "ext_ai_v13_13_exploit5_offense",
         "ext_ai_v13_24_popr1_read_loop"]


def series(run):
    rows = [json.loads(line) for line in open(MODELS / run / "eval_results.jsonl")]
    return [{"step": r["step"], "regime": r.get("sentinel_regime"),
             "matchup_hash": r.get("matchup_hash"),
             **{s: r["externals"][s]["counts"] for s in SPECS}} for r in rows]


def main():
    out = {"what": "§4.1 manipulation check, population loop round 2", "arms": {}}
    for k, run in ARMS.items():
        ser = series(run)
        assert len(ser) == 4, (run, len(ser))
        w = sum(r[s][0] for r in ser[-2:] for s in SPECS)
        n = sum(r[s][1] for r in ser[-2:] for s in SPECS)
        assert n == 600, n
        out["arms"][k] = {"run": run, "series": ser, "last_two_pooled": [w, n], "rate": w / n,
                          "curve_all_three": [sum(r[s][0] for s in SPECS) / 300 for r in ser]}
    (wb, nb), (wc, nc) = out["arms"]["B2"]["last_two_pooled"], out["arms"]["C2"]["last_two_pooled"]
    m, lo, hi = newcombe_diff_ci(wb, nb, wc, nc)
    out["M2"] = {"point": m, "ci95": [lo, hi], "ABSORBED": lo > 0}
    per = {}
    for s in SPECS:
        bw = sum(r[s][0] for r in out["arms"]["B2"]["series"][-2:])
        cw = sum(r[s][0] for r in out["arms"]["C2"]["series"][-2:])
        d, l2, h2 = newcombe_diff_ci(bw, 200, cw, 200)
        per[s] = {"B2": bw / 200, "C2": cw / 200, "B2_minus_C2": d, "ci95": [l2, h2]}
    out["per_specialist_descriptor"] = per
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("M2", "per_specialist_descriptor")}, indent=1))
    for k in ARMS:
        print(k, [(r["step"], *(r[s][0] for s in SPECS)) for r in out["arms"][k]["series"]],
              "curve", out["arms"][k]["curve_all_three"])


if __name__ == "__main__":
    main()
