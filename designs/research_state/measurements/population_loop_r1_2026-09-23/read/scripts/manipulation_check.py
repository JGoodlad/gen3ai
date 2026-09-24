#!/usr/bin/env python3
"""§4.1 MANIPULATION CHECK — did B (the loop generalist) absorb the specialists it was shown?

M = [B's pooled greedy rate over its LAST TWO eval cycles against both specialists]
  − [C's (the no-exploiter control's) same], n = 400 per arm, Newcombe 95 %.
ABSORBED iff M's CI lower bound > 0. Read from each run's own ``eval_results.jsonl`` ``externals``
(100 greedy-vs-greedy games per specialist per cycle, the eval regime; ``sentinel_regime`` is
recorded beside it). The specialists: A = ai_v13_18_teach5_offense_hidose, A13 = ai_v13_13_exploit5_offense.

Run with PYTHONPATH containing src (the Newcombe helper is the meter's own,
``agents.training.best_response_gap.newcombe_diff_ci``).  Usage: manipulation_check.py <out.json>
"""
import json
import sys
from pathlib import Path

from agents.training.best_response_gap import newcombe_diff_ci

MODELS = Path("/home/goodlad/dev/gen3ai/models")
ARMS = {"B": "ai_v13_22_popr1_loop", "C": "ai_v13_23_popr1_ctrl"}
SPECS = ["ext_ai_v13_18_teach5_offense_hidose", "ext_ai_v13_13_exploit5_offense"]
G0_RATES = {"ext_ai_v13_18_teach5_offense_hidose": 0.360, "ext_ai_v13_13_exploit5_offense": 0.3425}


def series(run):
    rows = [json.loads(l) for l in open(MODELS / run / "eval_results.jsonl")]
    return [{"step": r["step"], "regime": r.get("sentinel_regime"),
             "matchup_hash": r.get("matchup_hash"),
             **{s: r["externals"][s]["counts"] for s in SPECS}} for r in rows]


def last_two(ser):
    w = n = 0
    for r in ser[-2:]:
        for s in SPECS:
            w += r[s][0]
            n += r[s][1]
    return w, n


def main():
    out = {"what": "§4.1 manipulation check, population loop round 1", "arms": {}}
    for k, run in ARMS.items():
        ser = series(run)
        w, n = last_two(ser)
        out["arms"][k] = {"run": run, "series": ser, "last_two_pooled": [w, n], "rate": w / n}
    (wb, nb), (wc, nc) = out["arms"]["B"]["last_two_pooled"], out["arms"]["C"]["last_two_pooled"]
    m, lo, hi = newcombe_diff_ci(wb, nb, wc, nc)
    out["M"] = {"point": m, "ci95": [lo, hi], "ABSORBED": lo > 0}
    per = {}
    for s in SPECS:
        bw = sum(r[s][0] for r in out["arms"]["B"]["series"][-2:])
        cw = sum(r[s][0] for r in out["arms"]["C"]["series"][-2:])
        d, l2, h2 = newcombe_diff_ci(bw, 200, cw, 200)
        per[s] = {"B": bw / 200, "C": cw / 200, "G0": G0_RATES[s], "B_minus_C": d, "ci95": [l2, h2]}
    out["per_specialist_descriptor"] = per
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: out[k] for k in ("M", "per_specialist_descriptor")}, indent=1))
    for k in ARMS:
        print(k, [(r["step"], r[SPECS[0]][0], r[SPECS[1]][0]) for r in out["arms"][k]["series"]])


if __name__ == "__main__":
    main()
