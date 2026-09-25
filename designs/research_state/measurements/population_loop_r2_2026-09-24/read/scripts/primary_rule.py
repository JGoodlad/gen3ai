#!/usr/bin/env python3
"""Round 2 §4.2 PRIMARY — apply the registered rule to Δ2 = gap(RB2) − gap(RC2).

RB2 = ai_v13_29_popr2_read_loop (fresh offense reader of B2, the round-2 loop generalist);
RC2 = ai_v13_30_popr2_read_ctrl (fresh offense reader of C2, the round-2 control).
Input: ``brgap_r2.json`` (the registered invocation with ``--rounds RC2=2 RB2=3``).
The bar is REUSED (registration §2): F = 2.00 pp (round 1: |gap(A) − gap(A2)|), bar = max(F, 5.0) = 5.0.
Rule (§4.2, verbatim): OUTSIDE iff |Δ2| > bar AND the Newcombe CI excludes the bar point on that side.
EQUIVALENT iff the whole CI sits inside [−bar, +bar]. Otherwise NOT DETECTED.
Also: each reader's convergence descriptor (pooled cycles 3–4 minus cycles 1–2; a rise >= 5 pp on one
reader and not the other is a named caveat).

Usage: primary_rule.py <read dir>   (writes <read dir>/primary_verdict.json)
"""
import json
import sys
from pathlib import Path

F_PP = 2.00
BAR_PP = max(F_PP, 5.0)
RB2, RC2 = "ai_v13_29_popr2_read_loop", "ai_v13_30_popr2_read_ctrl"


def main():
    d = Path(sys.argv[1])
    r2 = json.loads((d / "brgap_r2.json").read_text())
    delta = next(x for x in r2["deltas"] if (x["earlier_round"], x["later_round"]) == (2, 3))
    row = delta["per_archetype"][0]
    assert row["runs_earlier"] == [RC2], row
    assert row["runs_later"] == [RB2], row
    assert not r2["mismatches"] and not r2["unmatched"], (r2["mismatches"], r2["unmatched"])
    D, lo, hi = 100 * row["delta"], 100 * row["lo"], 100 * row["hi"]
    if abs(D) > BAR_PP and ((D < 0 and hi < -BAR_PP) or (D > 0 and lo > BAR_PP)):
        v = "OUTSIDE, " + ("BELOW" if D < 0 else "ABOVE")
    elif -BAR_PP <= lo and hi <= BAR_PP:
        v = "EQUIVALENT"
    else:
        v = "NOT DETECTED"
    conv = {}
    for run in r2["runs"]:
        s = [c for c in run["series"] if c["post_fork"]]
        assert len(s) == 4, (run["run"], len(s))
        early = sum(c["wins"] for c in s[:2]) / sum(c["games"] for c in s[:2])
        late = sum(c["wins"] for c in s[2:]) / sum(c["games"] for c in s[2:])
        conv[run["run"]] = {"curve": [c["win_rate"] for c in s], "cycles_1_2": early,
                            "cycles_3_4": late, "rise_pp": round(100 * (late - early), 2)}
    rises = {k: v2["rise_pp"] >= 5.0 for k, v2 in conv.items() if k in (RB2, RC2)}
    out = {"statistic": "gap(RB2) - gap(RC2), offense, pooled 400 games per reader, Newcombe 95%",
           "delta_pp": round(D, 2), "ci95_pp": [round(lo, 2), round(hi, 2)],
           "F_pp_reused": F_PP, "bar_pp": BAR_PP, "verdict": v,
           "clause1_abs_gt_bar": abs(D) > BAR_PP,
           "clause2_ci_excludes_bar_point": (hi < -BAR_PP) if D < 0 else (lo > BAR_PP),
           "gaps_pp": {rr["run"]: round(100 * rr["gap"], 2) for rnd in r2["rounds"] for rr in rnd["rows"]},
           "convergence_descriptor": conv,
           "convergence_caveat": (rises[RB2] != rises[RC2]),
           }
    (d / "primary_verdict.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
