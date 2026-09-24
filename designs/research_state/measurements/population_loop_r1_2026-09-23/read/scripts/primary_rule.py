#!/usr/bin/env python3
"""§4.2 PRIMARY — apply the registered rule to Δ = gap(RB) − gap(RC).

RB = ai_v13_24_popr1_read_loop (the fresh offense exploiter that reads B, the loop generalist);
RC = ai_v13_25_popr1_read_ctrl (the one that reads C, the no-exploiter control).
Inputs: ``brgap_r1.json`` (the registered invocation with ``--rounds RC=2 RB=3``) and
``brgap_A2.json`` (A2 = ai_v13_26_popr0_exploit5_offense_s1002, A's seed-1002 replicate, read in
its OWN invocation — finding F3). Without brgap_A2.json the verdict is PENDING; nothing is issued.

The rule (registration §4.2, verbatim): F = |gap(A) − gap(A2)| (pooled); bar = max(F, 5.0 pp).
OUTSIDE iff |Δ| > bar AND the Newcombe CI excludes the bar point on that side. EQUIVALENT iff the
whole CI sits inside [−bar, +bar]. Otherwise NOT DETECTED.

Usage: primary_rule.py <read dir>   (writes <read dir>/primary_verdict.json)
"""
import json
import sys
from pathlib import Path

MIN_BAR_PP = 5.0


def gap_of(doc, run):
    for rnd in doc["rounds"]:
        for row in rnd["rows"]:
            if row["run"] == run:
                return row
    raise SystemExit(f"{run} not in the artifact")


def main():
    d = Path(sys.argv[1])
    r1 = json.loads((d / "brgap_r1.json").read_text())
    delta = next(x for x in r1["deltas"] if (x["earlier_round"], x["later_round"]) == (2, 3))
    row = delta["per_archetype"][0]
    assert row["runs_earlier"] == ["ai_v13_25_popr1_read_ctrl"], row
    assert row["runs_later"] == ["ai_v13_24_popr1_read_loop"], row
    D, lo, hi = 100 * row["delta"], 100 * row["lo"], 100 * row["hi"]
    gA = gap_of(r1, "ai_v13_18_teach5_offense_hidose")
    out = {"statistic": "gap(RB) - gap(RC), offense, pooled 400 games per reader, Newcombe 95%",
           "delta_pp": round(D, 2), "ci95_pp": [round(lo, 2), round(hi, 2)],
           "gap_A_pp": round(100 * gA["gap"], 2),
           "arithmetic_note": ("bar >= 5.0 always; OUTSIDE-below needs hi < -bar <= -5.0, "
                               f"and hi = {hi:+.2f}, so OUTSIDE is impossible for ANY F; "
                               f"EQUIVALENT would need bar >= {max(abs(lo), abs(hi)):.2f}")}
    a2p = d / "brgap_A2.json"
    if not a2p.exists():
        out["verdict"] = "PENDING — A2 has not finished; no verdict from an interim read"
    else:
        a2 = json.loads(a2p.read_text())
        gA2 = gap_of(a2, "ai_v13_26_popr0_exploit5_offense_s1002")
        F = abs(100 * gA["gap"] - 100 * gA2["gap"])
        bar = max(F, MIN_BAR_PP)
        if abs(D) > bar and ((D < 0 and hi < -bar) or (D > 0 and lo > bar)):
            v = "OUTSIDE, " + ("BELOW" if D < 0 else "ABOVE")
        elif -bar <= lo and hi <= bar:
            v = "EQUIVALENT"
        else:
            v = "NOT DETECTED"
        out.update({"gap_A2_pp": round(100 * gA2["gap"], 2), "A2_pooled": [gA2["wins"], gA2["games"]],
                    "A2_n_cycles": gA2["n_cycles"], "F_pp": round(F, 2), "bar_pp": round(bar, 2),
                    "verdict": v})
    (d / "primary_verdict.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
