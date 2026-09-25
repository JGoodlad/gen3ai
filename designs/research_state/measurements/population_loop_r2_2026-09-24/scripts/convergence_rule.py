"""Apply the round-2 registration's §4.4 CONVERGENCE SIDE-CHECK table — a DESCRIPTOR, never a verdict,
and NOT a re-verdict of round 1.

    python convergence_rule.py [--rbx RUN] [--rcx RUN] [--json OUT]

Reads four exploiter runs through `agents.training.best_response_gap.read_exploiter` (the meter's
own reader, so the vs-target series is exactly what the meter sees):
    RB  = ai_v13_24_popr1_read_loop   (round-1 reader of B, the loop)      — cycles 3-4
    RC  = ai_v13_25_popr1_read_ctrl   (round-1 reader of C, the control)   — cycles 3-4
    RB+ = ai_v13_31_popr1_read_loop_ext  (RB forked +50 %)                 — every post-fork cycle
    RC+ = ai_v13_32_popr1_read_ctrl_ext  (RC forked +50 %)                 — every post-fork cycle
and prints Δ+ = rate(RB+) − rate(RC+), Δ_late = [RB cycles 3-4 + RB+] − [RC cycles 3-4 + RC+], each
reader's rise over its parent's cycles 3-4, all with Newcombe 95 % intervals, and the outcome row
(S, B, C, I — read in that order, the first that applies governs). The row conditions are on POINT
estimates by registration (a descriptor); the intervals are printed beside them.

Run from the repo root with PYTHONPATH=src; --rbx / --rcx take stand-in run dirs for validation.
"""
import argparse
import json
import sys

from agents.training.best_response_gap import load_teamsets, newcombe_diff_ci, read_exploiter

BAR = 0.05       # 5 pp: the round's bar, reused as the descriptor's threshold
RISE = 0.05      # 5 pp: "still climbing"


def pooled(points):
    return sum(p.wins for p in points), sum(p.games for p in points)


def rate(wn):
    return wn[0] / wn[1]


def diff(a, b):
    _, lo, hi = newcombe_diff_ci(a[0], a[1], b[0], b[1])
    return {"delta": rate(a) - rate(b), "ci": [lo, hi], "a": list(a), "b": list(b)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--rb", default="ai_v13_24_popr1_read_loop")
    ap.add_argument("--rc", default="ai_v13_25_popr1_read_ctrl")
    ap.add_argument("--rbx", default="ai_v13_31_popr1_read_loop_ext")
    ap.add_argument("--rcx", default="ai_v13_32_popr1_read_ctrl_ext")
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    ts = load_teamsets()
    rb, rc, rbx, rcx = (read_exploiter(r, ts) for r in (a.rb, a.rc, a.rbx, a.rcx))

    # the pairing is the point: each extension continues ITS reader against the SAME target
    assert rbx.target_file == rb.target_file and rcx.target_file == rc.target_file, "target moved"
    assert rbx.budget == rcx.budget, f"extensions unmatched in budget: {rbx.budget} vs {rcx.budget}"

    rb34, rc34 = pooled(rb.post_fork[2:4]), pooled(rc.post_fork[2:4])
    rbx_p, rcx_p = pooled(rbx.post_fork), pooled(rcx.post_fork)
    late_b = (rb34[0] + rbx_p[0], rb34[1] + rbx_p[1])
    late_c = (rc34[0] + rcx_p[0], rc34[1] + rcx_p[1])

    d_plus = diff(rbx_p, rcx_p)
    d_late = diff(late_b, late_c)
    rise_b = diff(rbx_p, rb34)
    rise_c = diff(rcx_p, rc34)

    if d_late["delta"] <= -BAR and abs(rise_b["delta"]) <= RISE:
        row = "S — the Δ SURVIVES"
    elif rise_b["delta"] >= RISE and rise_c["delta"] >= RISE:
        row = "B — BOTH still climbing"
    elif rise_b["delta"] >= RISE and d_plus["delta"] > -BAR:
        row = "C — the Δ CLOSES"
    else:
        row = "I — INCONCLUSIVE"

    doc = {
        "descriptor_only": True,
        "not_a_reverdict_of_round_1": True,
        "runs": {"RB": rb.run, "RC": rc.run, "RB+": rbx.run, "RC+": rcx.run},
        "budget_ext": rbx.budget,
        "curves": {k: [[p.step, p.wins, p.games] for p in r.post_fork]
                   for k, r in (("RB", rb), ("RC", rc), ("RB+", rbx), ("RC+", rcx))},
        "rb_cycles_3_4": list(rb34), "rc_cycles_3_4": list(rc34),
        "delta_plus": d_plus, "delta_late": d_late,
        "rise_RB_plus_over_RB34": rise_b, "rise_RC_plus_over_RC34": rise_c,
        "outcome": row,
    }
    pct = lambda d: f"{100 * d['delta']:+.2f} pp [{100 * d['ci'][0]:+.2f}, {100 * d['ci'][1]:+.2f}]"
    print("CONVERGENCE SIDE-CHECK (round-2 registration §4.4) — DESCRIPTOR, NOT a re-verdict of round 1")
    for k, v in doc["curves"].items():
        print(f"  {k:4s} {doc['runs'][k]:34s} " + " / ".join(f"{w}/{g}" for _, w, g in v))
    print(f"  Δ+     = RB+ − RC+                    {pct(d_plus)}   (n {rbx_p[1]} / {rcx_p[1]})")
    print(f"  Δ_late = (RB c3-4 + RB+) − (RC c3-4 + RC+) {pct(d_late)}   (n {late_b[1]} / {late_c[1]})")
    print(f"  rise RB+ over RB cycles 3-4            {pct(rise_b)}")
    print(f"  rise RC+ over RC cycles 3-4            {pct(rise_c)}")
    print(f"  OUTCOME: {row}")
    if a.json:
        with open(a.json, "w") as fh:
            json.dump(doc, fh, indent=2)
        print(f"  wrote {a.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
