"""X65 ExIt-0's delivery gate BY HAND (REVIEW R-M1; PREREG §5): recompute every panel's Δagree, the pooled Δ, its LB and
DELIVERED from the three score files `python -m main.x65.delivery score` wrote, with loops of its own that share no
code with `main.x65.delivery`, and compare them with the gate's file.

    python3 delivery_by_hand.py <scores dir> <delivery.json> > delivery_by_hand.json      # exit 0 = agrees

Only the bootstrap's index DRAW is shared with the gate, by construction: `numpy.random.default_rng(20261105)`, B = 4,000
replicates, each drawing `integers(0, n_deals, n_deals)` per panel in the order heldout72_hpfix, forum72, p72, over the
panel's deals in sorted order. Everything computed from the draw is a Python loop. The LB is the 2.5 % percentile by
linear interpolation between order statistics (numpy's default), written out here. A LB within 0.0005 of 0 is never
DELIVERED (root rule 8), compared by exact fractions.
"""
import json
import sys
from fractions import Fraction

import numpy as np

PANELS = ("heldout72_hpfix", "forum72", "p72")
B, SEED, EPS = 4000, 20261105, Fraction("0.0005")


def panel_sums(path):
    z = np.load(path)
    per = {}
    for deal, a_s, a_p in zip(z["deal"].tolist(), z["a_student"].tolist(), z["a_pstar"].tolist()):
        s, n = per.get(deal, (0, 0))
        per[deal] = (s + int(a_s) - int(a_p), n + 1)
    deals = sorted(per)
    return [per[d][0] for d in deals], [per[d][1] for d in deals]


def percentile(xs, q):
    v = sorted(xs)
    h = (len(v) - 1) * q / 100.0
    lo = int(h)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (h - lo) * (v[hi] - v[lo])


def recompute(scores):
    sums = {p: panel_sums(f"{scores}/{p}.npz") for p in PANELS}
    delta = {p: sum(s) / sum(n) for p, (s, n) in sums.items()}
    pooled = sum(delta[p] for p in PANELS) / len(PANELS)
    rng = np.random.default_rng(SEED)
    reps = []
    for _ in range(B):
        vals = []
        for p in PANELS:
            s, n = sums[p]
            ix = rng.integers(0, len(s), len(s)).tolist()
            vals.append(sum(s[i] for i in ix) / sum(n[i] for i in ix))
        reps.append(sum(vals) / len(vals))
    lb = percentile(reps, 2.5)
    d = Fraction(lb)
    near = abs(d) <= EPS
    return {"delta": delta, "pooled": pooled, "lb": lb, "delivered": (not near) and d > 0, "boundary": near}


def compare(mine, gate, tol=1e-9):
    bad = []
    for p in PANELS:
        if abs(mine["delta"][p] - gate["per_panel"][p]["delta"]) > tol:
            bad.append(f"{p}: delta {mine['delta'][p]} vs the gate's {gate['per_panel'][p]['delta']}")
    for k in ("pooled", "lb"):
        if abs(mine[k] - gate[k]) > tol:
            bad.append(f"{k}: {mine[k]} vs the gate's {gate[k]}")
    # a LB within the tolerance of the boundary itself is not a disagreement the float sums can settle (rule 8)
    if abs(abs(mine["lb"]) - float(EPS)) > tol:
        for k in ("delivered", "boundary"):
            if mine[k] != gate[k]:
                bad.append(f"{k}: {mine[k]} vs the gate's {gate[k]}")
    return bad


def main(argv):
    scores, gate_path = argv[1], argv[2]
    mine = recompute(scores)
    with open(gate_path) as fh:
        bad = compare(mine, json.load(fh))
    print(json.dumps({"by_hand": mine, "agrees": not bad, "problems": bad}, indent=1))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
