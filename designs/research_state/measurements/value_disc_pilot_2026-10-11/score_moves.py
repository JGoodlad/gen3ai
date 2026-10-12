"""PHASE 2c's READ (Amendment 6): MOVE-belief worlds (``det_c``) and the true-seen-moves ceiling (``det_ct``) beside the
belief-species (``det_b``), Smogon (``det``) and oracle worlds, on the states EVERY arm played, against Phase 1's
TRUE-world truth. R-MOVES, R-HONEST-C, R-MOVES-CEILING, and the closed share of the oracle − honest gap.

    python -I score_moves.py <cp> [<cp> ...] [--out-json result_moves.json] [--out-md result_moves.md]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as S1  # noqa: E402
import score_belief as SB  # noqa: E402

KEYS = ("ARGMAX|CRITIC", "ARGMAX|LEAF-K1", "ARGMAX|LEAF-K4", "ARGMAX|LEAF-K8", "ARGMAX|TRUNC-4-K1",
        "ARGMAX|TRUNC-4-K4", "ARGMAX|TRUNC-4-K8", "ARGMAX|MIX-K8")
ARMS = (("C (net moves)", "det_c"), ("CT (true seen moves)", "det_ct"), ("B (belief species)", "det_b"),
        ("A (Smogon)", "det"))
#: Amendment 8: the same arms built by the fixed (v2) builder
ARMS_V2 = (("C (net moves)", "det2_c"), ("CT (true seen moves)", "det2_ct"), ("B (belief species)", "det2_b"),
           ("A (Smogon)", "det2"))


def read(cp):
    SB.KEYS = KEYS
    per, oracle, st = {}, None, None
    for lab, d in ARMS:
        st, per[lab], o = SB.per_state(cp, d)
        oracle = o if oracle is None else {**oracle, **o}
    ids = sorted(set.intersection(*[set(v) for v in per.values()]))
    E = st["states"]
    bt = S1.Boot([E[i]["battle"] for i in ids], [1.0] * len(ids))
    arr = lambda D, k: np.array([D[i][k] for i in ids])  # noqa: E731
    C, CT, B, A = (per[x] for x, _ in ARMS)
    res = {"cp": cp, "n_states": len(ids), "n_battles": len(bt.ub),
           "n_played": {lab: len(per[lab]) for lab, _ in ARMS}, "imp": {}, "contrasts": {}, "closed_share": {},
           "strata": {}}
    for k in KEYS:
        res["imp"][k] = {**{lab: bt.stat(arr(per[lab], k)) for lab, _ in ARMS}, "oracle": bt.stat(arr(oracle, k))}
    L8 = "ARGMAX|LEAF-K8"
    con = {"R-MOVES: C LEAF-K8 - B LEAF-K8": arr(C, L8) - arr(B, L8),
           "R-HONEST-C: C LEAF-K8 - C CRITIC": arr(C, L8) - arr(C, "ARGMAX|CRITIC"),
           "R-MOVES-CEILING: CT LEAF-K8 - C LEAF-K8": arr(CT, L8) - arr(C, L8),
           "C LEAF-K8 - A LEAF-K8": arr(C, L8) - arr(A, L8),
           "R-HONEST-B: B LEAF-K8 - B CRITIC": arr(B, L8) - arr(B, "ARGMAX|CRITIC"),
           "R-BELIEF: B LEAF-K8 - A LEAF-K8": arr(B, L8) - arr(A, L8),
           "CT LEAF-K8 - CT CRITIC": arr(CT, L8) - arr(CT, "ARGMAX|CRITIC"),
           "oracle LEAF-K8 - CT LEAF-K8": arr(oracle, L8) - arr(CT, L8),
           "oracle LEAF-K8 - C LEAF-K8": arr(oracle, L8) - arr(C, L8),
           "C CRITIC - B CRITIC": arr(C, "ARGMAX|CRITIC") - arr(B, "ARGMAX|CRITIC"),
           "C TRUNC-4-K4 - C CRITIC": arr(C, "ARGMAX|TRUNC-4-K4") - arr(C, "ARGMAX|CRITIC"),
           "CT TRUNC-4-K4 - CT CRITIC": arr(CT, "ARGMAX|TRUNC-4-K4") - arr(CT, "ARGMAX|CRITIC")}
    for k, v in con.items():
        res["contrasts"][k] = bt.stat(v)
    # the closed share of the oracle − B gap at LEAF-K8: (X − B) / (oracle − B), ratio of means on the same draws
    nb = len(bt.ub)
    sb = lambda x: np.bincount(bt.bi, weights=x, minlength=nb)  # noqa: E731
    b_, o_ = sb(arr(B, L8)), sb(arr(oracle, L8))
    for lab, D in (("C", C), ("CT", CT)):
        x_ = sb(arr(D, L8))
        est = (x_.sum() - b_.sum()) / (o_.sum() - b_.sum())
        bs = (bt.C @ x_ - bt.C @ b_) / (bt.C @ o_ - bt.C @ b_)
        lo, hi = np.percentile(bs, [2.5, 97.5])
        res["closed_share"][lab] = {"mean": round(float(est), 4), "ci": [round(float(lo), 4), round(float(hi), 4)]}
    for lab_, lo, hi in (("unrev_0-1", 0, 1), ("unrev_2-3", 2, 3), ("unrev_4-5", 4, 5)):
        m = np.array([lo <= E[i]["unrevealed"] <= hi for i in ids])
        if m.sum():
            res["strata"][lab_] = {**{lab: bt.stat(arr(per[lab], L8), sel=m) for lab, _ in ARMS},
                                   "oracle": bt.stat(arr(oracle, L8), sel=m)}
    c = res["contrasts"]
    lo, hi = c["R-HONEST-C: C LEAF-K8 - C CRITIC"]["ci"]
    res["verdicts"] = {
        "R-MOVES": S1.verdict3(c["R-MOVES: C LEAF-K8 - B LEAF-K8"]["ci"]),
        "R-HONEST-C": "LEAF-AHEAD" if lo > S1.EDGE else ("CRITIC-EQUIVALENT" if lo > -0.01 and hi < 0.01 else "NOT DETECTED"),
        "R-MOVES-CEILING": S1.verdict3(c["R-MOVES-CEILING: CT LEAF-K8 - C LEAF-K8"]["ci"]),
        "R-BELIEF": S1.verdict3(c["R-BELIEF: B LEAF-K8 - A LEAF-K8"]["ci"])}
    lo_b, hi_b = c["R-HONEST-B: B LEAF-K8 - B CRITIC"]["ci"]
    res["verdicts"]["R-HONEST-B"] = ("LEAF-AHEAD" if lo_b > S1.EDGE else
                                     ("CRITIC-EQUIVALENT" if lo_b > -0.01 and hi_b < 0.01 else "NOT DETECTED"))
    return res


def md(results):
    f = lambda d: f"{d['mean']:+.4f} [{d['ci'][0]:+.4f}, {d['ci'][1]:+.4f}]"  # noqa: E731
    labs = [x for x, _ in ARMS] + ["oracle"]
    L = ["# Value-discrimination pilot, PHASE 2c (move-belief worlds) and 2c-T (true seen moves): result tables", "",
         "IMP against the TRUE world's 24 truth seeds, on the states every world kind played.", ""]
    for r in results:
        L += [f"## {r['cp']} ({r['n_states']} common states, {r['n_battles']} battles; played per arm: "
              + ", ".join(f"{k} {v}" for k, v in r["n_played"].items()) + ")", "",
              "Verdicts: " + ", ".join(f"{k} **{x}**" for k, x in r["verdicts"].items()), "",
              "| scorer | " + " | ".join(labs) + " |", "|---" * (len(labs) + 1) + "|"]
        for k, d in r["imp"].items():
            L.append(f"| {k} | " + " | ".join(f(d[s]) for s in labs) + " |")
        L += ["", "| contrast | Δ [95 %] |", "|---|---|"]
        for k, x in r["contrasts"].items():
            L.append(f"| {k} | {f(x)} |")
        L += ["", "| closed share of the oracle − B gap (LEAF-K8) | share [95 %] |", "|---|---|"]
        for k, x in r["closed_share"].items():
            L.append(f"| {k} | {x['mean']:+.3f} [{x['ci'][0]:+.3f}, {x['ci'][1]:+.3f}] |")
        L += ["", "| unrevealed slots | n | " + " | ".join(labs) + " (LEAF-K8) |", "|---" * (len(labs) + 2) + "|"]
        for s, d in r["strata"].items():
            L.append(f"| {s} | {d['oracle']['n']} | " + " | ".join(f"{d[x]['mean']:+.4f}" for x in labs) + " |")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cps", nargs="+")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    ap.add_argument("--v2", action="store_true", help="the fixed builder's worlds (Amendment 8)")
    a = ap.parse_args()
    global ARMS
    if a.v2:
        ARMS = ARMS_V2
    res = [read(c) for c in a.cps]
    if a.out_json:
        Path(a.out_json).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    if a.out_md:
        Path(a.out_md).write_text(md(res) + "\n")


if __name__ == "__main__":
    main()
