"""PHASE 2b's READ (Amendment 5): belief worlds (``det_b``) vs Smogon worlds (``det``) vs the oracle, on the states
every arm played, against Phase 1's TRUE-world truth. R-HONEST-B and R-BELIEF.

    python -I score_belief.py <cp> [<cp> ...] [--out-json result_belief.json] [--out-md result_belief.md]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score as S1  # noqa: E402
import score_det as SD  # noqa: E402

KEYS = ("ARGMAX|CRITIC", "ARGMAX|LEAF-K1", "ARGMAX|LEAF-K2", "ARGMAX|LEAF-K4", "ARGMAX|LEAF-K8",
        "ARGMAX|TRUNC-4-K1", "ARGMAX|TRUNC-4-K4", "ARGMAX|TRUNC-4-K8", "ARGMAX|MIX-K8", "GUMBEL|LEAF-K8",
        "SH-all|B=n*2")


def per_state(cp, det_dir, ids_filter=None):
    SD.DET_DIR = det_dir
    st, r1, _, _ = S1.load(cp, True)
    rd = SD.det_rows(cp)
    E = st["states"]
    ids = sorted(i for i in E if r1.get(i, {}).get("ok") and rd.get(i, {}).get("ok"))
    if ids_filter is not None:
        ids = [i for i in ids if i in ids_filter]
    out, oracle = {}, {}
    for did in ids:
        So, Sd = SD.det_arrays(r1[did], rd[did], E[did])
        do, _ = S1.decisions(So)
        n16 = S1.N_LEAF
        S1.N_LEAF = Sd["W"].shape[1]
        try:
            dd, _ = S1.decisions(Sd)
        finally:
            S1.N_LEAF = n16
        T, t1 = So["truth"], So["top1"]
        out[did] = {k: T[dd[k]] - T[t1] for k in KEYS}
        oracle[did] = {k: T[do[k]] - T[t1] for k in KEYS}
    return st, out, oracle


def read(cp):
    st, B, O = per_state(cp, "det_b")
    _, A, _ = per_state(cp, "det", set(B))
    ids = sorted(set(A) & set(B))
    E = st["states"]
    bt = S1.Boot([E[i]["battle"] for i in ids], [1.0] * len(ids))
    arr = lambda D, k: np.array([D[i][k] for i in ids])  # noqa: E731
    res = {"cp": cp, "n_states": len(ids), "n_battles": len(bt.ub), "imp": {}, "contrasts": {}, "strata": {}}
    for k in KEYS:
        res["imp"][k] = {"belief": bt.stat(arr(B, k)), "smogon": bt.stat(arr(A, k)), "oracle": bt.stat(arr(O, k))}
    res["contrasts"]["R-HONEST-B: det-B LEAF-K8 - det-B CRITIC"] = bt.stat(arr(B, "ARGMAX|LEAF-K8") - arr(B, "ARGMAX|CRITIC"))
    res["contrasts"]["R-BELIEF: det-B LEAF-K8 - det-A LEAF-K8"] = bt.stat(arr(B, "ARGMAX|LEAF-K8") - arr(A, "ARGMAX|LEAF-K8"))
    res["contrasts"]["oracle - det-B LEAF-K8"] = bt.stat(arr(O, "ARGMAX|LEAF-K8") - arr(B, "ARGMAX|LEAF-K8"))
    res["contrasts"]["det-B TRUNC-4-K4 - det-B CRITIC"] = bt.stat(arr(B, "ARGMAX|TRUNC-4-K4") - arr(B, "ARGMAX|CRITIC"))
    res["contrasts"]["det-B CRITIC - det-A CRITIC"] = bt.stat(arr(B, "ARGMAX|CRITIC") - arr(A, "ARGMAX|CRITIC"))
    for lab, lo, hi in (("unrev_0-1", 0, 1), ("unrev_2-3", 2, 3), ("unrev_4-5", 4, 5)):
        m = np.array([lo <= E[i]["unrevealed"] <= hi for i in ids])
        if m.sum():
            res["strata"][lab] = {s: bt.stat(arr(D, "ARGMAX|LEAF-K8"), sel=m) for s, D in
                                  (("belief", B), ("smogon", A), ("oracle", O))}
    c = res["contrasts"]
    lo, hi = c["R-HONEST-B: det-B LEAF-K8 - det-B CRITIC"]["ci"]
    v = {"R-HONEST-B": "LEAF-AHEAD" if lo > S1.EDGE else ("CRITIC-EQUIVALENT" if lo > -0.01 and hi < 0.01 else "NOT DETECTED"),
         "R-BELIEF": S1.verdict3(c["R-BELIEF: det-B LEAF-K8 - det-A LEAF-K8"]["ci"])}
    res["verdicts"] = v
    return res


def md(results):
    L = ["# Value-discrimination pilot, PHASE 2b (belief worlds): result tables", "",
         "IMP against the TRUE world's 24 truth seeds, on the states the belief, Smogon and oracle arms all played.", ""]
    for r in results:
        L += [f"## {r['cp']} ({r['n_states']} states, {r['n_battles']} battles)", "",
              "Verdicts: " + ", ".join(f"{k} **{x}**" for k, x in r["verdicts"].items()), "",
              "| scorer | belief worlds | Smogon worlds | oracle |", "|---|---|---|---|"]
        for k, d in r["imp"].items():
            L.append(f"| {k} | " + " | ".join(f"{d[s]['mean']:+.4f} [{d[s]['ci'][0]:+.4f}, {d[s]['ci'][1]:+.4f}]"
                                              for s in ("belief", "smogon", "oracle")) + " |")
        L += ["", "| contrast | Δ [95 %] |", "|---|---|"]
        for k, x in r["contrasts"].items():
            L.append(f"| {k} | {x['mean']:+.4f} [{x['ci'][0]:+.4f}, {x['ci'][1]:+.4f}] |")
        L += ["", "| unrevealed slots | n | belief LEAF-K8 | Smogon LEAF-K8 | oracle LEAF-K8 |", "|---|---|---|---|---|"]
        for s, d in r["strata"].items():
            L.append(f"| {s} | {d['belief']['n']} | {d['belief']['mean']:+.4f} | {d['smogon']['mean']:+.4f} | {d['oracle']['mean']:+.4f} |")
        L.append("")
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cps", nargs="+")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-md", default=None)
    a = ap.parse_args()
    res = [read(c) for c in a.cps]
    if a.out_json:
        Path(a.out_json).write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")
    if a.out_md:
        Path(a.out_md).write_text(md(res) + "\n")


if __name__ == "__main__":
    main()
