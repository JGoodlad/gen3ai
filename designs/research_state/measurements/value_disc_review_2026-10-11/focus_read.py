"""Reviewer's POST-HOC read: does the policy's top-2 logit gap ('torn') predict where a playout leaf
improves the decision? Held-out ELIGIBLE turns of crn_label_refit_2026-10-06; improvement of the
scorer's pick among ALL legal actions over the policy's top1, against the 24 truth seeds (unbiased),
on the WIN scale. Data as `decision_read.py`.

    python focus_read.py <repo root> <archive rows dir>   # design_value_discrimination.md §12"""
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO, ARCH = Path(sys.argv[1]), Path(sys.argv[2])
M = REPO / "designs/research_state/measurements"
R = M / "crn_label_refit_2026-10-06/rows"
sc = np.load(R / "scores.npz")
with gzip.open(M / "m5_laneS/bank_v1/decisions.jsonl.gz", "rt") as f:
    decs = [json.loads(x) for x in f]
dec = {d["id"]: d for d in decs}
z = np.load(ARCH / "reads/blob.probs.npz")
assert str(z["ids_sha256"]) == hashlib.sha256("\n".join(d["id"] for d in decs).encode()).hexdigest()
logit = {d["id"]: z["logits"][i].astype(np.float64) for i, d in enumerate(decs)}
T = defaultdict(lambda: {"a": [], "truth": [], "LEAF_K8": [], "LEAF_K1": [], "BASE": []})
for r in range(len(sc["ids"])):
    did = str(sc["ids"][r])
    e = T[did]
    e["battle"] = str(sc["battle"][r])
    e["a"].append(int(sc["action"][r]))
    e["truth"].append(float(sc["truth"][r].mean()) / 2)
    for s in ("LEAF_K8", "LEAF_K1", "BASE"):
        e[s].append(float(sc[s][r]))
rows = []
for did, e in T.items():
    d = dec[did]
    if d["kind"] != "free" or not (2 <= d["turn"] <= 40) or d["n_legal"] < 3:
        continue
    m = np.array([c == "1" for c in d["mask"]])
    lg = logit[did]
    acts = e["a"]
    t1 = max(acts, key=lambda a: lg[a])
    srt = np.sort(lg[m])[::-1]
    gap = srt[0] - srt[1]
    tr = dict(zip(acts, e["truth"]))
    rec = {"gap": gap, "battle": e["battle"], "spread": max(tr.values()) - min(tr.values())}
    for s in ("LEAF_K8", "LEAF_K1", "BASE"):
        v = dict(zip(acts, e[s]))
        pick = max(acts, key=lambda a: (round(v[a], 9), a == t1))
        rec[s] = tr[pick] - tr[t1]
    rows.append(rec)
g = np.array([r["gap"] for r in rows])
qs = np.quantile(g, [0.2, 0.4, 0.6, 0.8])
out = {"n_turns": len(rows), "gap_quintile_edges": qs.round(3).tolist(), "by_quintile": []}
for k in range(5):
    lo = -np.inf if k == 0 else qs[k - 1]
    hi = np.inf if k == 4 else qs[k]
    sel = [r for r in rows if lo <= r["gap"] < hi] if k < 4 else [r for r in rows if r["gap"] >= lo]
    out["by_quintile"].append({
        "q": k + 1, "n": len(sel),
        "imp_LEAF_K8": round(float(np.mean([r["LEAF_K8"] for r in sel])), 4),
        "imp_LEAF_K1": round(float(np.mean([r["LEAF_K1"] for r in sel])), 4),
        "imp_BASE": round(float(np.mean([r["BASE"] for r in sel])), 4),
        "truth_spread_mean": round(float(np.mean([r["spread"] for r in sel])), 4)})
print(json.dumps(out))


def bci(sel, key, B=2000, seed=20261011):
    by = defaultdict(list)
    for r in sel:
        by[r["battle"]].append(r)
    keys = sorted(by)
    rng = np.random.default_rng(seed)
    v = []
    for _ in range(B):
        ks = rng.choice(len(keys), len(keys))
        v.append(np.mean([key(r) for k in ks for r in by[keys[k]]]))
    return np.percentile(v, [2.5, 97.5]).round(4).tolist()


q12 = [r for r in rows if r["gap"] < qs[1]]
q5 = [r for r in rows if r["gap"] >= qs[3]]
mean = lambda S, k: round(float(np.mean([r[k] for r in S])), 4)
print(json.dumps({
    "all_K8": [mean(rows, "LEAF_K8"), bci(rows, lambda r: r["LEAF_K8"])],
    "all_K1": [mean(rows, "LEAF_K1"), bci(rows, lambda r: r["LEAF_K1"])],
    "all_BASE": [mean(rows, "BASE"), bci(rows, lambda r: r["BASE"])],
    "all_K8_minus_BASE": bci(rows, lambda r: r["LEAF_K8"] - r["BASE"]),
    "all_K8_minus_K1": bci(rows, lambda r: r["LEAF_K8"] - r["LEAF_K1"]),
    "torn40_K8": [mean(q12, "LEAF_K8"), bci(q12, lambda r: r["LEAF_K8"]), len(q12)],
    "top20gap_K8": [mean(q5, "LEAF_K8"), bci(q5, lambda r: r["LEAF_K8"]), len(q5)]}))
