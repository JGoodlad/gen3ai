"""Reviewer's POST-HOC read of crn_label_refit_2026-10-06's banked scores: the DECISION currency.

For each held-out CONTESTED turn: the policy's top1 / top2 (blob logits) and a hash-chosen random
alternative. For a scorer s, the pick among the option set is argmax_s. IMPROVEMENT(s) = mean over
turns of truth(pick) - truth(top1), truth = the mean of the 24 truth seeds (8-31), which no scorer
saw, so the estimate is UNBIASED (no max over noisy truth means). Win scale = (+-1 scale)/2.
Also: pairwise accuracy on the 3 option pairs (non-tied), and the bootstrap SE as a function of the
number of states (the pilot's power). Data as `pairing_read.py`; plus the committed `scores.npz`.

    python decision_read.py <repo root> <archive rows dir>   # design_value_discrimination.md §12"""
import gzip
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(sys.argv[1])
ARCH = Path(sys.argv[2])
M = REPO / "designs/research_state/measurements"
R = M / "crn_label_refit_2026-10-06/rows"
sc = np.load(R / "scores.npz")
with gzip.open(M / "m5_laneS/bank_v1/decisions.jsonl.gz", "rt") as f:
    decs = [json.loads(x) for x in f]
dec = {d["id"]: d for d in decs}
z = np.load(ARCH / "reads/blob.probs.npz")
assert str(z["ids_sha256"]) == hashlib.sha256("\n".join(d["id"] for d in decs).encode()).hexdigest()
logit = {d["id"]: z["logits"][i].astype(np.float64) for i, d in enumerate(decs)}
sub = json.loads((R / "subset_train.json").read_text())["ids"] + json.loads((R / "subset_held.json").read_text())["ids"]
gaps = {}
for i in sub:
    d = dec[i]
    if d["kind"] != "free" or not (2 <= d["turn"] <= 40) or d["n_legal"] < 3:
        continue
    m = np.array([c == "1" for c in d["mask"]])
    lg = np.sort(logit[i][m])[::-1]
    gaps[i] = lg[0] - lg[1]
thr = np.percentile(list(gaps.values()), 40)
cont = {i for i, g in gaps.items() if g < thr}

SC = ["BASE", "LEAF_K1", "LEAF_K4", "LEAF_K8", "REFIT_K8_s0"]
T = defaultdict(lambda: {"a": [], "truth": [], **{s: [] for s in SC}})
for r in range(len(sc["ids"])):
    did = str(sc["ids"][r])
    if did not in cont:
        continue
    e = T[did]
    e["battle"] = str(sc["battle"][r])
    e["a"].append(int(sc["action"][r]))
    e["truth"].append(float(sc["truth"][r].mean()))
    for s in SC:
        e[s].append(float(sc[s][r]))
recs = []
for did, e in sorted(T.items()):
    lg = logit[did]
    acts = e["a"]
    if len(acts) < 3:
        continue
    order = sorted(acts, key=lambda a: -lg[a])
    t1, t2 = order[0], order[1]
    rest = sorted(a for a in acts if a not in (t1, t2))
    rnd = rest[int(hashlib.sha256(f"review:{did}".encode()).hexdigest(), 16) % len(rest)]
    opts = [t1, t2, rnd]
    idx = {a: acts.index(a) for a in opts}
    tr = {a: e["truth"][idx[a]] for a in opts}
    rec = {"battle": e["battle"], "did": did}
    for s in SC:
        v = {a: e[s][idx[a]] for a in opts}
        # deterministic tie-break toward the policy's order (top1 first): a scorer tie keeps top1
        pick3 = max(opts, key=lambda a: (round(v[a], 9), -opts.index(a)))
        pick2 = max([t1, t2], key=lambda a: (round(v[a], 9), -[t1, t2].index(a)))
        rec[f"imp3_{s}"] = (tr[pick3] - tr[t1]) / 2
        rec[f"imp2_{s}"] = (tr[pick2] - tr[t1]) / 2
        acc = []
        for a, b in ((t1, t2), (t1, rnd), (t2, rnd)):
            dt = tr[a] - tr[b]
            if abs(dt) < 1e-9:
                continue
            ds = v[a] - v[b]
            acc.append(0.5 if abs(ds) < 1e-6 else float(np.sign(ds) == np.sign(dt)))
        rec[f"acc_{s}"] = acc
    rec["oracle3"] = (max(tr.values()) - tr[t1]) / 2  # winner's-curse biased UP (noisy max)
    recs.append(rec)


def boot(f, B=2000, seed=20261011, n_states=None):
    by = defaultdict(list)
    for p in recs:
        by[p["battle"]].append(p)
    keys = sorted(by)
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(B):
        if n_states is None:
            ks = rng.choice(len(keys), len(keys))
        else:  # resample battles until ~n_states states (power projection)
            ks, n = [], 0
            while n < n_states:
                k = rng.integers(len(keys))
                ks.append(k)
                n += len(by[keys[k]])
        out.append(f([p for k in ks for p in by[keys[k]]]))
    return np.percentile(out, [2.5, 97.5]).round(4).tolist(), round(float(np.std(out)), 4)


res = {"n_states": len(recs), "n_battles": len({p["battle"] for p in recs}), "thr": round(float(thr), 4)}
for s in SC:
    f3 = lambda S, s=s: float(np.mean([p[f"imp3_{s}"] for p in S]))
    f2 = lambda S, s=s: float(np.mean([p[f"imp2_{s}"] for p in S]))
    fa = lambda S, s=s: float(np.mean([x for p in S for x in p[f"acc_{s}"]]))
    res[s] = {"improvement_3opt_win": round(f3(recs), 4), "ci3": boot(f3)[0],
              "improvement_top12_win": round(f2(recs), 4), "ci2": boot(f2)[0],
              "acc_3pairs": round(fa(recs), 4), "acc_ci": boot(fa)[0],
              "n_pairs": sum(len(p[f"acc_{s}"]) for p in recs)}
res["oracle3_upper_biased"] = round(float(np.mean([p["oracle3"] for p in recs])), 4)
# contrasts, paired
for a, b in (("LEAF_K8", "BASE"), ("LEAF_K8", "LEAF_K1"), ("LEAF_K4", "LEAF_K1")):
    fd = lambda S, a=a, b=b: float(np.mean([p[f"imp3_{a}"] - p[f"imp3_{b}"] for p in S]))
    fda = lambda S, a=a, b=b: (float(np.mean([x for p in S for x in p[f"acc_{a}"]])) -
                               float(np.mean([x for p in S for x in p[f"acc_{b}"]])))
    res[f"{a}-{b}"] = {"imp3": round(fd(recs), 4), "imp3_ci": boot(fd)[0],
                       "acc": round(fda(recs), 4), "acc_ci_now": boot(fda)[0],
                       "acc_ci_at_500_states": boot(fda, n_states=500)[0],
                       "imp3_ci_at_500_states": boot(fd, n_states=500)[0]}
    fd2 = lambda S, a=a, b=b: float(np.mean([p[f"imp2_{a}"] - p[f"imp2_{b}"] for p in S]))
    res[f"{a}-{b}"].update({"imp2": round(fd2(recs), 4), "imp2_ci": boot(fd2)[0],
                            "imp2_ci_at_500_states": boot(fd2, n_states=500)[0]})
for s in SC:
    res[s] = {k: res[s][k] for k in ("improvement_3opt_win", "ci3", "improvement_top12_win", "ci2")}
print(json.dumps(res))
