"""EXACT permutation p for the arm-ORDERING agreement with the published off-slice KL.

Five arms -> 120 relabellings, enumerated in full. p = P(Spearman rho >= observed) under a uniform
random relabelling of the five arms. This is the only inference the ordering claim gets: five
points is five points, and a rho quoted without its 1-in-120 scale invites over-reading.
"""
import itertools
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJ = json.load(open(os.path.join(HERE, "projection.json")))
DISP = json.load(open(os.path.join(HERE, "displacement.json")))


def rho(a, b):
    def r(x):
        o = np.argsort(np.asarray(x, float), kind="mergesort")
        q = np.empty(len(x))
        q[o] = np.arange(len(x))
        return q
    return float(np.corrcoef(r(a), r(b))[0, 1])


out = {}
for sl in ("untaught", "taught"):
    o = PROJ["slices"][sl]["ordering"]
    pub = np.array(o["published"])
    cols = {"|Delta_theta|": [DISP["arms"][k]["l2"] for k in o["arms"]],
            "KL1 (first order)": o["kl1"],
            "actual KL(parent||arm)": o["actual"]}
    out[sl] = {}
    print(f"=== {sl}   arms {o['arms']}")
    for name, v in cols.items():
        obs = rho(v, pub)
        null = [rho(v, [pub[i] for i in p]) for p in itertools.permutations(range(5))]
        p = float(np.mean(np.array(null) >= obs - 1e-12))
        out[sl][name] = dict(rho=obs, exact_p_one_sided=p, n_perm=len(null))
        print(f"  {name:24s} rho={obs:+.3f}   exact one-sided p = {p:.4f}  "
              f"({int(round(p*120))}/120)")
with open(os.path.join(HERE, "ordering_perm.json"), "w") as f:
    json.dump(out, f, indent=1)
print("wrote ordering_perm.json")
