"""Diff two `one_update.py` dumps: per-parameter max |Δ| (and relative to the update's own size),
and every logged scalar's |Δ|. Usage: compare.py <legacy.npz> <new.npz> [--json out.json]"""
import json
import sys

import numpy as np

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
la, lb = (json.load(open(p + ".json")) for p in sys.argv[1:3])
worst = []
for k in a.files:
    d = float(np.max(np.abs(a[k] - b[k]))) if a[k].size else 0.0
    worst.append((d, k))
worst.sort(reverse=True)
scal = []
for k in sorted(set(la) | set(lb)):
    if k.endswith("_ms") or k.startswith("time/"):
        continue
    va, vb = la.get(k), lb.get(k)
    if va is None or vb is None:
        scal.append((float("inf"), k, va, vb))
        continue
    d = abs(va - vb)
    if not (d == d):
        d = 0.0 if (va != va and vb != vb) else float("inf")
    scal.append((d / max(1.0, abs(va)), k, va, vb))
scal.sort(reverse=True)
res = {"param_max_abs": worst[0][0], "param_worst": worst[:8],
       "scalar_worst_rel": scal[:15], "n_params": len(a.files), "n_scalars": len(scal),
       "missing_scalars": [s[1] for s in scal if s[0] == float("inf")]}
print(json.dumps(res, indent=1, default=str))
if "--json" in sys.argv:
    json.dump(res, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1, default=str)
