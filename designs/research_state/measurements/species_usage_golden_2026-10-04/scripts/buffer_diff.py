"""Per-field diff of two golden buffers: byte-identical, or how many elements moved and by how much."""
import json
import sys

import numpy as np

a, b = np.load(sys.argv[1]), np.load(sys.argv[2])
out = {}
for k in sorted(set(a.files) | set(b.files)):
    x, y = a[k], b[k]
    if x.shape != y.shape:
        out[k] = f"shape {x.shape} -> {y.shape}"
    elif x.tobytes() == y.tobytes():
        out[k] = "identical"
    else:
        d = np.abs(x.astype(np.float64) - y.astype(np.float64))
        out[k] = f"{int((x != y).sum())}/{x.size} moved, max|d| {float(d.max()):.3g}"
print(json.dumps(out, indent=1))
