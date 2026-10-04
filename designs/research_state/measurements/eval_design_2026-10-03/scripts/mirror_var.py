"""Mirrored-pair variance vs two independent games, from the P0 h2h rows (designs/research_state/measurements/x5_p0_h2h_2026-10-03/rows)."""
import json, glob, os
from collections import defaultdict
ROWS = os.path.join(os.path.dirname(__file__), "../../x5_p0_h2h_2026-10-03/rows/*.jsonl")
E = defaultdict(lambda: [0] * 5)
for p in glob.glob(ROWS):
    for l in open(p):
        r = json.loads(l)
        k = (r["player"]["id"], r["opponent"]["id"])
        for i, x in enumerate(r["pairs"]["counts"]):
            E[k][i] += x
for (a, b), c in sorted(E.items()):
    n = sum(c); s = [i / 4 for i in range(5)]
    m = sum(ci * si for ci, si in zip(c, s)) / n
    v = sum(ci * (si - m) ** 2 for ci, si in zip(c, s)) / n
    vind = m * (1 - m) / 2
    kind = "self" if a == b else "cross"
    print(f"{kind:5s} {a.split('/')[-1][:28]:28s} vs {b.split('/')[-1][:28]:28s} pairs {n} mean {m:.4f} pairSD {v**.5:.4f} indepSD {vind**.5:.4f} var_ratio {v/vind:.3f}")
