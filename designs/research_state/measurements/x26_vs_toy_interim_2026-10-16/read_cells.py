"""THE READ for the X26-vs-toy INTERIM head-to-head: the ledger's 3 cells -> production's score per cell (W / L / D, %),
and the POOLED score - 50 with a SEEDS-AS-UNITS interval: the three toy seeds are the units (one value each: production's
score against that toy seed), the interval is the one-sample t over those 3 (df 2): mean +- t_{0.95,2} * sd / sqrt(3).
That is the toy-crosses statistic (rect_stat.py) taken to its n = 1 column limit: with ONE production checkpoint there is
no column variance, so production's own seed variance is NOT in the interval (declared). Cell noise is inside each seed's
value. Also prints the per-cell binomial-ish 90 % interval over its 500 pairs for context (pair-level, df 499).
Usage: py.sh read_cells.py. Read-only on the ledger. Writes result/read.json."""
import json, math, sys
from pathlib import Path
from scipy.stats import t as student_t
import main.h2h.cross as X

HERE = Path(__file__).resolve().parent; OUT = HERE / "result"; OUT.mkdir(exist_ok=True)
ROOT = Path("/home/goodlad/dev/gen3ai/models/_ledger")
P = json.loads((HERE / "plan" / "refs.json").read_text()); R = P["refs"]
cells = X.look_cells(ROOT, "xvt_interim", "xvt_interim", "gen3_eval_protocol_v1_h2h")
rows = [R[k]["sha256"] for k in P["rows"]]; cols = [R[k]["sha256"] for k in P["cols"]]
m = X.matrix_from_cells(cells, rows, cols, min_pairs=500)
rec = {"rows": P["rows"], "cols": P["cols"], "inconclusive": list(m.inconclusive), "cells": list(m.cells)}
if m.h is None:
    (OUT / "read.json").write_text(json.dumps(rec, indent=1, default=str))
    print("INCOMPLETE:", len(m.inconclusive), m.inconclusive[:3]); sys.exit(1)
v = list(m.h[0])  # (1 row x 3 toy columns) production's score (pp) vs each toy seed
n = len(v); mean = math.fsum(v) / n; sd = math.sqrt(math.fsum((x - mean) ** 2 for x in v) / (n - 1)); se = sd / math.sqrt(n)
rec["per_seed_score_pp"] = v; rec["pooled_minus_50_pp"] = mean - 50.0; rec["se_pp"] = se; rec["df"] = n - 1
for lvl in (0.90, 0.95):
    q = float(student_t.ppf(0.5 + lvl / 2, n - 1)); rec[f"ci{int(lvl * 100)}_pp"] = [mean - 50 - q * se, mean - 50 + q * se]
rec["ci90_near_zero"] = any(abs(b) <= 1e-9 for b in rec["ci90_pp"])
(OUT / "read.json").write_text(json.dumps(rec, indent=1, default=str))
for c in m.cells:
    g = c["w"] + c["l"] + c["d"]
    print(f"production vs {c['opponent']}: W {c['w']} / L {c['l']} / D {c['d']} of {g}  score {c['score_pp']:.2f} %  ({c['score_pp'] - 50:+.2f} pp)  pairs {c['n_pairs']} aborted {c['aborted']} {c['verdict']}")
print(f"POOLED production - 50 = {rec['pooled_minus_50_pp']:+.2f} pp, 90% [{rec['ci90_pp'][0]:+.2f}, {rec['ci90_pp'][1]:+.2f}], "
      f"95% [{rec['ci95_pp'][0]:+.2f}, {rec['ci95_pp'][1]:+.2f}] (seeds-as-units, df {rec['df']})")
