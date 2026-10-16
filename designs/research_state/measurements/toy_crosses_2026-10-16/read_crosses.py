"""THE READ for the toy crosses (PREREG.md §5): the ledger's cells of R-T5 / R-T6 -> h matrices
(main.h2h.cross.matrix_from_cells, min_pairs 250 = the registered pairs), each through rect_stat (descriptive). For
comparison only, W4's R-T1 / R-T1b matrices (road/w4/result/w4_read.json, read as committed) go through the same
statistic. Writes <here>/result/read.json; prints a summary. Usage: py.sh read_crosses.py. Read-only on the ledger."""
import json, runpy, sys
from pathlib import Path
import main.h2h.cross as X

HERE = Path(__file__).resolve().parent; OUT = HERE / "result"; OUT.mkdir(exist_ok=True)
ROOT = Path("/home/goodlad/dev/gen3ai/models/_ledger")
RS = runpy.run_path(str(HERE / "rect_stat.py"))["rect_stat"]
P = json.loads((HERE / "plan" / "refs.json").read_text()); R = P["refs"]
FAM = {"rt5": ("toyx_rt5_eqcompute", "toyx_rt5"), "rt6": ("toyx_rt6_cap", "toyx_rt6")}
PAIRS = 250
out = {}
for c, (fam, req) in FAM.items():
    rows = [R[k]["sha256"] for k in P["crosses"][c]["rows"]]; cols = [R[k]["sha256"] for k in P["crosses"][c]["cols"]]
    try:
        cells = X.look_cells(ROOT, fam, req, "gen3_eval_protocol_v1_h2h")
    except Exception as e:  # an unregistered family (before the runner's first step) reads as no cells
        cells = []; print(f"{c}: {type(e).__name__}: {str(e)[:160]}", file=sys.stderr)
    m = X.matrix_from_cells(cells, rows, cols, min_pairs=PAIRS)
    rec = {"rows": P["crosses"][c]["rows"], "cols": P["crosses"][c]["cols"], "inconclusive": list(m.inconclusive),
           "cells": list(m.cells)}
    if m.h is not None:
        rec["h_pp"] = [list(r) for r in m.h]; rec["rect_stat"] = RS([list(r) for r in m.h])
        rec["pooled_mean_minus_50_pp"] = rec["rect_stat"]["delta_hat_pp"]
    out[c] = rec
w4 = HERE.parent / "x26_exit_check_2026-10-15" / "road" / "w4" / "result" / "w4_read.json"
if w4.is_file():
    W = json.loads(w4.read_text())
    for c in ("rt1", "rt1b"):
        if "h_pp" in W.get(c, {}): out[f"w4_{c}"] = {"rows": W[c]["rows"], "cols": W[c]["cols"], "rect_stat": RS(W[c]["h_pp"])}
(OUT / "read.json").write_text(json.dumps(out, indent=1, default=str))
for k, v in out.items():
    s = v.get("rect_stat")
    if s is None: print(k, "INCOMPLETE:", len(v["inconclusive"]), "reasons; first:", v["inconclusive"][:2]); continue
    print(f"{k}: pooled {s['delta_hat_pp']:+.2f} pp, 90% [{s['ci90_pp'][0]:+.2f}, {s['ci90_pp'][1]:+.2f}] (df {s['df_welch']:.1f}); "
          f"per seed " + " / ".join(f"{p['mean_minus_50_pp']:+.2f}" for p in s["per_seed"]))
