"""W4's READ: the ledger's cells for E7 / R-T1 / R-T1b -> h matrices (main.h2h.cross.matrix_from_cells); E7 through
the committed sanity_rule.decide with the run gates read from the xs run dirs; R-T1 / R-T1b descriptive (cross_stat).
Usage: read_w4.py <plan_dir> <out_dir> <sanity_rule.py>. Run with PYTHONPATH=<P_X26 checkout>/src."""
import json, runpy, sys
from pathlib import Path
import main.h2h.cross as X
PLAN, OUT, RULE = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
OUT.mkdir(parents=True, exist_ok=True)
ROOT = Path("/home/goodlad/dev/gen3ai/models/_ledger"); M = Path("/home/goodlad/dev/gen3ai/models")
P = json.loads((PLAN / "refs.json").read_text()); R = P["refs"]
FAM = {"e7": ("x26_e7_sanity", "x26_e7"), "rt1": ("x26_toy_rt1", "x26_rt1"), "rt1b": ("x26_toy_rt1b", "x26_rt1b")}
# the registered pairs per cell (design_x26_readiness.md §4.1): E7 / R-T1 500, R-T1b 250 — matrix_from_cells defaults to the
# closing test's MIN_PAIRS (1,000), which refused every cell on the first read (10-16 03:07; read_w4_v1_minpairs_bug.py).
PAIRS = {"e7": 500, "rt1": 500, "rt1b": 250}
def gates(run):
    d = M / run; log = (d / "launcher_child.log").read_text(errors="replace") if (d / "launcher_child.log").is_file() else ""
    md = json.loads((d / "metadata.json").read_text()); ph = md.get("pin_history") or []
    last = max([int(p.get("last_step") or 0) for p in ph] or [0])
    crash_files = list((d / "crashes").glob("*")) if (d / "crashes").is_dir() else []
    nan = any(("nan" in l.lower() or "non-finite" in l.lower()) and ("FATAL" in l or "Error" in l) for l in log.splitlines())
    k9 = any("K9(b)" in l and "FATAL" in l for l in log.splitlines())
    return {"reached_5M": bool((d / "final_model.zip").is_file() and last >= 5_000_000), "crash": bool(crash_files), "nan": nan, "k9b_real_stop": k9}
out = {}
for c, (fam, req) in FAM.items():
    rows = [R[k]["sha256"] for k in P["crosses"][c]["rows"]]; cols = [R[k]["sha256"] for k in P["crosses"][c]["cols"]]
    cells = X.look_cells(ROOT, fam, req, "gen3_eval_protocol_v1_h2h")
    m = X.matrix_from_cells(cells, rows, cols, min_pairs=PAIRS[c])
    (OUT / f"cells_{c}.json").write_text(json.dumps(list(m.cells), indent=1, default=str))
    rec = {"rows": P["crosses"][c]["rows"], "cols": P["crosses"][c]["cols"], "inconclusive": list(m.inconclusive)}
    if m.h is not None:
        rec["h_pp"] = [list(r) for r in m.h]
        try:
            s = X.cross_stat(m.h); rec["cross_stat"] = {k: getattr(s, k) for k in dir(s) if not k.startswith("_") and not callable(getattr(s, k))}
        except Exception as e: rec["cross_stat"] = f"{type(e).__name__}: {e}"
    out[c] = rec
rg = {r: gates(r) for r in ("rb_xs_s2701", "rb_xs_s2702")}
rule = runpy.run_path(RULE)
out["e7_verdict"] = rule["decide"](out["e7"]["h_pp"], rg) if "h_pp" in out["e7"] else {"outcome": "INCONCLUSIVE", "reasons": out["e7"]["inconclusive"]}
out["run_gates"] = rg
(OUT / "w4_read.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out["e7_verdict"], default=str))
