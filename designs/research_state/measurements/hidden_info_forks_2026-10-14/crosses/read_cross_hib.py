"""HIB knock-in forks: the READ of the §8.4 crosses, applying the committed read_hib.decide (f68f686b). Run with
PYTHONPATH=<the 8272c6cb checkout>/src (its look_cells knows the withheld protocol). K-VALID inputs: each fork's bank audit (bank_hib.py STRICT-BY-READ) passed in as
argv[3] (a JSON list of invalid reasons, [] when all valid)."""
import json, runpy, sys
from pathlib import Path
import main.h2h.cross as X
ROOT = Path("/home/goodlad/dev/gen3ai/models/_ledger"); C = Path(sys.argv[1]); OUT = Path(sys.argv[2]); invalid = json.loads(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)
S = json.loads((C / "shas.json").read_text())["shas"]; SEEDS = (3101, 3102, 3103)
rule = runpy.run_path(str(C / "read_hib.py"))
mats = {"invalid": invalid, "inconclusive": {}}
for lv in rule["LEVERS"]:
    cells = X.look_cells(ROOT, "hib_forks_primary", f"hib_primary_{lv}", "gen3_eval_protocol_v1_h2h_oracle_one_sided")
    m = X.matrix_from_cells(cells, [S[f"{lv}_s{s}"] for s in SEEDS], [S[f"ctrl_s{s}"] for s in SEEDS])
    (OUT / f"cells_{lv}.json").write_text(json.dumps(list(m.cells), indent=1, default=str))
    if m.h is None: mats["inconclusive"][lv] = list(m.inconclusive); mats[lv] = None
    else: mats[lv] = [list(r) for r in m.h]
cells = X.look_cells(ROOT, "hib_forks_v8", "hib_v8", "gen3_eval_protocol_v1_h2h")
m = X.matrix_from_cells(cells, [S[f"ctrl_s{s}"] for s in SEEDS], [S["parent"]])
(OUT / "cells_v8.json").write_text(json.dumps(list(m.cells), indent=1, default=str))
if m.h is not None: mats["v8"] = [r[0] for r in m.h]
hon = {}
for lv in rule["LEVERS"]:
    cells = X.look_cells(ROOT, "hib_forks_honest", f"hib_honest_{lv}", "gen3_eval_protocol_v1_h2h_oracle_withheld")
    sc, info = [], []
    for s in SEEDS:
        pair = (S[f"{lv}_s{s}"], S[f"ctrl_s{s}"])  # matrix_from_cells refuses cells outside rows x cols: hand it this pair's
        m = X.matrix_from_cells([c for c in cells if (c.player, c.opponent) == pair], [pair[0]], [pair[1]])
        info.extend(m.cells)
        sc.append(None if m.h is None else m.h[0][0])
    (OUT / f"cells_honest_{lv}.json").write_text(json.dumps(info, indent=1, default=str))
    if all(x is not None for x in sc): hon[lv] = sc
    else: mats["inconclusive"].setdefault(f"honest_{lv}", []).append("an honest cell is missing or invalid")
if hon: mats["honest"] = hon
(OUT / "h.json").write_text(json.dumps({k: v for k, v in mats.items() if k in rule["LEVERS"] or k in ("v8", "honest")}, indent=1))
v = rule["decide"](mats); (OUT / "verdict.json").write_text(json.dumps(v, indent=1, default=str))
print(json.dumps({"outcome": v.get("outcome"), "levers": {k: x.get("outcome") for k, x in (v.get("levers") or {}).items()},
                  "MOVES_FIRST": (v.get("k_rank") or {}).get("MOVES_FIRST"), "extend": (v.get("moves_extension") or {}).get("extend")}))
