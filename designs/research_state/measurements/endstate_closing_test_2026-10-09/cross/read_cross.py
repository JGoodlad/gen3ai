"""END-STATE closing test: the READ of the 8 x 8 cross (design_endstate_closing_test.md §3.1-3.3).

Run with PYTHONPATH=<P_end checkout>/src. Reads request `es_closing_steps` of family `es_closing_strength_steps` from
the eval COUNT ledger (protocol gen3_eval_protocol_v1_h2h), builds h[i][j] (pp; rows = end-state seeds 2001..2008,
columns = production seeds 2001..2008, keyed by each final's sha256) with main.h2h.cross.matrix_from_cells (a missing /
duplicate / non-OK / < 1,000-pair cell is an INCONCLUSIVE reason), then applies the CODED rule closing_rule.decide.
Writes <outdir>/h.json, <outdir>/cells.json, <outdir>/verdict.json and prints the verdict JSON.
"""
from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path

import main.h2h.cross as X

ROOT = Path("/home/goodlad/dev/gen3ai/models/_ledger")
FAMILY, REQUEST, PROTOCOL = "es_closing_strength_steps", "es_closing_steps", "gen3_eval_protocol_v1_h2h"


def main() -> int:
    shas_path, outdir, rule_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    outdir.mkdir(parents=True, exist_ok=True)
    shas = json.loads(shas_path.read_text())
    cells = X.look_cells(ROOT, FAMILY, REQUEST, PROTOCOL)
    m = X.matrix_from_cells(cells, shas["rows"], shas["cols"])
    (outdir / "cells.json").write_text(json.dumps(list(m.cells), indent=1))
    rule = runpy.run_path(str(rule_path))
    if m.h is None:
        verdict = rule["decide"]([], inconclusive=list(m.inconclusive))
    else:
        (outdir / "h.json").write_text(json.dumps({"h_pp": [list(r) for r in m.h], "rows": shas["row_runs"],
                                                   "cols": shas["col_runs"]}, indent=1))
        verdict = rule["decide"](m.h)
    verdict["n_cells_read"] = len(cells)
    (outdir / "verdict.json").write_text(json.dumps(verdict, indent=1, default=str))
    print(json.dumps(verdict, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
