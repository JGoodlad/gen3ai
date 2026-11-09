"""X67, the entropy screen (PREREG §5): the READ of the head-to-head cross.

Run with ``PYTHONPATH=<the pin checkout>/src``. Reads request ``x67_entropy_screen`` of family
``x67_entropy_screen_h2h`` from the eval COUNT ledger (protocol ``gen3_eval_protocol_v1_h2h``), builds h[i][j] (pp;
rows = the X67 forks, columns = the C0 finals of the plan, keyed by each final's sha256) with
``main.h2h.cross.matrix_from_cells`` (a missing / duplicate / non-OK / < 1,000-pair cell is an INCONCLUSIVE reason),
then applies the CODED rule ``screen_rule.decide``. Writes ``<outdir>/h.json``, ``<outdir>/cells.json`` and
``<outdir>/verdict.json`` and prints the verdict JSON.

    python read_cross.py <shas.json> <outdir> <screen_rule.py>
"""
from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path

import main.h2h.cross as X

ROOT = Path(os.environ.get("GEN3AI_MODELS_DIR") or "/home/goodlad/dev/gen3ai/models") / "_ledger"
FAMILY, REQUEST, PROTOCOL = "x67_entropy_screen_h2h", "x67_entropy_screen", "gen3_eval_protocol_v1_h2h"


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
        (outdir / "h.json").write_text(json.dumps({
            "h_pp": [list(r) for r in m.h], "rows": shas["row_runs"], "cols": shas["col_runs"],
            "row_seeds": shas["row_seeds"], "col_seeds": shas["col_seeds"]}, indent=1))
        verdict = rule["decide"](m.h, col_seeds=shas["col_seeds"])
    verdict["n_cells_read"] = len(cells)
    (outdir / "verdict.json").write_text(json.dumps(verdict, indent=1, default=str))
    print(json.dumps(verdict, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
