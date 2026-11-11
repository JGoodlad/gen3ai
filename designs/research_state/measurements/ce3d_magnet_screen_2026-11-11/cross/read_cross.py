"""CE-3d, the trailing-magnet screen, Amendments 1 and 2 (PREREG §5): the READ of a head-to-head cross. X67's reader
(``../../x67_entropy_screen_2026-11-09/cross/read_cross.py``) with this screen's families and requests.

Run with ``PYTHONPATH=<the CE-3d pin checkout>/src``. The plan's ``shas.json`` names the cross (``cols_arm``):

* ``c0`` (the REGISTERED cross): reads request ``ce3d_swap_vs_c0`` of family ``ce3d_swap_vs_c0_h2h`` from the eval
  COUNT ledger (protocol ``gen3_eval_protocol_v1_h2h``), builds h[i][j] (pp; rows = the swap forks, columns = the C0
  finals of the plan, keyed by each final's sha256) with ``main.h2h.cross.matrix_from_cells`` (a missing / duplicate
  / non-OK / < 1,000-pair cell is an INCONCLUSIVE reason), then applies the CODED rule ``screen_rule.decide``.
  Writes ``<outdir>/h.json``, ``<outdir>/cells.json`` and ``<outdir>/verdict.json`` and prints the verdict JSON.
* ``x67`` (Amendment 2's REPORTED cross, §5.4): the same read of request ``ce3d_swap_vs_x67`` of family
  ``ce3d_swap_vs_x67_h2h`` (columns = X67's three finals), handed to ``screen_rule.report_x67``. Writes
  ``<outdir>/h.json``, ``<outdir>/cells.json`` and ``<outdir>/report_x67.json``, NEVER a ``verdict.json``: it
  carries no class and cannot change the registered one.

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
PROTOCOL = "gen3_eval_protocol_v1_h2h"
#: cols_arm -> (family, request, the rule function in screen_rule.py, the file its result is written to)
CROSSES = {"c0": ("ce3d_swap_vs_c0_h2h", "ce3d_swap_vs_c0", "decide", "verdict.json"),
           "x67": ("ce3d_swap_vs_x67_h2h", "ce3d_swap_vs_x67", "report_x67", "report_x67.json")}


def main() -> int:
    shas_path, outdir, rule_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    outdir.mkdir(parents=True, exist_ok=True)
    shas = json.loads(shas_path.read_text())
    family, request, fn, out_name = CROSSES[shas["cols_arm"]]
    cells = X.look_cells(ROOT, family, request, PROTOCOL)
    m = X.matrix_from_cells(cells, shas["rows"], shas["cols"])
    (outdir / "cells.json").write_text(json.dumps(list(m.cells), indent=1))
    rule = runpy.run_path(str(rule_path))
    if m.h is None:
        verdict = rule[fn]([], inconclusive=list(m.inconclusive))
    else:
        (outdir / "h.json").write_text(json.dumps({
            "h_pp": [list(r) for r in m.h], "rows": shas["row_runs"], "cols": shas["col_runs"],
            "row_seeds": shas["row_seeds"], "col_seeds": shas["col_seeds"]}, indent=1))
        verdict = rule[fn](m.h, col_seeds=shas["col_seeds"]) if fn == "decide" else rule[fn](m.h)
    verdict["n_cells_read"] = len(cells)
    verdict["family"], verdict["request"] = family, request
    (outdir / out_name).write_text(json.dumps(verdict, indent=1, default=str))
    print(json.dumps(verdict, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
