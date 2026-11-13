"""X70, the ent-0 control of the trailing-magnet screen (PREREG §5): the READ of a head-to-head cross. CE-3d's reader
(``../../ce3d_magnet_screen_2026-11-11/cross/read_cross.py``) with this screen's families, requests and pair counts.

Run with ``PYTHONPATH=<the play-pin checkout>/src``. The plan's ``shas.json`` names the cross:

* ``anchor`` (the REGISTERED cross): reads request ``x70_swap_vs_ent0`` of family ``x70_swap_vs_ent0_h2h`` from the
  eval COUNT ledger (protocol ``gen3_eval_protocol_v1_h2h``), builds h[i][j] (pp; rows = the swap forks, columns =
  the X70 forks, keyed by each final's sha256) with ``main.h2h.cross.matrix_from_cells``, then applies the CODED
  rule ``screen_rule.decide``. Writes ``<outdir>/h.json``, ``<outdir>/cells.json`` and ``<outdir>/verdict.json``.
* ``c0`` / ``x67`` (the REPORTED crosses): the same read of request ``x70_ent0_vs_c0`` / ``x70_ent0_vs_x67`` of
  family ``x70_ent0_vs_c0_h2h`` / ``x70_ent0_vs_x67_h2h`` (rows = the X70 forks), handed to ``screen_rule.report``.
  Writes ``<outdir>/h.json``, ``<outdir>/cells.json`` and ``<outdir>/report_<cross>.json``, NEVER a
  ``verdict.json``: a reported cross carries no class and cannot change the registered one.

A cell must hold EXACTLY the cross's registered pair count (4,000 for ``anchor``, 1,000 for the reported two): a
missing / duplicate / non-OK cell, or one with fewer OR more completed pairs, is an INCONCLUSIVE reason and the
statistic is not computed.

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
#: cross -> (family, request, registered pairs a cell, the file its result is written to)
CROSSES = {"anchor": ("x70_swap_vs_ent0_h2h", "x70_swap_vs_ent0", 4000, "verdict.json"),
           "c0": ("x70_ent0_vs_c0_h2h", "x70_ent0_vs_c0", 1000, "report_c0.json"),
           "x67": ("x70_ent0_vs_x67_h2h", "x70_ent0_vs_x67", 1000, "report_x67.json")}


def main() -> int:
    shas_path, outdir, rule_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    outdir.mkdir(parents=True, exist_ok=True)
    shas = json.loads(shas_path.read_text())
    cross = shas["cross"]
    family, request, pairs, out_name = CROSSES[cross]
    if shas.get("pairs") != pairs:
        raise SystemExit(f"the plan's pairs {shas.get('pairs')!r} != the registered {pairs} for '{cross}'")
    cells = X.look_cells(ROOT, family, request, PROTOCOL)
    m = X.matrix_from_cells(cells, shas["rows"], shas["cols"], min_pairs=pairs)
    (outdir / "cells.json").write_text(json.dumps(list(m.cells), indent=1))
    reasons = list(m.inconclusive)
    reasons += [f"cell {c['player']} vs {c['opponent']}: {c['n_pairs']} completed pairs > the registered {pairs}"
                for c in m.cells if c["n_pairs"] > pairs]
    rule = runpy.run_path(str(rule_path))
    if m.h is not None:
        (outdir / "h.json").write_text(json.dumps({
            "h_pp": [list(r) for r in m.h], "rows": shas["row_runs"], "cols": shas["col_runs"],
            "row_seeds": shas["row_seeds"], "col_seeds": shas["col_seeds"], "pairs": pairs}, indent=1))
    h = [list(r) for r in m.h] if m.h is not None and not reasons else []
    if cross == "anchor":
        verdict = rule["decide"](h, inconclusive=reasons)
    else:
        verdict = rule["report"](h, cross, inconclusive=reasons)
    verdict["n_cells_read"] = len(cells)
    verdict["family"], verdict["request"], verdict["pairs"] = family, request, pairs
    (outdir / out_name).write_text(json.dumps(verdict, indent=1, default=str))
    print(json.dumps(verdict, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
