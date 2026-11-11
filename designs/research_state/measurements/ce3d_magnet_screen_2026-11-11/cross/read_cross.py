"""CE-3d, the trailing-magnet screen (PREREG §5): the READ of the three head-to-head crosses.

Run with ``PYTHONPATH=<the CE-3d pin checkout>/src``. For each cross of the plan it reads its request (family
``<request>_h2h``) from the eval COUNT ledger (protocol ``gen3_eval_protocol_v1_h2h``) and builds h[i][j]
(pp, keyed by each final's sha256) with ``main.h2h.cross.matrix_from_cells`` (a missing / duplicate / non-OK cell, or
one below the cross's registered pairs, is an INCONCLUSIVE reason). The DECIDING cross (magnet forks x control forks,
2,000 pairs a cell) gets the CODED rule ``screen_rule.decide``; the two REPORTED crosses (against the C0 finals,
1,000 pairs a cell) get ``screen_rule.report``, and ``screen_rule.indirect`` joins them. A reported cross that is
incomplete is named as unavailable; it never changes the deciding class.

Writes ``<outdir>/<cross>/h.json`` and ``cells.json``, and ``<outdir>/verdict.json``; prints the verdict JSON.

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
#: cross -> (request id, registered mirrored pairs a cell); the request's family is ``<request id>_h2h``
REQUESTS = {"deciding": ("ce3d_magnet_vs_control", 2000), "mag_vs_c0": ("ce3d_magnet_vs_c0", 1000),
            "ent_vs_c0": ("ce3d_control_vs_c0", 1000)}


def main() -> int:
    shas_path, outdir, rule_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    shas = json.loads(shas_path.read_text())["crosses"]
    rule = runpy.run_path(str(rule_path))
    res = {}
    n_cells = {}
    for cross, (request, pairs) in REQUESTS.items():
        s = shas[cross]
        d = outdir / cross
        d.mkdir(parents=True, exist_ok=True)
        cells = X.look_cells(ROOT, f"{request}_h2h", request, PROTOCOL)
        n_cells[cross] = len(cells)
        m = X.matrix_from_cells(cells, s["rows"], s["cols"], min_pairs=pairs)
        (d / "cells.json").write_text(json.dumps(list(m.cells), indent=1))
        if m.h is not None:
            (d / "h.json").write_text(json.dumps({
                "h_pp": [list(r) for r in m.h], "rows": s["row_runs"], "cols": s["col_runs"],
                "row_seeds": s["row_seeds"], "col_seeds": s["col_seeds"]}, indent=1))
        h, bad = ([], list(m.inconclusive)) if m.h is None else (m.h, [])
        if cross == "deciding":
            res[cross] = rule["decide"](h, inconclusive=bad)
        else:
            res[cross] = rule["report"](h, inconclusive=bad, col_seeds=s["col_seeds"])
    verdict = dict(res["deciding"])
    verdict["reported_crosses"] = {"magnet_vs_c0": res["mag_vs_c0"], "control_vs_c0": res["ent_vs_c0"],
                                   "indirect": rule["indirect"](res["mag_vs_c0"], res["ent_vs_c0"], res["deciding"])}
    verdict["n_cells_read"] = n_cells
    (outdir / "verdict.json").write_text(json.dumps(verdict, indent=1, default=str))
    print(json.dumps(verdict, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
