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

ONE VENUE A CROSS (PREREG §14, Amendment 1): every row of the cross's request must name ONE device and ONE backend
(``compute.device`` / ``compute.backend``: ``cpu`` / ``eager`` on the agent cores, ``cuda`` / ``graph`` on the
card). A greedy decision within a rounding error of a tie can flip with the device (``main.h2h.play``), so a cross
whose rows name two venues is REFUSED here, before any cell is pooled and before any file is written: exit 7,
nothing read, no class. That is a HOLD for the orchestrator, never ``INCONCLUSIVE``, because the one look has not
been spent. The venue the cross was played on is written into the result as ``played_on``.

    python read_cross.py <shas.json> <outdir> <screen_rule.py>
    python read_cross.py --venue-check <shas.json> <cpu|cuda>     # BEFORE playing: exit 7 if the request already
                                                                  # holds rows played on another device
"""
from __future__ import annotations

import json
import os
import runpy
import sys
from pathlib import Path

import main.h2h.cross as X
from agents.training import eval_ledger as L

ROOT = Path(os.environ.get("GEN3AI_MODELS_DIR") or "/home/goodlad/dev/gen3ai/models") / "_ledger"
PROTOCOL = "gen3_eval_protocol_v1_h2h"
#: cross -> (family, request, registered pairs a cell, the file its result is written to)
CROSSES = {"anchor": ("x70_swap_vs_ent0_h2h", "x70_swap_vs_ent0", 4000, "verdict.json"),
           "c0": ("x70_ent0_vs_c0_h2h", "x70_ent0_vs_c0", 1000, "report_c0.json"),
           "x67": ("x70_ent0_vs_x67_h2h", "x70_ent0_vs_x67", 1000, "report_x67.json")}


def venues(rows, request: str) -> dict:
    """``{(device, backend): row count}`` over the ledger rows of ``request`` (a row's ``request.id``)."""
    where: dict = {}
    for r in rows:
        if (r.get("request") or {}).get("id") != request:
            continue
        c = r.get("compute") or {}
        k = (str(c.get("device")), str(c.get("backend")))
        where[k] = where.get(k, 0) + 1
    return where


def venue_problem(where: dict) -> str:
    """Why the rows were not played at ONE venue (``""`` = they were, or there is no row yet)."""
    if len(where) <= 1:
        return ""
    return ("the cross was played on more than one device/backend: "
            + ", ".join(f"{d}/{b} x{n} rows" for (d, b), n in sorted(where.items()))
            + " (PREREG §14: one venue a cross; nothing was read)")


def venue_check(shas_path: Path, device: str) -> int:
    """The driver's check BEFORE it plays or resumes: the request's recorded rows, if any, were played on ``device``."""
    family, request, _pairs, _out = CROSSES[json.loads(shas_path.read_text())["cross"]]
    where = venues(L.read(X.FAMILY_READ_OFF, root=ROOT, family=family).rows, request)
    other = {k: n for k, n in where.items() if k[0].split(":")[0] != device}
    if other:
        print(f"REFUSED: request {request} already holds rows played on "
              + ", ".join(f"{d}/{b} x{n}" for (d, b), n in sorted(other.items()))
              + f"; it is finished on that device, never on '{device}' (PREREG §14: one venue a cross)", file=sys.stderr)
        return 7
    print(f"venue OK: {request} holds {sum(where.values())} row(s), all on '{device}'" if where
          else f"venue OK: {request} holds no row yet")
    return 0


def main() -> int:
    if sys.argv[1] == "--venue-check":
        return venue_check(Path(sys.argv[2]), sys.argv[3])
    shas_path, outdir, rule_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
    outdir.mkdir(parents=True, exist_ok=True)
    shas = json.loads(shas_path.read_text())
    cross = shas["cross"]
    family, request, pairs, out_name = CROSSES[cross]
    if shas.get("pairs") != pairs:
        raise SystemExit(f"the plan's pairs {shas.get('pairs')!r} != the registered {pairs} for '{cross}'")
    where = venues(L.read(X.FAMILY_READ_OFF, root=ROOT, family=family).rows, request)
    problem = venue_problem(where)
    if problem:
        print(f"REFUSED: {problem}", file=sys.stderr)
        return 7
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
    verdict["played_on"] = [{"device": d, "backend": b, "rows": n} for (d, b), n in sorted(where.items())]
    verdict["family"], verdict["request"], verdict["pairs"] = family, request, pairs
    (outdir / out_name).write_text(json.dumps(verdict, indent=1, default=str))
    print(json.dumps(verdict, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
