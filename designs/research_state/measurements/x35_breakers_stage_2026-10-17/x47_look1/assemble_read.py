"""X47 look 1's READ JSON (BATCH1_LAUNCH_SHEET §4.5 "The read JSON"), from the look's ledger cells, `seq`'s output
and the pooled G-C record; `python -m main.x35.x47 decide` reads it.

* S of each X47 node = ``readout.population_score`` over the UNION of the look's ``_a`` / ``_b`` cells; S of each C0
  final over its ``_c`` cells (R = the six members of ``reference_population_toy.json``, 500-pair floor).
* ``per_member`` = the same cells per member (AM-X47-7).
* ``g_f.h_pp`` = ``cross.matrix_from_cells(_gf cells, s_eq rows, C0-final columns, min_pairs=readout.MIN_PAIRS)``
  (F-AMX-2: the floor passed explicitly), rows and columns in seed order.
* ``g_c`` = the ``main.x35.gc pool`` record; ``comparison`` = ``seq``'s output.

    python assemble_read.py <seq.json> <ledger root> <gc record> <out read.json>
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

from main.h2h import cross as CR
from main.x35 import readout as R

M = "/home/goodlad/dev/gen3ai/models"
HERE = os.path.dirname(os.path.abspath(__file__))
REFPOP = os.path.join(HERE, "..", "reference_population_toy.json")
FAMILY, PROTOCOL = "x35_x47", "gen3_eval_protocol_v1_h2h"


def sha(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def seed_of(run: str) -> str:
    return run.rsplit("_s", 1)[1]


def main(seq_path: str, ledger: str, gc_path: str, out: str) -> int:
    seq = json.load(open(seq_path))
    assert seq["look"] == 1
    refpop = json.load(open(REFPOP))
    members = {m["label"]: m["sha256"] for m in refpop["members"]}
    cells = {k: CR.look_cells(ledger, FAMILY, f"x35_x47_look1_{k}", PROTOCOL) for k in ("a", "b", "c", "gf")}
    x47_cells = cells["a"] + cells["b"]

    def score(path: str, pool: list) -> R.PopulationScore:
        return R.population_score(pool, sha(path), members, min_pairs=R.MIN_PAIRS)

    doc: dict = {"look": 1, "c0_final": {}, "x47_s_eq": {}, "x47_final": {},
                 "per_member": {"c0_final": {}, "x47_s_eq": {}, "x47_final_reported": {}}, "scores": {}}
    c0_sha, seq_sha = {}, {}
    for run, v in sorted(seq["c0_final"].items()):
        p = os.path.join(M, run, v["file"])
        ps = score(p, cells["c"])
        s = seed_of(run)
        c0_sha[s] = ps.final
        doc["c0_final"][s] = ps.s_pp
        doc["per_member"]["c0_final"][s] = dict(ps.per_member_pp)
        doc["scores"][f"{run}/{v['file']}"] = ps.as_dict()
    for run, v in sorted(seq["s_eq"].items()):
        s = seed_of(run)
        for key, f in (("x47_s_eq", v["file"]), ("x47_final", "final_model.zip")):
            p = os.path.join(M, run, f)
            ps = score(p, x47_cells)
            doc[key][s] = ps.s_pp
            pm_key = "x47_s_eq" if key == "x47_s_eq" else "x47_final_reported"
            doc["per_member"][pm_key][s] = dict(ps.per_member_pp)
            doc["scores"][f"{run}/{f}"] = ps.as_dict()
            if key == "x47_s_eq":
                seq_sha[s] = ps.final
    seeds = sorted(seq_sha)
    mat = CR.matrix_from_cells(cells["gf"], [seq_sha[s] for s in seeds], [c0_sha[s] for s in seeds],
                               min_pairs=R.MIN_PAIRS)
    doc["g_f"] = {"h_pp": [list(r) for r in mat.h] if mat.h is not None else None,
                  "inconclusive": list(mat.inconclusive), "cells": list(mat.cells), "rows_seeds": seeds}
    doc["g_c"] = json.load(open(gc_path))
    doc["comparison"] = seq
    doc["cell_counts"] = {k: len(v) for k, v in cells.items()}
    with open(out, "w") as f:
        json.dump(doc, f, indent=1, default=str)
    print(json.dumps({k: doc[k] for k in ("c0_final", "x47_s_eq", "x47_final")}, indent=1))
    print("G-F h_pp:", doc["g_f"]["h_pp"], doc["g_f"]["inconclusive"][:3])
    for k, v in doc["scores"].items():
        if v["inconclusive"]:
            print("INCONCLUSIVE", k, v["inconclusive"][:3])
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:5]))
