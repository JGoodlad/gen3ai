"""D3 look 2: the record-time checks registered in AM-D3-L2 (items 7, 10 and m3), read-only over the look-2 state.

* item 10 / item 7 (H21): main's G14 re-derivation (``main.reads.rederive.unit_check``: forced ends in the captures,
  the front end's disconnect lines, the peers' long-tail lines) on every COUNTED unit of both anchor groups;
* m3: every counted unit stamps ``model_loader: bare``.

Run with MAIN's ``src`` on PYTHONPATH (the re-derivation exists only there). Prints one JSON document; exit 0 iff no
counted unit has a HIT and every counted unit is ``bare``.

usage: python precheck.py <d3 state>
"""
import json
import sys
from pathlib import Path

from main.reads import rederive as RD


def counted_units(state: Path):
    for ujl in sorted((state / "anchor").glob("*/*/units.jsonl")):
        last = {}
        for ln in ujl.read_text().splitlines():
            d = json.loads(ln)
            if d.get("event") == "end":
                last[d["unit"]] = bool(d.get("counted"))
        for unit, ok in sorted(last.items()):
            if ok:
                yield ujl.parent.parent.name, ujl.parent.name, ujl.parent / "cells" / unit


def main(argv):
    state = Path(argv[0])
    rows, hits, not_bare, unavailable = [], [], [], []
    for group, label, cell in counted_units(state):
        summary = json.loads((cell / "summary.json").read_text())
        chk = RD.unit_check(cell, summary)
        loader = (summary.get("cell") or {}).get("model_loader")
        rows.append({"group": group, "label": label, "unit": cell.name, "forced_end": chk["forced_end"],
                     "peer_fault": chk["peer_fault"], "hits": chk["hits"], "why": chk["why"], "model_loader": loader})
        if chk["hits"]:
            hits.append(cell.name)
        if RD.UNAVAILABLE in (chk["forced_end"], chk["peer_fault"]):
            unavailable.append(cell.name)
        if loader != "bare":
            not_bare.append([cell.name, loader])
    out = {"schema": "x41_d3_look2_precheck_v1", "state": str(state), "rederive_schema": RD.SCHEMA,
           "counted_units": len(rows), "hit_units": hits, "unavailable_units": unavailable,
           "not_bare": not_bare, "units": rows}
    print(json.dumps(out, indent=1, default=str))
    return 0 if not hits and not not_bare else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
