"""D3 look 2, AM-D3-L2 item 11 (LR-F's compare, copied from lrf_read/s_identity_compare.py): the S lane's identity at the play pin, against G1's banked C0 rows.

Compares every eval-ledger row of a re-play ledger (``--replay``) with the G1 row of the same cell and batch
(player sha256, opponent sha256, ``seed.batch``) in G1's ledger (``--bank``): the regime id, the seed block, the
outcome counts and both outcome digests (``outcome_digest``, ``outcome_digest_all``). Prints one JSON document; exit
0 iff every re-played row has its bank row and all five are equal (exact string / integer equality: rule 8 has no
boundary here), exit 1 otherwise. Read-only on both ledgers.

    python lrf_read/s_identity_compare.py --replay ~/gen3ai_archive/x35_lrf_read/s_identity/ledger \
        --bank ~/gen3ai_archive/x35_g1/ledger
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

from agents.training.eval_ledger.store import scan_rows

FIELDS = ("regime_id", "seed", "counts", "outcome_digest", "outcome_digest_all")


def _key(r: Dict[str, Any]) -> Tuple[str, str, int]:
    return (r["player"]["sha256"], r["opponent"]["sha256"], int(r["seed"]["batch"]))


def _view(r: Dict[str, Any]) -> Dict[str, Any]:
    return {"regime_id": r["regime"]["regime_id"], "seed": r["seed"], "counts": r["counts"],
            "outcome_digest": r["compute"].get("outcome_digest"),
            "outcome_digest_all": r["compute"].get("outcome_digest_all"),
            "core_stamp": r["compute"].get("core_stamp"), "commit": r.get("commit")}


def compare(replay: Path, bank: Path) -> Dict[str, Any]:
    problems: list = []
    banked = {_key(s.row): s.row for s in scan_rows(bank, problems)}
    cells = []
    for s in scan_rows(replay, problems):
        r = s.row
        b = banked.get(_key(r))
        mine = _view(r)
        row: Dict[str, Any] = {"player": r["player"]["id"], "opponent": r["opponent"]["id"],
                               "batch": int(r["seed"]["batch"]), "replay": mine}
        if b is None:
            row.update(equal=False, why="no bank row for this cell and batch")
        else:
            theirs = _view(b)
            diff = [f for f in FIELDS if mine[f] != theirs[f]]
            row.update(bank=theirs, differing=diff, equal=not diff)
        cells.append(row)
    ok = bool(cells) and all(c["equal"] for c in cells) and not problems
    return {"schema": "x41_d3_l2_s_identity_v1", "replay": str(replay), "bank": str(bank), "fields": list(FIELDS),
            "rows": cells, "ledger_problems": problems, "identical": ok}


def main(argv: Any = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--replay", required=True, type=Path)
    ap.add_argument("--bank", required=True, type=Path)
    a = ap.parse_args(argv)
    out = compare(a.replay.expanduser(), a.bank.expanduser())
    print(json.dumps(out, indent=1, sort_keys=True, default=str))
    return 0 if out["identical"] else 1


if __name__ == "__main__":
    sys.exit(main())
