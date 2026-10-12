"""The REGISTERED interim subset (orchestrator, 2026-10-12, written before any F / F+M outcome was read):
per checkpoint, the first 40 states of each of the pilot's five gap quintiles in the pilot's own registered
sampling order (``sha256("value_disc_pilot_2026:" + decision_id)``, design_value_discrimination.md §7.2), 200 states.
The registered states file lists its states sorted by id (opponent name first), so a plain file-order prefix would
be one cycle and a few opponents; the quintile-stratified hash order is the registered order the pilot drew in.

    python -m replica.interim            # writes interim_ids_<cp>.txt next to this package
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
PER_QUINTILE = 40


def interim_ids(cp: str) -> list:
    states = json.loads((HERE.parent / "value_disc_pilot_2026-10-11" / f"states_{cp}.json").read_text())["states"]
    out = []
    for q in sorted({s["quintile"] for s in states.values()}):
        ids = [i for i, s in states.items() if s["quintile"] == q]
        ids.sort(key=lambda i: hashlib.sha256(("value_disc_pilot_2026:" + i).encode()).hexdigest())
        out += ids[:PER_QUINTILE]
    return out


def main() -> int:
    for cp in ("end2", "prod"):
        ids = interim_ids(cp)
        (HERE / f"interim_ids_{cp}.txt").write_text("".join(i + "\n" for i in ids))
        print(cp, len(ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
