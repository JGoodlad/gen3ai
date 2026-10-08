"""The OBS-FACTS block (`gen3_obs_facts_v1`) — its READ-BACK (`describe`); the block is written by the Rust encoder.

Four facts the entity-coverage audit found the model never saw (`designs/endstate/design_entity_coverage_audit.md`
§5, ranks 4-7), APPENDED as the observation's LAST block (`constants.OFFSET_OBS_FACTS`; obs 2761 → 2845 at the X5
version break, config v144). Layout: `constants.FACTS_*` (never a literal). The Rust encoder writes it every decision
(`src/rust_sim/src/encoder/facts.rs`; the engine-truth test `src/rust_sim/tests/obs_facts_truth_test.rs` holds it
to the referee). The model READS it only under `--obs-facts v1` (`agents/model/obs_facts_inject.py`); `off`
(production) builds nothing.

1. **SEEN** — per OUR mon: what the OPPONENT has seen of it (on the field once, each move seen, item / ability public).
2. **CHOICE** — the opponent active's Choice-lock EVIDENCE: two not-locked proofs, the stint's first move, its run.
3. **VOL** — per side, the active's Encore / Taunt / Disable / Uproar / partial trap: elapsed residuals and the
   min / max turns left (`constants.FACTS_VOL_DURATION`).
4. **SCREENS** — per side: turns left on Reflect / Light Screen / Safeguard / Mist.

The Python writer (`encode_obs_facts`, over a poke-env-backed `LiveView` and the deleted event-window fold) is DELETED
with the Python encoder's encode path (T27 P6 slice 6d-2); the per-fact semantics are the Rust encoder's.
"""

from __future__ import annotations

import numpy as np

from .constants import (
    FACTS_CHOICE_OFFSET,
    FACTS_SCREEN_TURNS,
    FACTS_SCREENS,
    FACTS_SCREENS_OFFSET,
    FACTS_SEEN_OFFSET,
    FACTS_SEEN_ROW_DIM,
    FACTS_TURN_NORM,
    FACTS_VOL_CELL_DIM,
    FACTS_VOL_EFFECTS,
    FACTS_VOL_OFFSET,
    FACTS_VOL_SIDE_DIM,
    TEAM_SIZE,
)

_OURS = "ours"
_OPP = "opp"


def describe(block: np.ndarray) -> dict:
    """A human-readable decode of one block (tests, the prober)."""
    out: dict = {"seen": [], "choice": None, "vol": {}, "screens": {}}
    for i in range(TEAM_SIZE):
        r = block[FACTS_SEEN_OFFSET + i * FACTS_SEEN_ROW_DIM:
                  FACTS_SEEN_OFFSET + (i + 1) * FACTS_SEEN_ROW_DIM]
        out["seen"].append([int(round(float(x))) for x in r])
    c = block[FACTS_CHOICE_OFFSET:FACTS_CHOICE_OFFSET + 4]
    out["choice"] = {"not_locked_item": bool(c[0] > 0.5), "not_locked_moves": bool(c[1] > 0.5),
                     "first_move_num": int(c[2]), "run": float(c[3])}
    for s, side in enumerate((_OURS, _OPP)):
        for j, key in enumerate(FACTS_VOL_EFFECTS):
            o = FACTS_VOL_OFFSET + s * FACTS_VOL_SIDE_DIM + j * FACTS_VOL_CELL_DIM
            if block[o + 1] > 0:
                out["vol"][f"{side}_{key}"] = tuple(
                    int(round(float(block[o + q]) * FACTS_TURN_NORM)) for q in range(3))
        for j, name in enumerate(FACTS_SCREENS):
            v = float(block[FACTS_SCREENS_OFFSET + s * len(FACTS_SCREENS) + j])
            if v > 0:
                out["screens"][f"{side}_{name}"] = int(round(v * FACTS_SCREEN_TURNS))
    return out
