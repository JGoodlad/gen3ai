"""The OBS-FACTS block (`gen3_obs_facts_v1`) — four facts the entity-coverage audit found the model
never sees (`designs/endstate/design_entity_coverage_audit.md` §5, ranks 4-7; owner scope
2026-10-06). Layout: `constants.FACTS_*` (never a literal).

🚨 **COMPUTED, NOT YET IN THE OBSERVATION.** The block is built and gated (slice O compares it byte for
byte with the Rust core's `encoder::facts`, beside the row; the engine-truth test holds it to the
referee), but `Gen3ObservationEncoder.encode` does not write it: the append (2761 → 2845) and the
model's `--obs-facts {off,v1}` consumer land at the ONE planned checkpoint break, the X5 adoption
version break (orchestrator decision 2026-10-06; branch `obs-facts-append`).

1. **SEEN** (backlog E1) — per OUR mon: what the OPPONENT has seen of it. Read off the reading's
   own reveal rules pointed at our side: ``LivePokemon.revealed`` (on the field at least once),
   ``LiveMove.seen`` (a public ``|move|`` line revealed the move — the per-mon slot's move order),
   ``item_public`` / ``ability_public`` (a protocol line, never a ``|request|``).
2. **CHOICE** — the opponent active's Choice-lock EVIDENCE (owner: "we can never 100 % know if they
   are choice locked, only if we know they aren't"). Never a lock state: (a) two NOT-locked proofs —
   a revealed non-Choice item (or a known empty hand: in gen 3/4 a lock holds only while the item is
   a Choice item, ``data/conditions.ts`` ``choicelock.onBeforeMove``), and two DISTINCT freely
   selected moves this stint; (b) the stint's first move — the one a Choice Band would lock it into
   (``data/mods/gen4/items.ts`` ``choiceband.onAfterMove`` adds ``choicelock``, whose gen-4 ``onStart``
   stores ``pokemon.lastMove``); (c) the trailing run of the same move. Struggle and a called move
   are not a selection (the lock ignores Struggle, ``choicelock.onBeforeMove``).
3. **VOL** — per side, the ACTIVE's Encore / Taunt / Disable / Uproar / partial trap: elapsed
   residuals and the known min / max turns LEFT. Showdown counts these durations in RESIDUALS
   (``sim/battle.ts`` ``fieldEvent('Residual')`` decrements ``duration`` and ends the effect at 0),
   so elapsed = poke-env's ``|turn|``-counted counter + one when this turn's residual has run
   (``LiveView.residual_done``). Durations: ``constants.FACTS_VOL_DURATION``. An Encore / Disable
   whose target had already acted gets +1 (the fold records it at the ``-start``); with no fold the
   bounds take the union.
4. **SCREENS** — per side: turns left on Reflect / Light Screen / Safeguard / Mist (5 residuals
   each), from poke-env's stored start turn (``_side_start``) and the same residual correction.

Mirrored byte for byte by the Rust encoder (``src/rust_sim/src/encoder/facts.rs``, slice O).
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

import numpy as np

from agents.gen3_data import moves as gen3_movedex
from .assembler import SAT_LUT
from .constants import (
    FACTS_CHOICE_OFFSET,
    FACTS_SCREEN_TURNS,
    FACTS_SCREENS,
    FACTS_SCREENS_OFFSET,
    FACTS_SEEN_OFFSET,
    FACTS_SEEN_ROW_DIM,
    FACTS_TURN_NORM,
    FACTS_VOL_CELL_DIM,
    FACTS_VOL_DURATION,
    FACTS_VOL_EFFECTS,
    FACTS_VOL_OFFSET,
    FACTS_VOL_SIDE_DIM,
    OBS_FACTS_DIM,
    TEAM_SIZE,
)

# Every partial-trap move's volatile id in the reading (poke-env starts WRAP / BIND / … from the
# `-activate|…|move: Wrap` line; `partiallytrapped` is the collapsed id the volatile vocabulary uses).
PARTIAL_TRAP_IDS = ("partiallytrapped", "wrap", "bind", "clamp", "whirlpool", "firespin", "sandtomb")
_SAT_CAP = len(SAT_LUT) - 1
_OURS = "ours"
_OPP = "opp"


def _move_num(move_id: Optional[str]) -> float:
    if not move_id:
        return 0.0
    md = gen3_movedex.get(move_id)
    return float(md.num) if md is not None else 0.0


def _vol_counter(vols: Any, key: str) -> Optional[int]:
    """The reading's counter for a duration volatile, ``None`` when absent. The partial trap is
    the max counter over every trap id present."""
    if key != "partiallytrapped":
        c = vols.get(key)
        return None if c is None else int(c)
    best: Optional[int] = None
    for k in PARTIAL_TRAP_IDS:
        c = vols.get(k)
        if c is not None and (best is None or int(c) > best):
            best = int(c)
    return best


def _write_vol(vec: np.ndarray, base: int, mon: Any, side: str, fold: Any, residual: int) -> None:
    if mon is None or mon.fainted:
        return
    vols = mon.volatiles
    for j, key in enumerate(FACTS_VOL_EFFECTS):
        c = _vol_counter(vols, key)
        if c is None:
            continue
        lo, hi, adjustable = FACTS_VOL_DURATION[key]
        adj_lo = adj_hi = 0
        if adjustable:
            adj = fold.adjusted(side, key) if fold is not None else None
            if adj is None:
                adj_hi = 1          # unknown: the union of both cases
            elif adj:
                adj_lo = adj_hi = 1
        e = c + residual
        min_left = max(1, lo + adj_lo - e)
        max_left = max(min_left, hi + adj_hi - e)
        o = base + j * FACTS_VOL_CELL_DIM
        vec[o] = min(e, FACTS_TURN_NORM) / FACTS_TURN_NORM
        vec[o + 1] = min_left / FACTS_TURN_NORM
        vec[o + 2] = max_left / FACTS_TURN_NORM


def encode_obs_facts(vec: np.ndarray, off: int, live: Any, our_species: Sequence[Optional[str]],
                     event_window: Any) -> None:
    """Write the ``OBS_FACTS_DIM`` block into ``vec[off:off + OBS_FACTS_DIM]`` (every cell).

    ``our_species`` is OUR team list in per-mon slot order; ``event_window`` the decision's
    :class:`EventWindowTracker` (its ``facts`` fold), or ``None``."""
    vec[off:off + OBS_FACTS_DIM] = 0.0
    if live is None:
        return
    fold = getattr(event_window, "facts", None) if event_window is not None else None
    residual = 1 if live.residual_done else 0

    # 1. SEEN — row i is our team slot i
    for i in range(TEAM_SIZE):
        sp = our_species[i] if i < len(our_species) else None
        m = live.ours.get(sp) if sp is not None else None
        if m is None:
            continue
        o = off + FACTS_SEEN_OFFSET + i * FACTS_SEEN_ROW_DIM
        vec[o] = 1.0 if m.revealed else 0.0
        for k, mv in enumerate(m.moves[:4]):
            vec[o + 1 + k] = 1.0 if mv.seen else 0.0
        vec[o + 5] = 1.0 if m.item_public else 0.0
        vec[o + 6] = 1.0 if m.ability_public else 0.0

    # 2. CHOICE — the opponent active
    oa = live.opp.active
    if oa is not None and not oa.fainted:
        o = off + FACTS_CHOICE_OFFSET
        vec[o] = 1.0 if (oa.item_public and oa.item != "choiceband") else 0.0
        if fold is not None:
            first, distinct2, run = fold.stint(_OPP)
            vec[o + 1] = 1.0 if distinct2 else 0.0
            vec[o + 2] = _move_num(first)
            vec[o + 3] = SAT_LUT[run if run < _SAT_CAP else _SAT_CAP]

    # 3. VOL — each side's active
    _write_vol(vec, off + FACTS_VOL_OFFSET, live.ours.active, _OURS, fold, residual)
    _write_vol(vec, off + FACTS_VOL_OFFSET + FACTS_VOL_SIDE_DIM, live.opp.active, _OPP, fold,
               residual)

    # 4. SCREENS — side-major
    turn = int(live.turn)
    for s, side in enumerate((live.ours, live.opp)):
        conds = side.side_conditions
        for j, name in enumerate(FACTS_SCREENS):
            start = conds.get(name)
            if start is None:
                continue
            left = max(0, FACTS_SCREEN_TURNS - ((turn - int(start)) + residual))
            vec[off + FACTS_SCREENS_OFFSET + s * len(FACTS_SCREENS) + j] = left / FACTS_SCREEN_TURNS


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


def our_species_list(team_list: List[Any]) -> List[Optional[str]]:
    return [m.species if m is not None else None for m in team_list]
