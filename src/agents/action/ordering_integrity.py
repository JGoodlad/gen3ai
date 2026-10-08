"""Live integrity checks that the ordering the *model* sees matches the ordering
the *action space* uses.

Our active mon's moves exist in TWO orders in every observation row:

* the per-mon move slots — SORTED BY ``Move.id`` STRING (``MovesEncoder`` ->
  ``get_sorted_moves``; the Rust port's ``encoder/slot.rs``), the same for every mon;
* the request block (``reactive.active_req_moves``: ids / types / legality) — REQUEST
  order, so request slot ``k`` IS action ``6+k``, which the mapper sends to the sim as
  ``move <id of legal.move_slots[k]>`` (by NAME, so the executed move is that slot's move).

The two are related by MOVE-NUM IDENTITY only, never by position. Applying a
request-order tensor to the sorted slots positionally puts move ``k``'s fact on a
DIFFERENT move whenever the moveset is not alphabetical — silent when every move is
legal, wrong exactly on Choice-lock / Taunt / Disable / 0-PP turns. That happened twice:
the old prev-turn mask (fixed by a reorder helper) and then, from gen3_frame_deletion_v1
(bcdd868b) until gen3_move_legality_by_id_v1, the role encoder's per-move-slot legality
(6.8 % of real move-bearing decisions). The model now crosses the orders only through
``agents.model.extractor_ctx.active_request_sorted_match``.

``check_obs_move_order`` is the THROWING guard on the invariants that rule needs, run on
every row the inference service serves. (The switch-ordering check and the TurnDelta move-data
check that read a poke-env-backed ``LiveView`` / delta went with the Python action mapper and
masker, T27 P6 slice 6d-2.)
"""
import functools
from typing import Any, NamedTuple, Optional

import numpy as np

from agents.action.constants import MOVE_START, N_MOVE_SLOTS



class OrderingMismatchError(RuntimeError):
    """Raised when the model's view of move/team ordering disagrees with the
    action space — a data-integrity failure, not a recoverable condition."""


# --- gen3_move_legality_by_id_v1: the row-level move-order guard -------------------------------------
class RowOffsets(NamedTuple):
    """Absolute offsets into one observation row; ``ordering_integrity_test`` pins them against the
    schema and the extractor's own slicer (`slice_pokemon_categoricals`) on real rows."""
    req0: int                    # request block start: ids [4], type ids [4], legal [4]
    req_ids: slice
    req_legal: slice
    team0: int                   # our team block start
    team_size: int
    mon_dim: int
    active_col: int              # the per-mon active flag
    slot_id_cols: np.ndarray     # [4] the per-mon move-slot id columns (SORTED order)
    active_cols: np.ndarray      # [6] absolute columns of our six active flags
    id_cols: np.ndarray          # [6,4] absolute columns of our six mons' sorted move-slot ids


@functools.lru_cache(maxsize=1)
def row_offsets() -> RowOffsets:
    # Imported lazily: `agents.observation`'s package init imports the masker, which imports this module.
    from agents.observation.constants import (ACTIVE_REQ_MOVES_OFFSET, ACTIVE_REQ_MOVES_PER, MOVE_SLOT_DIM,
                                              OFFSET_OUR_TEAM, OFFSET_REACTIVE, POKEMON_ACTIVE_OFFSET,
                                              POKEMON_FULL_DIM, POKEMON_MOVES_OFFSET, TEAM_SIZE)
    r0 = OFFSET_REACTIVE + ACTIVE_REQ_MOVES_OFFSET
    slot_id_cols = np.array([POKEMON_MOVES_OFFSET + j * MOVE_SLOT_DIM for j in range(N_MOVE_SLOTS)])
    mon0 = OFFSET_OUR_TEAM + np.arange(TEAM_SIZE) * POKEMON_FULL_DIM
    return RowOffsets(r0, slice(r0, r0 + ACTIVE_REQ_MOVES_PER),
                      slice(r0 + 2 * ACTIVE_REQ_MOVES_PER, r0 + 3 * ACTIVE_REQ_MOVES_PER),
                      OFFSET_OUR_TEAM, TEAM_SIZE, POKEMON_FULL_DIM, POKEMON_ACTIVE_OFFSET, slot_id_cols,
                      mon0 + POKEMON_ACTIVE_OFFSET, mon0[:, None] + slot_id_cols[None, :])


def check_obs_move_order(obs: Any, mask: Optional[Any] = None, where: str = "obs") -> None:
    """THROWING guard (gen3_move_legality_by_id_v1): raise `OrderingMismatchError` unless every row of
    ``obs`` [n, D] (or one row [D]) satisfies the invariants the model's one cross-order rule
    (`extractor_ctx.active_request_sorted_match`) relies on:

    1. a row naming any request move has exactly ONE active mon on our side;
    2. every nonzero request-slot id matches EXACTLY ONE of our active's sorted per-mon slot ids (the
       identity map is total and unambiguous — an unmatched request move would be scored on a zero token);
    and, given ``mask`` [n, 11] (the action mask the same rows are served with):
    3. request slot ``k``'s legality bit == mask bit ``6+k`` (the feature says what the action allows);
    4. mask bit ``6+k`` set ⇒ request slot ``k`` names a move (a choosable move has an identity) — on
       every row with MORE THAN ONE legal action. A single forced action is exempt: there is nothing to
       choose, and the Hyper Beam RECHARGE turn is exactly that row (request ``[recharge]``, token
       ``move 1``, an id no moveset holds, so the block is all-zero and the head scores a zero token).

    Vectorised numpy, no torch — measured on 37,358 real move-bearing rows (Lane S bank_v1, Mimic and
    Transform included) with zero violations, so a raise is a real misalignment, never noise."""
    o = row_offsets()
    x = np.asarray(obs, dtype=np.float32)
    if x.ndim == 1:
        x = x[None]
    n = x.shape[0]
    req = x[:, o.req_ids]                                                       # [n,4]
    active = x[:, o.active_cols] > 0.5                                          # [n,6]
    has_req = (req > 0).any(axis=1)
    bad = has_req & (active.sum(axis=1) != 1)
    if bad.any():
        i = int(np.flatnonzero(bad)[0])
        raise OrderingMismatchError(
            f"{where} row {i}: the request names moves {req[i].tolist()} but our side has "
            f"{int(active[i].sum())} active mons — the per-mon move slots cannot be located.")
    if has_req.any():
        rows = np.flatnonzero(has_req)
        act = active[rows].argmax(axis=1)
        slot_ids = x[rows[:, None], o.id_cols[act]]                             # [r,4] SORTED order
        r = req[rows]                                                           # [r,4] REQUEST order
        hits = (r[:, :, None] == slot_ids[:, None, :]).sum(axis=2)              # [r,4req]
        bad2 = ((r > 0) & (hits != 1)).any(axis=1)
        if bad2.any():
            j = int(np.flatnonzero(bad2)[0])
            raise OrderingMismatchError(
                f"{where} row {int(rows[j])}: request-order move ids {r[j].tolist()} do not map one-to-one "
                f"onto our active's sorted move slots {slot_ids[j].tolist()} — the identity match the "
                "pointer head and the legality feature read would drop or double a move.")
    if mask is None:
        return
    mv = np.asarray(mask)[:, MOVE_START:MOVE_START + N_MOVE_SLOTS] > 0          # [n,4]
    if mv.shape[0] != n:
        raise OrderingMismatchError(f"{where}: {n} obs rows vs {mv.shape[0]} mask rows")
    legal = x[:, o.req_legal] > 0.5
    bad3 = has_req & (legal != mv).any(axis=1)
    if bad3.any():
        i = int(np.flatnonzero(bad3)[0])
        raise OrderingMismatchError(
            f"{where} row {i}: the request-order legality bits {legal[i].astype(int).tolist()} disagree with "
            f"the action mask's move bits {mv[i].astype(int).tolist()} (request ids {req[i].tolist()}).")
    decision = np.asarray(mask).astype(bool).sum(axis=1) > 1                   # a real choice exists
    bad4 = decision & (mv & ~(req > 0)).any(axis=1)
    if bad4.any():
        i = int(np.flatnonzero(bad4)[0])
        raise OrderingMismatchError(
            f"{where} row {i}: the action mask allows move actions {mv[i].astype(int).tolist()} but the "
            f"request block names only {req[i].tolist()} — a choosable move with no identity in the obs.")


# Moves that legitimately invoke a DIFFERENT move when used — the protocol then
# reports the called move, not the one pressed. Not a mapping error.
CALLER_MOVES = frozenset({
    "sleeptalk", "metronome", "mirrormove", "naturepower", "assist", "copycat",
})
