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
every row the inference service serves and every row the Python encoder builds.
"""
import functools
from typing import TYPE_CHECKING, Any, NamedTuple, Optional

import numpy as np

from agents.action.constants import MOVE_START, N_MOVE_SLOTS, SWITCH_END

if TYPE_CHECKING:
    from agents.battle.live_view import LiveView


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


def _same_move(a, b) -> bool:
    if a == b:
        return True
    # Hidden Power reports as type-suffixed ids in some paths; treat as one move.
    return bool(a and b and a.startswith("hiddenpower") and b.startswith("hiddenpower"))


def check_move_data_consistent(delta) -> None:
    """Training-path data-integrity guard for the recorded move identity.

    `TurnDelta.our_move_id` is derived from the protocol last_move (immune to the
    action-bookkeeping desync, and delegation-aware). This asserts it agrees with
    the *independent* protocol source — the DamagingMoveEvent captured at |move|
    parse time — whenever a damaging move resolved. A disagreement means the two
    protocol parse points contradict each other (corrupted capture), not a
    recoverable condition. Non-damaging moves have no event and are skipped.
    """
    ev = getattr(delta, "our_damaging_event", None)
    mv = getattr(delta, "our_move_id", None)
    if ev is None or mv is None:
        return
    if getattr(delta, "our_switch_to", None) is not None:
        return
    if not _same_move(ev.move_id, mv):
        raise OrderingMismatchError(
            f"Move-data inconsistency on turn {getattr(delta, 'turn', '?')}: "
            f"our_move_id='{mv}' (from protocol last_move) disagrees with the "
            f"DamagingMoveEvent move_id='{ev.move_id}' (captured at |move| parse). "
            f"The two protocol sources contradict each other."
        )


def check_switch_ordering_alignment(live: "LiveView", mask: np.ndarray, legal) -> None:
    """Assert the team ordering the mask/mapper used equals the ordering the
    feature extractor consumes, so switch action index *i*, switch-validity bit
    *i*, and per-Pokémon obs slot *i* all refer to the same Pokémon.

    Unlike moves, our team has no sort step — every consumer uses the
    ``list(battle.team.values())`` order (the encoder via
    ``ObservationEncoder.get_team_list(is_opponent=False)``, mirrored here by
    ``live.ours.mons``; the masker/mapper via the ``LegalActions`` snapshot's
    slot-indexed switches). This check guarantees that stays true: if the two ever
    diverge (a future reorder, or the team mutating between snapshot and check) a switch
    could silently target the wrong mon, so we crash instead. ``live`` is the
    current-board :class:`LiveView`; ``legal`` is the per-decision :class:`LegalActions`,
    each of whose switches names the species AND the team slot the action space maps it to.
    """
    if legal is None:
        return
    # The encoder's team order (what per-Pokémon slots + switch validity index).
    encoder_team = [m.species for m in live.ours.mons]
    for sw in legal.switches:
        if sw.slot >= SWITCH_END or sw.slot >= len(encoder_team):
            continue
        if encoder_team[sw.slot] != sw.species:
            raise OrderingMismatchError(
                f"Team/switch ordering mismatch on turn {live.turn}: "
                f"the action space maps switch slot {sw.slot} to '{sw.species}', but the "
                f"feature extractor's per-Pokémon slot {sw.slot} is "
                f"'{encoder_team[sw.slot]}'. Switch action {sw.slot} would target a "
                f"different mon than the model evaluated at that slot."
            )
