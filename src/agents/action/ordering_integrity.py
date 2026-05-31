"""Live integrity checks that the ordering the *model* sees matches the ordering
the *action space* uses.

Background: the feature extractor reads each Pokémon's move slots in
sorted-by-id order (``MovesEncoder`` -> ``get_sorted_moves``), and applies the
previous-turn move-validity mask to those slots positionally. The action mask /
mapper, however, index moves in *request order* (the order Pokémon Showdown sends
them, captured in the per-decision ``LegalActions`` snapshot's ``move_ids``).

When those two orderings differ (i.e. the moveset isn't already alphabetical)
the validity bit for action slot ``k`` lands on the embedding of a *different*
move. It is silent when every move is legal (permuting all-ones changes nothing)
but wrong exactly on disabled / zero-PP / Taunt / Disable / Choice-locked turns —
precisely the degenerate states where pathological behaviour shows up.

These checks turn that whole class of bug into a loud, immediate
``OrderingMismatchError`` instead of silently feeding the model scrambled
validity. They are cheap (a few short-list comparisons) and run on every masked
turn, in both training and inference.
"""
import numpy as np
from poke_env.battle.abstract_battle import AbstractBattle

from agents.action.constants import MOVE_START, N_MOVE_SLOTS, SWITCH_END


class OrderingMismatchError(RuntimeError):
    """Raised when the model's view of move/team ordering disagrees with the
    action space — a data-integrity failure, not a recoverable condition."""


def reorder_move_bits_to_sorted(action_mask: np.ndarray, action_order_ids: list) -> np.ndarray:
    """Return a copy of the 11-dim action mask whose MOVE bits (slots 6-9) are
    reordered from action/request order into sorted-by-id order — matching the
    order the feature extractor reads the active mon's move slots
    (``MovesEncoder`` -> ``get_sorted_moves``).

    Switch bits (0-5) and the struggle bit (10) are left untouched: switches
    already share one team ordering and struggle is a scalar.

    ``action_order_ids`` is the active mon's move ids in request order (e.g.
    ``BattleContext.active_move_ids`` / the masker's pinned ``move_ids``).
    """
    out = np.asarray(action_mask, dtype=np.float32).copy()
    ids = [mid for mid in (action_order_ids or []) if mid]
    if not ids:
        return out  # forced switch / forced struggle — no per-move slots
    out[MOVE_START:MOVE_START + N_MOVE_SLOTS] = 0.0
    for sorted_slot, mid in enumerate(sorted(ids)):
        if sorted_slot >= N_MOVE_SLOTS:
            break
        action_slot = action_order_ids.index(mid)
        out[MOVE_START + sorted_slot] = action_mask[MOVE_START + action_slot]
    return out


def assert_sorted_validity_correct(
    reordered_mask: np.ndarray, action_mask: np.ndarray, action_order_ids: list
) -> None:
    """Validate that ``reordered_mask`` carries, in sorted-move order, exactly the
    per-move legality the action-order ``action_mask`` encodes. This is the
    invariant the extractor relies on (it applies reordered bit *i* to sorted
    slot *i*). Raises ``OrderingMismatchError`` on any disagreement."""
    ids = [mid for mid in (action_order_ids or []) if mid]
    for sorted_slot, mid in enumerate(sorted(ids)):
        if sorted_slot >= N_MOVE_SLOTS:
            break
        action_slot = action_order_ids.index(mid)
        got = int(reordered_mask[MOVE_START + sorted_slot])
        want = int(action_mask[MOVE_START + action_slot])
        if got != want:
            raise OrderingMismatchError(
                f"Sorted move-validity is wrong for '{mid}': the model would see "
                f"legality={got} on sorted slot {sorted_slot}, but its true legality "
                f"is {want}. sorted_order={sorted(ids)} action_order={list(action_order_ids)}."
            )


def _sorted_move_ids(mon) -> list:
    """Mirror exactly what ``MovesEncoder`` feeds the extractor: the mon's moves
    sorted by id (``ObservationEncoder.get_sorted_moves``)."""
    if mon is None or not getattr(mon, "moves", None):
        return []
    return [m.id for m in sorted(mon.moves.values(), key=lambda m: m.id)]


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


def check_switch_ordering_alignment(battle: AbstractBattle, mask: np.ndarray, legal) -> None:
    """Assert the team ordering the mask/mapper used equals the ordering the
    feature extractor consumes, so switch action index *i*, switch-validity bit
    *i*, and per-Pokémon obs slot *i* all refer to the same Pokémon.

    Unlike moves, our team has no sort step — every consumer uses
    ``list(battle.team.values())`` (the encoder via
    ``ObservationEncoder.get_team_list(is_opponent=False)``, the masker/mapper via
    the ``LegalActions`` snapshot's slot-indexed switches). This check guarantees that
    stays true: if the two ever diverge (a future reorder, or the team dict mutating
    between snapshot and check) a switch could silently target the wrong mon, so we
    crash instead. ``legal`` is the per-decision :class:`LegalActions`; each of its
    switches names the species AND the team slot the action space maps it to.
    """
    if legal is None:
        return
    # The encoder's team order (what per-Pokémon slots + switch validity index).
    encoder_team = [getattr(p, "species", None) for p in list(battle.team.values())]
    for sw in legal.switches:
        if sw.slot >= SWITCH_END or sw.slot >= len(encoder_team):
            continue
        if encoder_team[sw.slot] != sw.species:
            raise OrderingMismatchError(
                f"Team/switch ordering mismatch on turn {getattr(battle, 'turn', '?')}: "
                f"the action space maps switch slot {sw.slot} to '{sw.species}', but the "
                f"feature extractor's per-Pokémon slot {sw.slot} is "
                f"'{encoder_team[sw.slot]}'. Switch action {sw.slot} would target a "
                f"different mon than the model evaluated at that slot."
            )


def check_move_validity_alignment(battle: AbstractBattle, mask: np.ndarray, legal) -> None:
    """Assert the per-move legality the feature extractor applies (by sorted
    slot) equals the move's true legality from the action mask (by request slot).

    Raises ``OrderingMismatchError`` on any disagreement. ``legal`` is the per-decision
    :class:`LegalActions` snapshot whose ``move_ids`` are in request/action order.
    """
    active = getattr(battle, "active_pokemon", None)
    if active is None:
        return
    if legal is None:
        return
    request_ids = list(legal.move_ids)
    if not request_ids:
        return  # forced switch / forced struggle — no per-move slots in play

    # True legality keyed by move id, from the action-order mask the masker built.
    legal_by_id = {}
    for k, mid in enumerate(request_ids):
        if k >= N_MOVE_SLOTS:
            break
        if mid:
            legal_by_id[mid] = int(mask[MOVE_START + k])

    # The extractor applies mask[MOVE_START + k] to the move at SORTED slot k.
    sorted_ids = _sorted_move_ids(active)
    for k, mid in enumerate(sorted_ids):
        if k >= N_MOVE_SLOTS:
            break
        true_legal = legal_by_id.get(mid)
        if true_legal is None:
            continue  # move not in the request (e.g. struggle) — nothing to compare
        applied = int(mask[MOVE_START + k])
        if applied != true_legal:
            raise OrderingMismatchError(
                f"Move-validity ordering mismatch on turn {getattr(battle, 'turn', '?')}: "
                f"the model sees legality={applied} on sorted slot {k} ('{mid}'), "
                f"but that move's true legality is {true_legal}. "
                f"sorted_order={sorted_ids[:N_MOVE_SLOTS]} "
                f"action_order={[m for m in request_ids[:N_MOVE_SLOTS]]}. "
                f"The prev-turn move mask is applied positionally without remapping "
                f"action-order -> sorted-order (features_extractor.py move_validity)."
            )
