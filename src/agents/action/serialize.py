"""The ONE poke-env touch in the action path: turn a :class:`Choice` into the
``BattleOrder`` the Showdown client actually sends.

Everything upstream of this — the action mask and the action→:class:`Choice` decode —
is poke-env-free and driven by our captured :class:`~agents.battle.live_view.LegalActions`
snapshot. This adapter is the only place that imports poke-env order/move types, so the
serialization seam (and the struggle ``BattleOrder``) is confined to a single function.

The resolution mirrors the historical ``Gen3ActionMapper.action_to_order`` exactly; the
index/legality logic already happened in ``action_to_choice``, so all that remains here is
looking the chosen move / bench mon up among poke-env's live objects.
"""

from __future__ import annotations

from poke_env.battle.move import Move
from poke_env.battle.pokemon import Pokemon
from poke_env.player.battle_order import BattleOrder, SingleBattleOrder

from agents.action.choice import Choice, ChoiceKind
from agents.action.constants import MOVE_START, STRUGGLE
from agents.battle.live_view import LegalActions


def choice_to_order(choice: Choice, battle) -> BattleOrder:
    """Serialize a decoded :class:`Choice` into a poke-env ``BattleOrder`` against the
    live ``battle`` objects. Raises ``ValueError`` if the choice can't be resolved to a
    real move / bench mon (a desync between our snapshot and poke-env's live state)."""
    if choice.kind is ChoiceKind.SWITCH:
        team_list = list(battle.team.values())
        slot = choice.switch_slot
        if slot is not None and slot < len(team_list):
            target_mon = team_list[slot]
            if target_mon in battle.available_switches:
                return SingleBattleOrder(target_mon)
        raise ValueError(
            f"Switch slot {slot} ({choice.switch_species}) is not a currently "
            f"available switch."
        )

    if choice.kind is ChoiceKind.MOVE:
        move_id = choice.move_id
        for move in battle.available_moves:
            if move.id == move_id:
                return SingleBattleOrder(move)
            # Hidden Power: the request keys it bare ("hiddenpower") while poke-env
            # resolves available_moves to the typed id ("hiddenpowerice"). One move.
            if (move.id.startswith("hiddenpower")
                    and move_id and move_id.startswith("hiddenpower")):
                return SingleBattleOrder(move)
        available = [m.id for m in battle.available_moves]
        raise ValueError(
            f"Move '{move_id}' (slot {choice.move_slot}) not found in "
            f"available_moves: {available}"
        )

    if choice.kind is ChoiceKind.STRUGGLE:
        for m in battle.available_moves:
            if m.id == "struggle":
                return SingleBattleOrder(m)
        return SingleBattleOrder(Move("struggle", gen=3))

    raise ValueError(f"Unhandled choice kind: {choice.kind!r}")


def order_to_action(order: BattleOrder, battle, legal=None) -> int:
    """Reverse the boundary: a poke-env ``BattleOrder`` → the 11-dim action index.

    Best-effort (returns 0 on anything unrecognized) — used for diagnostics / opponent
    labelling AND for the send-path round-trip conformance check. Move ordering is sourced
    from the server-authoritative :class:`LegalActions` snapshot (request order) so it
    matches the action space the mask/decoder use; pass ``legal`` to reuse the captured
    per-decision snapshot (otherwise a fresh one is taken from ``battle``).
    """
    if not isinstance(order, SingleBattleOrder):
        return 0

    # A SingleBattleOrder carries its payload on `.order` (a Pokemon for a switch, a
    # Move for a move) — NOT `.switch`/`.move`. Dispatch on the payload type.
    target = order.order

    if isinstance(target, Pokemon):
        for i, mon in enumerate(battle.team.values()):
            if mon == target:
                return i
        return 0

    if isinstance(target, Move):
        if target.id == "struggle":
            return STRUGGLE
        move_ids = (legal or LegalActions.from_battle(battle)).move_ids
        for i, mid in enumerate(move_ids):
            if mid == target.id:
                return MOVE_START + i
            if (mid and mid.startswith("hiddenpower")
                    and target.id.startswith("hiddenpower")):
                return MOVE_START + i

    return 0
