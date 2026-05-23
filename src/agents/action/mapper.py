import numpy as np
from typing import Optional
from poke_env.player.battle_order import SingleBattleOrder, BattleOrder
from poke_env.battle.move import Move
from agents.action.constants import SWITCH_END, MOVE_START, MOVE_END, STRUGGLE


class Gen3ActionMapper:
    """
    Maps between the 11-action discrete index and poke_env BattleOrder.

    Action space:
      0-5:  Switches (team slots 0-5)
      6-9:  Moves (slots 0-3)
      10:   Struggle

    Shared between training (Gen3Env) and inference (RLPlayer).
    """

    @staticmethod
    def action_to_order(
        action: int,
        battle,
        mask: Optional[np.ndarray] = None,
        latched_turn: int = -1,
    ) -> BattleOrder:
        # --- Mask validation ---
        if mask is not None:
            if latched_turn != -1 and latched_turn != battle.turn:
                raise ValueError(
                    f"Stale mask: latched turn {latched_turn}, current turn {battle.turn}"
                )
            if mask[action] == 0:
                raise ValueError(f"Illegal action {action}. Mask: {mask}")

        context = getattr(battle, "_gen3_decision_context", None)

        # Crash if the server state changed under us between observation and action.
        # The latched context is the ground truth; acting on stale server state
        # would send the wrong move.
        if context and battle.last_request:
            active_request = battle.last_request.get("active", [{}])[0]
            current_move_ids = [m.get("id") for m in active_request.get("moves", [])]
            latched_move_ids = context.get("move_ids", [])
            if latched_move_ids != current_move_ids:
                raise RuntimeError(
                    f"Mid-decision state change at turn {battle.turn}: "
                    f"latched={latched_move_ids}, server={current_move_ids}"
                )

        if context and context.get("turn") != battle.turn:
            context = None

        # 0-5: Switches
        if action < SWITCH_END:
            team_list = context.get("team_objects") if context else list(battle.team.values())
            species_list = context.get("team_species") if context else [p.species for p in team_list]
            if len(species_list) != len(set(species_list)):
                raise RuntimeError(f"Duplicate species in team: {species_list}")
            if action < len(team_list):
                target_mon = team_list[action]
                if target_mon in battle.available_switches:
                    return SingleBattleOrder(target_mon)
            raise ValueError(f"Switch action {action} invalid for current state.")

        # 6-9: Moves
        if action < MOVE_END:
            move_idx = action - MOVE_START
            if not context:
                raise RuntimeError(
                    f"No decision context for move action {action} at turn {battle.turn}. "
                    "Gen3ActionMasker.get_mask() must be called before action_to_order()."
                )
            move_ids = context.get("move_ids", [])
            if move_idx < len(move_ids):
                move_id = move_ids[move_idx]
                for move in battle.available_moves:
                    if move.id == move_id:
                        return SingleBattleOrder(move)
                    if move.id.startswith("hiddenpower") and move_id and move_id.startswith("hiddenpower"):
                        return SingleBattleOrder(move)
            available = [m.id for m in battle.available_moves]
            raise ValueError(f"Move slot {move_idx} not found in available_moves: {available}")

        # Struggle
        if action == STRUGGLE:
            for m in battle.available_moves:
                if m.id == "struggle":
                    return SingleBattleOrder(m)
            return SingleBattleOrder(Move("struggle", gen=3))

        raise ValueError(f"Unhandled action index: {action}")

    @staticmethod
    def validate_context(battle) -> dict:
        """Return the decision context if valid; raise RuntimeError otherwise.

        Centralises the strict check used by both Gen3Player and the mapper so
        the staleness logic lives in one place.
        """
        ctx = getattr(battle, "_gen3_decision_context", None)
        if ctx is None:
            raise RuntimeError(
                f"Decision context is missing at turn {battle.turn}. "
                "get_mask() / embed_battle() must be called before action_to_order()."
            )
        if ctx.get("turn") != battle.turn:
            raise RuntimeError(
                f"Decision context is from turn {ctx.get('turn')} but current turn is "
                f"{battle.turn}. get_mask() / embed_battle() must be called before action_to_order()."
            )
        return ctx

    @staticmethod
    def order_to_action(order: BattleOrder, battle) -> int:
        """Reverse mapping from BattleOrder to action index (0-10)."""
        if not isinstance(order, SingleBattleOrder):
            return 0

        if order.switch:
            team_list = list(battle.team.values())
            for i, mon in enumerate(team_list):
                if mon == order.switch:
                    return i
            return 0

        if order.move:
            if order.move.id == "struggle":
                return 10

            context = getattr(battle, "_gen3_decision_context", None)
            move_ids = context.get("move_ids") if context else None

            if move_ids is None and battle.last_request:
                active_request = battle.last_request.get("active", [{}])[0]
                move_ids = [m.get("id") for m in active_request.get("moves", [])]

            if move_ids:
                for i, mid in enumerate(move_ids):
                    if mid == order.move.id:
                        return i + 6
                    if mid and mid.startswith("hiddenpower") and order.move.id.startswith("hiddenpower"):
                        return i + 6

        return 0
