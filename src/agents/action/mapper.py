from typing import Optional
from poke_env.player.battle_order import SingleBattleOrder, BattleOrder
from poke_env.battle.move import Move

class Gen3ActionMapper:
    """
    Centralized mapping logic for Gen 3 RL actions (0-10).
    Shared between Training (Gen3Env) and Inference (RLPlayer).
    """
    
    @staticmethod
    def action_to_order(
        action: int, 
        battle, 
        mask: Optional[any] = None, 
        latched_turn: int = -1
    ) -> BattleOrder:
        """
        Maps an 11-action discrete index to a poke_env BattleOrder.
        Strictly enforces mapping and validation.
        """
        
        # --- 1. Validation Logic ---
        if mask is not None:
            if latched_turn != -1 and latched_turn != battle.turn:
                raise ValueError(f"STRICT MODE FAILURE: Stale mask detected! Latched: {latched_turn}, Current: {battle.turn}")
            
            if mask[action] == 0:
                raise ValueError(f"STRICT MODE FAILURE: Illegal action {action} requested. Mask: {mask}")

        # --- 2. Action Mapping ---
        # 0-5: Switches (Team Slots 1-6)
        if action < 6:
            team_list = list(battle.team.values())
            if action < len(team_list):
                target_mon = team_list[action]
                if target_mon in battle.available_switches:
                    return SingleBattleOrder(target_mon)
            
            raise ValueError(f"STRICT MODE FAILURE: Switch action {action} invalid for current state.")

        # 6-9: Moves (Slots 1-4)
        elif action < 10:
            move_idx = action - 6
            
            # Preferred Source: Server request (Most accurate for active choices)
            if battle.last_request:
                active_request = battle.last_request.get("active", [{}])[0]
                request_moves = active_request.get("moves", [])
                if move_idx < len(request_moves):
                    move_id = request_moves[move_idx].get("id")
                    # Match move_id to available_moves
                    for move in battle.available_moves:
                        if move.id == move_id:
                            return SingleBattleOrder(move)
                        # Handle Hidden Power variants
                        if move.id.startswith("hiddenpower") and move_id.startswith("hiddenpower"):
                            return SingleBattleOrder(move)
            
            # Secondary Source: Stable sorted move list (for non-standard states)
            active_pokemon = battle.active_pokemon
            if active_pokemon and battle.available_moves:
                stable_ids = sorted(active_pokemon.moves.keys())
                if move_idx < len(stable_ids):
                    m_id = stable_ids[move_idx]
                    for move in battle.available_moves:
                        if move.id == m_id:
                            return SingleBattleOrder(move)
            
            available = [m.id for m in battle.available_moves]
            raise ValueError(f"STRICT MODE FAILURE: Move slot {move_idx} not found in available_moves: {available}")

        # 10: Struggle
        elif action == 10:
            for m in battle.available_moves:
                if m.id == "struggle":
                    return SingleBattleOrder(m)
            
            # Final fallback for struggle
            from poke_env.battle.move import Move
            return SingleBattleOrder(Move("struggle", gen=3))

        raise ValueError(f"UNHANDLED ACTION: {action}")

    @staticmethod
    def _fallback(battle) -> BattleOrder:
        """
        Returns a Default Battle Order, letting the Pokémon Showdown server 
        decide the move (standard default behavior).
        """
        return SingleBattleOrder(None)
