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
        # Retrieve the latched context if available
        context = getattr(battle, "_gen3_decision_context", None)
        
        # --- FORENSIC INTEGRITY CHECK ---
        # Compare the pinned state (what the model saw) to the current server state
        if context and battle.last_request:
            active_request = battle.last_request.get("active", [{}])[0]
            current_move_ids = [m.get("id") for m in active_request.get("moves", [])]
            latched_move_ids = context.get("move_ids", [])
            
            if latched_move_ids != current_move_ids:
                # We detected a silent update! 
                # Instead of just fixing it, we report the discrepancy for diagnostics.
                report = (
                    f"\n🛑 STRICT MODE INTEGRITY FAILURE: Mid-Decision State Change Detected!\n"
                    f"  Turn: {battle.turn}\n"
                    f"  Pinned (Observation): {latched_move_ids}\n"
                    f"  Current (Server):      {current_move_ids}\n"
                    f"  Reason: The server state updated while the model was executing.\n"
                    f"  Action: Using the pinned context to ensure mapping integrity, but reporting for forensics."
                )
                raise RuntimeError(report)
        
        if context and context.get("turn") != battle.turn:
            # If the context is from a previous turn, it's stale
            context = None

        # 0-5: Switches (Team Slots 1-6)
        if action < 6:
            # Use latched team if available, otherwise fall back to current
            team_list = context.get("team_objects") if context else list(battle.team.values())
            
            # Integrity Check: No duplicate species allowed
            species_list = context.get("team_species") if context else [p.species for p in team_list]
            if len(species_list) != len(set(species_list)):
                raise RuntimeError(f"STRICT MODE FAILURE: Duplicate species detected in team: {species_list}")
                
            if action < len(team_list):
                target_mon = team_list[action]
                if target_mon in battle.available_switches:
                    return SingleBattleOrder(target_mon)
            
            raise ValueError(f"STRICT MODE FAILURE: Switch action {action} invalid for current state.")

        # 6-9: Moves (Slots 1-4)
        elif action < 10:
            move_idx = action - 6
            
            # Use latched move IDs if available
            if context:
                move_ids = context.get("move_ids", [])
                if move_idx < len(move_ids):
                    move_id = move_ids[move_idx]
                    # Match move_id to available_moves
                    for move in battle.available_moves:
                        if move.id == move_id:
                            return SingleBattleOrder(move)
                        # Handle Hidden Power variants
                        if move.id.startswith("hiddenpower") and (move_id and move_id.startswith("hiddenpower")):
                            return SingleBattleOrder(move)
            
            # Fallback to current request if no context (should not happen in training)
            if not context and battle.last_request:
                active_request = battle.last_request.get("active", [{}])[0]
                request_moves = active_request.get("moves", [])
                if move_idx < len(request_moves):
                    move_id = request_moves[move_idx].get("id")
                    for move in battle.available_moves:
                        if move.id == move_id:
                            return SingleBattleOrder(move)
            
            available = [m.id for m in battle.available_moves]
            raise ValueError(f"STRICT MODE FAILURE: Move slot {move_idx} not found in available_moves: {available}")

        # 10: Struggle
        elif action == 10:
            for m in battle.available_moves:
                if m.id == "struggle":
                    return SingleBattleOrder(m)
            
            # Final fallback for struggle (in case server request is slightly desynced)
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
