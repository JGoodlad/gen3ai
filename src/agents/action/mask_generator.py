import numpy as np
from poke_env.battle.abstract_battle import AbstractBattle

class Gen3ActionMasker:
    """
    Handles the generation of 11-dimensional action masks for Gen 3 OU.
    Indices:
    0-5: Switches (Team slots 1-6)
    6-9: Moves (Slots 1-4)
    10: Struggle (Dedicated)
    """
    
    @staticmethod
    def get_mask(battle: AbstractBattle) -> np.ndarray:
        """
        Generates a binary mask where 1 is a valid action and 0 is invalid.
        STRICT MODE: Only uses the server request. Crashes on ambiguity.
        """
        if not battle.last_request:
            # We crash if we are asked to mask but have no request context.
            # This is a 'junk in junk out' prevention measure.
            raise RuntimeError("STRICT MODE FAILURE: No last_request found in battle. Cannot mask.")

        mask = np.zeros(11, dtype=np.int8)
        
        # --- 1. Switches (0-5) ---
        team_list = list(battle.team.values())
        
        # Integrity Check: No duplicate species allowed (Gen 3 OU standard)
        species_list = [p.species for p in team_list]
        if len(species_list) != len(set(species_list)):
            raise RuntimeError(f"STRICT MODE FAILURE: Duplicate species detected in team: {species_list}")
            
        for i, pokemon in enumerate(team_list):
            if i < 6:
                # Use object equality check for maximum robustness
                if pokemon in battle.available_switches:
                    mask[i] = 1
                    
        # --- 2. Moves (6-9) ---
        active_request = battle.last_request.get("active", [{}])[0]
        request_moves = active_request.get("moves", [])

        # --- Decision Context Latch ---
        # Pin current move/team ordering onto the battle object so the mapper
        # uses the identical mapping that produced this mask.
        battle._gen3_decision_context = {
            "turn": battle.turn,
            "move_ids": [m.get("id") for m in request_moves],
            "team_species": [p.species for p in team_list],
            "team_objects": team_list,
        }

        # Map request move slots 0-3 to actions 6-9.
        # Struggle is excluded here — it has a dedicated slot (10) below.
        # This prevents the same move being enabled at two different indices.
        for i, move_data in enumerate(request_moves):
            if i < 4 and not move_data.get("disabled", False) and move_data.get("id") != "struggle":
                mask[i + 6] = 1

        # --- 3. Struggle (10) ---
        # Enabled only when the server forces it (all PP depleted).
        if any(m.id == "struggle" for m in battle.available_moves):
            mask[10] = 1
                
        return mask
