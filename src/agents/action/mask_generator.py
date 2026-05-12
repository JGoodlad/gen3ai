import numpy as np
import weakref
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
        Maps Actions 6-9 directly to Server Move Slots 1-4 for absolute stability.
        """
        mask = np.zeros(11, dtype=np.int8)
        
        # --- Switches (0-5) ---
        team_list = list(battle.team.values())
        available_pokemon_species = [p.species for p in battle.available_switches]
        
        for i, pokemon in enumerate(team_list):
            if i < 6:
                if pokemon.species in available_pokemon_species:
                    mask[i] = 1
                    
        # --- Moves (6-10) ---
        available_moves = battle.available_moves
        active_pokemon = battle.active_pokemon
        
        if active_pokemon and battle.last_request:
            # Use Server's Native Move Slots (most stable)
            active_request = battle.last_request.get("active", [{}])[0]
            request_moves = active_request.get("moves", [])
            
            # Map slots 0-3 to Actions 6-9
            for i, move_data in enumerate(request_moves):
                if i < 4 and not move_data.get("disabled", False):
                    mask[i + 6] = 1
            
            # Dedicated Struggle (10)
            if any(m.id == "struggle" for m in available_moves):
                mask[10] = 1
        
        # Final safety check: if no moves are masked but some are available, 
        # fall back to basic ID matching (handles edge cases where request is stale)
        if np.sum(mask[6:10]) == 0 and available_moves:
            avail_ids = [m.id for m in available_moves]
            if "struggle" in avail_ids:
                mask[10] = 1
            else:
                # Fallback to alphabetical just in case request is missing
                stable_ids = sorted(active_pokemon.moves.keys()) if active_pokemon else []
                for i, m_id in enumerate(stable_ids):
                    if i < 4 and m_id in avail_ids:
                        mask[i + 6] = 1
                
        return mask
