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
        """
        mask = np.zeros(11, dtype=np.int8)
        
        # --- Switches (0-5) ---
        team_list = list(battle.team.values())
        available_pokemon_species = [p.species for p in battle.available_switches]
        
        for i, pokemon in enumerate(team_list):
            if i < 6: # Gen 3 is 6 mons
                if pokemon.species in available_pokemon_species:
                    mask[i] = 1
                    
        # --- Moves (6-10) ---
        available_moves = battle.available_moves
        available_move_ids = [m.id for m in available_moves]
        active_pokemon = battle.active_pokemon
        
        if active_pokemon:
            # SORT moves by ID to ensure stable mapping across workers
            mon_moves = sorted(active_pokemon.moves.values(), key=lambda m: m.id)[:4]
            mon_move_ids = [m.id for m in mon_moves]
            
            # Normal Moves (6-9)
            for i, move in enumerate(mon_moves):
                if i < 4:
                    if move.id in available_move_ids:
                        mask[i + 6] = 1
            
            # Dedicated Struggle (10)
            if len(available_moves) == 1 and available_moves[0].id == "struggle":
                if available_moves[0].id not in mon_move_ids:
                    mask[10] = 1
        
        # Final safety
        if np.sum(mask) == 0:
            if available_moves:
                mask[6] = 1
            elif battle.available_switches:
                mask[0] = 1
            else:
                mask[6] = 1
                
        return mask
