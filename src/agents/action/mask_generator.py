import numpy as np
from poke_env.battle.abstract_battle import AbstractBattle

class Gen3ActionMasker:
    """
    Handles the generation of 10-dimensional action masks for Gen 3 OU.
    Indices:
    0-5: Switches (Team slots 1-6)
    6-9: Moves (Slots 1-4)
    """
    
    @staticmethod
    def get_mask(battle: AbstractBattle) -> np.ndarray:
        """
        Generates a binary mask where 1 is a valid action and 0 is invalid.
        """
        mask = np.zeros(10, dtype=np.int8)
        
        # --- Switches (0-5) ---
        # Map poke-env available_switches to our 0-5 indices
        # poke-env uses the battle.team dictionary which is ordered.
        team_list = list(battle.team.values())
        available_pokemon = battle.available_switches
        
        for i, pokemon in enumerate(team_list):
            if i < 6: # Gen 3 is 6 mons
                if pokemon in available_pokemon:
                    mask[i] = 1
                    
        # --- Moves (6-9) ---
        available_moves = battle.available_moves
        active_pokemon = battle.active_pokemon
        if active_pokemon:
            mon_moves = list(active_pokemon.moves.values())[:4]
            mon_move_ids = [m.id for m in mon_moves]
            
            # Replicate SinglesEnv logic for struggle
            mvs = (
                available_moves
                if len(available_moves) == 1 and available_moves[0].id not in mon_move_ids
                else mon_moves
            )
            
            for i, move in enumerate(mvs):
                if i < 4:
                    if move in available_moves:
                        mask[i + 6] = 1
        
        # Final safety: Ensure at least one action is valid to avoid NaNs
        if np.sum(mask) == 0:
            # If no moves or switches are valid, and there are available_moves,
            # it might be a special state. Try to pick the first available move.
            if available_moves:
                mask[6] = 1
            elif battle.available_switches:
                mask[0] = 1
            else:
                # Absolute last resort
                mask[0] = 1
                
        return mask
