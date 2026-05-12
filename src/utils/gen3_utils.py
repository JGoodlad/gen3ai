import re
import numpy as np
from typing import Optional, Tuple

class SwitchDetection:
    """
    Centralized logic for identifying and categorizing Pokémon switches.
    Ensures consistency between reward calculation and diagnostic logging.
    """
    
    @staticmethod
    def is_voluntary(mask: Optional[np.ndarray]) -> Optional[bool]:
        """
        Determines if a switch was voluntary based on the action mask.
        Returns:
            True: Moves/Struggle were available (Voluntary choice)
            False: Only switches/nothing available (Forced/Faint)
            None: Mask missing (Unknown status)
        """
        if mask is None:
            return None
        # 6-9 are moves, 10 is struggle.
        return any(mask[6:11] == 1)

    @staticmethod
    def get_switch_type(last_mon: str, current_mon: str, last_mask: Optional[np.ndarray]) -> Tuple[bool, bool, Optional[bool]]:
        """
        Analyzes a transition between two Pokémon.
        
        Returns:
            is_switch (bool): True if species changed.
            is_real (bool): True if neither side is 'NONE' (filters faints/transitions).
            is_voluntary (Optional[bool]): True (Voluntary), False (Forced), None (Unknown).
        """
        if not last_mon or not current_mon:
            return False, False, None
            
        has_changed = last_mon != current_mon
        
        # Normalize 'NONE' strings from various sources
        l_norm = str(last_mon).upper()
        c_norm = str(current_mon).upper()
        is_real = l_norm != "NONE" and c_norm != "NONE" and l_norm != "NULL" and c_norm != "NULL"
        
        voluntary = None
        if has_changed and is_real:
            voluntary = SwitchDetection.is_voluntary(last_mask)
            
        return has_changed, is_real, voluntary


# Single source of truth for Gen 3 Hidden Power IVs (ensures 70 Power)
GEN3_HP_IVS = {
    "Fighting": {"hp": 31, "atk": 31, "def": 30, "spa": 30, "spd": 30, "spe": 30},
    "Rock":     {"hp": 31, "atk": 31, "def": 30, "spa": 31, "spd": 30, "spe": 30},
    "Fire":     {"hp": 31, "atk": 30, "def": 31, "spa": 30, "spd": 31, "spe": 30},
    "Psychic":  {"hp": 31, "atk": 30, "def": 31, "spa": 31, "spd": 31, "spe": 30},
    "Flying":   {"hp": 30, "atk": 30, "def": 30, "spa": 30, "spd": 30, "spe": 31},
    "Poison":   {"hp": 31, "atk": 31, "def": 30, "spa": 30, "spd": 30, "spe": 31},
    "Ground":   {"hp": 31, "atk": 31, "def": 31, "spa": 30, "spd": 30, "spe": 31},
    "Bug":      {"hp": 31, "atk": 30, "def": 30, "spa": 31, "spd": 30, "spe": 31},
    "Ghost":    {"hp": 31, "atk": 30, "def": 31, "spa": 31, "spd": 30, "spe": 31},
    "Steel":    {"hp": 31, "atk": 31, "def": 31, "spa": 31, "spd": 30, "spe": 31},
    "Water":    {"hp": 31, "atk": 30, "def": 30, "spa": 30, "spd": 31, "spe": 31},
    "Grass":    {"hp": 31, "atk": 30, "def": 31, "spa": 30, "spd": 31, "spe": 31},
    "Electric": {"hp": 31, "atk": 31, "def": 31, "spa": 30, "spd": 31, "spe": 31},
    "Ice":      {"hp": 31, "atk": 30, "def": 30, "spa": 31, "spd": 31, "spe": 31},
    "Dragon":   {"hp": 31, "atk": 30, "def": 31, "spa": 31, "spd": 31, "spe": 31},
    "Dark":     {"hp": 31, "atk": 31, "def": 31, "spa": 31, "spd": 31, "spe": 31},
}

def fix_gen3_hp_ivs(pokemon_list):
    """Detects Hidden Power and applies the correct Gen 3 IVs ONLY if they are all 31."""
    for mon in pokemon_list:
        # Check if mon has a Hidden Power move
        hp_move = next((m for m in mon.moves if "hiddenpower" in m.lower().replace(" ", "")), None)
        if not hp_move:
            continue
            
        # Only fix if current IVs are default (all 31)
        if mon.ivs and any(iv != 31 for iv in mon.ivs):
            continue

        if mon.ivs is None or len(mon.ivs) == 0:
            mon.ivs = [31] * 6
            
        normalized = hp_move.lower().replace(" ", "").replace("[", "").replace("]", "")
        match = re.search(r"\[(\w+)\]", hp_move)
        hp_type = match.group(1).capitalize() if match else normalized.replace("hiddenpower", "").capitalize()
        
        if hp_type in GEN3_HP_IVS:
            indices = {"hp": 0, "atk": 1, "def": 2, "spa": 3, "spd": 4, "spe": 5}
            for stat, value in GEN3_HP_IVS[hp_type].items():
                mon.ivs[indices[stat]] = value
    return pokemon_list
