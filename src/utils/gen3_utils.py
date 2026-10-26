import re
import numpy as np
from typing import Optional, Tuple
from agents.gen3_data.hidden_power import HP_IVS

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
            
        # Normalize 'NONE' strings from various sources
        l_norm = str(last_mon).upper()
        c_norm = str(current_mon).upper()
        
        has_changed = l_norm != c_norm
        is_real = False
        voluntary = None
        
        if has_changed:
            # Case 1: MON -> MON (Standard switch)
            if l_norm not in ["NONE", "NULL", "NONE_P1", "NONE_P2"] and c_norm not in ["NONE", "NULL"]:
                is_real = True
                voluntary = SwitchDetection.is_voluntary(last_mask)
            # Case 2: NONE -> MON (Replacement after faint - ALWAYS FORCED)
            elif l_norm == "NONE" and c_norm not in ["NONE", "NULL"]:
                is_real = True
                voluntary = False
            # Case 3: NULL -> MON (Start of game - IGNORE)
            # Case 4: ANY -> NONE (Faint - IGNORE, wait for replacement)
            
        return has_changed, is_real, voluntary


# The teambuilder's Hidden Power fill: THE one table, agents.gen3_data.hidden_power.HP_IVS (Showdown's validator
# fill, fillStats(typechart.ts HPivs, 31); power 70). A name kept for the teambuilder's callers, not a copy.
GEN3_HP_IVS = HP_IVS

def fix_gen3_hp_ivs(pokemon_list):
    """Detects Hidden Power and applies the correct Gen 3 IVs ONLY if they are all 31."""
    for mon in pokemon_list:
        # Check if mon has a Hidden Power move
        hp_move = next((m for m in mon.moves if "hiddenpower" in m.lower().replace(" ", "")), None)
        if not hp_move:
            # Showdown's validator (team-validator.ts, gen 3: `set.hpType && maxedIVs` -> fillStats(HPivs, 31)) fills
            # the IVs from a `Hidden Power: <Type>` line EVEN WITHOUT a Hidden Power move; the authored team plays
            # those IVs (gen3_team_pack_hptype_ivs_v1, 2026-10-26 GIGO audit C-3).
            hp_type_line = getattr(mon, "hiddenpowertype", None)
            if not hp_type_line:
                continue
            hp_move = f"Hidden Power [{hp_type_line.strip()}]"
            
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
