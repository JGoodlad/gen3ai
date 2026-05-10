import re

# Single source of truth for Gen 3 Hidden Power IVs (ensures 70 Power)
# Authoritative Gen 3 Hidden Power IVs for Power 70
# Each entry is (HP, Atk, Def, SpA, SpD, Spe)
GEN3_HP_IVS = {
    "Fighting": [31, 30, 30, 30, 30, 30],
    "Flying":   [31, 30, 31, 30, 30, 30],
    "Poison":   [31, 30, 30, 30, 30, 31],
    "Ground":   [31, 30, 31, 30, 30, 31],
    "Rock":     [31, 30, 30, 31, 30, 30],
    "Bug":      [30, 31, 31, 31, 30, 30],
    "Ghost":    [30, 31, 30, 31, 30, 31],
    "Steel":    [30, 31, 31, 31, 30, 31],
    "Fire":     [30, 31, 30, 30, 31, 30],
    "Water":    [30, 31, 31, 30, 31, 30],
    "Grass":    [31, 31, 30, 30, 31, 31],
    "Electric": [31, 31, 31, 30, 31, 31],
    "Psychic":  [31, 31, 30, 31, 31, 30],
    "Ice":      [31, 31, 31, 31, 31, 30],
    "Dragon":   [31, 31, 30, 31, 31, 31],
    "Dark":     [30, 30, 30, 30, 30, 30],
}

def fix_gen3_hp_ivs(pokemon_list):
    """Detects Hidden Power and applies correct Gen 3 IVs ONLY if missing or default."""
    for mon in pokemon_list:
        # Check if the pokemon has Hidden Power
        hp_type = None
        for move in mon.moves:
            normalized = move.lower().replace(" ", "").replace("[", "").replace("]", "")
            if "hiddenpower" in normalized:
                match = re.search(r"\[(\w+)\]", move)
                hp_type = match.group(1).capitalize() if match else normalized.replace("hiddenpower", "").capitalize()
                break
        
        if not hp_type:
            continue
            
        # Check if IVs are default (all 31s or None/Empty)
        is_default = mon.ivs is None or len(mon.ivs) == 0 or all(iv == 31 for iv in mon.ivs)
        
        if is_default:
            if mon.ivs is None or len(mon.ivs) == 0:
                mon.ivs = [31] * 6
                
            if hp_type in GEN3_HP_IVS:
                mon.ivs = list(GEN3_HP_IVS[hp_type])
                
    return pokemon_list
