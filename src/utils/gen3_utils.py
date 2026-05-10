import re

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
            print(f"Fixing IVs for {mon.species} with {hp_move}...")
            indices = {"hp": 0, "atk": 1, "def": 2, "spa": 3, "spd": 4, "spe": 5}
            for stat, value in GEN3_HP_IVS[hp_type].items():
                mon.ivs[indices[stat]] = value
    return pokemon_list
