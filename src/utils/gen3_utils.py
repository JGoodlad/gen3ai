import re

# Single source of truth for Gen 3 Hidden Power IVs (ensures 70 Power)
GEN3_HP_IVS = {
    "Bug":      {"atk": 30, "def": 30, "spd": 30},
    "Dark":     {}, # All 31s
    "Dragon":   {"spd": 30},
    "Electric": {"spa": 30, "spd": 30},
    "Fighting": {"atk": 30, "def": 30, "spa": 30, "spd": 30, "spe": 30},
    "Fire":     {"atk": 30, "spa": 30, "spe": 30},
    "Flying":   {"hp": 30, "atk": 30, "def": 30, "spa": 30, "spd": 30},
    "Ghost":    {"def": 30, "spd": 30},
    "Grass":    {"spa": 30},
    "Ground":   {"spa": 30, "spd": 30, "spe": 30},
    "Ice":      {"atk": 30, "def": 30},
    "Poison":   {"hp": 30, "atk": 30, "def": 30, "spa": 30},
    "Psychic":  {"atk": 30, "spe": 30},
    "Rock":     {"atk": 30, "def": 30, "spa": 30, "spd": 30},
    "Steel":    {"atk": 30, "def": 30, "spd": 30},
    "Water":    {"atk": 30, "def": 30, "spa": 30},
}

def fix_gen3_hp_ivs(pokemon_list):
    """Detects Hidden Power and applies the correct Gen 3 IVs from the mapping above."""
    for mon in pokemon_list:
        if mon.ivs is None or len(mon.ivs) == 0:
            mon.ivs = [31] * 6
            
        for move in mon.moves:
            # Handle formats: "Hidden Power [Bug]", "hiddenpowerbug"
            normalized = move.lower().replace(" ", "").replace("[", "").replace("]", "")
            if "hiddenpower" in normalized:
                match = re.search(r"\[(\w+)\]", move)
                hp_type = match.group(1).capitalize() if match else normalized.replace("hiddenpower", "").capitalize()
                
                if hp_type in GEN3_HP_IVS:
                    indices = {"hp": 0, "atk": 1, "def": 2, "spa": 3, "spd": 4, "spe": 5}
                    for stat, value in GEN3_HP_IVS[hp_type].items():
                        mon.ivs[indices[stat]] = value
    return pokemon_list
