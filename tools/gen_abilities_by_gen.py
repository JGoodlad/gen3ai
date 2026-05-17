import json
import os
import re
import sys

def to_id_str(name):
    """Converts a string to a Showdown ID (lowercase alphanumeric only)."""
    return "".join([char for char in name.lower() if char.isalnum()])

def get_ability_id_to_num(abilities_path):
    """Parses abilities.ts to get a mapping of ability ID to its official index number."""
    if not os.path.exists(abilities_path):
        return {}
    
    with open(abilities_path, "r") as f:
        content = f.read()
    
    # Regex to find all ability blocks: abid: { ... num: 123, ... }
    # We look for the key (abid) and then the num within that block.
    # Matches: \n\tabid: {\n (with optional spaces)
    matches = re.finditer(r'^\t([a-z0-9]+):\s*\{', content, re.MULTILINE)
    
    id_to_num = {}
    for match in matches:
        ab_id = match.group(1)
        # Find the next few lines to get the 'num' field
        block = content[match.start():match.start()+5000]
        num_match = re.search(r'num:\s*(\d+)', block)
        if num_match:
            id_to_num[ab_id] = int(num_match.group(1))
            
    return id_to_num

def get_abilities_for_gen(gen, pokedex_path, id_to_num):
    if not os.path.exists(pokedex_path):
        print(f"Error: Pokedex file not found: {pokedex_path}", file=sys.stderr)
        return {}

    with open(pokedex_path, "r") as f:
        dex = json.load(f)

    # Max ability number for each generation
    # These are based on the order of introduction in the games
    gen_max_ab_num = {
        1: 0,
        2: 0,
        3: 77,
        4: 123,
        5: 164,
        6: 191,
        7: 233,
        8: 267,
        9: 1000 # All current
    }
    
    max_num = gen_max_ab_num.get(gen, 1000)
    abilities_map = {}

    for mon_id, mon_data in dex.items():
        num = mon_data.get("num", 0)
        
        # Skip CAP Pokemon
        if num <= 0: continue
            
        # Skip species introduced after target gen
        mon_intro_gen = mon_data.get("gen", 1 if num <= 151 else (2 if num <= 251 else (3 if num <= 386 else 4)))
        if mon_intro_gen > gen: continue
            
        # Skip non-base forms (Megas, etc.) if they didn't exist in that gen
        base_species = mon_data.get("baseSpecies", mon_id)
        if to_id_str(base_species) != mon_id:
            # Note: Castform forms exist in Gen 3 but usually share abilities.
            # Most Gen 3 forms (Castform, Deoxys) are handled by base species or separate entries.
            # However, Deoxys-Attack etc. have the SAME num as Deoxys.
            # We only care about unique abilities, so skipping non-base species is generally fine
            # unless a form has a DIFFERENT ability that existed in that gen.
            # In Gen 3, only base abilities mattered.
            continue
            
        for slot, ability_name in mon_data.get("abilities", {}).items():
            # Filter slots by gen
            if slot == "H" and gen < 5: continue
            if slot == "S" and gen < 7: continue
                
            ab_id = to_id_str(ability_name)
            
            # CRITICAL: Verify the ability existed in the target generation
            ab_num = id_to_num.get(ab_id, 999)
            if ab_num > max_num:
                continue
                
            if ab_id not in abilities_map:
                abilities_map[ab_id] = {
                    "name": ability_name,
                    "num": ab_num
                }

    # Order by ID
    sorted_ids = sorted(abilities_map.keys())
    return {ab_id: abilities_map[ab_id] for ab_id in sorted_ids}

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/gen_abilities_by_gen.py <generation_number>")
        sys.exit(1)
        
    try:
        target_gen = int(sys.argv[1])
    except ValueError:
        print("Error: Generation must be an integer.")
        sys.exit(1)

    # Paths
    pokedex_file = f"deps/venv/lib/python3.11/site-packages/poke_env/data/static/pokedex/gen{target_gen}pokedex.json"
    abilities_file = "deps/pokemon-showdown/data/abilities.ts"
    
    if not os.path.exists(pokedex_file):
        print(f"Error: Pokedex file not found: {pokedex_file}", file=sys.stderr)
        sys.exit(1)

    id_to_num = get_ability_id_to_num(abilities_file)
    abilities = get_abilities_for_gen(target_gen, pokedex_file, id_to_num)
    
    if abilities:
        print(json.dumps(abilities, indent=4))
    else:
        print(f"No abilities found for Gen {target_gen}.", file=sys.stderr)
