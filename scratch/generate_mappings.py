from poke_env.data import GenData
import json
import os

def generate_mappings():
    gen3_data = GenData.from_gen(3)
    
    # Species Mapping
    species_map = {}
    # poke-env pokedex includes all gens usually, but we want Gen 3 specific or up to 386.
    # Actually, we just need the IDs for the ones present in Gen 3.
    for species, data in gen3_data.pokedex.items():
        # In poke-env, data usually has 'num'
        if 'num' in data and 0 < data['num'] <= 386:
            species_map[species] = {
                "name": data.get("name", species),
                "num": data['num']
            }
            
    # Move Mapping
    move_map = {}
    for move, data in gen3_data.moves.items():
        if 'num' in data and data['num'] > 0:
            move_map[move] = {
                "name": data.get("name", move),
                "num": data['num']
            }
            
    # Item Mapping
    # Items are trickier as they don't always have 'num' in the same way.
    # We can just assign IDs.
    item_map = {}
    # poke-env items is a dict
    items = sorted(list(gen3_data.items.keys()))
    for i, item in enumerate(items):
        item_map[item] = {
            "name": item,
            "num": i + 1
        }
        
    os.makedirs("data/pokemon", exist_ok=True)
    
    with open("data/pokemon/gen3_species.json", "w") as f:
        json.dump(species_map, f, indent=4)
        
    with open("data/pokemon/gen3_moves.json", "w") as f:
        json.dump(move_map, f, indent=4)
        
    with open("data/pokemon/gen3_items.json", "w") as f:
        json.dump(item_map, f, indent=4)

    print("Mappings generated in data/pokemon/")

if __name__ == "__main__":
    generate_mappings()
