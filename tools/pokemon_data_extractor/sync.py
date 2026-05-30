"""Regenerate the gen-N Pokémon data mappings under data/pokemon/.

These mappings are the source of truth consumed by the observation encoders
(src/agents/observation/). They are *derived* from the poke-env static data
shipped in src/poke_env/data/static/ plus the Pokémon Showdown source tree in
deps/pokemon-showdown/. This tool rebuilds them so the derivation is
reproducible instead of a one-off hand edit.

Currently extracts:
  - abilities  -> data/pokemon/gen{N}_abilities.json
  - moves      -> data/pokemon/gen{N}_moves.json   (includes `accuracy`)

Usage:
  python tools/pokemon_data_extractor/sync.py                 # all, gen 3, write files
  python tools/pokemon_data_extractor/sync.py --datasets moves
  python tools/pokemon_data_extractor/sync.py --gen 3 --stdout
"""

import argparse
import json
import os
import re
import sys

# Repo root = two levels up from this file (tools/pokemon_data_extractor/sync.py).
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _static(*parts):
    return os.path.join(REPO_ROOT, "src", "poke_env", "data", "static", *parts)


def to_id_str(name):
    """Converts a string to a Showdown ID (lowercase alphanumeric only)."""
    return "".join(char for char in name.lower() if char.isalnum())


# --------------------------------------------------------------------------- #
# Abilities
# --------------------------------------------------------------------------- #

# Last ability `num` belonging to each generation (order of introduction in the
# games). Anything above the target gen's ceiling did not exist yet and is
# filtered out. Gen 3 ends at Air Lock (76); 77 is Tangled Feet, the first Gen 4
# ability — including it wrongly pulls Tangled Feet onto the Gen 3 Pidgey line.
_GEN_MAX_ABILITY_NUM = {
    1: 0, 2: 0, 3: 76, 4: 123, 5: 164, 6: 191, 7: 233, 8: 267, 9: 1000,
}


def get_ability_id_to_num(abilities_path):
    """Parse abilities.ts into a mapping of ability ID -> official index number."""
    if not os.path.exists(abilities_path):
        return {}

    with open(abilities_path, "r") as f:
        content = f.read()

    # Each ability block looks like:  \n\tabid: {\n ... num: 123, ... }
    matches = re.finditer(r"^\t([a-z0-9]+):\s*\{", content, re.MULTILINE)

    id_to_num = {}
    for match in matches:
        ab_id = match.group(1)
        block = content[match.start():match.start() + 5000]
        num_match = re.search(r"num:\s*(\d+)", block)
        if num_match:
            id_to_num[ab_id] = int(num_match.group(1))

    return id_to_num


def build_abilities(gen):
    """Build the gen-N ability map from the pokedex + Showdown abilities source."""
    pokedex_path = _static("pokedex", f"gen{gen}pokedex.json")
    abilities_path = os.path.join(REPO_ROOT, "deps", "pokemon-showdown", "data", "abilities.ts")

    if not os.path.exists(pokedex_path):
        raise FileNotFoundError(f"Pokedex file not found: {pokedex_path}")

    id_to_num = get_ability_id_to_num(abilities_path)
    if not id_to_num:
        raise FileNotFoundError(f"Could not parse abilities source: {abilities_path}")

    with open(pokedex_path, "r") as f:
        dex = json.load(f)

    max_num = _GEN_MAX_ABILITY_NUM.get(gen, 1000)
    abilities_map = {}

    for mon_id, mon_data in dex.items():
        num = mon_data.get("num", 0)
        if num <= 0:  # Skip CAP Pokémon
            continue

        mon_intro_gen = mon_data.get(
            "gen", 1 if num <= 151 else (2 if num <= 251 else (3 if num <= 386 else 4))
        )
        if mon_intro_gen > gen:
            continue

        # Skip non-base forms (Megas etc.); in Gen 3 only base abilities mattered.
        base_species = mon_data.get("baseSpecies", mon_id)
        if to_id_str(base_species) != mon_id:
            continue

        for slot, ability_name in mon_data.get("abilities", {}).items():
            if slot == "H" and gen < 5:
                continue
            if slot == "S" and gen < 7:
                continue

            ab_id = to_id_str(ability_name)
            ab_num = id_to_num.get(ab_id, 999)
            if ab_num > max_num:  # Did not exist in the target gen
                continue

            if ab_id not in abilities_map:
                abilities_map[ab_id] = {"name": ability_name, "num": ab_num}

    return {ab_id: abilities_map[ab_id] for ab_id in sorted(abilities_map)}


# --------------------------------------------------------------------------- #
# Moves
# --------------------------------------------------------------------------- #

def build_moves(gen):
    """Build the gen-N move map from the poke-env static move data.

    Mirrors the fields the observation encoder relies on. Accuracy is split into
    two fields so `accuracy` is always numeric: the source stores either an int
    percentage (30-100) or the boolean `true` for never-miss moves (Swift, Aerial
    Ace, all status/self moves). We map never-miss to `accuracy: 100` and flag it
    via `never_miss: true`. A 100%-accuracy move can still miss into evasion
    (Double Team) or after Sand-Attack; a never-miss move bypasses the
    accuracy/evasion check entirely — hence the dedicated bit.
    """
    moves_path = _static("moves", f"gen{gen}moves.json")
    if not os.path.exists(moves_path):
        raise FileNotFoundError(f"Moves file not found: {moves_path}")

    with open(moves_path, "r") as f:
        src = json.load(f)

    moves_map = {}
    for move_id, entry in src.items():
        if entry.get("num", 0) <= 0:  # Skip fake/CAP moves (negative nums)
            continue

        raw_accuracy = entry.get("accuracy")
        never_miss = raw_accuracy is True

        moves_map[move_id] = {
            "name": entry.get("name"),
            "num": entry.get("num"),
            "type": entry.get("type"),
            "basePower": entry.get("basePower"),
            "target": entry.get("target"),
            "hasSecondary": bool(entry.get("secondary") or entry.get("secondaries")),
            "hasRecoil": bool(entry.get("recoil")),
            "accuracy": 100 if never_miss else raw_accuracy,
            "never_miss": never_miss,
        }

    return moves_map


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

_BUILDERS = {
    "abilities": ("gen{gen}_abilities.json", build_abilities),
    "moves": ("gen{gen}_moves.json", build_moves),
}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--gen", type=int, default=3, help="Target generation (default: 3)")
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=["abilities", "moves", "all"],
        default=["all"],
        help="Which mappings to regenerate (default: all)",
    )
    parser.add_argument("--stdout", action="store_true", help="Print JSON instead of writing files")
    args = parser.parse_args(argv)

    datasets = list(_BUILDERS) if "all" in args.datasets else args.datasets
    out_dir = os.path.join(REPO_ROOT, "data", "pokemon")

    for name in datasets:
        filename_tmpl, builder = _BUILDERS[name]
        data = builder(args.gen)

        if args.stdout:
            print(json.dumps(data, indent=4))
            continue

        out_path = os.path.join(out_dir, filename_tmpl.format(gen=args.gen))
        with open(out_path, "w") as f:
            json.dump(data, f, indent=4)
            f.write("\n")
        print(f"Wrote {len(data)} {name} -> {os.path.relpath(out_path, REPO_ROOT)}", file=sys.stderr)


if __name__ == "__main__":
    main()
