"""Augment data/pokemon/gen3_species.json with each species' Gen 3-valid abilities.

For every species in gen3_species.json, looks up `abilities` from the Showdown
pokedex (slots 0 and 1 only — H/S are Gen 5+/Gen 7+), filters out abilities not
present in gen3_abilities.json (i.e. introduced after Gen 3), and writes the
result back as `"abilities": [id_str_0, id_str_1_or_null]`.

The model uses this to give opponent Pokémon a {known=0, ability1, ability2}
encoding before any ability is revealed — Snorlax becomes "thickfat OR immunity"
instead of "unknown" — so the network can reason about both possibilities.

Run from the repo root:
    python tools/populate_species_abilities.py
"""
from __future__ import annotations

import json
import os
import sys


def to_id_str(name: str) -> str:
    """Showdown ID: lowercase alphanumeric only."""
    return "".join(c for c in name.lower() if c.isalnum())


def main() -> int:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    species_path = os.path.join(repo_root, "data", "pokemon", "gen3_species.json")
    abilities_path = os.path.join(repo_root, "data", "pokemon", "gen3_abilities.json")
    pokedex_path = os.path.join(
        repo_root, "src", "poke_env", "data", "static", "pokedex", "gen3pokedex.json"
    )

    for p in (species_path, abilities_path, pokedex_path):
        if not os.path.exists(p):
            print(f"ERROR: missing {p}", file=sys.stderr)
            return 1

    with open(species_path, "r") as f:
        species = json.load(f)
    with open(abilities_path, "r") as f:
        gen3_abilities = json.load(f)
    with open(pokedex_path, "r") as f:
        pokedex = json.load(f)

    # Set of ability IDs that existed in Gen 3 (everything in the curated file).
    gen3_ability_ids = set(gen3_abilities.keys())

    augmented = 0
    missing_dex = []
    missing_abilities = []
    for sp_id in sorted(species.keys()):
        dex_entry = pokedex.get(sp_id)
        if dex_entry is None:
            missing_dex.append(sp_id)
            continue
        raw_abilities = dex_entry.get("abilities", {})
        # Slots 0 and 1 only — H is Gen 5+, S is Gen 7+
        ordered = []
        for slot in ("0", "1"):
            name = raw_abilities.get(slot)
            if name is None:
                continue
            ab_id = to_id_str(name)
            if ab_id in gen3_ability_ids:
                if ab_id not in ordered:
                    ordered.append(ab_id)

        if not ordered:
            missing_abilities.append(sp_id)
            # Leave the field absent so a downstream lookup raises rather than
            # silently encoding a zero where data should exist.
            continue

        # Pad to exactly 2 slots so the encoder can index ability1/ability2
        # without conditional shape handling. None signals "no second ability."
        if len(ordered) == 1:
            ordered.append(None)
        species[sp_id]["abilities"] = ordered
        augmented += 1

    with open(species_path, "w") as f:
        json.dump(species, f, indent=4, sort_keys=True)
        f.write("\n")

    print(f"Augmented {augmented} / {len(species)} species with Gen 3 abilities.")
    if missing_dex:
        print(f"Species absent from pokedex ({len(missing_dex)}): {missing_dex[:10]}{'...' if len(missing_dex) > 10 else ''}")
    if missing_abilities:
        print(f"Species with no Gen 3-valid ability ({len(missing_abilities)}): {missing_abilities}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
