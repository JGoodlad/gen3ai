"""
Compute per-species Hidden Power type probability distributions from Smogon Gen3 OU stats.

Reads data/pokemon/gen3_smogon_stats.json (chaos format, downloaded by
tools/smogon_stats_downloader/sync.py) and writes
data/pokemon/gen3_hidden_power_priors.json keyed by lowercase species name.
"""
import json
import os
import sys
from collections import defaultdict
from typing import Optional

# Canonical 16 hidden power types (alphabetical, matches design doc encoding order)
HIDDEN_POWER_TYPES = frozenset([
    "bug", "dark", "dragon", "electric", "fighting", "fire", "flying", "ghost",
    "grass", "ground", "ice", "poison", "psychic", "rock", "steel", "water",
])

HIDDEN_POWER_PREFIX = "hiddenpower"

STATS_PATH = "data/pokemon/gen3_smogon_stats.json"
SPECIES_PATH = "data/pokemon/gen3_species.json"
OUTPUT_PATH = "data/pokemon/gen3_hidden_power_priors.json"


def extract_hidden_power_type(move_key: str) -> Optional[str]:
    """Returns the hidden power type from a chaos move key, or None if not a hidden power move.

    Chaos JSON uses lowercase concatenated keys: "hiddenpowerice", "hiddenpowergrass", etc.
    """
    if not move_key.startswith(HIDDEN_POWER_PREFIX):
        return None
    hidden_power_type = move_key[len(HIDDEN_POWER_PREFIX):]
    if hidden_power_type not in HIDDEN_POWER_TYPES:
        return None
    return hidden_power_type


def compute_hidden_power_priors() -> None:
    for path in (STATS_PATH, SPECIES_PATH):
        if not os.path.exists(path):
            print(f"Error: {path} not found. Run tools/smogon_stats_downloader/sync.py first.", file=sys.stderr)
            sys.exit(1)

    with open(STATS_PATH) as f:
        chaos = json.load(f)
    with open(SPECIES_PATH) as f:
        species_lookup = json.load(f)  # lowercase name → {num, name, baseStats}

    # Aggregate hidden power type usage counts per species
    # chaos["data"][species_name]["Moves"] = {move_key: usage_count, ...}
    hidden_power_counts: dict[str, dict[str, float]] = defaultdict(dict)

    for species_name, species_data in chaos["data"].items():
        moves = species_data.get("Moves", {})
        type_counts: dict[str, float] = {}
        for move_key, usage_count in moves.items():
            hidden_power_type = extract_hidden_power_type(move_key.lower())
            if hidden_power_type is not None and usage_count > 0:
                type_counts[hidden_power_type] = usage_count
        if type_counts:
            hidden_power_counts[species_name.lower()] = type_counts

    # Normalize to conditional probabilities and build output keyed by lowercase species name
    priors: dict[str, dict[str, float]] = {}
    skipped = 0
    observation_totals: list[tuple[str, float]] = []  # for summary

    for species_key, type_counts in hidden_power_counts.items():
        if species_key not in species_lookup:
            skipped += 1
            continue
        total = sum(type_counts.values())
        if total == 0:
            continue
        priors[species_key] = {t: count / total for t, count in type_counts.items()}
        observation_totals.append((species_key, total))

    with open(OUTPUT_PATH, "w") as f:
        json.dump(priors, f, indent=2, sort_keys=True)

    # Summary
    total_observations = sum(t for _, t in observation_totals)
    observation_totals.sort(key=lambda x: x[1], reverse=True)
    top5 = observation_totals[:5]

    print(f"Species with hidden power data: {len(priors)}")
    print(f"Total hidden power observations: {total_observations:,.0f}")
    if skipped:
        print(f"Skipped (not in gen3_species.json): {skipped}")
    print("\nTop 5 species by hidden power usage:")
    for name, count in top5:
        types = priors[name]
        dominant = max(types, key=lambda t: types[t])
        print(f"  {name}: {count:>10,.0f} obs  (dominant: {dominant} {types[dominant]:.2%})")
    print(f"\nWrote {OUTPUT_PATH}")


if __name__ == "__main__":
    compute_hidden_power_priors()
