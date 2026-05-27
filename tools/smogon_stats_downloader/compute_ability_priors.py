"""Compute per-species ability probability priors from Smogon Gen3 OU stats.

Reads data/pokemon/gen3_smogon_stats.json (chaos format, downloaded and
aggregated by tools/smogon_stats_downloader/sync.py) and writes
data/pokemon/gen3_ability_priors.json keyed by lowercase species name with
shape `{species: {ability_id: probability}}`. Probabilities are normalized
within each species so they sum to 1.0.

Mirrors the pattern of src/scripts/compute_hidden_power_priors.py.

The encoder picks the top 2 abilities by probability and emits
[ability1_id, ability2_id, dominance, known_flag]. Examples:

  snorlax:    {immunity:   0.86, thickfat:   0.14}  → dominance 0.86
  aerodactyl: {rockhead:   0.95, pressure:   0.05}  → dominance 0.95
  shedinja:   {wonderguard: 1.0}                    → dominance 1.0
  salamence:  {intimidate:  1.0}                    → dominance 1.0

Run from repo root:
    python tools/smogon_stats_downloader/compute_ability_priors.py
"""
from __future__ import annotations

import json
import os
import sys

STATS_PATH = "data/pokemon/gen3_smogon_stats.json"
SPECIES_PATH = "data/pokemon/gen3_species.json"
ABILITIES_PATH = "data/pokemon/gen3_abilities.json"
OUTPUT_PATH = "data/pokemon/gen3_ability_priors.json"


def main() -> int:
    for path in (STATS_PATH, SPECIES_PATH, ABILITIES_PATH):
        if not os.path.exists(path):
            print(
                f"ERROR: {path} not found. Run "
                "tools/smogon_stats_downloader/sync.py first.",
                file=sys.stderr,
            )
            return 1

    with open(STATS_PATH) as f:
        chaos = json.load(f)
    with open(SPECIES_PATH) as f:
        species_lookup = json.load(f)
    with open(ABILITIES_PATH) as f:
        gen3_abilities = json.load(f)

    valid_ability_ids = set(gen3_abilities.keys())

    priors: dict[str, dict[str, float]] = {}
    skipped_species: list[str] = []
    skipped_post_gen3: dict[str, list[str]] = {}
    total_observations = 0.0

    for species_name, species_data in chaos["data"].items():
        species_key = species_name.lower()
        if species_key not in species_lookup:
            skipped_species.append(species_key)
            continue

        raw_abilities = species_data.get("Abilities", {})
        # Filter out abilities that don't exist in Gen 3 (e.g. someone using a
        # later-gen ability in a custom Gen 3 OU match). These leak in
        # occasionally on Smogon's ladder.
        filtered = {}
        rejected = []
        for ab_id, usage in raw_abilities.items():
            ab_key = ab_id.lower().replace(" ", "").replace("_", "")
            if ab_key not in valid_ability_ids:
                rejected.append(ab_key)
                continue
            if usage > 0:
                filtered[ab_key] = float(usage)
        if rejected:
            skipped_post_gen3[species_key] = rejected
        if not filtered:
            continue

        total = sum(filtered.values())
        priors[species_key] = {ab: count / total for ab, count in filtered.items()}
        total_observations += total

    with open(OUTPUT_PATH, "w") as f:
        json.dump(priors, f, indent=2, sort_keys=True)
        f.write("\n")

    # Summary — rank by raw ability-usage count (recovered from the chaos JSON)
    raw_counts = []
    for sp_key, p in priors.items():
        chaos_entry = next(
            (v for k, v in chaos["data"].items() if k.lower() == sp_key),
            None,
        )
        raw = sum((chaos_entry or {}).get("Abilities", {}).values())
        raw_counts.append((sp_key, raw, p))
    raw_counts.sort(key=lambda x: -x[1])

    print(f"Species with ability data: {len(priors)}")
    print(f"Total ability observations: {total_observations:,.0f}")
    if skipped_species:
        print(
            f"Skipped (not in gen3_species.json): {len(skipped_species)}"
        )
    if skipped_post_gen3:
        n = sum(len(v) for v in skipped_post_gen3.values())
        print(f"Rejected post-Gen3 ability entries: {n} across "
              f"{len(skipped_post_gen3)} species")
    print("\nTop 10 species by ability usage:")
    for sp, raw, p in raw_counts[:10]:
        dominant_ab = max(p, key=lambda a: p[a])
        n_abs = len(p)
        print(
            f"  {sp:<20} {raw:>12,.0f} obs  "
            f"{n_abs} ability/-ies  "
            f"(dominant: {dominant_ab} {p[dominant_ab]:.2%})"
        )
    print(f"\nWrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
