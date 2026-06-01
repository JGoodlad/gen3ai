"""``gen3_data`` — the single, domain-facing interface over the project's Pokémon data.

The runtime reads **only** ``data/`` through this facade; the three upstreams (poke-env static
data, the Showdown source tree, Smogon usage stats) are known only to ``tools/`` (acquisition),
which derives + normalizes them into ``data/pokemon/``. Everything downstream asks by *concept*,
never by *source*:

    from agents import gen3_data

    gen3_data.moves.get(move_id)            # MoveData(num, base_power, type, category, …)
    gen3_data.species.get(species_id)       # SpeciesData(num, base_stats)
    gen3_data.items.get(item_id)            # ItemData(num, name)
    gen3_data.abilities.get(ability_id)     # AbilityData(num, name)
    gen3_data.natures.get(nature_name)      # NatureData(multipliers)
    gen3_data.type_chart.chart()            # {DEF: {ATT: multiplier}}
    gen3_data.priors.ability(species)       # {ability_id: probability}   (Smogon)
    gen3_data.priors.hidden_power(species)  # {hp_type: probability}       (Smogon)

Each submodule follows the ``moves`` concept-module discipline: an immutable dataclass keyed by
id, parsed once via a lazy singleton (``_base``), poke-env enums borrowed as keys only. ``.raw()``
on a dex returns the parsed JSON dict (used by ``state_encoder.load_mappings`` to assemble the
encoder mappings from a single parse).
"""
from __future__ import annotations

from . import abilities, items, moves, natures, priors, species, type_chart

__all__ = [
    "abilities",
    "items",
    "moves",
    "natures",
    "priors",
    "species",
    "type_chart",
]
