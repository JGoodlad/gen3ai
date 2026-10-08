"""``HIDDEN_POWER_TYPE_ORDER`` — the 16 Hidden Power types in their canonical (alphabetical) index order.

The model's typed-HP tables (``agents.model.{belief_tables,damage_tables,dex_ids,extractor_ctx}``) index by
it. The per-species candidate TRACKER that used to live here (``HiddenPowerTracker``, owned by the deleted
``EpisodeTracker``) is DELETED with the Python battle layer (T27 P6 slice 6d-2); the Rust core's trackers
fold the same evidence.
"""
from __future__ import annotations

from agents.enums import PokemonType

# Fixed alphabetical order — index is canonical across tracker, encoder, and design doc
HIDDEN_POWER_TYPE_ORDER: list[PokemonType] = [
    PokemonType.BUG,
    PokemonType.DARK,
    PokemonType.DRAGON,
    PokemonType.ELECTRIC,
    PokemonType.FIGHTING,
    PokemonType.FIRE,
    PokemonType.FLYING,
    PokemonType.GHOST,
    PokemonType.GRASS,
    PokemonType.GROUND,
    PokemonType.ICE,
    PokemonType.POISON,
    PokemonType.PSYCHIC,
    PokemonType.ROCK,
    PokemonType.STEEL,
    PokemonType.WATER,
]
