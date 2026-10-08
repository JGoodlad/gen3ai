from __future__ import annotations
import numpy as np
from .base import ObservationEncoder
from .constants import (
    POKEMON_VECTOR_DIM,
    POKEMON_SPECIES_OFFSET,
    POKEMON_ITEMS_OFFSET,
    POKEMON_TYPES_OFFSET,
    POKEMON_ABILITIES_OFFSET,
    POKEMON_CONDITION_OFFSET,
    POKEMON_MOVES_OFFSET,
    POKEMON_HP_OFFSET,
    POKEMON_SPECIES_KNOWN_OFFSET,
    POKEMON_COUNTER_OFFSET,
    POKEMON_SPREAD_OFFSET,
    POKEMON_SPREAD_DIM,
    POKEMON_HP_REVEALED_OFFSET,
    POKEMON_HP_BLOCK_DIM,
    POKEMON_PROTECT_OFFSET,
)
from .abilities import AbilitiesEncoder
from .items import ItemsEncoder
from .moves import MovesEncoder
from .species import SpeciesEncoder
from .types import TypeEncoder
from typing import Any, Dict, Optional

# Status → condition one-hot slot, in both the read-model's id form (LivePokemon.status,
# e.g. "brn") and poke-env's Status enum (raw / unit-test fallback). Both map to the SAME
# slot so the emitted vector is byte-identical regardless of which source is read.
_STATUS_STR_IDX = {"brn": 1, "par": 2, "slp": 3, "frz": 4, "psn": 5, "tox": 6}


class PokemonEncoder(ObservationEncoder):
    """
    Aggregates all Pokémon-level encoders into a single POKEMON_VECTOR_DIM-wide vector.
    Layout: species(7) + items(3) + types(2) + abilities(2) + condition(7) + moves(36)
            + hp(1) + species_known(1) + status_counters(2) + spread(18) + hp_block(17) = 96 dims.
    The active flag (1 dim) is appended by state_encoder, making POKEMON_FULL_DIM = 98.

    hp_block carries Hidden Power information as `hp_revealed (1) + type_probs (16)`:
      - opponent unknown: hp_revealed=0, probs all zero
      - opponent narrowed (observed HP, types ruled out): hp_revealed=1, sparse probs
      - opponent ruled out (4 moves seen, none is HP): hp_revealed=1, all-zero probs
      - our mon (HP or not): hp_revealed=1, all-zero probs

    Own mons always read hp_revealed=1 with all-zero probs. The own HP *type* is not
    recoverable from the current-board read-model: in a live battle the server request
    re-keys a typed Hidden Power under the bare ``"hiddenpower"`` id (the move's own typed
    ``.id`` survives only on the raw poke-env ``Move`` object, which this boundary does not
    expose). The previous `_own_hp_type_index` therefore resolved to None on every real
    decision — the own type one-hot was already dead — so dropping it keeps the emitted
    vector byte-identical while removing the last raw ``mon.moves`` read here.
    """
    
    _NATURE_STAT_ORDER = ("atk", "def", "spa", "spd", "spe")

    def __init__(self,
                 species_encoder: SpeciesEncoder,
                 items_encoder: ItemsEncoder,
                 type_encoder: TypeEncoder,
                 abilities_encoder: AbilitiesEncoder,
                 moves_encoder: MovesEncoder,
                 natures: Optional[Dict[str, Any]] = None) -> None:
        self.species_encoder = species_encoder
        self.items_encoder = items_encoder
        self.type_encoder = type_encoder
        self.abilities_encoder = abilities_encoder
        self.moves_encoder = moves_encoder
        self._natures: Dict[str, Any] = natures or {}

    @property
    def dimension(self) -> int:
        return POKEMON_VECTOR_DIM

    def get_layout(self) -> Dict[str, Any]:
        return {
            "species": {
                "offset": POKEMON_SPECIES_OFFSET, 
                "dim": self.species_encoder.dimension,
                "layout": self.species_encoder.get_layout()
            },
            "items": {
                "offset": POKEMON_ITEMS_OFFSET, 
                "dim": self.items_encoder.dimension,
                "layout": self.items_encoder.get_layout()
            },
            "types": {
                "offset": POKEMON_TYPES_OFFSET, 
                "dim": self.type_encoder.dimension,
                "layout": self.type_encoder.get_layout()
            },
            "abilities": {
                "offset": POKEMON_ABILITIES_OFFSET, 
                "dim": self.abilities_encoder.dimension,
                "layout": self.abilities_encoder.get_layout()
            },
            "condition": {"offset": POKEMON_CONDITION_OFFSET, "dim": 7},
            "moves": {
                "offset": POKEMON_MOVES_OFFSET, 
                "dim": self.moves_encoder.dimension,
                "layout": self.moves_encoder.get_layout()
            },
            "hp": {"offset": POKEMON_HP_OFFSET, "dim": 1},
            "species_known": {"offset": POKEMON_SPECIES_KNOWN_OFFSET, "dim": 1},
            "status_counters": {"offset": POKEMON_COUNTER_OFFSET, "dim": 2},
            "spread": {
                "offset": POKEMON_SPREAD_OFFSET,
                "dim": POKEMON_SPREAD_DIM,
                "layout": {
                    "ivs": {"offset": 0, "dim": 6, "stats": ["hp", "atk", "def", "spa", "spd", "spe"]},
                    "evs": {"offset": 6, "dim": 6, "stats": ["hp", "atk", "def", "spa", "spd", "spe"]},
                    "spread_known": {"offset": 12, "dim": 1},
                    "nature": {"offset": 13, "dim": 5, "stats": list(self._NATURE_STAT_ORDER)},
                }
            },
            "hp_block": {
                "offset": POKEMON_HP_REVEALED_OFFSET,
                "dim": POKEMON_HP_BLOCK_DIM,
                "layout": {
                    "hp_revealed": {"offset": 0, "dim": 1},
                    "hp_type_probs": {"offset": 1, "dim": 16},
                },
            },
            # gen3_entity_rehome_v1: the entity-owned stall state (see encode()).
            "protect_odds": {"offset": POKEMON_PROTECT_OFFSET, "dim": 1},
            "pokemon_vector_dim": POKEMON_VECTOR_DIM,
        }

    def describe_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        species_part = vector[POKEMON_SPECIES_OFFSET : POKEMON_SPECIES_OFFSET + 7]
        species_desc = self.species_encoder.describe_vector(species_part)
        
        item_part = vector[POKEMON_ITEMS_OFFSET : POKEMON_ITEMS_OFFSET + self.items_encoder.dimension]
        item_name = self.items_encoder.describe_vector(item_part)
        
        type_part = vector[POKEMON_TYPES_OFFSET : POKEMON_TYPES_OFFSET + self.type_encoder.dimension]
        type_name = self.type_encoder.describe_vector(type_part)
        
        ability_part = vector[POKEMON_ABILITIES_OFFSET : POKEMON_ABILITIES_OFFSET + self.abilities_encoder.dimension]
        ability_name = self.abilities_encoder.describe_vector(ability_part)
        
        moves_part = vector[POKEMON_MOVES_OFFSET : POKEMON_MOVES_OFFSET + self.moves_encoder.dimension]
        moves_desc = self.moves_encoder.describe_vector(moves_part)
        
        return {
            "species": species_desc["name"],
            "hp": f"{vector[POKEMON_HP_OFFSET]*100:.1f}%",
            "types": type_name,
            "stats": {k: v for k, v in species_desc.items() if k != "name"},
            "status": self._decode_status(vector[POKEMON_CONDITION_OFFSET : POKEMON_CONDITION_OFFSET + 7]),
            "item": item_name,
            "ability": ability_name,
            "moves": moves_desc["moves"]
        }

    def _decode_status(self, vec: np.ndarray) -> str:
        names = ["NONE", "BRN", "PAR", "SLP", "FRZ", "PSN", "TOX"]
        for i, val in enumerate(vec):
            if val > 0.5:
                return names[i]
        return "NONE"
