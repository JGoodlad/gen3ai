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
    POKEMON_HP_OFFSET
)
from .species import SpeciesEncoder
from .items import ItemsEncoder
from .types import TypeEncoder
from .abilities import AbilitiesEncoder
from .moves import MovesEncoder
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.status import Status
from typing import Any, Dict

class PokemonEncoder(ObservationEncoder):
    """
    Aggregates all Pokémon-level encoders into a single 132-dim vector.
    """
    
    def __init__(self, species_encoder, items_encoder, type_encoder, abilities_encoder, moves_encoder):
        self.species_encoder = species_encoder
        self.items_encoder = items_encoder
        self.type_encoder = type_encoder
        self.abilities_encoder = abilities_encoder
        self.moves_encoder = moves_encoder

    @property
    def dimension(self) -> int:
        return POKEMON_VECTOR_DIM

    def encode(self, mon: Any, battle: AbstractBattle) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        if mon is None:
            return vec
            
        # 1. Species (1 ID + 6 Stats)
        species_vec = self.species_encoder.encode(mon, battle)
        vec[POKEMON_SPECIES_OFFSET : POKEMON_SPECIES_OFFSET + len(species_vec)] = species_vec
        
        # 2. Items (16 + 1)
        item_vec = self.items_encoder.encode(mon, battle)
        vec[POKEMON_ITEMS_OFFSET : POKEMON_ITEMS_OFFSET + len(item_vec)] = item_vec
        
        # 3. Combined Types (8)
        type_vec = self.type_encoder.encode(mon, battle)
        vec[POKEMON_TYPES_OFFSET : POKEMON_TYPES_OFFSET + len(type_vec)] = type_vec
        
        # 4. Abilities (25)
        ability_vec = self.abilities_encoder.encode(mon, battle)
        vec[POKEMON_ABILITIES_OFFSET : POKEMON_ABILITIES_OFFSET + len(ability_vec)] = ability_vec
        
        # 5. Condition (8)
        cursor = POKEMON_CONDITION_OFFSET
        status = mon.status
        if status:
            status_map = {
                Status.BRN: 1, Status.PAR: 2, Status.SLP: 3, 
                Status.FRZ: 4, Status.PSN: 5, Status.TOX: 6
            }
            idx = status_map.get(status, 0)
            if idx > 0:
                vec[cursor + idx] = 1.0
        
        # 6. Moves (36)
        moves_vec = self.moves_encoder.encode(mon, battle)
        vec[POKEMON_MOVES_OFFSET : POKEMON_MOVES_OFFSET + len(moves_vec)] = moves_vec
        
        # 7. HP (1)
        vec[POKEMON_HP_OFFSET] = mon.current_hp_fraction
        
        return vec

    def get_layout(self) -> Dict[str, Any]:
        return {
            "species": {"offset": POKEMON_SPECIES_OFFSET, "dim": self.species_encoder.dimension},
            "items": {"offset": POKEMON_ITEMS_OFFSET, "dim": self.items_encoder.dimension},
            "types": {"offset": POKEMON_TYPES_OFFSET, "dim": self.type_encoder.dimension},
            "abilities": {"offset": POKEMON_ABILITIES_OFFSET, "dim": self.abilities_encoder.dimension},
            "condition": {"offset": POKEMON_CONDITION_OFFSET, "dim": 8},
            "moves": {"offset": POKEMON_MOVES_OFFSET, "dim": self.moves_encoder.dimension},
            "hp": {"offset": POKEMON_HP_OFFSET, "dim": 1}
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
