import numpy as np
from .base import ObservationEncoder
from .constants import POKEMON_VECTOR_DIM, CONDITION_DIM
from .species import SpeciesEncoder
from .items import ItemsEncoder
from .types import TypeEncoder
from .abilities import AbilitiesEncoder
from .moves import MovesEncoder
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.status import Status
from typing import Any

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
            
        cursor = 0
        
        # 1. Species (32) + Stats (5) = 37
        species_vec = self.species_encoder.encode(mon, battle)
        vec[cursor:cursor+len(species_vec)] = species_vec
        cursor += 37 # Fixed size as per spec
        
        # 2. Items (16 + 1) = 17
        item_vec = self.items_encoder.encode(mon, battle)
        vec[cursor:cursor+len(item_vec)] = item_vec
        cursor += 17
        
        # 3. Combined Types (8)
        type_vec = self.type_encoder.encode(mon, battle)
        vec[cursor:cursor+len(type_vec)] = type_vec
        cursor += 8
        
        # 4. Abilities (8 + 1 + 16) = 25
        ability_vec = self.abilities_encoder.encode(mon, battle)
        vec[cursor:cursor+len(ability_vec)] = ability_vec
        cursor += 25
        
        # 5. Condition (Status 7 + Status Turn 1) = 8
        # None, BRN, PAR, SLP, FRZ, PSN, TOX
        status = mon.status
        if status:
            status_map = {
                Status.BRN: 1, Status.PAR: 2, Status.SLP: 3, 
                Status.FRZ: 4, Status.PSN: 5, Status.TOX: 6
            }
            idx = status_map.get(status, 0)
            if idx > 0:
                vec[cursor + idx] = 1.0
            
            # Status Turn (Normalized)
            # Sleep turns: 1-7 in Gen 3
            # Toxic: 1-15?
            # We'll just put the raw turn count for now if available.
            # poke-env doesn't always expose this directly, might need to track it.
            # TODO: Implement status turn tracking.
            
        cursor += 8
        
        # 6. Moves (32 + 4) = 36
        moves_vec = self.moves_encoder.encode(mon, battle)
        vec[cursor:cursor+len(moves_vec)] = moves_vec
        cursor += 36
        
        # 7. HP (1)
        vec[cursor] = mon.current_hp_fraction
        cursor += 1
        
        return vec
