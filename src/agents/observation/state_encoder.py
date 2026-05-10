import numpy as np
from .base import ObservationEncoder
from .pokemon import PokemonEncoder
from .active_context import ActiveContextEncoder
from .global_env import GlobalEnvEncoder
from .species import SpeciesEncoder
from .items import ItemsEncoder
from .types import TypeEncoder
from .abilities import AbilitiesEncoder
from .moves import MovesEncoder
from .reactive import ReactiveEncoder
from poke_env.battle.abstract_battle import AbstractBattle
from typing import Dict, Any

class Gen3ObservationEncoder(ObservationEncoder):
    """
    Top-level encoder that orchestrates the entire Gen 3 observation vector.
    Total dimensions: ~1657
    """
    
    def __init__(self, mappings: Dict[str, Any] = None):
        mappings = mappings or {}
        
        # Sub-encoders
        self.species_encoder = SpeciesEncoder(mappings.get("species"))
        self.items_encoder = ItemsEncoder(mappings.get("items"))
        self.type_encoder = TypeEncoder()
        self.abilities_encoder = AbilitiesEncoder(mappings.get("abilities"))
        self.moves_encoder = MovesEncoder(mappings.get("moves"))
        
        self.pokemon_encoder = PokemonEncoder(
            self.species_encoder, 
            self.items_encoder, 
            self.type_encoder, 
            self.abilities_encoder, 
            self.moves_encoder
        )
        
        self.active_context_encoder = ActiveContextEncoder(mappings.get("moves"))
        self.global_env_encoder = GlobalEnvEncoder()
        self.reactive_encoder = ReactiveEncoder()
        
        # Stable mapping for opponent slots (since opponent_team is a dict)
        self.opp_slot_map = {} 

    @property
    def dimension(self) -> int:
        # 12 * 133 + 2 * 31 + 11 + 15
        return (12 * 133) + (2 * 31) + 11 + 15

    def encode(self, battle: AbstractBattle) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        cursor = 0
        
        # 1. Pokemon Vectors (12 * 133)
        # Use STABLE order (same as battle.team.values()) to match action indices 0-5
        our_team_list = list(battle.team.values())
        
        for i in range(6):
            mon = our_team_list[i] if i < len(our_team_list) else None
            mon_vec = self.pokemon_encoder.encode(mon, battle)
            # Add active flag (at the end of the 132-dim vector)
            is_active = 1.0 if (mon and mon.active) else 0.0
            vec[cursor:cursor+132] = mon_vec
            vec[cursor+132] = is_active
            cursor += 133
            
        # Opponent Team: Fixed slots 0-5 based on discovery order
        # We use species as the key for the slot map for simplicity in this POC
        for mon_name in battle.opponent_team:
            if mon_name not in self.opp_slot_map and len(self.opp_slot_map) < 6:
                self.opp_slot_map[mon_name] = len(self.opp_slot_map)
        
        opp_slots = [None] * 6
        for mon_name, mon in battle.opponent_team.items():
            if mon_name in self.opp_slot_map:
                opp_slots[self.opp_slot_map[mon_name]] = mon
        
        for i in range(6):
            mon = opp_slots[i]
            mon_vec = self.pokemon_encoder.encode(mon, battle)
            is_active = 1.0 if (mon and mon.active) else 0.0
            vec[cursor:cursor+132] = mon_vec
            vec[cursor+132] = is_active
            cursor += 133
            
        # 2. Active Context (2 * 31)
        # Our Active
        vec[cursor:cursor+31] = self.active_context_encoder.encode(battle.active_pokemon, battle)
        cursor += 31
        
        # Opponent Active
        vec[cursor:cursor+31] = self.active_context_encoder.encode(battle.opponent_active_pokemon, battle)
        cursor += 31
        
        # 3. Global Environment (11)
        vec[cursor:cursor+11] = self.global_env_encoder.encode(battle)
        cursor += 11
        
        # 4. Reactive Features (15)
        vec[cursor:cursor+15] = self.reactive_encoder.encode(battle)
        
        return vec
