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
from .constants import (
    POKEMON_VECTOR_DIM,
    POKEMON_FULL_DIM,
    TEAM_SIZE,
    OFFSET_OUR_TEAM,
    OFFSET_OPP_TEAM,
    OFFSET_CONTEXT,
    OFFSET_GLOBAL,
    OFFSET_REACTIVE,
    REACTIVE_DIM,
    ACTIVE_CONTEXT_DIM,
    GLOBAL_ENV_DIM
)
from poke_env.battle.abstract_battle import AbstractBattle
from typing import Dict, Any, List, Tuple
import json
import os
from agents.action.mask_generator import Gen3ActionMasker
from agents.observation.reactive import ReactiveEncoder as _ReactiveEncoder

def load_mappings():
    """Loads move, species, and item mappings with validation."""
    mappings = {}
    mapping_files = {
        "species": "data/pokemon/gen3_species.json",
        "moves": "data/pokemon/gen3_moves.json",
        "abilities": "data/pokemon/gen3_abilities.json",
        "items": "data/pokemon/gen3_items.json"
    }
    for key, path in mapping_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"CRITICAL: Mapping file missing: {path}. Run data generation script first!")
        
        with open(path, "r") as f:
            data = json.load(f)
            if not data:
                raise ValueError(f"CRITICAL: Mapping file is empty: {path}")
            # Normalize data: Ensure every entry is a dict with a 'num' key
            normalized = {}
            for name, val in data.items():
                if isinstance(val, dict):
                    normalized[name] = val
                else:
                    normalized[name] = {"num": int(val)}
            mappings[key] = normalized
            
    # Pre-compute reverse mappings for IDs to names
    mappings["reverse"] = {}
    for category in ["species", "moves", "abilities", "items"]:
        rev = {}
        for name, data in mappings[category].items():
            if isinstance(data, dict) and "num" in data:
                rev[data["num"]] = name
            elif isinstance(data, (int, float)):
                rev[int(data)] = name
        mappings["reverse"][category] = rev
            
    return mappings

def get_observation_encoder(mappings):
    return Gen3ObservationEncoder(mappings)

class Gen3ObservationEncoder(ObservationEncoder):
    """
    Top-level encoder that orchestrates the entire Gen 3 observation vector.
    Total dimensions: 1613
    """
    
    def __init__(self, mappings: Dict[str, Any] = None):
        self.mappings = mappings or {}
        mappings = self.mappings
        
        # Sub-encoders
        rev = self.mappings.get("reverse", {})
        self.species_encoder = SpeciesEncoder(mappings.get("species"), rev.get("species"))
        self.items_encoder = ItemsEncoder(mappings.get("items"), rev.get("items"))
        self.type_encoder = TypeEncoder()
        self.abilities_encoder = AbilitiesEncoder(mappings.get("abilities"), rev.get("abilities"))
        self.moves_encoder = MovesEncoder(mappings.get("moves"), rev.get("moves"))
        
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
        self.current_battle_id = None

    @property
    def base_dimension(self) -> int:
        """Raw encoder output dimension, before the previous-turn mask is appended."""
        return OFFSET_REACTIVE + REACTIVE_DIM

    @property
    def dimension(self) -> int:
        """Full observation dimension including the 11-dim prev-mask and 29-dim TurnDelta block."""
        from agents.observation.turn_delta_encoder import TURN_DELTA_DIM
        return self.base_dimension + 11 + TURN_DELTA_DIM

    def encode(self, battle: AbstractBattle) -> np.ndarray:
        vec = np.zeros(self.base_dimension, dtype=np.float32)
        
        # 1. Our Team
        our_team_list = self.get_team_list(battle, is_opponent=False)
        for i in range(TEAM_SIZE):
            mon = our_team_list[i] if i < len(our_team_list) else None
            mon_vec = self.pokemon_encoder.encode(mon, battle)
            is_active = 1.0 if (mon and mon.active) else 0.0
            
            start = OFFSET_OUR_TEAM + (i * POKEMON_FULL_DIM)
            vec[start : start + POKEMON_VECTOR_DIM] = mon_vec
            vec[start + POKEMON_VECTOR_DIM] = is_active
            
        # 2. Opponent Team
        opponents = self.get_team_list(battle, is_opponent=True)
            
        for i in range(TEAM_SIZE):
            mon = opponents[i] if i < len(opponents) else None
            mon_vec = self.pokemon_encoder.encode(mon, battle)
            is_active = 1.0 if (mon and mon is battle.opponent_active_pokemon) else 0.0
            
            start = OFFSET_OPP_TEAM + (i * POKEMON_FULL_DIM)
            vec[start : start + POKEMON_VECTOR_DIM] = mon_vec
            vec[start + POKEMON_VECTOR_DIM] = is_active
            
        # 3. Active Context
        vec[OFFSET_CONTEXT : OFFSET_CONTEXT + ACTIVE_CONTEXT_DIM] = self.active_context_encoder.encode(battle.active_pokemon, battle)
        vec[OFFSET_CONTEXT + ACTIVE_CONTEXT_DIM : OFFSET_CONTEXT + (2 * ACTIVE_CONTEXT_DIM)] = self.active_context_encoder.encode(battle.opponent_active_pokemon, battle)
        
        # 4. Global Environment
        vec[OFFSET_GLOBAL : OFFSET_GLOBAL + GLOBAL_ENV_DIM] = self.global_env_encoder.encode(battle)
        
        # 5. Reactive Features
        vec[OFFSET_REACTIVE : OFFSET_REACTIVE + REACTIVE_DIM] = self.reactive_encoder.encode(battle)
        
        return vec

    def get_observation(self, battle: AbstractBattle) -> Dict[str, Any]:
        """
        Standardized entry point for getting the full observation dictionary.
        Includes both the encoded state vector and the action mask.
        """
        if battle.wait:
            error_msg = f"⚠️ [OBSERVATION] CRITICAL: Observation requested while battle.wait is True for {battle.battle_tag}"
            print(error_msg)
            raise RuntimeError(error_msg)
            
        obs = self.encode(battle)
        mask = Gen3ActionMasker.get_mask(battle)
        return {
            "observation": obs,
            "action_mask": mask
        }

    def get_layout(self) -> Dict[str, Any]:
        pokemon_layout = self.pokemon_encoder.get_layout()
        return {
            "parts": {
                "our_team": {
                    "start": OFFSET_OUR_TEAM, 
                    "end": OFFSET_OPP_TEAM, 
                    "reshape": (TEAM_SIZE, POKEMON_FULL_DIM)
                },
                "opp_team": {
                    "start": OFFSET_OPP_TEAM, 
                    "end": OFFSET_CONTEXT, 
                    "reshape": (TEAM_SIZE, POKEMON_FULL_DIM)
                },
                "context": {
                    "start": OFFSET_CONTEXT, 
                    "end": OFFSET_GLOBAL, 
                    "reshape": (2, self.active_context_encoder.dimension)
                },
                "global": {
                    "start": OFFSET_GLOBAL, 
                    "end": OFFSET_REACTIVE, 
                    "dim": self.global_env_encoder.dimension
                },
                "reactive": {
                    "start": OFFSET_REACTIVE, 
                    "end": self.dimension, 
                    "dim": self.reactive_encoder.dimension
                }
            },
            "pokemon": pokemon_layout,
            "total_dim": self.dimension,    # base + 11 prev_mask + 29 turn_delta
            "base_dim": self.base_dimension, # raw encoder output without prev_mask or turn_delta
            "prev_mask_dim": 11,
            "turn_delta_dim": 29,
            "active_context_dim": ACTIVE_CONTEXT_DIM,
            "reactive_layout": _ReactiveEncoder().get_layout(),
            "global_layout": self.global_env_encoder.get_layout(),
            "max_species": 400,
            "species_embedding_dim": 32,
            "max_moves": 400,
            "move_embedding_dim": 16,
            "max_items": 600,
            "item_embedding_dim": 16,
            "max_abilities": 100,
            "ability_embedding_dim": 16,
            "max_types": 20, # 18 types + placeholders
            "type_embedding_dim": 16
        }

    def get_features_extractor_kwargs(self) -> Dict[str, Any]:
        return {
            "layout": self.get_layout(),
            "mappings": self.mappings
        }

    def describe_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        desc = {"our_team": [], "opp_team": []}
        
        # 1. Teams
        for i in range(TEAM_SIZE):
            start = OFFSET_OUR_TEAM + (i * POKEMON_FULL_DIM)
            mon_vec = vector[start : start + POKEMON_VECTOR_DIM]
            is_active = vector[start + POKEMON_VECTOR_DIM] > 0.5
            if np.any(mon_vec):
                mon_desc = self.pokemon_encoder.describe_vector(mon_vec)
                mon_desc["active"] = is_active
                desc["our_team"].append(mon_desc)
                
            start_opp = OFFSET_OPP_TEAM + (i * POKEMON_FULL_DIM)
            opp_vec = vector[start_opp : start_opp + POKEMON_VECTOR_DIM]
            is_active_opp = vector[start_opp + POKEMON_VECTOR_DIM] > 0.5
            if np.any(opp_vec):
                opp_desc = self.pokemon_encoder.describe_vector(opp_vec)
                opp_desc["active"] = is_active_opp
                desc["opp_team"].append(opp_desc)
        
        # 2. Context
        our_active_ctx = vector[OFFSET_CONTEXT : OFFSET_CONTEXT + ACTIVE_CONTEXT_DIM]
        opp_active_ctx = vector[OFFSET_CONTEXT + ACTIVE_CONTEXT_DIM : OFFSET_CONTEXT + (2 * ACTIVE_CONTEXT_DIM)]
        desc["our_active"] = self.active_context_encoder.describe_vector(our_active_ctx)
        desc["opp_active"] = self.active_context_encoder.describe_vector(opp_active_ctx)
        
        # 3. Global
        global_vec = vector[OFFSET_GLOBAL : OFFSET_GLOBAL + GLOBAL_ENV_DIM]
        desc["world"] = self.global_env_encoder.describe_vector(global_vec)
        
        # 4. Reactive
        reactive_vec = vector[OFFSET_REACTIVE : OFFSET_REACTIVE + REACTIVE_DIM]
        desc["momentum"] = self.reactive_encoder.describe_vector(reactive_vec)
        
        return desc

    def integrity_check(self, vector: np.ndarray) -> Tuple[List[str], bool]:
        warnings = []
        is_critical = False
        desc = self.describe_vector(vector)
        
        # 1. Active Pokémon Check
        our_active = [mon for mon in desc['our_team'] if mon.get('active')]
        if len(our_active) > 1:
            warnings.append(f"CRITICAL: Multiple active Pokémon on our team: {[m['species'] for m in our_active]}")
            is_critical = True
        elif len(our_active) == 0:
            warnings.append("Note: No active Pokémon found on our team.")
            
        opp_active = [mon for mon in desc['opp_team'] if mon.get('active')]
        if len(opp_active) > 1:
            warnings.append(f"CRITICAL: Multiple active Pokémon on opponent team: {[m['species'] for m in opp_active]}")
            is_critical = True
            
        # 2. HP/Fainted Consistency
        fainted_our_list = len([mon for mon in desc['our_team'] if float(mon['hp'].strip('%')) == 0])
        fainted_our_momentum = desc['momentum']['fainted_our']
        if fainted_our_list != fainted_our_momentum:
             warnings.append(f"CRITICAL: Our fainted count mismatch! Team list ({fainted_our_list}) != momentum ({fainted_our_momentum})")
             is_critical = True
             
        fainted_opp_list = [mon['species'] for mon in desc['opp_team'] if float(mon['hp'].strip('%')) == 0]
        fainted_opp_count = len(fainted_opp_list)
        fainted_opp_momentum = desc['momentum']['fainted_opp']
        if fainted_opp_momentum > fainted_opp_count:
             warnings.append(f"CRITICAL: Opponent fainted count (momentum={fainted_opp_momentum}) > seen in team list ({fainted_opp_count}). Team fainted: {fainted_opp_list}")
             is_critical = True
        elif fainted_opp_momentum < fainted_opp_count:
             warnings.append(f"Mismatch: Opponent fainted count (momentum={fainted_opp_momentum}) < seen in team list ({fainted_opp_count}). Team fainted: {fainted_opp_list}")
             is_critical = True

        return warnings, is_critical
