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
    POKEMON_FULL_DIM,
    TEAM_SIZE,
    OFFSET_OUR_TEAM,
    OFFSET_OPP_TEAM,
    OFFSET_CONTEXT,
    OFFSET_GLOBAL,
    OFFSET_REACTIVE
)
from poke_env.battle.abstract_battle import AbstractBattle
from typing import Dict, Any, List, Tuple

class Gen3ObservationEncoder(ObservationEncoder):
    """
    Top-level encoder that orchestrates the entire Gen 3 observation vector.
    Total dimensions: 1684
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
        
        # Stable mapping for opponent slots
        self.opp_slot_map = {} 
        self.current_battle_id = None

    @property
    def dimension(self) -> int:
        return 1684

    def encode(self, battle: AbstractBattle) -> np.ndarray:
        # Reset opponent map if this is a new battle
        if battle.battle_tag != self.current_battle_id:
            self.opp_slot_map = {}
            self.current_battle_id = battle.battle_tag
            
        vec = np.zeros(self.dimension, dtype=np.float32)
        
        # 1. Our Team (0-797)
        our_team_list = list(battle.team.values())
        for i in range(TEAM_SIZE):
            mon = our_team_list[i] if i < len(our_team_list) else None
            mon_vec = self.pokemon_encoder.encode(mon, battle)
            is_active = 1.0 if (mon and mon.active) else 0.0
            
            start = OFFSET_OUR_TEAM + (i * POKEMON_FULL_DIM)
            vec[start : start + 132] = mon_vec
            vec[start + 132] = is_active
            
        # 2. Opponent Team (798-1595)
        for mon_name in battle.opponent_team:
            if mon_name not in self.opp_slot_map and len(self.opp_slot_map) < TEAM_SIZE:
                self.opp_slot_map[mon_name] = len(self.opp_slot_map)
        
        opp_slots = [None] * TEAM_SIZE
        for mon_name, mon in battle.opponent_team.items():
            if mon_name in self.opp_slot_map:
                opp_slots[self.opp_slot_map[mon_name]] = mon
        
        for i in range(TEAM_SIZE):
            mon = opp_slots[i]
            mon_vec = self.pokemon_encoder.encode(mon, battle)
            is_active = 1.0 if (mon and mon.active) else 0.0
            
            start = OFFSET_OPP_TEAM + (i * POKEMON_FULL_DIM)
            vec[start : start + 132] = mon_vec
            vec[start + 132] = is_active
            
        # 3. Active Context (1596-1657)
        vec[OFFSET_CONTEXT : OFFSET_CONTEXT+31] = self.active_context_encoder.encode(battle.active_pokemon, battle)
        vec[OFFSET_CONTEXT+31 : OFFSET_CONTEXT+62] = self.active_context_encoder.encode(battle.opponent_active_pokemon, battle)
        
        # 4. Global Environment (1658-1668)
        vec[OFFSET_GLOBAL : OFFSET_GLOBAL+11] = self.global_env_encoder.encode(battle)
        
        # 5. Reactive Features (1669-1683)
        vec[OFFSET_REACTIVE : OFFSET_REACTIVE+15] = self.reactive_encoder.encode(battle)
        
        return vec

    def get_layout(self) -> Dict[str, Any]:
        pokemon_layout = self.pokemon_encoder.get_layout()
        return {
            "our_team": [(OFFSET_OUR_TEAM + i*POKEMON_FULL_DIM, pokemon_layout) for i in range(TEAM_SIZE)],
            "opp_team": [(OFFSET_OPP_TEAM + i*POKEMON_FULL_DIM, pokemon_layout) for i in range(TEAM_SIZE)],
            "context": (OFFSET_CONTEXT, 62),
            "global": (OFFSET_GLOBAL, 11),
            "reactive": (OFFSET_REACTIVE, 15)
        }

    def describe_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        desc = {"our_team": [], "opp_team": []}
        
        # 1. Teams
        for i in range(TEAM_SIZE):
            start = OFFSET_OUR_TEAM + (i * POKEMON_FULL_DIM)
            mon_vec = vector[start : start + 132]
            is_active = vector[start + 132] > 0.5
            if np.any(mon_vec):
                mon_desc = self.pokemon_encoder.describe_vector(mon_vec)
                mon_desc["active"] = is_active
                desc["our_team"].append(mon_desc)
                
            start_opp = OFFSET_OPP_TEAM + (i * POKEMON_FULL_DIM)
            opp_vec = vector[start_opp : start_opp + 132]
            is_active_opp = vector[start_opp + 132] > 0.5
            if np.any(opp_vec):
                opp_desc = self.pokemon_encoder.describe_vector(opp_vec)
                opp_desc["active"] = is_active_opp
                desc["opp_team"].append(opp_desc)
        
        # 2. Context
        our_active_ctx = vector[OFFSET_CONTEXT : OFFSET_CONTEXT+31]
        opp_active_ctx = vector[OFFSET_CONTEXT+31 : OFFSET_CONTEXT+62]
        desc["our_active"] = self.active_context_encoder.describe_vector(our_active_ctx)
        desc["opp_active"] = self.active_context_encoder.describe_vector(opp_active_ctx)
        
        # 3. Global
        global_vec = vector[OFFSET_GLOBAL : OFFSET_GLOBAL+11]
        desc["world"] = self.global_env_encoder.describe_vector(global_vec)
        
        # 4. Reactive
        reactive_vec = vector[OFFSET_REACTIVE : OFFSET_REACTIVE+15]
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
