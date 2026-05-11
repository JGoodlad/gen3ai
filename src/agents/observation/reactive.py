import numpy as np
from .base import ObservationEncoder
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.side_condition import SideCondition
from poke_env.data import GenData
from typing import Any, Dict

class ReactiveEncoder(ObservationEncoder):
    """
    Encodes the original 'reactive' features:
    - Base Power of 4 moves (4)
    - Damage multipliers of 4 moves (4)
    - Fainted counts (2)
    - HP fractions (2)
    - Spikes (2)
    - Status flag (1)
    Total: 15 dims
    """
    
    @property
    def dimension(self) -> int:
        return 15

    def encode(self, battle: AbstractBattle) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        if battle is None:
            return vec
            
        # 1. Moves (Power and Multiplier)
        moves_base_power = np.zeros(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            if i >= 4: break
            moves_base_power[i] = move.base_power / 100.0
            if battle.opponent_active_pokemon is not None:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=GenData.from_gen(3).type_chart,
                )
        
        vec[0:4] = moves_base_power
        vec[4:8] = moves_dmg_multiplier
        
        # 2. Fainted Counts
        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6.0
        fainted_mon_opponent = len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6.0
        vec[8] = fainted_mon_team
        vec[9] = fainted_mon_opponent
        
        # 3. HP
        our_hp = battle.active_pokemon.current_hp_fraction if battle.active_pokemon else 0.0
        opp_hp = (
            battle.opponent_active_pokemon.current_hp_fraction 
            if battle.opponent_active_pokemon else 0.0
        )
        vec[10] = our_hp
        vec[11] = opp_hp
        
        # 4. Spikes
        our_spikes = battle.side_conditions.get(SideCondition.SPIKES, 0) / 3.0
        opp_spikes = battle.opponent_side_conditions.get(SideCondition.SPIKES, 0) / 3.0
        vec[12] = our_spikes
        vec[13] = opp_spikes
        
        # 5. Status
        vec[14] = 1.0 if battle.active_pokemon and battle.active_pokemon.status else 0.0
        
        return vec

    def get_layout(self) -> Dict[str, Any]:
        return {
            "move_power": {"offset": 0, "dim": 4},
            "move_multiplier": {"offset": 4, "dim": 4},
            "fainted": {"offset": 8, "dim": 2},
            "hp": {"offset": 10, "dim": 2},
            "spikes": {"offset": 12, "dim": 2},
            "active_status": {"offset": 14, "dim": 1}
        }

    def describe_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        return {
            "fainted_our": int(vector[8] * 6),
            "fainted_opp": int(vector[9] * 6),
            "move_mults": [f"{m:.1f}x" for m in vector[4:8].tolist()]
        }
