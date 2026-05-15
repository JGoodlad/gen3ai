import numpy as np
from .base import ObservationEncoder
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.side_condition import SideCondition
from poke_env.data import GenData
from .constants import REACTIVE_DIM, TEAM_SIZE
from typing import Any, Dict, List, Tuple

class ReactiveEncoder(ObservationEncoder):
    """
    Encodes reactive features:
    - Base Power of 4 active moves (4)
    - Damage multipliers of 4 active moves (4)
    - Fainted counts (2)
    - HP fractions (2)
    - Spikes (2)
    - Status flag (1)
    - Forced Struggle flag (1)
    - Matchup Matrix: Our moves vs Their mons (144)
    - Matchup Matrix: Their moves vs Our mons (144)
    Total: 304 dims
    """
    
    @property
    def dimension(self) -> int:
        return REACTIVE_DIM

    def encode(self, battle: AbstractBattle) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        if battle is None:
            return vec
            
        # 1. Active Moves (Power and Multiplier)
        moves_base_power = np.zeros(4)
        moves_dmg_multiplier = np.ones(4)
        
        mon_move_ids = []
        if battle.active_pokemon:
            mon_move_ids = [m.id for m in battle.active_pokemon.moves.values()]

        for i, move in enumerate(battle.available_moves):
            if i >= 4: break
            if move.id == "struggle" and move.id not in mon_move_ids:
                continue
                
            moves_base_power[i] = move.base_power / 100.0
            if battle.opponent_active_pokemon is not None:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=GenData.from_gen(3).type_chart,
                ) / 4.0
        
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
        
        # 6. Forced Struggle
        is_struggle = 0.0
        if len(battle.available_moves) == 1 and battle.available_moves[0].id == "struggle":
            if battle.available_moves[0].id not in mon_move_ids:
                is_struggle = 1.0
        vec[15] = is_struggle

        # --- Matchup Matrices ---
        our_team = self.get_team_list(battle, is_opponent=False)
        their_team = self.get_team_list(battle, is_opponent=True)
        type_chart = GenData.from_gen(3).type_chart

        # 7. Our moves vs Their mons (144 dims)
        cursor = 16
        for i in range(TEAM_SIZE):
            our_mon = our_team[i] if i < len(our_team) else None
            our_moves = self.get_sorted_moves(our_mon)
            for move_idx in range(4):
                move = our_moves[move_idx] if move_idx < len(our_moves) else None
                for j in range(TEAM_SIZE):
                    their_mon = their_team[j] if j < len(their_team) else None
                    if move and their_mon:
                        # Normalize by 4.0 to keep values in [0, 1] range for better MLP convergence
                        vec[cursor] = move.type.damage_multiplier(
                            their_mon.type_1, their_mon.type_2, type_chart=type_chart
                        ) / 4.0
                    cursor += 1

        # 8. Their moves vs Our mons (144 dims)
        for i in range(TEAM_SIZE):
            their_mon = their_team[i] if i < len(their_team) else None
            their_moves = self.get_sorted_moves(their_mon)
            for move_idx in range(4):
                move = their_moves[move_idx] if move_idx < len(their_moves) else None
                for j in range(TEAM_SIZE):
                    our_mon = our_team[j] if j < len(our_team) else None
                    if move and our_mon:
                        # Normalize by 4.0 to keep values in [0, 1] range for better MLP convergence
                        vec[cursor] = move.type.damage_multiplier(
                            our_mon.type_1, our_mon.type_2, type_chart=type_chart
                        ) / 4.0
                    cursor += 1
        
        return vec

    def get_layout(self) -> Dict[str, Any]:
        return {
            "move_power": {"offset": 0, "dim": 4},
            "move_multiplier": {"offset": 4, "dim": 4},
            "fainted": {"offset": 8, "dim": 2},
            "hp": {"offset": 10, "dim": 2},
            "spikes": {"offset": 12, "dim": 2},
            "active_status": {"offset": 14, "dim": 1},
            "forced_struggle": {"offset": 15, "dim": 1},
            "our_matchups": {"offset": 16, "dim": 144},
            "their_matchups": {"offset": 160, "dim": 144}
        }

    def describe_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        # Extract matrices and scale back up by 4.0 for human-readable display
        our_m = vector[16:160].reshape(TEAM_SIZE, 4, TEAM_SIZE) * 4.0
        their_m = vector[160:304].reshape(TEAM_SIZE, 4, TEAM_SIZE) * 4.0
        
        return {
            "fainted_our": int(vector[8] * 6),
            "fainted_opp": int(vector[9] * 6),
            "active_move_mults": [f"{m*4.0:.1f}x" for m in vector[4:8].tolist()],
            "struggle": bool(vector[15]),
            "our_vs_their": our_m, # Full matrix for deeper trace
            "their_vs_our": their_m
        }
