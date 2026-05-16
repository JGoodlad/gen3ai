import numpy as np
from .base import ObservationEncoder
from .constants import ACTIVE_CONTEXT_DIM, BOOSTS_DIM, VOLATILES_DIM
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.effect import Effect
from typing import Any, Dict

class ActiveContextEncoder(ObservationEncoder):
    """
    Encodes the active context for a Pokémon (Slot 0).
    """
    
    def __init__(self, move_to_id=None):
        self.move_to_id = move_to_id or {}

    @property
    def dimension(self) -> int:
        return ACTIVE_CONTEXT_DIM

    def encode(self, mon: Any, battle: AbstractBattle) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        if mon is None:
            return vec
            
        cursor = 0
        
        # 1. Boosts (14) - 2 dims per stage
        # Atk, Def, SpA, SpD, Spe, Acc, Eva
        stats = ["atk", "def", "spa", "spd", "spe", "accuracy", "evasion"]
        for stat in stats:
            stage = mon.boosts.get(stat, 0)
            # 2 dims: [positive magnitude, negative magnitude]
            vec[cursor] = max(0, stage) / 6.0
            vec[cursor+1] = max(0, -stage) / 6.0
            cursor += 2
            
        # 2. Volatiles (8)
        volatiles = getattr(mon, "volatiles", {})
        volatile_map = {
            Effect.CONFUSION: 0, Effect.SUBSTITUTE: 1, Effect.TAUNT: 2, Effect.ENCORE: 3,
            Effect.LEECH_SEED: 5, Effect.FOCUS_ENERGY: 6, Effect.ATTRACT: 7,
            Effect.DESTINY_BOND: 8,
        }
        for v, idx in volatile_map.items():
            if v in volatiles:
                vec[cursor + idx] = 1.0
        
        # Perish Song (Index 4) - Check all counts
        if any(e in volatiles for e in [Effect.PERISH0, Effect.PERISH1, Effect.PERISH2, Effect.PERISH3]):
            vec[cursor + 4] = 1.0
            
        cursor += 9

        return vec

    def get_layout(self) -> Dict[str, Any]:
        return {
            "boosts": {"offset": 0, "dim": 14},
            "volatiles": {"offset": 14, "dim": 9}
        }

    def describe_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        # Boosts
        stats = ["atk", "def", "spa", "spd", "spe", "acc", "eva"]
        boosts = {}
        for i, stat in enumerate(stats):
            pos = vector[i*2] * 6.0
            neg = vector[i*2 + 1] * 6.0
            val = int(pos - neg)
            if val != 0:
                boosts[stat] = f"{val:+d}"
        
        # Volatiles
        volatile_names = ["CONF", "SUB", "TAUNT", "ENC", "PERISH", "LEECH", "FOCUS", "ATTRACT", "DBOND"]
        active_volatiles = []
        for i in range(9):
            if vector[14 + i] > 0.5:
                active_volatiles.append(volatile_names[i])
        
        return {
            "boosts": boosts,
            "volatiles": active_volatiles
        }
