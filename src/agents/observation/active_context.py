import numpy as np
from .base import ObservationEncoder
from .constants import ACTIVE_CONTEXT_DIM, BOOSTS_DIM, VOLATILES_DIM, TEMPORAL_DIM
from poke_env.battle.abstract_battle import AbstractBattle
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
            "confusion": 0, "substitute": 1, "taunt": 2, "encore": 3,
            "perishsong": 4, "leechseed": 5, "focusenergy": 6, "attract": 7
        }
        for v, idx in volatile_map.items():
            if v in volatiles:
                vec[cursor + idx] = 1.0
        cursor += 8
        
        # 3. Temporal (9)
        # Turns on Field (1)
        # Last Move Used (8 dims embedding)
        cursor += 1 
        cursor += 8
        
        return vec

    def get_layout(self) -> Dict[str, Any]:
        return {
            "boosts": (0, 14),
            "volatiles": (14, 8),
            "temporal": (22, 9)
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
        volatile_names = ["CONF", "SUB", "TAUNT", "ENC", "PERISH", "LEECH", "FOCUS", "ATTRACT"]
        active_volatiles = []
        for i in range(8):
            if vector[14 + i] > 0.5:
                active_volatiles.append(volatile_names[i])
        
        return {
            "boosts": boosts,
            "volatiles": active_volatiles
        }
