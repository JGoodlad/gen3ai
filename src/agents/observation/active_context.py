import numpy as np
from .base import ObservationEncoder
from .constants import ACTIVE_CONTEXT_DIM, BOOSTS_DIM, VOLATILES_DIM, TEMPORAL_DIM
from poke_env.battle.abstract_battle import AbstractBattle
from typing import Any

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
        # Confusion, Substitute, Taunt, Encore, etc.
        # TODO: Map specific volatiles to indices.
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
        # TODO: Implement turns on field tracking.
        cursor += 1 
        
        # Last Move
        # last_move = mon.last_move # poke-env doesn't always have this directly on Pokemon
        # For now, we'll leave it 0.
        cursor += 8
        
        return vec
