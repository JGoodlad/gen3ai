import numpy as np
from .base import ObservationEncoder
from .constants import MOVE_SLOT_DIM, MOVES_KNOWN_DIM
from poke_env.battle.abstract_battle import AbstractBattle
from typing import Any, List, Dict

class MovesEncoder(ObservationEncoder):
    """
    Encodes move IDs and reveal status for 4 move slots.
    Enriches with metadata from mappings (Power, Secondary, Recoil).
    """
    
    def __init__(self, mapping=None, reverse_mapping=None):
        if not mapping:
            raise ValueError("MovesEncoder requires a non-empty mapping for enrichment!")
        self.mapping = mapping
        self.reverse_mapping = reverse_mapping or {}

    @property
    def dimension(self) -> int:
        return (4 * MOVE_SLOT_DIM) + MOVES_KNOWN_DIM

    def encode(self, mon: Any, battle: AbstractBattle) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        if mon is None:
            return vec
            
        # Get moves
        moves = list(mon.moves.values()) if hasattr(mon, "moves") else []
        
        for i in range(4):
            if i < len(moves):
                move = moves[i]
                move_id = move.id
                
                # Extract metadata from mapping
                entry = self.mapping.get(move_id, {})
                num = entry.get("num", 0)
                power = entry.get("basePower", move.base_power)
                secondary = 1.0 if entry.get("hasSecondary") else 0.0
                recoil = 1.0 if entry.get("hasRecoil") else 0.0
                
                base_idx = i * MOVE_SLOT_DIM
                # 1. Move ID
                vec[base_idx] = float(num)
                # 2. Base Power (Normalized 0-200)
                vec[base_idx + 1] = float(power) / 200.0
                # 3. Secondary Effect Flag
                vec[base_idx + 2] = secondary
                # 4. Recoil Flag
                vec[base_idx + 3] = recoil
                
                # Known Flag (Binary)
                vec[4 * MOVE_SLOT_DIM + i] = 1.0
                
        return vec

    def get_layout(self) -> Dict[str, Any]:
        return {
            "slots": [{"offset": i * MOVE_SLOT_DIM, "dim": MOVE_SLOT_DIM} for i in range(4)],
            "known": {"offset": 4 * MOVE_SLOT_DIM, "dim": MOVES_KNOWN_DIM}
        }

    def describe_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        move_names = []
        for i in range(4):
            if vector[4 * MOVE_SLOT_DIM + i] > 0.5:
                mid = int(vector[i * MOVE_SLOT_DIM])
                name = self.reverse_mapping.get(mid, f"Move({mid})")
                move_names.append(name)
        return {"moves": move_names}
