import numpy as np
from .base import ObservationEncoder
from .constants import MOVE_SLOT_DIM
from .types import TypeEncoder
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.move_category import MoveCategory
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
        return 4 * MOVE_SLOT_DIM

    def encode(self, mon: Any, battle: AbstractBattle) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        if mon is None:
            return vec
            
        # Get moves and SORT them by ID to ensure stable mapping across workers
        moves = self.get_sorted_moves(mon)
        
        for i in range(4):
            if i < len(moves):
                move = moves[i]
                move_id = move.id
                
                # Extract metadata from mapping
                if move_id not in self.mapping:
                    raise ValueError(f"Unrecognized move: {move_id}. Update data/pokemon/gen3_moves.json")
                entry = self.mapping[move_id]
                num = entry.get("num", 0)
                power = entry.get("basePower", move.base_power)
                secondary = 1.0 if entry.get("hasSecondary") else 0.0
                recoil = 1.0 if entry.get("hasRecoil") else 0.0
                
                # Get type from mapping and convert to ID using TypeEncoder
                move_type = entry.get("type", "Normal").upper()
                type_id = TypeEncoder.TYPE_TO_IDX.get(move_type, 0)
                
                base_idx = i * MOVE_SLOT_DIM
                # 1. Move ID
                vec[base_idx] = float(num)
                # 2. Base Power (Normalized 0-200)
                vec[base_idx + 1] = float(power) / 200.0
                # 3. Secondary Effect Flag
                vec[base_idx + 2] = secondary
                # 4. Recoil Flag
                vec[base_idx + 3] = recoil
                # 5. Type ID
                vec[base_idx + 4] = float(type_id)
                # 6. Category ID (0=Status, 1=Physical, 2=Special)
                category_val = 0.0
                if move.category == MoveCategory.PHYSICAL:
                    category_val = 1.0
                elif move.category == MoveCategory.SPECIAL:
                    category_val = 2.0
                vec[base_idx + 5] = category_val
                
                # 7. Known Flag (Binary) - Now interleaved
                vec[base_idx + 6] = 1.0
                
        return vec

    def get_layout(self) -> Dict[str, Any]:
        return {
            "slots": [{"offset": i * MOVE_SLOT_DIM, "dim": MOVE_SLOT_DIM} for i in range(4)],
            "slot_layout": {
                "id": {"offset": 0, "dim": 1},
                "power": {"offset": 1, "dim": 1},
                "secondary": {"offset": 2, "dim": 1},
                "recoil": {"offset": 3, "dim": 1},
                "type": {"offset": 4, "dim": 1},
                "category": {"offset": 5, "dim": 1},
                "known": {"offset": 6, "dim": 1}
            }
        }

    def describe_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        move_names = []
        for i in range(4):
            if vector[i * MOVE_SLOT_DIM + 6] > 0.5:
                mid = int(vector[i * MOVE_SLOT_DIM])
                name = self.reverse_mapping.get(mid, f"Move({mid})")
                move_names.append(name)
        return {"moves": move_names}
