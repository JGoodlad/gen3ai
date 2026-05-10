import numpy as np
from .base import ObservationEncoder
from .constants import ITEM_ID_DIM, ITEM_KNOWN_DIM
from poke_env.battle.abstract_battle import AbstractBattle
from typing import Any

class ItemsEncoder(ObservationEncoder):
    """
    Encodes item IDs and reveal status.
    """
    
    def __init__(self, item_to_id=None):
        self.item_to_id = item_to_id or {}

    @property
    def dimension(self) -> int:
        return ITEM_ID_DIM + ITEM_KNOWN_DIM

    def encode(self, mon: Any, battle: AbstractBattle) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        if mon is None:
            return vec
            
        item = mon.item
        if item:
            item_key = item.lower().replace(" ", "")
            entry = self.item_to_id.get(item_key, 0)
            if isinstance(entry, dict):
                item_num = entry.get("num", 0)
            else:
                item_num = entry
            vec[0] = float(item_num)
            # Known Flag
            vec[ITEM_ID_DIM] = 1.0
        else:
            vec[0] = 0.0
            vec[ITEM_ID_DIM] = 0.0
            
        return vec
