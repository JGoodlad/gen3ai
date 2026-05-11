import numpy as np
from .base import ObservationEncoder
from .constants import ITEM_ID_DIM, ITEM_KNOWN_DIM
from poke_env.battle.abstract_battle import AbstractBattle
from typing import Any

class ItemsEncoder(ObservationEncoder):
    """
    Encodes item IDs and reveal status.
    """
    
    def __init__(self, item_to_id=None, reverse_mapping=None):
        self.item_to_id = item_to_id or {}
        self.reverse_mapping = reverse_mapping or {}

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
            entry = self.item_to_id.get(item_key, {})
            vec[0] = float(entry.get("num", 0))
            # Known Flag
            vec[ITEM_ID_DIM] = 1.0
        else:
            vec[0] = 0.0
            vec[ITEM_ID_DIM] = 0.0
            
        return vec

    def get_layout(self) -> dict:
        return {
            "id": {"offset": 0, "dim": 1}, # We only use the first dim for the actual ID
            "known": {"offset": ITEM_ID_DIM, "dim": ITEM_KNOWN_DIM}
        }

    def describe_vector(self, vector: np.ndarray) -> str:
        # Index 0 is ID, ITEM_ID_DIM is Known Flag
        if vector[ITEM_ID_DIM] < 0.5:
            return "ITM-UNKN"
            
        item_id = int(vector[0])
        if item_id == 0:
            return "NONE"
            
        return self.reverse_mapping.get(item_id, f"Item({item_id})").upper()
