from __future__ import annotations
import numpy as np
from .base import ObservationEncoder
from .constants import ITEM_ID_DIM, ITEM_KNOWN_DIM, ITEM_CONSUMED_DIM
from typing import Any, Dict, Optional


class ItemsEncoder(ObservationEncoder):
    """
    Encodes item IDs and reveal status.
    """

    def __init__(self,
                 item_to_id: Optional[Dict[str, Any]] = None,
                 reverse_mapping: Optional[Dict[int, str]] = None) -> None:
        if not item_to_id:
            raise ValueError("ItemsEncoder requires a non-empty mapping!")
        self.item_to_id = item_to_id
        self.reverse_mapping = reverse_mapping or {}

    @property
    def dimension(self) -> int:
        return ITEM_ID_DIM + ITEM_KNOWN_DIM + ITEM_CONSUMED_DIM

    def get_layout(self) -> dict:
        return {
            "id": {"offset": 0, "dim": 1},
            "known": {"offset": ITEM_ID_DIM, "dim": ITEM_KNOWN_DIM},
            "consumed": {"offset": ITEM_ID_DIM + ITEM_KNOWN_DIM, "dim": ITEM_CONSUMED_DIM},
        }

    # Why the `type: ignore[override]` below — compact-string sub-encoder; see TypeEncoder.describe_vector.
    def describe_vector(self, vector: np.ndarray) -> str:  # type: ignore[override]
        known = vector[ITEM_ID_DIM] >= 0.5
        consumed = vector[ITEM_ID_DIM + ITEM_KNOWN_DIM] >= 0.5

        if not known:
            return "ITM-UNKN"

        item_id = int(vector[0])
        name = self.reverse_mapping.get(item_id, f"Item({item_id})").upper() if item_id else "NONE"
        return f"{name}(CONSUMED)" if consumed else name
