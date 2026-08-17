import numpy as np
from .base import ObservationEncoder
from .constants import ITEM_ID_DIM, ITEM_KNOWN_DIM, ITEM_CONSUMED_DIM
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env import to_id_str  # canonical id normalizer (accepted poke-env string-util touch)
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

    def encode(self, mon: Any, battle: AbstractBattle) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        if mon is None:
            return vec

        item = mon.item
        consumed = getattr(mon, "consumed_item", None)

        if item:
            item_key = item.lower().replace(" ", "").replace("_", "")

            if item_key == "unknownitem":
                # Opponent's item not yet revealed — all zeros
                return vec

            if item_key not in self.item_to_id:
                raise ValueError(f"Unrecognized item: {item_key}. Update data/pokemon/gen3_items.json")
            entry = self.item_to_id[item_key]
            vec[0] = float(entry.get("num", 0))
            vec[ITEM_ID_DIM] = 1.0  # known
            # consumed stays 0 — item is still held
        elif consumed:
            # `consumed_item` can arrive name-form (e.g. "King's Rock"); `to_id_str` gives the
            # canonical id ("kingsrock") that matches the id-form mapping keys. The old manual
            # space/underscore strip missed apostrophes/hyphens (King's Rock / Never-Melt Ice /
            # Up-Grade) → those fell through, yet known/consumed were still set → a phantom
            # "[id=0, known=1, consumed=1]" (consumed the NONE item). Gate the bits on a successful
            # map so a genuinely unmappable consumed item reads clean all-zeros (unknown).
            entry = self.item_to_id.get(to_id_str(consumed))
            if entry is not None:
                vec[0] = float(entry.get("num", 0))
                vec[ITEM_ID_DIM] = 1.0                    # we observed what it was
                vec[ITEM_ID_DIM + ITEM_KNOWN_DIM] = 1.0   # consumed

        return vec

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
