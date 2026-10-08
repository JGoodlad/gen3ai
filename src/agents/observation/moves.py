from __future__ import annotations
import numpy as np
from .base import ObservationEncoder
from .constants import MOVE_SLOT_DIM
from .types import TypeEncoder
from typing import Any, Dict, Optional

# Hidden Power's Pokémon ID. All 16 typed variants ("hiddenpowergrass" etc.)
# share this num with the bare "hiddenpower" in data/pokemon/gen3_moves.json,
# so a single equality check detects any HP slot regardless of how the type
# was (or wasn't) revealed. Used by the feature extractor to route HP slots
# through the weighted-type-embedding path.
HIDDEN_POWER_MOVE_NUM = 237


class MovesEncoder(ObservationEncoder):
    """
    The 4 × 11-dim move-slot block's layout and read-back (``describe_vector``; ``reverse_mapping``,
    move num → name, names a slot's move). The slots are written by the Rust encoder; the Python
    ``encode`` over a poke-env mon is DELETED (T27 P6 slice 6d-2).
    """

    def __init__(self, reverse_mapping: Optional[Dict[int, str]] = None) -> None:
        self.reverse_mapping: Dict[int, str] = reverse_mapping or {}

    @property
    def dimension(self) -> int:
        return 4 * MOVE_SLOT_DIM

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
                "known": {"offset": 6, "dim": 1},
                "current_pp": {"offset": 7, "dim": 1},
                "max_pp": {"offset": 8, "dim": 1},
                "accuracy": {"offset": 9, "dim": 1},
                "never_miss": {"offset": 10, "dim": 1}
            }
        }

    def describe_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        move_names = []
        for i in range(4):
            base = i * MOVE_SLOT_DIM
            if vector[base + 6] > 0.5:
                mid = int(vector[base])
                name = self.reverse_mapping.get(mid, f"Move({mid})")
                # Hidden Power display (gen3_typed_hidden_power_ids_v1). OUR own HP carries its DISTINCT
                # num (355-370) → the reverse map already gives the typed id (e.g. "hiddenpowergrass");
                # render it in the readable paren form. The OPPONENT's bare HP shares num 237 (type
                # unknowable) → recover the type from the move's TYPE channel only if it was revealed,
                # else it correctly stays the bare "hiddenpower".
                if mid == HIDDEN_POWER_MOVE_NUM:
                    t = TypeEncoder.IDX_TO_TYPE.get(int(vector[base + 4]), "UNKNOWN")
                    if t not in ("UNKNOWN", "NORMAL"):
                        name = f"hiddenpower({t.lower()})"
                elif name.startswith("hiddenpower") and name != "hiddenpower":
                    name = f"hiddenpower({name[len('hiddenpower'):]})"
                move_names.append(name)
        return {"moves": move_names}
