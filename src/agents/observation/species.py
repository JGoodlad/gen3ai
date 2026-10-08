from __future__ import annotations
import numpy as np
from .base import ObservationEncoder
from typing import Dict, Any, Optional


class SpeciesEncoder(ObservationEncoder):
    """
    Encodes species ID and base stats.
    Dimension: 7 (1 + 6)
    """

    def __init__(self,
                 mapping: Optional[Dict[str, Any]] = None,
                 reverse_mapping: Optional[Dict[int, str]] = None) -> None:
        if not mapping:
            raise ValueError("SpeciesEncoder requires a non-empty mapping for enrichment!")
        self.mapping = mapping
        self.reverse_mapping = reverse_mapping or {}

    @property
    def dimension(self) -> int:
        return 7

    def get_layout(self) -> Dict[str, Any]:
        return {
            "species_id": {"offset": 0, "dim": 1},
            "base_stats": {"offset": 1, "dim": 6}
        }

    def describe_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        sid = int(vector[0])
        name = self.reverse_mapping.get(sid, f"Unknown({sid})")
        return {
            "name": name,
            "hp": f"{vector[1]*255:.0f}",
            "atk": f"{vector[2]*255:.0f}",
            "def": f"{vector[3]*255:.0f}",
            "spa": f"{vector[4]*255:.0f}",
            "spd": f"{vector[5]*255:.0f}",
            "spe": f"{vector[6]*255:.0f}"
        }
