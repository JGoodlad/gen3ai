from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from typing import Any, Dict


class ObservationEncoder(ABC):
    """Base class for the observation block DESCRIPTORS: each declares its width (``dimension``), its
    layout and how to read its slice back (``describe_vector``).

    The battle-reading ``encode`` (and the team / move-order helpers it used) is DELETED with the Python
    encoder's encode path (T27 P6 slice 6d-2): every row is the Rust encoder's
    (``src/rust_sim/src/encoder/``)."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the total number of dimensions in the encoded vector."""
        pass

    def get_layout(self) -> Dict[str, Any]:
        """
        Returns a dictionary describing the layout of the encoded vector.
        Should return mappings of { field_name: (offset, size) } or nested layouts.
        """
        return {"root": (0, self.dimension)}

    def describe_vector(self, vector: np.ndarray) -> Dict[str, Any]:
        """
        Takes a raw numeric vector and returns a human-readable dictionary
        interpreting the values.
        """
        return {"raw_vector": vector.tolist()}
