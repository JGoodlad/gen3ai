from abc import ABC, abstractmethod
import numpy as np
from poke_env.battle.abstract_battle import AbstractBattle
from typing import Any, Dict

class ObservationEncoder(ABC):
    """Base class for all observation encoders."""
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the total number of dimensions in the encoded vector."""
        pass

    @abstractmethod
    def encode(self, item: Any, battle: AbstractBattle) -> np.ndarray:
        """Encodes the item into a numpy array."""
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
