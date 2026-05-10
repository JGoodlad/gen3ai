from abc import ABC, abstractmethod
import numpy as np
from poke_env.battle.abstract_battle import AbstractBattle
from typing import Any

class ObservationEncoder(ABC):
    """Base class for all observation encoders."""
    
    @property
    @abstractmethod
    def dimension(self) -> int:
        """Returns the number of dimensions in the encoded vector."""
        pass

    @abstractmethod
    def encode(self, item: Any, battle: AbstractBattle) -> np.ndarray:
        """Encodes the item into a numpy array."""
        pass
