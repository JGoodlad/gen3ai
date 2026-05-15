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

    @staticmethod
    def get_sorted_moves(mon: Any) -> list:
        """Returns a stable, sorted list of move objects for a Pokémon."""
        if mon is None or not hasattr(mon, "moves"):
            return []
        # Sort by ID string to ensure stable slot mapping across observations
        return sorted(mon.moves.values(), key=lambda m: m.id)

    @staticmethod
    def get_team_list(battle: AbstractBattle, is_opponent: bool) -> list:
        """Returns a stable list of Pokémon for a team."""
        if battle is None:
            return []
        if is_opponent:
            team = list(battle.opponent_team.values())
            # Safety: Ensure active opponent is always included in the list
            active_opp = battle.opponent_active_pokemon
            if active_opp and active_opp not in team:
                team.append(active_opp)
        else:
            team = list(battle.team.values())
        return team

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
