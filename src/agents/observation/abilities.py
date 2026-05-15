import numpy as np
from .base import ObservationEncoder
from .constants import ABILITY_SLOT_DIM, ABILITY_KNOWN_DIM
from poke_env.battle.abstract_battle import AbstractBattle
from typing import Any

class AbilitiesEncoder(ObservationEncoder):
    """
    Encodes revealed and possible abilities.
    """
    
    def __init__(self, ability_to_id=None, reverse_mapping=None):
        if not ability_to_id:
            raise ValueError("AbilitiesEncoder requires a non-empty mapping!")
        self.ability_to_id = ability_to_id
        self.reverse_mapping = reverse_mapping or {}

    @property
    def dimension(self) -> int:
        return ABILITY_SLOT_DIM + ABILITY_KNOWN_DIM

    def encode(self, mon: Any, battle: AbstractBattle) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        if mon is None:
            return vec
            
        # 1. Revealed Ability
        ability = mon.ability
        if ability:
            ability_key = ability.lower().replace(" ", "").replace("_", "")
            
            # Handle poke-env's special "unknown_ability" placeholder
            if ability_key == "unknownability":
                vec[0] = 0.0
                vec[ABILITY_SLOT_DIM] = 0.0
                return vec

            if ability_key not in self.ability_to_id:
                raise ValueError(f"Unrecognized ability: {ability_key}. Update data/pokemon/gen3_abilities.json")
            entry = self.ability_to_id[ability_key]
            vec[0] = float(entry.get("num", 0))
            # Known Flag
            # For our team, it's always known. For opponent, if it's set, it's revealed.
            vec[ABILITY_SLOT_DIM] = 1.0
        else:
            # Not revealed
            vec[0] = 0.0
            vec[ABILITY_SLOT_DIM] = 0.0
            
        return vec

    def get_layout(self) -> dict:
        return {
            "id": {"offset": 0, "dim": 1},
            "known": {"offset": ABILITY_SLOT_DIM, "dim": ABILITY_KNOWN_DIM}
        }

    def describe_vector(self, vector: np.ndarray) -> str:
        # Index 0 is ID, ABILITY_SLOT_DIM is Known Flag
        if vector[ABILITY_SLOT_DIM] < 0.5:
            return "ABLY-UNKN"
            
        ab_id = int(vector[0])
        if ab_id == 0:
            return "NONE"
            
        return self.reverse_mapping.get(ab_id, f"Ably({ab_id})").upper()
