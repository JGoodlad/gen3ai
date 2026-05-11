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
        self.ability_to_id = ability_to_id or {}
        self.reverse_mapping = reverse_mapping or {}

    @property
    def dimension(self) -> int:
        # Ability 1 (8) + Known (1) + Ability 2 (8) + Ability 3 (8) = 25
        return (3 * ABILITY_SLOT_DIM) + ABILITY_KNOWN_DIM

    def encode(self, mon: Any, battle: AbstractBattle) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        if mon is None:
            return vec
            
        # 1. Revealed Ability
        ability = mon.ability
        if ability:
            ability_key = ability.lower().replace(" ", "")
            entry = self.ability_to_id.get(ability_key, 0)
            if isinstance(entry, dict):
                ab_num = entry.get("num", 0)
            else:
                ab_num = entry
            vec[0] = float(ab_num)
            # Known Flag
            # For our team, it's always known. For opponent, if it's set, it's revealed.
            vec[ABILITY_SLOT_DIM] = 1.0
        else:
            # Not revealed
            vec[0] = 0.0
            vec[ABILITY_SLOT_DIM] = 0.0
            
        # 2. Possible Abilities (Ability 2/3)
        # This requires dex data to know what's possible for the species.
        # For now, we'll leave them as 0 or implement a basic lookup if possible.
        # TODO: Implement possible abilities lookup.
        
        return vec

    def get_layout(self) -> dict:
        return {
            "id": {"offset": 0, "dim": 1},
            "known": {"offset": ABILITY_SLOT_DIM, "dim": ABILITY_KNOWN_DIM}
        }
