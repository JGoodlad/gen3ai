import numpy as np
from .base import ObservationEncoder
from .constants import COMBINED_TYPES_DIM
from poke_env.battle.abstract_battle import AbstractBattle
from typing import Any, Dict

class TypeEncoder(ObservationEncoder):
    """
    Encodes Pokémon types into a shared embedding space.
    Dimension: 8 (Reserved for E(T1) + E(T2) in the model)
    """
    
    TYPE_TO_IDX = {
        "NORMAL": 1, "FIRE": 2, "WATER": 3, "GRASS": 4, "ELECTRIC": 5,
        "ICE": 6, "FIGHTING": 7, "POISON": 8, "GROUND": 9, "FLYING": 10,
        "PSYCHIC": 11, "BUG": 12, "ROCK": 13, "GHOST": 14, "DRAGON": 15,
        "STEEL": 16, "DARK": 17, "???": 18
    }
    
    IDX_TO_TYPE = {v: k for k, v in TYPE_TO_IDX.items()}
    IDX_TO_TYPE[0] = "UNKNOWN"  # idx 0 = unset/unknown sentinel (not a real type)

    @property
    def dimension(self) -> int:
        return COMBINED_TYPES_DIM

    def encode(self, mon: Any, battle: AbstractBattle) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        if mon is None:
            return vec
            
        # Get types and sort them for order invariance (e.g. Water/Ground == Ground/Water)
        types = []
        if mon.type_1:
            types.append(mon.type_1.name)
        if mon.type_2:
            types.append(mon.type_2.name)
            
        types.sort()
        
        # Place IDs in the first two slots; the model will sum their embeddings
        for i, tname in enumerate(types):
            if i < 2:
                vec[i] = float(self.TYPE_TO_IDX.get(tname, 0))
                
        return vec

    def get_layout(self) -> Dict[str, Any]:
        return {
            "type1": {"offset": 0, "dim": 1},
            "type2": {"offset": 1, "dim": 1}
        }

    # Why the `type: ignore[override]` below — the base returns a dict; the three SUB-encoders (types /
    # items / abilities) deliberately return a compact STRING, because they are never
    # rendered on their own: `PokemonEncoder.describe_vector` embeds each one as a single
    # dict VALUE ("types": "WATER/GROUND"). Narrowing the return here rather than widening
    # the base keeps the dict contract real for the encoders the prober calls directly.
    def describe_vector(self, vector: np.ndarray) -> str:  # type: ignore[override]
        t1_idx = int(vector[0])
        t2_idx = int(vector[1])
        
        t1 = self.IDX_TO_TYPE.get(t1_idx, "NONE")
        t2 = self.IDX_TO_TYPE.get(t2_idx, "NONE")
        
        if t2 == "NONE":
            return t1
        return f"{t1}/{t2}"
