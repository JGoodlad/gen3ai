import numpy as np
from .base import ObservationEncoder
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.pokemon import Pokemon
from typing import Dict, Any

class SpeciesEncoder(ObservationEncoder):
    """
    Encodes species ID and base stats.
    Dimension: 7 (1 + 6)
    """
    
    def __init__(self, mapping=None, reverse_mapping=None):
        if not mapping:
            raise ValueError("SpeciesEncoder requires a non-empty mapping for enrichment!")
        self.mapping = mapping
        self.reverse_mapping = reverse_mapping or {}

    @property
    def dimension(self) -> int:
        return 7

    def encode(self, pokemon: Pokemon, battle: AbstractBattle = None) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        if pokemon is None:
            return vec
            
        # 1. Species ID (1)
        species_id = str(pokemon.species)
        if species_id not in self.mapping:
            raise ValueError(f"Unrecognized species: {species_id}. Update data/mappings/gen3_mapping.json")
            
        entry = self.mapping.get(species_id, {})
        vec[0] = float(entry.get("num", 0))
        
        # 2. Base Stats (6)
        # Order: HP, Atk, Def, SpA, SpD, Spe
        if "baseStats" in entry:
            stats = entry["baseStats"]
        else:
            # Fallback for robustness during runtime
            stats = pokemon.base_stats
            
        vec[1] = stats.get("hp", 100) / 255.0
        vec[2] = stats.get("atk", 100) / 255.0
        vec[3] = stats.get("def", 100) / 255.0
        vec[4] = stats.get("spa", 100) / 255.0
        vec[5] = stats.get("spd", 100) / 255.0
        vec[6] = stats.get("spe", 100) / 255.0
            
        return vec

    def get_layout(self) -> Dict[str, Any]:
        return {
            "species_id": (0, 1),
            "base_stats": (1, 6)
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
