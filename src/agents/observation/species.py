import numpy as np
from .base import ObservationEncoder
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.battle.pokemon import Pokemon

class SpeciesEncoder(ObservationEncoder):
    """
    Encodes species ID and base stats.
    Dimension: 7 (1 + 6)
    """
    
    def __init__(self, mapping=None):
        if not mapping:
            raise ValueError("SpeciesEncoder requires a non-empty mapping for enrichment!")
        self.mapping = mapping

    @property
    def dimension(self) -> int:
        return 7

    def encode(self, pokemon: Pokemon, battle: AbstractBattle = None) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        if pokemon is None:
            return vec
            
        # 1. Species ID (1)
        species_id = pokemon.species
        entry = self.mapping.get(species_id, {})
        if isinstance(entry, dict):
            num = entry.get("num", 0)
        else:
            num = entry
        vec[0] = float(num)
        
        # 2. Base Stats (6)
        # Order: HP, Atk, Def, SpA, SpD, Spe
        if isinstance(entry, dict) and "baseStats" in entry:
            stats = entry["baseStats"]
        else:
            # If we are here, something is wrong because mapping should have it
            # But we fallback just in case for robustness during runtime
            stats = pokemon.base_stats
            
        vec[1] = stats.get("hp", 100) / 255.0
        vec[2] = stats.get("atk", 100) / 255.0
        vec[3] = stats.get("def", 100) / 255.0
        vec[4] = stats.get("spa", 100) / 255.0
        vec[5] = stats.get("spd", 100) / 255.0
        vec[6] = stats.get("spe", 100) / 255.0
            
        return vec
