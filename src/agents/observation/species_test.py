import pytest
import numpy as np
from unittest.mock import MagicMock
from .species import SpeciesEncoder

def test_species_encoder():
    mapping = {
        "jolteon": {
            "num": 135,
            "baseStats": {"hp": 65, "atk": 65, "def": 60, "spa": 110, "spd": 95, "spe": 130}
        }
    }
    encoder = SpeciesEncoder(mapping)
    
    mon = MagicMock()
    mon.species = "jolteon"
    
    vec = encoder.encode(mon)
    assert vec[0] == 135.0
    assert vec[1] == pytest.approx(65/255) # HP
    assert vec[6] == pytest.approx(130/255) # Spe
