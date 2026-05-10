import pytest
import numpy as np
from .pokemon import PokemonEncoder
from .species import SpeciesEncoder
from .items import ItemsEncoder
from .types import TypeEncoder
from .abilities import AbilitiesEncoder
from .moves import MovesEncoder
from unittest.mock import MagicMock

def test_pokemon_encoder_dimension():
    # Setup sub-encoders
    se = SpeciesEncoder()
    ie = ItemsEncoder()
    te = TypeEncoder()
    ae = AbilitiesEncoder()
    me = MovesEncoder()
    
    encoder = PokemonEncoder(se, ie, te, ae, me)
    assert encoder.dimension == 132

def test_pokemon_encoder_empty():
    se = SpeciesEncoder()
    ie = ItemsEncoder()
    te = TypeEncoder()
    ae = AbilitiesEncoder()
    me = MovesEncoder()
    encoder = PokemonEncoder(se, ie, te, ae, me)
    
    vec = encoder.encode(None, None)
    assert vec.shape == (132,)
    assert np.all(vec == 0)
