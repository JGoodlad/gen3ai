import pytest
import numpy as np
from .pokemon import PokemonEncoder
from .species import SpeciesEncoder
from .items import ItemsEncoder
from .types import TypeEncoder
from .abilities import AbilitiesEncoder
from .moves import MovesEncoder
from .state_encoder import load_mappings

def test_pokemon_encoder_dimension():
    mappings = load_mappings()
    se = SpeciesEncoder(mappings["species"])
    ie = ItemsEncoder(mappings["items"])
    te = TypeEncoder()
    ae = AbilitiesEncoder(mappings["abilities"])
    me = MovesEncoder(mappings["moves"])
    
    encoder = PokemonEncoder(se, ie, te, ae, me)
    # 98 dims (Reduced from 102)
    assert encoder.dimension == 98

def test_pokemon_encoder_empty():
    mappings = load_mappings()
    se = SpeciesEncoder(mappings["species"])
    ie = ItemsEncoder(mappings["items"])
    te = TypeEncoder()
    ae = AbilitiesEncoder(mappings["abilities"])
    me = MovesEncoder(mappings["moves"])
    
    encoder = PokemonEncoder(se, ie, te, ae, me)
    vec = encoder.encode(None, None)
    assert len(vec) == 98
    assert np.all(vec == 0)
