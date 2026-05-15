import pytest
import numpy as np
from .types import TypeEncoder
from unittest.mock import MagicMock

def test_type_encoder_dimension():
    encoder = TypeEncoder()
    assert encoder.dimension == 2

def test_type_encoder_order_invariance():
    encoder = TypeEncoder()
    
    # Water/Ground
    mon1 = MagicMock()
    mon1.type_1 = MagicMock()
    mon1.type_1.name = "WATER"
    mon1.type_2 = MagicMock()
    mon1.type_2.name = "GROUND"
    
    # Ground/Water
    mon2 = MagicMock()
    mon2.type_1 = MagicMock()
    mon2.type_1.name = "GROUND"
    mon2.type_2 = MagicMock()
    mon2.type_2.name = "WATER"
    
    vec1 = encoder.encode(mon1, None)
    vec2 = encoder.encode(mon2, None)
    
    assert np.array_equal(vec1, vec2)
