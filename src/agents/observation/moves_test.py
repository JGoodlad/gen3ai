import pytest
import numpy as np
from .moves import MovesEncoder
from .state_encoder import load_mappings
from unittest.mock import MagicMock

def test_moves_encoder_dimension():
    mappings = load_mappings()
    encoder = MovesEncoder(mappings["moves"])
    assert encoder.dimension == 4 * 8

def test_moves_encoder_known_flags():
    mappings = load_mappings()
    encoder = MovesEncoder(mappings["moves"])
    mon = MagicMock()
    move = MagicMock()
    move.id = "surf"
    mon.moves = {"surf": move}
    
    vec = encoder.encode(mon, None)
    # Move 1 ID is at index 0 (Surf ID is 57)
    assert vec[0] == 57.0
    # Known flag for Move 1 is at index 6 (Interleaved)
    assert vec[6] == 1.0
    # Known flag for Move 2 is at index 14 (8 + 6) (should be 0)
    assert vec[14] == 0.0
