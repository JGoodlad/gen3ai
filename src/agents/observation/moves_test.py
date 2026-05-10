import pytest
import numpy as np
from .moves import MovesEncoder
from unittest.mock import MagicMock

def test_moves_encoder_dimension():
    encoder = MovesEncoder()
    # (4 * 8) + 4 = 36
    assert encoder.dimension == 36

def test_moves_encoder_known_flags():
    encoder = MovesEncoder(move_to_id={"surf": 1})
    mon = MagicMock()
    move = MagicMock()
    move.id = "surf"
    mon.moves = {"surf": move}
    
    vec = encoder.encode(mon, None)
    # Move 1 ID is at index 0
    assert vec[0] == 1.0
    # Known flag for Move 1 is at index 32
    assert vec[32] == 1.0
    # Known flag for Move 2 is at index 33 (should be 0)
    assert vec[33] == 0.0
