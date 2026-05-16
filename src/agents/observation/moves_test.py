import pytest
import numpy as np
from .moves import MovesEncoder
from .state_encoder import load_mappings
from .constants import MOVE_SLOT_DIM, MAX_PP
from unittest.mock import MagicMock

def _make_move(move_id, current_pp, max_pp):
    move = MagicMock()
    move.id = move_id
    move.current_pp = current_pp
    move.max_pp = max_pp
    return move

def test_moves_encoder_dimension():
    mappings = load_mappings()
    encoder = MovesEncoder(mappings["moves"])
    assert encoder.dimension == 4 * MOVE_SLOT_DIM

def test_moves_encoder_known_flags():
    mappings = load_mappings()
    encoder = MovesEncoder(mappings["moves"])
    mon = MagicMock()
    mon.moves = {"surf": _make_move("surf", 16, 16)}

    vec = encoder.encode(mon, None)
    assert vec[0] == 57.0          # Surf ID
    assert vec[6] == 1.0           # known flag, slot 0
    assert vec[MOVE_SLOT_DIM + 6] == 0.0  # known flag, slot 1 (empty)

def test_moves_encoder_pp_full():
    """Full PP: current == max, fraction should be 1.0 for both."""
    mappings = load_mappings()
    encoder = MovesEncoder(mappings["moves"])
    mon = MagicMock()
    mon.moves = {"surf": _make_move("surf", 24, 24)}  # Surf max PP = 24

    vec = encoder.encode(mon, None)
    assert vec[7] == pytest.approx(24 / MAX_PP)  # current_pp
    assert vec[8] == pytest.approx(24 / MAX_PP)  # max_pp

def test_moves_encoder_pp_depleted():
    """Partially depleted PP: current < max, both values distinct."""
    mappings = load_mappings()
    encoder = MovesEncoder(mappings["moves"])
    mon = MagicMock()
    mon.moves = {"surf": _make_move("surf", 8, 24)}  # half depleted

    vec = encoder.encode(mon, None)
    assert vec[7] == pytest.approx(8 / MAX_PP)   # current_pp
    assert vec[8] == pytest.approx(24 / MAX_PP)  # max_pp
    assert vec[7] < vec[8]                        # depletion is visible

def test_moves_encoder_pp_empty_slot():
    """Empty move slot should have both PP dims zero."""
    mappings = load_mappings()
    encoder = MovesEncoder(mappings["moves"])
    mon = MagicMock()
    mon.moves = {"surf": _make_move("surf", 24, 24)}

    vec = encoder.encode(mon, None)
    # Slot 1 is empty — all dims including PP should be 0
    assert vec[MOVE_SLOT_DIM + 7] == 0.0
    assert vec[MOVE_SLOT_DIM + 8] == 0.0
