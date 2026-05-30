import pytest
import numpy as np
from .moves import MovesEncoder
from .state_encoder import load_mappings
from .constants import MOVE_SLOT_DIM, MAX_PP
from .types import TypeEncoder
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


# ---------------------------------------------------------------------------
# Accuracy encoding
# ---------------------------------------------------------------------------

def test_moves_encoder_accuracy_can_miss():
    """A move that can miss encodes accuracy/100 and never_miss = 0."""
    mappings = load_mappings()
    encoder = MovesEncoder(mappings["moves"])
    mon = MagicMock()
    mon.moves = {"hydropump": _make_move("hydropump", 8, 8)}  # 80% accuracy

    vec = encoder.encode(mon, None)
    assert vec[9] == pytest.approx(0.80)  # accuracy_norm
    assert vec[10] == 0.0                 # never_miss flag


def test_moves_encoder_never_miss():
    """A never-miss move carries accuracy=100 in the mapping (→ 1.0 normalized)
    plus the never_miss flag, so it encodes as [1.0, 1] — same scalar as a plain
    100%-accuracy move, distinguished only by the bit."""
    mappings = load_mappings()
    encoder = MovesEncoder(mappings["moves"])
    mon = MagicMock()
    mon.moves = {"swift": _make_move("swift", 32, 32)}  # never_miss in mapping

    vec = encoder.encode(mon, None)
    assert vec[9] == pytest.approx(1.0)  # accuracy 100 / 100
    assert vec[10] == 1.0                # never_miss flag set


def test_moves_encoder_ohko_accuracy():
    """OHKO moves keep their honest (equal-level) 30% accuracy and are NOT
    flagged never-miss."""
    mappings = load_mappings()
    encoder = MovesEncoder(mappings["moves"])
    mon = MagicMock()
    mon.moves = {"fissure": _make_move("fissure", 8, 8)}  # 30% accuracy, ohko

    vec = encoder.encode(mon, None)
    assert vec[9] == pytest.approx(0.30)
    assert vec[10] == 0.0


def test_moves_encoder_accuracy_empty_slot():
    """Empty move slot leaves accuracy dims zero (never_miss = 0, accuracy_norm = 0)."""
    mappings = load_mappings()
    encoder = MovesEncoder(mappings["moves"])
    mon = MagicMock()
    mon.moves = {"surf": _make_move("surf", 24, 24)}

    vec = encoder.encode(mon, None)
    assert vec[MOVE_SLOT_DIM + 9] == 0.0
    assert vec[MOVE_SLOT_DIM + 10] == 0.0


# ---------------------------------------------------------------------------
# Hidden Power type/power encoding
# ---------------------------------------------------------------------------

def test_hiddenpower_bare_id_is_unknown_type_70bp():
    """Opponent HP arrives as bare 'hiddenpower'. The mapping entry has
    basePower=0 and type=Normal — neither correct — so the encoder must
    override to 70bp and type_id=0 (unknown sentinel)."""
    mappings = load_mappings()
    encoder = MovesEncoder(mappings["moves"])
    mon = MagicMock()
    mon.moves = {"hiddenpower": _make_move("hiddenpower", 24, 24)}

    vec = encoder.encode(mon, None)
    assert vec[1] == pytest.approx(70.0 / 200.0), "bare HP must encode 70bp"
    assert vec[4] == 0.0, "bare HP must encode type_id=0 (unknown)"


def test_hiddenpower_typed_grass_uses_real_type():
    """Our own Hidden Power [Grass] arrives as 'hiddenpowergrass' after the
    poke-env raw_id patch. The mapping entry already has type=Grass and
    basePower=70, so the else branch picks them up automatically."""
    mappings = load_mappings()
    encoder = MovesEncoder(mappings["moves"])
    mon = MagicMock()
    mon.moves = {"hiddenpowergrass": _make_move("hiddenpowergrass", 24, 24)}

    vec = encoder.encode(mon, None)
    assert vec[1] == pytest.approx(70.0 / 200.0), "typed HP must encode 70bp"
    assert vec[4] == float(TypeEncoder.TYPE_TO_IDX["GRASS"]), (
        "typed HP must encode the real type_id (Grass)"
    )


def test_hiddenpower_typed_fire_uses_real_type():
    """A different type to confirm the mapping isn't accidentally constant."""
    mappings = load_mappings()
    encoder = MovesEncoder(mappings["moves"])
    mon = MagicMock()
    mon.moves = {"hiddenpowerfire": _make_move("hiddenpowerfire", 24, 24)}

    vec = encoder.encode(mon, None)
    assert vec[4] == float(TypeEncoder.TYPE_TO_IDX["FIRE"])
    assert vec[4] != float(TypeEncoder.TYPE_TO_IDX["GRASS"])
