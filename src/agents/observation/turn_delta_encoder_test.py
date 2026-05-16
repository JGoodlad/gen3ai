import numpy as np
import pytest
from agents.observation.turn_delta_encoder import TurnDeltaEncoder, TURN_DELTA_DIM
from agents.training.battle_context import TurnDelta

# Minimal move mapping covering the test cases.
_MOVES = {
    "rockslide": {"num": 100, "basePower": 75, "type": "Rock", "hasSecondary": True, "hasRecoil": False},
    "surf":      {"num": 57,  "basePower": 95, "type": "Water", "hasSecondary": False, "hasRecoil": False},
    "struggle":  {"num": 165, "basePower": 50, "type": "Normal", "hasSecondary": False, "hasRecoil": True},
}


def _enc():
    return TurnDeltaEncoder(_MOVES)


def _empty_delta():
    return TurnDelta.empty()


def _delta(**kwargs):
    base = dict(
        our_move_id=None, our_switch_to=None, our_prev_active="tyranitar",
        opp_move_id=None, opp_switch_to=None, opp_prev_active="salamence",
        opp_move_known=True,
        our_hp_delta=np.zeros(6, dtype=np.float32),
        opp_hp_delta=np.zeros(6, dtype=np.float32),
        we_fainted=False, opp_fainted=False,
        our_failed_to_move=False, our_cant_reason=None,
        opp_failed_to_move=False, opp_cant_reason=None,
    )
    base.update(kwargs)
    return TurnDelta(**base)


def test_dimension():
    assert _enc().dimension == TURN_DELTA_DIM
    assert TURN_DELTA_DIM == 29


def test_empty_delta_is_all_zeros_except_opp_move_known():
    enc = _enc()
    vec = enc.encode(_empty_delta())
    assert vec.shape == (TURN_DELTA_DIM,)
    assert vec.dtype == np.float32
    # empty() has opp_move_known=False, everything else 0
    assert vec.sum() == pytest.approx(0.0)


def test_known_move_populates_features():
    enc = _enc()
    delta = _delta(our_move_id="rockslide")
    vec = enc.encode(delta)
    # our_move block: [id_norm, power_norm, has_secondary, has_recoil, type_id_norm]
    our_block = vec[:5]
    assert our_block[0] > 0.0      # id_norm
    assert our_block[1] == pytest.approx(75.0 / 200.0)   # power
    assert our_block[2] == pytest.approx(1.0)             # has_secondary
    assert our_block[3] == pytest.approx(0.0)             # no recoil
    assert our_block[4] > 0.0                             # type_id_norm (Rock != 0)


def test_unknown_move_is_zeros():
    enc = _enc()
    delta = _delta(our_move_id=None)
    vec = enc.encode(delta)
    assert vec[:5].sum() == pytest.approx(0.0)


def test_opp_move_block_offset():
    enc = _enc()
    delta = _delta(opp_move_id="surf")
    vec = enc.encode(delta)
    opp_block = vec[5:10]
    assert opp_block[0] > 0.0
    assert opp_block[1] == pytest.approx(95.0 / 200.0)


def test_switched_flag():
    enc = _enc()
    delta = _delta(our_switch_to="skarmory")
    vec = enc.encode(delta)
    assert vec[10] == pytest.approx(1.0)   # our_switched at offset 10
    assert vec[11] == pytest.approx(0.0)   # opp_switched


def test_failed_to_move_flag():
    enc = _enc()
    delta = _delta(our_failed_to_move=True, our_cant_reason="par")
    vec = enc.encode(delta)
    assert vec[12] == pytest.approx(1.0)   # our_failed_to_move at offset 12
    assert vec[13] == pytest.approx(0.0)   # opp_failed_to_move
    # our_cant_reason onehot: par=index 0, at offset 14
    assert vec[14] == pytest.approx(1.0)


def test_opp_cant_reason_onehot():
    enc = _enc()
    delta = _delta(opp_failed_to_move=True, opp_cant_reason="flinch")
    vec = enc.encode(delta)
    # opp_cant_onehot starts at offset 19 (14 + 5)
    # flinch is index 3 in [par, slp, frz, flinch, confusion]
    assert vec[19 + 3] == pytest.approx(1.0)
    assert vec[19 + 0] == pytest.approx(0.0)


def test_hp_delta():
    enc = _enc()
    our_hp = np.zeros(6, dtype=np.float32)
    opp_hp = np.zeros(6, dtype=np.float32)
    our_hp[0] = -0.3   # we took 30% damage
    opp_hp[0] = -0.5   # opp took 50%
    delta = _delta(our_hp_delta=our_hp, opp_hp_delta=opp_hp)
    vec = enc.encode(delta)
    # hp deltas at offset 24 and 25
    assert vec[24] == pytest.approx(-0.3)
    assert vec[25] == pytest.approx(-0.5)


def test_faint_flags():
    enc = _enc()
    delta = _delta(opp_fainted=True)
    vec = enc.encode(delta)
    assert vec[26] == pytest.approx(0.0)   # we_fainted
    assert vec[27] == pytest.approx(1.0)   # opp_fainted


def test_opp_move_known_flag():
    enc = _enc()
    delta = _delta(opp_move_known=True)
    vec = enc.encode(delta)
    assert vec[28] == pytest.approx(1.0)

    delta2 = _delta(opp_move_known=False)
    vec2 = enc.encode(delta2)
    assert vec2[28] == pytest.approx(0.0)


def test_unknown_move_id_gracefully_zeros():
    enc = _enc()
    delta = _delta(our_move_id="shadowball")  # not in _MOVES
    vec = enc.encode(delta)
    assert vec[:5].sum() == pytest.approx(0.0)


def test_struggle_recoil_flag():
    enc = _enc()
    delta = _delta(our_move_id="struggle")
    vec = enc.encode(delta)
    assert vec[3] == pytest.approx(1.0)  # has_recoil
