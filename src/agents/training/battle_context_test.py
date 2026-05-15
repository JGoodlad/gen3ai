import pytest
import numpy as np
from agents.training.battle_context import BattleContext, TurnDelta


def _mask(*valid_actions):
    m = np.zeros(11, dtype=np.int8)
    for a in valid_actions:
        m[a] = 1
    return m


def _obs():
    return np.zeros(1021, dtype=np.float32)


def _hp(*fractions):
    hp = np.zeros(6, dtype=np.float32)
    for i, v in enumerate(fractions):
        hp[i] = v
    return hp


def _ctx(**overrides):
    defaults = dict(
        turn=1,
        phase="move_selection",
        mask=_mask(6, 7),
        obs=_obs(),
        our_slot_map={},
        opp_slot_map={},
        our_hp=np.zeros(6, dtype=np.float32),
        opp_hp=np.zeros(6, dtype=np.float32),
        our_active="tyranitar",
        opp_active="salamence",
        our_fainted_count=0,
        opp_fainted_count=0,
    )
    defaults.update(overrides)
    return BattleContext(**defaults)


def test_valid_move_selection_context():
    ctx = _ctx(turn=5, phase="move_selection", mask=_mask(6, 7, 8, 9),
               our_slot_map={"tyranitar": 0, "skarmory": 1},
               opp_slot_map={"salamence": 0})
    assert ctx.turn == 5
    assert ctx.phase == "move_selection"
    assert ctx.mask[6] == 1
    assert ctx.mask[0] == 0


def test_valid_forced_switch_context():
    ctx = _ctx(turn=12, phase="forced_switch", mask=_mask(0, 1, 2),
               our_slot_map={"tyranitar": 0}, opp_slot_map={})
    assert ctx.phase == "forced_switch"
    assert ctx.mask[0] == 1
    assert ctx.mask[6] == 0


def test_empty_mask_is_accepted():
    # Empty masks happen during poke-env transitional states. BattleContext itself
    # does not reject them — embed_battle guards against building a context from one.
    ctx = _ctx(mask=np.zeros(11, dtype=np.int8))
    assert ctx.mask.sum() == 0


def test_wrong_mask_shape_raises():
    with pytest.raises(RuntimeError, match="shape"):
        BattleContext(
            turn=1,
            phase="forced_switch",
            mask=np.ones(9, dtype=np.int8),
            obs=_obs(),
            our_slot_map={},
            opp_slot_map={},
            our_hp=np.zeros(6, dtype=np.float32),
            opp_hp=np.zeros(6, dtype=np.float32),
            our_active="NONE",
            opp_active="NONE",
            our_fainted_count=0,
            opp_fainted_count=0,
        )


def test_slot_maps_are_stored():
    ctx = _ctx(our_slot_map={"tyranitar": 0, "skarmory": 1}, opp_slot_map={"gengar": 0})
    assert ctx.our_slot_map["tyranitar"] == 0
    assert ctx.opp_slot_map["gengar"] == 0


def test_hp_fields_stored():
    ctx = _ctx(our_hp=_hp(1.0, 0.5, 0.0), opp_hp=_hp(0.75, 1.0))
    assert ctx.our_hp[0] == pytest.approx(1.0)
    assert ctx.our_hp[1] == pytest.approx(0.5)
    assert ctx.opp_hp[0] == pytest.approx(0.75)


def test_active_names_stored():
    ctx = _ctx(our_active="tyranitar", opp_active="salamence")
    assert ctx.our_active == "tyranitar"
    assert ctx.opp_active == "salamence"


def test_fainted_counts_stored():
    ctx = _ctx(our_fainted_count=2, opp_fainted_count=1)
    assert ctx.our_fainted_count == 2
    assert ctx.opp_fainted_count == 1


def test_turn_delta_build_detects_faint():
    prev = _ctx(our_fainted_count=0, opp_fainted_count=0,
                our_hp=_hp(1.0, 1.0), opp_hp=_hp(1.0))
    curr = _ctx(our_fainted_count=1, opp_fainted_count=0,
                our_hp=_hp(0.0, 1.0), opp_hp=_hp(0.5))
    delta = TurnDelta.build(prev, curr, action=7)
    assert delta.we_fainted is True
    assert delta.opp_fainted is False
    assert delta.our_hp_delta[0] == pytest.approx(-1.0)
    assert delta.opp_hp_delta[0] == pytest.approx(-0.5)


def test_turn_delta_build_switch_action():
    prev = _ctx(our_active="tyranitar", opp_active="salamence")
    curr = _ctx(our_active="skarmory", opp_active="salamence")
    delta = TurnDelta.build(prev, curr, action=1)  # action < 6 = switch
    assert delta.our_switch_to == "skarmory"
    assert delta.our_prev_active == "tyranitar"
    assert delta.opp_switch_to is None


def test_turn_delta_empty():
    delta = TurnDelta.empty()
    assert delta.we_fainted is False
    assert delta.opp_fainted is False
    assert delta.our_hp_delta.shape == (6,)
    assert delta.our_hp_delta.sum() == 0
