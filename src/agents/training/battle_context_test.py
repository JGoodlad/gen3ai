import pytest
import numpy as np
from unittest.mock import MagicMock
from agents.training.battle_context import BattleContext, TurnDelta
from agents.training.slot_registry import SlotRegistry


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
        our_moves=[None, None, None, None],
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
            our_moves=[None, None, None, None],
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


def _mock_mon(species, hp_fraction=1.0, fainted=False, active=False):
    mon = MagicMock()
    mon.species = species
    mon.current_hp_fraction = hp_fraction
    mon.fainted = fainted
    return mon


def test_from_battle_builds_correct_context():
    our_mon = _mock_mon("tyranitar", hp_fraction=0.8)
    opp_mon = _mock_mon("zapdos", hp_fraction=0.5)

    battle = MagicMock()
    battle.turn = 7
    battle.force_switch = False
    battle.team = {"p1: tyranitar": our_mon}
    battle.opponent_team = {"p2: zapdos": opp_mon}
    battle.active_pokemon = our_mon
    battle.opponent_active_pokemon = opp_mon

    mask = np.zeros(11, dtype=np.int8)
    mask[6] = 1
    obs = np.ones(5, dtype=np.float32)

    our_slots = SlotRegistry()
    opp_slots = SlotRegistry()
    ctx = BattleContext.from_battle(battle, mask, obs, our_slots, opp_slots)

    assert ctx.turn == 7
    assert ctx.phase == "move_selection"
    assert ctx.our_active == "tyranitar"
    assert ctx.opp_active == "zapdos"
    assert ctx.our_hp[our_slots.get("tyranitar")] == pytest.approx(0.8)
    assert ctx.opp_hp[opp_slots.get("zapdos")] == pytest.approx(0.5)
    assert ctx.our_fainted_count == 0
    assert ctx.opp_fainted_count == 0


def test_from_battle_forced_switch_phase():
    battle = MagicMock()
    battle.turn = 3
    battle.force_switch = True
    battle.team = {}
    battle.opponent_team = {}
    battle.active_pokemon = None
    battle.opponent_active_pokemon = None

    ctx = BattleContext.from_battle(
        battle, np.ones(11, dtype=np.int8), np.zeros(1), SlotRegistry(), SlotRegistry()
    )
    assert ctx.phase == "forced_switch"
    assert ctx.our_active == "NONE"


def test_from_battle_slot_registry_mutated():
    our_mon = _mock_mon("skarmory")
    battle = MagicMock()
    battle.turn = 1
    battle.force_switch = False
    battle.team = {"p1: skarmory": our_mon}
    battle.opponent_team = {}
    battle.active_pokemon = our_mon
    battle.opponent_active_pokemon = None

    our_slots = SlotRegistry()
    BattleContext.from_battle(
        battle, np.ones(11, dtype=np.int8), np.zeros(1), our_slots, SlotRegistry()
    )
    assert our_slots.get("skarmory") == 0


def test_turn_delta_empty():
    delta = TurnDelta.empty()
    assert delta.we_fainted is False
    assert delta.opp_fainted is False
    assert delta.our_hp_delta.shape == (6,)
    assert delta.our_hp_delta.sum() == 0


# ---------------------------------------------------------------------------
# our_moves field
# ---------------------------------------------------------------------------

def _mock_battle_with_moves(move_ids, disabled=None):
    """
    Build a minimal mock battle whose last_request contains the given move IDs.
    disabled: set of indices (0-based) that should be marked disabled.
    """
    if disabled is None:
        disabled = set()

    our_mon = _mock_mon("tyranitar", hp_fraction=1.0)
    our_mon.fainted = False

    request_moves = [
        {
            "id": move_id,
            "disabled": (i in disabled),
        }
        for i, move_id in enumerate(move_ids)
    ]

    battle = MagicMock()
    battle.turn = 1
    battle.force_switch = False
    battle.team = {"p1: tyranitar": our_mon}
    battle.opponent_team = {}
    battle.active_pokemon = our_mon
    battle.opponent_active_pokemon = None
    battle.last_request = {
        "active": [{"moves": request_moves}],
    }
    return battle


def test_our_moves_populated_from_last_request():
    """our_moves mirrors the masker's slot assignment for normal moves."""
    battle = _mock_battle_with_moves(
        ["rockslide", "earthquake", "thunderbolt", "surf"]
    )
    mask = _mask(6, 7, 8, 9)
    ctx = BattleContext.from_battle(battle, mask, np.zeros(1), SlotRegistry(), SlotRegistry())
    assert ctx.our_moves == ["rockslide", "earthquake", "thunderbolt", "surf"]


def test_our_moves_disabled_slot_is_none():
    """Disabled move slots are set to None regardless of mask."""
    battle = _mock_battle_with_moves(
        ["rockslide", "earthquake", "thunderbolt", "surf"],
        disabled={1},  # earthquake is disabled
    )
    mask = _mask(6, 8, 9)  # slot 7 (earthquake) disabled
    ctx = BattleContext.from_battle(battle, mask, np.zeros(1), SlotRegistry(), SlotRegistry())
    assert ctx.our_moves[0] == "rockslide"
    assert ctx.our_moves[1] is None   # disabled
    assert ctx.our_moves[2] == "thunderbolt"
    assert ctx.our_moves[3] == "surf"


def test_our_moves_struggle_slot_is_none():
    """Struggle is excluded from our_moves (it uses the dedicated action 10)."""
    battle = _mock_battle_with_moves(["struggle", "struggle", "struggle", "struggle"])
    mask = _mask(10)
    ctx = BattleContext.from_battle(battle, mask, np.zeros(1), SlotRegistry(), SlotRegistry())
    assert ctx.our_moves == [None, None, None, None]


def test_our_moves_always_length_4():
    """our_moves is always a 4-element list, even with fewer than 4 moves."""
    battle = _mock_battle_with_moves(["rockslide", "earthquake"])
    mask = _mask(6, 7)
    ctx = BattleContext.from_battle(battle, mask, np.zeros(1), SlotRegistry(), SlotRegistry())
    assert len(ctx.our_moves) == 4
    assert ctx.our_moves[0] == "rockslide"
    assert ctx.our_moves[1] == "earthquake"
    assert ctx.our_moves[2] is None
    assert ctx.our_moves[3] is None


def test_our_moves_no_request_is_all_none():
    """Forced-switch phase (no active request) yields all-None our_moves."""
    battle = MagicMock()
    battle.turn = 3
    battle.force_switch = True
    battle.team = {}
    battle.opponent_team = {}
    battle.active_pokemon = None
    battle.opponent_active_pokemon = None
    # last_request has no "active" key
    battle.last_request = {}

    ctx = BattleContext.from_battle(
        battle, np.ones(11, dtype=np.int8), np.zeros(1), SlotRegistry(), SlotRegistry()
    )
    assert ctx.our_moves == [None, None, None, None]


# ---------------------------------------------------------------------------
# TurnDelta.build() — our_move_id
# ---------------------------------------------------------------------------

def test_turn_delta_move_action_returns_move_id():
    """Action 6-9 resolves to the move ID stored in prev_ctx.our_moves."""
    prev = _ctx(
        our_active="tyranitar", opp_active="salamence",
        our_moves=["rockslide", "earthquake", "thunderbolt", "surf"],
    )
    curr = _ctx(our_active="tyranitar", opp_active="salamence")
    delta = TurnDelta.build(prev, curr, action=6)
    assert delta.our_move_id == "rockslide"
    assert delta.our_switch_to is None


def test_turn_delta_move_action_slot_7():
    prev = _ctx(
        our_moves=["rockslide", "earthquake", "thunderbolt", "surf"],
    )
    curr = _ctx()
    delta = TurnDelta.build(prev, curr, action=8)
    assert delta.our_move_id == "thunderbolt"


def test_turn_delta_switch_action_returns_none_move_id():
    """Action < 6 is a switch; our_move_id must be None."""
    prev = _ctx(our_active="tyranitar", opp_active="salamence")
    curr = _ctx(our_active="skarmory", opp_active="salamence")
    delta = TurnDelta.build(prev, curr, action=2)
    assert delta.our_move_id is None
    assert delta.our_switch_to == "skarmory"


def test_turn_delta_struggle_action_returns_struggle():
    """Action 10 (struggle) always yields our_move_id == 'struggle'."""
    prev = _ctx(our_moves=[None, None, None, None])
    curr = _ctx()
    delta = TurnDelta.build(prev, curr, action=10)
    assert delta.our_move_id == "struggle"
    assert delta.our_switch_to is None
