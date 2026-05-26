import pytest
import numpy as np
from unittest.mock import MagicMock
from agents.training.battle_context import BattleContext, TurnDelta
from agents.training.slot_registry import SlotRegistry
from agents.gen3_mechanics import BOOST_DIM, BOOST_STATS


def _mask(*valid_actions):
    m = np.zeros(11, dtype=np.int8)
    for a in valid_actions:
        m[a] = 1
    return m


def _hp(*fractions):
    hp = np.zeros(6, dtype=np.float32)
    for i, v in enumerate(fractions):
        hp[i] = v
    return hp


def _boosts(**stages):
    arr = np.zeros(BOOST_DIM, dtype=np.int8)
    for stat, val in stages.items():
        arr[BOOST_STATS.index(stat)] = val
    return arr


def _ctx(**overrides):
    defaults = dict(
        turn=1,
        phase="move_selection",
        mask=_mask(6, 7),
        our_slot_map={},
        opp_slot_map={},
        our_hp=np.zeros(6, dtype=np.float32),
        opp_hp=np.zeros(6, dtype=np.float32),
        our_active="tyranitar",
        opp_active="salamence",
        our_fainted_count=0,
        opp_fainted_count=0,
        active_move_ids=[None, None, None, None],
        opp_last_move_id=None,
        opp_all_last_move_ids={},
        opp_active_revealed_moves=frozenset(),
        our_cant_reason=None,
        opp_cant_reason=None,
        our_boosts=np.zeros(BOOST_DIM, dtype=np.int8),
        opp_boosts=np.zeros(BOOST_DIM, dtype=np.int8),
        our_last_effectiveness=None,
        opp_last_effectiveness=None,
        we_moved_first=None,
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
    ctx = _ctx(mask=np.zeros(11, dtype=np.int8))
    assert ctx.mask.sum() == 0


def test_wrong_mask_shape_raises():
    with pytest.raises(RuntimeError, match="shape"):
        BattleContext(
            turn=1,
            phase="forced_switch",
            mask=np.ones(9, dtype=np.int8),
            our_slot_map={},
            opp_slot_map={},
            our_hp=np.zeros(6, dtype=np.float32),
            opp_hp=np.zeros(6, dtype=np.float32),
            our_active="NONE",
            opp_active="NONE",
            our_fainted_count=0,
            opp_fainted_count=0,
            active_move_ids=[None, None, None, None],
            opp_last_move_id=None,
            opp_all_last_move_ids={},
            opp_active_revealed_moves=frozenset(),
            our_cant_reason=None,
            opp_cant_reason=None,
            our_boosts=np.zeros(BOOST_DIM, dtype=np.int8),
            opp_boosts=np.zeros(BOOST_DIM, dtype=np.int8),
            our_last_effectiveness=None,
            opp_last_effectiveness=None,
            we_moved_first=None,
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


# ---------------------------------------------------------------------------
# TurnDelta — our action
# ---------------------------------------------------------------------------

def test_turn_delta_build_switch_action():
    # our_team_order makes action=1 unambiguously map to "skarmory" — the species
    # we sent in — independent of who happens to be active in curr_ctx.
    prev = _ctx(
        our_active="tyranitar",
        opp_active="salamence",
        our_team_order=("tyranitar", "skarmory", "blissey", "swampert", "lanturn", "snorlax"),
    )
    curr = _ctx(our_active="skarmory", opp_active="salamence")
    delta = TurnDelta.build(prev, curr, action=1)  # action < 6 = switch
    assert delta.our_switch_to == "skarmory"
    assert delta.our_move_id is None
    assert delta.our_prev_active == "tyranitar"
    assert delta.opp_switch_to is None


def test_opp_resolved_move_id_prefers_event_then_falls_back():
    """`delta.opp_resolved_move_id` should return the damaging event's move_id
    when set (protocol truth) and fall back to delta.opp_move_id otherwise
    (non-damaging move case, or no event ever promoted)."""
    from poke_env.battle.abstract_battle import DamagingMoveEvent

    # 1. Event set → use event's move_id (even if delta.opp_move_id disagrees).
    event = DamagingMoveEvent(
        user_species="zapdos", target_species="snorlax",
        target_status=None, move_id="hiddenpower", effectiveness=0.5,
    )
    prev = _ctx(our_team_order=("snorlax",))
    curr = _ctx()
    # Inject the event via TurnDelta directly (build() pulls from curr_ctx).
    curr_with_event = _ctx(opp_last_damaging_event=event)
    delta = TurnDelta.build(prev, curr_with_event, action=6)
    assert delta.opp_resolved_move_id == "hiddenpower"

    # 2. No event → use delta.opp_move_id.
    curr_no_event = _ctx(opp_last_move_id="roar")  # opp_last_damaging_event=None by default
    delta_no_event = TurnDelta.build(prev, curr_no_event, action=6)
    assert delta_no_event.opp_resolved_move_id == "roar"

    # 3. Neither set → None.
    delta_none = TurnDelta.build(prev, _ctx(), action=6)
    assert delta_none.opp_resolved_move_id is None


def test_turn_delta_build_switch_action_uses_intent_not_end_state():
    """A voluntary switch's our_switch_to must reflect the action's intent (the
    species in the chosen team slot), not curr_ctx.our_active — which can be a
    forced-replacement after the switch-in fainted and a faint-chain cycled in
    more mons. Regression for the HiddenPower target-resolution bug caught by
    the fuzz validator at 0.05%/battle.
    """
    # Turn N: active=Arcanine. We choose action=3 (slot 3 = Swampert).
    prev = _ctx(
        our_active="arcanine",
        our_team_order=("arcanine", "salamence", "lanturn", "swampert", "snorlax", "forretress"),
    )
    # Turn N+1 snapshot: Swampert died to HP, Lanturn died to Spikes on switch-in,
    # Snorlax is the final replacement.
    curr = _ctx(our_active="snorlax")
    delta = TurnDelta.build(prev, curr, action=3)
    assert delta.our_switch_to == "swampert"  # intent, not "snorlax" (end-state)


def test_turn_delta_build_our_move_id():
    prev = _ctx(active_move_ids=["earthquake", "rockslide", "crunch", "fireblast"])
    curr = _ctx()
    delta = TurnDelta.build(prev, curr, action=7)   # slot 1 → "rockslide"
    assert delta.our_move_id == "rockslide"
    assert delta.our_switch_to is None


def test_turn_delta_build_our_move_id_first_slot():
    prev = _ctx(active_move_ids=["surf", "icebeam", None, None])
    curr = _ctx()
    delta = TurnDelta.build(prev, curr, action=6)   # slot 0 → "surf"
    assert delta.our_move_id == "surf"


def test_turn_delta_build_struggle():
    prev = _ctx(active_move_ids=[None, None, None, None])
    curr = _ctx()
    delta = TurnDelta.build(prev, curr, action=10)
    assert delta.our_move_id == "struggle"
    assert delta.our_switch_to is None


def test_turn_delta_build_our_move_slot_missing_ids():
    # Slot 3 has None in active_move_ids (e.g. PP-depleted move).
    prev = _ctx(active_move_ids=["surf", "icebeam", "thunderbolt", None])
    curr = _ctx()
    delta = TurnDelta.build(prev, curr, action=9)   # slot 3 — None move
    assert delta.our_move_id is None


# ---------------------------------------------------------------------------
# TurnDelta — opponent action
# ---------------------------------------------------------------------------

def test_turn_delta_opp_move_known_from_last_move():
    prev = _ctx(opp_active="salamence")
    curr = _ctx(opp_active="salamence", opp_last_move_id="surf")
    delta = TurnDelta.build(prev, curr, action=6)
    assert delta.opp_move_id == "surf"
    assert delta.opp_move_known is True
    assert delta.opp_switch_to is None


def test_turn_delta_opp_move_unknown_no_last_move():
    # Opponent attacked but last_move is None (e.g. Explosion aftermath)
    prev = _ctx(opp_active="electrode")
    curr = _ctx(opp_active="electrode", opp_last_move_id=None)
    delta = TurnDelta.build(prev, curr, action=6)
    assert delta.opp_move_id is None
    assert delta.opp_move_known is False


def test_turn_delta_opp_switch_known():
    prev = _ctx(opp_active="salamence")
    curr = _ctx(opp_active="tyranitar", opp_last_move_id=None)
    delta = TurnDelta.build(prev, curr, action=6)
    assert delta.opp_switch_to == "tyranitar"
    assert delta.opp_move_id is None
    assert delta.opp_move_known is True   # switch is always known


def test_turn_delta_opp_switch_ignores_last_move():
    # If the opponent switched AND the new mon has a last_move from a prior appearance,
    # we must not use it as opp_move_id (the switch guard prevents contamination).
    prev = _ctx(opp_active="salamence")
    curr = _ctx(opp_active="gengar", opp_last_move_id="shadowball")  # gengar's prior last_move
    delta = TurnDelta.build(prev, curr, action=6)
    assert delta.opp_switch_to == "gengar"
    assert delta.opp_move_id is None        # NOT "shadowball"
    assert delta.opp_move_known is True


def test_turn_delta_opp_switch_to_none_active():
    # Edge case: opp switches but new active is "NONE" (between faints, shouldn't normally occur)
    prev = _ctx(opp_active="salamence")
    curr = _ctx(opp_active="NONE")
    delta = TurnDelta.build(prev, curr, action=6)
    assert delta.opp_switch_to is None
    assert delta.opp_move_known is True


# ---------------------------------------------------------------------------
# TurnDelta — phaze (Roar / Whirlwind)
# ---------------------------------------------------------------------------

def test_turn_delta_roar_captures_phazed_mon_move():
    # We use Roar (action slot 0, which maps to "roar"). Opponent used Surf before being
    # phazed out. opp_last_move_id is None (new active mon hasn't moved), but
    # opp_all_last_move_ids retains the phazed mon's last_move.
    prev = _ctx(
        active_move_ids=["roar", None, None, None],
        opp_active="salamence",
    )
    curr = _ctx(
        opp_active="gengar",
        opp_last_move_id=None,   # new active mon — never moved
        opp_all_last_move_ids={"salamence": "surf", "gengar": None},
    )
    delta = TurnDelta.build(prev, curr, action=6)   # slot 0 → "roar"
    assert delta.our_move_id == "roar"
    assert delta.opp_move_id == "surf"              # recovered from phazed mon
    assert delta.opp_move_known is True
    assert delta.opp_switch_to == "gengar"


def test_turn_delta_whirlwind_captures_phazed_mon_move():
    prev = _ctx(
        active_move_ids=["whirlwind", None, None, None],
        opp_active="zapdos",
    )
    curr = _ctx(
        opp_active="raikou",
        opp_last_move_id=None,
        opp_all_last_move_ids={"zapdos": "thunderbolt", "raikou": None},
    )
    delta = TurnDelta.build(prev, curr, action=6)   # slot 0 → "whirlwind"
    assert delta.our_move_id == "whirlwind"
    assert delta.opp_move_id == "thunderbolt"
    assert delta.opp_switch_to == "raikou"


def test_turn_delta_roar_opp_hadnt_moved_before_phaze():
    # Opponent was paralyzed/frozen and couldn't move before being phazed.
    # opp_all_last_move_ids has None for the phazed mon.
    prev = _ctx(
        active_move_ids=["roar", None, None, None],
        opp_active="salamence",
    )
    curr = _ctx(
        opp_active="gengar",
        opp_last_move_id=None,
        opp_all_last_move_ids={"salamence": None, "gengar": None},
    )
    delta = TurnDelta.build(prev, curr, action=6)
    assert delta.our_move_id == "roar"
    assert delta.opp_move_id is None
    assert delta.opp_move_known is False   # genuinely unknown — they may have been unable to move
    assert delta.opp_switch_to == "gengar"


def test_turn_delta_voluntary_switch_not_treated_as_phaze():
    # Opponent voluntarily switches — should NOT recover their previous last_move.
    prev = _ctx(
        active_move_ids=["surf", None, None, None],
        opp_active="salamence",
    )
    curr = _ctx(
        opp_active="gengar",
        opp_last_move_id=None,
        opp_all_last_move_ids={"salamence": "outrage", "gengar": None},
    )
    delta = TurnDelta.build(prev, curr, action=6)   # slot 0 → "surf" (not a phazing move)
    assert delta.our_move_id == "surf"
    assert delta.opp_move_id is None        # voluntary switch — move must not bleed through
    assert delta.opp_move_known is True     # switch is always known
    assert delta.opp_switch_to == "gengar"


# ---------------------------------------------------------------------------
# TurnDelta — HP and faint deltas
# ---------------------------------------------------------------------------

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


def test_turn_delta_both_fainted():
    prev = _ctx(our_fainted_count=0, opp_fainted_count=0)
    curr = _ctx(our_fainted_count=1, opp_fainted_count=1)
    delta = TurnDelta.build(prev, curr, action=6)
    assert delta.we_fainted is True
    assert delta.opp_fainted is True


def test_turn_delta_newly_revealed_opp_mon_hp_zeroed():
    # Swampert was at 42% and fainted this turn (slot 0: 0.42 → 0).
    # Tyranitar entered as replacement, revealed for the first time at 100% (slot 1: 0 → 1.0).
    # Without the fix, opp_hp_delta.sum() = -0.42 + 1.0 = +0.58 → looks like opp gained HP.
    # With the fix, slot 1 delta is zeroed → sum = -0.42, correctly reflecting damage dealt.
    prev = _ctx(
        opp_slot_map={"swampert": 0},
        opp_hp=_hp(0.42, 0.0, 0.0, 0.0, 0.0, 0.0),
        opp_fainted_count=0,
    )
    curr = _ctx(
        opp_slot_map={"swampert": 0, "tyranitar": 1},
        opp_hp=_hp(0.0, 1.0, 0.0, 0.0, 0.0, 0.0),
        opp_fainted_count=1,
    )
    delta = TurnDelta.build(prev, curr, action=6)
    assert delta.opp_hp_delta[0] == pytest.approx(-0.42)   # Swampert fainted — real delta
    assert delta.opp_hp_delta[1] == pytest.approx(0.0)     # Tyranitar revealed — zeroed
    assert delta.opp_hp_delta.sum() == pytest.approx(-0.42)


def test_turn_delta_previously_revealed_opp_hp_not_zeroed():
    # Tyranitar was already known (in prev slot_map at 80%). This turn it took damage to 50%.
    # Delta should be -0.30, NOT zeroed out.
    prev = _ctx(
        opp_slot_map={"tyranitar": 0},
        opp_hp=_hp(0.80),
    )
    curr = _ctx(
        opp_slot_map={"tyranitar": 0},
        opp_hp=_hp(0.50),
    )
    delta = TurnDelta.build(prev, curr, action=6)
    assert delta.opp_hp_delta[0] == pytest.approx(-0.30)


def test_turn_delta_newly_revealed_at_partial_hp_zeroed():
    # Opp reveals a damaged mon (e.g. sent in from bench at 60%). We don't know if
    # we caused that damage, so the delta (0 → 0.60) should be zeroed.
    prev = _ctx(opp_slot_map={}, opp_hp=np.zeros(6, dtype=np.float32))
    curr = _ctx(
        opp_slot_map={"starmie": 0},
        opp_hp=_hp(0.60),
    )
    delta = TurnDelta.build(prev, curr, action=6)
    assert delta.opp_hp_delta[0] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# TurnDelta.empty()
# ---------------------------------------------------------------------------

def test_turn_delta_empty():
    delta = TurnDelta.empty()
    assert delta.we_fainted is False
    assert delta.opp_fainted is False
    assert delta.our_hp_delta.shape == (6,)
    assert delta.our_hp_delta.sum() == 0
    assert delta.our_boost_delta.shape == (BOOST_DIM,)
    assert delta.our_boost_delta.sum() == 0
    assert delta.opp_boost_delta.shape == (BOOST_DIM,)


# ---------------------------------------------------------------------------
# BattleContext — boost fields
# ---------------------------------------------------------------------------

def test_battle_context_stores_our_boosts():
    boosts = _boosts(atk=2, spa=1)
    ctx = _ctx(our_boosts=boosts)
    assert ctx.our_boosts[BOOST_STATS.index("atk")] == 2
    assert ctx.our_boosts[BOOST_STATS.index("spa")] == 1
    assert ctx.our_boosts[BOOST_STATS.index("def")] == 0


def test_battle_context_stores_opp_boosts():
    boosts = _boosts(spe=-1)
    ctx = _ctx(opp_boosts=boosts)
    assert ctx.opp_boosts[BOOST_STATS.index("spe")] == -1


def test_battle_context_wrong_boost_shape_raises():
    with pytest.raises(RuntimeError, match="our_boosts"):
        _ctx(our_boosts=np.zeros(5, dtype=np.int8))


# ---------------------------------------------------------------------------
# TurnDelta — boost delta
# ---------------------------------------------------------------------------

def test_turn_delta_boost_delta_gain():
    # Swords Dance: our atk went from 0 to +2
    prev = _ctx(our_boosts=_boosts())
    curr = _ctx(our_boosts=_boosts(atk=2))
    delta = TurnDelta.build(prev, curr, action=6)
    assert delta.our_boost_delta[BOOST_STATS.index("atk")] == 2
    assert delta.our_boost_delta.sum() == 2


def test_turn_delta_boost_delta_loss():
    # Opponent used Charm: our atk dropped from 0 to -2
    prev = _ctx(our_boosts=_boosts())
    curr = _ctx(our_boosts=_boosts(atk=-2))
    delta = TurnDelta.build(prev, curr, action=6)
    assert delta.our_boost_delta[BOOST_STATS.index("atk")] == -2


def test_turn_delta_opp_boost_delta():
    # Opp used Calm Mind: their spa and spd both +1
    prev = _ctx(opp_boosts=_boosts())
    curr = _ctx(opp_boosts=_boosts(spa=1, spd=1))
    delta = TurnDelta.build(prev, curr, action=6)
    assert delta.opp_boost_delta[BOOST_STATS.index("spa")] == 1
    assert delta.opp_boost_delta[BOOST_STATS.index("spd")] == 1


def test_turn_delta_boost_delta_zeroed_on_our_switch():
    # We had +2 atk and switched out — boost delta should be zero (new mon, new baseline).
    prev = _ctx(
        our_active="tyranitar",
        our_boosts=_boosts(atk=2),
        our_team_order=("tyranitar", "skarmory"),
    )
    curr = _ctx(our_active="skarmory", our_boosts=_boosts())
    delta = TurnDelta.build(prev, curr, action=1)  # action < 6 = switch
    np.testing.assert_array_equal(delta.our_boost_delta, np.zeros(BOOST_DIM, dtype=np.int8))


def test_turn_delta_boost_delta_zeroed_on_opp_switch():
    # Opp had +1 spa and voluntarily switched — delta should be zero.
    prev = _ctx(opp_active="gengar", opp_boosts=_boosts(spa=1))
    curr = _ctx(opp_active="starmie", opp_boosts=_boosts())
    delta = TurnDelta.build(prev, curr, action=6)
    np.testing.assert_array_equal(delta.opp_boost_delta, np.zeros(BOOST_DIM, dtype=np.int8))


def test_turn_delta_no_change_gives_zero_delta():
    prev = _ctx(our_boosts=_boosts(atk=1))
    curr = _ctx(our_boosts=_boosts(atk=1))
    delta = TurnDelta.build(prev, curr, action=6)
    assert delta.our_boost_delta.sum() == 0
    assert delta.opp_move_known is False
    assert delta.our_move_id is None
    assert delta.opp_move_id is None


# ---------------------------------------------------------------------------
# TurnDelta — failed to move
# ---------------------------------------------------------------------------

def test_turn_delta_opp_failed_to_move_par():
    prev = _ctx(opp_active="salamence")
    curr = _ctx(opp_active="salamence", opp_cant_reason="par")
    delta = TurnDelta.build(prev, curr, action=6)
    assert delta.opp_failed_to_move is True
    assert delta.opp_cant_reason == "par"
    assert delta.opp_move_id is None   # cant + no last_move = None


def test_turn_delta_opp_failed_to_move_flinch():
    prev = _ctx(opp_active="salamence")
    curr = _ctx(opp_active="salamence", opp_cant_reason="flinch")
    delta = TurnDelta.build(prev, curr, action=6)
    assert delta.opp_failed_to_move is True
    assert delta.opp_cant_reason == "flinch"


def test_turn_delta_opp_moved_normally():
    prev = _ctx(opp_active="salamence")
    curr = _ctx(opp_active="salamence", opp_last_move_id="surf", opp_cant_reason=None)
    delta = TurnDelta.build(prev, curr, action=6)
    assert delta.opp_failed_to_move is False
    assert delta.opp_cant_reason is None
    assert delta.opp_move_id == "surf"


def test_turn_delta_our_failed_to_move_slp():
    prev = _ctx()
    curr = _ctx(our_cant_reason="slp")
    delta = TurnDelta.build(prev, curr, action=7)
    assert delta.our_failed_to_move is True
    assert delta.our_cant_reason == "slp"


def test_turn_delta_empty_no_failed_to_move():
    delta = TurnDelta.empty()
    assert delta.our_failed_to_move is False
    assert delta.opp_failed_to_move is False
    assert delta.our_cant_reason is None
    assert delta.opp_cant_reason is None


def test_turn_delta_flinch_with_persisting_last_move():
    # Opponent used earthquake last turn, then flinched this turn.
    # last_move persists as "earthquake" (|cant| doesn't clear _is_last_used).
    # With cant_reason, we can still detect the flinch.
    prev = _ctx(opp_active="salamence")
    curr = _ctx(opp_active="salamence", opp_last_move_id="earthquake", opp_cant_reason="flinch")
    delta = TurnDelta.build(prev, curr, action=6)
    assert delta.opp_failed_to_move is True
    assert delta.opp_cant_reason == "flinch"
    assert delta.opp_move_id == "earthquake"   # persisted from prior turn


# ---------------------------------------------------------------------------
# BattleContext.from_battle()
# ---------------------------------------------------------------------------

def _mock_mon(species, hp_fraction=1.0, fainted=False):
    mon = MagicMock()
    mon.species = species
    mon.current_hp_fraction = hp_fraction
    mon.fainted = fainted
    mon.last_move = None
    mon.last_cant_reason = None
    mon.moves = {}
    mon.boosts = {"atk": 0, "def": 0, "spa": 0, "spd": 0, "spe": 0, "accuracy": 0, "evasion": 0}
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
    battle.last_request = {}

    mask = np.zeros(11, dtype=np.int8)
    mask[6] = 1

    our_slots = SlotRegistry()
    opp_slots = SlotRegistry()
    ctx = BattleContext.from_battle(battle, mask, our_slots, opp_slots)

    assert ctx.turn == 7
    assert ctx.phase == "move_selection"
    assert ctx.our_active == "tyranitar"
    assert ctx.opp_active == "zapdos"
    assert ctx.our_hp[our_slots.get("tyranitar")] == pytest.approx(0.8)
    assert ctx.opp_hp[opp_slots.get("zapdos")] == pytest.approx(0.5)
    assert ctx.our_fainted_count == 0
    assert ctx.opp_fainted_count == 0
    assert ctx.opp_last_move_id is None
    assert ctx.our_cant_reason is None
    assert ctx.opp_cant_reason is None


def test_from_battle_forced_switch_phase():
    battle = MagicMock()
    battle.turn = 3
    battle.force_switch = True
    battle.team = {}
    battle.opponent_team = {}
    battle.active_pokemon = None
    battle.opponent_active_pokemon = None

    ctx = BattleContext.from_battle(
        battle, np.ones(11, dtype=np.int8), SlotRegistry(), SlotRegistry()
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
        battle, np.ones(11, dtype=np.int8), our_slots, SlotRegistry()
    )
    assert our_slots.get("skarmory") == 0


# ---------------------------------------------------------------------------
# active_move_ids field (via from_battle)
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


def test_active_move_ids_populated_from_last_request():
    """active_move_ids mirrors the masker's slot assignment for normal moves."""
    battle = _mock_battle_with_moves(
        ["rockslide", "earthquake", "thunderbolt", "surf"]
    )
    mask = _mask(6, 7, 8, 9)
    ctx = BattleContext.from_battle(battle, mask, SlotRegistry(), SlotRegistry())
    assert ctx.active_move_ids == ["rockslide", "earthquake", "thunderbolt", "surf"]


def test_active_move_ids_disabled_slot_is_none():
    """Disabled move slots are set to None regardless of mask."""
    battle = _mock_battle_with_moves(
        ["rockslide", "earthquake", "thunderbolt", "surf"],
        disabled={1},  # earthquake is disabled
    )
    mask = _mask(6, 8, 9)  # slot 7 (earthquake) disabled
    ctx = BattleContext.from_battle(battle, mask, SlotRegistry(), SlotRegistry())
    assert ctx.active_move_ids[0] == "rockslide"
    assert ctx.active_move_ids[1] is None   # disabled
    assert ctx.active_move_ids[2] == "thunderbolt"
    assert ctx.active_move_ids[3] == "surf"


def test_active_move_ids_struggle_slot_is_none():
    """Struggle is excluded from active_move_ids (it uses the dedicated action 10)."""
    battle = _mock_battle_with_moves(["struggle", "struggle", "struggle", "struggle"])
    mask = _mask(10)
    ctx = BattleContext.from_battle(battle, mask, SlotRegistry(), SlotRegistry())
    assert ctx.active_move_ids == [None, None, None, None]


def test_active_move_ids_always_length_4():
    """active_move_ids is always a 4-element list, even with fewer than 4 moves."""
    battle = _mock_battle_with_moves(["rockslide", "earthquake"])
    mask = _mask(6, 7)
    ctx = BattleContext.from_battle(battle, mask, SlotRegistry(), SlotRegistry())
    assert len(ctx.active_move_ids) == 4
    assert ctx.active_move_ids[0] == "rockslide"
    assert ctx.active_move_ids[1] == "earthquake"
    assert ctx.active_move_ids[2] is None
    assert ctx.active_move_ids[3] is None


def test_active_move_ids_no_request_is_all_none():
    """Forced-switch phase (no active request) yields all-None active_move_ids."""
    battle = MagicMock()
    battle.turn = 3
    battle.force_switch = True
    battle.team = {}
    battle.opponent_team = {}
    battle.active_pokemon = None
    battle.opponent_active_pokemon = None
    battle.last_request = {}

    ctx = BattleContext.from_battle(
        battle, np.ones(11, dtype=np.int8), SlotRegistry(), SlotRegistry()
    )
    assert ctx.active_move_ids == [None, None, None, None]
