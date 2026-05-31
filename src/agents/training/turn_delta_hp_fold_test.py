"""Unit coverage for the event-fold of the per-slot HP delta, target-HP attribution, and
the (snapshot-sourced) boost delta in ``TurnDelta.build_from_events`` — the Phase-5 history
migration. Complements ``turn_delta_event_fold_test.py`` (which covers attempted-action /
faint-cause / status / empty-window) with the numeric-quantity corners:

  * HP delta folded per-slot from DAMAGE/HEAL ``hp_after`` — bit-identical to ``curr−prev``.
  * Pain Split (SETHP) folded as the post-value.
  * Self-KO (Explosion/Selfdestruct): faints the user with NO ``-damage`` line, so HP→0
    comes from the FAINT event (the key event-log-incompleteness finding).
  * Multi-hit: several DAMAGE events on one mon → the LAST ``hp_after`` is the end HP.
  * Double-KO: both sides faint in one window.
  * Newly-revealed-opponent HP zeroing (parity with the old snapshot path).
  * Target-HP attributed from the damaging event's named target.
  * Boost delta from the LiveView snapshot (zeroed on switch) and its Haze (CLEARBOOST) /
    Belly-Drum (SETBOOST) cases — folded from the snapshot because the event log carries no
    realized stage amount for those ops.
"""
from __future__ import annotations
import itertools
import numpy as np

from agents.battle.battle_event import OPP, OURS, BattleEvent, EventKind
from agents.gen3_mechanics import BOOST_DIM
from agents.training.battle_snapshot import BattleContext
from agents.training.turn_delta import TurnDelta

_seq = itertools.count()


def ev(kind, *, side=None, actor=None, target=None, turn=2, **value):
    return BattleEvent(seq=next(_seq), turn=turn, kind=kind, side=side,
                       actor_species=actor, target_species=target, value=value)


def _arr(slots=None) -> np.ndarray:
    """6-slot float32 HP array from a {slot_index: fraction} dict."""
    a = np.zeros(6, dtype=np.float32)
    for s, v in (slots or {}).items():
        a[s] = v
    return a


def _ctx(*, turn=1, phase="move_selection", our_active="zapdos", opp_active="tyranitar",
         our_slot_map=None, opp_slot_map=None, our_hp=None, opp_hp=None,
         our_boosts=None, opp_boosts=None, our_fainted=0, opp_fainted=0,
         active_move_ids=None) -> BattleContext:
    return BattleContext(
        turn=turn, phase=phase,
        mask=np.ones(11, dtype=np.int8),
        our_slot_map=our_slot_map if our_slot_map is not None else {"zapdos": 0},
        opp_slot_map=opp_slot_map if opp_slot_map is not None else {"tyranitar": 0},
        our_hp=our_hp if our_hp is not None else np.zeros(6, dtype=np.float32),
        opp_hp=opp_hp if opp_hp is not None else np.zeros(6, dtype=np.float32),
        our_active=our_active, opp_active=opp_active,
        our_fainted_count=our_fainted, opp_fainted_count=opp_fainted,
        active_move_ids=active_move_ids or ["thunderbolt", "icebeam", "rest", "sleeptalk"],
        opp_last_move_id=None, opp_all_last_move_ids={}, opp_active_revealed_moves=frozenset(),
        our_cant_reason=None, opp_cant_reason=None,
        our_boosts=our_boosts if our_boosts is not None else np.zeros(BOOST_DIM, dtype=np.int8),
        opp_boosts=opp_boosts if opp_boosts is not None else np.zeros(BOOST_DIM, dtype=np.int8),
        our_last_effectiveness=None, opp_last_effectiveness=None, we_moved_first=None,
        our_team_order=("zapdos", "skarmory"),
    )


def _build(prev, curr, action, events):
    return TurnDelta.build_from_events(prev, curr, action, events)


# ---------------------------------------------------------------------------
# Per-slot HP delta — folded from DAMAGE/HEAL hp_after
# ---------------------------------------------------------------------------

def test_hp_delta_folded_per_slot_from_damage():
    prev = _ctx(opp_hp=_arr({0: 1.0}))
    curr = _ctx(turn=2, opp_hp=_arr({0: 0.6}))
    events = [
        ev(EventKind.MOVE, side=OURS, actor="zapdos", target="tyranitar", move_id="thunderbolt"),
        ev(EventKind.DAMAGE, side=OPP, actor="tyranitar", amount=-0.4, hp_before=1.0, hp_after=0.6),
    ]
    d = _build(prev, curr, 6, events)
    assert np.isclose(d.opp_hp_delta[0], -0.4)
    assert np.allclose(d.our_hp_delta, 0.0)


def test_hp_delta_bit_identical_to_snapshot_diff():
    """The event-fold HP delta must be BIT-identical to ``curr_hp − prev_hp`` (the whole
    point — event-sourced AND value-identical, no float-sum noise)."""
    prev = _ctx(our_hp=_arr({0: 1.0, 1: 0.5}), opp_hp=_arr({0: 1.0}))
    curr = _ctx(turn=2, our_hp=_arr({0: 0.8125, 1: 0.5}), opp_hp=_arr({0: 0.25}))
    events = [
        ev(EventKind.MOVE, side=OPP, actor="tyranitar", target="zapdos", move_id="crunch"),
        ev(EventKind.DAMAGE, side=OURS, actor="zapdos", amount=-0.1875, hp_before=1.0, hp_after=0.8125),
        ev(EventKind.MOVE, side=OURS, actor="zapdos", target="tyranitar", move_id="thunderbolt"),
        ev(EventKind.DAMAGE, side=OPP, actor="tyranitar", amount=-0.75, hp_before=1.0, hp_after=0.25),
    ]
    d = _build(prev, curr, 6, events)
    assert np.array_equal(d.our_hp_delta, curr.our_hp - prev.our_hp)
    assert np.array_equal(d.opp_hp_delta, curr.opp_hp - prev.opp_hp)


def test_heal_folds_positive_delta():
    prev = _ctx(our_hp=_arr({0: 0.5}))
    curr = _ctx(turn=2, our_hp=_arr({0: 1.0}))
    events = [
        ev(EventKind.MOVE, side=OURS, actor="zapdos", move_id="rest"),
        ev(EventKind.HEAL, side=OURS, actor="zapdos", amount=0.5, hp_before=0.5, hp_after=1.0),
    ]
    d = _build(prev, curr, 8, events)
    assert np.isclose(d.our_hp_delta[0], 0.5)


# ---------------------------------------------------------------------------
# Pain Split (SETHP) folded as the post-value
# ---------------------------------------------------------------------------

def test_pain_split_sethp_folded_both_sides():
    # Our 1.0 + opp 0.3 → both averaged to 0.65 via Pain Split.
    prev = _ctx(our_hp=_arr({0: 1.0}), opp_hp=_arr({0: 0.3}))
    curr = _ctx(turn=2, our_hp=_arr({0: 0.65}), opp_hp=_arr({0: 0.65}))
    events = [
        ev(EventKind.MOVE, side=OURS, actor="zapdos", target="tyranitar", move_id="painsplit"),
        ev(EventKind.SETHP, side=OURS, actor="zapdos", hp=0.65, hp_before=1.0, amount=-0.35),
        ev(EventKind.SETHP, side=OPP, actor="tyranitar", hp=0.65, hp_before=0.3, amount=0.35),
    ]
    d = _build(prev, curr, 6, events)
    assert np.isclose(d.our_hp_delta[0], -0.35)
    assert np.isclose(d.opp_hp_delta[0], 0.35)


# ---------------------------------------------------------------------------
# Self-KO — the event-log-incompleteness finding
# ---------------------------------------------------------------------------

def test_self_ko_explosion_hp_from_faint_no_damage_line():
    """Explosion faints the USER with no -damage line on it — only |faint|. The fold must
    still report the user's HP→0 (delta = −prev_hp), sourced from the FAINT event."""
    prev = _ctx(our_hp=_arr({0: 1.0}), opp_hp=_arr({0: 0.9}))
    curr = _ctx(turn=2, our_hp=_arr({0: 1.0}), opp_hp=_arr({0: 0.0}), opp_fainted=1)
    events = [
        ev(EventKind.MOVE, side=OPP, actor="tyranitar", target="zapdos", move_id="explosion"),
        # NOTE: deliberately NO DAMAGE event on tyranitar (self-KO emits none).
        ev(EventKind.DAMAGE, side=OURS, actor="zapdos", amount=0.0, hp_before=1.0, hp_after=1.0),
        ev(EventKind.FAINT, side=OPP, actor="tyranitar"),
    ]
    d = _build(prev, curr, 6, events)
    assert np.isclose(d.opp_hp_delta[0], -0.9), "self-KO user HP→0 must come from the FAINT"
    assert d.opp_fainted and d.opp_faint_count == 1


def test_damage_faint_hp_reaches_zero():
    prev = _ctx(opp_hp=_arr({0: 0.2}))
    curr = _ctx(turn=2, opp_hp=_arr({0: 0.0}), opp_fainted=1)
    events = [
        ev(EventKind.MOVE, side=OURS, actor="zapdos", target="tyranitar", move_id="thunderbolt"),
        ev(EventKind.DAMAGE, side=OPP, actor="tyranitar", amount=-0.2, hp_before=0.2, hp_after=0.0),
        ev(EventKind.FAINT, side=OPP, actor="tyranitar"),
    ]
    d = _build(prev, curr, 6, events)
    assert np.isclose(d.opp_hp_delta[0], -0.2)


# ---------------------------------------------------------------------------
# Multi-hit — last hp_after is the end HP
# ---------------------------------------------------------------------------

def test_multi_hit_last_hp_after_is_end_hp():
    """Two DAMAGE events on one mon (multi-hit / attack + residual): the END HP is the LAST
    event's hp_after, not the sum of amounts — exact, no accumulation noise."""
    prev = _ctx(opp_hp=_arr({0: 1.0}))
    curr = _ctx(turn=2, opp_hp=_arr({0: 0.55}))
    events = [
        ev(EventKind.MOVE, side=OURS, actor="zapdos", target="tyranitar", move_id="bonemerang"),
        ev(EventKind.DAMAGE, side=OPP, actor="tyranitar", amount=-0.25, hp_before=1.0, hp_after=0.75),
        ev(EventKind.DAMAGE, side=OPP, actor="tyranitar", amount=-0.20, hp_before=0.75, hp_after=0.55),
    ]
    d = _build(prev, curr, 6, events)
    assert np.isclose(d.opp_hp_delta[0], -0.45)
    assert np.array_equal(d.opp_hp_delta, curr.opp_hp - prev.opp_hp)


# ---------------------------------------------------------------------------
# Double-KO
# ---------------------------------------------------------------------------

def test_double_ko_both_sides_faint():
    prev = _ctx(our_hp=_arr({0: 0.4}), opp_hp=_arr({0: 0.5}))
    curr = _ctx(turn=2, our_hp=_arr({0: 0.0}), opp_hp=_arr({0: 0.0}),
                our_fainted=1, opp_fainted=1)
    events = [
        ev(EventKind.MOVE, side=OURS, actor="zapdos", target="tyranitar", move_id="explosion"),
        ev(EventKind.DAMAGE, side=OPP, actor="tyranitar", amount=-0.5, hp_before=0.5, hp_after=0.0),
        ev(EventKind.FAINT, side=OPP, actor="tyranitar"),
        ev(EventKind.FAINT, side=OURS, actor="zapdos"),
    ]
    d = _build(prev, curr, 6, events)
    assert d.we_fainted and d.opp_fainted
    assert d.our_faint_count == 1 and d.opp_faint_count == 1
    assert np.isclose(d.our_hp_delta[0], -0.4)   # our zapdos: self-KO HP via FAINT
    assert np.isclose(d.opp_hp_delta[0], -0.5)


# ---------------------------------------------------------------------------
# Newly-revealed-opponent HP zeroing
# ---------------------------------------------------------------------------

def test_newly_revealed_opp_hp_zeroed_even_with_damage():
    """A mon first revealed THIS window (a fresh switch-in taking entry-hazard chip) has its
    HP delta zeroed — parity with the old snapshot path (prev_hp was the 0 sentinel)."""
    prev = _ctx(opp_slot_map={"tyranitar": 0}, opp_hp=_arr({0: 1.0}))
    curr = _ctx(turn=2, opp_active="skarmory",
                opp_slot_map={"tyranitar": 0, "skarmory": 1},
                opp_hp=_arr({0: 1.0, 1: 0.88}))
    events = [
        ev(EventKind.SWITCH, side=OPP, actor="skarmory", prev_active="tyranitar"),
        ev(EventKind.DAMAGE, side=OPP, actor="skarmory", amount=-0.12, hp_before=1.0, hp_after=0.88),
    ]
    d = _build(prev, curr, 6, events)
    assert d.opp_hp_delta[1] == 0.0, "freshly-revealed opp slot must be zeroed"
    assert d.opp_hp_delta[0] == 0.0


# ---------------------------------------------------------------------------
# Target-HP attribution
# ---------------------------------------------------------------------------

def test_target_hp_delta_from_damaging_event_target():
    prev = _ctx(opp_hp=_arr({0: 1.0}))
    curr = _ctx(turn=2, opp_hp=_arr({0: 0.4}))
    events = [
        ev(EventKind.MOVE, side=OURS, actor="zapdos", target="tyranitar", move_id="thunderbolt"),
        ev(EventKind.SUPEREFFECTIVE, side=OURS, actor="zapdos", target="tyranitar", multiplier=2.0),
        ev(EventKind.DAMAGE, side=OPP, actor="tyranitar", amount=-0.6, hp_before=1.0, hp_after=0.4),
    ]
    d = _build(prev, curr, 6, events)
    assert d.our_damaging_event is not None
    assert d.our_damaging_event.target_species == "tyranitar"
    # opp_target_hp_delta = HP loss on the mon OUR move hit (the opp's tyranitar).
    assert np.isclose(d.opp_target_hp_delta, -0.6)


# ---------------------------------------------------------------------------
# Boost delta — snapshot-sourced (event log carries no realized stage amount)
# ---------------------------------------------------------------------------

def _b(**stages) -> np.ndarray:
    order = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
    return np.array([stages.get(s, 0) for s in order], dtype=np.int8)


def test_boost_delta_from_snapshot_when_stayed_in():
    prev = _ctx(our_boosts=_b(atk=0, spe=0))
    curr = _ctx(turn=2, our_boosts=_b(atk=1, spe=1))
    events = [ev(EventKind.MOVE, side=OURS, actor="zapdos", move_id="dragondance"),
              ev(EventKind.BOOST, side=OURS, actor="zapdos", stat="atk", amount=1),
              ev(EventKind.BOOST, side=OURS, actor="zapdos", stat="spe", amount=1)]
    d = _build(prev, curr, 6, events)
    assert np.array_equal(d.our_boost_delta, curr.our_boosts - prev.our_boosts)
    assert d.our_boost_delta[0] == 1 and d.our_boost_delta[4] == 1


def test_clearboost_haze_reflected_via_snapshot():
    """Haze (CLEARBOOST) carries NO realized stage amount in the event — only ``op``. The
    boost delta is the snapshot stage diff, which captures the wipe (+2 → 0 ⇒ −2)."""
    prev = _ctx(opp_boosts=_b(atk=2, spa=2))
    curr = _ctx(turn=2, opp_boosts=_b())  # Haze zeroed them
    events = [
        ev(EventKind.MOVE, side=OURS, actor="zapdos", move_id="haze"),
        ev(EventKind.CLEARBOOST, side=OPP, actor="tyranitar", op="clearallboost"),
    ]
    d = _build(prev, curr, 6, events)
    assert np.array_equal(d.opp_boost_delta, curr.opp_boosts - prev.opp_boosts)
    assert d.opp_boost_delta[0] == -2 and d.opp_boost_delta[2] == -2


def test_setboost_belly_drum_reflected_via_snapshot():
    """Belly Drum (SETBOOST) sets Attack to +6; the event's amount is the SET target, not the
    realized stage change. The snapshot diff is the lossless source (here 0 → +6)."""
    prev = _ctx(our_boosts=_b(atk=0))
    curr = _ctx(turn=2, our_boosts=_b(atk=6))
    events = [
        ev(EventKind.MOVE, side=OURS, actor="zapdos", move_id="bellydrum"),
        ev(EventKind.SETBOOST, side=OURS, actor="zapdos", stat="atk", amount=6),
    ]
    d = _build(prev, curr, 6, events)
    assert d.our_boost_delta[0] == 6


def test_boost_delta_zeroed_on_switch():
    prev = _ctx(our_boosts=_b(atk=3))
    curr = _ctx(turn=2, our_active="skarmory", our_boosts=_b())  # switched in, own baseline
    events = [ev(EventKind.SWITCH, side=OURS, actor="skarmory", prev_active="zapdos")]
    d = _build(prev, curr, 0, events)
    assert d.our_switch_to == "skarmory"
    assert np.all(d.our_boost_delta == 0), "switch-in's stages are its own baseline"
