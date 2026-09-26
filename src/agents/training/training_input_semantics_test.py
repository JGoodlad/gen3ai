"""The TRAINING-INPUT semantics fixes of the Rust core M3 loss catalogue, pinned on hand-built event
logs (`designs/research_state/measurements/rust_core_m3_2026-09-24/` §5; the fix record is
`designs/research_state/measurements/training_input_gigo_fixes_2026-09-24/`).

Each test names the catalogue id it pins and FAILS if its fix is reverted. The same cases on REAL
constructed battles — the Python path and the Rust core together, through slice T — are
`agents/battle/tracker_semantics_fixtures_test.py`; the Rust core's own pins are
`src/rust_sim/tests/tracker_semantics_test.rs`.

* `gen3_event_window_semantics_fixes_v1` — W1 (a stat DROP is a negative BOOST row), W2 (a faint no
  damage line caused is not an `attack`), W3 (a transfer move's `|-item|` is SWAPPED), W4 (a Protect
  block FAILS the blocked move), W5 (a side condition's end is a −1 HAZARD row), and T1's class on
  the row (a bare `-damage` attaches to the open move only while its user is moving).
* `gen3_intent_label_semantics_fixes_v1` — L1 / L2 (drag), L3 (straddling replacement), L4 (the
  caller), L5 (Encore override).
* `gen3_progress_clock_attribution_fix_v1` — T1 (clause (i) needs our own hit), T2 (a blocked attack
  is an exogenous freeze).
"""
from __future__ import annotations

import itertools
from types import SimpleNamespace

import numpy as np
import pytest

from agents.battle.battle_event import OPP, OURS, BattleEvent, EventKind
from agents.battle.turn_view import TurnView
from agents.observation.assembler import write_event_row
from agents.observation.constants import (
    EVENT_T_BOOST, EVENT_T_FAINT, EVENT_T_HAZARD, EVENT_T_ITEM_REVEAL, EVENT_T_MOVE,
    EVENT_TOKEN_DIM, ITEM_TR_REVEALED, ITEM_TR_RECEIVED, EventCol as C,
)
from agents.training.episode_tracker import EventWindowTracker
from agents.training.opp_intent_labels import KIND_MOVE, KIND_SWITCH, KIND_UNKNOWN, build_opp_intent_label
from agents.training.progress_clock import ProgressClock

_seq = itertools.count(1)


def ev(kind, side=None, actor=None, turn=1, target=None, **value):
    return BattleEvent(seq=next(_seq), turn=turn, kind=kind, side=side, actor_species=actor,
                       target_species=target, value=value)


def _leads():
    return [ev(EventKind.SWITCH, OURS, "snorlax"), ev(EventKind.SWITCH, OPP, "skarmory")]


def _window(events):
    """Fold the two leads then ``events``, re-sequenced in list order (the tracker is
    seq-idempotent, so an out-of-order seq would be dropped as a replay)."""
    import dataclasses
    evs = [dataclasses.replace(e, seq=i) for i, e in enumerate(_leads() + events)]
    t = EventWindowTracker()
    t.update(1, evs, None, None)
    return t.window()[2:]   # drop the two lead rows


# ============================================================================ the event window

def test_w1_a_stat_drop_is_a_negative_boost_row():
    """The event builder already signs an `|-unboost|` (amount −1); the window negated it again."""
    rows = _window([ev(EventKind.UNBOOST, OURS, "snorlax", stat="atk", amount=-1),
                    ev(EventKind.BOOST, OURS, "snorlax", stat="def", amount=1)])
    assert [(r["t"], r["hp_delta"]) for r in rows] == [(EVENT_T_BOOST, -1.0), (EVENT_T_BOOST, 1.0)]
    vec = np.zeros(EVENT_TOKEN_DIM, dtype=np.float32)
    write_event_row(vec, 0, rows[0], 1)
    assert vec[C.MAGNITUDE] == pytest.approx(-1.0 / 6.0)


def test_w2_a_faint_no_damage_line_caused_is_other_not_attack():
    """Never `attack`. Since gen3_event_record_v2 (E12) the two named cases have their OWN causes in
    the live vocabulary (`FAINT_CAUSE_VOCAB_LIVE`): `destinybond` and `perishsong`."""
    # Destiny Bond: the attacker faints with NO damage line at all.
    rows = _window([ev(EventKind.MOVE, OPP, "skarmory", move_id="drillpeck"),
                    ev(EventKind.DAMAGE, OURS, "snorlax", amount=-1.0, hp_after=0.0),
                    ev(EventKind.FAINT, OURS, "snorlax"),
                    ev(EventKind.ACTIVATE, OURS, "snorlax", effect="move: Destiny Bond"),
                    ev(EventKind.FAINT, OPP, "skarmory")])
    faints = [(r["actor"], r["faint_cause"]) for r in rows if r["t"] == EVENT_T_FAINT]
    assert faints == [("snorlax", "attack"), ("skarmory", "destinybond")]
    # Perish Song: the mon took earlier NON-lethal chip, then its count hit 0.
    rows = _window([ev(EventKind.MOVE, OURS, "snorlax", move_id="bodyslam"),
                    ev(EventKind.DAMAGE, OPP, "skarmory", amount=-0.3, hp_after=0.7),
                    ev(EventKind.VOLATILE_START, OPP, "skarmory", turn=2, effect="perish0"),
                    ev(EventKind.FAINT, OPP, "skarmory", turn=2)])
    assert [r["faint_cause"] for r in rows if r["t"] == EVENT_T_FAINT] == ["perishsong"]
    # a faint with no lethal line and neither announcement stays the honest catch-all
    rows = _window([ev(EventKind.MOVE, OURS, "snorlax", move_id="bodyslam"),
                    ev(EventKind.DAMAGE, OPP, "skarmory", amount=-0.3, hp_after=0.7),
                    ev(EventKind.FAINT, OPP, "skarmory", turn=2)])
    assert [r["faint_cause"] for r in rows if r["t"] == EVENT_T_FAINT] == ["other"]


def test_r4_a_self_move_row_targets_its_user():
    """gen3_move_target_class_v1: the window's MOVE row target is the move's dex target class — a
    self / side / field move (a failed Refresh, a Snatch-stolen one, Recover, Rain Dance) is its
    USER's row. The row used to carry the OTHER side's active for every move (the 2026-09-25
    cutover-stress evidence: Swampert's failed Refresh rows read `blissey` / `forretress`)."""
    rows = _window([ev(EventKind.MOVE, OPP, "skarmory", move_id="refresh"),
                    ev(EventKind.MOVE, OURS, "snorlax", move_id="refresh", from_move="snatch"),
                    ev(EventKind.MOVE, OURS, "snorlax", move_id="raindance", turn=2),
                    ev(EventKind.MOVE, OPP, "skarmory", move_id="drillpeck", turn=2),
                    ev(EventKind.MOVE, OURS, "snorlax", move_id="curse", turn=3),
                    ev(EventKind.MOVE, OPP, "skarmory", move_id="spikes", turn=3)])
    assert [(r["actor"], r["target"]) for r in rows] == [
        ("skarmory", "skarmory"), ("snorlax", "snorlax"), ("snorlax", "snorlax"),
        ("skarmory", "snorlax"), ("snorlax", "snorlax"), ("skarmory", "snorlax")]
    # ... and the obs column carries the USER's dex num.
    from agents import gen3_data
    vec = np.zeros(EVENT_TOKEN_DIM, dtype=np.float32)
    write_event_row(vec, 0, rows[0], 1)
    assert vec[C.TARGET_SPECIES] == float(gen3_data.species.get("skarmory").num)


def test_w3_a_trick_or_thief_item_line_is_a_transfer_on_both_mons():
    """A transfer, never a plain reveal. gen3_event_record_v2 (E12) gives it a DIRECTION: an
    `|-item|` [from] Trick / Thief / Covet is this mon RECEIVING the item."""
    rows = _window([ev(EventKind.ITEM, OPP, "skarmory", item="choiceband", **{"from": "move: Trick"}),
                    ev(EventKind.ITEM, OURS, "snorlax", item="leftovers", **{"from": "move: Trick"}),
                    ev(EventKind.ITEM, OURS, "snorlax", item="leftovers", **{"from": "move: Thief"}),
                    ev(EventKind.ITEM, OPP, "skarmory", item="leftovers")])
    assert [r["item_tr"] for r in rows if r["t"] == EVENT_T_ITEM_REVEAL] == \
        [ITEM_TR_RECEIVED, ITEM_TR_RECEIVED, ITEM_TR_RECEIVED, ITEM_TR_REVEALED]


def test_w4_a_protect_block_fails_the_blocked_move():
    rows = _window([ev(EventKind.MOVE, OPP, "skarmory", move_id="protect"),
                    ev(EventKind.MOVE, OURS, "snorlax", move_id="bodyslam"),
                    ev(EventKind.ACTIVATE, OPP, "skarmory", effect="move: Protect")])
    slam = next(r for r in rows if r.get("move_id") == "bodyslam")
    prot = next(r for r in rows if r.get("move_id") == "protect")
    assert slam["failed"] is True and prot["failed"] is False
    vec = np.zeros(EVENT_TOKEN_DIM, dtype=np.float32)
    write_event_row(vec, 0, slam, 1)
    assert (vec[C.OUT_HIT], vec[C.OUT_FAIL]) == (0.0, 1.0)
    # Endure's -activate is NOT a block: the move still hit.
    rows = _window([ev(EventKind.MOVE, OURS, "snorlax", move_id="bodyslam"),
                    ev(EventKind.ACTIVATE, OPP, "skarmory", effect="move: Endure")])
    assert rows[0]["failed"] is False


def test_w5_a_side_condition_end_is_a_negative_hazard_row():
    rows = _window([ev(EventKind.SIDE, OURS, condition="Spikes", op="sidestart"),
                    ev(EventKind.SIDE, OURS, condition="Spikes", op="sideend", **{"from": "move: Rapid Spin"})])
    assert [(r["t"], r["hp_delta"]) for r in rows] == [(EVENT_T_HAZARD, 1.0), (EVENT_T_HAZARD, -1.0)]
    vec = np.zeros(EVENT_TOKEN_DIM, dtype=np.float32)
    write_event_row(vec, 0, rows[1], 1)
    assert vec[C.MAGNITUDE] == -1.0          # a HAZARD row's magnitude is NOT stage-scaled


def test_t1_row_a_bare_damage_attaches_only_while_its_move_user_is_moving():
    """The opponent's OWN Substitute cost prints a bare `-damage` on it while IT is moving — that is
    not our Body Slam's hit, which stays at its own −0.2."""
    rows = _window([ev(EventKind.MOVE, OURS, "snorlax", move_id="bodyslam"),
                    ev(EventKind.DAMAGE, OPP, "skarmory", amount=-0.2, hp_after=0.8),
                    ev(EventKind.MOVE, OPP, "skarmory", move_id="substitute"),
                    ev(EventKind.VOLATILE_START, OPP, "skarmory", effect="Substitute"),
                    ev(EventKind.DAMAGE, OPP, "skarmory", amount=-0.25, hp_after=0.55)])
    slam = next(r for r in rows if r["t"] == EVENT_T_MOVE and r["move_id"] == "bodyslam")
    assert slam["hp_delta"] == pytest.approx(-0.2)


# ============================================================================ TurnView

def test_turnview_a_blocked_move_is_outcome_fail_and_its_hit_is_its_own():
    tv = TurnView.from_events([
        ev(EventKind.MOVE, OPP, "skarmory", move_id="protect"),
        ev(EventKind.MOVE, OURS, "snorlax", move_id="bodyslam", target="skarmory"),
        ev(EventKind.ACTIVATE, OPP, "skarmory", effect="move: Protect"),
        ev(EventKind.DAMAGE, OPP, "skarmory", amount=-0.06, hp_after=0.94, reason="Sandstorm"),
    ])
    assert tv.ours.outcome == "fail" and tv.opp.outcome == "hit"
    assert tv.ours.hit_dealt == 0.0          # the sand chip is not our hit


def test_turnview_an_encore_before_the_target_moves_overrides_its_choice():
    tv = TurnView.from_events([
        ev(EventKind.MOVE, OURS, "gengar", move_id="encore"),
        ev(EventKind.VOLATILE_START, OPP, "snorlax", effect="Encore"),
        ev(EventKind.MOVE, OPP, "snorlax", move_id="curse"),
    ])
    assert tv.opp.choice_overridden is True and tv.ours.choice_overridden is False
    # an Encore AFTER the target already moved overrides nothing
    tv = TurnView.from_events([
        ev(EventKind.MOVE, OPP, "snorlax", move_id="curse"),
        ev(EventKind.MOVE, OURS, "gengar", move_id="encore"),
        ev(EventKind.VOLATILE_START, OPP, "snorlax", effect="Encore"),
    ])
    assert tv.opp.choice_overridden is False


# ============================================================================ the α/β label

def _d(**kw):
    base = dict(phase_is_forced_switch=False, opp_resolved_move_id=None, opp_switch_to=None,
                opp_fainted=False, opp_dragged=False, opp_switch_is_replacement=False,
                opp_called_via=None, opp_choice_overridden=False)
    base.update(kw)
    return SimpleNamespace(**base)


_MOVES = {"sleeptalk": 11, "rest": 12, "curse": 13, "rockslide": 14}


def _label(d):
    return build_opp_intent_label(d, _MOVES.get, lambda s: 3, lambda s: 99)


def test_l1_l2_a_dragged_entrant_is_masked():
    assert _label(_d(opp_switch_to="starmie"))[0] == KIND_SWITCH              # a real pivot
    assert _label(_d(opp_switch_to="starmie", opp_dragged=True))[0] == KIND_UNKNOWN


def test_l3_a_replacement_in_the_window_after_its_faint_is_masked():
    assert _label(_d(opp_switch_to="snorlax", opp_switch_is_replacement=True))[0] == KIND_UNKNOWN


def test_l4_a_called_move_is_labelled_as_its_caller():
    assert _label(_d(opp_resolved_move_id="rest", opp_called_via="sleeptalk"))[:2] == (KIND_MOVE, 11)
    assert _label(_d(opp_resolved_move_id="rest"))[:2] == (KIND_MOVE, 12)


def test_l5_an_encore_override_is_masked():
    assert _label(_d(opp_resolved_move_id="curse", opp_choice_overridden=True))[0] == KIND_UNKNOWN


# ============================================================================ the progress clock

def _clock_delta(**kw):
    base = dict(our_move_id="taunt", our_switch_to=None, our_prev_active="tyranitar",
                our_damaging_event=None, opp_target_hp_delta=None, our_move_hit_delta=0.0,
                opp_status_applied=None, opp_switch_to=None, our_failed_to_move=False,
                our_move_outcome="hit", our_status_applied=None, our_status_cured=None,
                opp_resolved_move_id=None, opp_fainted=False, phase_is_forced_switch=False,
                decision_was_forced_switch=False,
                our_hp_delta=np.zeros(6, dtype=np.float32), opp_hp_delta=np.zeros(6, dtype=np.float32))
    base.update(kw)
    return SimpleNamespace(**base)


def _live():
    mon = SimpleNamespace(boosts={}, volatiles=set(), move_ids=[], species="x", status=None, fainted=False)
    side = SimpleNamespace(active=mon, side_conditions={})
    return SimpleNamespace(ours=side, opp=side)


def _n(d):
    c = ProgressClock(0.15)
    c.update(d, _live(), SimpleNamespace(switches=[1]))
    return c.n


def test_t1_the_targets_net_fall_without_our_hit_is_not_progress():
    """A Taunt in Sandstorm: the target's net HP fell 6 %, our move dealt nothing."""
    ev_ = SimpleNamespace(move_id="taunt", target_species="snorlax")
    assert _n(_clock_delta(our_damaging_event=ev_, opp_target_hp_delta=-0.06)) == 1
    assert _n(_clock_delta(our_damaging_event=ev_, opp_target_hp_delta=-0.06, our_move_hit_delta=-0.06)) == 0


def test_t2_a_blocked_attack_freezes_the_clock():
    """Our Body Slam into their Protect: outcome "fail" with their Protect is the exogenous-denial
    branch — frozen, not charged. The branch always existed; what was broken is that a block never
    READ "fail", so the REVERT pin of T2 is ``test_turnview_a_blocked_move_is_outcome_fail…`` above
    (and the constructed battle in ``tracker_semantics_fixtures_test.py``). This one pins the
    contract the TurnView fix feeds."""
    assert _n(_clock_delta(our_move_id="bodyslam", our_move_outcome="fail", opp_resolved_move_id="protect")) == 0
    assert _n(_clock_delta(our_move_id="bodyslam", our_move_outcome="hit", opp_resolved_move_id="protect")) == 1
