"""Unit tests for the gen3_sleep_wake_belief_v1 belief (observation/sleep_belief.py).

The four P(wake) tables are pinned against the adversarially-verified gen3 re-simulation
(opp time=random(2,6)∈{2,3,4,5}, Rest time=3, Early Bird halves). The event-log extraction is
exercised with hand-built BattleEvents (the same shape Gen3Battle folds)."""
import math
from types import SimpleNamespace

from agents.battle.battle_event import BattleEvent, EventKind
from agents.observation.sleep_belief import (
    sleep_wake_probability, early_bird_probability, build_sleep_sources, sleep_belief_features,
)

_THIRD = 1.0 / 3.0
_TWO_THIRDS = 2.0 / 3.0


def test_pwake_tables_opp_no_earlybird():
    # K=0..5 → 0, 1/4, 1/3, 1/2, 1, 1  (1.0 sentinel on the unreachable K≥5)
    expected = [0.0, 0.25, _THIRD, 0.5, 1.0, 1.0]
    for k, e in enumerate(expected):
        assert math.isclose(sleep_wake_probability(k, is_rest=False, p_earlybird=0.0), e), k


def test_pwake_tables_opp_with_earlybird():
    expected = [0.25, _TWO_THIRDS, 1.0, 1.0, 1.0, 1.0]
    for k, e in enumerate(expected):
        assert math.isclose(sleep_wake_probability(k, is_rest=False, p_earlybird=1.0), e), k


def test_pwake_tables_rest_no_earlybird():
    expected = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
    for k, e in enumerate(expected):
        assert math.isclose(sleep_wake_probability(k, is_rest=True, p_earlybird=0.0), e), k


def test_pwake_tables_rest_with_earlybird():
    expected = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    for k, e in enumerate(expected):
        assert math.isclose(sleep_wake_probability(k, is_rest=True, p_earlybird=1.0), e), k


def test_pwake_marginalises_early_bird_prior():
    # opp, p_eb=0.5, K=0 → 0.5*0.25 + 0.5*0 = 0.125; K=2 → 0.5*1 + 0.5*(1/3) = 2/3
    assert math.isclose(sleep_wake_probability(0, False, 0.5), 0.125)
    assert math.isclose(sleep_wake_probability(2, False, 0.5), 0.5 * 1.0 + 0.5 * _THIRD)


def test_pwake_counter_clamped_to_reachable_max():
    # a corrupted (Sleep-Talk-inflated) counter clamps to the last index, not an index error
    assert sleep_wake_probability(99, False, 0.0) == 1.0
    assert sleep_wake_probability(-3, False, 0.0) == 0.0


def test_early_bird_probability_known_vs_prior():
    own_eb = SimpleNamespace(ability="earlybird", species="dodrio")
    own_other = SimpleNamespace(ability="pressure", species="suicune")
    assert early_bird_probability(own_eb) == 1.0
    assert early_bird_probability(own_other) == 0.0
    # unrevealed opp → species prior (suicune has no Early Bird → 0.0)
    opp_unrevealed = SimpleNamespace(ability=None, species="suicune")
    assert early_bird_probability(opp_unrevealed) == 0.0
    # an unknownability sentinel also falls through to the prior
    opp_unknown = SimpleNamespace(ability="unknownability", species="suicune")
    assert early_bird_probability(opp_unknown) == early_bird_probability(opp_unrevealed)


def _ev(seq, kind, side, species, **value):
    return BattleEvent(seq=seq, turn=1, kind=kind, side=side, actor_species=species, value=value)


def _battle(events):
    return SimpleNamespace(events=events)


def test_build_sleep_sources_rest_vs_opp():
    events = [
        _ev(1, EventKind.STATUS, "ours", "suicune", status="slp", reason="move: rest"),
        _ev(2, EventKind.STATUS, "opp", "snorlax", status="slp", reason="move: spore"),
    ]
    src = build_sleep_sources(_battle(events))
    assert src[("ours", "suicune")] == (True, False)    # Rest → deterministic, no sleep-usable move
    assert src[("opp", "snorlax")] == (False, False)    # Spore → random, reliable


def test_build_sleep_sources_sleep_talk_flags_unreliable():
    events = [
        _ev(1, EventKind.STATUS, "ours", "snorlax", status="slp", reason="move: spore"),
        _ev(2, EventKind.MOVE, "ours", "snorlax", move_id="sleeptalk"),
    ]
    src = build_sleep_sources(_battle(events))
    assert src[("ours", "snorlax")] == (False, True)    # sleep-usable move seen → unreliable


def test_build_sleep_sources_resleep_takes_latest_source():
    # slept by Spore, woke, re-slept by Rest, with a Sleep Talk BEFORE the re-sleep → the
    # re-sleep is deterministic and the pre-re-sleep Sleep Talk does NOT mark it unreliable.
    events = [
        _ev(1, EventKind.STATUS, "ours", "suicune", status="slp", reason="move: spore"),
        _ev(2, EventKind.MOVE, "ours", "suicune", move_id="sleeptalk"),
        _ev(5, EventKind.STATUS, "ours", "suicune", status="slp", reason="move: rest"),
    ]
    src = build_sleep_sources(_battle(events))
    assert src[("ours", "suicune")] == (True, False)


def test_build_sleep_sources_empty_when_no_sleep():
    assert build_sleep_sources(_battle([])) == {}
    assert build_sleep_sources(SimpleNamespace(events=None)) == {}


def test_sleep_belief_features_integration():
    sources = {("opp", "snorlax"): (False, False)}
    mon = SimpleNamespace(ability=None, species="snorlax")
    det, p_wake, reliable = sleep_belief_features(3, mon, is_own=False, sleep_sources=sources)
    assert det == 0.0 and reliable == 1.0
    assert math.isclose(p_wake, 0.5)                     # opp, K=3, no EB

    # Rest on our own mon (deterministic), K=2 → certain wake, reliable
    rest_sources = {("ours", "suicune"): (True, False)}
    own = SimpleNamespace(ability="pressure", species="suicune")
    det, p_wake, reliable = sleep_belief_features(2, own, is_own=True, sleep_sources=rest_sources)
    assert det == 1.0 and reliable == 1.0 and p_wake == 1.0

    # No sources map (mock path) → non-Rest, reliable, prior-only opp sleep
    det, p_wake, reliable = sleep_belief_features(1, mon, is_own=False, sleep_sources=None)
    assert det == 0.0 and reliable == 1.0 and math.isclose(p_wake, 0.25)
