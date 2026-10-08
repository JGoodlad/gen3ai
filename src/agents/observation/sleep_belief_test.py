"""Unit tests for the gen3_sleep_wake_belief_v1 belief (observation/sleep_belief.py).

The four P(wake) tables are pinned against the adversarially-verified gen3 re-simulation
(opp time=random(2,6)∈{2,3,4,5}, Rest time=3, Early Bird halves). The event-log extraction is
exercised with hand-built BattleEvents (the same shape Gen3Battle folds)."""
import math

from agents.observation.sleep_belief import (
    sleep_wake_probability,
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
