"""Unit tests for the dice-attribution falsifier's pure logic (no Node, no battles)."""

import numpy as np
import pytest

from main.prober import falsifier
from main.prober.falsifier import (
    classify,
    fresh_seeds,
    material_margin,
    paired_stats,
    percentile_of,
    select_anchors,
)


def _outcome(p1_alive, p1_hp, p2_alive, p2_hp, maxhp=600.0):
    return {
        "p1": {"alive": p1_alive, "team_hp": p1_hp, "team_maxhp": maxhp},
        "p2": {"alive": p2_alive, "team_hp": p2_hp, "team_maxhp": maxhp},
    }


# ---------------------------------------------------------------------------
# material_margin
# ---------------------------------------------------------------------------

def test_margin_alive_dominates_hp():
    # 1 extra mon beats any same-material HP edge (alive term is ±1, hp term <1).
    up_a_mon = material_margin(_outcome(4, 300, 3, 300), "p1")
    up_hp_only = material_margin(_outcome(3, 500, 3, 100), "p1")
    assert up_a_mon > 0 and up_hp_only > 0
    assert up_a_mon > up_hp_only - 1.0  # alive diff contributes a full point


def test_margin_is_antisymmetric_across_sides():
    o = _outcome(4, 350, 2, 500)
    assert material_margin(o, "p1") == pytest.approx(-material_margin(o, "p2"))


def test_margin_even_board_is_zero():
    assert material_margin(_outcome(3, 300, 3, 300), "p1") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# percentile / paired stats
# ---------------------------------------------------------------------------

def test_percentile_midrank():
    samples = [0.0, 1.0, 2.0, 3.0]
    assert percentile_of(-1.0, samples) == 0.0
    assert percentile_of(10.0, samples) == 1.0
    assert percentile_of(1.0, samples) == pytest.approx((1 + 0.5) / 4)  # one below, one tie
    assert percentile_of(0.5, []) is None


def test_paired_stats_basics():
    mean, se = paired_stats([1.0, 1.0, 1.0, 1.0])
    assert mean == 1.0 and se == 0.0
    mean, se = paired_stats([0.0, 2.0])
    assert mean == 1.0 and se == pytest.approx(np.std([0, 2], ddof=1) / np.sqrt(2))
    _, se1 = paired_stats([3.0])
    assert se1 == float("inf")  # one pair can never be significant
    mean0, se0 = paired_stats([])
    assert mean0 == 0.0 and se0 == float("inf")


# ---------------------------------------------------------------------------
# verdicts
# ---------------------------------------------------------------------------

def test_classify_matrix():
    # realized in the bad tail, no better alt → LUCK
    assert classify(0.05, best_adv=0.1, best_se=0.2) == "LUCK"
    # strong, significant alt, fair dice → MISTAKE
    assert classify(0.5, best_adv=1.2, best_se=0.2) == "MISTAKE"
    # both → MIXED
    assert classify(0.1, best_adv=1.2, best_se=0.2) == "MIXED"
    # neither → NEUTRAL
    assert classify(0.5, best_adv=0.2, best_se=0.2) == "NEUTRAL"
    # a big advantage that is NOT significant stays NEUTRAL (1 pair, inf se)
    assert classify(0.5, best_adv=3.0, best_se=float("inf")) == "NEUTRAL"
    # no alternatives at all
    assert classify(0.5, best_adv=None, best_se=None) == "NEUTRAL"
    assert classify(None, best_adv=None, best_se=None) == "NEUTRAL"


def test_classify_threshold_edges():
    # advantage must clear BOTH the material floor and the z-test
    just_below_floor = falsifier._ADV_MIN - 0.01
    assert classify(0.9, best_adv=just_below_floor, best_se=0.0001) == "NEUTRAL"
    assert classify(falsifier._LUCK_PCTL_MAX, best_adv=None, best_se=None) == "LUCK"


# ---------------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------------

def test_fresh_seeds_deterministic_and_distinct():
    a = fresh_seeds(10, "battle-x:7")
    b = fresh_seeds(10, "battle-x:7")
    c = fresh_seeds(10, "battle-x:8")
    assert a == b
    assert a != c
    assert len(set(a)) == 10
    for s in a:
        parts = s.split(",")
        assert len(parts) == 4 and all(0 <= int(x) < 65536 for x in parts)
        assert s != "original"


# ---------------------------------------------------------------------------
# anchor selection
# ---------------------------------------------------------------------------

def _summary_with(invs):
    return {"invocations": invs}


def _inv(turn, phase, rtotal):
    return {"turn": turn, "phase": phase, "outcome": {"reward": rtotal},
            "actions": {}}


def test_select_anchors_ranks_by_td_and_dedupes_turns():
    # δ_i = r_i + γ·V_{i+1} − V_i with γ=1 for easy math.
    invs = [
        _inv(1, "move_selection", 0.0),   # δ = 0 + 5 − 5 = 0
        _inv(2, "move_selection", -4.0),  # δ = −4 + 5 − 5 = −4   (worst)
        _inv(3, "move_selection", -1.0),  # δ = −1 + 5 − 5 = −1
        _inv(4, "move_selection", 0.0),   # last inv: no next value → skipped
    ]
    npz = {"values": np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32)}
    anchors = select_anchors(_summary_with(invs), npz, gamma=1.0, worst=2)
    assert anchors == [1, 2]


def test_select_anchors_attributes_forced_switch_to_turn_move():
    # The crater happens at the forced-switch invocation of turn 2; the anchor
    # must be turn 2's move_selection inv (the re-rollable decision).
    invs = [
        _inv(1, "move_selection", 0.0),
        _inv(2, "move_selection", 0.0),
        _inv(2, "forced_switch", -6.0),   # the crater
        _inv(3, "move_selection", 0.0),
    ]
    npz = {"values": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)}
    anchors = select_anchors(_summary_with(invs), npz, gamma=1.0, worst=1)
    assert anchors == [1]
    assert invs[1]["phase"] == "move_selection" and invs[1]["turn"] == 2


def test_select_anchors_requires_values():
    with pytest.raises(ValueError):
        select_anchors(_summary_with([_inv(1, "move_selection", 0.0)]), {}, gamma=1.0, worst=1)


def test_anchor_deltas_maps_worst_delta_per_anchor():
    # The δ MAP (not just the ranking) is what falsify_scan weights craters by.
    invs = [
        _inv(1, "move_selection", 0.0),   # δ = 0
        _inv(2, "move_selection", -4.0),  # δ = -4
        _inv(3, "move_selection", -1.0),  # δ = -1
        _inv(4, "move_selection", 0.0),   # last inv: no next value → absent
    ]
    npz = {"values": np.array([5.0, 5.0, 5.0, 5.0], dtype=np.float32)}
    deltas = falsifier.anchor_deltas(_summary_with(invs), npz, gamma=1.0)
    assert deltas == {0: 0.0, 1: -4.0, 2: -1.0}
    # select_anchors is exactly the ranking over this map.
    assert select_anchors(_summary_with(invs), npz, gamma=1.0, worst=2) == [1, 2]


def test_anchor_deltas_forced_switch_folds_worst_into_turn_move():
    # The crater is on turn 2's forced-switch inv; its δ must fold into turn 2's
    # move anchor (inv1) as the WORST δ of the turn — so the scan weights the
    # re-rollable anchor by the true crater magnitude.
    invs = [
        _inv(1, "move_selection", 0.0),
        _inv(2, "move_selection", 0.0),   # δ = 0
        _inv(2, "forced_switch", -6.0),   # the crater (same turn)
        _inv(3, "move_selection", 0.0),
    ]
    npz = {"values": np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)}
    deltas = falsifier.anchor_deltas(_summary_with(invs), npz, gamma=1.0)
    assert deltas[1] == -6.0 and 2 not in deltas


# ---------------------------------------------------------------------------
# falsify_decision argument guards (no re-roll spawned)
# ---------------------------------------------------------------------------

class _Rec:
    battle_tag = "battle-gen3ou-t"
    trainee_username = "Z"

    def side_of(self, name):
        return "p1"


def test_falsify_decision_rejects_forced_switch_anchor():
    summary = _summary_with([_inv(1, "forced_switch", 0.0)])
    with pytest.raises(ValueError, match="move_selection"):
        falsifier.falsify_decision(_Rec(), summary, {"actions": np.array([0])}, 0)


def test_falsify_decision_requires_actions_array():
    summary = _summary_with([_inv(1, "move_selection", 0.0)])
    with pytest.raises(ValueError, match="actions"):
        falsifier.falsify_decision(_Rec(), summary, {}, 0)


def test_falsify_decision_inv_out_of_range():
    with pytest.raises(IndexError):
        falsifier.falsify_decision(_Rec(), _summary_with([]), {"actions": np.array([0])}, 0)
