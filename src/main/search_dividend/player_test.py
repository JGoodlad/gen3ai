"""What a played battle's OUTCOME is — and the tie that used to be recorded as a defeat."""

from __future__ import annotations

from main.search_dividend.player import OUTCOMES, battle_outcome


class _Battle:
    def __init__(self, finished, won):
        self.finished = finished
        self.won = won


def test_a_won_and_a_lost_battle_read_straight_off_the_battle_object():
    assert battle_outcome(_Battle(True, True)) == "win"
    assert battle_outcome(_Battle(True, False)) == "loss"


def test_a_TIE_is_its_own_outcome_and_is_never_a_loss():
    """THE regression. poke-env reports a draw as ``finished`` with ``won is None`` — so the win
    COUNTER the battery used to diff (`n_won_battles`, which counts only truthy `won`) cannot tell
    a draw from a defeat, and every gen3 tie was silently recorded as `result="loss"` with no
    error and nothing anywhere saying a draw had happened.

    It matters most in the MIRROR mode this test was written for: two copies of one network draw
    far more often than a policy and a scripted bot do, and every draw would have been charged to
    the searched side — a bias with a DIRECTION, pointing the one way that makes the search look
    worse than it is."""
    assert battle_outcome(_Battle(True, None)) == "tie"


def test_an_unfinished_or_missing_battle_is_UNFINISHED_not_a_loss():
    """A crash / a bridge child that died at spawn is never a semantic outcome — the contention
    lesson, which this project has already paid for once."""
    assert battle_outcome(_Battle(False, None)) == "unfinished"
    assert battle_outcome(None) == "unfinished"


def test_the_outcome_vocabulary_is_closed():
    """Every consumer branches on these four strings; a fifth would show up as a silent no-match
    in a summary rather than as an error."""
    for b in (_Battle(True, True), _Battle(True, False), _Battle(True, None),
              _Battle(False, None), None):
        assert battle_outcome(b) in OUTCOMES
