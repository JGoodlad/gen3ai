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


# -- the ROOT P(win) the defensive gate reads (see defensive.py) --------------


class _Extractor:
    def __init__(self, logits):
        self.last_win_prob_logits = logits


class _Policy:
    def __init__(self, logits):
        self.features_extractor = _Extractor(logits)


class _Model:
    def __init__(self, logits):
        self.policy = _Policy(logits)


def test_the_root_win_prob_is_read_off_the_live_forwards_own_stash():
    import torch

    from main.search_dividend.player import _safe_win_prob

    assert _safe_win_prob(_Model(torch.tensor([[0.0]]))) == 0.5
    assert _safe_win_prob(_Model(torch.tensor([[2.0]]))) > 0.88


def test_a_run_with_no_win_prob_head_yields_None_never_an_imputed_half():
    """0.5 is the MOST contested value the gate knows, so imputing it on a missing measurement
    would route every decision into the searched class. The engine turns this ``None`` into a
    counted `defensive_no_win_prob` refusal instead."""
    from main.search_dividend.player import _safe_win_prob

    assert _safe_win_prob(_Model(None)) is None


def test_the_search_hop_carries_the_win_prob_through_to_the_engine():
    """The stash is clobbered by the search's own forwards, so the value has to be captured at the
    live decision and PASSED. A signature that silently dropped it would leave the gate reading
    None on every decision — 100% `defensive_no_win_prob`, a cell that measures nothing."""
    import inspect

    from main.search_dividend.player import SearchDividendPlayer
    from main.search_dividend.search import SearchEngine

    assert "root_win_prob" in inspect.signature(SearchEngine.choose).parameters
    assert "root_win_prob" in inspect.signature(SearchDividendPlayer._search).parameters
    src = inspect.getsource(SearchDividendPlayer.choose_move)
    assert "_safe_win_prob(self.model)" in src, "read it off the LIVE forward"
    assert src.index("_safe_win_prob") < src.index("run_in_executor"), \
        "...and BEFORE the search's own forwards overwrite the stash"
    assert "root_win_prob=root_win_prob" in inspect.getsource(SearchDividendPlayer._search)
