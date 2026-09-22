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


# -- gen3_caller_sized_battle_budget_v1 (2026-09-22) -------------------------------------------


def test_battle_bounds_SCALE_with_the_per_decision_cost_and_never_go_below_the_defaults():
    """🚨 FAILS ON REVERT. The runner's 180 s TOTAL cap killed the longer orientation of every
    side-swapped playoff cell: a game measured at 246.6 s (60 decisions at ~4 s) was reported as
    `livelock, not a stall`, and it FINISHED — a WIN, 3 searched, 2 changed — as soon as the cap
    was raised. On a search arm both bounds are functions of the argv: the idle bound is the
    longest gap between two protocol chunks and a searched decision emits none."""
    from utils.bridge.local_battle_runner import _BATTLE_IDLE_BUDGET, _PER_BATTLE_TIMEOUT
    from main.search_dividend.player import battle_bounds

    assert battle_bounds(0.05) == (_BATTLE_IDLE_BUDGET, _PER_BATTLE_TIMEOUT), (
        "a cheap cell keeps the tighter, better bound — sizing must never LOOSEN one")

    idle, total = battle_bounds(5.0)
    assert idle == _BATTLE_IDLE_BUDGET, "5 s a decision still fits the 30 s chunk gap"
    # The measured game: 60 decisions at 4.1 s = 246.6 s. The bound must cover it.
    assert total >= 246.6, total
    assert battle_bounds(30.0)[0] > _BATTLE_IDLE_BUDGET, (
        "a decision longer than the chunk gap must widen the IDLE bound too, or the search's own "
        "thinking time reads as a wedged bridge")
    assert battle_bounds(20.0)[1] > total, "the total must be monotone in the decision cost"


def test_a_timed_out_battle_KEEPS_the_decisions_it_made(monkeypatch):
    """A game that timed out used to reach the results file as `decisions: []` — so 41 real
    decisions read as `dec=0`, which looks like a battle that never started and is exactly how
    the livelock was mis-described. The outcome stays `unfinished` (a timeout is never a
    semantic outcome); what changes is that the evidence survives."""
    import asyncio

    from main.search_dividend import player as P

    class _Builder:
        battle_tag = "battle-x-1"
        n_commands = 3

    class _Player:
        username = "T"

        def __init__(self):
            self.decisions: list = []
            self._battles: dict = {}

        def open_battle(self, seed, chunk_sink, our_side="p1"):
            return _Builder()

    p_obj = _Player()

    async def _boom(*a, **kw):
        p_obj.decisions.append({"turn": 1, "fallback": None})
        p_obj.decisions.append({"turn": 2, "fallback": "deadline"})
        raise TimeoutError("exceeded total budget: 180.5s")

    monkeypatch.setattr("utils.bridge.local_battle_runner.run_local_battles", _boom)
    out = asyncio.run(P.play_one_battle(p_obj, object(), battle_format="gen3ou",
                                        seed="sodium,00", impl="rust"))
    assert out["outcome"] == "unfinished" and out["finished"] == 0
    assert "exceeded total budget" in (out["error"] or "")
    assert len(out["decisions"]) == 2, "the decisions the battle DID make must survive"
