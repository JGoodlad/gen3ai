"""The driver CLI's two non-obvious knobs: the arm DEFAULT and the per-battle time bounds."""

from __future__ import annotations

import pytest

from main.search_dividend.__main__ import DEFAULT_ARMS, _raise_battle_backstop, build_parser
from main.search_dividend.search import ARMS


def test_playoff_is_selectable_but_not_a_default_arm():
    """A flagless run must play what it always played. ``playoff`` costs orders of magnitude more
    than a critic sweep (it plays whole battles inside a decision), so adding it to ``ARMS`` must
    not silently change every existing invocation into a different, much longer experiment."""
    assert "playoff" in ARMS
    assert "playoff" not in DEFAULT_ARMS
    assert set(DEFAULT_ARMS) < set(ARMS)
    assert build_parser().parse_args(["m", "--arm", "playoff"]).arm == ["playoff"]


def test_the_playoff_knobs_default_to_the_registered_values():
    a = build_parser().parse_args(["m", "--arm", "playoff"])
    assert (a.playoff_rollouts, a.playoff_se_k, a.playoff_min_pairs) == (12, 2.0, 4)
    assert a.battle_timeout_s is None and a.battle_idle_s is None


def test_raising_the_battle_bounds_patches_BOTH_and_only_when_asked():
    """The two bounds answer different questions and the playoff arm breaks both assumptions — a
    nested rollout silences the live stream (idle) and a 25-decision game outruns 180 s (total).
    Raising one without the other still loses games, and a lost game poisons the rest of the cell.
    """
    from utils.bridge import local_battle_runner as lbr

    total0, idle0 = lbr._PER_BATTLE_TIMEOUT, lbr._BATTLE_IDLE_BUDGET
    try:
        _raise_battle_backstop(None, None)
        assert (lbr._PER_BATTLE_TIMEOUT, lbr._BATTLE_IDLE_BUDGET) == (total0, idle0)
        _raise_battle_backstop(5400.0, 120.0)
        assert lbr._PER_BATTLE_TIMEOUT == pytest.approx(5400.0)
        assert lbr._BATTLE_IDLE_BUDGET == pytest.approx(120.0)
    finally:
        lbr._PER_BATTLE_TIMEOUT, lbr._BATTLE_IDLE_BUDGET = total0, idle0


def test_the_idle_bound_is_read_at_CALL_time_so_patching_it_takes_effect():
    """A module-level constant captured at import would make the flag a no-op that looks like it
    worked. ``_await_battle`` reads the global when it builds its ``ProgressDeadline``."""
    import inspect

    from utils.bridge import local_battle_runner as lbr

    src = inspect.getsource(lbr._await_battle)
    assert "ProgressDeadline(_BATTLE_IDLE_BUDGET" in src
    assert "total_budget_s=_PER_BATTLE_TIMEOUT" in src
