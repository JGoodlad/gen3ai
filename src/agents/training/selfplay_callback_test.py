"""Unit tests for selfplay_callback pure functions and regression guard logic."""

from unittest.mock import MagicMock, patch, call
import os

import pytest

from agents.training.selfplay_callback import (
    _monotonicity_score,
    SelfPlayCallback,
    _REGRESSION_WARN_THRESHOLD,
)


# ── _monotonicity_score ──────────────────────────────────────────────────────

def test_monotonicity_single_entry():
    # n < 2 → always 1.0
    assert _monotonicity_score([0.7]) == pytest.approx(1.0)


def test_monotonicity_empty():
    assert _monotonicity_score([]) == pytest.approx(1.0)


def test_monotonicity_perfectly_monotone():
    # index 0 = most recent (hardest) → lowest win rate; index n-1 = oldest (easiest)
    # Perfectly healthy: win rates increase from index 0 to index n-1
    assert _monotonicity_score([0.4, 0.5, 0.6, 0.7, 0.8]) == pytest.approx(1.0)


def test_monotonicity_perfectly_inverted():
    # Win rates decrease from index 0 to n-1 → fully inverted = -1.0
    assert _monotonicity_score([0.8, 0.7, 0.6, 0.5, 0.4]) == pytest.approx(-1.0)


def test_monotonicity_two_entries_ordered():
    assert _monotonicity_score([0.4, 0.8]) == pytest.approx(1.0)


def test_monotonicity_two_entries_inverted():
    assert _monotonicity_score([0.8, 0.4]) == pytest.approx(-1.0)


def test_monotonicity_ties_count_as_concordant():
    # Ties (equal win rates) are treated as concordant in the ≤ check
    assert _monotonicity_score([0.5, 0.5, 0.5]) == pytest.approx(1.0)


def test_monotonicity_mixed():
    # [0.4, 0.8, 0.6]: pairs (0,1)→concordant, (0,2)→concordant, (1,2)→discordant
    # 2 concordant / 3 total → τ = 2*(2/3) - 1 = 1/3
    result = _monotonicity_score([0.4, 0.8, 0.6])
    assert result == pytest.approx(1 / 3, abs=1e-6)


# ── SelfPlayCallback helper: build a minimal instance ───────────────────────

def _make_callback(best_model_save_path=None):
    pool = MagicMock()
    pool.load_persisted_win_rate.return_value = 0.0
    pool.load_persisted_bot_peaks.return_value = {}
    cb = SelfPlayCallback(
        pool=pool,
        bot_opponents=[],
        trainee_teambuilder=MagicMock(),
        opp_teambuilder=MagicMock(),
        mappings=MagicMock(),
        best_model_save_path=best_model_save_path,
    )
    cb.model = MagicMock()
    cb.model.save = MagicMock()
    cb.num_timesteps = 5_000_000
    return cb


# ── _check_bot_regression ────────────────────────────────────────────────────

def test_regression_no_warning_before_threshold():
    cb = _make_callback()
    with patch("agents.training.selfplay_callback.emit") as mock_emit:
        cb._check_bot_regression({"Heuristic": 0.40})
        mock_emit.assert_not_called()


def test_regression_peak_recorded():
    cb = _make_callback()
    cb._check_bot_regression({"Heuristic": 0.75})
    assert cb._bot_peak["Heuristic"] == pytest.approx(0.75)


def test_regression_warning_fires_when_drop_below_threshold():
    cb = _make_callback()
    # First eval: reach above warn threshold
    cb._check_bot_regression({"Heuristic": 0.72})
    with patch("agents.training.selfplay_callback.emit") as mock_emit:
        # Second eval: drop below warn threshold
        cb._check_bot_regression({"Heuristic": 0.55})
        assert mock_emit.call_count == 1
        assert "BOT_REGRESSION" in mock_emit.call_args[0][0]
        assert "Heuristic" in mock_emit.call_args[0][0]


def test_regression_fires_even_if_peak_below_old_trigger_threshold():
    # Old code required peak >= 0.70 first. New code warns from 0.60 onward.
    cb = _make_callback()
    cb._check_bot_regression({"Heuristic": 0.65})  # peak set to 0.65 (above warn threshold)
    with patch("agents.training.selfplay_callback.emit") as mock_emit:
        cb._check_bot_regression({"Heuristic": 0.50})  # drop below 0.60
        assert mock_emit.call_count == 1


def test_regression_no_warning_if_still_above_threshold():
    cb = _make_callback()
    cb._check_bot_regression({"Heuristic": 0.80})
    with patch("agents.training.selfplay_callback.emit") as mock_emit:
        cb._check_bot_regression({"Heuristic": 0.65})  # still above 0.60
        mock_emit.assert_not_called()


def test_regression_only_fires_for_named_bots():
    cb = _make_callback()
    cb._check_bot_regression({"Random": 0.90})
    with patch("agents.training.selfplay_callback.emit") as mock_emit:
        cb._check_bot_regression({"Random": 0.10})
        mock_emit.assert_not_called()


# ── regression guard — edge-trigger ─────────────────────────────────────────

def test_regression_fires_only_once_while_regressed():
    cb = _make_callback()
    cb._check_bot_regression({"Heuristic": 0.75})
    with patch("agents.training.selfplay_callback.emit") as mock_emit:
        cb._check_bot_regression({"Heuristic": 0.50})  # enters warned state → fires
        cb._check_bot_regression({"Heuristic": 0.45})  # still in warned state → silent
        cb._check_bot_regression({"Heuristic": 0.40})  # still in warned state → silent
        assert mock_emit.call_count == 1


def test_regression_re_arms_after_recovery():
    cb = _make_callback()
    cb._check_bot_regression({"Heuristic": 0.75})
    with patch("agents.training.selfplay_callback.emit") as mock_emit:
        cb._check_bot_regression({"Heuristic": 0.50})  # first regression → fires
        cb._check_bot_regression({"Heuristic": 0.70})  # recovered → re-arms
        cb._check_bot_regression({"Heuristic": 0.45})  # regressed again → fires again
        assert mock_emit.call_count == 2


# ── _abort ───────────────────────────────────────────────────────────────────

def test_abort_calls_abort_fn_when_set():
    cb = _make_callback()
    abort_fn = MagicMock()
    cb.abort_fn = abort_fn
    cb._abort("test reason")
    abort_fn.assert_called_once_with("test reason")


def test_abort_falls_back_to_os_exit_when_unset():
    cb = _make_callback()
    cb.abort_fn = None
    with patch("os._exit") as mock_exit:
        cb._abort("test reason")
        mock_exit.assert_called_once_with(1)


# ── _schedule delegated to eval_schedule ────────────────────────────────────

def test_schedule_uses_shared_function():
    cb = _make_callback()
    cb.num_timesteps = 5_000_000
    assert cb._schedule() == (1_000_000, 100)
    cb.num_timesteps = 25_000_000
    assert cb._schedule() == (2_000_000, 200)
    cb.num_timesteps = 55_000_000
    assert cb._schedule() == (3_000_000, 300)
