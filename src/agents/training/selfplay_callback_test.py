"""Unit tests for selfplay_callback pure functions and regression guard logic."""

from unittest.mock import MagicMock, patch, call
import os

import pytest

from agents.training.selfplay_callback import (
    _monotonicity_score,
    SelfPlayCallback,
    _BOT_NAMES,
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


def test_regression_peak_persisted_after_check():
    cb = _make_callback()
    cb._check_bot_regression({"Heuristic": 0.75, "Staller": 0.60})
    cb._pool.persist_bot_peaks.assert_not_called()  # persist called from _eval_all, not here


# ── _emergency_save ──────────────────────────────────────────────────────────

def test_emergency_save_writes_checkpoint(tmp_path):
    cb = _make_callback(best_model_save_path=str(tmp_path))
    cb._emergency_save()
    cb.model.save.assert_called_once()
    saved_path = cb.model.save.call_args[0][0]
    assert "crash_" in saved_path
    assert str(cb.num_timesteps) in saved_path


def test_emergency_save_no_op_without_save_path():
    cb = _make_callback(best_model_save_path=None)
    cb._emergency_save()
    cb.model.save.assert_not_called()


def test_emergency_save_no_op_without_model():
    cb = _make_callback(best_model_save_path="/tmp")
    cb.model = None
    cb._emergency_save()  # should not raise


def test_emergency_save_swallows_save_errors(tmp_path):
    cb = _make_callback(best_model_save_path=str(tmp_path))
    cb.model.save.side_effect = OSError("disk full")
    cb._emergency_save()  # should not raise


# ── _schedule delegated to eval_schedule ────────────────────────────────────

def test_schedule_uses_shared_function():
    cb = _make_callback()
    cb.num_timesteps = 5_000_000
    assert cb._schedule() == (1_000_000, 100)
    cb.num_timesteps = 25_000_000
    assert cb._schedule() == (2_000_000, 200)
    cb.num_timesteps = 55_000_000
    assert cb._schedule() == (3_000_000, 300)
