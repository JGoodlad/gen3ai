from unittest.mock import MagicMock, patch

import pytest

from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from agents.opponents import Gen3StallerPlayer
from agents.training.eval_callback import (
    PerOpponentEvalCallback, bot_mean, opponent_name, RANDOM_OPPONENT_NAME,
    _per_opponent_concurrency, _EVAL_TOTAL_CONCURRENCY, _EVAL_CONCURRENCY,
)


# ── bot_mean ─────────────────────────────────────────────────────────────────

def test_bot_mean_excludes_random():
    assert bot_mean({"Random": 0.9, "Heuristic": 0.4, "Staller": 0.6}) == pytest.approx(0.5)


def test_bot_mean_all_random_returns_zero():
    assert bot_mean({"Random": 0.9}) == pytest.approx(0.0)


def test_bot_mean_empty_returns_zero():
    assert bot_mean({}) == pytest.approx(0.0)


def test_bot_mean_no_random_averages_all():
    assert bot_mean({"Heuristic": 0.4, "Staller": 0.6}) == pytest.approx(0.5)


# ── opponent_name ─────────────────────────────────────────────────────────────

def test_opponent_name_random():
    assert opponent_name(RandomPlayer) == "Random"


def test_opponent_name_heuristic():
    assert opponent_name(SimpleHeuristicsPlayer) == "Heuristic"


def test_opponent_name_staller():
    assert opponent_name(Gen3StallerPlayer) == "Staller"


def test_opponent_name_unknown_falls_back_to_class_name():
    class MyCustomPlayer:
        pass
    assert opponent_name(MyCustomPlayer) == "MyCustomPlayer"


def test_random_opponent_name_constant_matches_function():
    assert RANDOM_OPPONENT_NAME == opponent_name(RandomPlayer)


# ── _per_opponent_concurrency ─────────────────────────────────────────────────

def test_per_opponent_concurrency_splits_budget():
    # Aggregate (n × per-player) stays at/below the total budget.
    assert _per_opponent_concurrency(5) == _EVAL_TOTAL_CONCURRENCY // 5
    assert _per_opponent_concurrency(9) * 9 <= _EVAL_TOTAL_CONCURRENCY + 9


def test_per_opponent_concurrency_floored_at_16():
    # Many opponents still get enough concurrency to saturate inference.
    assert _per_opponent_concurrency(100) == 16


def test_per_opponent_concurrency_zero_falls_back():
    assert _per_opponent_concurrency(0) == _EVAL_CONCURRENCY


def _make_callback(best_model_save_path=None):
    opp_a = MagicMock()
    opp_a.n_finished_battles = 0
    opp_b = MagicMock()
    opp_b.n_finished_battles = 0
    cb = PerOpponentEvalCallback(
        opponents=[("Random", opp_a), ("Heuristic", opp_b)],
        trainee_teambuilder=MagicMock(),
        mappings=MagicMock(),
        best_model_save_path=best_model_save_path,
    )
    cb.model = MagicMock()
    cb.model.save = MagicMock()
    cb.model.logger = MagicMock()
    cb.num_timesteps = 0
    return cb


# --- Schedule ---

def test_schedule_early_phase():
    cb = _make_callback()
    cb.num_timesteps = 5_000_000
    freq, n_games = cb._schedule()
    assert freq == 1_000_000
    assert n_games == 100


def test_schedule_mid_phase():
    cb = _make_callback()
    cb.num_timesteps = 30_000_000
    freq, n_games = cb._schedule()
    assert freq == 2_000_000
    assert n_games == 200


def test_schedule_late_phase():
    cb = _make_callback()
    cb.num_timesteps = 60_000_000
    freq, n_games = cb._schedule()
    assert freq == 3_000_000
    assert n_games == 300


def test_schedule_boundary_20m():
    cb = _make_callback()
    cb.num_timesteps = 20_000_000
    freq, n_games = cb._schedule()
    assert freq == 2_000_000
    assert n_games == 200


def test_schedule_boundary_50m():
    cb = _make_callback()
    cb.num_timesteps = 50_000_000
    freq, n_games = cb._schedule()
    assert freq == 3_000_000
    assert n_games == 300


# --- Trigger logic ---

def test_no_eval_at_step_zero():
    cb = _make_callback()
    cb.num_timesteps = 0
    with patch.object(cb, '_run_async_eval') as mock_run:
        cb._on_step()
        mock_run.assert_not_called()


def test_no_eval_before_first_freq_boundary():
    cb = _make_callback()
    cb.num_timesteps = 500_000  # below the first 1M-step boundary
    with patch.object(cb, '_run_async_eval') as mock_run:
        cb._on_step()
        mock_run.assert_not_called()


def test_triggers_at_early_freq():
    cb = _make_callback()
    cb.num_timesteps = 1_000_000
    with patch.object(cb, '_run_async_eval') as mock_run:
        cb._on_step()
        mock_run.assert_called_once_with(100)


def test_triggers_at_mid_freq():
    cb = _make_callback()
    cb._last_eval_step = 20_000_000
    cb.num_timesteps = 22_000_000
    with patch.object(cb, '_run_async_eval') as mock_run:
        cb._on_step()
        mock_run.assert_called_once_with(200)


def test_triggers_at_late_freq():
    cb = _make_callback()
    cb._last_eval_step = 51_000_000
    cb.num_timesteps = 54_000_000
    with patch.object(cb, '_run_async_eval') as mock_run:
        cb._on_step()
        mock_run.assert_called_once_with(300)


def test_no_double_trigger_within_interval():
    cb = _make_callback()
    cb._last_eval_step = 1_000_000
    cb.num_timesteps = 1_500_000
    with patch.object(cb, '_run_async_eval') as mock_run:
        cb._on_step()
        mock_run.assert_not_called()


def test_updates_last_eval_step_on_trigger():
    cb = _make_callback()
    cb.num_timesteps = 1_000_000
    with patch.object(cb, '_run_async_eval'):
        cb._on_step()
    assert cb._last_eval_step == 1_000_000


# --- Best model saving ---

def test_saves_best_model_on_first_improvement(tmp_path):
    cb = _make_callback(best_model_save_path=str(tmp_path))
    cb._best_aggregate_win_rate = -1.0
    aggregate = 0.65
    if aggregate > cb._best_aggregate_win_rate:
        cb._best_aggregate_win_rate = aggregate
        cb.model.save(str(tmp_path / "best_model"))
    cb.model.save.assert_called_once()
    assert cb._best_aggregate_win_rate == pytest.approx(0.65)


def test_does_not_save_when_aggregate_does_not_improve(tmp_path):
    cb = _make_callback(best_model_save_path=str(tmp_path))
    cb._best_aggregate_win_rate = 0.80
    aggregate = 0.75
    if aggregate > cb._best_aggregate_win_rate:
        cb.model.save(str(tmp_path / "best_model"))
    cb.model.save.assert_not_called()
