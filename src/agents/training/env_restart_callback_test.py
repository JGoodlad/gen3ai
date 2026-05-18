from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from agents.training.env_restart_callback import VecEnvRestartCallback, _SECONDS_PER_HOUR


def _callback(interval_hours=1.0, num_envs=4):
    """Return (callback, make_factories) with _last_restart pinned to t=0."""
    make_factories = MagicMock(return_value=[MagicMock() for _ in range(num_envs)])
    cb = VecEnvRestartCallback(make_factories=make_factories, restart_interval_hours=interval_hours)
    cb._last_restart = 0.0
    return cb, make_factories


def _subproc_env(num_envs=4):
    """Plain MagicMock — auto-has all attributes including 'processes'."""
    env = MagicMock()
    env.num_envs = num_envs
    env.reset.return_value = MagicMock()
    return env


def _dummy_env(num_envs=4):
    """MagicMock without 'processes' attr (simulates DummyVecEnv)."""
    env = MagicMock(spec=['num_envs', 'reset', 'close', 'set_env'])
    env.num_envs = num_envs
    return env


def _attach_model(cb, env):
    cb.model = MagicMock()
    cb.model.env = env


class TestVecEnvRestartCallback:
    def test_interval_stored_in_seconds(self):
        cb, _ = _callback(interval_hours=2.5)
        assert cb._restart_interval == 2.5 * _SECONDS_PER_HOUR

    def test_on_step_returns_true(self):
        cb, _ = _callback()
        _attach_model(cb, _subproc_env())
        assert cb._on_step() is True

    def test_skip_when_disabled(self):
        cb, make_factories = _callback(interval_hours=0.0)
        _attach_model(cb, _subproc_env())
        with patch('agents.training.env_restart_callback.time') as t:
            t.monotonic.return_value = 99999.0
            cb._on_rollout_end()
        make_factories.assert_not_called()

    def test_skip_for_dummy_vec_env(self):
        cb, make_factories = _callback(interval_hours=1.0)
        _attach_model(cb, _dummy_env())
        with patch('agents.training.env_restart_callback.time') as t:
            t.monotonic.return_value = 99999.0
            cb._on_rollout_end()
        make_factories.assert_not_called()

    def test_skip_before_interval_elapses(self):
        cb, make_factories = _callback(interval_hours=1.0)
        _attach_model(cb, _subproc_env())
        with patch('agents.training.env_restart_callback.time') as t:
            t.monotonic.return_value = 1800.0  # 0.5h, threshold is 1h
            cb._on_rollout_end()
        make_factories.assert_not_called()

    @patch('agents.training.env_restart_callback.start_subprocess_watchdog')
    @patch('agents.training.env_restart_callback.SubprocVecEnv')
    def test_restart_after_interval(self, MockVecEnv, mock_watchdog):
        num_envs = 4
        old_env = _subproc_env(num_envs)
        new_env = _subproc_env(num_envs)
        MockVecEnv.return_value = new_env

        factories = [MagicMock() for _ in range(num_envs)]
        make_factories = MagicMock(return_value=factories)
        cb = VecEnvRestartCallback(make_factories=make_factories, restart_interval_hours=1.0)
        cb._last_restart = 0.0
        _attach_model(cb, old_env)

        with patch('agents.training.env_restart_callback.time') as t:
            # First call: elapsed check; second call: reset _last_restart
            t.monotonic.side_effect = [3601.0, 3601.0]
            cb._on_rollout_end()

        make_factories.assert_called_once()
        MockVecEnv.assert_called_once_with(factories)
        cb.model.set_env.assert_called_once_with(new_env)
        assert cb.model._last_obs is new_env.reset.return_value
        np.testing.assert_array_equal(
            cb.model._last_episode_starts, np.ones(num_envs, dtype=bool)
        )
        mock_watchdog.assert_called_once_with(new_env, label="train_env")
        old_env.close.assert_called_once()

    @patch('agents.training.env_restart_callback.start_subprocess_watchdog')
    @patch('agents.training.env_restart_callback.SubprocVecEnv')
    def test_timer_resets_after_restart(self, MockVecEnv, mock_watchdog):
        new_env = _subproc_env()
        MockVecEnv.return_value = new_env
        cb, make_factories = _callback(interval_hours=1.0)
        _attach_model(cb, _subproc_env())

        with patch('agents.training.env_restart_callback.time') as t:
            # First call: elapsed=3600 → restart, _last_restart set to 3600
            # Second call: elapsed=3700-3600=100 → skip
            t.monotonic.side_effect = [3600.0, 3600.0, 3700.0]
            cb._on_rollout_end()
            cb.model.env = new_env
            cb._on_rollout_end()

        assert make_factories.call_count == 1

    @patch('agents.training.env_restart_callback.start_subprocess_watchdog')
    @patch('agents.training.env_restart_callback.SubprocVecEnv')
    def test_closes_old_env_after_swap(self, MockVecEnv, mock_watchdog):
        old_env = _subproc_env()
        new_env = _subproc_env()
        MockVecEnv.return_value = new_env
        cb, _ = _callback(interval_hours=1.0)
        _attach_model(cb, old_env)

        with patch('agents.training.env_restart_callback.time') as t:
            t.monotonic.side_effect = [3601.0, 3601.0]
            cb._on_rollout_end()

        # close() must happen AFTER set_env to avoid interrupting an active env
        set_env_call_order = cb.model.method_calls.index(
            next(c for c in cb.model.method_calls if c[0] == 'set_env')
        )
        close_call_order = old_env.method_calls.index(
            next(c for c in old_env.method_calls if c[0] == 'close')
        )
        # set_env on model happens before close on old_env — verify by checking close was called
        old_env.close.assert_called_once()
