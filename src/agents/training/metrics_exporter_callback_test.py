import json
import os
from unittest.mock import MagicMock, patch

import pytest

from agents.training.metrics_exporter_callback import MetricsExporterCallback


def _make_callback(fd: int = -1) -> MetricsExporterCallback:
    with patch.dict(os.environ, {"LAUNCHER_METRICS_FD": str(fd)}):
        return MetricsExporterCallback()


def _attach_logger(cb: MetricsExporterCallback, metrics: dict) -> None:
    """Wire a fake model+logger onto cb so self.logger.name_to_value works."""
    mock_log = MagicMock()
    mock_log.name_to_value = metrics
    mock_model = MagicMock()
    mock_model.logger = mock_log
    cb.model = mock_model


class TestMetricsExporterCallback:
    def test_noop_when_env_var_absent(self):
        env = {k: v for k, v in os.environ.items() if k != "LAUNCHER_METRICS_FD"}
        with patch.dict(os.environ, env, clear=True):
            cb = MetricsExporterCallback()
        assert cb._pipe is None

    def test_noop_when_fd_negative(self):
        cb = _make_callback(fd=-1)
        assert cb._pipe is None

    def test_on_rollout_end_noop_without_pipe(self):
        cb = _make_callback(fd=-1)
        _attach_logger(cb, {"x": 1.0})
        cb.num_timesteps = 0
        cb._on_rollout_end()  # must not raise

    def test_writes_json_line_on_rollout_end(self):
        r_fd, w_fd = os.pipe()
        cb = _make_callback(fd=w_fd)
        assert cb._pipe is not None

        _attach_logger(cb, {"rollout/ep_rew_mean": -5.0, "time/fps": 1200.0})
        cb.num_timesteps = 50000
        cb._on_rollout_end()
        cb._on_training_end()  # closes pipe → read side gets EOF

        with os.fdopen(r_fd, "r") as f:
            payload = json.loads(f.readline())

        assert payload["rollout/ep_rew_mean"] == pytest.approx(-5.0)
        assert payload["time/fps"] == pytest.approx(1200.0)
        assert payload["_step"] == 50000

    def test_skips_non_numeric_values(self):
        r_fd, w_fd = os.pipe()
        cb = _make_callback(fd=w_fd)
        _attach_logger(cb, {"metric/num": 1.0, "metric/str": "hello"})
        cb.num_timesteps = 0
        cb._on_rollout_end()
        cb._on_training_end()

        with os.fdopen(r_fd, "r") as f:
            payload = json.loads(f.readline())

        assert "metric/str" not in payload
        assert "metric/num" in payload

    def test_sets_pipe_none_on_broken_pipe(self):
        r_fd, w_fd = os.pipe()
        cb = _make_callback(fd=w_fd)
        os.close(r_fd)  # close read end → write will raise BrokenPipeError

        _attach_logger(cb, {"x": 1.0})
        cb.num_timesteps = 0
        cb._on_rollout_end()  # must not raise
        assert cb._pipe is None

    def test_training_end_closes_pipe(self):
        r_fd, w_fd = os.pipe()
        cb = _make_callback(fd=w_fd)
        cb._on_training_end()
        assert cb._pipe is None
        os.close(r_fd)  # cleanup read end
