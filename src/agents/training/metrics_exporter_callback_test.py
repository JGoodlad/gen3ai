import json
import os
from unittest.mock import MagicMock, patch

import pytest

import main.launcher.ipc as ipc
from agents.training.metrics_exporter_callback import MetricsExporterCallback


@pytest.fixture(autouse=True)
def reset_ipc():
    ipc._reset_for_testing()
    yield
    ipc._reset_for_testing()


def _make_callback() -> MetricsExporterCallback:
    return MetricsExporterCallback()


def _attach_logger(cb: MetricsExporterCallback, metrics: dict) -> None:
    mock_log = MagicMock()
    mock_log.name_to_value = metrics
    mock_model = MagicMock()
    mock_model.logger = mock_log
    mock_model.ep_info_buffer = None  # prevent MagicMock truthy fallthrough into safe_mean
    cb.model = mock_model


def _read_lines(r_fd: int) -> list[dict]:
    with os.fdopen(r_fd, "r") as f:
        return [json.loads(l) for l in f if l.strip()]


class TestMetricsExporterCallback:
    def test_rollout_end_noop_without_pipe(self):
        env = {k: v for k, v in os.environ.items() if k != "LAUNCHER_METRICS_FD"}
        with patch.dict(os.environ, env, clear=True):
            cb = _make_callback()
        _attach_logger(cb, {"x": 1.0})
        cb.num_timesteps = 0
        cb._on_rollout_end()  # must not raise

    def test_rollout_end_noop_with_negative_fd(self):
        with patch.dict(os.environ, {"LAUNCHER_METRICS_FD": "-1"}):
            cb = _make_callback()
        _attach_logger(cb, {"x": 1.0})
        cb.num_timesteps = 0
        cb._on_rollout_end()  # must not raise

    def test_writes_json_line_on_rollout_end(self):
        r_fd, w_fd = os.pipe()
        with patch.dict(os.environ, {"LAUNCHER_METRICS_FD": str(w_fd)}):
            cb = _make_callback()
            _attach_logger(cb, {"rollout/ep_rew_mean": -5.0, "time/fps": 1200.0})
            cb.num_timesteps = 50000
            cb._on_rollout_end()
            cb._on_training_end()  # closes pipe → EOF

        lines = _read_lines(r_fd)
        assert len(lines) == 1
        assert lines[0]["rollout/ep_rew_mean"] == pytest.approx(-5.0)
        assert lines[0]["time/fps"] == pytest.approx(1200.0)
        assert lines[0]["_step"] == 50000

    def test_skips_non_numeric_values(self):
        r_fd, w_fd = os.pipe()
        with patch.dict(os.environ, {"LAUNCHER_METRICS_FD": str(w_fd)}):
            cb = _make_callback()
            _attach_logger(cb, {"metric/num": 1.0, "metric/str": "hello"})
            cb.num_timesteps = 0
            cb._on_rollout_end()
            cb._on_training_end()

        lines = _read_lines(r_fd)
        assert "metric/str" not in lines[0]
        assert "metric/num" in lines[0]

    def test_payload_has_no_event_key(self):
        r_fd, w_fd = os.pipe()
        with patch.dict(os.environ, {"LAUNCHER_METRICS_FD": str(w_fd)}):
            cb = _make_callback()
            _attach_logger(cb, {"x": 1.0})
            cb.num_timesteps = 0
            cb._on_rollout_end()
            cb._on_training_end()

        lines = _read_lines(r_fd)
        assert "_event" not in lines[0]

    def test_training_end_closes_pipe(self):
        r_fd, w_fd = os.pipe()
        with patch.dict(os.environ, {"LAUNCHER_METRICS_FD": str(w_fd)}):
            cb = _make_callback()
            cb._on_training_end()
        # pipe write-end closed; reading should return EOF immediately
        with os.fdopen(r_fd, "r") as f:
            assert f.read() == ""

    def test_broken_pipe_does_not_raise(self):
        r_fd, w_fd = os.pipe()
        with patch.dict(os.environ, {"LAUNCHER_METRICS_FD": str(w_fd)}):
            cb = _make_callback()
            os.close(r_fd)  # close read end → next write raises BrokenPipeError
            _attach_logger(cb, {"x": 1.0})
            cb.num_timesteps = 0
            cb._on_rollout_end()  # must not raise
