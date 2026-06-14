import os
from unittest.mock import MagicMock, patch

import pytest

from main.train_rl_agent import _write_latest_txt, _TrackingCheckpointCallback


# ── _write_latest_txt ─────────────────────────────────────────────────────────

class TestWriteLatestTxt:
    def test_creates_file(self, tmp_path):
        _write_latest_txt(str(tmp_path), "checkpoint_100_steps.zip")
        latest = tmp_path / "latest.txt"
        assert latest.exists()
        assert latest.read_text() == "checkpoint_100_steps.zip\n"

    def test_atomic_overwrite(self, tmp_path):
        _write_latest_txt(str(tmp_path), "checkpoint_100_steps.zip")
        _write_latest_txt(str(tmp_path), "checkpoint_200_steps.zip")
        assert (tmp_path / "latest.txt").read_text() == "checkpoint_200_steps.zip\n"

    def test_cleans_up_tmp_file(self, tmp_path):
        _write_latest_txt(str(tmp_path), "checkpoint_100_steps.zip")
        assert not (tmp_path / "latest.txt.tmp").exists()

    def test_accepts_run_relative_path(self, tmp_path):
        # Periodic/forced checkpoints record a run-relative path (into checkpoints/);
        # latest.txt still lives at the run root and a reader joins it back with run_dir.
        _write_latest_txt(str(tmp_path), os.path.join("checkpoints", "checkpoint_100_steps.zip"))
        assert (tmp_path / "latest.txt").read_text() == "checkpoints/checkpoint_100_steps.zip\n"


# ── _TrackingCheckpointCallback ───────────────────────────────────────────────

class TestTrackingCheckpointCallback:
    def _make_cb(self, tmp_path, save_freq=10, save_path=None):
        cb = _TrackingCheckpointCallback(
            save_freq=save_freq,
            save_path=str(save_path if save_path is not None else tmp_path),
            name_prefix="checkpoint",
        )
        mock_model = MagicMock()
        mock_model.num_timesteps = 0
        cb.model = mock_model
        cb.n_calls = 0
        return cb

    def test_writes_latest_txt_on_save_freq(self, tmp_path):
        # save_path == run root (no checkpoints/ subdir): the recorded name is bare.
        cb = self._make_cb(tmp_path, save_freq=10)
        cb.num_timesteps = 50000
        cb.n_calls = 10

        with patch.object(type(cb).__bases__[0], "_on_step", return_value=True):
            cb._on_step()

        assert (tmp_path / "latest.txt").exists()
        assert (tmp_path / "latest.txt").read_text() == "checkpoint_50000_steps.zip\n"

    def test_writes_checkpoints_relative_latest_txt(self, tmp_path):
        # save_path == <run>/checkpoints/: latest.txt lands at the run root and records
        # the run-RELATIVE path into checkpoints/; _run_dir is derived as the parent.
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        cb = self._make_cb(tmp_path, save_freq=10, save_path=ckpt_dir)
        assert cb._run_dir == str(tmp_path)
        cb.num_timesteps = 50000
        cb.n_calls = 10

        with patch.object(type(cb).__bases__[0], "_on_step", return_value=True):
            cb._on_step()

        # latest.txt at the run root, NOT inside checkpoints/.
        assert not (ckpt_dir / "latest.txt").exists()
        assert (tmp_path / "latest.txt").read_text() == "checkpoints/checkpoint_50000_steps.zip\n"

    def test_no_write_between_saves(self, tmp_path):
        cb = self._make_cb(tmp_path, save_freq=10)
        cb.num_timesteps = 50000
        cb.n_calls = 5

        with patch.object(type(cb).__bases__[0], "_on_step", return_value=True):
            cb._on_step()

        assert not (tmp_path / "latest.txt").exists()
