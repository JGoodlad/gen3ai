import os
from unittest.mock import MagicMock


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
    """The periodic checkpointer saves at TOTAL-ENV-STEP boundaries (`constants.checkpoint_due`),
    never on a call count. `model.save` is a MagicMock here, so these pin the trigger + bookkeeping."""

    def _make_cb(self, tmp_path, interval=100, save_path=None, start=0):
        cb = _TrackingCheckpointCallback(
            interval_env_steps=interval,
            save_path=str(save_path if save_path is not None else tmp_path),
            name_prefix="checkpoint",
        )
        mock_model = MagicMock()
        mock_model.num_timesteps = start
        cb.init_callback(mock_model)
        cb.on_training_start({}, {})
        return cb

    def _advance(self, cb, steps):
        """One collector call: the model's counter advances, then SB3's public `on_step` (which
        counts `n_calls` — the old trigger — and calls `_on_step`)."""
        cb.model.num_timesteps += steps
        return cb.on_step()

    def _saved_steps(self, cb):
        return [int(os.path.basename(c.args[0]).split("_")[1]) for c in cb.model.save.call_args_list]

    def test_writes_latest_txt_at_a_boundary(self, tmp_path):
        # save_path == run root (no checkpoints/ subdir): the recorded name is bare.
        cb = self._make_cb(tmp_path, interval=50000, start=49952)
        assert self._advance(cb, 48) is True
        assert (tmp_path / "latest.txt").read_text() == "checkpoint_50000_steps.zip\n"

    def test_writes_checkpoints_relative_latest_txt(self, tmp_path):
        # save_path == <run>/checkpoints/: latest.txt lands at the run root and records
        # the run-RELATIVE path into checkpoints/; _run_dir is derived as the parent.
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        cb = self._make_cb(tmp_path, interval=50000, save_path=ckpt_dir, start=49952)
        assert cb._run_dir == str(tmp_path)
        self._advance(cb, 48)
        assert not (ckpt_dir / "latest.txt").exists()
        assert (tmp_path / "latest.txt").read_text() == "checkpoints/checkpoint_50000_steps.zip\n"

    def test_no_write_between_boundaries(self, tmp_path):
        cb = self._make_cb(tmp_path, interval=50000, start=10000)
        self._advance(cb, 48)
        assert not (tmp_path / "latest.txt").exists()
        assert cb.model.save.call_count == 0

    def test_a_sync_stream_saves_where_the_old_call_count_did(self, tmp_path):
        """N = 48 from step 0: calls of exactly N steps hit every multiple of the interval
        exactly, so the boundary rule saves at the same steps as `n_calls % (interval / N)`."""
        cb = self._make_cb(tmp_path, interval=4800)          # = 100 calls of 48
        for _ in range(1000):
            self._advance(cb, 48)
        assert self._saved_steps(cb) == [4800 * k for k in range(1, 11)]

    def test_an_async_wave_stream_saves_at_total_step_boundaries(self, tmp_path):
        """`--async-rollout` fires one call per WAVE (< N envs). A call count would save every
        `interval / N` WAVES — early, by the mean wave fraction; the boundary rule saves once per
        interval of TOTAL env steps, at the first call at or past each boundary."""
        import random
        rng = random.Random(7)
        n, interval = 48, 4800
        cb = self._make_cb(tmp_path, interval=interval)
        while cb.model.num_timesteps < 10 * interval:
            self._advance(cb, rng.randint(1, n - 1))       # an uneven wave, always < N
        saved = self._saved_steps(cb)
        assert len(saved) == 10, saved
        for k, step in enumerate(saved, start=1):
            assert k * interval <= step < k * interval + n, (k, step)

    def test_a_restart_resumes_at_the_next_global_multiple(self, tmp_path):
        """The anchor is the step the process resumed at; boundaries stay GLOBAL multiples, so a
        resume at 7,000 saves at the first call at or past 9,600 — 9,640 in 48-step calls (not at
        7,000 + 4,800 = 11,800 as a per-process call count gave) — and never re-saves its own step."""
        cb = self._make_cb(tmp_path, interval=4800, start=7000)
        for _ in range(100):
            self._advance(cb, 48)
        assert self._saved_steps(cb) == [9640]

    def test_a_resume_exactly_on_a_boundary_does_not_resave_it(self, tmp_path):
        cb = self._make_cb(tmp_path, interval=4800, start=9600)
        self._advance(cb, 48)
        assert cb.model.save.call_count == 0

    def test_one_call_crossing_two_boundaries_saves_once(self, tmp_path):
        cb = self._make_cb(tmp_path, interval=100)
        self._advance(cb, 250)
        assert self._saved_steps(cb) == [250]
