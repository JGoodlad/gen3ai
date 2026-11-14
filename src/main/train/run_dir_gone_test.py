"""A save whose run dir is GONE refuses, never re-creates it, and the child exits FATAL_CONFIG (3), never 15.

The defect: `run_io.commit_checkpoint` wrote through `utils.atomic_io.atomic_write`, whose `os.makedirs` re-created
a missing run dir. A run dir deleted (or moved aside) under a live run came back as a half-run at its old name on
the next save: the SIGTERM / deferred-abort save (`final_model_interrupted.zip` + `latest.txt`), a periodic
checkpoint, or a forced one. The abort then exited INTERRUPTED (15), the code that claims a save.

Every test below fails on a revert of the part it names (`run_io.require_run_dir` in `commit_checkpoint`, the
abort's re-raise in `lifecycle._commit_abort`, `DeferredAbort.abort`'s mapped exit code). The controls pin the
first save of a FRESH run, which must still create `checkpoints/` under an existing run dir. No marker: stubs, no
torch work, milliseconds.
"""
from __future__ import annotations

import os
import shutil
import threading
import zipfile
from typing import List

import pytest

from main.exit_codes import TrainExitCode, exit_code_for
from main.launcher.run import _fatal_config_reason, _stop_was_unsaved
from main.train import lifecycle
from main.train.run_io import RunDirGoneError, _TrackingCheckpointCallback, commit_checkpoint


class _Model:
    """A learner stub: `save(path)` writes a real zip at the path it is handed (SB3 appends `.zip`)."""

    def __init__(self) -> None:
        self.num_timesteps = 2_000
        self.saves: List[str] = []

    def dump_logs(self) -> None:
        pass

    def save(self, path: str) -> None:
        self.saves.append(path)
        if not os.path.splitext(path)[1]:
            path += ".zip"
        with zipfile.ZipFile(path, "w") as z:
            z.writestr("data", "new")


def _gone_run(tmp_path):
    """A run dir under a `models/` archive, deleted while its run is live."""
    run = tmp_path / "models" / "rb_gone_s1"
    (run / "checkpoints").mkdir(parents=True)
    (run / "latest.txt").write_text("checkpoints/checkpoint_1000_steps.zip\n")
    shutil.rmtree(run)
    return run


@pytest.fixture
def quiet_sidecars(monkeypatch):
    monkeypatch.setattr(lifecycle, "save_model_snapshot", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "record_checkpoint", lambda *a, **k: None)
    monkeypatch.setattr(lifecycle, "_model_hparams", lambda m: {})


def _handlers(model, run_dir, exits: List[int]):
    return lifecycle._setup_signal_handlers(model, str(run_dir), threading.Event(), None, lambda: 1e-3, lambda: 1,
                                            exit_fn=exits.append, start_watchdog=False)


# ── the protocol ───────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name", ["final_model_interrupted", "checkpoints/checkpoint_2000_steps.zip",
                                  "final_model", "final_model_exception"])
def test_commit_checkpoint_refuses_a_gone_run_dir_and_creates_nothing(tmp_path, name):
    run = _gone_run(tmp_path)
    model = _Model()
    with pytest.raises(RunDirGoneError, match="no longer exists"):
        commit_checkpoint(model, str(run), str(run / name))
    assert not run.exists(), "the save re-created the deleted run dir"
    assert model.saves == [], "the model was saved although the run dir is gone"
    assert exit_code_for(RunDirGoneError("x")) == int(TrainExitCode.FATAL_CONFIG)


def test_control_a_fresh_runs_first_save_still_creates_checkpoints_under_its_run_dir(tmp_path):
    run = tmp_path / "models" / "rb_fresh_s1"
    run.mkdir(parents=True)                       # the trainer creates the run dir at startup, before any save
    final = commit_checkpoint(_Model(), str(run), str(run / "checkpoints" / "checkpoint_1000_steps.zip"))
    assert os.path.isfile(final) and (run / "latest.txt").read_text().strip() == \
        "checkpoints/checkpoint_1000_steps.zip"


# ── the SIGTERM / deferred-abort save ──────────────────────────────────────────────────────────────

def test_the_interrupted_save_into_a_gone_run_dir_exits_FATAL_CONFIG_and_writes_nothing(tmp_path, quiet_sidecars,
                                                                                       capsys):
    run = _gone_run(tmp_path)
    exits: List[int] = []
    ab = _handlers(_Model(), run, exits)
    try:
        with pytest.raises(RunDirGoneError):
            ab.abort("SIGTERM received")
    finally:
        ab.close()
    assert exits == [int(TrainExitCode.FATAL_CONFIG)], f"exit {exits}: 15 claims a save that never happened"
    assert not run.exists(), "the interrupted save re-created the deleted run dir"
    out = capsys.readouterr().out
    assert "[RunDir] FATAL" in out and "Checkpoint saved" not in out, out
    # the launcher's reading of that child: a quit is an UNSAVED stop, and the reason line surfaces
    lines = out.splitlines()
    assert _stop_was_unsaved(int(TrainExitCode.FATAL_CONFIG), lines)
    assert any("[RunDir] FATAL" in ln for ln in _fatal_config_reason(int(TrainExitCode.FATAL_CONFIG), lines))


def test_control_the_interrupted_save_into_a_live_run_dir_still_exits_15(tmp_path, quiet_sidecars):
    run = tmp_path / "models" / "rb_live_s1"
    run.mkdir(parents=True)
    exits: List[int] = []
    ab = _handlers(_Model(), run, exits)
    try:
        ab.abort("SIGTERM received")
    finally:
        ab.close()
    assert exits == [int(TrainExitCode.INTERRUPTED)]
    assert (run / "final_model_interrupted.zip").is_file()


def test_a_forced_checkpoint_into_a_gone_run_dir_stops_the_run(tmp_path, quiet_sidecars):
    """SIGUSR1's save used to `makedirs(<run>/checkpoints)` itself and report a failure as "training continues"."""
    run = _gone_run(tmp_path)
    ab = _handlers(_Model(), run, [])
    try:
        ab.request_checkpoint("SIGUSR1")
        with pytest.raises(RunDirGoneError):
            ab.safe_point("rollout_start")
    finally:
        ab.close()
    assert not run.exists()


# ── the periodic checkpoint ────────────────────────────────────────────────────────────────────────

def test_a_periodic_checkpoint_into_a_gone_run_dir_raises_and_creates_nothing(tmp_path, capsys):
    run = tmp_path / "models" / "rb_periodic_s1"
    cb = _TrackingCheckpointCallback(interval_env_steps=1_000, save_path=str(run / "checkpoints"))
    model = _Model()
    cb.model = model
    cb._last_step = 1_000
    shutil.rmtree(run, ignore_errors=True)        # (never created here; deleted under a live run in the field)
    with pytest.raises(RunDirGoneError):
        cb._on_step()
    assert not run.exists(), "the periodic save re-created the deleted run dir"
    assert "[RunDir] FATAL" in capsys.readouterr().out
