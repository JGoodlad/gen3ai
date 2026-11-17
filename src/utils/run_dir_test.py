"""The run's CHOKE-POINT writers refuse a gone run dir and never re-create it (`utils.run_dir`).

The defect class: a writer that ``mkdir -p``\\ s its target re-creates a run dir deleted or moved aside under a live run,
resurrecting it at its old name as a half-run. `utils.atomic_io.atomic_write` did (every metadata, pool and eval-results
write goes through it), and so did `save_model_snapshot`, `record_eval_results`, `record_team_win_rates`, the team
win-rate instrument and the checkpointer's ``checkpoints/`` mkdir. Each test deletes the run dir (in a `run_archive`),
calls the writer, and asserts the typed refusal (`RunDirGoneError`, exit FATAL_CONFIG 3) and that nothing came back;
each fails on a revert of the guard it names. The controls pin that a LIVE run dir still gets its subdirectories.
No marker: no torch work, milliseconds.
"""
from __future__ import annotations

import os
import shutil

import pytest

from agents.model import snapshot
from agents.training.team_winrate_callback import TeamWinRateCallback
from main.exit_codes import TrainExitCode, exit_code_for
from main.train.run_io import _TrackingCheckpointCallback
from utils import atomic_io
from utils.run_dir import RunDirGoneError, ensure_subdir, require_run_dir


@pytest.fixture
def gone_run(run_archive):
    """A run dir in the test's own archive, deleted while its run is live."""
    run = run_archive / "rb_gone_s1"
    (run / "checkpoints").mkdir(parents=True)
    (run / "metadata.json").write_text("{}")
    shutil.rmtree(run)
    return run


def test_run_dir_gone_maps_to_FATAL_CONFIG():
    assert exit_code_for(RunDirGoneError("x")) == int(TrainExitCode.FATAL_CONFIG)


# ── the primitives ─────────────────────────────────────────────────────────────────────────────────

def test_require_and_ensure_subdir_refuse_a_gone_root(gone_run):
    with pytest.raises(RunDirGoneError, match="no longer exists"):
        require_run_dir(gone_run)
    with pytest.raises(RunDirGoneError):
        ensure_subdir(gone_run, "best_model/sub")
    assert not gone_run.exists()


def test_ensure_subdir_never_recreates_a_root_that_vanishes_between_levels(run_archive, monkeypatch):
    """The check-then-create race: the root goes AFTER the existence check, before the first level is made."""
    run = run_archive / "rb_race_s1"
    run.mkdir()
    real_mkdir = os.mkdir

    def _mkdir_after_rmtree(path, *a, **k):
        shutil.rmtree(run, ignore_errors=True)
        return real_mkdir(path, *a, **k)

    monkeypatch.setattr(os, "mkdir", _mkdir_after_rmtree)
    with pytest.raises(RunDirGoneError):
        ensure_subdir(run, "a/b")
    assert not run.exists()


def test_ensure_subdir_refuses_a_path_outside_the_root(run_archive):
    with pytest.raises(ValueError):
        ensure_subdir(run_archive, "../escape")


def test_control_ensure_subdir_creates_nested_levels_under_a_live_root(run_archive):
    run = run_archive / "rb_live_s1"
    run.mkdir()
    assert ensure_subdir(run, "a/b") == str(run / "a" / "b") and (run / "a" / "b").is_dir()
    assert ensure_subdir(run, "a/b") == str(run / "a" / "b")         # idempotent


# ── atomic_write(root=) ────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("rel", ["metadata.json", "best_model/model_config.json", "pool/a/b.json"])
def test_atomic_write_with_root_refuses_a_gone_run_dir(gone_run, rel):
    with pytest.raises(RunDirGoneError):
        atomic_io.atomic_write_json(gone_run / rel, {"x": 1}, root=gone_run)
    assert not gone_run.exists(), "atomic_write re-created the deleted run dir"


def test_atomic_write_with_root_refuses_a_path_outside_its_root(run_archive):
    run = run_archive / "rb_live_s1"
    run.mkdir()
    with pytest.raises(ValueError):
        atomic_io.atomic_write_text(run_archive / "elsewhere" / "x.txt", "x", root=run)
    assert not (run_archive / "elsewhere").exists()


def test_control_atomic_write_with_root_creates_subdirs_under_a_live_run(run_archive):
    run = run_archive / "rb_live_s1"
    run.mkdir()
    atomic_io.atomic_write_text(run / "best_model" / "x.txt", "ok", root=run)
    assert (run / "best_model" / "x.txt").read_text() == "ok"


# ── the snapshot writers ───────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("write", [
    pytest.param(lambda d: snapshot.save_model_snapshot(d, object()), id="save_model_snapshot"),
    pytest.param(lambda d: snapshot.record_eval_results(d, 1000, {"win_rate": 0.5}), id="record_eval_results"),
    pytest.param(lambda d: snapshot.record_team_win_rates(d, {"teams": {}}), id="record_team_win_rates"),
    pytest.param(lambda d: snapshot.record_snapshot_in_history(d, "checkpoint_1_steps.zip", 1e-3, 1),
                 id="record_snapshot_in_history"),
])
def test_snapshot_writers_refuse_a_gone_run_dir(gone_run, write):
    with pytest.raises(RunDirGoneError):
        write(str(gone_run))
    assert not gone_run.exists(), "the snapshot writer re-created the deleted run dir"


def test_control_record_eval_results_writes_into_a_live_run_dir(run_archive):
    run = run_archive / "rb_live_s1"
    run.mkdir()
    snapshot.record_eval_results(str(run), 1000, {"win_rate": 0.5})
    assert '"latest_eval"' in (run / "metadata.json").read_text()


# ── the team win-rate instrument and the checkpointer ─────────────────────────────────────────────

def test_team_winrate_persist_into_a_gone_run_dir_raises_and_creates_nothing(gone_run, capsys):
    """It used to `makedirs(run_dir)` and swallow every failure as a non-fatal instrument error."""
    cb = TeamWinRateCallback(run_dir=gone_run)
    cb.num_timesteps = 1000
    with pytest.raises(RunDirGoneError):
        cb._persist()
    assert not gone_run.exists()
    assert "[RunDir] FATAL" in capsys.readouterr().out


def test_checkpointer_init_refuses_a_gone_run_dir(gone_run):
    cb = _TrackingCheckpointCallback(interval_env_steps=1000, save_path=str(gone_run / "checkpoints"))
    with pytest.raises(RunDirGoneError):
        cb._init_callback()
    assert not gone_run.exists()


def test_control_checkpointer_init_creates_checkpoints_under_a_live_run(run_archive):
    run = run_archive / "rb_live_s1"
    run.mkdir()
    _TrackingCheckpointCallback(interval_env_steps=1000, save_path=str(run / "checkpoints"))._init_callback()
    assert (run / "checkpoints").is_dir()
