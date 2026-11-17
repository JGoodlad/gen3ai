"""The run ROOT is created once, at startup, and a same-run resume never re-creates it (`utils.run_dir`, unit E).

The trainer's directory setup runs on EVERY child start, a launcher restart included: a run dir deleted or moved
aside between two children came back at its old name as a half-run when the next child ``makedirs``'d it.
``create_run_root(resume=True)`` requires the dir instead, so the child exits FATAL_CONFIG (3) on its own, not only
because the launcher refuses the restart (ef6132069). The startup-only writers that seed a fork (the pool seed, the
TensorBoard inheritance) create only below an existing run dir. Each refusal test fails on a revert of its site;
the controls pin that a fresh run and a fork still get their dirs. No marker: no torch work, milliseconds.
"""
from __future__ import annotations

import argparse
import shutil

import pytest

from agents.training import pool_seed
from agents.training import tb_inherit
from main.exit_codes import TrainExitCode, exit_code_for
from main.train.run_io import _create_run_root
from utils.run_dir import RunDirGoneError, create_run_root


@pytest.fixture
def gone_run(run_archive):
    """A run dir in the test's own archive, checkpointed once, then deleted between two children."""
    run = run_archive / "rb_gone_s1"
    (run / "checkpoints").mkdir(parents=True)
    (run / "checkpoints" / "ckpt_1000_steps.zip").write_bytes(b"zip")
    shutil.rmtree(run)
    return run


def test_create_run_root_refuses_a_gone_root_on_a_resume(gone_run):
    with pytest.raises(RunDirGoneError):
        create_run_root(gone_run, resume=True)
    assert not gone_run.exists()


def test_control_create_run_root_makes_a_fresh_root_and_keeps_a_live_one(run_archive):
    fresh = run_archive / "rb_fresh_s1"
    assert create_run_root(fresh, resume=False) == str(fresh) and fresh.is_dir()
    assert create_run_root(fresh, resume=True) == str(fresh)         # a resume of a LIVE dir: nothing to refuse


def test_the_trainers_startup_on_a_same_run_resume_refuses_a_gone_run_dir(gone_run, capsys):
    """The child's own refusal: `--model <run>/checkpoints/<ckpt>.zip --run-dir <run>` after <run> was deleted."""
    args = argparse.Namespace(model=str(gone_run / "checkpoints" / "ckpt_1000_steps.zip"))
    with pytest.raises(RunDirGoneError) as exc:
        _create_run_root(args, str(gone_run))
    assert exit_code_for(exc.value) == int(TrainExitCode.FATAL_CONFIG)
    assert "[RunDir] FATAL (startup)" in capsys.readouterr().out
    assert not gone_run.exists()


@pytest.mark.parametrize("model", [None, "parent"])
def test_control_the_trainers_startup_creates_a_fresh_runs_and_a_forks_dir(run_archive, model):
    parent_ckpt = run_archive / "rb_parent_s1" / "checkpoints" / "ckpt_1000_steps.zip"
    args = argparse.Namespace(model=str(parent_ckpt) if model else None)
    run = run_archive / "rb_new_s1"
    assert _create_run_root(args, str(run)) == str(run) and run.is_dir()


def test_the_pool_seed_into_a_gone_fork_dir_refuses_and_creates_nothing(run_archive, gone_run):
    parent = run_archive / "rb_parent_s1"
    (parent / "snapshots").mkdir(parents=True)
    (parent / "snapshots" / "snapshot_1000.zip").write_bytes(b"zip")
    with pytest.raises(RunDirGoneError):
        pool_seed.seed_pool(str(parent), pool_seed.pool_dir_for(str(gone_run)))
    assert not gone_run.exists()


def test_tb_inheritance_into_a_gone_fork_dir_refuses_and_creates_nothing(gone_run):
    with pytest.raises(RunDirGoneError):
        tb_inherit._write_provenance(str(gone_run), str(gone_run.parent / "rb_parent_s1"),
                                     tb_inherit.InheritResult(True, "wrote"))
    assert not gone_run.exists()

