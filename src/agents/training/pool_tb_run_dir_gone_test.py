"""The POOL, TENSORBOARD and VALUE-SIDECAR writers refuse a gone run dir and never re-create it.

The self-play pool `mkdir -p`'d its dir at construction and wrote its summary, win rate, arch record and snapshots
through ``atomic_write``'s ``makedirs``; the TB logger and the value sidecar ``makedirs``'d their dirs. Each test
deletes the run dir (in a `run_archive`), calls the writer, and asserts the typed refusal (`RunDirGoneError`, exit
FATAL_CONFIG 3) and that nothing came back; each fails on a revert of its site.

TensorBoard, measured (tensorboard 2.20, torch 2.8, CPU): an OPEN ``SummaryWriter`` never re-creates its dir: with
the run dir deleted its ``flush`` raises ``FileNotFoundError`` (its writer thread dies the same way) -- untyped, a
CRASH the launcher would restart. A NEW ``SummaryWriter`` on the gone dir does re-create it. So the guard is at
construction (``configure(root=)``: ``ensure_subdir``) and at the flush every dump makes (``fatal_if_gone``: the
``FileNotFoundError`` under a gone run dir is raised as ``RunDirGoneError``).
"""
from __future__ import annotations

import shutil
from unittest.mock import MagicMock

import pytest

from agents.training.snapshot_pool import SnapshotPool
from agents.training.train_logger import configure
from agents.training.value_sidecar import SIDECAR_DIRNAME, ValueSidecarCallback, sidecar_path
from utils.run_dir import RunDirGoneError


@pytest.fixture
def run(run_archive):
    r = run_archive / "rb_pool_s1"
    r.mkdir()
    return r


def _version():
    v = MagicMock()
    v.to_json.return_value = "{}"
    return v


# ── the self-play pool ─────────────────────────────────────────────────────────────────────────────

def test_a_pool_constructed_under_a_gone_run_dir_refuses_and_creates_nothing(run):
    shutil.rmtree(run)
    with pytest.raises(RunDirGoneError):
        SnapshotPool(run / "snapshots", current_version=_version())
    assert not run.exists()


def test_a_live_pools_writes_into_a_gone_run_dir_refuse(run, tmp_path):
    pool = SnapshotPool(run / "snapshots", current_version=_version())
    assert (run / "snapshots").is_dir()
    src = tmp_path / "frozen.zip"
    src.write_bytes(b"weights")
    shutil.rmtree(run)
    with pytest.raises(RunDirGoneError):
        pool.persist_summary(win_rate_vs_bots=0.5, last_eval_step=1000)
    with pytest.raises(RunDirGoneError):
        pool.add_from_path(src, 1000)
    assert not run.exists(), "the pool re-created the deleted run dir"


# ── TensorBoard ────────────────────────────────────────────────────────────────────────────────────

def test_a_tb_logger_configured_under_a_gone_run_dir_refuses(run):
    shutil.rmtree(run)
    with pytest.raises(RunDirGoneError):
        configure(str(run / "tb"), ["tensorboard"], root=str(run))
    assert not run.exists()


def test_an_open_tb_logger_whose_run_dir_is_deleted_raises_typed_at_its_flush(run, capsys):
    logger = configure(str(run / "tb"), ["tensorboard"], root=str(run))
    try:
        logger.record("x", 1.0)
        logger.dump(1)
        assert any((run / "tb").iterdir())               # the control: a live run's events land
        shutil.rmtree(run)
        logger.record("x", 2.0)
        with pytest.raises(RunDirGoneError):
            logger.dump(2)
        assert not run.exists(), "the TB writer re-created the deleted run dir"
        assert "[RunDir] FATAL (tensorboard)" in capsys.readouterr().out
    finally:
        try:
            logger.close()
        except OSError:
            pass                                          # its writer's file is gone: closing it fails too


# ── the value sidecar ──────────────────────────────────────────────────────────────────────────────

def test_the_value_sidecar_header_refuses_a_gone_run_dir(run):
    cb = ValueSidecarCallback(str(run), fraction=1.0)
    shutil.rmtree(run)
    with pytest.raises(RunDirGoneError):
        cb._ensure_header()
    assert not run.exists()


def test_control_the_value_sidecar_header_lands_under_a_live_run(run):
    cb = ValueSidecarCallback(str(run), fraction=1.0)
    cb._ensure_header()
    assert (run / SIDECAR_DIRNAME).is_dir() and '"kind": "header"' in open(sidecar_path(str(run))).read()
