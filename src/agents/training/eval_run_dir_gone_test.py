"""The EVAL writers refuse a gone run dir and never re-create it; the run ends FATAL_CONFIG (3), never resurrected.

Every eval-side writer used to ``makedirs`` its target: both callbacks' ``best_model/`` and ``.eval_runs/`` at init and
the cycle's ``.eval_runs/step_<N>`` at launch, the cycle manifest's and the persisted snapshot's
``eval_traces/step_<N>``, ``best_model/`` for its config copy, the plan, the shard results and the core traces (each
through ``atomic_write``'s ``makedirs``), the SPRT candidate's and batch's dirs, and ``snapshot_ladder/``. A run dir
deleted or moved aside under a live run came back at its old name at the next eval. Each test deletes the run dir (in a
`run_archive`), calls the writer, and asserts the typed refusal and that nothing came back; each fails on a revert of
its site. The callbacks print ``[RunDir] FATAL`` (a launcher FATAL signature) and re-raise, the way 77abda32a's
periodic checkpoint does. No marker: stubs, no torch work, milliseconds.
"""
from __future__ import annotations

import shutil
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agents.training import eval_collect, eval_launch
from agents.training.eval_callback import EVAL_FREQ_STEPS, PerOpponentEvalCallback
from agents.training.eval_sharding import BOT, EvalItem, ShardedEvalPool
from agents.training.eval_sharding.results import ShardResult, write_shard_result
from agents.training.rust_eval.traces import write_core_trace
from agents.training.selfplay_callback import SelfPlayCallback
from agents.training.sprt_promotion import SprtPromotionMixin
from utils.run_dir import RunDirGoneError


@pytest.fixture
def run(run_archive):
    r = run_archive / "rb_eval_s1"
    r.mkdir()
    return r


def _gone(run):
    shutil.rmtree(run)
    return run


def _callbacks(run):
    return [PerOpponentEvalCallback(model_dir=str(run), best_model_save_path=str(run / "best_model"), eval_games=4),
            SelfPlayCallback(pool=MagicMock(), model_dir=str(run), best_model_save_path=str(run / "best_model"),
                             eval_games=4, snapshot_ladder_games=0)]


# ── the callbacks ──────────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("which", [0, 1], ids=["per_opponent", "selfplay"])
def test_eval_init_into_a_gone_run_dir_raises_and_creates_nothing(run, which, capsys):
    cb = _callbacks(run)[which]
    _gone(run)
    with pytest.raises(RunDirGoneError):
        cb._init_callback()
    assert not run.exists(), "the eval callback's init re-created the deleted run dir"
    assert "[RunDir] FATAL" in capsys.readouterr().out


@pytest.mark.parametrize("which", [0, 1], ids=["per_opponent", "selfplay"])
def test_an_eval_cycle_into_a_gone_run_dir_raises_and_creates_nothing(run, which, capsys, monkeypatch):
    cb = _callbacks(run)[which]
    cb._init_callback()                                   # a live run: its eval dirs exist
    assert (run / ".eval_runs").is_dir() and (run / "best_model").is_dir()
    _gone(run)
    cb.num_timesteps = cb._schedule()[0]                  # the first cadence boundary
    cb._last_eval_step = 0
    with pytest.raises(RunDirGoneError):
        cb._on_step()
    assert not run.exists(), "the eval cycle re-created the deleted run dir"
    assert "[RunDir] FATAL" in capsys.readouterr().out


def test_the_snapshot_ladder_spawn_into_a_gone_run_dir_is_not_swallowed_as_telemetry(run):
    cb = SelfPlayCallback(pool=MagicMock(), model_dir=str(run), eval_games=4, snapshot_ladder_games=10)
    _gone(run)
    with pytest.raises(RunDirGoneError):
        cb._spawn_snapshot_ladder_update(1000)
    assert not run.exists()


def test_control_a_live_cycle_dir_is_created_under_the_run(run):
    cb = _callbacks(run)[0]
    cb._init_callback()
    assert cb._eval_root == str(run / ".eval_runs")
    assert EVAL_FREQ_STEPS > 0


# ── the cycle's own writers ────────────────────────────────────────────────────────────────────────

def test_the_cycle_manifest_refuses_a_gone_run_dir(run):
    _gone(run)
    with pytest.raises(RunDirGoneError):
        eval_launch.write_eval_manifest(str(run), 1000, opponents=["random"], n_games=4)
    assert not run.exists()


def test_the_persisted_eval_snapshot_and_the_best_model_refuse_a_gone_run_dir(run, tmp_path):
    snap = tmp_path / "snapshot.zip"                      # the cycle's transient snapshot (elsewhere here)
    snap.write_bytes(b"weights")
    _gone(run)
    with pytest.raises(RunDirGoneError):
        eval_collect.persist_eval_snapshot(str(run), 1000, str(snap), keep_n=2)
    with pytest.raises(RunDirGoneError):
        eval_collect.store_best_model(str(run), 1000, str(snap), str(run / "best_model" / "best_model.zip"))
    eval_collect.copy_run_config_to_best_model(str(run), str(run / "best_model"))   # no config to copy: a no-op
    assert not run.exists()


def test_the_plan_and_a_shard_result_refuse_a_gone_cycle_dir(run):
    cycle = run / ".eval_runs" / "step_1000"
    cycle.mkdir(parents=True)
    _gone(run)
    with pytest.raises(RunDirGoneError):
        ShardedEvalPool([EvalItem("random", BOT, 4)], 2, step=1000).write_plan(str(cycle))
    with pytest.raises(RunDirGoneError):
        write_shard_result(str(cycle), ShardResult(unit_id="u0", item_key="random", worker_id=0, n_won=0,
                                                   n_finished=0, sum_reward=0.0, n_episodes=0, sum_ep_len=0.0,
                                                   duration_sec=0.0))
    assert not run.exists()


def test_a_core_trace_refuses_a_gone_forensic_root(run):
    root = run / "eval_traces" / "step_1000"
    root.mkdir(parents=True)
    _gone(run)
    with pytest.raises(RunDirGoneError):
        write_core_trace(str(root / "random" / "win_s0_1"), trace={}, step=1000, battle_id="b", result="win",
                         draw_kind=None, turns=1, trainee_username="t", obs=None, logp=None, values=None,
                         actions=None, masks=None, root=str(root))
    assert not run.exists()


def test_an_sprt_batch_into_a_gone_run_dir_refuses(run):
    host = SimpleNamespace(_model_dir=str(run), _eval_root=str(run / ".eval_runs"), _eval_shard_games=2)
    job = SimpleNamespace(step=1000, batch=0, state=SimpleNamespace(next_batch_pairs=lambda: 1),
                          plan_items=lambda n: [EvalItem("s", BOT, 2)])
    _gone(run)
    with pytest.raises(RunDirGoneError):
        SprtPromotionMixin._sprt_batch_pool(host, job)
    assert not run.exists()
