"""The eval callbacks under ``--env-core rust`` (M5 Lane H): the cycle plays IN PROCESS on the model's
declared eval core — never on Python workers — publishes the workers' shard records, and the
UNCHANGED collect records every metric; a run without a declared eval core is REFUSED."""
from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from agents.training import eval_callback as ec
from agents.training.eval_callback import PerOpponentEvalCallback, eval_opponent_names
from agents.training.eval_sharding import ShardResult
from agents.training.rust_eval.launch import RustEvalUnavailable


class _Stats:
    def __init__(self, n):
        self.n = n

    def as_dict(self):
        return {"games": self.n, "units": self.n, "host_steps": 1, "trainee_decisions": 10 * self.n,
                "p2_policy_decisions": 0, "traces": 0, "near_ties": 0,
                "seconds": {k: 0.0 for k in ("load", "stage", "submit", "drain", "act", "core", "finish", "trace",
                                             "total")}, "lifecycle": {}}


class _FakeEvaluator:
    """Publishes one ShardResult per unit (win 3 of 4, 40 turns each) — what ``RustEvalCore`` publishes."""

    def __init__(self):
        self.calls = []

    def run_cycle(self, pool, run_dir, **kw):
        self.calls.append(kw)
        for u in pool.units:
            pool.publish(run_dir, ShardResult(
                unit_id=u.unit_id, item_key=u.item.key, worker_id=0, n_won=3 * u.n_games // 4,
                n_finished=u.n_games, sum_reward=0.0, n_episodes=u.n_games, sum_ep_len=40.0 * u.n_games,
                duration_sec=1.0, td_residuals=[-0.1, 0.2], traces_written=1, traces_won=0))
        return _Stats(len(pool.units))


def _cb(tmp_path, evaluator):
    cb = PerOpponentEvalCallback(model_dir=str(tmp_path), server_config=MagicMock(), env_core="rust",
                                 best_model_save_path=str(tmp_path / "best"), eval_games=8)
    cb.model = MagicMock()
    cb.model.save = lambda base: open(base + ".zip", "w").close()
    cb.model.gamma = 0.99
    cb.model._rust_collector = SimpleNamespace(evaluator=evaluator, cfg=SimpleNamespace(run_seed=5))
    cb._logger = MagicMock()
    cb.num_timesteps = 2_000_000
    cb._init_callback()
    return cb


def test_a_rust_cycle_runs_in_process_and_the_unchanged_collect_records_it(tmp_path, monkeypatch):
    def no_workers(*_a, **_k):
        raise AssertionError("--env-core rust spawned Python eval workers")

    monkeypatch.setattr(ec, "spawn_eval_workers", no_workers)
    ev = _FakeEvaluator()
    cb = _cb(tmp_path, ev)
    cb._on_step()
    assert cb._pending is None, "the blocking cycle is collected in the same step"
    assert len(ev.calls) == 1 and ev.calls[0]["step"] == 2_000_000
    recorded = {c.args[0]: c.args[1] for c in cb.logger.record.call_args_list}
    for name in eval_opponent_names():
        assert recorded[f"eval/win_rate_vs_{name}"] == pytest.approx(0.75)
    assert recorded["eval/win_rate_vs_bots"] == pytest.approx(0.75)
    assert "rust_eval/cycle_wall_s" in recorded
    man = json.loads((tmp_path / "eval_traces" / "step_2000000" / "eval_manifest.json").read_text())
    assert man["selection"] is not None, "the trace selection is recorded at collect, as on the Python path"
    assert (tmp_path / "best" / "best_model.zip").exists()


def test_a_rust_run_without_a_declared_eval_core_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(ec, "spawn_eval_workers", lambda *_a, **_k: [])
    cb = _cb(tmp_path, None)
    with pytest.raises(RustEvalUnavailable):
        cb._on_step()


def test_a_failed_rust_cycle_is_logged_as_missing_results_not_raised(tmp_path, monkeypatch):
    from agents.training.rust_eval.executor import EvalCoreError

    class _Broken:
        def run_cycle(self, *_a, **_k):
            raise EvalCoreError("a battle was QUARANTINED")

    sent = []
    monkeypatch.setattr(ec, "send_event", sent.append)
    cb = _cb(tmp_path, _Broken())
    cb._on_step()
    assert cb._pending is None
    assert any("Rust eval cycle failed" in s for s in sent)
    assert any("failed (no results)" in s for s in sent)
