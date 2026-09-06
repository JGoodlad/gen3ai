"""`RewardTermMetricsCallback` — the transport half of the `reward/` group.

The math is pinned in `reward_term_stats_test`; what is pinned HERE is the seam: the pull is an
`env_method` (so it is correct under `--async-rollout` as well as the sync barrier), a worker that
cannot answer costs nothing, and a diagnostic never takes down a run.
"""
from __future__ import annotations

import types

import pytest

from agents.training.reward_term_callback import RewardTermMetricsCallback


class _FakeLogger:
    def __init__(self):
        self.recorded = {}

    def record(self, key, value, exclude=None):
        self.recorded[key] = value


class _FakeVecEnv:
    def __init__(self, payloads, raises=False):
        self._payloads = payloads
        self._raises = raises
        self.calls = []

    def env_method(self, name, *args, **kwargs):
        self.calls.append(name)
        if self._raises:
            raise RuntimeError("worker pipe broke")
        return list(self._payloads)


def _cb(env, term_class=None):
    cb = RewardTermMetricsCallback(term_class=term_class or {"win_loss": "terminal"})
    # BaseCallback.training_env / .logger are getter-only properties over self.model.
    cb.model = types.SimpleNamespace(logger=_FakeLogger(), get_env=lambda: env)
    return cb


def _payload(n=1, total=30.0, sums=None, abs_=None, resid=0.0):
    sums = sums if sums is not None else {"win_loss": total}
    abs_ = abs_ if abs_ is not None else {k: abs(v) for k, v in sums.items()}
    return {"n": n, "total_sum": total, "total_abs_sum": abs(total),
            "residual_abs_sum": resid, "sum": sums, "abs": abs_}


class TestTheSeam:
    def test_it_PULLS_via_env_method_and_not_the_info_dicts(self):
        # The env_method route is what makes this correct under the async collector, whose
        # wave-batched step locals cannot say which buffer row a step landed on.
        env = _FakeVecEnv([_payload()])
        cb = _cb(env)
        cb.locals = {"infos": [{"reward_breakdown": {"win_loss": 999.0}}], "dones": [True]}
        cb._on_rollout_end()
        assert env.calls == ["drain_reward_terms"]
        assert cb.logger.recorded["reward/win_loss_mean"] == pytest.approx(30.0)

    def test_it_merges_every_worker(self):
        env = _FakeVecEnv([_payload(total=30.0), _payload(total=-30.0)])
        cb = _cb(env)
        cb._on_rollout_end()
        assert cb.logger.recorded["reward/n_decisions"] == 2.0
        assert cb.logger.recorded["reward/total_mean"] == pytest.approx(0.0)
        assert cb.logger.recorded["reward/total_abs_mean"] == pytest.approx(30.0)

    def test_a_worker_that_cannot_answer_is_skipped_not_counted(self):
        env = _FakeVecEnv([None, _payload()])
        cb = _cb(env)
        cb._on_rollout_end()
        assert cb.logger.recorded["reward/n_decisions"] == 1.0

    def test_an_env_with_no_reward_manager_publishes_NOTHING(self):
        # Absent curves, never zeros: a non-Gen3 env has no composition to report.
        cb = _cb(_FakeVecEnv([None, None]))
        cb._on_rollout_end()
        assert cb.logger.recorded == {}

    def test_a_broken_pipe_costs_the_reading_and_not_the_run(self):
        cb = _cb(_FakeVecEnv([], raises=True))
        cb._on_rollout_end()                       # must not raise
        assert cb.logger.recorded == {}

    def test_no_training_env_is_survivable(self):
        cb = _cb(None)
        cb._on_rollout_end()
        assert cb.logger.recorded == {}

    def test_every_key_carries_the_reward_prefix(self):
        env = _FakeVecEnv([_payload()])
        cb = _cb(env)
        cb._on_rollout_end()
        assert cb.logger.recorded
        assert all(k.startswith("reward/") for k in cb.logger.recorded)

    def test_on_step_is_a_no_op_that_keeps_the_run_going(self):
        assert _cb(_FakeVecEnv([]))._on_step() is True

    def test_the_residual_rides_out_as_the_GIGO_meter(self):
        env = _FakeVecEnv([_payload(resid=0.5)])
        cb = _cb(env)
        cb._on_rollout_end()
        assert cb.logger.recorded["reward/untracked_abs_mean"] == pytest.approx(0.5)


class TestTheRegistration:
    def test_the_launch_path_registers_it_with_this_runs_own_class_map(self):
        # The grouping and the startup composition line must read ONE declaration.
        from agents.training.reward_manager import RewardConfig, reward_class_composition
        from agents.training.reward_term_stats import term_class_map

        cmap = term_class_map(reward_class_composition(RewardConfig()))
        cb = RewardTermMetricsCallback(term_class=cmap)
        assert cb.term_class["win_loss"] == "terminal"
        assert cb.term_class["pbrs_material"] == "pbrs"
        assert cb.term_class["no_progress_tax"] == "bias"

    def test_it_is_in_the_base_callback_list_of_a_default_run(self):
        import inspect

        from main.train import callbacks as cb_mod

        src = inspect.getsource(cb_mod.build_callbacks)
        assert "RewardTermMetricsCallback" in src
        assert "reward_term_callback" in src
