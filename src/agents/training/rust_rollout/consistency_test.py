"""Pins for K9(b) BEHAVIOUR-POLICY CONSISTENCY and the STALENESS probe (M5 Lane G).

On a buffer whose stored behaviour log-probs are the policy's own (real obs rows, the committed
parity fixture): the probe passes (|Δ| at float rounding on CPU); a stored log-prob moved by 1e-3 on ONE current
row FAILS it (`fatal` raises, `warn` prints); rows of older versions are NOT held to the bar — they are
bucketed by age with their ratio / clip fraction / KL recorded; the row choice takes current rows first
and every present age; `off` does nothing."""
from __future__ import annotations

from collections import deque

import numpy as np
import pytest
import torch as th

from agents.training.rust_rollout import consistency as K


def _model(n_steps=16, n_envs=4):
    import gymnasium as gym
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.vec_env import DummyVecEnv

    from agents.model.compile_parity_fixture import load_parity_rows
    from agents.model.parity_probe import PERTURB_SCALE, perturb_
    from agents.model.policy import Gen3DualHeadMaskablePolicy
    from main.fresh_checkpoint import _production_policy_kwargs

    _args, layout, pk = _production_policy_kwargs()
    dim = layout["total_dim"]
    space = gym.spaces.Dict({"observation": gym.spaces.Box(-np.inf, np.inf, (dim,), np.float32),
                             "action_mask": gym.spaces.MultiBinary(11)})

    class _E(gym.Env):
        observation_space = space
        action_space = gym.spaces.Discrete(11)

        def reset(self, **kw):
            return {"observation": np.zeros(dim, np.float32), "action_mask": np.ones(11, np.int8)}, {}

        def step(self, a):
            return self.reset()[0], 0.0, False, False, {}

    th.manual_seed(0)
    m = MaskablePPO(Gen3DualHeadMaskablePolicy, DummyVecEnv([_E] * n_envs), n_steps=n_steps, batch_size=32,
                    policy_kwargs=pk, verbose=0, device="cpu", seed=0)
    perturb_(m.policy, seed=77, scale=PERTURB_SCALE)
    obs, mask = load_parity_rows(dim)
    n = n_steps * n_envs
    idx = np.arange(n) % len(obs)
    buf = m.rollout_buffer
    buf.reset()
    buf.observations["observation"][...] = obs[idx].reshape(n_steps, n_envs, dim)
    buf.observations["action_mask"][...] = mask[idx].reshape(n_steps, n_envs, 11)
    buf.action_masks[...] = mask[idx].reshape(n_steps, n_envs, 11).astype(np.float32)
    rng = np.random.default_rng(1)
    acts = np.array([rng.choice(np.flatnonzero(mask[i])) for i in idx])
    buf.actions[...] = acts.reshape(n_steps, n_envs, 1)
    m.policy.set_training_mode(False)
    with th.no_grad():
        _v, lp, _e = m.policy.evaluate_actions(
            {"observation": th.as_tensor(obs[idx]), "action_mask": th.as_tensor(mask[idx])},
            th.as_tensor(acts), action_masks=th.as_tensor(mask[idx]))
    buf.log_probs[...] = lp.numpy().reshape(n_steps, n_envs)
    buf.full = True
    m.ep_info_buffer = deque(maxlen=10)
    m._current_progress_remaining = 1.0
    m.num_timesteps = 1234
    return m


def test_off_does_nothing():
    m = _model()
    m.behaviour_check = "off"
    assert K.behaviour_probe(m) is None


def test_current_rows_match_on_cpu_to_float_rounding():
    m = _model()
    m.behaviour_check = "fatal"
    out = K.behaviour_probe(m)
    # the stored log-probs came from a 64-row batch, the probe runs 32: the CPU matmul's batch-size
    # rounding (~6e-7) is the whole difference, 170x under the bar
    assert out["behaviour/rows_current"] == 32 and out["behaviour/max_abs_dlogp_current"] < 5e-6
    assert out["staleness/probe_age_0_ratio_mean"] == pytest.approx(1.0, abs=1e-5)
    assert out["staleness/probe_age_0_clip_frac"] == 0.0


def test_one_current_row_off_by_1e_3_fails_fatal_and_warns_under_warn(capsys):
    m = _model()
    m.behaviour_check = "fatal"
    m.rollout_buffer.log_probs += 1e-3          # every stored behaviour log-prob moved: the probe sees it
    with pytest.raises(K.BehaviourMismatch, match="CURRENT policy"):
        K.behaviour_probe(m)
    m.behaviour_check = "warn"
    out = K.behaviour_probe(m)
    assert out["behaviour/max_abs_dlogp_current"] == pytest.approx(1e-3, rel=1e-3)
    assert "BEHAVIOUR-POLICY MISMATCH" in capsys.readouterr().out


def test_older_rows_are_bucketed_by_age_not_held_to_the_bar():
    m = _model()
    m.behaviour_check = "fatal"
    versions = np.full((16, 4), 5, dtype=np.int64)
    versions[8:, :] = 3                           # half the buffer played two versions ago
    m._rust_row_versions, m._rust_version = versions, 5
    m.rollout_buffer.log_probs[8:, :] -= 0.5      # those rows' behaviour policy differed
    out = K.behaviour_probe(m)
    assert out["behaviour/max_abs_dlogp_current"] < 5e-6
    assert out["staleness/probe_age_0_rows"] == 16 and out["staleness/probe_age_2_rows"] == 16
    assert out["staleness/probe_age_2_ratio_mean"] == pytest.approx(np.exp(0.5), rel=1e-5)
    assert out["staleness/probe_age_2_clip_frac"] == 1.0
    assert out["staleness/probe_age_2_approx_kl"] == pytest.approx(np.exp(0.5) - 1 - 0.5, rel=1e-5)


def test_the_row_choice_takes_current_rows_first_and_every_present_age():
    rng = np.random.default_rng(0)
    ages = np.array([0] * 10 + [1] * 50 + [4] * 50 + [20] * 3)
    pick = K.choose_rows(ages, 32, rng)
    assert len(pick) == 32 and len(set(pick.tolist())) == 32
    got = ages[pick]
    assert (got == 0).sum() == 10 and {1, 4, 20} <= set(got.tolist())
    assert K.choose_rows(np.zeros(5, int), 32, rng).tolist() == [0, 1, 2, 3, 4]


def test_the_probe_reads_the_precision_keyed_gate():
    """Lane G's probe and the python-core in-loop gate share ONE table (K9, M5 Lane K): fp32 = max < 1e-4;
    TF32 (`--matmul-precision high`) = the current rows' p99 < 3.6e-3 AND max < 0.071. A uniform shift
    between the fp32 bar and the TF32 p99 bar FAILS at fp32 and PASSES under TF32; above the p99 bar it
    fails under both. (The localized half — p99 clean, max fires — is `learner_gates_test`'s, on a
    production-size micro-batch; this probe's 32 rows make its p99 ~ its max.)"""
    assert K.BEHAVIOUR_GATES["highest"] == (K.GateCondition("max", 1e-4, 1),)
    (p99_name, p99_bar, p99_k), (max_name, max_bar, max_k) = K.BEHAVIOUR_GATES["high"]
    assert (p99_name, max_name, p99_k, max_k) == ("p99", "max", 1, K.TF32_MAX_PERSISTENCE)
    assert 1e-4 < p99_bar < max_bar
    prev = th.get_float32_matmul_precision()
    try:
        for shift, precision, fails in ((0.5 * p99_bar, "highest", True), (0.5 * p99_bar, "high", False),
                                        (2.0 * p99_bar, "high", True)):
            th.set_float32_matmul_precision(precision)
            m = _model()
            m.behaviour_check = "fatal"
            m.rollout_buffer.log_probs += shift
            if fails:
                with pytest.raises(K.BehaviourMismatch, match=precision):
                    K.behaviour_probe(m)
            else:
                assert K.behaviour_probe(m)["behaviour/bar_p99"] == p99_bar
    finally:
        th.set_float32_matmul_precision(prev)
