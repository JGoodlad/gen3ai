"""Correctness tests for the non-barrier async rollout collector (`async_vec_env.py`).

Uses a REAL ``AsyncSubprocVecEnv`` over a tiny deterministic **Dict-obs** env (mirroring
``Gen3Env``'s ``{"observation", "action_mask"}`` space) + a fake deterministic policy, so the
collector's bookkeeping (per-env-column dict buffer fills, episode_starts, num_timesteps,
auto-reset obs/mask hand-off, GAE) is checked against an exactly-predictable trajectory —
independent of the async scheduling order. Also covers the drain-safe ``env_method`` (the desync
hazard when the eval callback calls ``env_method`` mid-collection while pipes hold un-recv'd step
results).

No server / bridge needed; spawns lightweight subprocess envs.
"""

import numpy as np
import gymnasium as gym
import torch as th
from gymnasium import spaces

from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer

from agents.training.async_vec_env import AsyncSubprocVecEnv, collect_rollouts_async


class _CounterDictEnv(gym.Env):
    """Deterministic Dict-obs env: obs = {"observation": [step], "action_mask": [1,1]}, reward
    1.0/step, terminates after ``ep_len`` steps. The mask rides in the obs (like Gen3Env)."""

    def __init__(self, ep_len=3):
        super().__init__()
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(low=0.0, high=1e4, shape=(1,), dtype=np.float32),
            "action_mask": spaces.Box(0, 1, shape=(2,), dtype=np.int8),
        })
        self.action_space = spaces.Discrete(2)
        self._ep_len = ep_len
        self._t = 0

    def action_masks(self):
        return np.ones(2, dtype=np.int8)

    def _obs(self):
        return {"observation": np.array([float(self._t)], dtype=np.float32),
                "action_mask": np.ones(2, dtype=np.int8)}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self._t += 1
        terminated = self._t >= self._ep_len
        return self._obs(), 1.0, terminated, False, {}


class _FakePolicy:
    """Deterministic stand-in: action 0, value 0, log_prob 0 (so trajectories are predictable)."""
    def set_training_mode(self, mode): pass

    @staticmethod
    def _n(obs_t):
        return next(iter(obs_t.values())).shape[0] if isinstance(obs_t, dict) else obs_t.shape[0]

    def __call__(self, obs_t, action_masks=None):
        n = self._n(obs_t)
        return th.zeros(n, dtype=th.long), th.zeros(n), th.zeros(n)

    def predict_values(self, obs_t):
        return th.zeros((self._n(obs_t), 1))


class _FakeModel:
    def __init__(self, env, last_obs):
        self.policy = _FakePolicy()
        self.device = "cpu"
        self.num_timesteps = 0
        self.gamma = 0.99
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self._last_obs = last_obs
        self._last_episode_starts = np.ones(env.num_envs, dtype=np.float32)

    def _update_info_buffer(self, infos, dones): pass


class _NullCallback:
    def on_rollout_start(self): pass
    def on_rollout_end(self): pass
    def update_locals(self, loc): pass
    def on_step(self): return True


def _buf(env, n_steps):
    return MaskableDictRolloutBuffer(n_steps, env.observation_space, env.action_space,
                                     device="cpu", gae_lambda=0.95, gamma=0.99, n_envs=env.num_envs)


def test_async_collect_fills_dict_buffer_with_correct_trajectory():
    n_envs, n_steps, ep_len = 2, 5, 3
    env = AsyncSubprocVecEnv([(lambda el=ep_len: _CounterDictEnv(el)) for _ in range(n_envs)])
    last_obs = env.reset()
    try:
        model = _FakeModel(env, last_obs)
        buf = _buf(env, n_steps)
        ok = collect_rollouts_async(model, env, _NullCallback(), buf, n_steps, use_masking=True)
        assert ok is True
        assert buf.full and buf.pos == n_steps
        assert model.num_timesteps == n_envs * n_steps

        for i in range(n_envs):
            # obs acted on cycles 0,1,2 then resets (ep_len=3); episode_starts mark the initial step
            # + the post-terminal reset; the mask is stored for every transition.
            np.testing.assert_array_equal(buf.observations["observation"][:, i, 0], [0, 1, 2, 0, 1])
            np.testing.assert_array_equal(buf.episode_starts[:, i], [1, 0, 0, 1, 0])
            np.testing.assert_array_equal(buf.rewards[:, i], [1, 1, 1, 1, 1])
            assert np.all(buf.action_masks[:, i] == 1.0)
        assert np.all(np.isfinite(buf.returns)) and np.all(np.isfinite(buf.advantages))
        np.testing.assert_array_equal(model._last_obs["observation"][:, 0], [2.0, 2.0])
    finally:
        env.close()


def test_async_collect_handles_uneven_episode_lengths():
    """Different ep_len per env exercises the async ready-subset + per-env auto-reset independently."""
    n_steps = 6
    env = AsyncSubprocVecEnv([(lambda el=el: _CounterDictEnv(el)) for el in (2, 4)])
    last_obs = env.reset()
    try:
        model = _FakeModel(env, last_obs)
        buf = _buf(env, n_steps)
        assert collect_rollouts_async(model, env, _NullCallback(), buf, n_steps, use_masking=True)
        # env0 (ep_len 2): obs 0,1 | 0,1 | 0,1 ; starts 1,0,1,0,1,0
        np.testing.assert_array_equal(buf.observations["observation"][:, 0, 0], [0, 1, 0, 1, 0, 1])
        np.testing.assert_array_equal(buf.episode_starts[:, 0], [1, 0, 1, 0, 1, 0])
        # env1 (ep_len 4): obs 0,1,2,3 | 0,1 ; starts 1,0,0,0,1,0
        np.testing.assert_array_equal(buf.observations["observation"][:, 1, 0], [0, 1, 2, 3, 0, 1])
        np.testing.assert_array_equal(buf.episode_starts[:, 1], [1, 0, 0, 0, 1, 0])
        assert model.num_timesteps == 2 * n_steps
    finally:
        env.close()


class _WinCounterDictEnv(_CounterDictEnv):
    """`_CounterDictEnv` that emits ``info["win_outcome"]`` on the terminal step — so the async
    collector's INLINE win-prob capture (`async_vec_env.py`, the production --async-rollout path)
    can be exercised. `MaskableAgentWrapper.step` sets this key live; here we stub it directly."""

    def __init__(self, ep_len=3, outcome=1.0):
        super().__init__(ep_len)
        self._outcome = outcome

    def step(self, action):
        obs, r, term, trunc, _ = super().step(action)
        return obs, r, term, trunc, ({"win_outcome": float(self._outcome)} if term else {})


def test_async_inline_captures_win_outcome_at_correct_row():
    """The async collector records each terminal's win_outcome into model._win_terminal_scratch at the
    env's JUST-WRITTEN (t, i) buffer row (the SYNC callback can't recover that row, so the collector
    owns it). Distinct per-env outcomes (1.0 vs 0.0) confirm no cross-env mixup; 0.0-vs-NaN confirms a
    captured LOSS is distinguished from an un-captured (in-progress) row."""
    n_envs, n_steps, ep_len = 2, 5, 3
    outcomes = [1.0, 0.0]
    env = AsyncSubprocVecEnv(
        [(lambda o=outcomes[k], el=ep_len: _WinCounterDictEnv(el, o)) for k in range(n_envs)])
    last_obs = env.reset()
    try:
        model = _FakeModel(env, last_obs)
        model._async_rollout = True
        model._win_terminal_scratch = np.full((n_steps, n_envs), np.nan, dtype=np.float32)
        buf = _buf(env, n_steps)
        assert collect_rollouts_async(model, env, _NullCallback(), buf, n_steps, use_masking=True)
        scr = model._win_terminal_scratch
        # ep_len=3, n_steps=5 → the done lands at buffer row 2 (episode_starts == [1,0,0,1,0]); the 2nd
        # episode (rows 3,4) is in progress → no terminal captured there.
        for i in range(n_envs):
            np.testing.assert_array_equal(buf.episode_starts[:, i], [1, 0, 0, 1, 0])
            assert scr[2, i] == outcomes[i], f"env{i} outcome must land at its done row"
            assert np.isnan(scr[[0, 1, 3, 4], i]).all(), f"env{i} non-terminal rows must stay NaN"
    finally:
        env.close()


def test_async_inline_capture_noop_when_winprob_off():
    """No _win_terminal_scratch on the model (win-prob off) → the inline capture is a clean no-op."""
    env = AsyncSubprocVecEnv([(lambda: _WinCounterDictEnv(3, 1.0))])
    last_obs = env.reset()
    try:
        model = _FakeModel(env, last_obs)  # no _win_terminal_scratch attr
        buf = _buf(env, 5)
        assert collect_rollouts_async(model, env, _NullCallback(), buf, 5, use_masking=True)
        assert not hasattr(model, "_win_terminal_scratch")
    finally:
        env.close()


def test_env_method_is_drain_safe_with_inflight_steps():
    """env_method mid-collection (eval callback's set_self_play_target etc.) must not desync against
    un-recv'd step results: AsyncSubprocVecEnv drains+stashes in-flight steps first."""
    env = AsyncSubprocVecEnv([(lambda: _CounterDictEnv(10)) for _ in range(3)])
    env.reset()
    try:
        env.send_step(0, 0)
        env.send_step(2, 0)            # envs 0 and 2 in flight (un-recv'd results on their pipes)
        assert env._inflight == {0, 2}

        masks = env.env_method("action_masks")   # must drain 0,2 into the stash, then RPC cleanly
        assert len(masks) == 3 and not env._inflight
        assert set(env._stash) == {0, 2}

        # The collector still reads the stashed step results (no desync, no double-read).
        r0 = env.recv_step(0)
        r2 = env.recv_step(2)
        assert isinstance(r0, tuple) and len(r0) == 5 and not env._stash
        assert isinstance(r2, tuple) and len(r2) == 5
    finally:
        env.close()
