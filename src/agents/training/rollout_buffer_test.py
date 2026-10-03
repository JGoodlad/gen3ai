"""`gen3_owned_rollout_buffer_v1` — the owned rollout buffer IS sb3-contrib's ``MaskableDictRolloutBuffer`` for
the learner's layout (`agents/training/rollout_buffer.py`).

While sb3-contrib is installed (until stage 4 drops it) it is the ORACLE: the same writes into both buffers
leave the same arrays (dtype, shape, bytes), the same window GAE, and — from the same global numpy seed —
the same ``get()`` permutation and micro-batches, field for field. The K9 learner golden is the end-to-end
bar (one real ``train()`` on the owned buffer, bit-identical); this pins the buffer alone, and each
assertion fails if the owned buffer drifts from sb3's (e.g. the permutation drawn after the flatten, a
float64 mask, a missing swap).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch as th
from gymnasium import spaces

from agents.training.rollout_buffer import RolloutBuffer, RolloutBufferLayoutError, RolloutSamples

N_STEPS, N_ENVS, N_ACT = 6, 3, 11


def _spaces():
    obs = spaces.Dict({"observation": spaces.Box(-np.inf, np.inf, (5,), np.float32),
                       "action_mask": spaces.Box(0, 1, (N_ACT,), np.int8),
                       "flags": spaces.MultiBinary(4)})
    return obs, spaces.Discrete(N_ACT)


def _pair(gamma=0.97, lam=0.8):
    from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer

    obs, act = _spaces()
    kw = dict(device="cpu", gamma=gamma, gae_lambda=lam, n_envs=N_ENVS)
    return RolloutBuffer(N_STEPS, obs, act, **kw), MaskableDictRolloutBuffer(N_STEPS, obs, act, **kw)


def _fill(bufs, seed=0):
    rng = np.random.default_rng(seed)
    obs_space, _ = _spaces()
    for _t in range(N_STEPS):
        obs = {k: rng.random((N_ENVS, *sp.shape)).astype(sp.dtype) for k, sp in obs_space.spaces.items()}
        act = rng.integers(0, N_ACT, (N_ENVS, 1))
        rew = rng.standard_normal(N_ENVS).astype(np.float32)
        start = (rng.random(N_ENVS) < 0.3)
        val = th.as_tensor(rng.standard_normal(N_ENVS).astype(np.float32))
        lp = th.as_tensor(rng.standard_normal(N_ENVS).astype(np.float32))
        masks = (rng.random((N_ENVS, N_ACT)) < 0.7)
        for b in bufs:
            b.add(obs, act, rew, start, val, lp, action_masks=masks)
    last = th.as_tensor(rng.standard_normal(N_ENVS).astype(np.float32))
    dones = rng.random(N_ENVS) < 0.5
    for b in bufs:
        b.compute_returns_and_advantage(last_values=last, dones=dones)


def _arrays(b):
    out = {f"obs:{k}": v for k, v in b.observations.items()}
    out.update({k: getattr(b, k) for k in ("actions", "rewards", "returns", "episode_starts", "values",
                                           "log_probs", "advantages", "action_masks")})
    return out


def _same(a, b):
    assert a.keys() == b.keys()
    for k in a:
        assert a[k].dtype == b[k].dtype and a[k].shape == b[k].shape, k
        assert np.array_equal(a[k], b[k]), k


def test_a_fresh_buffer_has_sb3s_arrays_dtypes_and_shapes():
    ours, theirs = _pair()
    _same(_arrays(ours), _arrays(theirs))
    assert (ours.pos, ours.full, ours.mask_dims, ours.action_dim) == (theirs.pos, theirs.full, theirs.mask_dims, 1)


def test_the_same_writes_and_window_gae_leave_the_same_bytes():
    ours, theirs = _pair()
    _fill([ours, theirs])
    assert ours.full and theirs.full and ours.size() == theirs.size() == N_STEPS
    _same(_arrays(ours), _arrays(theirs))


@pytest.mark.parametrize("batch", [None, 4, 9])
def test_get_draws_the_same_permutation_and_serves_the_same_micro_batches(batch):
    ours, theirs = _pair()
    _fill([ours, theirs])
    for epoch in range(2):                                  # the second epoch reads the flattened arrays
        np.random.seed(100 + epoch)
        mine = list(ours.get(batch))
        np.random.seed(100 + epoch)
        ref = list(theirs.get(batch))
        assert len(mine) == len(ref)
        for m, r in zip(mine, ref):
            assert isinstance(m, RolloutSamples) and m._fields == r._fields
            for f in m._fields:
                if f == "observations":
                    assert m.observations.keys() == r.observations.keys()
                    for k in m.observations:
                        assert th.equal(m.observations[k], r.observations[k]) and \
                            m.observations[k].dtype == r.observations[k].dtype, k
                else:
                    assert th.equal(getattr(m, f), getattr(r, f)) and getattr(m, f).dtype == getattr(r, f).dtype, f
    _same(_arrays(ours), _arrays(theirs))                  # both flattened in place, identically


def test_reset_restores_the_unflattened_layout():
    ours, _ = _pair()
    _fill([ours])
    list(ours.get(4))
    ours.reset()
    assert ours.actions.shape == (N_STEPS, N_ENVS, 1) and not ours.generator_ready and ours.pos == 0


def test_a_layout_the_learner_never_uses_is_refused():
    obs, act = _spaces()
    with pytest.raises(RolloutBufferLayoutError, match="Dict"):
        RolloutBuffer(4, spaces.Box(0, 1, (3,), np.float32), act)
    with pytest.raises(RolloutBufferLayoutError, match="Discrete"):
        RolloutBuffer(4, obs, spaces.MultiDiscrete([2, 3]))
    with pytest.raises(RolloutBufferLayoutError, match="keys only"):
        RolloutBuffer(4, spaces.Dict({"d": spaces.Discrete(3)}), act)
