"""Pure unit tests for the search-teacher CorrectionBuffer (ring eviction, sampling, tensor stacking)."""

import numpy as np
import pytest

from agents.training.teacher.buffer import Correction, CorrectionBuffer

_OBS_DIM, _N_ACT = 8, 11


def _corr(action=6, adv=0.5, step=0, opp="sentinel_0", pi_target=None):
    return Correction(
        obs=np.arange(_OBS_DIM, dtype=np.float32),
        action_mask=np.ones(_N_ACT, dtype=np.int8),
        better_action=action, advantage=adv, confirmed_value=1.0,
        step_produced=step, opponent=opp, pi_target=pi_target)


def _pi(action=6):
    """A π' one-hot on ``action`` over the 11 slots (a valid OPD target)."""
    p = np.zeros(_N_ACT, dtype=np.float32)
    p[action] = 1.0
    return p


def test_capacity_must_be_positive():
    with pytest.raises(ValueError):
        CorrectionBuffer(0)


def test_ring_evicts_oldest_at_capacity():
    b = CorrectionBuffer(3)
    for i in range(5):
        b.add(_corr(step=i))
    assert len(b) == 3
    steps = sorted(c.step_produced for c in b.sample(3))
    assert steps == [2, 3, 4]                      # the 2 oldest (0,1) were dropped


def test_sample_none_on_empty_and_clamps():
    b = CorrectionBuffer(10)
    assert b.sample(4) is None
    b.add(_corr()); b.add(_corr())
    s = b.sample(8)                                # asks for 8, only 2 present
    assert len(s) == 2


def test_sample_without_replacement_distinct():
    b = CorrectionBuffer(10, rng=np.random.default_rng(0))
    for i in range(6):
        b.add(_corr(step=i))
    s = b.sample(6)
    assert sorted(c.step_produced for c in s) == [0, 1, 2, 3, 4, 5]   # all distinct


def test_to_tensors_shapes_and_dict_wrapping():
    import torch as th
    b = CorrectionBuffer(10)
    for a, adv in [(6, 0.3), (2, 0.7), (8, 0.1)]:
        b.add(_corr(action=a, adv=adv))
    td = CorrectionBuffer.to_tensors(b.sample(3), device=th.device("cpu"))
    assert set(td["obs_dict"]) == {"observation", "action_mask"}
    assert td["obs_dict"]["observation"].shape == (3, _OBS_DIM)
    assert td["obs_dict"]["action_mask"].shape == (3, _N_ACT)
    assert td["better_action"].dtype == th.int64 and td["better_action"].shape == (3,)
    assert td["advantage"].dtype == th.float32 and td["advantage"].shape == (3,)
    assert td["confirmed_value"].shape == (3,)


def test_as_record_drops_arrays():
    r = _corr(action=7, adv=0.4, pi_target=_pi(7)).as_record()
    assert "obs" not in r and "action_mask" not in r and "pi_target" not in r
    assert r["better_action"] == 7 and r["advantage"] == 0.4


def test_pi_target_defaults_none_and_backward_compatible():
    """The OPD π' field is appended LAST with default None → a Correction built the old way (AWR-only)
    has pi_target None."""
    c = _corr(action=6)
    assert c.pi_target is None


def test_to_tensors_pi_target_all_present_stacks():
    """When EVERY sampled correction carries a π', to_tensors adds a [n, 11] pi_target tensor."""
    import torch as th
    b = CorrectionBuffer(10)
    for a in (6, 2, 8):
        b.add(_corr(action=a, pi_target=_pi(a)))
    td = CorrectionBuffer.to_tensors(b.sample(3), device=th.device("cpu"))
    assert td["pi_target"] is not None
    assert td["pi_target"].shape == (3, _N_ACT)
    assert td["pi_target"].dtype == th.float32
    assert th.allclose(td["pi_target"].sum(-1), th.ones(3))   # each row is a distribution


def test_to_tensors_pi_target_any_none_is_none_key():
    """If ANY sampled correction lacks a π' (a mixed / AWR-only buffer), the pi_target key is None so the
    KL loss None-guards (never a partial / mis-stacked batch)."""
    import torch as th
    b = CorrectionBuffer(10)
    b.add(_corr(action=6, pi_target=_pi(6)))
    b.add(_corr(action=2, pi_target=None))                    # one without a target
    td = CorrectionBuffer.to_tensors(b.sample(2), device=th.device("cpu"))
    assert td["pi_target"] is None
