"""Regression test for the resume optimizer-state realignment guard
(`_validate_or_reset_optimizer_state`).

The bug it pins (gen3_nature_ev_belief_v1 / the v40 SpreadBelief reorder): SB3/torch save+load the
Adam optimizer state BY PARAMETER POSITION. A refactor that reorders a module's parameters between a
checkpoint save and a resume silently misassigns the saved per-param momentum to the WRONG params —
weights still load (name-keyed) so the arch check passes, then `AdamW.step()` crashes
("size of tensor a (128) must match b (5)") the moment a misassigned param of a different shape first
gets a gradient. The guard treats ANY param↔state shape mismatch as proof the whole state is
misaligned and resets the momentum (fresh zero-init), so the resume proceeds instead of crashing."""
import types

import torch
from torch import nn

from main.train_rl_agent import _validate_or_reset_optimizer_state


def _fake_model_with_populated_optimizer():
    """A tiny two-Linear policy whose AdamW has real, correctly-shaped momentum after one step."""
    policy = nn.Module()
    policy.a = nn.Linear(5, 128)    # weight [128, 5]
    policy.b = nn.Linear(128, 5)    # weight [5, 128]  (distinct shape — a reorder is shape-detectable)
    opt = torch.optim.AdamW(policy.parameters(), lr=1e-3)
    loss = policy.b(policy.a(torch.randn(4, 5))).sum()
    loss.backward()
    opt.step()                      # populates exp_avg / exp_avg_sq with correct shapes
    model = types.SimpleNamespace(policy=policy)
    model.policy.optimizer = opt
    return model, policy, opt


def test_aligned_state_is_left_untouched():
    # A normal resume: every momentum buffer matches its param → no reset.
    model, _policy, opt = _fake_model_with_populated_optimizer()
    n_before = len(opt.state)
    assert n_before == 4            # a.weight, a.bias, b.weight, b.bias
    _validate_or_reset_optimizer_state(model)
    assert len(opt.state) == n_before   # untouched — clean resume is unchanged


def test_misaligned_state_is_reset():
    # Simulate the reorder: give a.weight ([128,5]) the momentum SHAPE meant for b.weight ([5,128]).
    model, policy, opt = _fake_model_with_populated_optimizer()
    opt.state[policy.a.weight]["exp_avg"] = torch.zeros(5, 128)     # wrong shape for a.weight
    _validate_or_reset_optimizer_state(model)
    assert len(opt.state) == 0      # whole state dropped → AdamW reinitialises fresh momentum


def test_reset_preserves_lr_and_param_groups():
    # The reset must clear momentum ONLY — LR / param_groups (where the resume LR lives) stay intact.
    model, policy, opt = _fake_model_with_populated_optimizer()
    opt.param_groups[0]["lr"] = 7e-5
    opt.state[policy.b.bias]["exp_avg_sq"] = torch.zeros(128)       # wrong shape for b.bias ([5])
    _validate_or_reset_optimizer_state(model)
    assert len(opt.state) == 0
    assert opt.param_groups[0]["lr"] == 7e-5    # LR survives the momentum reset


def test_no_optimizer_is_noop():
    # An inference-only load (env=None) may have no optimizer — must not raise.
    _validate_or_reset_optimizer_state(types.SimpleNamespace(policy=types.SimpleNamespace(optimizer=None)))
    _validate_or_reset_optimizer_state(types.SimpleNamespace(policy=None))
