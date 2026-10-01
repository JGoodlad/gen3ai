"""The in-run parity canary (K6, `gen3_compile_canary_v1`): it passes on a healthy compiled learner,
FAILS (typed) on a compiled forward that drifted from eager, runs on its cadence only, and changes
nothing about training (RNG, `.grad`, training mode). CPU, the production policy surface, dynamo's
`eager` backend standing in for Inductor (the canary compares whatever is installed as `fe.forward`
against the class's eager forward)."""
from __future__ import annotations

import pytest
import torch

from agents.model.compile_canary import CompileCanary, CompileCanaryError
from agents.model.compile_trainer import CompileTrainerError


@pytest.fixture
def model():
    from agents.training.diagnostics_cadence_test import _real_gen3_ppo
    torch._dynamo.reset()
    m = _real_gen3_ppo(device="cpu")
    try:
        yield m
    finally:
        torch._dynamo.reset()


def _compile(fe):
    fe.forward = torch.compile(fe.forward, backend="eager")


def test_a_healthy_compiled_learner_passes_with_the_gradient_checked(model):
    fe = model.policy.features_extractor
    _compile(fe)
    can = CompileCanary(model, n_envs=2, batch_size=4, every=1, grad_every=1)
    out = can.after_update()
    assert out["compile/canary_ok"] == 1.0 and out["compile/canary_grad_checked"] == 1.0
    assert out["compile/canary_grad_cosine"] > 0.9999
    assert out["compile/canary_max_abs_legal_logprob"] < 1e-5


def test_a_compiled_forward_that_DRIFTED_from_eager_is_the_typed_fatal(model):
    """The canary's whole point: the installed (compiled) forward no longer computes the eager
    function. Fails if the canary stops comparing (or compares eager with itself)."""
    fe = model.policy.features_extractor

    def drifted(obs):
        pi, vf = type(fe).forward(fe, obs)
        return pi * 1.01, vf
    fe.forward = drifted
    can = CompileCanary(model, n_envs=2, batch_size=4, every=1, grad_every=1)
    with pytest.raises(CompileCanaryError, match="no longer equals eager") as ei:
        can.after_update()
    assert isinstance(ei.value, CompileTrainerError)          # -> FATAL_CONFIG, not restarted


def test_a_backward_only_drift_is_caught_on_a_gradient_canary(model):
    """A forward that matches but whose gradient is wrong: only the train-graph check sees it."""
    fe = model.policy.features_extractor

    class _ScaleGrad(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            return x.view_as(x)

        @staticmethod
        def backward(ctx, g):
            return g * 3.0

    def bad_backward(obs):
        pi, vf = type(fe).forward(fe, obs)
        return _ScaleGrad.apply(pi), vf
    fe.forward = bad_backward
    decision_only = CompileCanary(model, n_envs=2, batch_size=4, every=1, grad_every=10)
    decision_only.after_update()                              # the forward agrees: passes
    with_grad = CompileCanary(model, n_envs=2, batch_size=4, every=1, grad_every=1)
    with pytest.raises(CompileCanaryError):
        with_grad.after_update()


def test_the_cadence_runs_every_N_updates_and_writes_nothing_between(model):
    _compile(model.policy.features_extractor)
    can = CompileCanary(model, n_envs=2, batch_size=4, every=3, grad_every=2)
    got = [can.after_update() for _ in range(6)]
    assert [bool(g) for g in got] == [False, False, True, False, False, True]
    assert got[2]["compile/canary_grad_checked"] == 0.0 and got[5]["compile/canary_grad_checked"] == 1.0


def test_the_canary_changes_nothing_about_training(model):
    fe = model.policy.features_extractor
    _compile(fe)
    model.policy.set_training_mode(True)
    before = [p.detach().clone() for p in model.policy.parameters()]
    torch.manual_seed(7)
    expect = torch.rand(4)
    torch.manual_seed(7)
    CompileCanary(model, n_envs=2, batch_size=4, every=1, grad_every=1).after_update()
    assert torch.equal(torch.rand(4), expect)                 # RNG-neutral
    assert model.policy.training                              # mode restored
    assert all(p.grad is None for p in model.policy.parameters())
    assert all(torch.equal(a, b) for a, b in zip(before, model.policy.parameters()))
