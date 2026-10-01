"""Batch 1 through the COMPILED learner forward on CUDA (`gen3_batch1_eager_v1`; GPU tier: `slow`,
run with `GEN3AI_TEST_ALLOW_GPU=1` under `scripts/ops/gpu_lock.sh`, on `gen3ai_torch28`).

On torch 2.8.0+cu126 a batch-1 CUDA eval/no-grad graph of the production extractor does not lower
(Triton `CompilationError` — the K1 finding), and two learner-process paths reach batch 1: the
`--critic shaped` truncation value and the in-process FINAL EVALUATION. `compile_trainer` routes
every batch in `EAGER_BATCHES` to the eager forward. The first test FAILS if that routing is reverted
(the batch-1 call then compiles and, on 2.8, raises); the second is the CONTRACT that the bug is real
on 2.8 (if torch fixes it, it fails, and the routing can be retired)."""
from __future__ import annotations

import pytest
import torch

from agents.model import compile_control as cc
from agents.model import compile_trainer as ct
from agents.model.extractor_compiles_test import _cuda_skip_reason

_skip_cuda = pytest.mark.skipif(_cuda_skip_reason() is not None, reason=_cuda_skip_reason() or "")


def _compiled_model():
    """The production policy surface (`_real_gen3_ppo`: `main.fresh_checkpoint`'s kwargs) on CUDA,
    through the REAL `--compile-trainer` gate."""
    from agents.training.diagnostics_cadence_test import _real_gen3_ppo
    model = _real_gen3_ppo(device="cuda")
    ct.compile_trainer_extractor(model, True, batch=16)
    return model


def _final_eval_call(model):
    """What the in-process final evaluation does per decision: the policy's eval/no-grad forward
    at batch 1 (`policy.predict` -> the compiled `fe.forward`)."""
    obs = ct._prewarm_obs(model, 1)
    model.policy.set_training_mode(False)
    with torch.no_grad():
        return model.policy(obs, deterministic=True, action_masks=obs["action_mask"].cpu().numpy()
                            if "action_mask" in obs else None)


@pytest.fixture
def clean_dynamo():
    cc._reset_control_for_tests()
    torch._dynamo.reset()
    # SB3's seeding (`set_random_seed(using_cuda=True)` inside `_real_gen3_ppo`) sets
    # `cudnn.deterministic`; restore it with everything else this test touches.
    prev_det = torch.backends.cudnn.deterministic
    try:
        yield
    finally:
        torch.backends.cudnn.deterministic = prev_det
        cc._reset_control_for_tests()
        torch._dynamo.reset()


@_skip_cuda
@pytest.mark.slow
def test_batch_1_through_the_compiled_learner_forward_runs_eager_on_cuda(clean_dynamo):
    model = _compiled_model()
    fe = model.policy.features_extractor
    assert "forward" in vars(fe), "the learner is not compiled — the test is vacuous"
    actions, values, logp = _final_eval_call(model)       # the final eval's shape: must not raise
    assert actions.shape[0] == 1 and torch.isfinite(values).all() and torch.isfinite(logp).all()
    obs1 = ct._prewarm_obs(model, 1)
    with torch.no_grad():
        got = fe(obs1)
        ref = type(fe).forward(fe, obs1)
        two = fe(ct._prewarm_obs(model, 2))                # the compiled graph still serves batch 2
    assert all(torch.equal(a, b) for a, b in zip(got, ref))
    assert two[0].shape[0] == 2


@_skip_cuda
@pytest.mark.slow
@pytest.mark.skipif(not torch.__version__.startswith("2.8"), reason="the K1 finding is torch 2.8's")
def test_contract_torch_2_8_cannot_lower_the_batch_1_cuda_graph(clean_dynamo, monkeypatch):
    model = _compiled_model()
    monkeypatch.setattr(ct, "EAGER_BATCHES", frozenset())   # the revert
    with pytest.raises(Exception) as ei:
        _final_eval_call(model)
    assert "CompilationError" in repr(ei.value) or "constexpr" in str(ei.value), repr(ei.value)[:2000]
