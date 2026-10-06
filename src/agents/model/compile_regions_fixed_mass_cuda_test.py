"""F-XC-4 on the GPU: region R1 under `--belief-tokens fixed_mass` compiles to a FINITE gradient and passes
the real startup gate, CUDA, Inductor, fp32 'highest', at the production micro-batch (GPU tier: `slow`,
run with `GEN3AI_TEST_ALLOW_GPU=1` under a lease and `scripts/ops/gpu_lock.sh`).

Before `gen3_fm_index_max_v1` the Inductor-compiled R1 gradient was NaN on 41 of 257 parameters in 8 of 8
processes (`amax`'s backward over a RECOMPUTED 400-wide sweep: no element equalled the saved max once
Triton's FMA contraction rounded the recompute differently, so 0/0), and this gate FATAL'd
`NonFiniteGateArmError` — so this test FAILS on revert. Fresh weights: the gate's fresh path also runs
its seeded-perturbation rung (the trained-like state). The CPU half of the fix (blob never selects by
index; the value is `amax` bit for bit) is `damage_op_index_max_test.py`."""
from __future__ import annotations

import pytest
import torch

from agents.model import compile_control as cc
from agents.model import compile_regions as cr
from agents.model import compile_trainer as ct
from agents.model.extractor_compiles_test import _cuda_skip_reason

_skip_cuda = pytest.mark.skipif(_cuda_skip_reason() is not None, reason=_cuda_skip_reason() or "")
B = 2048


@pytest.fixture
def clean_dynamo():
    cc._reset_control_for_tests()
    torch._dynamo.reset()
    prev_prec = torch.get_float32_matmul_precision()
    try:
        yield
    finally:
        torch.set_float32_matmul_precision(prev_prec)
        cc._reset_control_for_tests()
        torch._dynamo.config.error_on_recompile = False
        torch._dynamo.reset()


@_skip_cuda
@pytest.mark.slow
def test_fixed_mass_R1_compiles_to_a_finite_gradient_and_passes_the_real_gate_on_cuda(clean_dynamo, monkeypatch):
    from agents.model import parity_probe as PP
    from agents.training import learner_golden as LG
    monkeypatch.setattr(PP, "PERTURB_SCALE", 0.0)          # a FRESH launch's weights (zero-init pointer head)
    model = LG.build_arm_learner("fixed_mass")
    assert model.policy.features_extractor.hypothesis_builder is not None      # PRECONDITION: the X5 arm
    dev = torch.device("cuda")
    model.policy.to(dev)
    model.device = dev
    torch.set_float32_matmul_precision("highest")
    model.batch_size = B
    cc.control().install()
    cr.install(model)
    try:
        assert cr.weights_regime(model) == "fresh"
        model.policy.set_training_mode(True)
        comp, eager = cr._r1_pair(model, cr._r1_args(model, cr.r1_batch(model, B)))
        names = [n for n, _ in ct.grad_parameters(model, model.policy.features_extractor)]
        assert ct.nonfinite_grad_params(eager, names) == []                      # PRECONDITION: eager finite
        assert ct.nonfinite_grad_params(comp, names) == [], "the compiled R1 gradient is NON-FINITE"
        rules = cr.gate_regions(model, batch_size=B, say=lambda _m: None)
        assert any(r.startswith("R1 [fresh weights]") for r in rules), rules
        # the `84f8569f` checks still bite on the REAL compiled graph: a planted NaN is named, a planted
        # 10 % gradient miscompile on one encoder parameter is FATAL (the graph is cached: no recompile)
        from agents.model.compile_regions_independence_test import _plant
        with monkeypatch.context() as mp:
            _plant(mp, nan=True)
            with pytest.raises(ct.NonFiniteGateArmError, match=r"COMPILED arm's gradient is NON-FINITE"):
                cr.gate_regions(model, batch_size=B, say=lambda _m: None)
        with monkeypatch.context() as mp:
            _plant(mp, grad_rel=0.1)
            with pytest.raises(ct.CompileTrainerError, match="DISAGREES with eager.*move_network"):
                cr.gate_regions(model, batch_size=B, say=lambda _m: None)
    finally:
        cr.uninstall(model)
