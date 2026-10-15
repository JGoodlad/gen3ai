"""F-XC-4 on the GPU: region R1 under X5's fixed_mass hypothesis tokens compiles to a FINITE gradient and passes
the real startup gate, CUDA, Inductor, fp32 'highest', at the production micro-batch (GPU tier: `slow`, run with
`GEN3AI_TEST_ALLOW_GPU=1` under a lease and `scripts/ops/gpu_lock.sh` — `scripts/ops/verify_compile_gate_cuda.sh`).

Before `gen3_fm_index_max_v1` the Inductor-compiled R1 gradient was NaN on 41 of 257 parameters in 8 of 8
processes (`amax`'s backward over a RECOMPUTED 400-wide sweep: no element equalled the saved max once
Triton's FMA contraction rounded the recompute differently, so 0/0), and this gate FATAL'd
`NonFiniteGateArmError` — so this test FAILS on revert. Fresh weights: the gate's fresh path also runs
its seeded-perturbation rung (the trained-like state). The CPU half of the fix (blob never selects by
index; the value is `amax` bit for bit) is `damage_op_index_max_test.py`.

TWO GATES, TWO TESTS (2026-10-15). Since the bf16 trunk became production (`10b3a6b6`) the default learner is
bf16, and R1 is then judged by THE bf16 PARITY RULE, whose per-parameter resolution is K x that parameter's own
bf16 error — a planted 10 % miscompile on `move_network.0.weight` PASSED it on CUDA (the 2026-10-15 failure of this
file, which still built the default learner). So:

* the FP32 GATE's teeth are judged on an explicitly fp32 learner (`trunk_precision="fp32"`, the opt-out), where
  the 10 % plant must be FATAL as before;
* the bf16 GATE is judged for what it IS: its honest pass, a NaN named, and its RESOLUTION — the smallest planted
  miscompile it refuses, per representative parameter (`compile_regions_resolution_test`: computed from the gate's
  own arms, then confirmed by planting 0.5x (passes) and 2x (refused)); the record goes to
  `$GEN3AI_GATE_RESOLUTION_OUT` for `designs/training/compile_flags.md` "The bf16 gate's resolution"."""
from __future__ import annotations

import pytest
import torch

from agents.model import compile_control as cc
from agents.model import compile_regions as cr
from agents.model import compile_trainer as ct
from agents.model import trunk_precision as TP
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


def _fresh_cuda_learner(monkeypatch, trunk_precision):
    """The K9 golden learner at a FRESH launch's weights on CUDA, R1 installed (Inductor). ``trunk_precision``:
    "fp32" (the opt-out: the fp32 gate) or None (the production recipe: bf16)."""
    from agents.model import parity_probe as PP
    from agents.training import learner_golden as LG
    monkeypatch.setattr(PP, "PERTURB_SCALE", 0.0)          # a FRESH launch's weights (zero-init pointer head)
    model = LG.build_learner(trunk_precision=trunk_precision)
    assert model.policy.features_extractor.hypothesis_builder is not None      # PRECONDITION: the X5 arm
    dev = torch.device("cuda")
    model.policy.to(dev)
    model.device = dev
    torch.set_float32_matmul_precision("highest")
    model.batch_size = B
    cc.control().install()
    cr.install(model)
    return model


def _finite_compiled_gradient(model):
    model.policy.set_training_mode(True)
    comp, eager = cr._r1_pair(model, cr._r1_args(model, cr.r1_batch(model, B)))
    names = [n for n, _ in ct.grad_parameters(model, model.policy.features_extractor)]
    assert ct.nonfinite_grad_params(eager, names) == []                      # PRECONDITION: eager finite
    assert ct.nonfinite_grad_params(comp, names) == [], "the compiled R1 gradient is NON-FINITE"


@_skip_cuda
@pytest.mark.slow
def test_fixed_mass_R1_compiles_to_a_finite_gradient_and_passes_the_real_gate_on_cuda(clean_dynamo, monkeypatch):
    """THE fp32 GATE (the `--trunk-precision fp32` opt-out): finite, passes, and its `84f8569f` teeth bite."""
    model = _fresh_cuda_learner(monkeypatch, "fp32")
    assert TP.trunk_precision_of(model.policy) == "fp32"     # PRECONDITION: the fp32 gate's arithmetic
    try:
        assert cr.weights_regime(model) == "fresh"
        _finite_compiled_gradient(model)
        rules = cr.gate_regions(model, batch_size=B, say=lambda _m: None)
        assert any(r.startswith("R1 [fresh weights]") for r in rules), rules
        assert not any("bf16" in r for r in rules), rules
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


@_skip_cuda
@pytest.mark.slow
def test_fixed_mass_R1_under_the_bf16_rule_on_cuda_passes_names_a_NaN_and_has_its_measured_resolution(
        clean_dynamo, monkeypatch):
    """THE bf16 GATE (the production recipe): finite, passes under the bf16 rule, a NaN is named, and its
    resolution on the representative parameters is CONFIRMED by plants at 0.5x / 2x (the numbers are recorded,
    not pinned: they are bf16's rounding on this graph, `designs/training/compile_flags.md`)."""
    from agents.model import compile_regions_resolution_test as RT
    model = _fresh_cuda_learner(monkeypatch, None)
    assert TP.trunk_precision_of(model.policy) == "bf16"     # PRECONDITION: the production recipe is the bf16 trunk
    try:
        assert cr.weights_regime(model) == "fresh"
        _finite_compiled_gradient(model)
        rules = cr.gate_regions(model, batch_size=B, say=lambda _m: None)
        assert any(r.startswith("R1 [fresh weights] bf16 rule:") for r in rules), rules
        with monkeypatch.context() as mp:
            RT.plant(mp, RT.REPRESENTATIVE[0], nan=True)
            with pytest.raises(ct.NonFiniteGateArmError, match=r"COMPILED arm's gradient is NON-FINITE"):
                cr.gate_regions(model, batch_size=B, say=lambda _m: None)
        rec = RT.confirm_resolution(model, monkeypatch, B, RT.REPRESENTATIVE)
        rec["what"] = "fixed_mass K9 golden learner, FRESH weights, production bf16 recipe, CUDA Inductor, B=2048"
        RT.write_record("bf16_fresh", rec)
    finally:
        cr.uninstall(model)
