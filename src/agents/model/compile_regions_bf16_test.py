"""THE bf16 PARITY RULE for region R1 (`gen3_r1_bf16_parity_v1`, T25 lever A; `compile_regions._r1_verdict_bf16`).

CPU, the K9 golden's production-surface learner with its trunk rounds set to bf16, R1 installed on dynamo's `eager`
backend (so the "compiled" arm computes exactly eager bf16, and the only compiled-vs-eager error is a planted one):

* the rule REFUSES while it has no measured bars (never a guess);
* with bars, a healthy bf16 R1 PASSES and names the bf16 rule; the fp32 REFERENCE arm really runs the fp32 base
  classes, runs R1's eager body once and the compiled route zero times, and leaves the bf16 classes in place;
* a COMPILED-arm error beyond K x eager bf16's own error on a parameter is the typed FATAL (the ENVELOPE);
* an eager bf16 arm that drifted from the fp32 function is the typed FATAL (the DRIFT bound);
* an fp32 learner is untouched (the fp32 rule's line, no reference arm);
* the committed bars equal what `derive_bf16_bars.py` derives from the committed GPU reads (when they exist).
"""
from __future__ import annotations

import json
import os

import pytest
import torch

from agents.model import compile_control as cc
from agents.model import compile_regions as cr
from agents.model import compile_trainer as ct
from agents.model import trunk_precision as TP

BATCH = 16
#: generous bars for the CPU tests: the rule's MECHANICS are under test here, not the GPU-derived values
_BARS = {"loss": 1.0, "grad_rel_l2": 5.0}
_K = {"loss": 4.0, "grad_rel_l2": 4.0, "param": 4.0}


@pytest.fixture
def learner():
    from agents.training import learner_golden as LG
    cc._reset_control_for_tests()
    torch._dynamo.reset()
    model = LG.build_learner()
    LG.load_buffer_into(model)
    TP.set_trunk_precision(model.policy, "bf16")
    try:
        yield model
    finally:
        cr.uninstall(model)
        cc._reset_control_for_tests()
        torch._dynamo.config.error_on_recompile = False
        torch._dynamo.reset()


def _install(model):
    cc.control().install()
    cr.install(model, backend="eager")


def _with_bars(monkeypatch, bars=None, k=None):
    monkeypatch.setattr(cr, "BF16_DRIFT_BARS", dict(bars or _BARS))
    monkeypatch.setattr(cr, "BF16_ENVELOPE_K", dict(k or _K))


def test_a_bf16_R1_is_REFUSED_while_the_rule_has_no_measured_bars(learner, monkeypatch):
    monkeypatch.setattr(cr, "BF16_DRIFT_BARS", None)
    _install(learner)
    with pytest.raises(ct.CompileTrainerError, match="no measured bars"):
        cr.gate_regions(learner, batch_size=BATCH, say=lambda _m: None)


def test_a_healthy_bf16_R1_PASSES_under_the_bf16_rule_and_the_reference_arm_is_the_fp32_function(
        learner, monkeypatch):
    _with_bars(monkeypatch)
    _install(learner)
    seen = []
    real = cr._r1_arm

    def spy(model, fn, args):
        seen.append(TP.trunk_precision_of(model.policy))
        return real(model, fn, args)
    monkeypatch.setattr(cr, "_r1_arm", spy)
    said = []
    rules = cr.gate_regions(learner, batch_size=BATCH, say=said.append)
    assert any("bf16 rule:" in r for r in rules), rules
    assert "bf16 trunk: the bf16 rule" in said[-1]
    # per rung: compiled (bf16), eager (bf16), the reference (fp32) — in that order
    assert seen[:3] == ["bf16", "bf16", "fp32"], seen
    assert TP.trunk_precision_of(learner.policy) == "bf16"      # the scope restored the classes


def _plant(scale_compiled=None, scale_eager=None):
    """An `_r1_arm` that multiplies the largest-gradient parameter's segment of the compiled (or eager bf16)
    arm by ``1 + scale``."""
    real = cr._r1_arm

    def arm(model, fn, args):
        out = real(model, fn, args)
        is_comp = fn is model._compiled_micro_step
        is_ref = TP.trunk_precision_of(model.policy) == "fp32"
        eps = scale_compiled if is_comp else (None if is_ref else scale_eager)
        if eps:
            g = out["grad"].clone()
            sizes = [int(x) for x in out["grad_sizes"].tolist()]
            big = max(range(len(sizes)), key=lambda i: float(torch.split(g, sizes)[i].norm()))
            start = sum(sizes[:big])
            g[start:start + sizes[big]] *= (1.0 + eps)
            out = {**out, "grad": g}
        return out
    return arm


def test_a_COMPILED_error_beyond_K_x_eager_bf16s_own_error_is_the_typed_FATAL(learner, monkeypatch):
    # K for the global gradient is set out of the way, so the PER-PARAMETER envelope is the rule under test
    _with_bars(monkeypatch, k={"loss": 4.0, "grad_rel_l2": 1e6, "param": 4.0})
    _install(learner)
    monkeypatch.setattr(cr, "_r1_arm", _plant(scale_compiled=0.0))
    cr.gate_regions(learner, batch_size=BATCH, say=lambda _m: None)          # nothing planted: passes
    monkeypatch.setattr(cr, "_r1_arm", _plant(scale_compiled=50.0))
    with pytest.raises(ct.CompileTrainerError, match="farther from the fp32 reference than 4 x eager bf16"):
        cr.gate_regions(learner, batch_size=BATCH, say=lambda _m: None)


def test_the_GLOBAL_gradient_envelope_bites_on_its_own(learner, monkeypatch):
    _with_bars(monkeypatch, k={"loss": 4.0, "grad_rel_l2": 4.0, "param": 1e6})
    _install(learner)
    monkeypatch.setattr(cr, "_r1_arm", _plant(scale_compiled=50.0))
    with pytest.raises(ct.CompileTrainerError, match="grad_rel_l2: compiled"):
        cr.gate_regions(learner, batch_size=BATCH, say=lambda _m: None)


def test_an_eager_bf16_arm_that_DRIFTED_from_the_fp32_function_is_the_typed_FATAL(learner, monkeypatch):
    _with_bars(monkeypatch, bars={"loss": 1.0, "grad_rel_l2": 0.5})
    _install(learner)
    # both bf16 arms carry the same 3x error on the largest parameter: no envelope breach, a DRIFT one
    monkeypatch.setattr(cr, "_r1_arm", _plant(scale_compiled=2.0, scale_eager=2.0))
    with pytest.raises(ct.CompileTrainerError, match="DRIFTED from the fp32"):
        cr.gate_regions(learner, batch_size=BATCH, say=lambda _m: None)


def test_the_reference_arm_must_be_EAGER(learner, monkeypatch):
    _with_bars(monkeypatch)
    _install(learner)
    from agents.training.instrumented_ppo import micro_step as MS
    compiled = learner._compiled_micro_step
    real = cr._r1_arm

    def arm(model, fn, args):              # a mis-wired reference that dispatches the compiled route
        if fn is MS.micro_step and TP.trunk_precision_of(model.policy) == "fp32":
            return real(model, compiled, args)
        return real(model, fn, args)
    monkeypatch.setattr(cr, "_r1_arm", arm)
    with pytest.raises(ct.GateArmsNotIndependentError, match="REFERENCE arm"):
        cr.gate_regions(learner, batch_size=BATCH, say=lambda _m: None)


def test_an_fp32_learner_keeps_the_fp32_rule_and_runs_no_reference_arm(learner, monkeypatch):
    TP.set_trunk_precision(learner.policy, "fp32")
    monkeypatch.setattr(cr, "BF16_DRIFT_BARS", None)          # the bf16 rule is never consulted
    _install(learner)
    calls = []
    monkeypatch.setattr(cr, "_r1_reference", lambda *a, **k: calls.append(1))
    said = []
    rules = cr.gate_regions(learner, batch_size=BATCH, say=said.append)
    assert not calls and not any("bf16" in r for r in rules)
    assert "parity PASS (fp32)" in said[-1]


_BARS_JSON = os.path.join(os.path.dirname(__file__), "..", "..", "..", "designs", "research_state", "measurements",
                          "perf_phase_2026-10-10", "results", "bf16_parity_bars.json")


def test_the_committed_bars_are_the_DERIVED_ones():
    """`BF16_DRIFT_BARS` / `BF16_ENVELOPE_K` equal `derive_bf16_bars.py` re-run on the committed GPU reads; with
    no committed derivation both must be None (the rule refuses)."""
    if not os.path.exists(_BARS_JSON):
        assert cr.BF16_DRIFT_BARS is None and cr.BF16_ENVELOPE_K is None
        return
    import importlib.util
    import sys
    rec = json.load(open(_BARS_JSON))
    here = os.path.dirname(_BARS_JSON)
    spec = importlib.util.spec_from_file_location(
        "derive_bf16_bars", os.path.join(here, "..", "scripts", "derive_bf16_bars.py"))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("derive_bf16_bars", mod)
    spec.loader.exec_module(mod)
    out = mod.derive(mod.load_reads([os.path.join(os.path.dirname(here), p) for p in rec["inputs"]]),
                     param_floor_frac=ct._PARAM_GRAD_FLOOR, loss_floor=cr.BF16_LOSS_FLOOR,
                     grad_floor=cr.BF16_GRAD_FLOOR, param_bar=dict(cr.R1_PARAM_BAR))
    assert out["BF16_DRIFT_BARS"] == rec["BF16_DRIFT_BARS"] == cr.BF16_DRIFT_BARS
    assert out["BF16_ENVELOPE_K"] == rec["BF16_ENVELOPE_K"] == cr.BF16_ENVELOPE_K
