"""Region R1's parity gate under `--belief-tokens fixed_mass` (X5): the gate SEES a difference, names a
non-finite gradient as such, and never compares a thing with itself (F-XC-4, 2026-10-05).

THE FINDING (`designs/research_state/measurements/x5_fxc4_compile_gate_2026-10-05/`). Every fixed_mass
gate read compiled − eager loss exactly 0.00e+00, and one launch FATAL'd at "cosine 0.000000". Measured
on CUDA at HEAD: the two arms ARE independent and R1 IS compiled (one graph, no graph break); the
cosine 0 was a NON-FINITE compiled gradient (41 parameters NaN, eager finite) that `_cos` read as an
orthogonal one, and the bit-equal loss is the forward's own arithmetic, not shared state.

THE CLASS FIX, each pinned here on the production fixed_mass learner (CPU, dynamo's `eager` backend,
faults PLANTED on R1's compiled arm through `compile_regions._r1_arm`):

  * `gen3_gate_nonfinite_named_v1` — a NaN / inf gradient on either arm is `NonFiniteGateArmError`
    naming the parameters (fails on revert: the old gate said "cosine 0.000000");
  * `gen3_gate_independent_arms_v1` — `_r1_pair` proves from `region_calls`' counters that the compiled
    arm dispatched the compiled route once and ran R1's Python body zero times, and the eager arm the
    reverse; a compiled slot holding the eager function is REFUSED (fails on revert: such a gate
    compared eager with eager and PASSED);
  * the gate can SEE a planted miscompile under fixed_mass, on the loss and on one gradient."""
from __future__ import annotations

import pytest
import torch

from agents.model import compile_control as cc
from agents.model import compile_regions as cr
from agents.model import compile_trainer as ct

BATCH = 16
#: a parameter on the fixed_mass path whose gradient the golden rows judge (the encoder the hypothesis
#: pass shares; one of the 41 the CUDA measurement found non-finite)
PARAM = "features_extractor.pokemon_encoder.move_network.0.weight"


@pytest.fixture(scope="module")
def fixed_mass():
    """The fixed_mass K9 golden learner (production + `--belief-tokens fixed_mass`, its perturbed
    weights), R1 installed under dynamo's `eager` backend (compiled == eager arithmetic)."""
    from agents.training import learner_golden as LG
    cc._reset_control_for_tests()
    torch._dynamo.reset()
    m = LG.build_arm_learner("fixed_mass")
    LG.load_buffer_into(m, LG.arm_buffer("fixed_mass"))
    m.batch_size = BATCH
    assert m.policy.features_extractor.hypothesis_builder is not None      # PRECONDITION: the X5 arm
    cc.control().install()
    cr.install(m, backend="eager")
    try:
        yield m
    finally:
        cr.uninstall(m)
        cc._reset_control_for_tests()
        torch._dynamo.config.error_on_recompile = False
        torch._dynamo.reset()


def _plant(monkeypatch, *, loss_rel: float = 0.0, grad_rel: float = 0.0, nan: bool = False) -> None:
    """Corrupt R1's COMPILED arm only: the loss by ``loss_rel``, `PARAM`'s gradient by a relative
    ``grad_rel`` (alternating sign), or `PARAM`'s gradient to NaN."""
    real = cr._r1_arm

    def arm(model, fn, args):
        out = real(model, fn, args)
        if fn is not model._compiled_micro_step:
            return out
        names = [n for n, _ in ct.grad_parameters(model, model.policy.features_extractor)]
        sizes = [int(x) for x in out["grad_sizes"].tolist()]
        i = names.index(PARAM)
        start = sum(sizes[:i])
        g = out["grad"].clone()
        seg = g[start:start + sizes[i]]
        if nan:
            seg[0] = float("nan")
        elif grad_rel:
            sign = torch.where(torch.arange(seg.numel()) % 2 == 0, 1.0, -1.0).to(seg)
            g[start:start + sizes[i]] = seg + grad_rel * float(seg.norm()) * sign / float(sign.norm())
        return {**out, "grad": g, "loss": out["loss"] * (1.0 + loss_rel)}
    monkeypatch.setattr(cr, "_r1_arm", arm)


def test_the_fixed_mass_gate_PASSES_honest_arms_and_judges_the_encoder(fixed_mass):
    """PRECONDITION for every plant below: the honest gate passes, and `PARAM` is judged (its gradient
    is above the per-parameter floor on the golden rows)."""
    from agents.model.compile_gate_probe import per_param_grad_errors
    from agents.training.instrumented_ppo.micro_step import micro_step
    rules = cr.gate_regions(fixed_mass, batch_size=BATCH, say=lambda _m: None)
    assert rules and all("R1" in r for r in rules), rules
    fixed_mass.policy.set_training_mode(True)
    e = cr._r1_arm(fixed_mass, micro_step, cr._r1_args(fixed_mass, cr.r1_batch(fixed_mass, BATCH)))
    names = [n for n, _ in ct.grad_parameters(fixed_mass, fixed_mass.policy.features_extractor)]
    sizes = [int(x) for x in e["grad_sizes"].tolist()]
    judged = dict(per_param_grad_errors(e["grad"], e["grad"], sizes, floor_frac=ct._PARAM_GRAD_FLOOR))
    assert names.index(PARAM) in judged


def test_a_planted_LOSS_miscompile_under_fixed_mass_is_FATAL(fixed_mass, monkeypatch):
    """The gate CAN see a loss difference under fixed_mass (F-XC-4's 0.00e+00 was not blindness)."""
    _plant(monkeypatch, loss_rel=1e-3)
    with pytest.raises(ct.CompileTrainerError, match="LOSS disagrees"):
        cr.gate_regions(fixed_mass, batch_size=BATCH, say=lambda _m: None)


def test_a_planted_GRADIENT_miscompile_under_fixed_mass_is_FATAL(fixed_mass, monkeypatch):
    """A 10 % backward error on one encoder parameter — far under the global cosine's resolution —
    is FATAL by the per-parameter rule, named."""
    _plant(monkeypatch, grad_rel=0.1)
    with pytest.raises(ct.CompileTrainerError, match="DISAGREES with eager.*move_network"):
        cr.gate_regions(fixed_mass, batch_size=BATCH, say=lambda _m: None)


def test_a_NONFINITE_compiled_gradient_is_NAMED_not_read_as_a_cosine(fixed_mass, monkeypatch):
    """The fmA1b FATAL's class: a NaN in the compiled gradient. Fails on revert — the old gate raised
    "cosine 0.000000 < 0.9999", a direction claim about an undefined direction."""
    _plant(monkeypatch, nan=True)
    with pytest.raises(ct.NonFiniteGateArmError, match=r"COMPILED arm's gradient is NON-FINITE.*move_network"):
        cr.gate_regions(fixed_mass, batch_size=BATCH, say=lambda _m: None)


def test_the_CANARY_confirms_a_nonfinite_compiled_gradient_and_names_it(fixed_mass, monkeypatch):
    from agents.model.compile_canary import CompileCanary, CompileCanaryError
    _plant(monkeypatch, nan=True)
    with pytest.raises(CompileCanaryError, match="CONFIRMED.*NON-FINITE"):
        CompileCanary(fixed_mass, batch_size=BATCH, every=1, saver=lambda p: None).after_update()


def test_cos_of_a_nonfinite_gradient_is_NaN_never_zero():
    a = torch.tensor([1.0, 2.0])
    assert ct._cos(a, a) == pytest.approx(1.0)
    nan = ct._cos(torch.tensor([float("nan"), 1.0]), a)
    assert nan != nan                                            # NaN, not 0.0 (fails on revert)


def test_the_gate_REFUSES_a_compiled_slot_that_runs_EAGER(fixed_mass, monkeypatch):
    """The compiled slot holds a wrapper around the EAGER micro-step (it even carries the dispatcher's
    marker, so only what it EXECUTES can expose it): the gate would compare eager with eager — bit-equal
    loss and gradients — and PASS. Fails on revert."""
    from agents.training.instrumented_ppo.micro_step import micro_step
    real = fixed_mass._compiled_micro_step

    def eager_in_disguise(*a):
        return micro_step(*a)
    eager_in_disguise._gen3_compiled = real._gen3_compiled               # type: ignore[attr-defined]
    monkeypatch.setattr(fixed_mass, "_compiled_micro_step", eager_in_disguise)
    with pytest.raises(ct.CompileTrainerError, match="NOT independent"):
        cr.gate_regions(fixed_mass, batch_size=BATCH, say=lambda _m: None)


def test_the_gate_REFUSES_the_bare_eager_function_in_the_compiled_slot(fixed_mass, monkeypatch):
    """`micro_step` itself installed as the compiled route: it runs the eager body. Fails on revert."""
    from agents.training.instrumented_ppo.micro_step import micro_step
    monkeypatch.setattr(fixed_mass, "_compiled_micro_step", micro_step)
    with pytest.raises(ct.CompileTrainerError, match="NOT independent"):
        cr.gate_regions(fixed_mass, batch_size=BATCH, say=lambda _m: None)


def test_the_gate_REFUSES_an_eager_arm_that_dispatches_the_compiled_graph(fixed_mass, monkeypatch):
    """The other direction: the 'eager' reference secretly runs the compiled route."""
    real = cr._r1_arm

    def arm(model, fn, args):
        from agents.training.instrumented_ppo.micro_step import micro_step
        return real(model, model._compiled_micro_step if fn is micro_step else fn, args)
    monkeypatch.setattr(cr, "_r1_arm", arm)
    with pytest.raises(ct.CompileTrainerError, match="NOT independent"):
        cr.gate_regions(fixed_mass, batch_size=BATCH, say=lambda _m: None)
