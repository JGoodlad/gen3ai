"""Region R1's UNMOVED zero-init parameters (`gen3_r1_unmoved_param_v1`, 2026-10-04).

THE CLASS. `weights_regime` classifies the MODEL; gradient conditioning is a property of each
parameter. Under `--oracle-reveal` the blob arm's species belief labels are all PAD, so
`belief_head.species_head` (zero-init) receives no gradient in training and stays bit-exactly 0.0
while the rest of the model trains — and the canary's golden rows still supervise it, so its gradient
there is the FRESH-weights one (healthy compiled-vs-eager 1.0-2.5e-2, inside CPU eager fp32's own
error against float64). Judged at the TRAINED bar (9.88e-3) it FATAL'd `rb_x5ab_oracle_sp_s1001` at
update 10 (1.40e-2; `designs/research_state/measurements/oracle_canary_2026-10-04/`).

THE RULE. An unmoved parameter (every element == 0.0 — categorical, no tolerance) is judged at the
FRESH bar on the live weights, and again on a rung where ONLY the unmoved parameters are moved off zero
(name-keyed seeded noise, restored bit-exactly), with EVERY parameter at the TRAINED bar.

CPU, the production-surface learner (the K9 golden's perturbed weights: `weights_regime` == trained),
dynamo's `eager` backend (compiled == eager), the species head zeroed to stand in for the oracle's;
faults are PLANTED on R1's compiled arm through `compile_regions._r1_arm`:

  * NOISE — a relative error BETWEEN the two bars on the species head's gradient, present only while
    that head is exactly zero (fresh-weights conditioning, gone once it moves);
  * MISCOMPILE — the same relative error on every call, whatever the weights.

The noise must PASS (fails on revert: the old rule judged the head at the trained bar) and the
miscompile must be FATAL (fails if the unmoved rung is dropped, or the head simply excluded)."""
from __future__ import annotations

import pytest
import torch

from agents.model import compile_control as cc
from agents.model import compile_regions as cr
from agents.model import compile_trainer as ct

BATCH = 16
SP = "features_extractor.belief_head.species_head.weight"
#: between the trained bar (9.88e-3) and the fresh bar (0.1004): absorbed only by the fresh bar
EPS = 0.5 * (cr.R1_PARAM_BAR["trained"] + cr.R1_PARAM_BAR["fresh"])


@pytest.fixture
def oracle_like(tmp_path):
    """The golden learner, R1 installed under dynamo's `eager` backend, with the species head's weight
    and bias zeroed — the oracle run's state (`final_model_canary_fatal.zip`: both bit-exactly 0.0)."""
    from agents.training import learner_golden as LG
    cc._reset_control_for_tests()
    torch._dynamo.reset()
    m = LG.build_learner()
    LG.load_buffer_into(m)
    m.behaviour_dump_dir = str(tmp_path)
    head = m.policy.features_extractor.belief_head.species_head
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
    cc.control().install()
    cr.install(m, backend="eager")
    try:
        yield m
    finally:
        cr.uninstall(m)
        cc._reset_control_for_tests()
        torch._dynamo.config.error_on_recompile = False
        torch._dynamo.reset()


def _plant(monkeypatch, *, only_while_zero: bool) -> None:
    """Add a relative error of exactly ``EPS`` to the species head's gradient on R1's COMPILED arm —
    only while that head is bit-exactly zero (NOISE), or always (MISCOMPILE)."""
    real = cr._r1_arm

    def arm(model, fn, args):
        out = real(model, fn, args)
        head = model.policy.features_extractor.belief_head.species_head.weight
        if fn is model._compiled_micro_step and (bool(head.detach().any()) is False or not only_while_zero):
            names = [n for n, _ in ct.grad_parameters(model, model.policy.features_extractor)]
            sizes = [int(x) for x in out["grad_sizes"].tolist()]
            i = names.index(SP)
            start = sum(sizes[:i])
            g = out["grad"].clone()
            seg = g[start:start + sizes[i]]
            sign = torch.where(torch.arange(seg.numel()) % 2 == 0, 1.0, -1.0).to(seg)
            g[start:start + sizes[i]] = seg + EPS * float(seg.norm()) * sign / float(sign.norm())
            out = {**out, "grad": g}
        return out
    monkeypatch.setattr(cr, "_r1_arm", arm)


def _judged(model) -> bool:
    """PRECONDITION: the golden rows put the zeroed head's gradient above the per-parameter floor (it
    is judged at all), and the model's regime is TRAINED."""
    from agents.model.compile_gate_probe import per_param_grad_errors
    from agents.training.instrumented_ppo.micro_step import micro_step
    model.policy.set_training_mode(True)
    e = cr._r1_arm(model, micro_step, cr._r1_args(model, cr.r1_batch(model, BATCH)))
    names = [n for n, _ in ct.grad_parameters(model, model.policy.features_extractor)]
    sizes = [int(x) for x in e["grad_sizes"].tolist()]
    judged = dict(per_param_grad_errors(e["grad"], e["grad"], sizes, floor_frac=ct._PARAM_GRAD_FLOOR))
    return names.index(SP) in judged and cr.weights_regime(model) == "trained"


def test_unmoved_parameters_is_EXACT_zero_only():
    """Categorical: every element == 0.0. A parameter moved by any amount (1e-30) is not unmoved."""
    from agents.training import learner_golden as LG
    m = LG.build_learner()
    assert cr.unmoved_parameters(m) == []                     # the golden's perturbed weights
    head = m.policy.features_extractor.belief_head.species_head
    with torch.no_grad():
        head.weight.zero_()
    assert cr.unmoved_parameters(m) == [SP]
    with torch.no_grad():
        head.weight[0, 0] = 1e-30
    assert cr.unmoved_parameters(m) == []


def test_perturbed_parameters_ONLY_moves_the_named_and_restores_bit_exactly():
    from agents.model.parity_probe import perturbed_parameters
    lin = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.Linear(3, 2))
    with torch.no_grad():
        lin[1].weight.zero_()
    before = {n: p.detach().clone() for n, p in lin.named_parameters()}
    with perturbed_parameters(lin, seed=7, scale=0.05, only=["1.weight"]):
        now = dict(lin.named_parameters())
        assert bool(now["1.weight"].detach().any())
        assert all(torch.equal(now[n].detach(), before[n]) for n in before if n != "1.weight")
    assert all(torch.equal(p.detach(), before[n]) for n, p in lin.named_parameters())
    with pytest.raises(KeyError):
        with perturbed_parameters(lin, only=["no.such"]):
            pass


def test_the_GATE_judges_an_unmoved_head_at_the_FRESH_bar_and_on_its_own_perturbed_rung(oracle_like, monkeypatch):
    """A resume of an oracle run: the head's fresh-weights noise passes the startup gate (fails on
    revert — the trained bar refused it), and the rule line names the unmoved rung."""
    assert _judged(oracle_like)
    _plant(monkeypatch, only_while_zero=True)
    rules = cr.gate_regions(oracle_like, batch_size=BATCH, say=lambda _m: None)
    assert any("declared-exception param(s)" in r and SP in r for r in rules), rules
    assert any("unmoved zero-init param(s) perturbed" in r for r in rules), rules


def test_the_GATE_still_catches_a_REAL_miscompile_on_the_unmoved_head(oracle_like, monkeypatch):
    """The same error on EVERY call is a miscompile: the live rung's fresh bar absorbs it, the unmoved
    rung's trained bar refuses it. Fails if the unmoved rung is dropped or the head excluded."""
    assert _judged(oracle_like)
    _plant(monkeypatch, only_while_zero=False)
    with pytest.raises(ct.CompileTrainerError, match="DISAGREES with eager.*species_head"):
        cr.gate_regions(oracle_like, batch_size=BATCH, say=lambda _m: None)


def test_the_CANARY_passes_the_unmoved_heads_noise_and_FATALs_its_miscompile(oracle_like, monkeypatch):
    """The 2026-10-04 FATAL's class, through the in-run canary: noise passes (fails on revert), a
    persistent miscompile is CONFIRMED and FATAL."""
    from agents.model.compile_canary import CompileCanary, CompileCanaryError
    assert _judged(oracle_like)
    _plant(monkeypatch, only_while_zero=True)
    out = CompileCanary(oracle_like, batch_size=BATCH, every=1, saver=lambda p: None).after_update()
    assert out["compile/canary_ok"] == 1.0 and out["compile/canary_unmoved_params"] >= 1.0
    monkeypatch.undo()                                         # the noise plant off; the miscompile alone
    _plant(monkeypatch, only_while_zero=False)
    with pytest.raises(CompileCanaryError, match="CONFIRMED"):
        CompileCanary(oracle_like, batch_size=BATCH, every=1, saver=lambda p: None).after_update()
