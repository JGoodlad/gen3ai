"""gen3_gate_grad_coverage_v1 — the learner compile gate's TRAIN-graph check must reach every path
the production loss reads, and must refuse to judge a backward it cannot see.

THE DEFECT CLASS (2026-09-29): the gate's train-step loss was ``pi^2 + vf^2`` over the extractor's
two feature outputs. The pointer head reads the extractor's STASH (move seats, team tokens, the
per-action cells), so every parameter feeding only those paths got a ZERO gradient from the gate's
loss (43 of 232 extractor parameters on the perturbed production policy), and a backward-only
miscompile there passed on ANY weights. And the global gradient cosine, even once the loss reaches
them, is dominated by the largest gradients: dropping the whole move-cell gradient left it at 1.0000.

These tests pin: the coverage of the new probe loss on the real production policy (and that the old
loss FAILS the coverage guard — the premise); the guard itself; the per-parameter rule; and the
WIRING — a backward-only miscompile in a pointer-only path is REFUSED by `compile_trainer_extractor`
on FRESH weights (and on informative ones), and the same fake passes a correct-forward gate check.

CPU, routine tier, driven through `parity_probe_test._drive_gate` (device, timings and
`torch.compile` stubbed; the real compile is CUDA-only).
"""
from __future__ import annotations

import pytest
import torch

from agents.model import compile_gate_probe as cgp
from agents.model import parity_probe as pp
from agents.model.compile_trainer import (_MAX_PARAM_GRAD_REL, _MAX_PARAM_GRAD_REL_TRAINED,
                                          _PARAM_GRAD_FLOOR, CompileTrainerError,
                                          VacuousCompileParityError, _cos, _train_step,
                                          train_verdict)
from agents.model.parity_probe_test import _drive_gate

_ROWS = 16


@pytest.fixture(scope="module")
def fresh():
    from agents.model.compile_trainer import _parity_obs
    from agents.model.extra_obs_keys import zero_extra_obs
    from main.fresh_checkpoint import build_fresh_model

    torch.set_num_threads(2)
    model, _, _ = build_fresh_model(0)
    fe = model.policy.features_extractor
    fe.train()
    obs, mask = _parity_obs(fe.layout["total_dim"], _ROWS, torch.device("cpu"))
    obs.update(zero_extra_obs(fe, batch=_ROWS, device=torch.device("cpu")))
    return model, obs, mask


def _names(model):
    return [n for n, _ in cgp.grad_parameters(model, model.policy.features_extractor)]


# --------------------------------------------------------------------------- coverage

def test_the_probe_loss_COVERS_the_policy_and_the_old_features_only_loss_did_not(fresh):
    """The measured premise and the bar. New loss, perturbed weights: <= bar zero-grad parameters
    (measured 1/254: the Baton-Pass edge map, no fixture row has a BP seat). The OLD features-only
    loss on the same weights FAILS the guard (measured 65/254) — revert `gate_loss` to it and the
    gate refuses every launch, so this cannot silently regress to the blind loss."""
    model, obs, mask = fresh
    fe = model.policy.features_extractor
    names = _names(model)
    with pp.perturbed_parameters(model.policy):
        new = _train_step(model, fe, obs, mask)
        line = cgp.coverage_verdict(new["grad_absmax"], names)
        zero = cgp.zero_grad_names(new["grad_absmax"], names)
        assert len(zero) / len(names) <= cgp.MAX_ZERO_GRAD_FRACTION, zero
        assert not any(n.startswith("pointer_head.") for n in zero), zero
        assert not any("belief" in n for n in zero), zero

        model.policy.zero_grad(set_to_none=True)
        pi, vf = fe(obs)
        (pi.square().mean() + vf.square().mean()).backward()
        old = torch.stack([(p.grad.abs().max() if p.grad is not None else torch.zeros(()))
                           for _, p in cgp.grad_parameters(model, fe)])
        model.policy.zero_grad(set_to_none=True)
        with pytest.raises(cgp.GradCoverageError, match="GRAD COVERAGE"):
            cgp.coverage_verdict(old, names)
    assert "grad coverage" in line


def test_coverage_guard_counts_NaN_as_zero_and_names_the_parameters():
    a = torch.tensor([1.0, 0.0, float("nan"), 2.0])
    assert cgp.zero_grad_names(a, ["a", "b", "c", "d"]) == ["b", "c"]
    with pytest.raises(cgp.GradCoverageError, match=r"\['b', 'c'\]"):
        cgp.coverage_verdict(a, ["a", "b", "c", "d"], bar=0.25)
    assert "1/4" in cgp.coverage_verdict(torch.tensor([1.0, 0.0, 1.0, 1.0]), bar=0.25)


def test_train_verdict_REFUSES_poor_coverage_as_VACUOUS_and_allow_vacuous_skips_it():
    f = torch.randn(3, 4, generator=torch.Generator().manual_seed(0))
    g = torch.randn(40, generator=torch.Generator().manual_seed(1))
    g[:20] = 0.0
    arm = {"features": f, "grad": g, "grad_sizes": torch.full((10,), 4, dtype=torch.long),
           "grad_absmax": torch.stack([c.abs().max() for c in g.split(4)])}
    with pytest.raises(VacuousCompileParityError, match="GRAD COVERAGE"):
        train_verdict(eager=arm, compiled=dict(arm), precision="highest")
    assert "per-param" in train_verdict(eager=arm, compiled=dict(arm), precision="highest",
                                        allow_vacuous=True)


# --------------------------------------------------------------------------- the per-param rule

def test_per_param_rule_catches_a_small_path_the_global_cosine_cannot():
    """One big parameter dominates the norm; a 10% error on a small one leaves the global cosine
    above 0.9999 and must still fail."""
    big = torch.full((1000,), 10.0)
    small = torch.linspace(0.1, 1.0, 10)
    e = torch.cat([big, small])
    c = torch.cat([big, small * 1.1])
    assert _cos(c, e) >= 0.9999
    f = torch.randn(3, 4, generator=torch.Generator().manual_seed(0))
    sizes = torch.tensor([1000, 10])

    def arm(grad):
        return {"features": f, "grad": grad, "grad_sizes": sizes,
                "grad_absmax": torch.stack([x.abs().max() for x in grad.split([1000, 10])])}
    with pytest.raises(CompileTrainerError, match="per-param.*FAILED"):
        train_verdict(eager=arm(e), compiled=arm(c), precision="highest", param_names=["B", "S"])
    assert "per-param" in train_verdict(eager=arm(e), compiled=arm(e.clone()),
                                        precision="highest")
    assert 0 < _PARAM_GRAD_FLOOR < 1 and 0 < _MAX_PARAM_GRAD_REL < 0.1


def test_cosine_is_float64_and_never_reads_above_one():
    g = torch.randn(10_000_000, generator=torch.Generator().manual_seed(0)) * 1e3
    assert abs(_cos(g, g.clone()) - 1.0) < 1e-12


# --------------------------------------------------------------------------- the wiring

class _ScaleGrad(torch.autograd.Function):
    """Identity forward; the backward scales the incoming gradient — a BACKWARD-ONLY defect."""

    @staticmethod
    def forward(ctx, x, k):                      # type: ignore[override]
        ctx.k = k
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):                        # type: ignore[override]
        return g * ctx.k, None


def _pointer_backward_miscompile(fe, original, field="move_cells", k=0.0):
    """A 'compiled' forward whose FORWARD is exact (so every decision readout matches) but whose
    backward through one pointer-only stash path is wrong — the defect the old train loss, which
    never reached the stash, could not see on any weights."""
    def fwd(obs):
        out = original(obs)
        pin = fe.stash.pointer_inputs
        fe.stash.pointer_inputs = pin._replace(**{field: _ScaleGrad.apply(getattr(pin, field), k)})
        return out
    return fwd


@pytest.mark.parametrize("field,k", [("move_cells", 0.0), ("switch_cells", 0.5),
                                     ("move_tokens", 0.9)])
def test_the_GATE_refuses_a_BACKWARD_only_pointer_miscompile_on_FRESH_weights(
        monkeypatch, fresh, field, k):
    """Reverting `_train_step` to the features-only loss makes this PASS the gate (verified
    2026-09-29 — see compile_flags.md)."""
    model, _obs, _mask = fresh
    before = {n: v.clone() for n, v in model.policy.state_dict().items()}
    with pytest.raises(CompileTrainerError, match="TRAIN graph.*per-param"):
        _drive_gate(monkeypatch, model,
                    lambda fe, f: _pointer_backward_miscompile(fe, f, field, k), [])
    assert all(torch.equal(before[n], v) for n, v in model.policy.state_dict().items())


def test_the_GATE_refuses_it_on_INFORMATIVE_weights_too_and_passes_a_correct_compile(
        monkeypatch, fresh):
    model, _obs, _mask = fresh
    with pp.perturbed_parameters(model.policy, seed=7):        # a stand-in for TRAINED weights
        with pytest.raises(CompileTrainerError, match="TRAIN graph.*per-param"):
            _drive_gate(monkeypatch, model, _pointer_backward_miscompile, [])
        lines: list = []
        assert _drive_gate(monkeypatch, model, lambda fe, f: f, lines) == 2.0
        assert any("grad coverage" in ln and "per-param" in ln for ln in lines), lines
        # REAL weights are judged at the TRAINED bar (healthy CUDA noise there reaches 9.3e-4, with
        # isolated 4.9e-2 outliers on perturbed trained states); only the fresh pass is strict.
        assert any(f"<= {_MAX_PARAM_GRAD_REL_TRAINED:g}" in ln for ln in lines), lines


def test_the_FRESH_pass_is_judged_at_the_STRICT_per_param_bar(monkeypatch, fresh):
    model, _obs, _mask = fresh
    lines: list = []
    assert _drive_gate(monkeypatch, model, lambda fe, f: f, lines) == 2.0
    line = next(ln for ln in lines if "parity PASS" in ln)
    fresh_part, real_part = line.split("| features:", 1)
    assert f"<= {_MAX_PARAM_GRAD_REL:g}" in fresh_part and "seeded perturbation" in fresh_part
    assert f"<= {_MAX_PARAM_GRAD_REL_TRAINED:g}" in real_part
    assert _MAX_PARAM_GRAD_REL < _MAX_PARAM_GRAD_REL_TRAINED
