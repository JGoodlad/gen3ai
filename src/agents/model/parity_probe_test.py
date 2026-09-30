"""gen3_fresh_parity_probe_v1 — a parity check on FRESH weights must bite, or refuse.

THE DEFECT CLASS (M5 T2, 2026-09-29): a fresh production policy's pointer head is zero-init, so its
legal log-probs are ``-log(n_legal)`` on every row whatever the extractor computed, and a
compiled-vs-eager comparison of them passes any miscompile (T2: max|dlogp| 0.0 fresh vs 0.68 real).
The learner's startup compile gate runs on exactly those weights at every FRESH launch.

These tests pin: the premise (a fresh production policy IS vacuous on the committed fixture, and the
seeded perturbation is NOT); the fail-closed guard (`decision_verdicts` / `train_verdict` refuse a
vacuous comparison); the perturbation's hygiene (bit-exact restore, the global RNG untouched,
deterministic); and the WIRING through `compile_trainer_extractor` itself — a miscompile that only
the pointer head can see (the per-action move cells it reads from the extractor's stash) is REFUSED
on fresh weights. Reverting the gate's fresh path lets that miscompile through (see the last test).

CPU, routine tier: the gate is driven with the device, the timings and `torch.compile` stubbed, the
same seam `compile_trainer_test`'s TF32 wiring test uses (the CPU backward does not lower, so the
real compile is CUDA-only; this is the control flow + verdicts on the REAL production policy).
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from agents.model import parity_probe as pp
from agents.model.compile_trainer import (_FP32_TOL, _readout, CompileTrainerError,
                                          VacuousCompileParityError, compile_trainer_extractor,
                                          decision_verdicts, train_verdict)
from utils.torch_state_guard import torch_globals

_ROWS = 16


# --------------------------------------------------------------------------- pure: the guard

def _logp(rows, legal=3, vary=True):
    x = torch.zeros(rows, 11)
    base = -torch.log(torch.tensor(float(legal)))
    x[:, :legal] = base
    if vary:
        x[:, 0] += 0.5
        x[:, 1] -= 0.5
    return x


def test_spread_reads_a_constant_per_row_logprob_as_VACUOUS_and_a_real_one_as_not():
    assert pp.spread("legal_logprob", _logp(4, vary=False)) == 0.0
    assert pp.spread("legal_logprob", _logp(4, vary=True)) == pytest.approx(1.0)
    single = torch.zeros(4, 11)
    single[:, 2] = 0.0                               # one legal action: logp 0, nothing to compare
    assert pp.spread("legal_logprob", single) == 0.0
    assert pp.spread("value", torch.full((5,), 0.5)) == 0.0
    assert pp.spread("value", torch.tensor([0.1, 0.9])) == pytest.approx(0.8)
    assert pp.spread("grad", torch.zeros(7)) == 0.0


def test_require_informative_refuses_constant_and_NaN_and_passes_informative():
    bars = {"legal_logprob": 1e-3, "value": 1e-4}
    with pytest.raises(pp.VacuousParityError, match="legal_logprob"):
        pp.require_informative({"legal_logprob": _logp(4, vary=False),
                                "value": torch.tensor([0.1, 0.2, 0.3, 0.4])}, bars)
    with pytest.raises(pp.VacuousParityError, match="value"):
        pp.require_informative({"value": torch.tensor([float("nan")] * 3)}, bars)
    pp.require_informative({"legal_logprob": _logp(4), "value": torch.tensor([0.1, 0.3])}, bars)


def test_decision_verdicts_REFUSES_a_vacuous_comparison_and_allow_vacuous_is_explicit():
    """The routine-tier pin: a parity check that runs on constant outputs FAILS."""
    const = {"features": torch.randn(4, 8, generator=torch.Generator().manual_seed(0)),
             "legal_logprob": _logp(4, vary=False), "value": torch.linspace(0.1, 0.9, 4)}
    with pytest.raises(VacuousCompileParityError, match="VACUOUS.*legal_logprob"):
        decision_verdicts(eager=const, compiled={k: v.clone() for k, v in const.items()},
                          precision="highest")
    assert isinstance(VacuousCompileParityError("x"), CompileTrainerError), \
        "the launcher classifies CompileTrainerError as config-fatal; a vacuous gate must be too"
    assert len(decision_verdicts(eager=const, compiled=const, precision="highest",
                                 allow_vacuous=True)) == 3


def test_train_verdict_REFUSES_an_all_zero_gradient():
    f = torch.randn(3, 4, generator=torch.Generator().manual_seed(0))
    with pytest.raises(VacuousCompileParityError, match="grad"):
        train_verdict(eager={"features": f, "grad": torch.zeros(10)},
                      compiled={"features": f, "grad": torch.zeros(10)}, precision="highest")


# --------------------------------------------------------------------------- the perturbation

def test_perturbed_parameters_restores_BIT_EXACTLY_touches_no_RNG_and_is_deterministic():
    m = torch.nn.Sequential(torch.nn.Linear(5, 4), torch.nn.Linear(4, 2))
    before = {k: v.clone() for k, v in m.state_dict().items()}
    rng = torch.get_rng_state().clone()
    seen = []
    for _ in range(2):
        with pp.perturbed_parameters(m):
            seen.append({k: v.clone() for k, v in m.state_dict().items()})
    assert torch.equal(torch.get_rng_state(), rng), "the perturbation advanced the global RNG"
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())
    assert all(torch.equal(seen[0][k], seen[1][k]) for k in before), "not deterministic"
    assert any(not torch.equal(seen[0][k], before[k]) for k in before), "perturbed nothing"
    with pytest.raises(ValueError):
        with pp.perturbed_parameters(m):
            raise ValueError("boom")
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items()), \
        "an exception inside the block must still restore the weights"
    cp = pp.perturbed_copy(m)
    assert all(torch.equal(before[k], v) for k, v in m.state_dict().items())
    assert all(torch.equal(seen[0][k], v) for k, v in cp.state_dict().items())


# --------------------------------------------------------------------------- the real policy

@pytest.fixture(scope="module")
def fresh():
    """The production policy exactly as a FRESH launch builds it (seeded, untrained, CPU), plus the
    committed REAL-obs fixture rows as the gate reads them."""
    from agents.model.compile_trainer import _parity_obs
    from agents.model.extra_obs_keys import zero_extra_obs
    from main.fresh_checkpoint import build_fresh_model

    with torch_globals(num_threads=2):          # restored at module end (the global-state guard)
        model, _, _ = build_fresh_model(0)
        fe = model.policy.features_extractor
        obs, mask = _parity_obs(fe.layout["total_dim"], _ROWS, torch.device("cpu"))
        obs.update(zero_extra_obs(fe, batch=_ROWS, device=torch.device("cpu")))
        yield model, obs, mask


def _decision(model, obs, mask):
    return {k: v for k, v in _readout(model, model.policy.features_extractor, obs, mask).items()
            if k in _FP32_TOL}


def test_PREMISE_a_fresh_production_policy_is_vacuous_and_its_perturbation_is_not(fresh):
    """If this flips (a non-zero pointer init), the fresh path goes quiet — revisit the gate."""
    model, obs, mask = fresh
    bad = pp.vacuous_keys(_decision(model, obs, mask), _FP32_TOL)
    assert "legal_logprob" in bad and bad["legal_logprob"] == 0.0, bad
    with pp.perturbed_parameters(model.policy):
        assert pp.vacuous_keys(_decision(model, obs, mask), _FP32_TOL) == {}


def _drive_gate(monkeypatch, model, compiled_factory, lines):
    """Run `compile_trainer_extractor` on the real CPU policy with the CUDA-only parts stubbed."""
    import agents.model.compile_control as cc
    cc._reset_control_for_tests()
    fe = model.policy.features_extractor
    monkeypatch.setattr("agents.model.compile_trainer.resolve_device",
                        lambda f: torch.device("cuda"))
    import agents.model.compile_trainer as ct
    import agents.model.extra_obs_keys as eok
    cpu = torch.device("cpu")
    real_obs, real_extra = ct._parity_obs, eok.zero_extra_obs
    monkeypatch.setattr("agents.model.compile_trainer._parity_obs",
                        lambda d, b, device: real_obs(d, b, cpu))
    monkeypatch.setattr("agents.model.extra_obs_keys.zero_extra_obs",
                        lambda f, batch=1, device=None: real_extra(f, batch=batch, device=cpu))
    times = iter([20.0, 10.0])
    monkeypatch.setattr("agents.model.compile_trainer._time_steps", lambda *a, **k: next(times))
    monkeypatch.setattr("agents.model.compile_trainer.torch.compile",
                        lambda f, **k: compiled_factory(fe, f))
    try:
        return compile_trainer_extractor(model, True, batch=_ROWS, emit=lines.append)
    finally:
        if "forward" in vars(fe):
            del fe.forward
        cc._reset_control_for_tests()


def _pointer_cell_miscompile(fe, original):
    """A 'compiled' forward whose features and gradient are EXACT but whose per-action move cells
    — read only by the pointer head, through the extractor's stash — are wrong. On fresh weights
    the zero-init scorers multiply them by 0, so only a perturbed pass can see it."""
    def fwd(obs):
        out = original(obs)
        pin = fe.stash.pointer_inputs
        fe.stash.pointer_inputs = pin._replace(move_cells=pin.move_cells * 1.5 + 0.25)
        return out
    return fwd


def test_the_GATE_refuses_a_pointer_only_miscompile_on_FRESH_weights(monkeypatch, fresh):
    model, _obs, _mask = fresh
    before = {k: v.clone() for k, v in model.policy.state_dict().items()}
    rng = torch.get_rng_state().clone()
    lines: list = []
    with pytest.raises(CompileTrainerError, match="DISAGREES.*legal_logprob"):
        _drive_gate(monkeypatch, model, _pointer_cell_miscompile, lines)
    assert all(torch.equal(before[k], v) for k, v in model.policy.state_dict().items()), \
        "the gate's perturbed pass must leave the run's weights bit-identical"
    assert torch.equal(torch.get_rng_state(), rng), "the gate advanced training's RNG stream"


def test_the_GATE_passes_a_correct_compile_on_FRESH_weights_and_says_so(monkeypatch, fresh):
    model, _obs, _mask = fresh
    before = {k: v.clone() for k, v in model.policy.state_dict().items()}
    lines: list = []
    assert _drive_gate(monkeypatch, model, lambda fe, f: f, lines) == 2.0
    assert any("FRESH weights" in ln and "legal_logprob" in ln for ln in lines), lines
    assert any("parity PASS" in ln and "[fresh weights, seeded perturbation]" in ln
               for ln in lines), lines
    assert all(torch.equal(before[k], v) for k, v in model.policy.state_dict().items())


def test_on_INFORMATIVE_weights_the_gate_takes_no_fresh_path(monkeypatch, fresh):
    model, _obs, _mask = fresh
    with pp.perturbed_parameters(model.policy, seed=7):    # a stand-in for TRAINED weights
        lines: list = []
        assert _drive_gate(monkeypatch, model, lambda fe, f: f, lines) == 2.0
        assert not any("FRESH weights" in ln for ln in lines), lines
        with pytest.raises(CompileTrainerError, match="DISAGREES.*legal_logprob"):
            _drive_gate(monkeypatch, model, _pointer_cell_miscompile, [])


def test_the_fixture_rows_carry_multi_action_rows(fresh):
    """The within-row spread needs rows with >= 2 legal actions; the fixture has them."""
    _model, _obs, mask = fresh
    assert int((np.asarray(mask).sum(axis=1) >= 2).sum()) >= _ROWS // 2


def test_a_single_row_is_judged_on_its_logprobs_only():
    one = {"legal_logprob": _logp(1, vary=False), "value": torch.tensor([0.5])}
    assert pp.vacuous_keys(one, {"legal_logprob": 1e-3, "value": 1e-4}) == {"legal_logprob": 0.0}
    one["legal_logprob"] = _logp(1, vary=True)
    assert pp.vacuous_keys(one, {"legal_logprob": 1e-3, "value": 1e-4}) == {}
