"""gen3_opponent_compile_parity_v1 — a compiled CPU OPPONENT must match eager at the decision level.

Before this, `--compile-opponents` checked only speed on an all-zero observation, so a compiled
opponent whose decisions were wrong played every self-play / eval game with every gate
green. These tests drive `maybe_compile_extractor` itself on the REAL production policy (fresh, as a
fresh pool snapshot would be, and perturbed as a stand-in for trained weights) with `torch.compile`
and the timing stubbed, and pin: a correct compile passes and says so; a miscompile only the pointer
head can see is REFUSED with the learner gate's typed error and uninstalled; the weights come back
bit-identical; a stand-in policy is refused rather than installed unvalidated. Removing the check
from `maybe_compile_extractor` lets the miscompile through (`test_..._REVERT_...` below proves the
fake is otherwise invisible to the path).
"""
from __future__ import annotations

import pytest
import torch

from agents.model import compile_opponents as S
from agents.model import parity_probe as pp
from agents.model.compile_trainer import CompileTrainerError
from utils.torch_state_guard import torch_globals


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.setattr(S, "_COMPILE_VALIDATED", False)
    monkeypatch.setattr(S, "_LOCAL_TALLY", {"reverts": 0, "total": 0})
    monkeypatch.delenv(S.COMPILE_QUORUM_ENV, raising=False)
    monkeypatch.setattr(S, "_measure_arms", lambda e, c, o: ([10.0] * 5, [2.0] * 5))
    from agents.model import opponent_parity as op
    monkeypatch.setattr(op, "_PASSED", {})        # the per-process PASS cache: per test


@pytest.fixture(scope="module")
def fresh():
    from main.fresh_checkpoint import build_fresh_model
    with torch_globals(num_threads=2):          # restored at module end (the global-state guard)
        model, _, _ = build_fresh_model(0)
        model.policy.set_training_mode(False)
        yield model


def _uninstall(fe):
    if "forward" in vars(fe):
        del fe.forward


def _pointer_cell_miscompile(fe):
    """Features exact; the per-action move cells only the pointer head reads are wrong."""
    def factory(fn):
        def fwd(obs):
            out = fn(obs)
            pin = fe.stash.pointer_inputs
            fe.stash.pointer_inputs = pin._replace(move_cells=pin.move_cells * 1.5 + 0.25)
            return out
        return fwd
    return factory


def test_a_correct_compile_PASSES_parity_on_fresh_weights_and_says_so(monkeypatch, fresh, capsys):
    fe = fresh.policy.features_extractor
    before = {k: v.clone() for k, v in fresh.policy.state_dict().items()}
    monkeypatch.setattr(torch, "compile", lambda fn, **k: fn)
    try:
        assert S.maybe_compile_extractor(fresh, True, label="t-ok") is True
    finally:
        _uninstall(fe)
    out = capsys.readouterr().out
    assert "parity PASS on 16 REAL obs rows at B=1" in out, out
    assert "[fresh weights, seeded perturbation scale=0.05 seed+0]" in out and "legal_logprob" in out, out
    assert all(torch.equal(before[k], v) for k, v in fresh.policy.state_dict().items())


@pytest.mark.parametrize("strict", [False, True])
def test_a_pointer_only_miscompile_is_REFUSED_with_the_typed_error_whatever_strict_says(
        monkeypatch, fresh, strict):
    fe = fresh.policy.features_extractor
    before = {k: v.clone() for k, v in fresh.policy.state_dict().items()}
    monkeypatch.setattr(torch, "compile", _pointer_cell_miscompile(fe))
    try:
        with pytest.raises(CompileTrainerError, match="--compile-opponents parity.*legal_logprob"):
            S.maybe_compile_extractor(fresh, True, label="t-bad", strict=strict)
        assert "forward" not in vars(fe) or getattr(fe.forward, "__func__", None) is \
            type(fe).forward, "a refused compile must be uninstalled"
    finally:
        _uninstall(fe)
    assert all(torch.equal(before[k], v) for k, v in fresh.policy.state_dict().items())


def test_it_is_refused_on_INFORMATIVE_weights_too(monkeypatch, fresh):
    fe = fresh.policy.features_extractor
    monkeypatch.setattr(torch, "compile", _pointer_cell_miscompile(fe))
    with pp.perturbed_parameters(fresh.policy, seed=7):
        try:
            with pytest.raises(CompileTrainerError, match="DISAGREES.*legal_logprob"):
                S.maybe_compile_extractor(fresh, True, label="t-bad-trained")
        finally:
            _uninstall(fe)


def test_REVERT_without_the_check_the_same_miscompile_is_INSTALLED(monkeypatch, fresh):
    """The fail-on-revert proof: with the parity seam neutralised (== the pre-fix path), the
    miscompile passes the timing gate and is installed as the opponent's forward."""
    fe = fresh.policy.features_extractor
    monkeypatch.setattr(torch, "compile", _pointer_cell_miscompile(fe))
    monkeypatch.setattr(S, "_check_parity", lambda *a, **k: "no check")
    try:
        assert S.maybe_compile_extractor(fresh, True, label="t-revert") is True
    finally:
        _uninstall(fe)


def test_a_stand_in_policy_is_REFUSED_not_installed_unvalidated():
    import types
    from agents.model.opponent_parity import check_opponent_parity
    fe = torch.nn.Linear(2, 2)
    m = types.SimpleNamespace(policy=types.SimpleNamespace(features_extractor=fe))
    with pytest.raises(CompileTrainerError, match="not a Gen3 dual-head policy"):
        check_opponent_parity(m, fe.forward, fe.forward)


def test_a_PASS_is_cached_per_distinct_weights_and_a_weight_change_is_rechecked(monkeypatch, fresh):
    """Once per distinct weights, not per load: the eval worker re-loads its opponent
    every game. The same weights hit the cache; ANY weight change is a new key and is judged."""
    from agents.model import opponent_parity as op
    fe = fresh.policy.features_extractor
    orig = fe.forward
    calls = {"n": 0}
    real_arm = op._arm

    def counting(*a, **k):
        calls["n"] += 1
        return real_arm(*a, **k)
    monkeypatch.setattr(op, "_arm", counting)
    first = op.check_opponent_parity(fresh, orig, orig, label="c1")
    n1 = calls["n"]
    again = op.check_opponent_parity(fresh, orig, orig, label="c2")
    assert calls["n"] == n1 and "cached" in again and "cached" not in first
    bad = _pointer_cell_miscompile(fe)(orig)
    with pp.perturbed_parameters(fresh.policy, seed=11):     # different weights: judged afresh
        with pytest.raises(CompileTrainerError, match="legal_logprob"):
            op.check_opponent_parity(fresh, orig, bad, label="c3")
    assert calls["n"] > n1


def test_a_COLLAPSED_critic_opponent_passes_parity_above_the_first_rung(monkeypatch, fresh, capsys):
    """gen3_parity_perturb_ladder_v1: a pool snapshot whose win-prob critic collapsed (saturated at
    logit −9) is vacuous on V at the fresh-weights scale; the gate climbs the declared ladder. Revert
    the ladder ⇒ `VacuousCompileParityError` on a legitimate opponent."""
    fe = fresh.policy.features_extractor
    head = fe.win_head.net[3]
    saved = {k: v.clone() for k, v in head.state_dict().items()}
    with torch.no_grad():
        head.bias.fill_(-9.0)
        head.weight.mul_(0.01)
    monkeypatch.setattr(torch, "compile", lambda fn, **k: fn)
    try:
        assert S.maybe_compile_extractor(fresh, True, label="t-collapsed") is True
    finally:
        _uninstall(fe)
        head.load_state_dict(saved)
    out = capsys.readouterr().out
    assert "parity PASS" in out and "seeded perturbation scale=" in out, out
    assert f"seeded perturbation scale={pp.PERTURB_SCALE:g} seed+0]" not in out, out
