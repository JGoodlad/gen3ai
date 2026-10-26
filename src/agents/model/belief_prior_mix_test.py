"""`belief_prior_mix` (INFERENCE ONLY): b ← (1 − λ)·b + λ·prior on the species, moves and items legs.

Asserted on a REAL ``MaskablePPO``-built production-surface learner (``learner_golden.build_learner``: SB3's
``_build`` re-init plus the golden's perturbation, so every learned delta is NON-zero) over the golden buffer's real
rows. The pins:

* λ = 0 is BIT-IDENTICAL to a model that never set it, including after a λ > 0 forward (the flag restores);
* λ = 1 is BIT-IDENTICAL to D0f's surgery (the three deltas zeroed in the weights) on the policy logits, the critic
  and every belief leg, and it CHANGES the policy (so the equality is not vacuous). Reverting the mix in any one leg
  fails the leg's own equality;
* 0 < λ < 1 mixes the species presences linearly (the delta's inputs precede every belief), and the move / item
  mixing helpers are the stated mixtures;
* a graph-building forward with λ > 0 raises, and a λ outside [0, 1] or a model without the three legs refuses.
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Tuple

import pytest
import torch as th

from agents.model import belief_prior_mix as BPM


@pytest.fixture(scope="module")
def learner() -> Tuple[Any, Dict[str, th.Tensor], th.Tensor]:
    from stable_baselines3.common.utils import obs_as_tensor

    from agents.training import learner_golden as LG

    th.manual_seed(0)
    model = LG.build_learner(trunk_precision="fp32")
    LG.load_buffer_into(model)
    rb = model.rollout_buffer
    obs = obs_as_tensor({k: v.reshape((-1,) + v.shape[2:]) for k, v in rb.observations.items()}, "cpu")
    masks = th.as_tensor(rb.action_masks.reshape(obs["observation"].shape[0], -1))
    model.policy.set_training_mode(False)
    return model, obs, masks


def _forward(model: Any, obs: Dict[str, th.Tensor], masks: th.Tensor) -> Dict[str, th.Tensor]:
    """Everything the mix can reach: the policy logits, the critic, and each leg's published belief."""
    pol = model.policy
    with th.no_grad():
        logits = pol.get_distribution(obs, action_masks=masks).distribution.logits.clone()
        fe = pol.features_extractor
        hs = fe.last_hypothesis
        out = {"logits": logits, "species_pi": hs.species.pi.clone(), "species_log_pi": hs.species.log_pi.clone(),
               "slot_species": hs.slot_species.clone(), "moves": fe.last_move_belief_logits.clone(),
               "items": fe.last_item_logits.clone()}
        out["value"] = pol.predict_values(obs).clone()
    return out


def _at(model: Any, lam: float, obs: Any, masks: Any) -> Dict[str, th.Tensor]:
    BPM.set_belief_prior_mix(model, lam)
    try:
        return _forward(model, obs, masks)
    finally:
        BPM.set_belief_prior_mix(model, 0.0)


def _zeroed(model: Any) -> Any:
    """D0f's surgery: a copy with the species, move and item deltas' last layers zeroed (= the prior by
    construction)."""
    m = copy.deepcopy(model)
    fe = m.policy.features_extractor
    with th.no_grad():
        for lin in (fe.hypothesis_builder.delta_out, fe.move_belief.move_head, fe.item_belief_head.item_head):
            lin.weight.zero_()
            lin.bias.zero_()
    return m


def _equal(a: Dict[str, th.Tensor], b: Dict[str, th.Tensor]) -> Dict[str, bool]:
    return {k: bool(th.equal(a[k], b[k])) for k in a}


def test_the_deltas_are_live(learner: Any) -> None:
    """Precondition: every delta the mix shrinks is non-zero, so prior-only differs from the posterior."""
    fe = learner[0].policy.features_extractor
    for lin in (fe.hypothesis_builder.delta_out, fe.move_belief.move_head, fe.item_belief_head.item_head):
        assert float(lin.weight.abs().sum()) > 0.0
    assert fe.move_belief.prior_fusion


def test_lambda_zero_is_bit_identical_and_restores(learner: Any) -> None:
    model, obs, masks = learner
    assert BPM.belief_prior_mix_of(model) == 0.0
    base = _forward(model, obs, masks)
    zero = _at(model, 0.0, obs, masks)
    _at(model, 0.25, obs, masks)                        # a mixed forward in between ...
    after = _forward(model, obs, masks)                 # ... leaves nothing behind
    assert all(_equal(base, zero).values()), _equal(base, zero)
    assert all(_equal(base, after).values()), _equal(base, after)


def test_lambda_one_is_the_deltas_zeroed(learner: Any) -> None:
    model, obs, masks = learner
    base = _forward(model, obs, masks)
    one = _at(model, 1.0, obs, masks)
    d0f = _forward(_zeroed(model), obs, masks)
    eq = _equal(one, d0f)
    assert all(eq.values()), f"λ = 1 differs from the zeroed deltas on {[k for k, v in eq.items() if not v]}"
    moved = _equal(one, base)
    # not vacuous: prior-only moves the beliefs AND the action logits on these rows
    assert not moved["logits"] and not moved["species_pi"] and not moved["moves"] and not moved["items"], moved


def test_species_presence_mixes_linearly(learner: Any) -> None:
    """π(λ) = (1 − λ)·π(0) + λ·π(1): δ_θ reads the pre-belief tokens, so nothing the mix moves feeds back."""
    model, obs, masks = learner
    p0 = _forward(model, obs, masks)["species_pi"]
    p1 = _at(model, 1.0, obs, masks)["species_pi"]
    lam = 0.25
    pm = _at(model, lam, obs, masks)
    th.testing.assert_close(pm["species_pi"], (1 - lam) * p0 + lam * p1, rtol=0, atol=1e-6)
    th.testing.assert_close(pm["species_pi"].sum(-1), p0.sum(-1), rtol=0, atol=1e-4)   # still Σ = k
    assert th.isfinite(pm["logits"][masks.bool()]).all() and th.isfinite(pm["value"]).all()
    assert not th.equal(pm["logits"], _forward(model, obs, masks)["logits"])


@pytest.mark.parametrize("lam", [0.1, 0.25, 0.5, 0.9])
def test_the_mixing_helpers_are_the_stated_mixtures(lam: float) -> None:
    g = th.Generator().manual_seed(7)
    a, b = th.randn(64, 30, generator=g) * 6, th.randn(64, 30, generator=g) * 6
    want = (1 - lam) * th.sigmoid(a) + lam * th.sigmoid(b)
    th.testing.assert_close(th.sigmoid(BPM.mix_bernoulli_logits(a, b, lam)), want, rtol=1e-5, atol=1e-7)
    want_c = (1 - lam) * th.softmax(a, -1) + lam * th.softmax(b, -1)
    th.testing.assert_close(th.softmax(BPM.mix_categorical_logits(a, b, lam), -1), want_c, rtol=1e-5, atol=1e-7)
    th.testing.assert_close(BPM.mix_categorical_logits(a, b, lam).exp().sum(-1), th.ones(64), rtol=0, atol=1e-5)
    # extreme logits (an illegal move's prior, a pinned reveal) stay finite
    big = th.tensor([-40.0, -13.0, 0.0, 13.0, 40.0])
    assert th.isfinite(BPM.mix_bernoulli_logits(big, big.flip(0), lam)).all()


def test_a_gradient_forward_refuses(learner: Any) -> None:
    model, obs, masks = learner
    BPM.set_belief_prior_mix(model, 0.25)
    try:
        with pytest.raises(BPM.BeliefPriorMixError, match="INFERENCE-ONLY"):
            model.policy.get_distribution(obs, action_masks=masks)
    finally:
        BPM.set_belief_prior_mix(model, 0.0)


def test_refusals() -> None:
    for bad in (-0.1, 1.5, float("nan"), "x"):
        with pytest.raises(BPM.BeliefPriorMixError):
            BPM.check_lambda(bad)

    class _Bare(th.nn.Module):
        hypothesis_builder = None

    with pytest.raises(BPM.BeliefPriorMixError, match="hypothesis_builder"):
        BPM.set_belief_prior_mix(_Bare(), 0.25)
