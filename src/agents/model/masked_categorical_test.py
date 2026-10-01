"""`masked_categorical` is BIT-IDENTICAL to sb3's `MaskableCategorical` (gen3_functional_masking_v1, K8):
masked logits, log-prob, entropy, probs, the sampled actions (same RNG draw) and the deterministic mode —
masked and unmasked, every mask spelling the learner and the rollout pass. Each test fails if its line of
the reconstruction drifts from sb3 + torch (e.g. dropping the double normalisation, masking from the
renormalised logits instead of `_original_logits`, or a different multinomial call)."""
from __future__ import annotations

import numpy as np
import pytest
import torch
from sb3_contrib.common.maskable.distributions import MaskableCategoricalDistribution

from agents.model import masked_categorical as mc

A = 11


def _cases(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(257, A, generator=g) * 4.0
    logits[3] = torch.tensor([30.0, -30.0] + [0.0] * (A - 2))      # extreme spread
    logits[4] = 0.0                                                 # the fresh pointer head: all equal
    masks = torch.rand(257, A, generator=g) > 0.45
    masks[:, 0] |= ~masks.any(dim=-1)                               # >= 1 legal per row
    masks[5] = False
    masks[5, 7] = True                                              # exactly one legal
    masks[6] = True                                                 # all legal
    actions = torch.stack([torch.multinomial(m.float(), 1, generator=g)[0] for m in masks])
    return logits, masks, actions


def _sb3(logits, masks):
    d = MaskableCategoricalDistribution(A).proba_distribution(action_logits=logits)
    if masks is not None:
        d.apply_masking(masks)
    return d


@pytest.mark.parametrize("spelling", ["bool", "float", "numpy"])
def test_masked_logits_log_prob_entropy_probs_are_bit_identical(spelling):
    logits, masks, actions = _cases()
    given = {"bool": masks, "float": masks.float(), "numpy": masks.numpy()}[spelling]
    ref = _sb3(logits, given)
    lp = mc.masked_logits(logits, given, A)
    mb = mc.mask_bool(given, lp)
    assert torch.equal(lp, ref.distribution.logits)
    assert torch.equal(mc.log_prob(lp, actions), ref.log_prob(actions))
    assert torch.equal(mc.entropy(lp, mb), ref.entropy())
    assert torch.equal(mc.probs(lp), ref.distribution.probs)
    pi = mc.MaskedPi(lp, mb)
    assert torch.equal(pi.distribution.logits, ref.distribution.logits)
    assert torch.equal(pi.distribution.probs, ref.distribution.probs)
    assert torch.equal(pi.entropy(), ref.entropy())


def test_unmasked_is_the_double_normalised_logits_and_torchs_entropy():
    logits, _, actions = _cases(1)
    ref = _sb3(logits, None)
    lp = mc.masked_logits(logits, None, A)
    assert torch.equal(lp, ref.distribution.logits)
    assert torch.equal(mc.log_prob(lp, actions), ref.log_prob(actions))
    assert torch.equal(mc.entropy(lp, None), ref.entropy())


def test_logits_of_any_leading_shape_are_viewed_as_rows():
    logits, masks, _ = _cases(2)
    ref = _sb3(logits, masks)
    assert torch.equal(mc.masked_logits(logits.reshape(257, 1, A), masks, A), ref.distribution.logits)


def test_sample_draws_the_same_actions_from_the_same_rng_stream():
    logits, masks, _ = _cases(3)
    ref = _sb3(logits, masks)
    lp = mc.masked_logits(logits, masks, A)
    with torch.random.fork_rng():
        torch.manual_seed(11)
        a_ref = ref.get_actions(deterministic=False)
        after_ref = torch.rand(3)
    with torch.random.fork_rng():
        torch.manual_seed(11)
        a = mc.sample(lp)
        after = torch.rand(3)
    assert torch.equal(a, a_ref)
    assert torch.equal(after, after_ref)          # the same number of draws consumed
    assert torch.equal(mc.mode(lp), ref.get_actions(deterministic=True))
    assert bool(masks.gather(1, a[:, None]).all())


def test_the_policy_hot_paths_match_the_sb3_object_path():
    """The production-surface policy: `evaluate_actions` and `forward` (functional) against the
    same forward scored through sb3's distribution object (`_get_action_dist_from_latent`)."""
    from agents.training import learner_golden as LG

    model = LG.build_learner()
    pol = model.policy
    LG.load_buffer_into(model)
    rb = model.rollout_buffer
    obs = {k: torch.as_tensor(v.reshape(-1, *v.shape[2:])[:32]) for k, v in rb.observations.items()}
    masks = torch.as_tensor(rb.action_masks.reshape(-1, A)[:32])
    acts = torch.as_tensor(rb.actions.reshape(-1)[:32]).long()
    pol.set_training_mode(True)
    with torch.no_grad():
        v, lp, ent = pol.evaluate_actions(obs, acts, action_masks=masks)
        pi_f, vf_f = pol.extract_features(obs)
        d = pol._get_action_dist_from_latent(pol.mlp_extractor.forward_actor(pi_f))
        d.apply_masking(masks)
        assert torch.equal(lp, d.log_prob(acts))
        assert torch.equal(ent, d.entropy())
        assert torch.equal(pol._last_pi_distribution.distribution.logits, d.distribution.logits)
        assert torch.equal(pol._last_pi_distribution.distribution.probs, d.distribution.probs)
        pol.set_training_mode(False)
        with torch.random.fork_rng():
            torch.manual_seed(5)
            a, v2, lp2 = pol(obs, action_masks=masks.numpy())
        pi_f, vf_f = pol.extract_features(obs)
        d = pol._get_action_dist_from_latent(pol.mlp_extractor.forward_actor(pi_f))
        d.apply_masking(masks.numpy())
        with torch.random.fork_rng():
            torch.manual_seed(5)
            a_ref = d.get_actions(deterministic=False)
        assert torch.equal(a, a_ref)
        assert torch.equal(lp2, d.log_prob(a_ref))
        assert np.isfinite(v2.numpy()).all()
