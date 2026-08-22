"""The EVIDENTIAL Beta head (v98, gen3_cf_evidential_head_v1).

Four things are pinned here, and they are pinned in the order of how badly a silent break would
hurt:

1. **The MATH is right.** The Beta-Binomial marginal NLL is checked against
   `scipy.stats.betabinom.logpmf` on hand-chosen cases, not against a re-derivation of itself. A
   closed form that is wrong by a constant would still train (constants have no gradient) but a
   form wrong in α or β would train the head toward nonsense while the loss curve fell.
2. **OFF is byte-identical, and ON-at-coefficient-0 is BIT-identical.** The head is built LAST in
   `__init__`, so building it must not move any earlier module's initialization RNG draw — a
   property that is easy to break by "tidying" the constructor and impossible to notice without
   this test.
3. **The version gate fires.** The head is never called by the forward, so a mismatched resume
   produces NO shape error anywhere: `check_compatible` is the only thing between a flipped flag
   and a run that silently supervises a freshly-random head (or nothing) for good.
4. **The regularizer does what it says.** KL to the uniform is exactly 0 at the reachable floor
   Beta(1,1), and descending it actually moves α, β toward 1.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from gymnasium import spaces

from agents.model.aux_value_heads import CfEvidentialHead
from agents.model.arch_constants import D_MODEL
from agents.model.features_extractor import Gen3FeaturesExtractor
from agents.model.model_version import (MODEL_CONFIG_VERSION, ModelVersion, ModelVersionError,
                                        _migrate_config)
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings


@pytest.fixture(scope="module")
def ek_and_space():
    mappings = load_mappings()
    ek = Gen3ObservationEncoder(mappings).get_features_extractor_kwargs()
    total = ek["layout"]["total_dim"]
    space = spaces.Dict({
        "observation": spaces.Box(-np.inf, np.inf, (total,), np.float32),
        "action_mask": spaces.Box(0, 1, (11,), np.int8),
    })
    return ek, space, total


def _build(ek, space, on, seed=0):
    torch.manual_seed(seed)
    return Gen3FeaturesExtractor(space, **{**ek, "cf_evidential": on}).eval()


# ── the module itself ─────────────────────────────────────────────────────────

def test_alpha_and_beta_are_at_least_one_and_beta_one_one_is_reachable():
    """`softplus(x) + 1` keeps the Beta UNIMODAL (α<1 puts mass at an endpoint, turning
    "uncertain" into "certain of both extremes") while leaving the uniform Beta(1,1) — maximum
    ignorance — exactly attainable in the limit, so an unresolved state has an honest place to be."""
    head = CfEvidentialHead()
    a, b = head(torch.randn(64, D_MODEL) * 10.0)
    assert a.shape == (64,) and b.shape == (64,)
    assert float(a.min()) >= 1.0 and float(b.min()) >= 1.0
    # the floor is APPROACHED, not merely bounded: a very negative pre-activation lands at ~1, so
    # Beta(1,1) is a state the head can actually reach rather than an unattainable limit.
    raw = torch.nn.functional.softplus(torch.tensor([-40.0])) + 1.0
    assert float(raw) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("alpha,beta,w,n", [
    (2.0, 3.0, 3.0, 8.0),
    (1.0, 1.0, 0.0, 4.0),      # uniform prior: P(w) is flat = 1/(n+1) for every w
    (5.5, 1.5, 7.0, 16.0),
    (1.0, 9.0, 16.0, 16.0),    # all wins under a pessimistic prior — the expensive tail
])
def test_beta_binomial_nll_matches_scipy(alpha, beta, w, n):
    """The AUTHORITATIVE check: an INDEPENDENT implementation, not a re-derivation.

    `betabinom.logpmf` includes the binomial coefficient `log C(n, w)`; ours deliberately drops it
    (it does not depend on α or β, so it contributes no gradient), hence the `+ logC` here. Getting
    that offset wrong in the SOURCE would be invisible — which is exactly why it is stated in the
    comparison rather than assumed away.
    """
    from scipy.stats import betabinom
    mine = float(CfEvidentialHead.beta_binomial_nll(
        torch.tensor([alpha]), torch.tensor([beta]), torch.tensor([w]), torch.tensor([n])))
    ref = -float(betabinom.logpmf(w, n, alpha, beta)) + math.log(math.comb(int(n), int(w)))
    assert mine == pytest.approx(ref, rel=1e-5, abs=1e-5)


def test_uniform_beta_gives_the_flat_count_likelihood_by_hand():
    """A hand-computed anchor that does not go through scipy at all.

    Under Beta(1,1) the Beta-Binomial is UNIFORM over w ∈ {0..n}, so the true NLL is log(n+1) for
    every w — and OURS, which drops `log C(n, w)`, must therefore read `log(n+1) + log C(n, w)`.
    Writing the dropped constant out explicitly is the point: it says in the test what the loss is
    NOT computing, so a future reader cannot mistake the value for a probability.
    """
    n = 5.0
    for w in range(6):
        nll = float(CfEvidentialHead.beta_binomial_nll(
            torch.tensor([1.0]), torch.tensor([1.0]), torch.tensor([float(w)]), torch.tensor([n])))
        assert nll == pytest.approx(math.log(n + 1.0) + math.log(math.comb(5, w)), abs=1e-5)


def test_kl_to_uniform_is_exactly_zero_at_beta_one_one():
    kl = CfEvidentialHead.kl_to_uniform(torch.ones(3), torch.ones(3))
    assert torch.allclose(kl, torch.zeros(3), atol=1e-6)


def test_kl_to_uniform_is_positive_and_grows_with_evidence():
    """KL is non-negative, and a sharper Beta (more claimed evidence) is further from uniform —
    which is the whole mechanism by which the regularizer bounds precision."""
    kl_small = float(CfEvidentialHead.kl_to_uniform(torch.tensor([2.0]), torch.tensor([2.0])))
    kl_big = float(CfEvidentialHead.kl_to_uniform(torch.tensor([50.0]), torch.tensor([50.0])))
    assert 0.0 < kl_small < kl_big


def test_the_regularizer_pulls_alpha_and_beta_toward_one():
    """Descending the KL must actually MOVE the parameters toward the uniform — a term that is
    merely positive proves nothing about its direction."""
    raw = torch.tensor([3.0, 3.0], requires_grad=True)     # softplus(3)+1 ≈ 4.05
    start = float(torch.nn.functional.softplus(raw.detach()).max()) + 1.0
    for _ in range(2000):
        ab = torch.nn.functional.softplus(raw) + 1.0
        loss = CfEvidentialHead.kl_to_uniform(ab[0:1], ab[1:2]).sum()
        loss.backward()
        with torch.no_grad():
            raw -= 2.0 * raw.grad
            raw.grad = None
    final = torch.nn.functional.softplus(raw.detach()) + 1.0
    assert start > 4.0                                     # preconditions: it started far away
    assert float(final.max()) < 1.05, f"KL descent left α,β at {final.tolist()}"


def test_epistemic_std_matches_the_closed_form():
    """Beta(1,1) is Uniform(0,1), whose std is 1/sqrt(12) — an anchor with a known value."""
    s = float(CfEvidentialHead.epistemic_std(torch.tensor([1.0]), torch.tensor([1.0])))
    assert s == pytest.approx(1.0 / math.sqrt(12.0), rel=1e-6)
    # and a sharper posterior is NARROWER, which is the property the metric is read for.
    sharp = float(CfEvidentialHead.epistemic_std(torch.tensor([50.0]), torch.tensor([50.0])))
    assert sharp < s


# ── the extractor integration ─────────────────────────────────────────────────

def test_off_builds_no_head_and_on_adds_only_the_head(ek_and_space):
    ek, space, _total = ek_and_space
    off, on = _build(ek, space, False), _build(ek, space, True)
    assert off.cf_evid_head is None and off.cf_evidential is False
    assert on.cf_evid_head is not None and on.cf_evidential is True
    delta = (sum(p.numel() for p in on.parameters())
             - sum(p.numel() for p in off.parameters()))
    assert delta == sum(p.numel() for p in on.cf_evid_head.parameters()) > 0


def test_on_is_BIT_identical_in_pi_and_vf(ek_and_space):
    """The strong form, and the reason the head is built LAST.

    Adding a module mid-constructor shifts the initialization RNG stream for every module built
    after it, so an "optional" head can silently re-roll the whole network. Built last, and never
    called by the forward, ON must reproduce OFF's outputs BIT for bit from the same seed — not
    merely match in shape (which is all the win_prob/value_dist precedents claim).
    """
    ek, space, total = ek_and_space
    obs = {"observation": torch.zeros(3, total)}
    off, on = _build(ek, space, False), _build(ek, space, True)
    with torch.no_grad():
        a, b = off(obs), on(obs)
    assert all(torch.equal(x, y) for x, y in zip(a, b)), \
        "building the evidential head perturbed pi/vf — is it still built LAST in __init__?"


def test_the_head_is_not_called_by_the_forward(ek_and_space):
    """It is a training-side readout applied to the STASHED value_pooled, so the rollout pays
    nothing for it and no label can reach the acting path through it."""
    ek, space, total = ek_and_space
    fe = _build(ek, space, True)
    calls = []
    fe.cf_evid_head.register_forward_hook(lambda *_a: calls.append(1))
    with torch.no_grad():
        fe({"observation": torch.zeros(2, total)})
    assert calls == [], "the evidential head was called during the extractor forward"


def test_the_stashed_value_pooled_is_what_the_head_consumes(ek_and_space):
    """The training-side contract: `stash.value_pooled` exists after a forward and has the width
    the head expects. If that stash ever stops being written, the term would silently no-op."""
    ek, space, total = ek_and_space
    fe = _build(ek, space, True)
    with torch.no_grad():
        fe({"observation": torch.zeros(2, total)})
        alpha, beta = fe.cf_evid_head(fe.stash.value_pooled)
    assert fe.stash.value_pooled.shape[-1] == D_MODEL
    assert alpha.shape == (2,) and beta.shape == (2,)


# ── the v98 version gate ──────────────────────────────────────────────────────

def _ver(on=False):
    mappings = load_mappings()
    ek = Gen3ObservationEncoder(mappings).get_features_extractor_kwargs()
    ek["cf_evidential"] = on
    pk = {"features_extractor_class": Gen3FeaturesExtractor, "features_extractor_kwargs": ek,
          "net_arch": [512, 512]}
    return ModelVersion.from_layout_and_policy_kwargs(ek["layout"], pk)


def test_version_records_the_toggle():
    assert _ver(True).cf_evidential is True
    assert _ver(False).cf_evidential is False


@pytest.mark.parametrize("saved,current", [(False, True), (True, False)])
def test_mismatch_fatals(saved, current):
    with pytest.raises(ModelVersionError, match="cf_evidential"):
        _ver(current).check_compatible(_ver(saved))


@pytest.mark.parametrize("on", [False, True])
def test_matching_toggle_loads(on):
    _ver(on).check_compatible(_ver(on))       # no raise


def test_migration_defaults_a_v97_config_off():
    """A pre-v98 checkpoint could not have built the head — the module did not exist — so False is
    not a guess, it is the only possible past."""
    out = _migrate_config({"config_version": 97})
    assert out["cf_evidential"] is False
    assert out["config_version"] == MODEL_CONFIG_VERSION == 98


def test_a_recorded_on_config_round_trips():
    """The other migration leg: a config that already RECORDS the flag must survive untouched."""
    out = _migrate_config({"config_version": 98, "cf_evidential": True})
    assert out["cf_evidential"] is True and out["config_version"] == 98


def test_the_flag_is_in_the_registry_as_a_structural_cli_toggle():
    """It builds a MODULE from an extractor constructor kwarg, which is the registry's declared
    scope — the win_prob_mode / value_dist_mode precedent. Its two COEFFICIENTS are deliberately
    absent (training-only, the --opd-coef class)."""
    from agents.model.flag_registry import BY_NAME, Klass, Tier
    row = BY_NAME["cf_evidential"]
    assert row.tier is Tier.CLI and row.klass is Klass.STRUCTURAL and row.since == 98
    assert row.requires == ()
    assert "cf_evidential_coef" not in BY_NAME and "cf_evidential_reg" not in BY_NAME
    assert "cf_label_likelihood" not in BY_NAME
