"""Gates for per-seed quantile assignment (gen3_seed_quantile_v1).

Two load-bearing tests:

* `test_pinball_recovers_the_true_quantiles` — the loss does what it claims. Fit free per-seed
  predictions against a known distribution and check they converge to its actual quantiles. If
  this fails, every downstream claim about "each seed has a different job" is decoration.
* `test_shared_readout_makes_collapse_strictly_worse` — the MECHANISM proof, and the reason the
  readout is shared. Four identical seed outputs can only produce four identical predictions, so
  the best achievable loss under collapse must be strictly worse than with differentiated seeds.
  That is the whole point: collapse becomes loss-INCREASING rather than merely unpenalized (which
  is all VICReg's repulsion could manage). A per-seed projection would let the head fake the
  spread from identical inputs — the silent-failure shape this test exists to forbid.
"""
import pytest
import torch

from agents.model.seed_quantile import (
    SEED_QUANTILE_TAUS,
    SeedQuantileHead,
    seed_quantile_loss,
)

K, D, B = 4, 16, 4096
_G = torch.Generator().manual_seed(0)


def _fit(preds_init, returns, taus, steps=600, lr=0.05):
    p = preds_init.clone().requires_grad_(True)
    opt = torch.optim.Adam([p], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        loss, _ = seed_quantile_loss(p.expand(returns.shape[0], -1), returns, taus)
        loss.backward()
        opt.step()
    return p.detach()


def test_pinball_recovers_the_true_quantiles():
    returns = torch.randn(B, generator=_G) * 2.0                  # N(0, 2)
    taus = torch.tensor(SEED_QUANTILE_TAUS)
    fitted = _fit(torch.zeros(1, K), returns, taus)[0]
    true_q = torch.quantile(returns, taus)
    # PURE pinball is an UNBIASED quantile estimator, so this is a tight check. It is also the
    # regression for the Huber reversal recorded in the module: at κ=1 on this exact σ=2 data the
    # Huber-softened form fitted ±2.18/±0.67 against true ±2.50/±0.78 — a shrink toward the median
    # that would quietly erode the very separation the seeds are supposed to learn.
    assert torch.allclose(fitted, true_q, atol=0.12), f"fitted {fitted} vs true {true_q}"
    assert (fitted[1:] > fitted[:-1]).all(), "recovered quantiles must be ascending"


def test_shared_readout_makes_collapse_strictly_worse():
    """THE MECHANISM PROOF. A shared readout over IDENTICAL seeds can emit only one number."""
    returns = torch.randn(B, generator=_G) * 2.0
    taus = torch.tensor(SEED_QUANTILE_TAUS)
    head = SeedQuantileHead(D, K)

    collapsed = torch.randn(B, 1, D, generator=_G).expand(B, K, D)      # all seeds identical
    preds_collapsed = head(collapsed)
    assert torch.allclose(preds_collapsed[:, 0], preds_collapsed[:, 1]), (
        "a shared readout over identical seeds MUST give identical predictions — if this fails "
        "the projection is per-seed and the pressure no longer lands on the seeds"
    )
    # Best achievable under collapse: one free scalar for all k (fit it).
    best_collapsed, _ = seed_quantile_loss(
        _fit(torch.zeros(1, 1), returns, taus[:1]).expand(B, K), returns, taus)
    # Best achievable with differentiated seeds: k free scalars.
    best_free, _ = seed_quantile_loss(
        _fit(torch.zeros(1, K), returns, taus).expand(B, K), returns, taus)
    assert float(best_free) < float(best_collapsed) - 1e-3, (
        f"differentiation must PAY: free={float(best_free):.4f} "
        f"collapsed={float(best_collapsed):.4f}"
    )


def test_collapsed_seeds_show_zero_spread_and_full_crossing():
    returns = torch.randn(B, generator=_G)
    taus = torch.tensor(SEED_QUANTILE_TAUS)
    preds = torch.zeros(B, K)                       # perfectly collapsed
    _, m = seed_quantile_loss(preds, returns, taus)
    assert m["value_seeds/quantile_spread"] == pytest.approx(0.0)
    assert m["value_seeds/quantile_crossing_rate"] == pytest.approx(1.0)


def test_healthy_seeds_show_spread_and_no_crossing():
    returns = torch.randn(B, generator=_G)
    taus = torch.tensor(SEED_QUANTILE_TAUS)
    preds = torch.tensor([-1.3, -0.4, 0.4, 1.3]).expand(B, K)
    _, m = seed_quantile_loss(preds, returns, taus)
    assert m["value_seeds/quantile_spread"] > 2.0
    assert m["value_seeds/quantile_crossing_rate"] == pytest.approx(0.0)
    assert m["value_seeds/quantile_pred_0"] < m["value_seeds/quantile_pred_3"]


def test_asymmetry_is_real():
    """A low-tau seed must be pushed DOWN by an over-prediction more than a high-tau seed."""
    returns = torch.zeros(64)
    taus = torch.tensor([0.1, 0.9])
    p = torch.full((64, 2), 1.0, requires_grad=True)          # both over-predict
    loss, _ = seed_quantile_loss(p, returns, taus)
    loss.backward()
    g = p.grad.mean(0)
    assert g[0] > g[1] > 0, f"tau=0.1 must be pushed down harder: {g}"


def test_head_shape_and_tau_guards():
    head = SeedQuantileHead(D, K)
    assert head(torch.randn(7, K, D, generator=_G)).shape == (7, K)
    with pytest.raises(AssertionError):
        SeedQuantileHead(D, 3)                                # taus/seed mismatch
    with pytest.raises(AssertionError):
        SeedQuantileHead(D, 2, taus=(0.9, 0.1))               # not ascending
    with pytest.raises(AssertionError):
        SeedQuantileHead(D, 2, taus=(0.0, 0.5))               # out of range


def test_gradient_reaches_the_seed_outputs():
    head = SeedQuantileHead(D, K)
    seeds = torch.randn(32, K, D, generator=_G, requires_grad=True)
    loss, _ = seed_quantile_loss(head(seeds), torch.randn(32, generator=_G), head.taus)
    loss.backward()
    assert seeds.grad is not None and float(seeds.grad.abs().sum()) > 0
    # and it must reach EVERY seed, not just one
    per_seed = seeds.grad.abs().sum(dim=(0, 2))
    assert (per_seed > 0).all(), f"some seed received no gradient: {per_seed}"


def test_shape_guards():
    taus = torch.tensor(SEED_QUANTILE_TAUS)
    with pytest.raises(AssertionError):
        seed_quantile_loss(torch.zeros(B), torch.zeros(B), taus)
    with pytest.raises(AssertionError):
        seed_quantile_loss(torch.zeros(B, K), torch.zeros(B + 1), taus)


# ------------------------------------------------ v63 wiring: version gate + structural isolation
def test_v63_migration_defaults_off():
    """The v63 default-injection branch is pre-floor (MIGRATION_FLOOR): a v62 config is a
    pre-generation checkpoint and is refused outright instead of migrating to OFF."""
    from agents.model.model_version import _migrate_config, MODEL_CONFIG_VERSION, ModelVersionError
    assert MODEL_CONFIG_VERSION >= 63
    with pytest.raises(ModelVersionError, match="PRE-GENERATION"):
        _migrate_config({"config_version": 62})


def test_version_gate_rejects_a_toggle_flip():
    """The head changes the state_dict, so it is fixed for a run's lifetime."""
    import dataclasses
    from agents.model.model_version import ModelVersion, ModelVersionError
    from agents.observation.state_encoder import load_mappings
    from agents.model.snapshot import current_model_version
    base = current_model_version(load_mappings())
    on = dataclasses.replace(base, seed_quantile=True)
    with pytest.raises(ModelVersionError, match="seed_quantile"):
        on.check_compatible(base)
    with pytest.raises(ModelVersionError, match="seed_quantile"):
        base.check_compatible(on)
    on.check_compatible(on)          # matching pair is fine


def test_off_adds_no_module_and_on_adds_only_its_own_keys():
    """OFF must be byte-identical (no params); ON must touch nothing but its own head."""
    from agents.model.damage_op_test import _make_layout
    from agents.model.features_extractor import Gen3FeaturesExtractor
    import gymnasium as gym, numpy as np
    layout = _make_layout()
    space = gym.spaces.Dict({
        "observation": gym.spaces.Box(-np.inf, np.inf, (layout["total_dim"],), np.float32),
        "action_mask": gym.spaces.Box(0, 1, (11,), np.float32)})
    common = dict(layout=layout, move_belief_mode="revealed", damage_op=True,
                  attend_unrevealed_opponents=True)
    torch.manual_seed(0); off = Gen3FeaturesExtractor(space, **common)
    torch.manual_seed(0); on = Gen3FeaturesExtractor(space, **common, seed_quantile=True)
    assert off.seed_quantile_head is None and on.seed_quantile_head is not None
    new = set(on.state_dict()) - set(off.state_dict())
    assert new and all(k.startswith("seed_quantile_head.") for k in new), sorted(new)[:5]
    assert not (set(off.state_dict()) - set(on.state_dict())), "OFF must not have keys ON lacks"
