"""Unit tests for the seed-VICReg regularizer (gen3_seed_vicreg_v1, v62).

The load-bearing test is `test_constant_offset_cheat_is_caught_by_covariance_term` — the
"covariance is actually WIRED" proof. The z_arch precedent: a VICReg whose covariance term was
specified but never wired shipped, and ≈2/3 of the code's energy collapsed into one shared
direction, discovered only post-hoc. Seeds that share one batch-varying vector and differ only by
per-seed constant offsets satisfy the variance hinge PERFECTLY while carrying zero extra
information; only the cross-seed covariance term sees it. If that term is ever dropped, dead-coded,
or detached, this test fails.
"""
import math

import pytest
import torch

from agents.model.seed_vicreg import (
    SEED_VICREG_GAMMA,
    assert_seed_vicreg_wirable,
    seed_vicreg_loss,
)

B, K, D = 32, 4, 16
_G = torch.Generator().manual_seed(0)


def test_collapsed_seeds_pay_the_variance_hinge():
    one = torch.randn(B, 1, D, generator=_G)
    outputs = one.expand(B, K, D).clone()          # all k seeds identical per sample
    loss, m = seed_vicreg_loss(outputs)
    # std across seeds is exactly 0 → the hinge pays the full gamma.
    assert math.isclose(m["value_seeds/vicreg_var_term"], SEED_VICREG_GAMMA, rel_tol=1e-5)
    assert float(loss) > 0.9 * SEED_VICREG_GAMMA


def test_distinct_decorrelated_high_variance_seeds_cost_nothing():
    # Give each seed its own orthogonal direction with std >> gamma across the seed axis,
    # and batch-decorrelated content: loss ≈ 0 on both terms.
    outputs = torch.zeros(B, K, D)
    for k in range(K):
        outputs[:, k, k] = torch.randn(B, generator=_G) * 10.0
    loss, m = seed_vicreg_loss(outputs)
    # Variance hinge: per-dim std across seeds is large on the k active dims but ZERO on the
    # rest, so the hinge cannot reach 0 exactly — check the cov term is ~0 and the loss is
    # strictly below the collapsed case instead.
    assert m["value_seeds/vicreg_cov_term"] < 1e-2
    collapsed, _ = seed_vicreg_loss(outputs[:, :1, :].expand(B, K, D).clone())
    assert float(loss) < float(collapsed)


def test_constant_offset_cheat_is_caught_by_covariance_term():
    # THE COV GATE. shared batch-varying signal + per-seed constant offsets:
    #   variance term: std across seeds is the offsets' std — large ⇒ hinge ≈ 0 (cheat passes).
    #   covariance term: batch-centering kills the offsets, leaving k IDENTICAL batch-varying
    #   parts ⇒ cross-seed covariance is maximal ⇒ MUST be large positive.
    shared = torch.randn(B, 1, D, generator=_G) * 3.0
    offsets = torch.arange(K, dtype=torch.float32).view(1, K, 1) * 5.0
    outputs = shared.expand(B, K, D) + offsets
    _, m = seed_vicreg_loss(outputs)
    assert m["value_seeds/vicreg_var_term"] < 0.05, "offsets should satisfy the variance hinge"
    assert m["value_seeds/vicreg_cov_term"] > 1.0, (
        "the constant-offset cheat MUST be caught by the covariance term — if this fails, "
        "the covariance term is unwired/dead (the z_arch failure mode)"
    )


def test_gradient_flows_to_a_leaf():
    leaf = torch.randn(B, K, D, generator=_G, requires_grad=True)
    loss, _ = seed_vicreg_loss(leaf * 1.0)
    loss.backward()
    assert leaf.grad is not None and float(leaf.grad.abs().sum()) > 0.0


def test_metrics_are_detached_floats_and_loss_keeps_grad():
    leaf = torch.randn(B, K, D, generator=_G, requires_grad=True)
    loss, m = seed_vicreg_loss(leaf * 1.0)
    assert loss.requires_grad
    assert set(m) == {"value_seeds/vicreg_var_term", "value_seeds/vicreg_cov_term"}
    assert all(isinstance(v, float) for v in m.values())


def test_shape_and_k_guards():
    with pytest.raises(AssertionError):
        seed_vicreg_loss(torch.zeros(B, D))
    with pytest.raises(AssertionError):
        seed_vicreg_loss(torch.zeros(B, 1, D))


def test_wirable_assert_raises_without_readout():
    class _A:  # assembler with no seed readout (damage_op off)
        seed_readout = None

    class _FE:
        assembler = _A()

    class _P:
        features_extractor = _FE()

    with pytest.raises(RuntimeError, match="seed-vicreg-coef"):
        assert_seed_vicreg_wirable(_P())


def test_wirable_assert_passes_with_readout():
    class _A:
        seed_readout = object()

    class _FE:
        assembler = _A()

    class _P:
        features_extractor = _FE()

    assert_seed_vicreg_wirable(_P())  # no raise
