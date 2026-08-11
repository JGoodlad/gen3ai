"""Unit tests for the seed-VICReg regularizer (gen3_seed_vicreg_v1, v62).

Two load-bearing tests:

* `test_constant_offset_cheat_is_caught_by_covariance_term` — the "covariance is actually WIRED"
  proof. The z_arch precedent: a VICReg whose covariance term was specified but never wired
  shipped, and ≈2/3 of the code's energy collapsed into one shared direction, discovered only
  post-hoc. Seeds that share one batch-varying vector and differ only by per-seed constant offsets
  satisfy the variance hinge PERFECTLY while carrying zero extra information; only the cross-seed
  correlation term sees it. If that term is ever dropped, dead-coded, or detached, this fails.
* `test_targets_are_scale_free` — the regression for the bug that killed gen-6's first launch.
  ABSOLUTE targets (a std hinge at γ=1.0) were unreachable on a readout whose whole signal has
  RMS 0.207, so `out_effective_rank` sat at exactly 1.000 for 2M steps while the hinge saturated.
  Both terms must now be invariant to a global rescale of the outputs; this test scales by 100×
  and asserts the loss is unchanged.
"""
import pytest
import torch

from agents.model.seed_vicreg import (
    SEED_VICREG_GAMMA_REL,
    assert_seed_vicreg_wirable,
    seed_vicreg_loss,
)

B, K, D = 32, 4, 16
_G = torch.Generator().manual_seed(0)


def _decorrelated_seeds(scale: float = 1.0) -> torch.Tensor:
    """k seeds, each reading its own independent direction — the healthy target state."""
    out = torch.zeros(B, K, D)
    for k in range(K):
        out[:, k, :] = torch.randn(B, D, generator=_G) * scale
    return out


def test_collapsed_seeds_pay_the_variance_hinge():
    one = torch.randn(B, 1, D, generator=_G)
    outputs = one.expand(B, K, D).clone()          # all k seeds identical per sample
    _, m = seed_vicreg_loss(outputs)
    # std across seeds is exactly 0 → the relative hinge pays the full gamma.
    assert m["value_seeds/vicreg_var_term"] == pytest.approx(SEED_VICREG_GAMMA_REL, rel=1e-4)
    assert m["value_seeds/out_std_rel"] == pytest.approx(0.0, abs=1e-6)


def test_healthy_seeds_cost_almost_nothing():
    loss, m = seed_vicreg_loss(_decorrelated_seeds())
    # Independent per-seed directions: cross-seed std ≈ the dim's own RMS ⇒ ratio ≈ 1 ≫ gamma,
    # and near-zero cross-seed correlation.
    assert m["value_seeds/out_std_rel"] > 0.5 * SEED_VICREG_GAMMA_REL * 2
    assert m["value_seeds/vicreg_cov_term"] < 0.05
    assert float(loss) < 0.05


def test_targets_are_scale_free():
    """THE GEN-6 REGRESSION. A 100x global rescale must not change either term — an absolute
    target is what made the first launch's hinge unreachable (module docstring)."""
    base = _decorrelated_seeds()
    small = base * 0.01
    big = base * 100.0
    l0, m0 = seed_vicreg_loss(base)
    l1, m1 = seed_vicreg_loss(small)
    l2, m2 = seed_vicreg_loss(big)
    assert float(l1) == pytest.approx(float(l0), rel=1e-4), "shrinking the readout changed the loss"
    assert float(l2) == pytest.approx(float(l0), rel=1e-4), "growing the readout changed the loss"
    for m in (m1, m2):
        assert m["value_seeds/out_std_rel"] == pytest.approx(m0["value_seeds/out_std_rel"], rel=1e-4)


def test_out_rms_watchdog_tracks_magnitude():
    """The relative target's one degenerate path (shrink everything) must be VISIBLE."""
    base = _decorrelated_seeds()
    _, m0 = seed_vicreg_loss(base)
    _, m1 = seed_vicreg_loss(base * 0.01)
    assert m1["value_seeds/out_rms"] == pytest.approx(0.01 * m0["value_seeds/out_rms"], rel=1e-4)


def test_realistic_collapsed_scale_still_pays_full_hinge():
    """The gen-5/gen-6 operating point in NUMBERS: RMS ~0.21, cross-seed std ~0.0015. Under the
    OLD absolute γ=1.0 this was 0.998 (saturated, unreachable); under the relative target it is a
    normal, closable gap."""
    shared = torch.randn(B, 1, D, generator=_G) * 0.207
    jitter = torch.randn(B, K, D, generator=_G) * 0.0015
    _, m = seed_vicreg_loss(shared.expand(B, K, D) + jitter)
    assert m["value_seeds/out_std_rel"] < 0.02                      # far from the 0.25 target
    assert m["value_seeds/vicreg_var_term"] > 0.9 * SEED_VICREG_GAMMA_REL
    assert m["value_seeds/vicreg_var_term"] <= SEED_VICREG_GAMMA_REL + 1e-6   # bounded by gamma


def test_constant_offset_cheat_is_caught_by_covariance_term():
    # THE COV GATE. shared batch-varying signal + per-seed constant offsets:
    #   variance term: std across seeds is the offsets' std — large ⇒ hinge ≈ 0 (cheat passes).
    #   covariance term: batch-centering kills the offsets, leaving k IDENTICAL batch-varying
    #   parts ⇒ cross-seed CORRELATION is ~1 ⇒ MUST be large positive.
    shared = torch.randn(B, 1, D, generator=_G) * 3.0
    offsets = torch.arange(K, dtype=torch.float32).view(1, K, 1) * 5.0
    outputs = shared.expand(B, K, D) + offsets
    _, m = seed_vicreg_loss(outputs)
    assert m["value_seeds/vicreg_var_term"] < 0.05, "offsets should satisfy the variance hinge"
    assert m["value_seeds/vicreg_cov_term"] > 0.9, (
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
    assert set(m) == {"value_seeds/vicreg_var_term", "value_seeds/vicreg_cov_term",
                      "value_seeds/out_std_rel", "value_seeds/out_rms"}
    assert all(isinstance(v, float) for v in m.values())


def test_shape_and_k_guards():
    with pytest.raises(AssertionError):
        seed_vicreg_loss(torch.zeros(B, D))
    with pytest.raises(AssertionError):
        seed_vicreg_loss(torch.zeros(B, 1, D))


def test_all_zero_outputs_do_not_nan():
    loss, m = seed_vicreg_loss(torch.zeros(B, K, D))
    assert torch.isfinite(loss) and all(v == v for v in m.values())


def test_wirable_assert_raises_without_readout():
    class _A:  # assembler with no seed readout (damage_op off)
        seed_readout = None

    class _FE:
        assembler = _A()

    class _P:
        features_extractor = _FE()

    with pytest.raises(RuntimeError, match="value-seed-vicreg-coef"):
        assert_seed_vicreg_wirable(_P())


def test_wirable_assert_passes_with_readout():
    class _A:
        seed_readout = object()

    class _FE:
        assembler = _A()

    class _P:
        features_extractor = _FE()

    assert_seed_vicreg_wirable(_P())  # no raise
