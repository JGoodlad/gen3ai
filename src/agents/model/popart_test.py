"""Unit tests for the PopArt value-target normalizer (no SB3, no battle)."""
import torch as th
from torch import nn

from agents.model.popart import PopArtNormalizer


def test_normalize_denormalize_are_inverse():
    pop = PopArtNormalizer()
    layer = nn.Linear(4, 1)
    pop.update(th.tensor([10.0, -30.0, 50.0, 0.0]), layer)
    x = th.randn(7)
    assert th.allclose(pop.denormalize(pop.normalize(x)), x, atol=1e-5)


def test_stats_track_target_distribution():
    pop = PopArtNormalizer()
    layer = nn.Linear(4, 1)
    targets = th.randn(20000) * 23.0 - 5.0  # mean ≈ -5, std ≈ 23 (like real returns)
    pop.update(targets, layer)
    # First update initializes directly to the batch stats.
    assert abs(float(pop.mu) - float(targets.mean())) < 1e-3
    assert abs(float(pop.sigma) - float(targets.std(unbiased=False))) < 0.1


def test_pop_preserves_outputs_across_update():
    """THE load-bearing invariant: a (mu, sigma) update must not change de-normalized outputs."""
    th.manual_seed(0)
    pop = PopArtNormalizer()
    layer = nn.Linear(8, 1)
    x = th.randn(16, 8)

    pop.update(th.randn(5000) * 20.0 + 3.0, layer)            # initialize stats
    before = pop.denormalize(layer(x))
    pop.update(th.randn(5000) * 35.0 - 10.0, layer)           # shift the distribution
    after = pop.denormalize(layer(x))

    assert th.allclose(before, after, atol=1e-4), "POP must preserve de-normalized outputs"
    # And the stats actually moved (otherwise the test is vacuous). EMA (beta=0.1) shifts sigma
    # only ~10% toward the wider 2nd batch, so it rises from ~20 to ~22 (not all the way to 35).
    assert float(pop.sigma) > 21.0


def test_sigma_floor_on_constant_targets():
    pop = PopArtNormalizer(sigma_floor=1e-2)
    layer = nn.Linear(2, 1)
    pop.update(th.full((100,), 7.0), layer)  # zero-variance batch
    assert float(pop.sigma) >= 1e-2 - 1e-6   # floored, never 0 (allow float32 rounding of 1e-2)
    assert th.isfinite(pop.normalize(th.tensor(7.0)))


def test_ema_smooths_after_init():
    pop = PopArtNormalizer(beta=0.1)
    layer = nn.Linear(2, 1)
    pop.update(th.zeros(1000), layer)            # init mu≈0
    pop.update(th.full((1000,), 100.0), layer)   # big jump; EMA should move only ~10%
    assert 5.0 < float(pop.mu) < 20.0            # ≈ 0.1 * 100, not a full jump to 100


def test_buffers_are_in_state_dict():
    pop = PopArtNormalizer()
    keys = set(pop.state_dict().keys())
    assert {"mu", "sigma", "nu", "initialized"} <= keys  # save/restore across checkpoints
    assert sum(p.numel() for p in pop.parameters()) == 0  # no learnable params
