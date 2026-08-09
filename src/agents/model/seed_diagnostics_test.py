"""Pins for the seed-collapse diagnostics (OpTensors step-4 TB contract).

The metric must SEE the z_arch failure mode: collapsed seeds → query_cos ~1 / effective rank
~1; healthy orthogonal seeds → query_cos ~0 / effective rank ~k. If these pins hold, a
collapsing k-seed readout is visible on TB from step 0 and the pre-registered VICReg trigger
(module docstring) can fire on evidence instead of vibes."""
import torch

from agents.model.seed_diagnostics import seed_collapse_diagnostics


def test_collapsed_seeds_read_as_collapsed():
    k, d, b = 4, 16, 8
    q = torch.randn(1, d).expand(k, d).contiguous()          # identical queries
    out = torch.randn(b, 1, d).expand(b, k, d).contiguous()  # identical outputs per seed
    m = seed_collapse_diagnostics(q, out)
    assert m["seeds/query_cos"] > 0.99
    assert m["seeds/out_cos"] > 0.99
    assert m["seeds/out_effective_rank"] < 1.05, "identical seeds must read ~1 effective seed"
    assert m["seeds/out_var"] < 1e-10


def test_orthogonal_seeds_read_as_healthy():
    k, b = 4, 8
    q = torch.eye(k)                                          # orthogonal queries [k, k]
    out = torch.zeros(b, k, k)
    out[:, range(k), range(k)] = 1.0                          # orthogonal outputs per seed
    out = out + 0.01 * torch.randn(b, k, k)                   # a whiff of noise
    m = seed_collapse_diagnostics(q, out)
    assert m["seeds/query_cos"] < 0.05
    assert m["seeds/out_cos"] < 0.2
    assert m["seeds/out_effective_rank"] > 0.7 * k, "orthogonal seeds must read ~k effective seeds"
    assert m["seeds/out_var"] > 0.01


def test_partial_collapse_sits_between():
    """Two distinct directions shared by 4 seeds → effective rank ~2 (the z_arch shape:
    a 'shared dominant direction' reads well below k without being exactly 1)."""
    b, d = 16, 12
    dir1, dir2 = torch.zeros(d), torch.zeros(d)
    dir1[0], dir2[1] = 1.0, 1.0
    q = torch.stack([dir1, dir1, dir2, dir2])
    out = torch.stack([dir1, dir1, dir2, dir2])[None].expand(b, 4, d).contiguous()
    out = out + 0.01 * torch.randn(b, 4, d)
    m = seed_collapse_diagnostics(q, out)
    assert 1.3 < m["seeds/out_effective_rank"] < 2.7
    assert 0.2 < m["seeds/query_cos"] < 0.8


def test_shapes_are_enforced():
    import pytest
    with pytest.raises(AssertionError):
        seed_collapse_diagnostics(torch.randn(2, 3, 4), torch.randn(5, 2, 4))
    with pytest.raises(AssertionError):
        seed_collapse_diagnostics(torch.randn(3, 4), torch.randn(5, 2, 4))
