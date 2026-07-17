"""Unit tests for the effective-rank probe (rank_metrics.py)."""
import numpy as np

from agents.training.rank_metrics import effective_rank, rank_probe


def test_full_rank_isotropic_is_high():
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((2000, 16))          # isotropic → all 16 dims equally used
    r = effective_rank(Z)
    assert r["pr"] > 13.0                          # participation ratio ≈ D (16)
    assert r["effrank"] > 13.0
    assert r["n90"] >= 13 and r["n95"] >= 14 and r["n99"] >= 15


def test_rank_one_collapses_to_one():
    rng = np.random.default_rng(1)
    u = rng.standard_normal((2000, 1))
    v = rng.standard_normal((1, 16))
    Z = u @ v                                      # exact rank-1 (plus a constant offset)
    Z += rng.standard_normal((1, 16))              # a mean shift — centering removes it
    r = effective_rank(Z)
    assert r["pr"] < 1.5                            # one dominant direction
    assert r["effrank"] < 1.5
    assert r["n90"] == 1 and r["n95"] == 1 and r["n99"] == 1


def test_low_rank_between():
    rng = np.random.default_rng(2)
    # 3 strong directions + tiny noise → pr ≈ 3, most variance in 3 dims.
    base = rng.standard_normal((2000, 3)) * np.array([5.0, 4.0, 3.0])
    mix = rng.standard_normal((3, 16))
    Z = base @ mix + 0.01 * rng.standard_normal((2000, 16))
    r = effective_rank(Z)
    assert 2.0 < r["pr"] < 4.5
    assert r["n90"] <= 3


def test_degenerate_returns_zeros():
    assert effective_rank(np.zeros((10, 8)))["pr"] == 0.0        # zero variance
    assert effective_rank(np.ones((1, 8)))["n90"] == 0          # <2 rows


class _NonGen3Extractor:
    """No team_transformer / cls_pool → the probe must no-op gracefully."""


def test_rank_probe_non_gen3_returns_empty():
    assert rank_probe(_NonGen3Extractor(), obs={}, extract_features_fn=lambda o: (None, None)) == {}


def test_effective_rank_keys_present():
    r = effective_rank(np.random.default_rng(3).standard_normal((100, 8)))
    assert set(r) == {"pr", "effrank", "n90", "n95", "n99"}
