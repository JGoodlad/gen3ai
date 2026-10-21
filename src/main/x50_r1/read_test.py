"""X50 R1's read: the registered rule and the cluster bootstrap (pure; no model)."""
from __future__ import annotations

import numpy as np

from main.x50_r1.read import MARGIN, boot_means, rule


def test_the_registered_rule_reads_the_ci_against_the_margin():
    assert rule(MARGIN + 0.01, 0.4) == "GAP"
    assert rule(-MARGIN + 0.01, MARGIN - 0.01) == "NO GAP"
    assert rule(0.0, MARGIN + 0.01) == "INCONCLUSIVE"
    assert rule(-0.3, -MARGIN - 0.01) == "INCONCLUSIVE"     # the head reads humans BETTER: not a gap


def test_the_bootstrap_resamples_whole_battles():
    # three battles: two of one row at 0, one of nine rows at 1. Three battle draws, k of them the nine-row
    # battle, give 9k / (9k + 3 - k): only {0, 9/11, 18/19, 1} — a per-row resample would give other values
    x = np.array([0.0, 0.0] + [1.0] * 9)
    cl = np.array([0, 1] + [2] * 9)
    b = boot_means(x, cl, np.random.default_rng(0))
    allowed = np.array([0.0, 9 / 11, 18 / 19, 1.0])
    assert len(b) == 2000 and all(np.isclose(allowed, v).any() for v in b)
    assert np.isclose(boot_means(np.ones(5), np.arange(5), np.random.default_rng(1)), 1.0).all()
