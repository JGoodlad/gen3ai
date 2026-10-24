"""X50 rung 1b R0's read: the registered rule with rule 8's band, and the JOINT battle bootstrap (pure; no model)."""
from __future__ import annotations

import numpy as np

from main.x50_r1.read_r0 import BAND, FIX_BAR, REDUCE_BAR, fixed_rule, gap_rule, joint_boot


def test_the_r0_rule_reads_the_upper_end_against_the_bar_with_rule_8s_band():
    assert fixed_rule(FIX_BAR - BAND - 0.001, -1.0) == "FIXED"
    assert fixed_rule(FIX_BAR - BAND / 2, -1.0) == "NOT-FIXED"            # within 0.002 of the bar: conservative
    assert fixed_rule(FIX_BAR + 0.1, REDUCE_BAR + BAND + 0.001) == "REDUCED"
    assert fixed_rule(FIX_BAR + 0.1, REDUCE_BAR + BAND / 2) == "NOT-FIXED"
    assert fixed_rule(FIX_BAR + 0.1, 0.0) == "NOT-FIXED"


def test_r1s_gap_rule_reread_with_the_band():
    assert gap_rule(0.05 + BAND + 0.001, 0.3) == "GAP"
    assert gap_rule(0.05 + BAND / 2, 0.3) == "INCONCLUSIVE"
    assert gap_rule(-0.04, 0.04) == "NO GAP"
    assert gap_rule(-0.049, 0.04) == "INCONCLUSIVE"


def test_the_joint_bootstrap_draws_the_same_battles_for_every_population():
    # population A is 1 on battle 0 and 0 on battle 1; B is the same rows: their paired difference is 0 in EVERY
    # resample only if both read the SAME battle draw (an independent resample would scatter it)
    b = np.array([0, 1])
    got = joint_boot({"A": (np.array([1.0, 0.0]), b), "B": (np.array([1.0, 0.0]), b)}, np.random.default_rng(0))
    assert np.allclose(got["A"] - got["B"], 0.0) and got["A"].std() > 0
    # a population absent from some battles is read over its own rows in the drawn ones
    got = joint_boot({"A": (np.array([2.0, 4.0]), b), "C": (np.array([3.0]), np.array([1]))},
                     np.random.default_rng(1))
    assert np.isfinite(got["C"]).sum() > 0
