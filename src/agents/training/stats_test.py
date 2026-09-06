"""Unit tests for `agents.training.stats` — the shared small-sample statistics.

These are the load-bearing tests of the whole counterfactual programme's arithmetic, which is why
they live with the estimators rather than with any one instrument that calls them.
``sd_true_excess`` is the programme's PRIMARY meter, and an estimator that reports structure where
there is none would license a lever on noise — so it is validated **at zero true effect** (a
synthetic cell whose entire spread IS the binomial floor must return ~0) as well as at a known
nonzero one.

Moved here (2026-09-06) with the functions themselves, out of `cf_audit_test.py`.
"""

from __future__ import annotations

import numpy as np
import pytest

from agents.training.stats import (_is_flat, _ranks, cluster_bootstrap_ci,
                                   cluster_bootstrap_diff_ci, sd_true_excess, spearman, wilson_ci)


# --------------------------------------------------------------------- Wilson

def test_wilson_ci_is_not_degenerate_at_zero_wins():
    lo, hi = wilson_ci(0, 8)
    assert lo == 0.0 and 0.0 < hi < 1.0, "a normal approximation would give the useless [0, 0]"
    lo, hi = wilson_ci(8, 8)
    assert hi == 1.0 and 0.0 < lo < 1.0


def test_wilson_ci_brackets_the_point_estimate_and_narrows_with_n():
    lo, hi = wilson_ci(4, 8)
    assert lo < 0.5 < hi
    lo2, hi2 = wilson_ci(400, 800)
    assert (hi2 - lo2) < (hi - lo)


def test_wilson_ci_of_no_samples_is_maximally_uninformative():
    assert wilson_ci(0, 0) == (0.0, 1.0)


# ------------------------------------------------------- sd_true_excess (THE meter)

def test_sd_true_excess_returns_zero_when_the_spread_IS_the_binomial_noise():
    """ZERO TRUE EFFECT. Every state in the cell shares one true p, so all the observed
    spread of the R-rollout means is sampling noise and there is nothing to resolve. An
    estimator that reports excess here would license a lever on noise."""
    rng = np.random.default_rng(0)
    R = 8
    excesses = []
    for p in (0.2, 0.5, 0.85):
        for _ in range(40):
            mc = rng.binomial(R, p, size=300) / R
            excesses.append(sd_true_excess(mc, R)["sd_true_excess"])
    # The estimator is unbiased in VARIANCE and clamped at 0, so the sd reads slightly
    # positive on average — and it carries the sampling noise of a variance estimate
    # (relative sd ~sqrt(2/n)), which at n=300 puts an occasional cell near 0.1. The
    # property that matters is therefore a DISTRIBUTIONAL one, stated against the real
    # spreads the G0 map reports (0.11-0.36): spurious excess is small on average and its
    # worst cell never reaches the smallest genuine signal.
    assert np.mean(excesses) < 0.03, f"mean spurious sd_true_excess {np.mean(excesses):.4f}"
    assert np.percentile(excesses, 95) < 0.08, f"p95 {np.percentile(excesses, 95):.4f}"
    assert max(excesses) < 0.11, f"max {max(excesses):.4f}"


def test_sd_true_excess_recovers_a_KNOWN_spread():
    """Nonzero true effect: true p is spread with sd 0.20, and the estimator must find it
    after subtracting the R=8 floor (which alone is ~0.17 and would otherwise swamp it)."""
    rng = np.random.default_rng(1)
    R = 8
    true_p = np.clip(rng.normal(0.5, 0.20, size=4000), 0.01, 0.99)
    mc = rng.binomial(R, true_p) / R
    st = sd_true_excess(mc, R)
    assert st["sd_observed"] > st["sd_true_excess"] > 0.17, st
    assert abs(st["sd_true_excess"] - 0.20) < 0.03, st
    assert 0.4 < st["frac_variance_real"] < 0.75, st


def test_sd_true_excess_is_degenerate_on_a_tiny_cell():
    st = sd_true_excess([0.5, 0.25], 8)
    assert st["sd_true_excess"] is None and st["n"] == 2


def test_sd_true_excess_weights_recombine_subcells_at_population_shares():
    """A cell sampled 50/50 from two sub-populations that occur 90/10 must be recombined at
    the POPULATION shares — otherwise this probe's own oversampling inflates the spread."""
    a = [0.9] * 50                      # the common sub-population, tight
    b = [0.1] * 50                      # the rare one, far away
    unweighted = sd_true_excess(a + b, 8)["sd_true_excess"]
    weighted = sd_true_excess(a + b, 8, weights=[0.9 / 50] * 50 + [0.1 / 50] * 50)["sd_true_excess"]
    assert weighted < unweighted, (weighted, unweighted)


# --------------------------------------------------------------- clustered CI

def test_cluster_bootstrap_ci_is_wider_than_a_state_level_ci_when_battles_correlate():
    """Decisions inside a battle are not independent. Resampling STATES would understate the
    width; resampling BATTLES is the honest unit (the pooled-correlation Simpson lesson)."""
    rng = np.random.default_rng(3)
    values, clusters = [], []
    for b in range(12):                                  # a strong per-battle offset
        off = rng.normal(0, 1.0)
        for _ in range(20):
            values.append(off + rng.normal(0, 0.05))
            clusters.append(f"b{b}")
    lo_c, hi_c = cluster_bootstrap_ci(values, clusters, draws=800, seed=1)
    lo_s, hi_s = cluster_bootstrap_ci(values, [f"s{i}" for i in range(len(values))],
                                      draws=800, seed=1)
    assert (hi_c - lo_c) > 3 * (hi_s - lo_s)


def test_cluster_bootstrap_ci_refuses_a_single_cluster():
    assert cluster_bootstrap_ci([1.0, 2.0], ["b", "b"]) == (None, None)


# ------------------------------------------- the DIFFERENCE of means (conviction-class readout)

def _two_arms(mean_a, mean_b, n_batt_a, n_batt_b, per_battle=8, sd=0.02, seed=11):
    """Two disjoint battle-clustered arms with KNOWN means. Deliberately lopsided in battle
    count, which is where the old pooled construction came apart."""
    rng = np.random.default_rng(seed)

    def arm(mean, n_batt, tag):
        vals, cls = [], []
        for b in range(n_batt):
            for _ in range(per_battle):
                vals.append(mean + rng.normal(0, sd))
                cls.append(f"{tag}{b}")
        return vals, cls
    return arm(mean_a, n_batt_a, "L"), arm(mean_b, n_batt_b, "W")


def test_cluster_bootstrap_diff_ci_recovers_a_KNOWN_difference_and_brackets_it():
    """Synthetic arms at +0.30 and +0.10 ⇒ the difference is +0.20, and the interval must contain
    both the truth and its own point estimate — at a 3:1 arm imbalance, which is the real shape."""
    (va, ca), (vb, cb) = _two_arms(0.30, 0.10, n_batt_a=30, n_batt_b=10)
    point, lo, hi = cluster_bootstrap_diff_ci(va, ca, vb, cb, draws=800, seed=1)
    assert point == pytest.approx(0.20, abs=0.01)
    assert lo <= point <= hi, f"the CI [{lo}, {hi}] does not contain its own estimate {point}"
    assert lo <= 0.20 <= hi                                  # …and it covers the truth


def test_cluster_bootstrap_diff_ci_beats_the_POOLED_construction_it_replaced():
    """THE DEFECT, reproduced. The readout used to take its point estimate from
    ``mean(a) − mean(b)`` and its CI from ``cluster_bootstrap_ci(a + [−x for x in b])`` — the mean
    of the CONCATENATION, i.e. ``(Σa − Σb)/(n_a + n_b)``, a size-weighted pooled mean. The two
    coincide only at equal arm sizes; at the real imbalance they diverge far enough that the
    published interval excluded its own point estimate (the observed +0.205 vs [+0.070, +0.158]).
    Nothing about that reads as broken — it reads as a precise result.

    The cleanest demonstration is two arms with the SAME mean at 3:1 sizes: the difference is
    exactly 0, while the pooled statistic is ``(3m − m)/4 = m/2`` and confidently reports an effect
    that does not exist."""
    (va, ca), (vb, cb) = _two_arms(0.30, 0.30, n_batt_a=30, n_batt_b=10)
    point = float(np.mean(va) - np.mean(vb))
    assert point == pytest.approx(0.0, abs=0.01)             # the truth: no difference

    pooled = list(va) + [-x for x in vb]
    assert float(np.mean(pooled)) == pytest.approx(0.15, abs=0.01), (   # (3m − m)/4, not m − m
        "the pooled construction is supposed to invent an effect here — if it no longer does, "
        "this test has stopped reproducing the defect it guards")
    old_lo, old_hi = cluster_bootstrap_ci(pooled, list(ca) + list(cb), draws=800, seed=1)
    assert not (old_lo <= point <= old_hi), (
        f"the old CI [{old_lo}, {old_hi}] is supposed to exclude the true difference {point}")

    _p, lo, hi = cluster_bootstrap_diff_ci(va, ca, vb, cb, draws=800, seed=1)
    assert lo <= point <= hi and lo <= 0.0 <= hi             # …the honest statistic finds none


def test_cluster_bootstrap_diff_ci_refuses_when_an_ARM_has_one_cluster():
    """Same refusal convention as the single-arm CI: the point estimate still stands (it needs no
    resampling), but an un-resamplable arm gets no interval rather than a fake-narrow one."""
    point, lo, hi = cluster_bootstrap_diff_ci([0.3, 0.4], ["b", "b"], [0.1, 0.2], ["w1", "w2"])
    assert point == pytest.approx(0.2) and (lo, hi) == (None, None)
    assert cluster_bootstrap_diff_ci([], [], [0.1], ["w"]) == (None, None, None)



def test_spearman_is_none_when_a_side_is_FLAT_not_zero():
    """"Wide everywhere" and "width unrelated to blur" are DIFFERENT findings, and the more damning
    one is the first. A constant input must not be reported as a correlation of 0."""
    assert spearman([1.0, 1.0, 1.0, 1.0], [1.0, 2.0, 3.0, 4.0]) is None
    assert spearman([1.0, 2.0, 3.0], [1.0, 2.0]) is None          # mismatched / too short
    assert spearman([1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]) == 1.0
    assert spearman([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]) == -1.0
    # RANK, not Pearson: a monotone but wildly nonlinear relation is still a perfect 1.0.
    assert spearman([1.0, 2.0, 3.0, 4.0], [1.0, 10.0, 1e3, 1e9]) == 1.0


# ------------------------------------------------- the rank transform and its degeneracy test

def test_ranks_share_a_tie_so_a_constant_row_cannot_score_an_artifact():
    """Average ranks, ties shared. Without the averaging, a row of equal values would be ordered
    by array position and correlate perfectly with anything sorted the same way."""
    r = _ranks(np.array([5.0, 5.0, 1.0, 9.0]))
    assert list(r) == [1.5, 1.5, 0.0, 3.0]
    assert list(_ranks(np.array([3.0, 3.0, 3.0]))) == [1.0, 1.0, 1.0]


def test_is_flat_is_RELATIVE_because_a_weighted_average_is_constant_to_one_ulp():
    """The reason `spearman` does not test ``std() == 0``: a constant that arrives through a
    weighted average is equal to ~1 ulp, not exactly — and an exact test lets it through to
    `corrcoef`, which divides by ~1e-17 and reports a confident correlation of pure float noise."""
    assert _is_flat(np.array([1.0, 1.0, 1.0]))
    assert _is_flat(np.array([1.0, 1.0 + 1e-16, 1.0 - 1e-16]))     # constant to a ulp
    assert not _is_flat(np.array([1.0, 1.0001, 1.0]))
    assert _is_flat(np.array([0.0, 0.0]))                          # the max-is-zero guard
    # …and the whole point: the near-flat row must REFUSE rather than report noise.
    assert spearman([1.0, 1.0 + 1e-16, 1.0 - 1e-16], [1.0, 2.0, 3.0]) is None
