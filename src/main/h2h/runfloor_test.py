"""``main.h2h.runfloor`` — σ_h from a round-robin, the meter's variance subtracted EXACTLY.

The unbiasedness of the estimator is checked as an IDENTITY, not by simulation: average σ̂² over every sign
pattern of the edge noise (``±sqrt(v_e)`` on each edge, 2^E patterns) and the cross term vanishes, leaving
exactly ``sum_i (s_i - s̄)^2 / (K - 1)`` — so the test is deterministic and has no tolerance to tune."""
from __future__ import annotations

import itertools
import math

import pytest

from main.h2h import runfloor as RF


def noiseless(strengths, v=1.0, u=0.0):
    runs = list(strengths)
    return runs, [RF.EdgeRead(i, j, strengths[i] - strengths[j] + u, v, 1000) for i, j in itertools.combinations(runs, 2)]


def test_two_runs_reduce_to_the_notes_per_edge_formula():
    r = RF.sigma_h(["a", "b"], [RF.EdgeRead("a", "b", 4.0, 1.0)])
    assert r["df"] == 1 and r["sigma_hat_sq"] == pytest.approx((16.0 - 1.0) / 2.0)
    assert r["meter_variance_subtracted"] == pytest.approx(0.5)


def test_noiseless_strengths_are_recovered_centred_and_the_spread_is_exact():
    s = {"a": 3.0, "b": -1.0, "c": -2.0}
    runs, reads = noiseless(s, v=1e-12)
    r = RF.sigma_h(runs, reads)
    mean = sum(s.values()) / 3
    for k in s:
        assert r["strengths_pp"][k] == pytest.approx(s[k] - mean)
    assert r["raw_sd_of_strengths"] ** 2 == pytest.approx(sum((x - mean) ** 2 for x in s.values()) / 2)


@pytest.mark.parametrize("k", [3, 4])
def test_the_estimator_is_exactly_unbiased_over_the_sign_patterns_of_the_noise(k):
    s = {f"r{i}": x for i, x in enumerate([2.5, -1.0, 0.5, -3.0][:k])}
    runs = list(s)
    pairs = list(itertools.combinations(runs, 2))
    v = [0.4 + 0.1 * e for e in range(len(pairs))]                  # unequal edge variances
    mean = sum(s.values()) / k
    truth = sum((x - mean) ** 2 for x in s.values()) / (k - 1)
    ests = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(pairs)):
        reads = [RF.EdgeRead(i, j, s[i] - s[j] + sg * math.sqrt(ve), ve) for (i, j), sg, ve in zip(pairs, signs, v)]
        ests.append(RF.sigma_h(runs, reads)["sigma_hat_sq"])
    assert sum(ests) / len(ests) == pytest.approx(truth, rel=1e-12)


def test_the_seat_effect_is_subtracted_before_anything_else():
    s = {"a": 3.0, "b": -1.0, "c": -2.0}
    runs, reads = noiseless(s, v=1e-12, u=1.7)
    biased = RF.sigma_h(runs, reads, u=0.0)["strengths_pp"]
    fixed = RF.sigma_h(runs, reads, u=1.7)
    for k in s:
        assert fixed["strengths_pp"][k] == pytest.approx(s[k] - sum(s.values()) / 3)
    assert max(abs(biased[k] - fixed["strengths_pp"][k]) for k in s) > 0.1, "u must matter to be worth measuring"


def test_seat_effect_from_self_edges_and_from_reversals_agree_on_a_planted_u():
    u = 0.8
    selfs = [RF.EdgeRead("a", "a", u + 0.1, 0.25), RF.EdgeRead("b", "b", u - 0.1, 0.25)]
    got = RF.seat_effect(selfs)
    assert got["measured"] and got["u"] == pytest.approx(u) and got["se"] == pytest.approx(math.sqrt(0.125))
    both = [RF.EdgeRead("a", "b", 2.0 + u, 0.5), RF.EdgeRead("b", "a", -2.0 + u, 0.5)]
    r = RF.seat_effect_from_reversals(both)
    assert r["measured"] and r["u"] == pytest.approx(u) and r["n_pairs_both_ways"] == 1
    assert RF.seat_effect([RF.EdgeRead("a", "b", 1.0, 1.0)])["measured"] is False
    assert RF.seat_effect_from_reversals([RF.EdgeRead("a", "b", 1.0, 1.0)])["measured"] is False


def test_edges_of_one_pair_in_both_directions_combine_by_inverse_variance():
    reads = [RF.EdgeRead("a", "b", 4.0, 1.0), RF.EdgeRead("b", "a", -2.0, 1.0)]
    d, v = RF.combine_pairs(["a", "b"], reads)[("a", "b")]
    assert d == pytest.approx(3.0) and v == pytest.approx(0.5)
    heavy = [RF.EdgeRead("a", "b", 4.0, 0.1), RF.EdgeRead("b", "a", -2.0, 10.0)]
    assert RF.combine_pairs(["a", "b"], heavy)[("a", "b")][0] == pytest.approx((4.0 / 0.1 + 2.0 / 10.0) / (1 / 0.1 + 1 / 10.0))


def test_an_incomplete_round_robin_is_refused():
    with pytest.raises(ValueError, match="complete round-robin"):
        RF.sigma_h(["a", "b", "c"], [RF.EdgeRead("a", "b", 1.0, 1.0), RF.EdgeRead("a", "c", 1.0, 1.0)])


def test_an_estimate_inside_the_meter_noise_is_reported_as_such_and_floored():
    r = RF.sigma_h(["a", "b"], [RF.EdgeRead("a", "b", 0.3, 1.0)])
    assert r["inside_meter_noise"] and r["sigma_hat"] == 0.0 and r["sigma_hat_sq"] < 0


def test_the_interval_is_the_exact_chi_square_one_shifted_by_the_meter_term():
    from scipy import stats

    r = RF.sigma_h(["a", "b"], [RF.EdgeRead("a", "b", 10.0, 1.0)])
    s2, df, shift = 100.0 / 2, 1, 1.0 / 2
    lo = df * s2 / stats.chi2.ppf(0.975, df) - shift
    hi = df * s2 / stats.chi2.ppf(0.025, df) - shift
    assert r["sigma_ci"][0] == pytest.approx(math.sqrt(lo)) and r["sigma_ci"][1] == pytest.approx(math.sqrt(hi))
    assert r["sigma_upper_one_sided"] == pytest.approx(math.sqrt(df * s2 / stats.chi2.ppf(0.05, df) - shift))
    assert r["sigma_ci"][0] < r["sigma_hat"] < r["sigma_ci"][1] < r["sigma_upper_one_sided"] * 1000


def test_edge_read_takes_the_pp_scale_from_a_summary():
    e = RF.edge_read("a", "b", {"score": 0.5231, "se_pp": 0.42, "pairs": 5000})
    assert e.d == pytest.approx(2.31) and e.v == pytest.approx(0.1764) and e.pairs == 5000
    with pytest.raises(ValueError):
        RF.edge_read("a", "b", {"score": None, "se_pp": None, "pairs": 0})
