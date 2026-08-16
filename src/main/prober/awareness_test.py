"""awareness.py — the 'did it KNOW?' verdict fold, pinned on synthetic distributions.

Every scenario builds per-decision atom distributions by hand (a Gaussian bump placed on the
support), so each verdict definition is asserted against a trajectory whose ground truth is
constructed, not inferred: an aware loss, a blind loss, the stall signature (mean positive
while the tail piles up), the sustained-suffix rule (a late rally resets knew_by_turn), and
the popart denorm fit recovered from the trace's own scalar values.
"""
import numpy as np
import pytest

from main.prober.awareness import AwarenessVerdict, build_awareness, fit_denorm

SUPPORT = (-12.0, 12.0, 51)
Z = np.linspace(*SUPPORT[:2], SUPPORT[2])


def _bump(center: float, width: float = 1.5) -> np.ndarray:
    p = np.exp(-0.5 * ((Z - center) / width) ** 2)
    return p / p.sum()


def _mix(centers_weights, width: float = 1.5) -> np.ndarray:
    p = np.zeros_like(Z)
    for c, w in centers_weights:
        p += w * _bump(c, width)
    return p / p.sum()


def test_aware_loss_gets_onset_and_lead_time():
    # optimistic for 3 decisions, then mass collapses below zero from turn 4 onward
    dists = [_bump(6), _bump(5), _bump(4), _bump(-5), _bump(-7), _bump(-8)]
    v = build_awareness(dists, [1, 2, 3, 4, 5, 6], "loss", SUPPORT)
    assert isinstance(v, AwarenessVerdict)
    assert v.knew_by_turn == 4
    assert v.lead_time == 2          # last decision turn 6 − onset 4
    assert not v.blind_loss
    assert v.p_loss[0] < 0.1 and v.p_loss[-1] > 0.9


def test_blind_loss_never_sees_it_coming():
    dists = [_bump(6)] * 5
    v = build_awareness(dists, [1, 2, 3, 4, 5], "loss", SUPPORT)
    assert v.knew_by_turn is None and v.lead_time is None
    assert v.blind_loss


def test_a_win_is_never_a_blind_loss():
    dists = [_bump(6)] * 5
    v = build_awareness(dists, [1, 2, 3, 4, 5], "win", SUPPORT)
    assert not v.blind_loss


def test_sustained_rule_a_rally_resets_the_onset():
    # pessimistic early, rallies above the bar at turn 3, collapses again from turn 5:
    # knew_by_turn must be 5 (the last all-the-way-to-the-end suffix), not 1.
    dists = [_bump(-6), _bump(-6), _bump(6), _bump(6), _bump(-6), _bump(-6)]
    v = build_awareness(dists, [1, 2, 3, 4, 5, 6], "loss", SUPPORT)
    assert v.knew_by_turn == 5


def test_stall_signature_mean_positive_tail_growing():
    # a bimodal drift: the mean stays POSITIVE throughout (0.65·7 − 0.35·11.5 ≈ +0.5) while
    # bottom-atom mass grows to 35% — the exact shape a scalar critic cannot surface
    dists = [
        _bump(6),
        _mix([(7, 0.9), (-11.5, 0.1)]),
        _mix([(7, 0.75), (-11.5, 0.25)]),
        _mix([(7, 0.65), (-11.5, 0.35)]),
    ]
    v = build_awareness(dists, [1, 2, 3, 4], "loss", SUPPORT)
    assert v.mean_tail_divergence >= 0.25
    assert v.divergence_turn == 4
    # and a clean optimistic battle shows ~none
    clean = build_awareness([_bump(6)] * 4, [1, 2, 3, 4], "loss", SUPPORT)
    assert clean.mean_tail_divergence < 0.05


def test_skips_missing_rows_and_requires_two():
    dists = [None, _bump(6), None, _bump(-6)]
    v = build_awareness(dists, [1, 2, 3, 4], "loss", SUPPORT)
    assert v is not None and v.n_decisions == 2 and v.turns == (2, 4)
    assert build_awareness([_bump(6), None, None], [1, 2, 3], "loss", SUPPORT) is None
    assert build_awareness([], [], "loss", SUPPORT) is None


def test_fit_denorm_recovers_popart_and_shifts_the_loss_bar():
    mu, sigma = 3.0, 2.0
    # dists whose NORMALIZED mean is ~0 (below the real-return zero once denormalized:
    # real mean = 0·2+3 = 3 > 0 — but a normalized mean of −2 maps to real −1 < 0)
    dists = [_bump(1.0), _bump(0.5), _bump(0.0), _bump(-2.0)]
    means = [float((d / d.sum() * Z).sum()) for d in dists]
    values = [m * sigma + mu for m in means]           # what value_from_dist records
    est = fit_denorm(means, values)
    assert est == pytest.approx((mu, sigma), abs=1e-6)

    v = build_awareness(dists, [1, 2, 3, 4], "loss", SUPPORT, values=values)
    assert v.denorm == pytest.approx((mu, sigma), abs=1e-5)
    # normalized mean −2 → real −1: most mass sits below real zero (normalized −1.5)
    assert v.p_loss[-1] > 0.5
    # WITHOUT the fit, the same dist reads p_loss at the normalized zero instead
    raw = build_awareness(dists, [1, 2, 3, 4], "loss", SUPPORT)
    assert raw.p_loss[-1] > v.p_loss[-1]               # stricter bar once denormalized

    # degenerate inputs fall back to identity
    assert fit_denorm([0.5, 0.5], [3.0, 3.0]) == (0.0, 1.0)
    assert fit_denorm([0.5], [3.0]) == (0.0, 1.0)
    # a negative-slope (nonsense) fit is refused too
    assert fit_denorm([0.0, 1.0], [1.0, -1.0]) == (0.0, 1.0)


def test_coverage_calibrated_head_reads_calibrated():
    """Dists centered ON the realized return → PIT ≈ 0.5 each, everything inside the 10–90 band."""
    from main.prober.awareness import coverage_stats

    returns = [4.0, 1.0, -3.0, -6.0]
    dists = [_bump(g) for g in returns]
    c = coverage_stats(dists, returns, SUPPORT)
    assert c is not None and c["n"] == 4
    assert abs(c["pit_mean"] - 0.5) < 0.05
    assert c["coverage80"] == 1.0


def test_coverage_optimistic_head_pit_below_half():
    """Dists sitting ABOVE the realized returns → outcomes land in the lower tail: PIT ≪ 0.5 and
    the narrow bands miss — the over-confident/optimistic signature."""
    from main.prober.awareness import coverage_stats

    returns = [-6.0, -7.0, -5.0, -8.0]
    dists = [_bump(6.0, width=0.8) for _ in returns]
    c = coverage_stats(dists, returns, SUPPORT)
    assert c["pit_mean"] < 0.05
    assert c["coverage80"] == 0.0


def test_coverage_denorm_shifts_the_axis():
    """With popart (mu=3, sigma=2) recovered from the paired scalar values, a dist whose
    NORMALIZED mean matches (G-mu)/sigma is calibrated in REAL units; without the fit the same
    inputs read miscalibrated."""
    from main.prober.awareness import coverage_stats

    mu, sigma = 3.0, 2.0
    returns = [5.0, 3.0, 1.0, -1.0]
    dists = [_bump((g - mu) / sigma, width=0.75) for g in returns]
    means = [float((d / d.sum() * Z).sum()) for d in dists]
    values = [m * sigma + mu for m in means]
    c = coverage_stats(dists, returns, SUPPORT, values=values)
    assert abs(c["pit_mean"] - 0.5) < 0.05 and c["coverage80"] == 1.0
    raw = coverage_stats(dists, returns, SUPPORT)               # no denorm: axis is wrong
    assert abs(raw["pit_mean"] - 0.5) > 0.1


def test_coverage_skips_missing_and_requires_two():
    from main.prober.awareness import coverage_stats

    dists = [None, _bump(0.0), _bump(1.0)]
    c = coverage_stats(dists, [0.0, 0.0, 1.0], SUPPORT)
    assert c is not None and c["n"] == 2
    assert coverage_stats([_bump(0.0), None], [0.0, 0.0], SUPPORT) is None


def test_from_npz_reads_the_trace_arrays():
    from main.prober.awareness import awareness_from_npz

    dists = np.stack([_bump(6), _bump(-6), _bump(-7)])
    npz = {"value_dist": dists, "values": np.array([6.0, -6.0, -7.0])}
    v = awareness_from_npz(npz, [1, 2, 3], "loss", SUPPORT)
    assert v is not None and v.knew_by_turn == 2
    assert awareness_from_npz({}, [1, 2, 3], "loss", SUPPORT) is None
    # NaN rows (uncaptured) are skipped, not crashed on
    dists_nan = dists.copy()
    dists_nan[0] = np.nan
    v2 = awareness_from_npz({"value_dist": dists_nan}, [1, 2, 3], "loss", SUPPORT)
    assert v2 is not None and v2.n_decisions == 2
