"""The X5 A/B's registered cross statistic and decision rule (``main.h2h.cross``) on SYNTHETIC matrices: known Δ,
a hand computation in exact fractions, every outcome row of §7.4's table, rule 8 at every boundary, the oracle
contrast's column cancellation, the floor and H, and the ledger-cell adapter's INCONCLUSIVE reasons. Deterministic:
no random draw anywhere."""
from __future__ import annotations

import dataclasses
import math
from fractions import Fraction

import pytest

from agents.training.eval_ledger.cells import Cell
from main.h2h import cross as X

#: Student t quantiles on 4 df (tables; scipy is checked against them below).
T975_4 = 2.7764451051977987
T95_4 = 2.1318467863266495


def additive(delta, a, b):
    """h_ij = 50 + Δ + a_i − b_j, no noise."""
    return [[50.0 + delta + ai - bj for bj in b] for ai in a]


def stat_with(**kw):
    """A CrossStat at look-1 shape with the fields a decision reads overridden."""
    base = X.cross_stat(additive(0.0, [0.5, -0.5, 0.0], [0.25, -0.25, 0.0]))
    return dataclasses.replace(base, **kw)


# ------------------------------------------------------------------------------------------------ the statistic
def test_scipy_t_quantiles_match_the_tables():
    assert X.t_ppf(0.975, 4) == pytest.approx(T975_4, abs=1e-12)
    assert X.t_ppf(0.95, 4) == pytest.approx(T95_4, abs=1e-12)


@pytest.mark.parametrize("delta", [-6.0, -3.5, 0.0, 2.25, 7.0])
def test_a_noise_free_additive_matrix_returns_its_delta_and_its_seed_effects(delta):
    a, b = [1.0, -2.0, 1.0], [0.5, 0.0, -0.5]
    s = X.cross_stat(additive(delta, a, b))
    assert s.delta_hat == pytest.approx(delta, abs=1e-12)
    # row i = 50 + Δ + a_i − mean(b); column j = 50 + Δ + mean(a) − b_j
    assert s.row_means == pytest.approx(tuple(50 + delta + x for x in a), abs=1e-12)
    assert s.col_means == pytest.approx(tuple(50 + delta - x for x in b), abs=1e-12)
    assert s.s2_rows == pytest.approx(3.0, abs=1e-12)      # var(1, -2, 1)
    assert s.s2_cols == pytest.approx(0.25, abs=1e-12)     # var(0.5, 0, -0.5)
    assert s.v_hat == pytest.approx(3.25 / 3, abs=1e-12)
    assert s.df == 4
    assert s.t == pytest.approx((delta + 3.5) / math.sqrt(3.25 / 3), abs=1e-12)


def test_the_statistic_equals_a_hand_computation_in_exact_fractions():
    h = [[52, 49, 51], [48, 47, 50], [53, 50, 52]]
    n = 3
    F = [[Fraction(x) for x in r] for r in h]
    grand = sum(sum(r) for r in F) / 9
    rows = [sum(r) / 3 for r in F]
    cols = [sum(F[i][j] for i in range(3)) / 3 for j in range(3)]
    var = (lambda xs: sum((x - sum(xs) / 3) ** 2 for x in xs) / 2)
    v = (var(rows) + var(cols)) / n
    s = X.cross_stat(h)
    assert s.delta_hat == pytest.approx(float(grand - 50), abs=1e-12)
    assert s.v_hat == pytest.approx(float(v), abs=1e-12)
    assert s.t == pytest.approx((float(grand - 50) + 3.5) / math.sqrt(float(v)), abs=1e-12)
    se = math.sqrt(float(v))
    assert s.ci95 == pytest.approx((float(grand - 50) - T975_4 * se, float(grand - 50) + T975_4 * se), abs=1e-9)
    assert s.upper95_one_sided == pytest.approx(float(grand - 50) + T95_4 * se, abs=1e-9)
    # the worked numbers, so a regression in the formula is visible in the failure
    assert s.delta_hat == pytest.approx(0.2222222222, abs=1e-9)
    assert s.t == pytest.approx(2.961010, abs=1e-5)


def test_the_statistic_uses_every_cell_not_just_the_diagonal():
    h = additive(0.0, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0])
    h[0][1] += 9.0                                          # an off-diagonal cell moves Δ̂ by 1
    h[1][2] -= 0.5
    h[2][0] += 0.5
    assert X.cross_stat(h).delta_hat == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize("h, match", [
    ([[50.0]], "n ≥ 2"),
    ([[50.0, 51.0], [49.0]], "n × n"),
    ([[50.0, 51.0, 52.0], [49.0, 50.0, 51.0]], "n × n"),
    ([[50.0, float("nan")], [49.0, 50.0]], "non-finite"),
    ([[50.0, 50.0], [50.0, 50.0]], "zero variance"),
])
def test_a_matrix_the_statistic_is_not_defined_on_is_refused(h, match):
    with pytest.raises(X.CrossInputError, match=match):
        X.cross_stat(h)


# ---------------------------------------------------------------------------------------- every outcome row
def test_non_inferior_at_look_1_when_t_clears_5_761():
    d = X.decide(1, X.cross_stat(additive(2.0, [0.1, -0.1, 0.0], [0.1, 0.0, -0.1])))
    assert d.stat.t > 5.761 and d.outcome == X.NON_INFERIOR and d.label is None
    assert X.verdict_text(d) == "NON-INFERIOR"


def test_continue_at_look_1_when_neither_boundary_is_crossed():
    d = X.decide(1, X.cross_stat(additive(0.0, [2.0, -2.0, 0.0], [1.0, -1.0, 0.0])))
    assert -3.5 < d.stat.delta_hat and d.stat.t < 5.761
    assert d.outcome == X.CONTINUE and d.label is None


def test_futility_stop_labelled_inferior_when_the_upper_bound_is_below_minus_delta():
    d = X.decide(1, X.cross_stat(additive(-6.0, [0.2, -0.2, 0.0], [0.1, -0.1, 0.0])))
    assert d.stat.upper95_one_sided < -3.5
    assert d.outcome == X.FUTILITY_STOP and d.label == X.INFERIOR
    assert X.verdict_text(d) == "FUTILITY STOP (verdict NOT DETECTED, INFERIOR)"


def test_futility_stop_without_the_inferior_label_when_the_upper_bound_reaches_minus_delta():
    d = X.decide(2, stat_with(n=5, delta_hat=-4.0, t=-0.5, upper95_one_sided=-1.0))
    assert d.outcome == X.FUTILITY_STOP and d.label is None
    assert X.verdict_text(d) == "FUTILITY STOP (verdict NOT DETECTED)"


@pytest.mark.parametrize("upper, label", [(-4.0, X.INFERIOR), (1.0, None)])
def test_look_3_reads_not_detected_with_the_inferior_label_by_the_upper_bound(upper, label):
    d = X.decide(3, stat_with(n=8, delta_hat=-5.0, t=1.0, upper95_one_sided=upper))
    assert d.outcome == X.NOT_DETECTED and d.label == label        # no futility stop at the last look


def test_look_3_non_inferior_at_its_own_boundary():
    assert X.decide(3, stat_with(n=8, t=1.9, delta_hat=1.0, upper95_one_sided=3.0)).outcome == X.NON_INFERIOR
    assert X.decide(2, stat_with(n=5, t=1.9, delta_hat=1.0, upper95_one_sided=3.0)).outcome == X.CONTINUE


@pytest.mark.parametrize("reasons", [["missing cell a vs b"], ["cell a vs b: aborted 300 of 1000 attempted > 25 %"],
                                     ["a run did not reach 15M"]])
def test_any_invalid_input_is_inconclusive_whatever_the_statistic_says(reasons):
    strong = X.cross_stat(additive(5.0, [0.1, -0.1, 0.0], [0.1, 0.0, -0.1]))
    d = X.decide(1, strong, reasons)
    assert d.outcome == X.INCONCLUSIVE and d.reasons == tuple(reasons)


def test_a_seed_count_other_than_the_looks_registration_is_inconclusive():
    d = X.decide(2, X.cross_stat(additive(5.0, [0.1, -0.1, 0.0], [0.1, 0.0, -0.1])))
    assert d.outcome == X.INCONCLUSIVE and "look 2 registers 5" in d.reasons[0]


def test_an_unregistered_look_is_refused():
    with pytest.raises(ValueError, match="registered looks"):
        X.decide(4, stat_with())


# ------------------------------------------------------------------------------------------------- rule 8
@pytest.mark.parametrize("dt, outcome, near", [
    (0.0, X.CONTINUE, True), (+9e-10, X.CONTINUE, True), (-9e-10, X.CONTINUE, True), (+5e-10, X.CONTINUE, True),
    (+2e-9, X.NON_INFERIOR, False), (-2e-9, X.CONTINUE, False)])
def test_rule_8_a_t_within_1e_9_of_the_boundary_is_not_a_crossing(dt, outcome, near):
    d = X.decide(1, stat_with(t=5.761 + dt, delta_hat=0.0, upper95_one_sided=2.0))
    assert d.outcome == outcome
    assert bool(d.near_boundary) == near


@pytest.mark.parametrize("dd, outcome", [(0.0, X.CONTINUE), (-5e-10, X.CONTINUE), (-2e-9, X.FUTILITY_STOP),
                                         (+2e-9, X.CONTINUE)])
def test_rule_8_on_the_futility_boundary(dd, outcome):
    d = X.decide(1, stat_with(t=0.0, delta_hat=-3.5 + dd, upper95_one_sided=1.0))
    assert d.outcome == outcome


@pytest.mark.parametrize("du, label", [(0.0, None), (-5e-10, None), (-2e-9, X.INFERIOR)])
def test_rule_8_on_the_inferior_label(du, label):
    d = X.decide(1, stat_with(t=-1.0, delta_hat=-5.0, upper95_one_sided=-3.5 + du))
    assert d.outcome == X.FUTILITY_STOP and d.label == label


# ------------------------------------------------------------------------------------------- the oracle reads
def test_the_shared_column_contrast_cancels_the_reference_seed_effects():
    a1, a2 = [1.0, -1.0, 0.5], [0.0, 0.4, -0.4]
    for b in ([0.0, 0.0, 0.0], [3.0, -1.0, -2.0], [10.0, 0.0, -10.0]):
        full = X.cross_stat(additive(8.0, a1, b))
        sp = X.cross_stat(additive(4.0, a2, b))
        c = X.shared_column_contrast(full, sp)
        assert c.estimate == pytest.approx(4.0 + (sum(a1) - sum(a2)) / 3, abs=1e-12)
        assert c.v == pytest.approx(full.s2_rows / 3 + sp.s2_rows / 3, abs=1e-12)
        assert c.df == 4
        assert c.ci95 == pytest.approx((c.estimate - T975_4 * c.se, c.estimate + T975_4 * c.se), abs=1e-9)


def test_the_floor_is_the_larger_of_p0_and_the_blob_columns_spread():
    small = X.cross_stat(additive(0.0, [1.0, -1.0, 0.0], [1.0, 0.0, -1.0]))      # spread 2
    big = X.cross_stat(additive(0.0, [1.0, -1.0, 0.0], [4.0, 0.0, -2.0]))       # spread 6
    assert X.replicate_floor(small)["floor_pp"] == X.P0_H2H_FLOOR_PP == 4.57
    f = X.replicate_floor(big)
    assert f["floor_pp"] == pytest.approx(6.0) and f["in_experiment_blob_spread_pp"] == pytest.approx(6.0)


def test_h_is_reported_only_when_cs_lower_bound_clears_the_floor():
    delta = stat_with(delta_hat=-1.0)
    clear = dataclasses.replace(stat_with(), delta_hat=10.0, ci95=(6.0, 14.0))
    assert X.headroom(delta, clear, 4.57)["H"] == pytest.approx(-0.1)
    within = dataclasses.replace(clear, ci95=(2.0, 18.0))
    out = X.headroom(delta, within, 4.57)
    assert out["H"] is None and "within the noise" in out["read"]
    touching = dataclasses.replace(clear, ci95=(4.57 + 5e-10, 18.0))
    assert X.headroom(delta, touching, 4.57)["H"] is None and X.headroom(delta, touching, 4.57)["near_floor"]


@pytest.mark.parametrize("point, want", [(4.0, "IN"), (0.0, "IN"), (10.0, "IN"), (-0.01, "OUT"), (10.01, "OUT")])
def test_a_prediction_is_graded_in_iff_its_point_lies_in_the_closed_interval(point, want):
    assert X.grade(point, 0.0, 10.0) == want


# ------------------------------------------------------------------------------------- matrix from ledger cells
def cell(p, o, score_counts, verdict="OK", reasons=(), aborted=0):
    w = 2 * score_counts[4] + score_counts[3]
    return Cell(request_id="look1", family="f", opened="t", player=p, opponent=o, player_id=p, opponent_id=o, rows=2,
                w=w, l=0, d=0, aborted=aborted, pairs=tuple(score_counts), verdict=verdict, reasons=tuple(reasons))


def full_cells(n_pairs=1000):
    # pentanomial with mean score 0.5 + small offsets per cell
    return [cell(p, o, (0, 0, n_pairs - k, k, 0)) for k, (p, o) in
            enumerate((p, o) for p in ("x1", "x2", "x3") for o in ("b1", "b2", "b3"))]


def test_the_adapter_builds_the_matrix_in_declared_order_from_pair_scores():
    m = X.matrix_from_cells(full_cells(), ["x1", "x2", "x3"], ["b1", "b2", "b3"])
    assert m.inconclusive == ()
    # cell k has k pairs at 3/4 instead of 1/2: score = 50 + 25 k / 1000 pp
    assert m.h[0] == pytest.approx((50.0, 50.025, 50.05)) and m.h[2][2] == pytest.approx(50.2)


@pytest.mark.parametrize("mutate, match", [
    (lambda cs: cs[:-1], "missing cell"),
    (lambda cs: cs[:1] + [cell("x1", "b2", (0, 0, 999, 0, 0))] + cs[2:], "999 completed pairs < 1000"),
    (lambda cs: [dataclasses.replace(cs[0], verdict="INCONCLUSIVE", reasons=("aborted 300 of 1000 attempted games "
                                                                            "> 25 %",))] + cs[1:], "> 25 %"),
    (lambda cs: cs + [cell("x9", "b1", (0, 0, 1000, 0, 0))], "outside the declared"),
    (lambda cs: cs + [cs[0]], "twice"),
])
def test_the_adapter_turns_every_invalid_cell_into_an_inconclusive_reason(mutate, match):
    m = X.matrix_from_cells(mutate(full_cells()), ["x1", "x2", "x3"], ["b1", "b2", "b3"])
    assert any(match in r for r in m.inconclusive), m.inconclusive
    d = X.summarize(1, m)
    assert d.outcome == X.INCONCLUSIVE and d.stat is None


def test_a_precondition_failure_makes_the_read_inconclusive():
    m = X.matrix_from_cells(full_cells(), ["x1", "x2", "x3"], ["b1", "b2", "b3"])
    assert X.summarize(1, m, ["blob-path identity broken"]).outcome == X.INCONCLUSIVE
    assert X.summarize(1, m).outcome in X.OUTCOMES and X.summarize(1, m).stat is not None
