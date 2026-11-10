"""BOUNDARY-1's 2 x 2, meter 1 (``main.b1.cross``): the registered rule on matrices whose answer is known by hand.

Each test names the revert it catches: dropping the variance floor, pooling the variance across a factor's two arms
(the other factor's effect would enter it), reading a cell from the wrong side, a panel weight, a class threshold,
the AT-BOUNDARY band, breadth's net rule, the confirming meter, the exact-pairs and drawn-games rules of a cell,
and a class read after a STOP-HARM.
"""
from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Dict, Mapping, Optional, Tuple

import numpy as np
import pytest

from main.b1 import cross as X
from main.h2h.cross import CrossInputError, cross_stat, t_ppf

RUNS = {"C": ("c1", "c2"), "W": ("w1", "w2"), "Br": ("b1", "b2"), "WB": ("x1", "x2")}
DESIGN = X.design_of(RUNS)
T4 = t_ppf(0.95, 4)


def model_scores(strength: Mapping[str, float], design: X.Design = DESIGN,
                 noise: Optional[Mapping[Tuple[str, str], float]] = None) -> Dict[Tuple[str, str], float]:
    """h(player, opponent) = 50 + strength[player] - strength[opponent] (+ a per-cell term), the design's cells."""
    return {(p, o): 50.0 + strength[p] - strength[o] + (noise or {}).get((p, o), 0.0) for p, o in design.cells()}


def strengths(width: float = 0.0, breadth: float = 0.0, inter: float = 0.0,
              own: Optional[Mapping[str, float]] = None, design: X.Design = DESIGN) -> Dict[str, float]:
    eff = {"C": 0.0, "W": width, "Br": breadth, "WB": width + breadth + inter}
    return {r: eff[a] + (own or {}).get(r, 0.0) for a in X.ARMS for r in design.runs[a]}


def all_panels(scores: Mapping[Tuple[str, str], float]) -> Dict[str, Dict[Tuple[str, str], float]]:
    return {p: dict(scores) for p in X.PANELS}


def mk(lo: float, hi: float, what: str = "width", view: str = "N", estimate: Optional[float] = None) -> X.Stat:
    est = 0.5 * (lo + hi) if estimate is None else estimate
    return X.Stat(what=what, view=view, estimate=est, groups=(), v_hat=1.0, se=1.0, df=4, level=0.9, lo=lo, hi=hi,
                  floor_pp=X.FLOOR_PP)


# ------------------------------------------------------------------------------------------------- the design
def test_the_plan_is_every_cross_arm_pair_once_with_the_later_arm_as_the_player() -> None:
    cells = DESIGN.cells()
    assert len(cells) == 24 and len(set(cells)) == 24
    order = {a: i for i, a in enumerate(X.ARMS)}
    for p, o in cells:
        assert order[DESIGN.arm_of(p)] > order[DESIGN.arm_of(o)]
    pairs = {frozenset(c) for c in cells}
    want = {frozenset((r, s)) for a in X.ARMS for b in X.ARMS if a != b for r in RUNS[a] for s in RUNS[b]}
    assert pairs == want


def test_each_factor_splits_the_arms_two_against_two_and_holds_one_arm_of_the_other_factor_on_each_side() -> None:
    assert X.FACTORS["width"] == (("W", "WB"), ("C", "Br"))
    assert X.FACTORS["breadth"] == (("Br", "WB"), ("C", "W"))
    for on, off in X.FACTORS.values():
        assert sorted(on + off) == sorted(X.ARMS)


@pytest.mark.parametrize("runs", [
    {"C": ("c1", "c2"), "W": ("w1", "w2"), "Br": ("b1", "b2")},
    {"C": ("c1",), "W": ("w1",), "Br": ("b1",), "WB": ("x1",)},
    {"C": ("c1", "c2"), "W": ("w1", "w2"), "Br": ("b1", "b2"), "WB": ("x1", "x2", "x3")},
    {"C": ("c1", "c2"), "W": ("w1", "w2"), "Br": ("b1", "b2"), "WB": ("x1", "c1")},
])
def test_a_design_that_is_not_four_equal_arms_of_distinct_runs_is_refused(runs: Dict[str, Tuple[str, ...]]) -> None:
    with pytest.raises(CrossInputError):
        X.design_of(runs)


def test_a_cell_read_from_the_other_side_is_one_hundred_minus_the_score() -> None:
    scores = {("w1", "c1"): 61.5}
    assert X.oriented(scores, "w1", "c1") == 61.5
    assert X.oriented(scores, "c1", "w1") == 38.5
    with pytest.raises(CrossInputError, match="missing"):
        X.oriented(scores, "w2", "c1")
    with pytest.raises(CrossInputError, match="both sides"):
        X.oriented({("w1", "c1"): 61.5, ("c1", "w1"): 38.5}, "w1", "c1")


# ---------------------------------------------------------------------------------------------------- blend
def test_n_is_half_home72_and_a_sixth_of_each_unfamiliar_panel() -> None:
    base = model_scores(strengths())
    by = {"home72": {c: 60.0 for c in base}, "heldout72_hpfix": {c: 30.0 for c in base},
          "forum72": {c: 42.0 for c in base}, "p72": {c: 54.0 for c in base}}
    n = X.blend(DESIGN, by, "N")
    assert set(n) == set(DESIGN.cells())
    assert all(v == pytest.approx(0.5 * 60.0 + (30.0 + 42.0 + 54.0) / 6.0, abs=1e-12) for v in n.values())
    assert all(v == pytest.approx(42.0, abs=1e-12) for v in X.blend(DESIGN, by, "U").values())
    assert all(v == 42.0 for v in X.blend(DESIGN, by, "forum72").values())
    assert all(v == 60.0 for v in X.blend(DESIGN, by, "home72").values())
    for view, w in X.VIEWS.items():
        assert math.fsum(w.values()) == pytest.approx(1.0, abs=1e-12), view
    assert X.PAIRS == {"home72": 1000, "heldout72_hpfix": 500, "forum72": 500, "p72": 500}


def test_a_panel_with_a_missing_an_extra_or_an_impossible_cell_is_refused() -> None:
    good = all_panels(model_scores(strengths()))
    first = DESIGN.cells()[0]
    missing = {p: dict(s) for p, s in good.items()}
    del missing["p72"][first]
    with pytest.raises(CrossInputError, match="p72"):
        X.blend(DESIGN, missing, "N")
    X.blend(DESIGN, missing, "home72")  # a view that does not weight p72 does not need it
    extra = {p: dict(s) for p, s in good.items()}
    extra["forum72"][("c1", "c2")] = 50.0
    with pytest.raises(CrossInputError, match="forum72"):
        X.blend(DESIGN, extra, "N")
    flipped = {p: dict(s) for p, s in good.items()}
    flipped["home72"][(first[1], first[0])] = 100.0 - flipped["home72"].pop(first)
    with pytest.raises(CrossInputError, match="home72"):
        X.blend(DESIGN, flipped, "N")
    bad = {p: dict(s) for p, s in good.items()}
    bad["home72"][first] = float("nan")
    with pytest.raises(CrossInputError, match="not a rate"):
        X.blend(DESIGN, bad, "N")
    with pytest.raises(CrossInputError, match="no scores"):
        X.blend(DESIGN, {"home72": good["home72"]}, "N")


# ------------------------------------------------------------------------------------------- the factor statistic
def test_the_factor_statistic_by_hand_when_the_floor_does_not_bind() -> None:
    own = {"c1": -3.0, "c2": 3.0, "w1": -2.0, "w2": 2.0, "b1": -4.0, "b2": 4.0, "x1": -5.0, "x2": 5.0}
    s = X.factor_stat(DESIGN, model_scores(strengths(width=3.0, own=own)), "width")
    assert s.estimate == pytest.approx(3.0, abs=1e-12)
    by_arm = {g.arm: g for g in s.groups}
    # within-arm variance of two runs at -a and +a is 2 a^2
    assert by_arm["W"].s2_raw == pytest.approx(8.0) and by_arm["WB"].s2_raw == pytest.approx(50.0)
    assert by_arm["C"].s2_raw == pytest.approx(18.0) and by_arm["Br"].s2_raw == pytest.approx(32.0)
    assert not any(g.floored for g in s.groups)
    v = ((8.0 + 50.0) / 2.0) / 4.0 + ((18.0 + 32.0) / 2.0) / 4.0
    assert s.v_hat == pytest.approx(v) and s.df == 4 and s.se == pytest.approx(math.sqrt(v))
    assert s.lo == pytest.approx(3.0 - T4 * math.sqrt(v)) and s.hi == pytest.approx(3.0 + T4 * math.sqrt(v))
    assert T4 == pytest.approx(2.1318, abs=1e-4)


def test_the_floor_binds_on_a_lucky_small_spread_and_changes_the_class() -> None:
    scores = model_scores(strengths(width=1.0, own={"w1": 0.01, "c2": -0.01}))
    s = X.factor_stat(DESIGN, scores, "width")
    assert all(g.floored and g.s2 == X.FLOOR_PP ** 2 for g in s.groups)
    assert s.v_hat == pytest.approx(X.FLOOR_PP ** 2 / 2.0)
    half = T4 * X.FLOOR_PP / math.sqrt(2.0)
    assert half == pytest.approx(2.653, abs=1e-3)
    assert X.classify("100M", s).outcome == X.NO_USEFUL_GAIN
    # without the floor the same matrix would be a WIN: the floor is what stands between two lucky seeds and a class
    unfloored = X.factor_stat(DESIGN, scores, "width", floor_pp=1e-6)
    assert unfloored.hi - unfloored.lo < 0.1 and X.classify("100M", unfloored).outcome == X.WIN


def test_each_arm_is_floored_on_its_own_never_the_pooled_variance() -> None:
    # W spreads widely, WB not at all: pooling first and flooring second would leave WB's zero in the mean
    own = {"w1": -4.0, "w2": 4.0}
    s = X.factor_stat(DESIGN, model_scores(strengths(own=own)), "width")
    by_arm = {g.arm: g for g in s.groups}
    assert by_arm["W"].s2 == pytest.approx(32.0) and not by_arm["W"].floored
    assert by_arm["WB"].s2 == X.FLOOR_PP ** 2 and by_arm["WB"].floored
    assert s.v_hat == pytest.approx((32.0 + X.FLOOR_PP ** 2) / 2.0 / 4.0 + X.FLOOR_PP ** 2 / 4.0)


def test_the_other_factors_effect_cancels_in_the_estimate_and_in_the_variance() -> None:
    own = {"c1": -3.0, "c2": 3.0, "w1": -2.0, "w2": 2.0, "b1": -4.0, "b2": 4.0, "x1": -5.0, "x2": 5.0}
    plain = X.factor_stat(DESIGN, model_scores(strengths(width=2.0, own=own)), "width")
    with_b = X.factor_stat(DESIGN, model_scores(strengths(width=2.0, breadth=9.0, own=own)), "width")
    assert with_b.estimate == pytest.approx(plain.estimate, abs=1e-12) == pytest.approx(2.0)
    assert with_b.v_hat == pytest.approx(plain.v_hat, abs=1e-12)
    # the ungrouped cross statistic on the same 4 x 4 matrix does NOT have this property (why the rule groups by arm)
    _r, _c, h0 = X.factor_matrix(DESIGN, model_scores(strengths(width=2.0, own=own)), "width")
    _r, _c, h1 = X.factor_matrix(DESIGN, model_scores(strengths(width=2.0, breadth=9.0, own=own)), "width")
    assert cross_stat(h1).v_hat > cross_stat(h0).v_hat + 5.0
    # and breadth's own statistic reads the breadth effect
    b = X.factor_stat(DESIGN, model_scores(strengths(width=2.0, breadth=9.0, own=own)), "breadth")
    assert b.estimate == pytest.approx(9.0, abs=1e-12)


def test_the_factor_matrix_reads_reversed_cells_from_the_rows_side() -> None:
    scores = model_scores(strengths(width=3.0, breadth=1.0))
    rows, cols, h = X.factor_matrix(DESIGN, scores, "width")
    assert rows == ("w1", "w2", "x1", "x2") and cols == ("c1", "c2", "b1", "b2")
    # W against Br is stored as (Br, W): the row's score is 100 - the stored one = 50 + 3 - 1
    assert ("b1", "w1") in scores and ("w1", "b1") not in scores
    assert h[0][2] == pytest.approx(52.0) and h[0][0] == pytest.approx(53.0)
    assert h[2][0] == pytest.approx(54.0) and h[2][2] == pytest.approx(53.0)


def test_three_runs_an_arm_reads_by_the_same_rule_on_eight_df() -> None:
    runs = {a: tuple(f"{a}{i}" for i in range(3)) for a in X.ARMS}
    d = X.design_of(runs)
    assert len(d.cells()) == 54
    own = {f"{a}{i}": 3.0 * (i - 1) for a in X.ARMS for i in range(3)}
    s = X.factor_stat(d, model_scores(strengths(width=1.5, own=own, design=d), d), "width")
    assert s.df == 8 and s.estimate == pytest.approx(1.5)
    assert s.v_hat == pytest.approx(9.0 / 6.0 + 9.0 / 6.0)  # each within-arm variance is 9


def test_the_rule_matches_an_independent_numpy_implementation_on_random_matrices() -> None:
    rng = np.random.default_rng(20261109)
    cells = DESIGN.cells()
    idx = {r: i for i, r in enumerate(r for a in X.ARMS for r in RUNS[a])}
    for _ in range(200):
        a = rng.normal(0.0, 2.0, 8) + np.repeat([0.0, 1.0, 2.0, 4.0], 2)
        e = rng.normal(0.0, 1.0, len(cells))
        scores = {c: 50.0 + a[idx[c[0]]] - a[idx[c[1]]] + e[k] for k, c in enumerate(cells)}
        full = np.full((8, 8), np.nan)
        for (p, o), v in scores.items():
            full[idx[p], idx[o]] = v
            full[idx[o], idx[p]] = 100.0 - v
        for factor, (on, off) in (("width", ([2, 3, 6, 7], [0, 1, 4, 5])), ("breadth", ([4, 5, 6, 7], [0, 1, 2, 3]))):
            h = full[np.ix_(on, off)]
            rm, cm = h.mean(1), h.mean(0)
            s2r = np.mean([max(np.var(rm[:2], ddof=1), X.FLOOR_PP ** 2), max(np.var(rm[2:], ddof=1), X.FLOOR_PP ** 2)])
            s2c = np.mean([max(np.var(cm[:2], ddof=1), X.FLOOR_PP ** 2), max(np.var(cm[2:], ddof=1), X.FLOOR_PP ** 2)])
            half = T4 * math.sqrt(s2r / 4.0 + s2c / 4.0)
            got = X.factor_stat(DESIGN, scores, factor)
            assert got.estimate == pytest.approx(h.mean() - 50.0, abs=1e-9)
            assert got.hi - got.estimate == pytest.approx(half, abs=1e-9)


# ------------------------------------------------------------------------------------------------ the classes
@pytest.mark.parametrize("lo,hi,want", [
    (0.5, 6.0, X.WIN), (-6.0, -0.5, X.LOSS), (-2.0, 3.5, X.NO_USEFUL_GAIN), (-1.0, 4.5, X.UNDECIDED),
    (0.006, 6.0, X.WIN), (-6.0, -0.006, X.LOSS), (-2.0, 3.994, X.NO_USEFUL_GAIN),
])
def test_the_100m_classes(lo: float, hi: float, want: str) -> None:
    v = X.classify("100M", mk(lo, hi))
    assert v.outcome == want and v.at_boundary == ()


@pytest.mark.parametrize("lo,hi,want", [
    (0.004, 6.0, X.UNDECIDED),  # a lower end AT the WIN threshold is not a WIN
    (-0.004, 6.0, X.UNDECIDED),
    (0.0, 6.0, X.UNDECIDED),
    (-6.0, -0.004, X.NO_USEFUL_GAIN),  # an upper end AT zero is not a LOSS (it is still below the bar)
    (-6.0, 0.004, X.NO_USEFUL_GAIN),
    (-2.0, 3.996, X.UNDECIDED),  # an upper end AT the bar is not NO-USEFUL-GAIN
    (-2.0, 4.004, X.UNDECIDED),
])
def test_a_bound_within_the_band_of_its_threshold_takes_the_conservative_class_and_is_named(
        lo: float, hi: float, want: str) -> None:
    v = X.classify("100M", mk(lo, hi))
    assert v.outcome == want and len(v.at_boundary) == 1


@pytest.mark.parametrize("hi,want,near", [
    (-2.5, X.STOP_HARM, 0), (-2.006, X.STOP_HARM, 0), (-2.004, X.CONTINUE, 1), (-1.996, X.CONTINUE, 1),
    (-1.5, X.CONTINUE, 0), (3.0, X.CONTINUE, 0),
])
def test_the_50m_look_only_stops_for_harm(hi: float, want: str, near: int) -> None:
    v = X.classify("50M", mk(hi - 6.0, hi))
    assert v.outcome == want and len(v.at_boundary) == near
    # a clear gain at 50M is still only CONTINUE: no other 50M outcome exists
    assert X.classify("50M", mk(2.0, 9.0)).outcome == X.CONTINUE


def test_the_75m_look_classifies_nothing() -> None:
    for lo, hi in ((0.5, 6.0), (-9.0, -3.0), (-1.0, 1.0)):
        assert X.classify("75M", mk(lo, hi)).outcome == X.REPORTED
        assert X.classify("EQ", mk(lo, hi)).outcome == X.REPORTED
    with pytest.raises(CrossInputError):
        X.classify("25M", mk(0.0, 1.0))


def test_the_thresholds_are_the_registered_ones() -> None:
    assert (X.FLOOR_PP, X.BAR_PP, X.HARM_PP, X.LEVEL, X.EPS_PP) == (1.76, 4.0, -2.0, 0.90, 0.005)
    assert X.DECIDING == {"width": "N", "breadth": "forum72"}
    assert X.HARM_VIEW == {"width": "N", "breadth": "HF"} and X.VIEWS["HF"] == {"home72": 0.5, "forum72": 0.5}
    assert X.NET == {"breadth": ("home72", "forum72")}
    assert X.CONFIRMING == {"width": "width_U", "breadth": "breadth_F"}


# ---------------------------------------------------------------------------------------- breadth's net rule
def _net(home: float, forum: float, lo: float = 0.5, hi: float = 6.0):
    win = X.classify("100M", mk(lo, hi, what="breadth", view="forum72", estimate=forum))
    return X.net_rule("breadth", win, mk(home - 3.0, home + 3.0, what="breadth", view="home72", estimate=home),
                      mk(lo, hi, what="breadth", view="forum72", estimate=forum))


@pytest.mark.parametrize("home,forum,want,near", [
    (0.0, 3.0, X.WIN, 0), (-2.0, 3.0, X.WIN, 0), (-2.994, 3.0, X.WIN, 0),
    (-2.996, 3.0, X.WIN_NET_NEGATIVE, 1),  # a net within the band of zero does not pass, and is named
    (-3.0, 3.0, X.WIN_NET_NEGATIVE, 1), (-3.004, 3.0, X.WIN_NET_NEGATIVE, 1),
    (-3.006, 3.0, X.WIN_NET_NEGATIVE, 0), (-9.0, 3.0, X.WIN_NET_NEGATIVE, 0),
])
def test_a_breadth_win_is_adoptable_only_if_home72_plus_forum72_is_above_zero(
        home: float, forum: float, want: str, near: int) -> None:
    outcome, info, notes = _net(home, forum)
    assert outcome == want and len(notes) == near
    assert info["net_pp"] == pytest.approx(home + forum) and info["passes"] == (want == X.WIN)
    assert (info["cost_view"], info["gain_view"]) == ("home72", "forum72")


def test_home72_alone_never_vetoes() -> None:
    # home72's own interval is entirely below zero (-7 +/- 3: a LOSS if it were classed), yet the net is +1: a WIN
    outcome, info, _ = _net(-7.0, 8.0)
    assert outcome == X.WIN and info["cost_pp"] == -7.0
    assert X.classify("100M", mk(-10.0, -4.0, what="breadth", view="home72")).outcome == X.LOSS


def test_the_net_rule_touches_only_a_win_of_breadth() -> None:
    # width has no net rule, whatever its home72 statistic says
    assert X.net_rule("width", X.classify("100M", mk(0.5, 6.0)), None, None) == (X.WIN, None, ())
    for lo, hi, cls in ((-6.0, -0.5, X.LOSS), (-2.0, 3.5, X.NO_USEFUL_GAIN), (-1.0, 4.5, X.UNDECIDED)):
        outcome, info, notes = _net(-20.0, 0.5 * (lo + hi), lo, hi)
        assert outcome == cls and info["passes"] is False and notes == ()
    win = X.classify("100M", mk(0.5, 6.0, what="breadth", view="forum72"))
    gain = mk(0.5, 6.0, what="breadth", view="forum72")
    for cost in (None, mk(-1.0, 1.0, what="breadth", view="N"), mk(-1.0, 1.0, what="width", view="home72")):
        with pytest.raises(CrossInputError):  # the rule needs breadth's own home72 and forum72 statistics
            X.net_rule("breadth", win, cost, gain)


# ----------------------------------------------------------------------------------------- the confirming meter
@pytest.mark.parametrize("outcome,kakuna,want,to_owner", [
    (X.WIN, "POSITIVE", X.CONFIRMED, False), (X.WIN, "SPANS-ZERO", X.UNCONFIRMED, False),
    (X.WIN, "NEGATIVE", X.IN_HOUSE_ONLY, True),
    (X.UNDECIDED, "POSITIVE", X.KAKUNA_ONLY, True), (X.NO_USEFUL_GAIN, "POSITIVE", X.KAKUNA_ONLY, True),
    (X.LOSS, "POSITIVE", X.KAKUNA_ONLY, True), (X.WIN_NET_NEGATIVE, "POSITIVE", X.KAKUNA_ONLY, True),
    (X.UNDECIDED, "SPANS-ZERO", X.NO_CONTRADICTION, False), (X.LOSS, "NEGATIVE", X.NO_CONTRADICTION, False),
    (X.NO_USEFUL_GAIN, "NEGATIVE", X.NO_CONTRADICTION, False),
    (X.WIN_NET_NEGATIVE, "NEGATIVE", X.NO_CONTRADICTION, False),
    # an INCONCLUSIVE read confirms and contradicts nothing, whatever the outcome: UNREAD, and nothing is acted on
    (X.WIN, "INCONCLUSIVE", X.UNREAD, True), (X.UNDECIDED, "INCONCLUSIVE", X.UNREAD, True),
    (X.LOSS, "INCONCLUSIVE", X.UNREAD, True), (X.NO_USEFUL_GAIN, "INCONCLUSIVE", X.UNREAD, True),
    (X.WIN_NET_NEGATIVE, "INCONCLUSIVE", X.UNREAD, True),
])
def test_kakuna_confirms_or_contradicts_and_never_decides(outcome: str, kakuna: str, want: str, to_owner: bool) -> None:
    assert X.confirm(outcome, kakuna) == (want, to_owner)


def test_the_confirming_meter_refuses_a_class_it_does_not_know() -> None:
    for outcome, kakuna in ((X.CONTINUE, "POSITIVE"), (X.STOP_HARM, "NEGATIVE"), (X.WIN, "GAIN"), ("win", "POSITIVE")):
        with pytest.raises(CrossInputError):
            X.confirm(outcome, kakuna)


# ---------------------------------------------------------------------------------------------- the interaction
def test_the_interaction_by_hand() -> None:
    own = {"c1": -3.0, "c2": 3.0, "w1": -2.0, "w2": 2.0, "b1": -4.0, "b2": 4.0, "x1": -5.0, "x2": 5.0}
    scores = model_scores(strengths(width=2.0, breadth=1.0, inter=3.0, own=own))
    s = X.interaction_stat(DESIGN, scores)
    assert s.estimate == pytest.approx(3.0, abs=1e-12) and s.df == 4
    assert {g.arm: round(g.s2_raw, 9) for g in s.groups} == {"C": 18.0, "W": 8.0, "Br": 32.0, "WB": 50.0}
    assert s.v_hat == pytest.approx((18.0 + 8.0 + 32.0 + 50.0) / 2.0)
    # the main effects absorb half the interaction each; the interaction is not double counted in their variance
    assert X.factor_stat(DESIGN, scores, "width").estimate == pytest.approx(3.5, abs=1e-12)
    assert X.factor_stat(DESIGN, scores, "breadth").estimate == pytest.approx(2.5, abs=1e-12)
    no_inter = X.factor_stat(DESIGN, model_scores(strengths(width=2.0, breadth=1.0, own=own)), "width")
    assert X.factor_stat(DESIGN, scores, "width").v_hat == pytest.approx(no_inter.v_hat, abs=1e-12)


def test_the_interactions_variance_is_floored_per_arm_and_its_class_is_reported() -> None:
    s = X.interaction_stat(DESIGN, model_scores(strengths(inter=4.0)))
    assert all(g.floored for g in s.groups) and s.v_hat == pytest.approx(2.0 * X.FLOOR_PP ** 2)
    assert s.se == pytest.approx(2.489, abs=1e-3) and s.hi - s.estimate == pytest.approx(5.306, abs=1e-3)
    assert X.interaction_class(s) == (X.NOT_SHOWN, ())
    assert X.interaction_class(mk(0.5, 9.0, what="interaction"))[0] == X.SYNERGY
    assert X.interaction_class(mk(-9.0, -0.5, what="interaction"))[0] == X.ANTAGONISM
    cls, near = X.interaction_class(mk(0.004, 9.0, what="interaction"))
    assert cls == X.NOT_SHOWN and len(near) == 1


def test_a_simple_effect_by_hand() -> None:
    own = {"c1": -3.0, "c2": 3.0, "x1": -5.0, "x2": 5.0}
    scores = model_scores(strengths(width=2.0, breadth=1.0, inter=3.0, own=own))
    s = X.simple_effect(DESIGN, scores, "WB", "C")
    assert s.estimate == pytest.approx(6.0) and s.df == 2
    assert s.v_hat == pytest.approx(50.0 / 2.0 + 18.0 / 2.0)
    rev = X.simple_effect(DESIGN, scores, "C", "WB")
    assert rev.estimate == pytest.approx(-6.0) and rev.v_hat == pytest.approx(s.v_hat)
    with pytest.raises(CrossInputError):
        X.simple_effect(DESIGN, scores, "C", "C")


# ----------------------------------------------------------------------------------------------------- a look
OWN = {"c1": -1.0, "c2": 1.0, "w1": -1.5, "w2": 1.5, "b1": -1.0, "b2": 1.0, "x1": -2.0, "x2": 2.0}


def _panels(width: Mapping[str, float], breadth: Mapping[str, float]) -> Dict[str, Dict[Tuple[str, str], float]]:
    return {p: model_scores(strengths(width=width[p], breadth=breadth[p], own=OWN)) for p in X.PANELS}


def test_a_look_end_to_end_reads_each_factor_on_its_own_view_and_reports_every_view() -> None:
    by = _panels({"home72": 8.0, "heldout72_hpfix": 6.0, "forum72": 6.0, "p72": 6.0},
                 {"home72": -2.0, "heldout72_hpfix": 22.0, "forum72": -1.0, "p72": 21.0})
    out = X.read_look(DESIGN, by, "100M")
    w, b = out["factors"]["width"], out["factors"]["breadth"]
    assert w["views"]["N"]["estimate_pp"] == pytest.approx(0.5 * 8.0 + 0.5 * 6.0)
    assert b["views"]["N"]["estimate_pp"] == pytest.approx(0.5 * -2.0 + (22.0 - 1.0 + 21.0) / 6.0)
    assert set(w["views"]) == set(X.VIEWS) == set(b["views"])
    assert w["outcome"] == X.WIN and w["read_on"] == "N" and w["net"] is None
    assert (w["confirming_contrast"], b["confirming_contrast"]) == ("width_U", "breadth_F")
    # breadth is far above zero on N, but only through the two panels that overlap its training teams (hazard M4):
    # its class is read on forum72 alone, where it shows nothing
    assert b["views"]["N"]["interval_pp"][0] > 0.0
    assert b["read_on"] == "forum72" and b["outcome"] == X.NO_USEFUL_GAIN
    assert b["h_pp"] == [list(r) for r in X.factor_matrix(DESIGN, by["forum72"], "breadth")[2]]
    assert b["net"]["net_pp"] == pytest.approx(-3.0) and b["net"]["passes"] is False
    assert out["interaction"]["class_is"] == X.REPORTED and out["interaction"]["class"] == X.NOT_SHOWN
    assert set(out["simple_effects"]) == {f"{a}_vs_{c}" for a, c in X.SIMPLE_EFFECTS}
    assert X.read_look(DESIGN, by, "75M")["factors"]["breadth"]["outcome"] == X.REPORTED


@pytest.mark.parametrize("home,forum,want", [(-7.0, 8.0, X.WIN), (0.0, 8.0, X.WIN), (-9.0, 8.0, X.WIN_NET_NEGATIVE),
                                             (-8.0, 8.0, X.WIN_NET_NEGATIVE)])
def test_a_looks_breadth_outcome_is_forum72s_class_after_the_net_rule(home: float, forum: float, want: str) -> None:
    flat = {p: 0.0 for p in X.PANELS}
    out = X.read_look(DESIGN, _panels(flat, {**flat, "home72": home, "forum72": forum}), "100M")
    b = out["factors"]["breadth"]
    assert b["class_before_net_rule"] == X.WIN and b["outcome"] == want
    assert b["net"]["net_pp"] == pytest.approx(home + forum, abs=1e-9)
    assert bool(b["at_boundary"]) == (home + forum == 0.0)
    # home72 alone: at -7 its own interval is entirely below zero, and the lever still wins on a net of +1
    assert (b["views"]["home72"]["interval_pp"][1] < 0.0) == (home <= -7.0)
    assert out["factors"]["width"]["outcome"] == X.NO_USEFUL_GAIN  # width is flat: breadth's panels do not leak in


def test_the_50m_look_reads_width_on_n_and_breadth_on_the_net_view() -> None:
    flat = {p: 0.0 for p in X.PANELS}
    look = lambda w, b: {f: v["outcome"] for f, v in X.read_look(DESIGN, _panels(w, b), "50M")["factors"].items()}
    assert look(flat, flat) == {"width": X.CONTINUE, "breadth": X.CONTINUE}
    assert look({p: -9.0 for p in X.PANELS}, flat) == {"width": X.STOP_HARM, "breadth": X.CONTINUE}
    # breadth: a home cost the clean panel does not pay back stops it (HF = -7); the same cost paid back does not
    assert look(flat, {**flat, "home72": -14.0}) == {"width": X.CONTINUE, "breadth": X.STOP_HARM}
    assert look(flat, {**flat, "home72": -14.0, "forum72": 14.0}) == {"width": X.CONTINUE, "breadth": X.CONTINUE}
    # the panels that overlap breadth's training teams cannot rescue it, nor sink it
    assert look(flat, {**flat, "home72": -14.0, "heldout72_hpfix": 40.0, "p72": 40.0})["breadth"] == X.STOP_HARM
    assert look(flat, {**flat, "heldout72_hpfix": -40.0, "p72": -40.0})["breadth"] == X.CONTINUE
    out = X.read_look(DESIGN, _panels(flat, {**flat, "home72": -14.0}), "50M")["factors"]["breadth"]
    assert out["read_on"] == "HF" and out["net"]["net_pp"] == pytest.approx(-14.0) and out["net"]["passes"] is None


# ------------------------------------------------------------------------------------- scores from the ledger
def _cell(player: str, opponent: str, score: Optional[float] = 0.55, n_pairs: int = 1000, verdict: str = "OK",
          reasons: Tuple[str, ...] = (), d: int = 0) -> SimpleNamespace:
    return SimpleNamespace(player=player, opponent=opponent, score=score, n_pairs=n_pairs, verdict=verdict,
                           reasons=reasons, d=d, games=2 * n_pairs)


def _ledger_cells() -> list:
    return [_cell("sha_" + p, "sha_" + o) for p, o in DESIGN.cells()]


RUN_OF = {"sha_" + r: r for a in X.ARMS for r in RUNS[a]}


def test_ledger_cells_become_scores_keyed_by_run() -> None:
    scores, reasons = X.scores_from_cells(_ledger_cells(), RUN_OF, DESIGN, 1000)
    assert reasons == () and scores is not None
    assert set(scores) == set(DESIGN.cells()) and all(v == pytest.approx(55.0) for v in scores.values())


def test_any_defect_in_a_panels_cells_is_inconclusive_never_a_score() -> None:
    def reasons_of(cells: list, pairs: int = 1000) -> str:
        scores, reasons = X.scores_from_cells(cells, RUN_OF, DESIGN, pairs)
        assert scores is None and reasons
        return " | ".join(reasons)

    assert "missing cell" in reasons_of(_ledger_cells()[1:])
    assert "appears twice" in reasons_of(_ledger_cells() + _ledger_cells()[:1])
    short = _ledger_cells()
    short[3] = _cell(short[3].player, short[3].opponent, n_pairs=999)
    assert "999 completed pairs < the registered 1000" in reasons_of(short)
    # MORE than the registered pairs is a defect too (games added after the plan was read): never "at least"
    over = _ledger_cells()
    over[3] = _cell(over[3].player, over[3].opponent, n_pairs=1500)
    assert "1500 completed pairs > the registered 1000" in reasons_of(over)
    assert "> the registered 500" in reasons_of(_ledger_cells(), pairs=500)
    # drawn games are the head-to-head's timeouts: more than 25 % of a cell is INCONCLUSIVE, exactly 25 % is not
    drawn = _ledger_cells()
    drawn[4] = _cell(drawn[4].player, drawn[4].opponent, d=501)
    assert "501 of 2000 games drawn" in reasons_of(drawn)
    drawn[4] = _cell(drawn[4].player, drawn[4].opponent, d=500)
    assert X.scores_from_cells(drawn, RUN_OF, DESIGN, 1000)[0] is not None
    aborted = _ledger_cells()
    aborted[5] = _cell(aborted[5].player, aborted[5].opponent, verdict="INCONCLUSIVE",
                       reasons=("aborted 600 of 2000 attempted games > 25 %",))
    assert "> 25 %" in reasons_of(aborted)
    wrong_side = _ledger_cells()
    wrong_side[0] = _cell(wrong_side[0].opponent, wrong_side[0].player)
    assert "outside the design" in reasons_of(wrong_side)
    stranger = _ledger_cells() + [_cell("sha_other", "sha_c1")]
    assert "outside the look" in reasons_of(stranger)
    same_arm = _ledger_cells() + [_cell("sha_c1", "sha_c2")]
    assert "outside the design" in reasons_of(same_arm)
    no_score = _ledger_cells()
    no_score[7] = _cell(no_score[7].player, no_score[7].opponent, score=None)
    assert "no score" in reasons_of(no_score)


# ------------------------------------------------------------------------------------------ after a STOP-HARM
PAIR_RUNS = {"C": ("c1", "c2"), "Br": ("b1", "b2")}


def test_a_survivor_design_is_the_surviving_factors_on_arm_against_the_control() -> None:
    pair = X.pair_design_of(PAIR_RUNS)
    assert pair.arms == ("Br", "C") and pair.survivor == "breadth" and pair.n == 2
    assert pair.cells() == [("b1", "c1"), ("b1", "c2"), ("b2", "c1"), ("b2", "c2")]
    assert X.pair_design_of({"C": ("c1", "c2"), "W": ("w1", "w2")}).survivor == "width"
    for bad in ({"W": ("w1", "w2"), "Br": ("b1", "b2")}, {"C": ("c1", "c2"), "WB": ("x1", "x2")}, RUNS,
                {"C": ("c1",), "Br": ("b1",)}, {"C": ("c1", "c2"), "Br": ("b1", "c1")}):
        with pytest.raises(CrossInputError):
            X.pair_design_of(bad)


def test_after_a_stop_harm_the_survivor_is_reported_on_every_view_and_never_classed() -> None:
    # a survivor +20 pp ahead on every panel (an interval far above zero AND above the bar) still reads NO class
    pair = X.pair_design_of(PAIR_RUNS)
    scores = {c: 70.0 for c in pair.cells()}
    out = X.read_survivor(pair, {p: dict(scores) for p in X.PANELS}, "100M")
    assert out["outcome"] == X.NO_CLASS and out["class_is"] == X.REPORTED and out["to_owner"] is True
    assert (out["survivor"], out["stopped"], out["effect"]) == ("breadth", "width", "Br_vs_C")
    assert set(out["views"]) == set(X.VIEWS)
    half = t_ppf(0.95, 2) * X.FLOOR_PP  # V = floor^2 / 2 + floor^2 / 2, 2 df
    for view in X.VIEWS:
        v = out["views"][view]
        assert v["estimate_pp"] == pytest.approx(20.0) and v["df"] == 2
        assert v["interval_pp"][1] - v["estimate_pp"] == pytest.approx(half) and v["interval_pp"][0] > X.BAR_PP
    assert "factors" not in out and "interaction" not in out
    # breadth's net is printed and never judged; width has none
    assert out["net"]["net_pp"] == pytest.approx(40.0) and out["net"]["passes"] is None
    wide = X.pair_design_of({"C": ("c1", "c2"), "W": ("w1", "w2")})
    w = X.read_survivor(wide, {p: {c: 40.0 for c in wide.cells()} for p in X.PANELS}, "75M")
    assert w["outcome"] == X.NO_CLASS and w["net"] is None and w["effect"] == "W_vs_C"
    assert w["views"]["N"]["interval_pp"][1] < 0  # an interval wholly below zero is no LOSS either


def test_a_survivor_look_is_never_the_50m_look_and_needs_every_panel() -> None:
    pair = X.pair_design_of(PAIR_RUNS)
    full = {p: {c: 50.0 for c in pair.cells()} for p in X.PANELS}
    with pytest.raises(CrossInputError):
        X.read_survivor(pair, full, "50M")
    with pytest.raises(CrossInputError):
        X.read_survivor(pair, {p: full[p] for p in X.PANELS[:-1]}, "100M")


def test_ledger_cells_of_a_survivor_look_are_its_four_cells_alone() -> None:
    pair = X.pair_design_of(PAIR_RUNS)
    run_of = {"sha_" + r: r for rs in PAIR_RUNS.values() for r in rs}
    cells = [_cell("sha_" + p, "sha_" + o) for p, o in pair.cells()]
    scores, reasons = X.scores_from_cells(cells, run_of, pair, 1000)
    assert reasons == () and scores is not None and set(scores) == set(pair.cells())
    scores, reasons = X.scores_from_cells(cells[1:], run_of, pair, 1000)
    assert scores is None and "missing cell" in " | ".join(reasons)
