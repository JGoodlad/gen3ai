"""The ``main.reads`` rule ``b1_fresh_cell`` (``main.b1.fresh_rule``): the statistic, its classes and boundaries, the
INCONCLUSIVE rules, and what the result does and does not depend on."""
import json
import math
import random
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import pytest

from main.b1 import cross as X
from main.b1 import fresh_rule as FR

PIN = "a" * 40
FRESH = ("rb_q9n_s5901", "rb_q9n_s5902", "rb_q9n_s5903")
BANKED = ("rb_b1c_s5101", "rb_b1c_s5102", "rb_b1c_s5110")
#: Student's t at 0.95 on 4 df (tables: 2.1318), the 90 % two-sided quantile at three runs a side.
T95_4 = 2.131846786326649
#: The hand-worked case: fresh run i's scores (pp) against banked runs 1..3.
HAND = ((56.0, 54.0, 58.0), (52.0, 50.0, 54.0), (60.0, 58.0, 62.0))


def _sha(run: str) -> str:
    return (run.encode().hex() * 8)[:64]


def look_doc(look: str = "100M", fresh: Sequence[str] = FRESH, banked: Sequence[str] = BANKED) -> Dict[str, Any]:
    def side(runs: Sequence[str]) -> List[Dict[str, Any]]:
        return [{"run": r, "sha256": _sha(r), "steps": 100_000_044, "checkpoint": f"/m/{r}.zip"} for r in runs]

    return {"look": look, "card": "q9n", "pin": PIN, "runs": {"fresh": side(fresh), "banked": side(banked)}}


def grid(rows: Sequence[Sequence[float]], fresh: Sequence[str] = FRESH, banked: Sequence[str] = BANKED
         ) -> Dict[Tuple[str, str], float]:
    return {(f, b): float(rows[i][j]) for i, f in enumerate(fresh) for j, b in enumerate(banked)}


def ledger(scores: Mapping[Tuple[str, str], float], panel: str, *, draws: int = 0, n_pairs: int = 0
           ) -> List[SimpleNamespace]:
    n = n_pairs or X.PAIRS[panel]
    return [SimpleNamespace(player=_sha(p), opponent=_sha(o), score=v / 100.0, n_pairs=n, verdict="OK", reasons=(),
                            d=draws, games=2 * n, w=0, l=0) for (p, o), v in scores.items()]


def same_on_every_panel(scores: Mapping[Tuple[str, str], float]) -> Dict[str, List[SimpleNamespace]]:
    return {p: ledger(scores, p) for p in X.PANELS}


def decide(rows: Sequence[Sequence[float]], look: str = "100M") -> Dict[str, Any]:
    return FR.read(look_doc(look), same_on_every_panel(grid(rows)))[1]


def uniform(v: float) -> List[List[float]]:
    return [[v] * 3 for _ in range(3)]


#: Nine equal cells: both within-side variances are 0, so both are FLOORED and the half-width is the floor's.
FLOOR_HALF = T95_4 * math.sqrt(2 * X.FLOOR_PP ** 2 / 3)


# ------------------------------------------------------------------------------------------- the statistic
def test_a_hand_worked_3x3_reads_its_estimate_variance_df_interval_and_class():
    """Worked by hand: the mean of the nine cells is 56, so D = +6. The fresh runs' means are 56, 52 and 60
    (variance (0 + 16 + 16) / 2 = 16); the banked runs' columns average 56, 54 and 58 (variance 4); neither is below
    the floor 1.76^2 = 3.0976. V = 16/3 + 4/3 = 20/3, se = 2.58199, 4 df, half = 2.13185 x 2.58199 = 5.50440:
    [+0.49560, +11.50440], so WIN at 100M (lower end above 0) and CONTINUE at 50M."""
    v = decide(HAND)
    n = v["views"]["N"]
    assert n["estimate_pp"] == pytest.approx(6.0, abs=1e-12) and n["df"] == 4
    assert n["v_hat"] == pytest.approx(20.0 / 3.0, abs=1e-12)
    assert [g["s2"] for g in n["groups"]] == pytest.approx([16.0, 4.0], abs=1e-12)
    assert [g["floored"] for g in n["groups"]] == [False, False]
    half = T95_4 * math.sqrt(20.0 / 3.0)
    assert half == pytest.approx(5.50440, abs=1e-5)
    assert n["interval_pp"] == pytest.approx([6.0 - half, 6.0 + half], abs=1e-9)
    assert v["decision"]["outcome"] == X.WIN and v["decision"]["at_boundary"] == []
    assert [x for r in v["h_pp"] for x in r] == pytest.approx([x for r in HAND for x in r], abs=1e-12) and v["rows"] == list(FRESH) and v["cols"] == list(BANKED)
    assert decide(HAND, "50M")["decision"]["outcome"] == FR.CONTINUE


def test_it_is_b1s_simple_effect_as_coded_and_the_floor_binds_a_side_whose_seeds_agree():
    """The rule's number IS ``main.b1.cross.simple_effect`` (the frame's §3), and a side whose runs all score alike
    is floored at 1.76^2, so the interval is never narrower than +-3.06 pp at three runs a side."""
    flat = decide(uniform(53.0))["views"]["N"]
    assert [g["floored"] for g in flat["groups"]] == [True, True]
    assert flat["interval_pp"][1] - flat["estimate_pp"] == pytest.approx(FLOOR_HALF, abs=1e-12)
    assert FLOOR_HALF == pytest.approx(3.0635, abs=1e-4)
    design = FR.design_of({"fresh": FRESH, "banked": BANKED})
    raw = X.simple_effect(design, grid(HAND), "fresh", "banked", view="N")
    assert FR.stat(design, grid(HAND)).as_dict() == raw.as_dict()


def test_the_deciding_view_is_n_and_every_other_view_is_reported():
    """home72 at +10 and the unfamiliar panels at 0: N = 1/2 home72 + 1/6 each of H, F, P72 = +5."""
    rows = [[60.0 + d for d in (1.0, -1.0, 0.0)] for _ in range(3)]
    by = {p: ledger(grid(rows if p == "home72" else [[r - 10.0 for r in row] for row in rows]), p) for p in X.PANELS}
    v = FR.read(look_doc(), by)[1]
    assert v["read_on"] == "N" and set(v["views"]) == set(X.VIEWS)
    assert v["views"]["home72"]["estimate_pp"] == pytest.approx(10.0, abs=1e-12)
    assert v["views"]["U"]["estimate_pp"] == pytest.approx(0.0, abs=1e-12)
    assert v["views"]["N"]["estimate_pp"] == pytest.approx(5.0, abs=1e-12)
    assert v["decision"]["outcome"] == X.WIN  # +5 on N with both sides floored: lo = 5 - 3.06 > 0


# --------------------------------------------------------------------------------------------- the classes
def _at(estimate: float, look: str = "100M") -> Dict[str, Any]:
    """Nine equal cells at 50 + ``estimate``: the interval is estimate +- FLOOR_HALF exactly (both sides floored)."""
    return decide(uniform(50.0 + estimate), look)["decision"]


@pytest.mark.parametrize("estimate,outcome,near", [
    # the WIN line: lower end +0.010 clears it; +0.003 is within 0.005 pp, fails, and is named
    (FLOOR_HALF + 0.010, X.WIN, False),
    (FLOOR_HALF + 0.003, X.UNDECIDED, True),
    # the LOSS line: upper end -0.010 clears it; -0.003 does not
    (-FLOOR_HALF - 0.010, X.LOSS, False),
    (-FLOOR_HALF - 0.003, X.NO_USEFUL_GAIN, True),
    # the +4 bar: upper end 3.990 is NO-USEFUL-GAIN; 3.997 is within 0.005 of the bar, so UNDECIDED and named
    (4.0 - FLOOR_HALF - 0.010, X.NO_USEFUL_GAIN, False),
    (4.0 - FLOOR_HALF - 0.003, X.UNDECIDED, True),
    (4.0 - FLOOR_HALF + 0.010, X.UNDECIDED, False),
])
def test_the_100m_classes_and_their_0005_pp_boundaries(estimate, outcome, near):
    """Inputs sit 0.003 or 0.010 pp from a threshold, never within a rounding error of the 0.005 pp band's own edge
    (root rule 8): the class is the conservative one inside the band, and the band is named."""
    got = _at(estimate)
    assert got["outcome"] == outcome
    assert bool(got["at_boundary"]) is near


@pytest.mark.parametrize("estimate,outcome,labels,near", [
    (0.010, FR.CONTINUE, [], False),
    (0.003, FR.STOPPED_AT_50M, [], True),       # ahead by less than 0.005 pp: not ahead
    (-0.003, FR.STOPPED_AT_50M, [], True),
    (-1.0, FR.STOPPED_AT_50M, [], False),
    # the harm line: an upper end 0.010 below -2.0 is HARM-AT-50M; 0.003 below is not, and is named
    (-2.0 - FLOOR_HALF - 0.010, FR.STOPPED_AT_50M, [FR.HARM_AT_50M], False),
    (-2.0 - FLOOR_HALF - 0.003, FR.STOPPED_AT_50M, [], True),
])
def test_the_50m_rule_reads_the_point_estimate_and_its_0005_pp_boundary(estimate, outcome, labels, near):
    got = _at(estimate, "50M")
    assert (got["outcome"], got["labels"], bool(got["at_boundary"])) == (outcome, labels, near)


def test_a_50m_continue_never_reads_the_interval():
    """The 50M rule only stops: an interval far below zero with a point estimate above +0.005 still continues."""
    rows = [[50.6, 50.6, 50.6], [44.0, 44.0, 44.0], [57.5, 57.5, 57.5]]
    got = decide(rows, "50M")
    assert got["views"]["N"]["interval_pp"][0] < -5.0 and got["decision"]["outcome"] == FR.CONTINUE


# ----------------------------------------------------------------------------------------- what it reads by
def test_run_order_within_a_side_and_cell_and_panel_order_change_nothing():
    rng = random.Random(7)
    rows = [[rng.uniform(35.0, 65.0) for _ in range(3)] for _ in range(3)]
    base = FR.read(look_doc(), same_on_every_panel(grid(rows)))[1]
    perm_f, perm_b = [2, 0, 1], [1, 2, 0]
    f2, b2 = [FRESH[i] for i in perm_f], [BANKED[j] for j in perm_b]
    by = {p: list(reversed(ledger(grid(rows), p))) for p in reversed(X.PANELS)}
    moved = FR.read(look_doc(fresh=f2, banked=b2), by)[1]
    for view in X.VIEWS:
        a, b = base["views"][view], moved["views"][view]
        assert (a["estimate_pp"], a["v_hat"]) == pytest.approx((b["estimate_pp"], b["v_hat"]), abs=1e-12)
        assert a["interval_pp"] == pytest.approx(b["interval_pp"], abs=1e-12)
    assert base["decision"]["outcome"] == moved["decision"]["outcome"]


def test_reading_the_other_way_round_negates_the_estimate_and_mirrors_the_interval_and_class():
    """The rule is NOT symmetric in which side is fresh (module docstring): swapping the sides on the same games
    negates D, mirrors [lo, hi] and keeps V, so a WIN reads as a LOSS."""
    fwd = FR.design_of({"fresh": FRESH, "banked": BANKED})
    rev = FR.design_of({"fresh": BANKED, "banked": FRESH})
    s = grid(HAND)
    flipped = {(b, f): 100.0 - v for (f, b), v in s.items()}
    a, b = FR.stat(fwd, s), FR.stat(rev, flipped)
    assert b.estimate == pytest.approx(-a.estimate, abs=1e-12) and b.v_hat == pytest.approx(a.v_hat, abs=1e-12)
    assert (b.lo, b.hi) == pytest.approx((-a.hi, -a.lo), abs=1e-12)
    assert FR.classify("100M", a)["outcome"] == X.WIN and FR.classify("100M", b)["outcome"] == X.LOSS
    # and the statistic reads an oriented cell the same whichever side it was stored from
    assert FR.stat(fwd, flipped).estimate == pytest.approx(a.estimate, abs=1e-12)


# ---------------------------------------------------------------------------------------------- INCONCLUSIVE
def _inconclusive(by: Mapping[str, List[SimpleNamespace]], row_reasons: Sequence[str] = ()) -> Dict[str, Any]:
    inp, v = FR.read(look_doc(), by, row_reasons)
    assert v["decision"]["outcome"] == FR.INCONCLUSIVE and v["decision"]["verdict"] == FR.INCONCLUSIVE
    assert "views" not in v and inp["scores"] == {}
    return v


def test_a_missing_duplicated_foreign_or_short_cell_is_inconclusive_never_a_class():
    good = same_on_every_panel(grid(HAND))
    missing = dict(good, p72=good["p72"][:-1])
    assert any("missing cell" in r for r in _inconclusive(missing)["decision"]["reasons"])
    doubled = dict(good, home72=good["home72"] + good["home72"][:1])
    assert any("appears twice" in r for r in _inconclusive(doubled)["decision"]["reasons"])
    foreign = dict(good, forum72=good["forum72"] + [SimpleNamespace(player="f" * 64, opponent="e" * 64, score=0.5,
                                                                    n_pairs=500, verdict="OK", reasons=(), d=0,
                                                                    games=1000)])
    assert any("outside the look" in r for r in _inconclusive(foreign)["decision"]["reasons"])
    short = dict(good, heldout72_hpfix=ledger(grid(HAND), "heldout72_hpfix", n_pairs=499))
    assert any("499 completed pairs < the registered 500" in r for r in _inconclusive(short)["decision"]["reasons"])
    wrong_side = dict(good, home72=ledger({(b, f): v for (f, b), v in grid(HAND).items()}, "home72"))
    assert any("wrong side" in r for r in _inconclusive(wrong_side)["decision"]["reasons"])


def test_more_than_a_quarter_of_a_cells_games_drawn_is_inconclusive_and_a_quarter_is_not():
    """B1's draw rule, in integers: 250 of 1,000 games is a quarter (read); 251 is more (INCONCLUSIVE)."""
    at = dict(same_on_every_panel(grid(HAND)), forum72=ledger(grid(HAND), "forum72", draws=250))
    assert FR.read(look_doc(), at)[1]["decision"]["outcome"] == X.WIN
    over = dict(at, forum72=ledger(grid(HAND), "forum72", draws=251))
    v = _inconclusive(over)
    assert len(v["decision"]["reasons"]) == 9 and all("251 of 1000 games drawn" in r for r in v["decision"]["reasons"])


def test_a_row_not_played_as_registered_makes_the_look_inconclusive_whatever_the_cells_say():
    v = _inconclusive(same_on_every_panel(grid(HAND)), ["p72: rows played at ['bbbbbbbbbbbb'], not the pin aaaa"])
    assert v["decision"]["reasons"][0].startswith("p72: rows played at")


# ------------------------------------------------------------------------------- the rule as main.reads holds it
def test_the_decision_is_recomputed_from_the_input_document_alone():
    inp, v = FR.read(look_doc(), same_on_every_panel(grid(HAND)))
    again = FR.decide(json.loads(json.dumps(inp)), {"look": "100M"})
    assert json.loads(json.dumps(again)) == json.loads(json.dumps(v))
    with pytest.raises(ValueError, match="not a gen3_read_b1_fresh_cell_v1 of the 50M look"):
        FR.decide(inp, {"look": "50M"})


def test_it_is_registered_in_main_reads_and_refuses_a_spec_and_unknown_params():
    from main.reads import rules as RU

    rule = RU.RULES[FR.RULE]
    assert rule.name == "b1_fresh_cell" and not rule.x35_record
    assert rule.verdict_of(decide(HAND)) == X.WIN
    assert FR.check({"look": "100M"}) == [] and FR.check({"look": "50M"}) == []
    assert any("not by a main.reads spec" in p for p in rule.check_params({"look": "100M"}, models={"a": 1},
                                                                          anchor={}, s={"g": 1}))
    assert any("unknown params" in p for p in FR.check({"look": "100M", "bar_pp": 3.0}))
    assert any("is not one of" in p for p in FR.check({"look": "75M"}))
    with pytest.raises(RU.RuleInputError, match="the '50M' look, the params read '100M'"):
        FR.inputs({"look": look_doc("50M"), "cells_by_panel": {}}, {"look": "100M"})


def test_a_design_needs_two_equal_sides_of_at_least_two_distinct_runs():
    for runs in ({"fresh": FRESH, "banked": BANKED[:2]}, {"fresh": FRESH[:1], "banked": BANKED[:1]},
                 {"fresh": FRESH, "banked": (FRESH[0],) + BANKED[1:]}, {"fresh": FRESH, "control": BANKED}):
        with pytest.raises(X.CrossInputError):
            FR.design_of(runs)
    assert FR.design_of({"fresh": FRESH[:2], "banked": BANKED[:2]}).cells() == [
        (FRESH[0], BANKED[0]), (FRESH[0], BANKED[1]), (FRESH[1], BANKED[0]), (FRESH[1], BANKED[1])]
