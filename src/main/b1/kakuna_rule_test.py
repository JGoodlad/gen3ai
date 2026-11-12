"""``main.reads`` rule ``b1_kakuna`` (BOUNDARY-1's Kakuna meter): the clustering addendum's ``our_team`` and ``dyadic``
standard errors reproduced from its ``deals.csv``; the variance composition, the seed floor, the wider scheme and the
Satterthwaite df on cases computed by hand (and by a longhand twin with its own loops); rule 8 at and just off each
boundary; every typed refusal and INCONCLUSIVE; ``inputs`` -> ``decide`` end to end on cells in the engine's shape with
real cell dirs; the registered spec; and the operating-characteristics script holding the rule's OWN functions.
Synthetic cells, no game."""
from __future__ import annotations

import csv
import importlib.util
import json
import math
import random
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pytest

from main.reads import rules as RU
from main.b1 import kakuna_rule as KB
from utils.paths import repo_path

ADDENDUM = repo_path("designs/research_state/measurements/sprime_honest_play_2026-11-06/addendum_clustering")
B1_DIR = repo_path("designs/research_state/measurements/boundary1_scratch_2026-11-09")
SPEC = B1_DIR / "read_spec_kakuna.json"
OC = B1_DIR / "kakuna_oc.py"
SHORT = {"heldout72_hpfix": "H", "forum72": "F", "p72": "P72"}

T90_DF2 = 0.9 / math.sqrt(2 * 0.95 * 0.05)    # Student-t 0.95 quantile on 2 df, its closed form (the table's 2.920)


def _cell(a: Sequence[int], b: Sequence[int], n: Sequence[float], w: Sequence[float]) -> KB.Cell:
    return (np.array(a, dtype=int), np.array(b, dtype=int), np.array(n, dtype=float), np.array(w, dtype=float))


# ------------------------------------------------------------------------------------------------ the addendum
def _addendum() -> Dict[str, Dict[str, KB.Cell]]:
    """``{statistic: {panel: (our team, their team, den, num)}}`` from the addendum's deals, its ``load`` re-typed:
    ``raw_paired`` is the raw win share over the paired deals, ``searched_goal_b`` the searched win share over all."""
    rows = list(csv.DictReader(open(ADDENDUM / "deals.csv")))
    assert len(rows) == 5100
    out: Dict[str, Dict[str, KB.Cell]] = {"raw_paired": {}, "searched_goal_b": {}}
    for p in SHORT:
        rs = [r for r in rows if r["panel"] == p]
        ours = [r["our_team"].rsplit(".", 1)[0] for r in rs]
        theirs = [r["their_team"].rsplit(".", 1)[0] for r in rs]
        tid = {t: i for i, t in enumerate(sorted(set(ours) | set(theirs)))}
        a, b = [tid[t] for t in ours], [tid[t] for t in theirs]
        pair = [float(r["raw"] != "missing" and r["searched"] != "missing") for r in rs]
        out["raw_paired"][p] = _cell(a, b, pair, [float(r["raw"] == "win") * f for r, f in zip(rs, pair)])
        out["searched_goal_b"][p] = _cell(a, b, [1.0] * len(rs), [float(r["searched"] == "win") for r in rs])
    return out


@pytest.mark.parametrize("stat", ["raw_paired", "searched_goal_b"])
def test_the_addendums_our_team_and_dyadic_se_are_reproduced_from_its_deals(stat):
    """One run with weight 1 IS the addendum's single score: V_T is its sandwich (clustering.py ``v_cluster`` /
    ``v_dyadic``), per panel and pooled U (panels as equal-weight strata)."""
    want = json.loads((ADDENDUM / "clustering.json").read_text())["results"][stat]
    data = {"run": _addendum()[stat]}
    scopes = {SHORT[p]: {p: 1.0} for p in SHORT}
    scopes["U"] = {p: 1 / 3 for p in SHORT}
    for scope, pw in scopes.items():
        psi = KB.linear_terms(data, {"run": 1.0}, pw)
        assert KB.run_scores(data, ["run"], pw)["run"] == pytest.approx(want[scope]["our_team"]["point"], rel=1e-12)
        ours, dyad = KB.team_variance(psi, data, "our_team"), KB.team_variance(psi, data, "dyadic")
        assert math.sqrt(ours["v_t"]) == pytest.approx(want[scope]["our_team"]["se_sandwich"], rel=1e-10)
        assert math.sqrt(dyad["v_t"]) == pytest.approx(want[scope]["dyadic"]["se"], rel=1e-10)
        assert ours["v_ts"] == pytest.approx(ours["v_t"], rel=1e-12)          # one run: split by run = not split
        assert dyad["v_ts"] == pytest.approx(dyad["v_t"], rel=1e-12)
    assert want["H"]["our_team"]["se_sandwich"] != want["H"]["dyadic"]["se"]  # the two schemes are distinct numbers


def test_the_dyadic_shortcut_equals_the_pair_matrix_on_mirrors_and_repeats():
    """``q_dyadic`` never builds the n x n matrix; the addendum's does. Equal on data with mirror rows (one team in
    both roles), repeated pairs and the same two teams in swapped roles."""
    rng = np.random.default_rng(5)
    for n, k in ((40, 5), (60, 9), (25, 3)):
        a, b = rng.integers(0, k, n), rng.integers(0, k, n)
        a[:3], b[:3] = [0, 1, 1], [0, 0, 0]                                   # a mirror, a pair, the pair swapped
        a[3], b[3] = 0, 1
        psi = rng.standard_normal(n)
        m = np.zeros((n, k))
        m[np.arange(n), a] = 1.0
        m[np.arange(n), b] = 1.0
        teams = len(set(a) | set(b))
        want = float(psi @ ((m @ m.T) > 0) @ psi * teams / (teams - 1))
        got, seen = KB.q_dyadic(psi, a, b)
        assert seen == teams and got == pytest.approx(want, rel=1e-10)
        sums: Dict[int, float] = {}
        for x, t in zip(psi, a):
            sums[int(t)] = sums.get(int(t), 0.0) + float(x)
        g = len(sums)
        assert KB.q_our_team(psi, a) == (pytest.approx(sum(v * v for v in sums.values()) * g / (g - 1), rel=1e-10), g)


# ------------------------------------------------------------------------------------------------ by hand
ARMS2 = {"X": ["x1", "x2"], "Y": ["y1", "y2"]}
COEF2 = {"X": 1.0, "Y": -1.0}


def _tiny(wins: Mapping[str, Tuple[int, int]]) -> Dict[str, Dict[str, KB.Cell]]:
    """One panel, two deals a run (our team 0 against their team 2, our team 1 against their team 3), 10 games a deal."""
    return {r: {"P": _cell([0, 1], [2, 3], [10, 10], list(w))} for r, w in wins.items()}


def test_the_composition_on_a_case_computed_by_hand():
    """Rates 0.7, 0.5 | 0.4, 0.5: psi = 0.15, s2 = 0.0125, sum w^2 = 1 so V_S = 0.0125. The linear terms are +-0.025
    in x1, x2, y1 and 0 in y2, all with team 0 on the + side: across runs team 0 sums to 0.075.

    our_team (G = 2): V_T = 2 * 0.075^2 * 2 = 0.0225; V_TS = 3 runs * (2 * 0.025^2 * 2) = 0.0075;
      V = max(0.0125, 0.0075) + max(0, 0.0225 - 0.0075) = 0.0275; df = 0.0275^2 / (0.0125^2 / 2 + 0.015^2 / 1).
    dyadic (4 teams, the same sharing): V_T = 0.01125 * 4/3 = 0.015; V_TS = 3 * 0.00125 * 4/3 = 0.005;
      V = 0.0125 + 0.010 = 0.0225; df = 0.0225^2 / (0.0125^2 / 2 + 0.010^2 / 3)."""
    st = KB.contrast(_tiny({"x1": (8, 6), "x2": (6, 4), "y1": (3, 5), "y2": (5, 5)}), ARMS2, COEF2, {"P": 1.0}, 0.90)
    assert st["psi"] == pytest.approx(0.15, abs=1e-15) and st["s2"] == pytest.approx(0.0125, abs=1e-15)
    assert st["df_s"] == 2 and st["run_weights"] == {"x1": 0.5, "x2": 0.5, "y1": -0.5, "y2": -0.5}
    o, d = st["schemes"]["our_team"], st["schemes"]["dyadic"]
    assert (o["v_s"], o["v_t"], o["v_ts"], o["v"]) == pytest.approx((0.0125, 0.0225, 0.0075, 0.0275), abs=1e-15)
    assert (d["v_s"], d["v_t"], d["v_ts"], d["v"]) == pytest.approx((0.0125, 0.015, 0.005, 0.0225), abs=1e-15)
    assert o["df_t"] == 1 and d["df_t"] == 3 and o["clusters"] == {"P": 2} and d["clusters"] == {"P": 4}
    assert o["df"] == pytest.approx(0.0275 ** 2 / (0.0125 ** 2 / 2 + 0.015 ** 2 / 1), rel=1e-12)
    assert d["df"] == pytest.approx(0.0225 ** 2 / (0.0125 ** 2 / 2 + 0.010 ** 2 / 3), rel=1e-12)
    assert o["df"] == pytest.approx(2.494845360824742, rel=1e-12) and d["df"] == pytest.approx(4.542056074766355, rel=1e-12)
    for x in (o, d):
        assert x["half"] == pytest.approx(KB.t_quantile(0.90, x["df"]) * math.sqrt(x["v"]), rel=1e-12)
        assert (x["lo"], x["hi"]) == pytest.approx((0.15 - x["half"], 0.15 + x["half"]), abs=1e-15)
    # THE WIDER scheme decides: here our_team's (fewer clusters, a smaller df, a larger V)
    assert o["half"] > d["half"] and st["deciding"] == "our_team" and (st["lo"], st["hi"]) == (o["lo"], o["hi"])


def test_team_noise_that_cancels_across_runs_still_counts_split_by_run():
    """Every run favours team 0 alike, so across runs the +1 / -1 weights cancel the team sums: V_T = 0. V_TS (0.0025 a
    run) is the floor, the seed variance is 0, and V = max(0, 0.01) + max(0, 0 - 0.01) = 0.01 on df_S: a sum
    V_S + V_T would call this contrast exact."""
    st = KB.contrast(_tiny({"x1": (8, 6), "x2": (8, 6), "y1": (5, 3), "y2": (5, 3)}), ARMS2, COEF2, {"P": 1.0}, 0.90)
    o = st["schemes"]["our_team"]
    assert st["psi"] == pytest.approx(0.30, abs=1e-15) and o["v_s"] == 0.0
    assert o["v_t"] == pytest.approx(0.0, abs=1e-18) and o["v_ts"] == pytest.approx(0.01, abs=1e-15)
    assert o["v"] == pytest.approx(0.01, abs=1e-15) and o["df"] == 2.0
    assert o["half"] == pytest.approx(T90_DF2 * 0.1, rel=1e-9)


def test_the_seed_floor():
    """No team noise at all (each run wins alike on both teams): V_T = V_TS = 0 and V is V_S alone, on df_S = 2.
    s2 = (0.2^2 + 0.2^2) / 2 = 0.04, half = t(0.95, 2) * 0.2."""
    st = KB.contrast(_tiny({"x1": (9, 9), "x2": (5, 5), "y1": (5, 5), "y2": (5, 5)}), ARMS2, COEF2, {"P": 1.0}, 0.90)
    for s in KB.SCHEMES:
        x = st["schemes"][s]
        assert (x["v_t"], x["v_ts"]) == (0.0, 0.0) and x["v"] == x["v_s"] == pytest.approx(0.04, abs=1e-15)
        assert x["df"] == 2.0 and x["half"] == pytest.approx(T90_DF2 * 0.2, rel=1e-9)
    assert st["deciding"] == "our_team"                                       # a tie takes the first scheme
    # V never falls below V_S, whatever the team parts
    assert KB.compose(4.0, 1.0, 0.5, 2, 10)["v"] == 4.0 and KB.compose(4.0, 1.0, 3.0, 2, 10)["v"] == 6.0


def test_one_run_an_arm_has_no_seed_variance_and_is_refused():
    with pytest.raises(RU.RuleInputError, match="no degree of freedom"):
        KB.seed_variance({"x1": 0.5, "y1": 0.4}, {"X": ["x1"], "Y": ["y1"]}, {"x1": 1.0, "y1": -1.0})
    # ONE arm with two runs is one degree of freedom: its variance stands for the single-run arm's too
    sv = KB.seed_variance({"x1": 0.5, "x2": 0.7, "y1": 0.4}, {"X": ["x1", "x2"], "Y": ["y1"]},
                          {"x1": 0.5, "x2": 0.5, "y1": -1.0})
    assert sv["df_s"] == 1 and sv["s2"] == pytest.approx(0.02, abs=1e-15) and sv["v_s"] == pytest.approx(0.03, abs=1e-15)


def test_compose_and_the_satterthwaite_bounds():
    c = KB.compose(4.0, 1.0, 3.0, 2, 10)                                      # A = V_S = 4, B = 3 - 1 = 2
    assert (c["a"], c["b"], c["v"]) == (4.0, 2.0, 6.0) and c["df"] == pytest.approx(36 / (16 / 2 + 4 / 10), rel=1e-12)
    c = KB.compose(1.0, 4.0, 3.0, 2, 10)                                      # A = V_TS = 4, B = max(0, 3 - 4) = 0
    assert (c["a"], c["b"], c["v"], c["df"]) == (4.0, 0.0, 4.0, 2.0)
    c = KB.compose(1.0, 2.0, 5.0, 2, 10)                                      # A = V_TS = 2, B = 3
    assert (c["a"], c["b"], c["v"]) == (2.0, 3.0, 5.0) and c["df"] == pytest.approx(25 / (4 / 2 + 9 / 10), rel=1e-12)
    # never below df_S: the raw Satterthwaite here is ~1.0 (a team part on ONE df), the registered df stays 4
    raw = (0.01 + 10.0) ** 2 / (0.01 ** 2 / 4 + 10.0 ** 2 / 1)
    assert raw < 1.01 and KB.compose(0.01, 0.0, 10.0, 4, 1)["df"] == 4.0
    # never above df_S + df_T (equal parts on equal dfs reach it exactly)
    assert KB.compose(1.0, 1.0, 2.0, 4, 4)["df"] == 8.0
    assert KB.compose(0.0, 0.0, 0.0, 2, 10) == {"a": 0.0, "b": 0.0, "v": 0.0, "df": 2.0}
    assert KB.t_quantile(0.90, 2) == pytest.approx(T90_DF2, rel=1e-9) and T90_DF2 == pytest.approx(2.920, abs=5e-4)
    # on 2 df the quantile has a closed form, (2u - 1) / sqrt(2u(1 - u)); a non-integer df sits between its integer
    # neighbours' table values (4.303 on 2 df, 3.182 on 3 df)
    assert KB.t_quantile(0.95, 2) == pytest.approx(0.95 / math.sqrt(2 * 0.975 * 0.025), rel=1e-9)
    assert KB.t_quantile(0.95, 3) == pytest.approx(3.182446, rel=1e-6)
    assert KB.t_quantile(0.95, 3) < KB.t_quantile(0.95, 2.5) < KB.t_quantile(0.95, 2)


def test_the_wider_scheme_decides_and_a_tie_takes_our_team():
    assert KB.wider({"our_team": {"half": 1.0}, "dyadic": {"half": 2.0}}) == "dyadic"
    assert KB.wider({"our_team": {"half": 2.0}, "dyadic": {"half": 1.0}}) == "our_team"
    assert KB.wider({"our_team": {"half": 1.0}, "dyadic": {"half": 1.0}}) == "our_team"
    # dyadic decides when the dependence runs through THEIR team: every deal a different team of ours, two of theirs
    wins = {"x1": [2, 2, 2, 0, 0, 0], "x2": [2, 2, 2, 0, 0, 0], "y1": [0, 0, 0, 2, 2, 2], "y2": [0, 0, 0, 2, 2, 2]}
    data = {r: {"P": _cell([0, 1, 2, 3, 4, 5], [6, 6, 6, 7, 7, 7], [2] * 6, w)} for r, w in wins.items()}
    st = KB.contrast(data, ARMS2, COEF2, {"P": 1.0}, 0.90)
    assert st["schemes"]["dyadic"]["half"] > st["schemes"]["our_team"]["half"] > 0
    assert st["deciding"] == "dyadic" and (st["lo"], st["hi"]) == (st["schemes"]["dyadic"]["lo"], st["schemes"]["dyadic"]["hi"])


def _longhand(data: Mapping[str, Mapping[str, KB.Cell]], arms: Mapping[str, Sequence[str]], coef: Mapping[str, float],
              pw: Mapping[str, float]) -> Dict[str, Any]:
    """The module docstring written out with its own loops (no numpy reduction, the dyadic sum over EVERY pair)."""
    w = {r: coef[a] / len(arms[a]) for a in coef for r in arms[a]}
    rate = {r: {p: sum(data[r][p][3]) / sum(data[r][p][2]) for p in pw} for r in w}
    score = {r: sum(pw[p] * rate[r][p] for p in pw) for r in w}
    df_s = sum(len(arms[a]) - 1 for a in coef)
    ss = 0.0
    for a in coef:
        mean = sum(score[r] for r in arms[a]) / len(arms[a])
        ss += sum((score[r] - mean) ** 2 for r in arms[a])
    v_s = ss / df_s * sum(x * x for x in w.values())
    rows = {p: [(r, int(a), int(b), w[r] * pw[p] * (float(win) - rate[r][p] * float(n)) / float(sum(data[r][p][2])))
                for r in w for a, b, n, win in zip(*data[r][p])] for p in pw}

    def ours(rs: List[Tuple[str, int, int, float]]) -> Tuple[float, int]:
        sums: Dict[int, float] = {}
        for _r, a, _b, x in rs:
            sums[a] = sums.get(a, 0.0) + x
        return sum(v * v for v in sums.values()) * len(sums) / (len(sums) - 1), len(sums)

    def dyad(rs: List[Tuple[str, int, int, float]]) -> Tuple[float, int]:
        k = len({t for _r, a, b, _x in rs for t in (a, b)})
        tot = sum(xi * xj for _ri, ai, bi, xi in rs for _rj, aj, bj, xj in rs if {ai, bi} & {aj, bj})
        return tot * k / (k - 1), k

    out: Dict[str, Any] = {"psi": sum(w[r] * score[r] for r in w), "v_s": v_s, "df_s": df_s}
    for name, q in (("our_team", ours), ("dyadic", dyad)):
        v_t = sum(q(rows[p])[0] for p in pw)
        v_ts = sum(q([x for x in rows[p] if x[0] == r])[0] for p in pw for r in w)
        df_t = sum(q(rows[p])[1] - 1 for p in pw)
        a_part, b_part = max(v_s, v_ts), max(0.0, v_t - v_ts)
        v = a_part + b_part
        df = min(max(v * v / (a_part ** 2 / df_s + b_part ** 2 / df_t), df_s), df_s + df_t)
        out[name] = {"v_t": v_t, "v_ts": v_ts, "v": v, "df": df, "df_t": df_t}
    return out


ARMS4 = {"C": ["c1", "c2"], "W": ["w1", "w2"], "Br": ["b1", "b2"], "WB": ["wb1", "wb2"]}
WIDTH = {"W": 0.5, "WB": 0.5, "C": -0.5, "Br": -0.5}
BREADTH = {"Br": 0.5, "WB": 0.5, "C": -0.5, "W": -0.5}
INTER = {"WB": 1, "W": -1, "Br": -1, "C": 1}


@pytest.mark.parametrize("seed", [1, 2, 3])
@pytest.mark.parametrize("coef", [WIDTH, INTER, {"W": 1.0, "C": -1.0}])
def test_contrast_equals_the_longhand_twin_on_random_shared_deals(seed, coef):
    """Shared deals (one draw of teams for every run), a team-by-arm effect so that V_T, V_TS and V_S all matter, two
    panels of unequal weight."""
    rng = np.random.default_rng(seed)
    pw = {"P": 0.75, "Q": 0.25}
    data: Dict[str, Dict[str, KB.Cell]] = {r: {} for labs in ARMS4.values() for r in labs}
    for p in pw:
        a, b = rng.integers(0, 7, 30), rng.integers(0, 7, 30)
        for arm, labs in ARMS4.items():
            lean = rng.uniform(0.2, 0.8, 7)
            for r in labs:
                data[r][p] = _cell(a, b, [2] * 30, rng.binomial(2, lean[a]))
    st, want = KB.contrast(data, ARMS4, coef, pw, 0.95), _longhand(data, ARMS4, coef, pw)
    assert st["psi"] == pytest.approx(want["psi"], abs=1e-14) and st["df_s"] == want["df_s"]
    for s in KB.SCHEMES:
        got = st["schemes"][s]
        assert got["v_s"] == pytest.approx(want["v_s"], rel=1e-10, abs=1e-18)
        for k in ("v_t", "v_ts", "v", "df"):
            assert got[k] == pytest.approx(want[s][k], rel=1e-9, abs=1e-18), (s, k)
        assert got["df_t"] == want[s]["df_t"] and st["df_s"] <= got["df"] <= st["df_s"] + got["df_t"]
    assert st["schemes"][st["deciding"]]["half"] == max(x["half"] for x in st["schemes"].values())


# ------------------------------------------------------------------------------------------------ rule 8
def test_at_boundary_exactly_at_and_just_off():
    eps, tick = Fraction(5, 1000), Fraction(1, 10 ** 12)                      # the registered band: 0.005 pp
    assert KB.EPS_PP == "0.005" and KB.TIMEOUT_FRAC == Fraction(1, 4)
    assert KB.classify(eps, 3.0) == {"class": KB.SPANS, "at_boundary": True}             # exactly at: no crossing
    assert KB.classify(eps + tick, 3.0) == {"class": KB.POSITIVE, "at_boundary": False}
    assert KB.classify(eps - tick, 3.0) == {"class": KB.SPANS, "at_boundary": True}
    assert KB.classify(-3.0, -eps) == {"class": KB.SPANS, "at_boundary": True}
    assert KB.classify(-3.0, -eps - tick) == {"class": KB.NEGATIVE, "at_boundary": False}
    assert KB.classify(-3.0, -eps + tick) == {"class": KB.SPANS, "at_boundary": True}
    assert KB.classify(0.0049, 3.0)["class"] == KB.SPANS and KB.classify(0.0051, 3.0)["class"] == KB.POSITIVE
    assert KB.classify(-3.0, -0.0049)["class"] == KB.SPANS and KB.classify(-3.0, -0.0051)["class"] == KB.NEGATIVE
    assert KB.classify(-1.0, 1.0) == {"class": KB.SPANS, "at_boundary": False}
    assert KB.classify(0.5, 3.0) == {"class": KB.POSITIVE, "at_boundary": False}
    assert KB.classify(-3.0, -0.5) == {"class": KB.NEGATIVE, "at_boundary": False}
    # a net: >= 0 only when it is clearly above 0; AT-BOUNDARY takes < 0
    assert KB.net_sign(eps) == {"sign": KB.NET_LT, "at_boundary": True}
    assert KB.net_sign(eps + tick) == {"sign": KB.NET_GE, "at_boundary": False}
    assert KB.net_sign(-eps) == {"sign": KB.NET_LT, "at_boundary": True}
    assert KB.net_sign(0.0) == {"sign": KB.NET_LT, "at_boundary": True}
    assert KB.net_sign(-1.0) == {"sign": KB.NET_LT, "at_boundary": False}
    assert KB.net_sign(2.0) == {"sign": KB.NET_GE, "at_boundary": False}


# ------------------------------------------------------------------------------------------------ the bootstraps
def test_the_bootstraps_are_seeded_and_the_literal_one_is_anti_conservative_at_two_seeds():
    """No team noise, so only the seed picks move: arm X's runs score 0.9 and 0.5. n-of-n picks the mean of two (SD
    0.2 / sqrt(2) = 0.1414); n - 1 = one pick has SD 0.2 = sqrt(V_S), the unbiased size."""
    data = _tiny({"x1": (9, 9), "x2": (5, 5), "y1": (5, 5), "y2": (5, 5)})
    boot = KB.bootstrap_rates(data, ARMS2, ["P"], 20000, 11)
    got = KB.bootstrap_interval(boot, ARMS2, COEF2, {"P": 1.0}, 0.90)
    assert got["literal"]["computed"] and got["rao_wu"]["computed"]
    assert got["literal"]["used_draws"] == got["rao_wu"]["used_draws"] == 20000
    assert got["literal"]["sd"] == pytest.approx(0.2 / math.sqrt(2), rel=0.02)
    assert got["rao_wu"]["sd"] == pytest.approx(0.2, rel=0.02)
    assert (got["rao_wu"]["lo"], got["rao_wu"]["hi"]) == pytest.approx((0.0, 0.4), abs=1e-12)
    again = KB.bootstrap_interval(KB.bootstrap_rates(data, ARMS2, ["P"], 20000, 11), ARMS2, COEF2, {"P": 1.0}, 0.90)
    assert again == got
    other = KB.bootstrap_interval(KB.bootstrap_rates(data, ARMS2, ["P"], 20000, 12), ARMS2, COEF2, {"P": 1.0}, 0.90)
    assert other["literal"]["sd"] != got["literal"]["sd"]


# ------------------------------------------------------------------------------------------------ end to end
ENGINE = "port=aaaa/1;env=bbbb/2"
PANELS = ("forum72", "p72")
UNIT_SEEDS = (0, 1, 2, 3)
LABELS = [r for labs in ARMS4.values() for r in labs]
SEEDS = {lab: 5101 + i for i, lab in enumerate(LABELS)}
P_WIN = {"C": 0.25, "W": 0.85, "Br": 0.25, "WB": 0.85}
HALVES = tuple(KB.HALF_PEER_REPORT)


def _deal(panel: str, seed: int, index: int) -> Tuple[str, str]:
    """One deal's two team FILES (ours a ``.txt``, Kakuna's a ``.gen3ou_team``): shared by every run and both halves."""
    r = random.Random(f"{panel}/{seed}/{index}")
    return f"team_{r.randrange(12):03d}.txt", f"team_{r.randrange(12):03d}.gen3ou_team"


def _params(**over: Any) -> Dict[str, Any]:
    p: Dict[str, Any] = {
        "group": "kakuna",
        "contrasts": {"width": {"arms": dict(WIDTH), "panels": list(PANELS)},
                      "breadth_f": {"arms": dict(BREADTH), "panels": ["forum72"]},
                      "inter": {"arms": dict(INTER), "panels": list(PANELS)}},
        "reported_contrasts": ["inter"], "nets": {}, "level": 0.90, "games_per_panel": 80, "min_games_frac": 0.95,
        "bootstrap": {"draws": 200, "seed": 7}}
    p.update(over)
    return p


def _spec(**group: Any) -> SimpleNamespace:
    arm_of = {r: a for a, labs in ARMS4.items() for r in labs}
    models = {lab: SimpleNamespace(arm=arm_of[lab], seed=SEEDS[lab], load={}, ref=SimpleNamespace(kind="final"))
              for lab in LABELS}
    g = dict(name="kakuna", models=tuple(LABELS), teamsets=PANELS, per_game=False, imports=None, unit_seeds=UNIT_SEEDS)
    g.update(group)
    return SimpleNamespace(models=models, anchor={"kakuna": SimpleNamespace(**g)})


def _world(root: Path, p_win: Mapping[str, float] = P_WIN) -> Dict[str, Any]:
    """The assembled cells of a finished read, in the engine's shape, with every unit's cell dir on disk
    (``games.jsonl`` and Kakuna's draw log per half)."""
    arm_of = {r: a for a, labs in ARMS4.items() for r in labs}
    cells: Dict[str, Dict[str, Any]] = {lab: {} for lab in LABELS}
    units: List[Dict[str, Any]] = []
    for lab in LABELS:
        for panel in PANELS:
            wins = 0
            for seed in UNIT_SEEDS:
                name = f"{lab}__{panel}__{seed:03d}"
                games, rows = [], []
                for half in HALVES:
                    draws = []
                    for i in range(10):
                        our, their = _deal(panel, seed, i)
                        res = "win" if random.Random(f"{name}/{half}/{i}").random() < p_win[arm_of[lab]] else "loss"
                        games.append([half, i, our, res, 30])
                        rows.append({"half": half, "index": i, "our_team": our, "result": res, "hit_forfeit_limit": False})
                        draws.append(their)
                    (root / name / half).mkdir(parents=True)
                    (root / name / half / KB.HALF_PEER_REPORT[half]).write_text(json.dumps({"team_draws": draws}))
                (root / name / "games.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
                w = sum(g[3] == "win" for g in games)
                wins += w
                units.append({"unit": name, "label": lab, "panel": panel, "unit_seed": seed, "counted": True, "n": 20,
                              "wins": w, "engine": ENGINE, "games": games, "cell_dir": str(root / name)})
            cells[lab][panel] = {"wins": wins, "ties": 0, "n": 80, "units": 4, "planned": 4, "skipped": []}
    return {"anchor": {"kakuna": {"teamsets": list(PANELS), "engine": ENGINE, "regime_id": "b" * 16, "inconclusive": [],
                                  "unplayed": 0, "cells": cells, "units": units}}}


def _read(cells: Mapping[str, Any], p: Optional[Dict[str, Any]] = None, spec: Optional[SimpleNamespace] = None
          ) -> Dict[str, Any]:
    p, spec = p or _params(), spec or _spec()
    assert RU.RULES["b1_kakuna"].check_params(p, models=spec.models, anchor=spec.anchor, s={}) == []
    doc = json.loads(json.dumps(RU.inputs("b1_kakuna", cells, p, spec)))     # the decision reads the FILE's form
    return RU.decide("b1_kakuna", doc, p)


def _unit(cells: Mapping[str, Any], lab: str, panel: str, seed: int) -> Dict[str, Any]:
    return next(u for u in cells["anchor"]["kakuna"]["units"]
                if (u["label"], u["panel"], u["unit_seed"]) == (lab, panel, seed))


def _rewrite_games(u: Mapping[str, Any], edit: Any) -> None:
    path = Path(u["cell_dir"]) / "games.jsonl"
    rows = [json.loads(ln) for ln in path.read_text().splitlines()]
    path.write_text("".join(json.dumps(edit(i, r)) + "\n" for i, r in enumerate(rows)))


def _deal_level(cells: Mapping[str, Any]) -> Dict[str, Dict[str, KB.Cell]]:
    """The fixture's games as one row a GAME, straight from the unit rows and ``_deal`` (never through ``inputs``)."""
    out: Dict[str, Dict[str, List[List[float]]]] = {lab: {p: [] for p in PANELS} for lab in LABELS}
    for u in cells["anchor"]["kakuna"]["units"]:
        for _half, i, our, res, _turns in u["games"]:
            their = _deal(u["panel"], u["unit_seed"], i)[1]
            out[u["label"]][u["panel"]].append([int(our[5:8]), int(their[5:8]), 1.0, float(res == "win")])
    return {lab: {p: _cell(*zip(*rows)) for p, rows in by.items()} for lab, by in out.items()}


def test_inputs_to_decide_end_to_end(tmp_path):
    cells = _world(tmp_path)
    d = _read(cells)
    assert d["schema"] == KB.SCHEMA and d["level"] == 0.90 and d["engine"] == ENGINE
    assert d["arms"] == ARMS4 and d["seeds"] == SEEDS and d["bootstrap"] == {"draws": 200, "seed": 7}
    rates = {lab: {p: cells["anchor"]["kakuna"]["cells"][lab][p]["wins"] / 80 for p in PANELS} for lab in LABELS}
    for lab in LABELS:
        for p in PANELS:
            run = d["per_run"][lab][p]
            assert (run["n"], run["units"], run["planned"], run["skipped"], run["caps"], run["ties"]) == (80, 4, 4, 0, 0, 0)
            assert run["rate"] == rates[lab][p]

    def arm(a: str, panels: Sequence[str]) -> float:
        return sum(rates[r][p] for r in ARMS4[a] for p in panels) / (2 * len(panels))

    w, bf, it = d["contrasts"]["width"], d["contrasts"]["breadth_f"], d["contrasts"]["inter"]
    assert w["psi_pp"] == pytest.approx(50 * (arm("W", PANELS) + arm("WB", PANELS) - arm("C", PANELS) - arm("Br", PANELS)),
                                        abs=1e-10)
    assert bf["psi_pp"] == pytest.approx(
        50 * (arm("Br", ["forum72"]) + arm("WB", ["forum72"]) - arm("C", ["forum72"]) - arm("W", ["forum72"])), abs=1e-10)
    assert it["psi_pp"] == pytest.approx(100 * (arm("WB", PANELS) - arm("W", PANELS) - arm("Br", PANELS) + arm("C", PANELS)),
                                         abs=1e-10)
    assert w["psi_pp"] == pytest.approx(sum(w["per_panel_pp"].values()) / 2, abs=1e-10) and set(w["per_panel_pp"]) == set(PANELS)
    assert set(bf["per_panel_pp"]) == {"forum72"} and bf["panels"] == ["forum72"] and bf["arms"] == BREADTH
    # the statistic is the rule's on the GAME-level rows: aggregating a run-panel's games by team pair loses nothing
    direct = _deal_level(cells)
    for name, coef, panels in (("width", WIDTH, PANELS), ("breadth_f", BREADTH, ("forum72",)), ("inter", INTER, PANELS)):
        st = KB.contrast(direct, ARMS4, coef, {p: 1 / len(panels) for p in panels}, 0.90)
        blk = d["contrasts"][name]
        assert blk["deciding_scheme"] == st["deciding"] and blk["df_s"] == st["df_s"] == 4
        assert (blk["lo_pp"], blk["hi_pp"]) == pytest.approx((100 * st["lo"], 100 * st["hi"]), abs=1e-9)
        for s in KB.SCHEMES:
            got, want = blk["schemes"][s], st["schemes"][s]
            assert got["v_pp2"] == pytest.approx(1e4 * want["v"], rel=1e-9) and got["df"] == pytest.approx(want["df"], rel=1e-9)
            assert got["v_t_pp2"] == pytest.approx(1e4 * want["v_t"], rel=1e-9)
            assert got["v_ts_pp2"] == pytest.approx(1e4 * want["v_ts"], rel=1e-9)
            assert got["clusters"] == want["clusters"]
        assert blk["class"] == KB.classify(blk["lo_pp"], blk["hi_pp"])["class"]
        assert blk["bootstrap"]["literal"]["computed"] and blk["bootstrap"]["rao_wu"]["computed"]
        assert blk["bootstrap"]["literal"]["lo_pp"] < blk["psi_pp"] < blk["bootstrap"]["literal"]["hi_pp"]
    # 60 pp of width on 8 runs: POSITIVE; nothing else was put in
    assert w["class"] == KB.POSITIVE and w["lo_pp"] > 40 and bf["class"] == KB.SPANS and it["class"] == KB.SPANS
    dec = d["decision"]
    assert dec["class"] == KB.CLASSIFIED and dec["reasons"] == [] and dec["nets"] == {}
    assert dec["contrasts"] == {"width": KB.POSITIVE, "breadth_f": KB.SPANS}
    assert dec["reported_contrasts"] == {"inter": KB.SPANS}
    assert dec["verdict"] == "width POSITIVE; breadth_f SPANS-ZERO"           # a REPORTED contrast stays out of the line
    assert RU.RULES["b1_kakuna"].verdict_of(d) == dec["verdict"]
    # the design block says what the lane dealt: shared deals, drawn with replacement, never balanced
    for p in PANELS:
        des = d["design"][p]
        assert des["deals"] == 4 * 2 * 10 and des["teams"] <= 12 and des["our_use_max"] > des["our_use_min"] >= 1
        assert des["mirror_deals"] == sum(_deal(p, s, i)[0][:8] == _deal(p, s, i)[1][:8] for s in UNIT_SEEDS
                                          for i in range(10)) * 2


def test_the_arm_order_flips_the_sign_and_a_negative_is_classed(tmp_path):
    d = _read(_world(tmp_path, {"C": 0.85, "W": 0.25, "Br": 0.85, "WB": 0.25}))
    assert d["contrasts"]["width"]["class"] == KB.NEGATIVE and d["contrasts"]["width"]["hi_pp"] < -40
    assert d["decision"]["verdict"].startswith("width NEGATIVE; ")


def test_a_net_sums_two_contrasts_of_one_coefficient_set(tmp_path):
    p = _params(contrasts={"width_f": {"arms": dict(WIDTH), "panels": ["forum72"]},
                           "width_p": {"arms": dict(WIDTH), "panels": ["p72"]}},
                nets={"width_net": ["width_f", "width_p"]}, reported_contrasts=[])
    d = _read(_world(tmp_path), p)
    net = d["nets"]["width_net"]
    assert net["pair"] == ["width_f", "width_p"] and net["sign"] == KB.NET_GE and not net["at_boundary"]
    assert net["sum_pp"] == d["contrasts"]["width_f"]["psi_pp"] + d["contrasts"]["width_p"]["psi_pp"]
    assert net["reported_interval"]["psi_pp"] == pytest.approx(net["sum_pp"], abs=1e-9)
    assert d["decision"]["nets"] == {"width_net": KB.NET_GE}
    assert d["decision"]["verdict"] == "width_f POSITIVE; width_p POSITIVE; width_net >= 0"


def test_a_tie_is_a_non_win_and_is_counted(tmp_path):
    cells = _world(tmp_path)
    u = _unit(cells, "w1", "forum72", 0)
    k = next(i for i, g in enumerate(u["games"]) if g[3] == "win")
    u["games"][k][3] = "tie"
    u["wins"] -= 1
    cells["anchor"]["kakuna"]["cells"]["w1"]["forum72"]["wins"] -= 1
    cells["anchor"]["kakuna"]["cells"]["w1"]["forum72"]["ties"] = 1
    _rewrite_games(u, lambda i, r: dict(r, result="tie") if i == k else r)
    run = _read(cells)["per_run"]["w1"]["forum72"]
    assert (run["n"], run["ties"]) == (80, 1) and run["rate"] == run["wins"] / 80


# ------------------------------------------------------------------------------------------------ refusals
def _refused(cells: Mapping[str, Any], match: str) -> None:
    with pytest.raises(RU.RuleInputError, match=match):
        RU.inputs("b1_kakuna", cells, _params(), _spec())


def test_no_opponent_team_per_game_is_a_typed_refusal(tmp_path):
    cells = _world(tmp_path)
    u = _unit(cells, "c2", "p72", 2)
    report = Path(u["cell_dir"]) / "peer_challenge" / KB.HALF_PEER_REPORT["peer_challenge"]
    kept = report.read_text()
    report.unlink()
    _refused(cells, "no readable peer report .*opponent's team per game is unknown")
    report.write_text(json.dumps({"team_draws": json.loads(kept)["team_draws"][:9]}))
    _refused(cells, "holds 9 team draw\\(s\\) for 10 game\\(s\\)")
    report.write_text(json.dumps({"other": 1}))
    _refused(cells, "holds no team draw\\(s\\)")
    report.write_text(kept)
    RU.inputs("b1_kakuna", cells, _params(), _spec())                         # whole again: no refusal
    u["games"][0][0] = "third_half"
    _refused(cells, "half 'third_half' has no known peer report")


def test_a_counted_unit_without_its_rows_or_its_turn_cap_flag_is_refused(tmp_path):
    cells = _world(tmp_path)
    u = _unit(cells, "b1", "forum72", 1)
    _rewrite_games(u, lambda i, r: {k: v for k, v in r.items() if not (i == 4 and k == "hit_forfeit_limit")})
    _refused(cells, "carries no hit_forfeit_limit")
    flip = {"win": "loss", "loss": "win"}
    _rewrite_games(u, lambda i, r: dict(r, hit_forfeit_limit=False, result=flip[r["result"]] if i == 4 else r["result"]))
    _refused(cells, "is not the assembled row")
    (Path(u["cell_dir"]) / "games.jsonl").unlink()
    _refused(cells, "no readable .*games.jsonl")
    cells2 = _world(tmp_path / "again")
    _unit(cells2, "b1", "forum72", 1)["games"].pop()
    _refused(cells2, "19 per-game row\\(s\\), n 20")
    cells3 = _world(tmp_path / "third")
    _unit(cells3, "b1", "forum72", 1)["cell_dir"] = None
    _refused(cells3, "names no cell dir")


def test_a_deal_played_on_other_teams_by_one_run_is_refused(tmp_path):
    """The contrast's team clusters cross runs: one deal is ONE pair of teams in every run (the lane's seeded draw)."""
    cells = _world(tmp_path)
    u = _unit(cells, "wb2", "forum72", 3)
    report = Path(u["cell_dir"]) / "ours_challenge" / KB.HALF_PEER_REPORT["ours_challenge"]
    draws = json.loads(report.read_text())["team_draws"]
    draws[0] = "team_099.gen3ou_team"
    report.write_text(json.dumps({"team_draws": draws}))
    _refused(cells, "one deal is ONE pair of teams in every run")


def test_the_per_game_rows_must_be_the_assembled_cell(tmp_path):
    cells = _world(tmp_path)
    cells["anchor"]["kakuna"]["cells"]["c1"]["p72"]["wins"] += 1
    _refused(cells, "c1 on p72: the per-game rows hold")


def _uncount(cells: Mapping[str, Any], lab: str, panel: str, seed: int) -> None:
    u = _unit(cells, lab, panel, seed)
    u["counted"] = False
    c = cells["anchor"]["kakuna"]["cells"][lab][panel]
    c.update(n=c["n"] - 20, wins=c["wins"] - u["wins"], units=c["units"] - 1, skipped=[u["unit"]])


def test_a_short_run_is_inconclusive_never_dropped_or_reweighted(tmp_path):
    cells = _world(tmp_path)
    _uncount(cells, "w2", "p72", 1)                                           # 60 of 80 games, the floor 0.95 * 80 = 76
    d = _read(cells)
    assert d["decision"]["verdict"] == d["decision"]["class"] == KB.INCONCLUSIVE and d["contrasts"] == {}
    assert d["decision"]["inconclusive_kind"] == "inputs"
    assert d["decision"]["reasons"] == ["w2 on p72: 60 counted game(s), the registered 80 (at least 76)"]
    assert d["per_run"]["w2"]["p72"]["n"] == 60 and d["per_run"]["w2"]["p72"]["skipped"] == 1
    # the registered fraction is the bar: at 0.75 the same 60 games are a counted run-panel
    assert _read(cells, _params(min_games_frac=0.75))["decision"]["class"] == KB.CLASSIFIED


def _cap(cells: Mapping[str, Any], lab: str, panel: str, k: int) -> None:
    """Mark the first ``k`` games of the run-panel as ended at OUR turn cap (10 a unit-half, in file order)."""
    left = k
    for seed in UNIT_SEEDS:
        take, left = min(left, 20), left - min(left, 20)
        _rewrite_games(_unit(cells, lab, panel, seed), lambda i, r, t=take: dict(r, hit_forfeit_limit=i < t))


def test_more_than_a_quarter_of_a_run_panel_at_the_turn_cap_is_inconclusive(tmp_path):
    cells = _world(tmp_path)
    _cap(cells, "c1", "forum72", 20)                                          # exactly 1/4: not MORE than a quarter
    d = _read(cells)
    assert d["decision"]["class"] == KB.CLASSIFIED and d["per_run"]["c1"]["forum72"]["caps"] == 20
    _cap(cells, "c1", "forum72", 21)
    d = _read(cells)
    assert d["decision"]["verdict"] == KB.INCONCLUSIVE
    assert d["decision"]["reasons"] == ["c1 on forum72: 21 of 80 games ended at our turn cap (> 1/4)"]


def test_the_groups_own_reasons_block_the_class(tmp_path):
    cells = _world(tmp_path)
    g = cells["anchor"]["kakuna"]
    g.update(inconclusive=["a unit's regime differs"], unplayed=2, engine=None)
    d = _read(cells)
    assert d["decision"]["verdict"] == KB.INCONCLUSIVE and d["decision"]["reasons"] == [
        "kakuna: a unit's regime differs", "kakuna: 2 unit(s) unplayed",
        "kakuna: its counted units are not on ONE known engine"]
    d = RU.decide("b1_kakuna", json.loads(json.dumps(RU.inputs("b1_kakuna", {"anchor": {}}, _params(), _spec()))), _params())
    assert d["decision"]["verdict"] == KB.INCONCLUSIVE and d["decision"]["reasons"] == [
        "kakuna: no assembled anchor group 'kakuna'"]


def test_team_aliases_merge_one_team_under_two_file_names(tmp_path):
    cells = _world(tmp_path)
    base = _read(cells)["design"]["forum72"]
    merged = _read(cells, _params(team_aliases={"forum72": {"team_001": "team_000"}}))["design"]["forum72"]
    assert merged["teams"] == base["teams"] - 1 and merged["deals"] == base["deals"]
    assert KB.team_id("team_001.gen3ou_team", {"team_001": "team_000"}) == "team_000" == KB.team_id("team_000.txt", {})
    with pytest.raises(RU.RuleInputError, match="no team name"):
        KB.team_id(None, {})


# ------------------------------------------------------------------------------------------------ check (spec load)
def _check(p: Mapping[str, Any], spec: Optional[SimpleNamespace] = None) -> List[str]:
    spec = spec or _spec()
    return RU.RULES["b1_kakuna"].check_params(p, models=spec.models, anchor=spec.anchor, s={})


def test_the_rule_is_registered():
    rule = RU.RULES["b1_kakuna"]
    assert rule.check_params is KB.check and rule.inputs is KB.inputs and rule.decide is KB.decide
    assert rule.doc == KB.RULE_DOCS["b1_kakuna"] and rule.x35_record is False
    assert _check(_params()) == []


def test_an_unknown_or_missing_param_refuses_the_spec():
    assert _check(_params(bar_pp=2.0)) == ["unknown 'bar_pp'"]
    assert _check(_params(head_to_head="DELIVERED")) == ["unknown 'head_to_head'"]
    for k in KB.KEYS:
        p = _params()
        del p[k]
        assert f"missing {k!r}" in _check(p), k
    p = _params()
    del p["reported_contrasts"]                                               # optional: every contrast decides
    assert _check(p) == []


@pytest.mark.parametrize("over,want", [
    ({"group": "nope"}, "group: 'nope' is not an anchor group"),
    ({"level": 1.0}, "level: the two-sided confidence level"),
    ({"level": 0.4}, "level: the two-sided confidence level"),
    ({"level": True}, "level: the two-sided confidence level"),
    ({"games_per_panel": 720}, "games_per_panel: 720 is not the group's 4 unit seeds x 20 games"),
    ({"min_games_frac": 0}, "min_games_frac: a number in (0, 1]"),
    ({"min_games_frac": 1.01}, "min_games_frac: a number in (0, 1]"),
    ({"bootstrap": {"draws": 99, "seed": 7}}, "bootstrap: {draws: an int >= 100"),
    ({"bootstrap": {"draws": 200}}, "bootstrap: {draws: an int >= 100"),
    ({"contrasts": {}}, "contrasts: {name: {arms"),
    ({"contrasts": {"w": {"arms": {"W": 1, "C": -0.5}, "panels": list(PANELS)}}, "reported_contrasts": []},
     "contrasts.w.arms: the coefficients sum to 0.5, not 0"),
    ({"contrasts": {"w": {"arms": {"W": 1, "Z": -1}, "panels": list(PANELS)}}, "reported_contrasts": []},
     "contrasts.w.arms: ['Z'] are not arms of the group's models"),
    ({"contrasts": {"w": {"arms": {"W": 1, "C": 0}, "panels": list(PANELS)}}, "reported_contrasts": []},
     "contrasts.w.arms: {arm: a non-zero coefficient}"),
    ({"contrasts": {"w": {"arms": dict(WIDTH), "panels": ["forum72", "p72", "home72"]}}, "reported_contrasts": []},
     "contrasts.w.panels: ['home72'] are not team sets of the group"),
    ({"contrasts": {"w": {"arms": dict(WIDTH), "panels": ["forum72"]}}, "reported_contrasts": []},
     "group 'kakuna': team set(s) ['p72'] are read by no contrast"),
    ({"contrasts": {"w": {"arms": dict(WIDTH), "panels": list(PANELS), "bar": 1}}, "reported_contrasts": []},
     "contrasts.w: exactly {arms, panels}"),
    ({"reported_contrasts": ["width", "breadth_f", "inter"]}, "reported_contrasts: a list of distinct registered"),
    ({"reported_contrasts": ["nope"]}, "reported_contrasts: a list of distinct registered"),
    ({"nets": {"n": ["width", "inter"]}}, "nets.n: its two contrasts differ in arm coefficients"),
    ({"nets": {"n": ["width", "width"]}}, "nets.n: two distinct registered contrasts"),
    ({"nets": {"width": ["width", "inter"]}}, "nets.width: two distinct registered contrasts"),
    ({"nets": None}, "nets: {name: [contrast, contrast]}"),
    ({"team_aliases": {"home72": {"a": "b"}}}, "team_aliases: {team set of the group"),
    ({"team_aliases": {"p72": {"a": "b", "b": "c"}}}, "team_aliases: {team set of the group"),
])
def test_check_names_each_bad_param(over, want):
    probs = _check(_params(**over))
    assert len(probs) == 1 and probs[0].startswith(want), probs


def test_check_refuses_a_group_the_meter_cannot_read():
    assert _check(_params(), _spec(per_game=True)) == ["group 'kakuna' is per_game (the meter plays whole 20-game units)"]
    assert _check(_params(), _spec(imports={"from": "x"}))[0].startswith("group 'kakuna' imports cells")
    spec = _spec()
    spec.models["c1"].load = {"mode": "sample"}
    assert _check(_params(), spec) == ["group 'kakuna': ['c1'] carry load options (the meter plays greedy finals)"]
    spec = _spec()
    spec.models["c2"].seed = spec.models["c1"].seed
    assert _check(_params(), spec) == ["arm 'C': two of its runs share a seed"]
    # the seed floor at spec load: a contrast none of whose arms holds two runs has no seed variance
    spec = _spec(models=("c1", "w1"))
    p = _params(contrasts={"w": {"arms": {"W": 1, "C": -1}, "panels": list(PANELS)}}, reported_contrasts=[])
    assert _check(p, spec) == ["contrasts.w.arms: no arm holds two runs, so the seed variance has no degree of freedom"]


# ------------------------------------------------------------------------------------------------ the registration
def _oc() -> Any:
    spec = importlib.util.spec_from_file_location("b1_kakuna_oc", OC)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_registered_spec_parses_and_holds_the_registered_statistic():
    """BOUNDARY-1's Kakuna spec: until the registration fills ``pin`` it is checked as ``check-spec --pin-pending``
    does, after it as ``check-spec`` loads it."""
    from main.reads import spec as SP
    from main.reads import speccheck

    doc = json.loads(SPEC.read_text())
    if "pin" not in doc:
        got = speccheck.check_spec(SPEC, pin_pending=True)
        assert got["rule"] == "b1_kakuna" and got["pending"] == [] and got["pin"]["state"] == "PENDING"
    s = SP.load(SPEC)
    from main.b1 import arms as A

    assert s.rule == "b1_kakuna" and len(s.models) == 12, "three runs a cell (PREREG §18 Amendment 1, the re-pin)"
    runs = {m.ref.run for m in s.models.values()}
    assert runs == {f"models/{A.run_name(arm, seed)}" for arm, seed in A.LAUNCH_ORDER}
    assert all(m.ref.kind == "final" and m.ref.min_step == 100_000_000 for m in s.models.values())
    assert sorted((m.arm, m.seed) for m in s.models.values()) == [
        ("Br", 5105), ("Br", 5106), ("Br", 5112), ("C", 5101), ("C", 5102), ("C", 5110),
        ("W", 5103), ("W", 5104), ("W", 5111), ("WB", 5107), ("WB", 5108), ("WB", 5113)]
    g = s.anchor["kakuna"]
    assert set(g.models) == set(s.models), "every final plays Kakuna"
    assert tuple(g.teamsets) == ("heldout72_hpfix", "forum72", "p72") and tuple(g.unit_seeds) == tuple(range(36))
    p = s.params
    u = ["heldout72_hpfix", "forum72", "p72"]
    assert p["contrasts"] == {
        "width_U": {"arms": WIDTH, "panels": u}, "breadth_F": {"arms": BREADTH, "panels": ["forum72"]},
        "breadth_U": {"arms": BREADTH, "panels": u}, "width_F": {"arms": WIDTH, "panels": ["forum72"]},
        "interaction_U": {"arms": INTER, "panels": u}}
    # the CONFIRMING contrasts are width_U and breadth_F: everything else is classified but REPORTED
    assert [k for k in p["contrasts"] if k not in p["reported_contrasts"]] == ["width_U", "breadth_F"]
    assert p["nets"] == {} and p["games_per_panel"] == 720 and p["min_games_frac"] == 0.95
    assert p["bootstrap"] == {"draws": 10000, "seed": 20261109}
    assert p["level"] == _oc().REGISTERED_LEVEL                               # the level the OC log reports as registered


def test_the_oc_script_runs_the_rules_own_functions():
    """``kakuna_oc.py``'s coverage and power are the RULE's: it holds this module's functions (never a copy), deals
    as the lane does, and its second-level half-width is ``contrast``'s at that level."""
    oc = _oc()
    for name in ("contrast", "classify", "t_quantile", "linear_terms", "bootstrap_rates", "bootstrap_interval"):
        assert getattr(oc, name) is getattr(KB, name), name
    assert (oc.POSITIVE, oc.NEGATIVE) == (KB.POSITIVE, KB.NEGATIVE)
    assert oc.SEED == 20261109 and oc.LEVELS == (0.95, 0.90) and oc.REGISTERED_LEVEL in oc.LEVELS
    assert (oc.TEAMS, oc.UNIT_SEEDS, oc.DEALS) == (72, 36, 10) and len(oc.A) == len(oc.B) == 360   # x 2 halves = 720
    assert oc.ARMS.keys() == ARMS4.keys() and all(len(v) == 2 for v in oc.ARMS.values()) and oc.MAIN == WIDTH
    assert oc.CONFIRMING == {"width_U": "U", "breadth_F": "F"}
    assert set(oc.TARGETS["U"]) == {"H", "F", "P72"} and oc.TARGETS["F"] == {"F": 1.0}
    data = oc.experiment(np.random.default_rng(3), 1.0, 3.0)
    assert all(float(data[r][p][2].sum()) == 720.0 for r in data for p in oc.PANELS)
    for target in ("U", "F"):
        at95 = KB.contrast(data, oc.ARMS, oc.MAIN, oc.TARGETS[target], 0.95)
        at90 = KB.contrast(data, oc.ARMS, oc.MAIN, oc.TARGETS[target], 0.90)
        assert oc.deciding_half(at95, 0.90) == pytest.approx(at90["schemes"][at90["deciding"]]["half"], rel=1e-12)
        assert oc.deciding_half(at95, 0.95) == pytest.approx(at95["schemes"][at95["deciding"]]["half"], rel=1e-12)
