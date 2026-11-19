"""V-RANK's estimator (:mod:`main.value_search.v_rank_stat`) on planted positions: the within-node slope is recovered
whatever each position's level, every class is read at its own bar, rule 8 turns a bound on a bar into UNDECIDED, look
1 reads NULL only, and the position builder refuses every input it cannot read honestly."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pytest

from main.value_search import v_rank_stat as ST

U, HOME = ["h", "f"], "home"
PARAMS = {"u_panels": U, "home_panel": HOME, "look": 2, "null_bar": 0.25, "margin": 0.25, "rule8_eps": 0.0005,
          "bootstrap": {"B": 300, "seed": 7}}


def _positions(slopes: Dict[str, float], n: int = 60, noise: float = 0.0, seed: int = 0) -> List[Dict[str, Any]]:
    """``n`` deals a panel (each its own team), one 4-action position each: y = level + slope x q_c (+ noise), each position at its own
    level of Q̂ and of y (a level the within-position centring must cancel)."""
    rng = np.random.default_rng(seed)
    out = []
    for panel, slope in slopes.items():
        for i in range(n):
            q = rng.normal(0.0, 0.1, 4)
            y = 0.5 + slope * (q - q.mean()) + rng.normal(0.0, noise, 4) + rng.normal(0.0, 0.2)
            out.append({"key": f"{panel}{i}", "panel": panel, "deal": f"{panel}{i}", "team": f"{panel}t{i}",
                        "stratum": "AB"[i % 2],
                        "turn": 12, "actions": [1, 2, 3, 6], "top": 0, "pick": int(np.argmax(q)),
                        "q": q + rng.normal(0.0, 0.3), "lg": rng.normal(0.0, 1.0, 4), "y": y})
    return out


def test_a_planted_within_node_slope_is_recovered_whatever_the_positions_levels() -> None:
    res = ST.read(_positions({"h": 0.6, "f": 0.4, "home": 0.5}), PARAMS)
    pp = res["primary"]["per_panel"]
    assert pp["h"]["est"] == pytest.approx(0.6) and pp["f"]["est"] == pytest.approx(0.4)
    assert res["primary"]["beta_u"]["est"] == pytest.approx(0.5)             # the U panels weigh equally
    assert res["primary"]["delta"]["est"] == pytest.approx(0.0, abs=1e-9)
    assert res["class"] == ST.HOLDS and res["positions"] == {"h": 60, "f": 60, "home": 60}
    assert res["reported"]["beta_stratum_A"]["u"]["est"] == pytest.approx(0.5)
    assert res["reported"]["legal_actions"]["home"]["est"] == 4.0


def test_the_bootstrap_resamples_teams_not_deals_or_positions() -> None:
    pos = _positions({"h": 0.5, "f": 0.5, "home": 0.5}, noise=0.05)
    res = ST.read(pos, PARAMS)
    one = res["primary"]["beta_u"]
    assert res["primary"]["per_panel"]["h"]["clusters"] == 60
    # one team a deal: the registered team bootstrap IS the design's deal bootstrap
    deal = res["reported"]["beta_deal_clustered"]["u"]
    assert (deal["est"], deal["lo"], deal["hi"]) == (one["est"], one["lo"], one["hi"])
    # the same positions, each team dealt twice (a 20-game unit's shape): the estimate stands, the clusters halve and
    # the interval is the teams', while the REPORTED deal bootstrap still reads the 60 deals
    paired = [dict(x, team=f"{x['panel']}t{i // 2}") for i, x in enumerate(pos)]
    res2 = ST.read(paired, PARAMS)
    two = res2["primary"]["beta_u"]
    assert res2["primary"]["per_panel"]["h"]["clusters"] == 30
    assert one["est"] == pytest.approx(two["est"]) and (one["lo"], one["hi"]) != (two["lo"], two["hi"])
    deal2 = res2["reported"]["beta_deal_clustered"]["u"]
    assert (deal2["lo"], deal2["hi"]) == (one["lo"], one["hi"])
    again = ST.read(pos, PARAMS)["primary"]["beta_u"]
    assert (one["lo"], one["hi"]) == (again["lo"], again["hi"])               # one seeded stream: reproducible


def test_a_team_effect_on_the_slope_widens_the_registered_interval_only() -> None:
    """Teams that differ in their slope, each dealt four times: the deal bootstrap treats the 60 deals as independent
    and is too narrow; the team bootstrap carries the between-team spread."""
    rng = np.random.default_rng(3)
    pos = []
    for panel in ("h", "f", "home"):
        for t in range(15):
            slope = 0.5 + (0.8 if t % 2 else -0.8)
            for d in range(4):
                q = rng.normal(0.0, 0.1, 4)
                pos.append({"key": f"{panel}{t}_{d}", "panel": panel, "deal": f"{panel}{t}_{d}", "team": f"{panel}t{t}",
                            "stratum": "A", "turn": 12, "actions": [1, 2, 3, 6], "top": 0, "pick": 0, "q": q,
                            "lg": rng.normal(0.0, 1.0, 4), "y": 0.5 + slope * (q - q.mean())})
    res = ST.read(pos, PARAMS)
    team, deal = res["primary"]["per_panel"]["home"], res["reported"]["beta_deal_clustered"]["per_panel"]["home"]
    assert team["clusters"] == 15 and deal["clusters"] == 60 and team["est"] == pytest.approx(deal["est"])
    assert team["hi"] - team["lo"] > 1.4 * (deal["hi"] - deal["lo"])


@pytest.mark.parametrize("bu,d,want,dcls", [
    ({"lo": -0.1, "hi": 0.20}, {"lo": -0.5, "hi": -0.1}, ST.NULL, ST.WORSE),          # NULL dominates WORSE
    ({"lo": 0.30, "hi": 0.60}, {"lo": -0.2, "hi": 0.2}, ST.HOLDS, ST.HOLDS),
    ({"lo": 0.30, "hi": 0.60}, {"lo": -0.2, "hi": -0.05}, ST.HOLDS, ST.HOLDS),        # worse, inside the margin
    ({"lo": 0.10, "hi": 0.40}, {"lo": -0.5, "hi": -0.05}, ST.WORSE, ST.WORSE),
    ({"lo": -0.05, "hi": 0.40}, {"lo": -0.2, "hi": 0.2}, ST.UNDECIDED, ST.UNDECIDED),  # LB(β_U) not above 0
    ({"lo": 0.10, "hi": 0.40}, {"lo": -0.4, "hi": 0.1}, ST.UNDECIDED, ST.UNDECIDED),
])
def test_each_class_is_read_at_its_own_bar(bu, d, want, dcls) -> None:
    got = ST.classes(bu, d, null_bar=0.25, margin=0.25, eps=0.0005, look=2)
    assert (got["class"], got["delta_class"], got["boundary"]) == (want, dcls, False)
    assert (got["note"] == "worse, within the margin") == (want == ST.HOLDS and d["hi"] < 0)


@pytest.mark.parametrize("bu,d", [
    ({"lo": -0.1, "hi": 0.2498}, {"lo": -0.5, "hi": 0.3}),       # UB(β_U) within eps of the NULL bar
    ({"lo": 0.3, "hi": 0.6}, {"lo": -0.2497, "hi": 0.3}),        # LB(Δβ) within eps of the margin
    ({"lo": 0.0003, "hi": 0.6}, {"lo": -0.1, "hi": 0.3}),        # LB(β_U) within eps of 0
    ({"lo": 0.3, "hi": 0.6}, {"lo": -0.4, "hi": -0.0002}),       # UB(Δβ) within eps of 0
])
def test_rule_8_a_bound_within_eps_of_a_bar_that_matters_is_undecided(bu, d) -> None:
    got = ST.classes(bu, d, null_bar=0.25, margin=0.25, eps=0.0005, look=2)
    assert got["class"] == ST.UNDECIDED and got["boundary"] is True
    clear = ST.classes(bu, d, null_bar=0.25, margin=0.25, eps=0.0, look=2)
    assert clear["boundary"] is False and clear["class"] != ST.UNDECIDED       # with no band the same bounds decide


def test_look_1_reads_null_only() -> None:
    kw = {"null_bar": 0.25, "margin": 0.25, "eps": 0.0005}
    holds = ST.classes({"lo": 0.3, "hi": 0.6}, {"lo": -0.2, "hi": 0.2}, look=1, **kw)
    assert holds == {"class": ST.CONTINUE, "boundary": False, "delta_class": None, "note": None}
    worse = ST.classes({"lo": 0.1, "hi": 0.4}, {"lo": -0.5, "hi": -0.05}, look=1, **kw)
    assert worse["class"] == ST.CONTINUE and worse["delta_class"] is None
    null = ST.classes({"lo": -0.1, "hi": 0.2}, {"lo": -0.5, "hi": -0.1}, look=1, **kw)
    assert null["class"] == ST.NULL and null["delta_class"] == ST.WORSE
    edge = ST.classes({"lo": -0.1, "hi": 0.2498}, {"lo": -0.5, "hi": -0.1}, look=1, **kw)
    assert edge["class"] == ST.CONTINUE and edge["boundary"] is True
    with pytest.raises(ST.VRankError, match="look"):
        ST.classes({"lo": 0, "hi": 1}, {"lo": 0, "hi": 1}, look=3, **kw)


def test_a_null_value_reads_null_and_a_weaker_u_slope_reads_worse() -> None:
    null = ST.read(_positions({"h": 0.0, "f": 0.0, "home": 0.5}, n=200, noise=0.02), PARAMS)
    assert null["class"] == ST.NULL and null["delta_class"] == ST.WORSE
    worse = ST.read(_positions({"h": 0.3, "f": 0.3, "home": 0.9}, n=200, noise=0.02), PARAMS)
    assert worse["class"] == ST.WORSE and worse["primary"]["delta"]["hi"] < -0.25


def test_the_noise_of_a_mean_is_what_common_draws_leave_of_it() -> None:
    # every action shifted by the same draw: the centred values never move, so no noise
    common = np.array([[0.1, 0.5, 0.9], [0.2, 0.6, 1.0]])
    assert ST.mean_noise(common) == pytest.approx(0.0)
    rng = np.random.default_rng(3)
    s = rng.normal(0.0, 1.0, (4, 4000))
    # independent unit draws, 4 actions: Σ_a Var(centred mean) = (A - 1) / S
    assert ST.mean_noise(s) == pytest.approx(3 / 4000, rel=0.1)
    with pytest.raises(ST.VRankError, match="S >= 2"):
        ST.mean_noise(np.ones((3, 1)))


def _state(key: str = "k", **over: Any) -> Dict[str, Any]:
    s = {"key": key, "panel": "h", "deal": "d0", "team": "t0", "stratum": "A", "turn": 12, "pi_argmax": 2,
         "search": {"searched": True, "played": 3, "q": {"2": 0.5, "3": 0.7, "6": 0.3},
                    "log_pi": {"2": -0.2, "3": -2.0, "6": -3.0}},
         "worlds": {"2": [0.4, 0.6], "3": [0.6, 0.8], "6": [0.2, 0.4]},
         "referee": {"out": {"2": "+-", "3": "++", "6": "--"}}}
    s.update(over)
    return s


def _branch(action: int, y: float, key: str = "k", **over: Any) -> Dict[str, Any]:
    r = {"kind": "branch", "key": key, "action": action, "status": "counted", "y_raw": 0.0, "y_branch": y}
    r.update(over)
    return r


def test_a_position_takes_pis_outcome_from_the_raw_game_and_the_others_from_their_branches() -> None:
    pos, counts = ST.build_positions([_state()], [_branch(3, 1.0), _branch(6, 0.5),
                                                  {"kind": "null", "key": "k", "action": 2, "status": "counted"}])
    (x,) = pos
    assert x["actions"] == [2, 3, 6] and x["top"] == 0 and x["pick"] == 1
    assert list(x["y"]) == [0.0, 1.0, 0.5] and list(x["q"]) == [0.5, 0.7, 0.3]
    assert list(x["q_self"]) == [0.5, 1.0, 0.0] and x["q_noise"] == pytest.approx(0.0)
    assert counts["h"] == {"scored": 1, "excluded": 0, "searched": 1, "complete": 1, "incomplete": 0,
                           "peer_dropped": 0}


def test_a_position_with_a_dropped_branch_is_left_out_whole_and_an_excluded_state_has_no_branch() -> None:
    ex = _state("x", search={"searched": False, "why": "excluded"})
    pos, counts = ST.build_positions([_state(), ex], [_branch(3, 1.0), _branch(6, 0.5, status="dropped",
                                                                               peer_drop=True)])
    assert pos == [] and counts["h"]["incomplete"] == 1 and counts["h"]["peer_dropped"] == 1
    assert counts["h"]["excluded"] == 1 and counts["h"]["scored"] == 2


@pytest.mark.parametrize("scored,branches,why", [
    ([_state()], [_branch(3, 1.0)], "are not its legal actions"),
    ([_state()], [_branch(3, 1.0), _branch(6, 0.5), _branch(7, 0.5)], "are not its legal actions"),
    ([_state()], [_branch(3, 1.0), _branch(3, 0.0), _branch(6, 0.5)], "branched twice"),
    ([_state()], [_branch(3, 1.0), _branch(6, 0.5, status="pending")], "not yet played"),
    ([_state()], [_branch(3, 1.0), _branch(6, 0.5, y_raw=1.0)], "disagree on the raw game"),
    ([_state(), _state()], [_branch(3, 1.0), _branch(6, 0.5)], "scored twice"),
    ([_state()], [_branch(3, 1.0), _branch(6, 0.5), _branch(3, 1.0, key="other")], "never scored"),
    ([_state(search={"searched": False})], [_branch(3, 1.0)], "the search excluded"),
])
def test_the_position_builder_refuses_what_it_cannot_read_honestly(scored, branches, why) -> None:
    with pytest.raises(ST.VRankError, match=why):
        ST.build_positions(scored, branches)


def test_the_reported_rows_never_stop_the_read_and_the_self_referee_decomposes() -> None:
    pos = _positions({"h": 0.5, "f": 0.5, "home": 0.5}, noise=0.05)
    for x in pos:                                     # a referee that IS the outcome, noise-free; Q̂ with no MC noise
        x["q_self"], x["self_noise"], x["q_noise"] = x["y"], 0.0, 0.0
    rep = ST.read(pos, PARAMS)["reported"]
    assert rep["self_referee"]["y_on_q_self"]["u"]["est"] == pytest.approx(1.0)
    assert rep["self_referee"]["q_self_reliability"]["home"]["est"] == pytest.approx(1.0)
    assert rep["qhat_monte_carlo"]["reliability"]["u"]["est"] == pytest.approx(1.0)
    assert rep["pikl_deviation_rate"]["u"]["est"] > 0.5
    only_a = [dict(x, stratum="A") for x in pos]
    rep = ST.read(only_a, PARAMS)["reported"]
    assert "error" in rep["beta_stratum_B"] and "error" not in rep["beta_stratum_A"]


def test_a_panel_with_no_position_refuses_the_read() -> None:
    with pytest.raises(ST.VRankError, match="panel f: no cluster"):
        ST.read(_positions({"h": 0.5, "home": 0.5}), PARAMS)
