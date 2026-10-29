"""``main.reads`` rule ``x63_kvk`` (X63 PREREG §6–§7) on constructed cells: a hand computation of every class
(CONTINUE, NOT PURSUED; GAIN → ADOPT-CANDIDATE / TRADE / OVER-RATE by price / OVER-RATE by the floor with r unset /
UNPRICED; RETIRE, UNDECIDED, INCONCLUSIVE), rule 8 on each side of every bar, the z-vs-t switch at σ̂ = the floor,
Stage A's σ̂ = X63-C's σ̂_CC, the spec-load check, the reported-only groups and the draft spec. Synthetic cells, no
game."""
from __future__ import annotations

import json
import math
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pytest

from main.reads import rules as RU
from main.reads import rules_x63 as X

EPS = 1e-9
ENGINE = "port=aaaa/1;env=bbbb/2"
TEAMS = [f"t{i:02d}" for i in range(72)]
STAGE_LABELS = {"A": (["u1"], ["c1", "c2"]), "B": (["u1", "u2"], ["c1", "c2"]),
                "C": (["u1", "u2", "u3"], ["c1", "c2", "c3"])}
SEEDS = {"u1": 2601, "u2": 6302, "u3": 6303, "c1": 2601, "c2": 6302, "c3": 6303}


def _params(stage: str = "B", **over: Any) -> Dict[str, Any]:
    on, off = STAGE_LABELS[stage]
    p: Dict[str, Any] = {"on": list(on), "off": list(off), "primary": "primary", "home": "home", "stage": stage,
                         "screen": 0.010, "kill_bar": 0.015, "stage_c_bar": 0.015, "sigma_floor": 0.010,
                         "inconclusive_sigma": 0.030, "r": 1.0, "rule8_eps": EPS}
    p.update(over)
    return p


def _spec(labels: Sequence[str], reported: Sequence[str] = ()) -> SimpleNamespace:
    models = {lab: SimpleNamespace(arm="x63u" if lab.startswith("u") else ("x26c" if lab == "c1" else "x63c"),
                                   seed=SEEDS[lab], load={}, ref=SimpleNamespace(kind="pinned"))
              for lab in labels}
    groups = {"primary": ("forum72",), "home": ("home72",), **{g: ("p72",) for g in reported}}
    anchor = {g: SimpleNamespace(name=g, models=tuple(labels), teamsets=ts, per_game=False, imports=None)
              for g, ts in groups.items()}
    return SimpleNamespace(models=models, anchor=anchor)


def _group(panel: str, counts: Dict[str, Tuple[int, int]], *, teams: bool = True, engine: Optional[str] = ENGINE,
           inconclusive: Sequence[str] = (), unplayed: int = 0) -> Dict[str, Any]:
    """Each model's ``(wins, games)`` as 20-game units on ``panel``; game i plays our team ``TEAMS[i % 72]`` and the
    first ``wins`` games are wins."""
    cells, units = {}, []
    for lab, (w, n) in counts.items():
        cells[lab] = {panel: {"wins": w, "ties": 0, "n": n, "units": n // 20, "planned": n // 20, "skipped": []}}
        for u in range(n // 20):
            games = [["a" if j < 10 else "b", j, TEAMS[(20 * u + j) % 72] if teams else None,
                      "win" if 20 * u + j < w else "loss", 30] for j in range(20)]
            units.append({"unit": f"{lab}__{panel}__{u:03d}", "label": lab, "panel": panel, "unit_seed": 1000 + u,
                          "counted": True, "n": 20, "wins": sum(g[3] == "win" for g in games), "engine": engine,
                          "games": games})
    return {"teamsets": [panel], "engine": engine, "regime_id": "b" * 16, "inconclusive": list(inconclusive),
            "unplayed": unplayed, "cells": cells, "units": units}


def _run(stage: str, prim: Dict[str, Tuple[int, int]], home: Dict[str, Tuple[int, int]], *,
         rep: Optional[Dict[str, Dict[str, Any]]] = None, **over: Any) -> Dict[str, Any]:
    p = _params(stage, **over)
    labels = list(p["on"]) + list(p["off"])
    cells = {"anchor": {"primary": _group("forum72", prim), "home": _group("home72", home), **(rep or {})}}
    spec = _spec(labels, list(rep or {}))
    assert RU.RULES["x63_kvk"].check_params(p, models=spec.models, anchor=spec.anchor, s={}) == []
    doc = json.loads(json.dumps(RU.inputs("x63_kvk", cells, p, spec)))     # the decision reads the FILE's form
    return RU.decide("x63_kvk", doc, p)


def _pq(w: int, n: int) -> float:
    return (w / n) * (1 - w / n) / n


def _hand(on: List[Tuple[int, int]], off: List[Tuple[int, int]], floor: float = 0.010) -> Dict[str, float]:
    """§6 written out longhand (independent of the module)."""
    pon, poff = [w / n for w, n in on], [w / n for w, n in off]
    d = sum(pon) / len(pon) - sum(poff) / len(poff)
    b = sum(_pq(*x) for x in on) / len(on) ** 2 + sum(_pq(*x) for x in off) / len(off) ** 2
    terms = []
    for xs, ps in ((on, pon), (off, poff)):
        if len(ps) >= 2:
            m = sum(ps) / len(ps)
            terms.append(sum((v - m) ** 2 for v in ps) / (len(ps) - 1) - sum(_pq(*x) for x in xs) / len(xs))
    sig = math.sqrt(max(0.0, sum(terms) / len(terms)))
    se = math.sqrt(b + max(floor, sig) ** 2 * (1 / len(on) + 1 / len(off)))
    return {"delta": d, "b": b, "sigma_hat": sig, "se": se}


G, H = 4000, 2000
B_GAIN = {"u1": (2240, G), "u2": (2260, G), "c1": (2100, G), "c2": (2120, G)}       # Δ +0.035, σ̂ 0, LB +0.0101
HOME_FLAT = {"u1": (1200, H), "u2": (1200, H), "c1": (1200, H), "c2": (1200, H)}   # Δ_home 0, floor LB ≈ 0.585


# ------------------------------------------------------------------------------------------------ the arithmetic
def test_stage_b_gain_matches_the_hand_computation_and_literals():
    d = _run("B", B_GAIN, HOME_FLAT)
    f, hand = d["primary"], _hand([B_GAIN["u1"], B_GAIN["u2"]], [B_GAIN["c1"], B_GAIN["c2"]])
    for k in ("delta", "sigma_hat", "se"):
        assert f[k] == pytest.approx(hand[k], abs=1e-15)
    assert f["binomial_part"] == pytest.approx(hand["b"], abs=1e-18)
    # literals computed by hand (PREREG §6): B = 6.1915625e-05, σ̂² < 0 so σ̂ = 0, SE = √(B + 0.01²)
    assert f["binomial_part"] == pytest.approx(6.1915625e-05, rel=1e-12)
    assert f["sigma_hat_sq"] == pytest.approx(-4.9415625e-05, rel=1e-9) and f["sigma_hat"] == 0.0
    assert f["se"] == pytest.approx(0.012724607066624887, rel=1e-12)
    assert f["q"] == X.Z == 1.959964 and f["q_kind"] == "z" and f["df"] == 2
    assert f["lb"] == pytest.approx(0.010060228235269541, rel=1e-12)
    assert d["decision"]["class"] == X.GAIN and d["decision"]["verdict"] == X.ADOPT
    assert d["decision"]["stage_c_fires"] is False            # Δ̂ ≥ stage_c_bar, but C fires only on an UNDECIDED


def test_stage_a_sigma_is_x63c_sigma_cc():
    """Stage A (1 v 2): σ̂ is the OFF side's alone, = √(max(0, d² − b₁ − b₂) / 2) (X63-C PREREG §3), df 1."""
    prim = {"u1": (2200, G), "c1": (2040, G), "c2": (2160, G)}
    d = _run("A", prim, {k: (1200, H) for k in prim})
    f = d["primary"]
    cc = X.sigma_cc(2160 / G - 2040 / G, _pq(2040, G), _pq(2160, G))
    assert f["sigma_hat"] == pytest.approx(cc, rel=1e-12)
    assert cc == pytest.approx(0.019690416450649305, rel=1e-12)
    assert f["df"] == 1 and set(f["seed_terms"]) == {"off"}
    assert d["decision"]["sigma_cc"] == f["sigma_hat"]
    assert f["delta"] == pytest.approx(0.025, abs=1e-15)
    assert d["decision"]["verdict"] == X.CONTINUE


def test_stage_a_not_pursued_and_the_screen_bar_exactly():
    base = {"c1": (2040, G), "c2": (2160, G)}                                       # mean OFF 0.525
    assert _run("A", dict(base, u1=(2100, G)), {k: (1200, H) for k in ("u1", "c1", "c2")}
                )["decision"]["verdict"] == X.NOT_PURSUED                            # Δ 0
    d = _run("A", dict(base, u1=(2140, G)), {k: (1200, H) for k in ("u1", "c1", "c2")})
    assert abs(d["primary"]["delta"] - 0.010) < EPS                                 # Δ̂ = the screen: rule 8
    assert d["decision"]["verdict"] == X.NOT_PURSUED and d["decision"]["tests"]["delta >= screen"]["near"]


@pytest.mark.parametrize("offset,want", [(+5e-10, X.NOT_PURSUED), (-5e-10, X.NOT_PURSUED), (+2e-9, X.CONTINUE),
                                         (-2e-9, X.NOT_PURSUED)])
def test_rule8_screen_each_side(offset, want):
    assert X.classify({"delta": 0.010 + offset}, _params("A"))["class"] == want


def test_stage_a_power_row_nearest_sigma_cc():
    sp = {"source": "power_x63_v2.log", "rows": [{"sigma_pp": 1.5, "p_np_at_2pp": 0.311, "p_np_at_3pp": 0.168},
                                                 {"sigma_pp": 2.0, "p_np_at_2pp": 0.351, "p_np_at_3pp": 0.227}]}
    assert X.stage_a_power(sp, 0.0196)["row"]["sigma_pp"] == 2.0
    assert X.stage_a_power(sp, 0.0175)["row"]["sigma_pp"] == 2.0                   # a tie goes to the larger σ
    assert X.stage_a_power(sp, 0.0160)["row"]["sigma_pp"] == 1.5
    assert X.stage_a_power(None, 0.02) is None


# ------------------------------------------------------------------------------------------------ B / C classes
def test_gain_trade():
    home = {"u1": (1110, H), "u2": (1110, H), "c1": (1200, H), "c2": (1200, H)}     # Δ_home −0.045
    d = _run("B", B_GAIN, home)
    pr, f = d["decision"]["price"], d["primary"]
    h = _hand([home["u1"], home["u2"]], [home["c1"], home["c2"]])
    net = h["delta"] + 1.0 * f["delta"]
    se_net = math.sqrt(h["se"] ** 2 + f["se"] ** 2)
    assert pr["net"] == pytest.approx(net, abs=1e-15) and net < 0 < net + X.Z * se_net
    assert pr["se_net"] == pytest.approx(se_net, abs=1e-15)
    assert d["decision"]["floor"]["pass"] and d["decision"]["verdict"] == X.TRADE


def test_gain_over_rate_by_price():
    home = {"u1": (1200, H), "u2": (1200, H), "c1": (1400, H), "c2": (1400, H)}     # Δ_home −0.10
    d = _run("B", B_GAIN, home)
    pr = d["decision"]["price"]
    assert pr["net_upper"] < 0 and pr["exceeded"] and d["decision"]["floor"]["pass"]
    assert d["decision"]["verdict"] == X.OVER_RATE


def test_gain_over_rate_by_floor_whatever_r():
    """FLOOR fails ⇒ OVER-RATE before r is looked at: with r unset it is NOT UNPRICED (the re-review's order)."""
    home = {"u1": (980, H), "u2": (980, H), "c1": (1000, H), "c2": (1000, H)}      # pooled 0.49
    for r in (None, 1.0, 50.0):
        d = _run("B", B_GAIN, home, r=r)
        assert not d["decision"]["floor"]["pass"] and d["decision"]["verdict"] == X.OVER_RATE, r


def test_gain_unpriced_prints_r_star():
    home = {"u1": (1150, H), "u2": (1150, H), "c1": (1200, H), "c2": (1200, H)}     # Δ_home −0.025
    d = _run("B", B_GAIN, home, r=None)
    assert d["decision"]["verdict"] == X.UNPRICED and d["decision"]["floor"]["pass"]
    assert d["decision"]["price"]["r_star"] == pytest.approx(0.025 / 0.035, rel=1e-12)
    assert d["decision"]["price"]["within"] is None


def test_floor_is_the_pooled_wilson_lb():
    from main.anchors.results import wilson

    d = _run("B", B_GAIN, {"u1": (1060, H), "u2": (1000, H), "c1": (1200, H), "c2": (1200, H)}, r=None)
    fl = d["decision"]["floor"]
    assert (fl["wins"], fl["n"]) == (2060, 4000) and fl["wilson_lb"] == wilson(2060, 4000)[1]
    assert fl["pass"] == (wilson(2060, 4000)[1] > 0.5)


def test_retire():
    prim = {"u1": (2040, G), "u2": (2060, G), "c1": (2100, G), "c2": (2120, G)}     # Δ −0.015
    d = _run("B", prim, HOME_FLAT)
    h = _hand([prim["u1"], prim["u2"]], [prim["c1"], prim["c2"]])
    assert d["primary"]["ub"] == pytest.approx(h["delta"] + X.Z * h["se"], abs=1e-15)
    assert d["primary"]["ub"] < 0.015
    assert d["decision"]["verdict"] == X.RETIRE and d["decision"]["stage_c_fires"] is False


def test_undecided_and_stage_c():
    fire = _run("B", {"u1": (2170, G), "u2": (2190, G), "c1": (2100, G), "c2": (2120, G)}, HOME_FLAT)  # Δ +0.0175
    assert fire["decision"]["verdict"] == X.UNDECIDED and fire["decision"]["stage_c_fires"] is True
    hold = _run("B", {"u1": (2120, G), "u2": (2140, G), "c1": (2100, G), "c2": (2120, G)}, HOME_FLAT)  # Δ +0.005
    assert hold["decision"]["verdict"] == X.UNDECIDED and hold["decision"]["stage_c_fires"] is False


def test_inconclusive_on_sigma():
    prim = {"u1": (2000, G), "u2": (2480, G), "c1": (2100, G), "c2": (2120, G)}
    d = _run("B", prim, HOME_FLAT)
    h = _hand([prim["u1"], prim["u2"]], [prim["c1"], prim["c2"]])
    assert d["primary"]["sigma_hat"] == pytest.approx(h["sigma_hat"], abs=1e-15)
    assert d["primary"]["sigma_hat"] > 0.030
    assert d["decision"]["verdict"] == X.INCONCLUSIVE and d["decision"]["inconclusive_kind"] == "sigma"


def test_inconclusive_precedes_gain_and_retire():
    p = _params("B")
    for lb, ub in ((0.01, 0.05), (-0.05, 0.01)):
        assert X.classify({"delta": 0.0, "sigma_hat": 0.031, "lb": lb, "ub": ub}, p)["class"] == X.INCONCLUSIVE
    for s in (0.030 + 5e-10, 0.030 - 5e-10):                                      # near the bar: INCONCLUSIVE
        assert X.classify({"delta": 0.0, "sigma_hat": s, "lb": 0.01, "ub": 0.05}, p)["class"] == X.INCONCLUSIVE
    assert X.classify({"delta": 0.0, "sigma_hat": 0.030 - 2e-9, "lb": 0.01, "ub": 0.05}, p)["class"] == X.GAIN


def test_stage_c_t_on_df_4():
    prim = {"u1": (2200, G), "u2": (2400, G), "u3": (2260, G), "c1": (2100, G), "c2": (2120, G), "c3": (2110, G)}
    home = {k: (1200, H) for k in prim}
    d = _run("C", prim, home)
    f = d["primary"]
    assert f["df"] == 4 and f["sigma_hat"] > 0.010 and f["q_kind"] == "t"
    assert f["q"] == pytest.approx(2.776445, abs=1e-6)
    assert "stage_c_fires" not in d["decision"]


# ------------------------------------------------------------------------------------------------ rule 8 and the quantile
@pytest.mark.parametrize("lb,want", [(5e-10, X.UNDECIDED), (-5e-10, X.UNDECIDED), (2e-9, X.GAIN)])
def test_rule8_gain_each_side(lb, want):
    assert X.classify({"delta": 0.0, "sigma_hat": 0.0, "lb": lb, "ub": 0.05}, _params("B"))["class"] == want


@pytest.mark.parametrize("ub,want", [(0.015 - 5e-10, X.UNDECIDED), (0.015 + 5e-10, X.UNDECIDED),
                                     (0.015 - 2e-9, X.RETIRE)])
def test_rule8_retire_each_side(ub, want):
    assert X.classify({"delta": 0.0, "sigma_hat": 0.0, "lb": -0.02, "ub": ub}, _params("B"))["class"] == want


@pytest.mark.parametrize("d,fires", [(0.015 + 5e-10, False), (0.015 - 5e-10, False), (0.015 + 2e-9, True)])
def test_rule8_stage_c_each_side(d, fires):
    assert X.classify({"delta": d, "sigma_hat": 0.0, "lb": -0.01, "ub": 0.05}, _params("B"))["stage_c_fires"] is fires


@pytest.mark.parametrize("dh,within", [(-0.035 + 5e-10, False), (-0.035 - 5e-10, False), (-0.035 + 2e-9, True)])
def test_rule8_within_each_side(dh, within):
    pr = X.price({"delta": 0.035, "se": 0.01}, {"delta": dh, "se": 0.01}, 1.0, EPS)
    assert pr["within"] is within


def test_rule8_floor_each_side(monkeypatch):
    from main.anchors import results as AR

    blk = {"teamsets": ["home72"], "models": {"u1": {"home72": {"wins": 1, "n": 2}}}}
    for lb, ok in ((0.5 + 5e-10, False), (0.5 - 5e-10, False), (0.5 + 2e-9, True)):
        monkeypatch.setattr(AR, "wilson", lambda k, n, z=AR.Z95, lb=lb: (0.5, lb, 0.6))
        assert X.floor_check(blk, ["u1"], EPS)["pass"] is ok


def test_quantile_switch_at_the_floor():
    """z iff σ̂ is clearly ≤ the floor; at the floor (and within eps of it) t (rule 8: the larger quantile)."""
    assert X.quantile(0.0, 0.010, 2, EPS) == (X.Z, "z")
    assert X.quantile(0.010 - 2e-9, 0.010, 2, EPS) == (X.Z, "z")
    for s in (0.010, 0.010 - 5e-10, 0.010 + 5e-10, 0.0101):
        q, kind = X.quantile(s, 0.010, 2, EPS)
        assert kind == "t" and q == pytest.approx(4.302653, abs=1e-6)
    assert [round(X.t975(df), 6) for df in (1, 2, 3, 4)] == [12.706205, 4.302653, 3.182446, 2.776445]


def test_t_vs_z_on_counts_either_side_of_the_floor():
    below = _run("B", B_GAIN, HOME_FLAT)["primary"]
    above = _run("B", {"u1": (2200, G), "u2": (2340, G), "c1": (2100, G), "c2": (2120, G)}, HOME_FLAT)["primary"]
    assert below["sigma_hat"] <= 0.010 and below["q_kind"] == "z"
    assert above["sigma_hat"] > 0.010 and above["q_kind"] == "t" and above["q"] == pytest.approx(4.302653, abs=1e-6)
    assert above["sigma_eff"] == above["sigma_hat"] and below["sigma_eff"] == 0.010


# ------------------------------------------------------------------------------------------------ inputs and reported
def test_input_inconclusive_group():
    p = _params("B")
    spec = _spec(p["on"] + p["off"])
    cells = {"anchor": {"primary": _group("forum72", B_GAIN, engine=None), "home": _group("home72", HOME_FLAT)}}
    d = X.decide(X.inputs(cells, p, spec), p)
    assert d["decision"]["verdict"] == X.INCONCLUSIVE and d["decision"]["inconclusive_kind"] == "inputs"
    cells["anchor"]["primary"] = _group("forum72", B_GAIN, unplayed=3)
    assert X.decide(X.inputs(cells, p, spec), p)["decision"]["verdict"] == X.INCONCLUSIVE


def test_cluster_bootstrap_reported_and_seeded():
    a = _run("B", B_GAIN, HOME_FLAT)["reported"]["cluster_bootstrap_delta_f"]
    b = _run("B", B_GAIN, HOME_FLAT)["reported"]["cluster_bootstrap_delta_f"]
    assert a["computed"] and a["clusters"] == 72 and a == b
    assert a["ci95"][0] < 0.035 < a["ci95"][1]


def test_cluster_bootstrap_without_teams_says_so():
    p = _params("B")
    spec = _spec(p["on"] + p["off"])
    cells = {"anchor": {"primary": _group("forum72", B_GAIN, teams=False), "home": _group("home72", HOME_FLAT)}}
    rep = X.decide(X.inputs(cells, p, spec), p)["reported"]["cluster_bootstrap_delta_f"]
    assert rep["computed"] is False and "our_team" in rep["why"]


def test_reported_groups_and_home_at_stage_a():
    assert X.reported(_params("A", reported=["traj"])) == [("anchor", "traj"), ("anchor", "home")]
    assert X.reported(_params("B", reported=["traj"])) == [("anchor", "traj")]
    rep = {"traj": _group("p72", {k: (600, 1200) for k in ("u1", "u2", "c1", "c2")}, unplayed=2)}
    d = _run("B", B_GAIN, HOME_FLAT, rep=rep, reported=["traj"])
    assert d["reported"]["traj"]["status"] == "INCOMPLETE" and d["decision"]["verdict"] == X.ADOPT


# ------------------------------------------------------------------------------------------------ the spec-load check
def _probs(stage: str = "B", labels: Optional[Sequence[str]] = None, **over: Any) -> List[str]:
    p = _params(stage)
    p.update(over.pop("_over", {}), **over)
    spec = _spec(list(labels or ["u1", "u2", "u3", "c1", "c2", "c3"]))
    return X.check(p, models=spec.models, anchor=spec.anchor, s={})


def test_check_accepts_each_stage():
    for st in ("A", "B", "C"):
        assert _probs(st) == []
    assert _probs("B", r=None) == [] and _probs("B", r=0) == []


@pytest.mark.parametrize("over,needle", [
    ({"stage": "D"}, "stage"), ({"on": []}, "on:"), ({"off": []}, "off:"), ({"on": ["u1"]}, "registered stage"),
    ({"stage": "A"}, "registered stage"), ({"r": -0.5}, "r:"), ({"r": "one"}, "r:"), ({"screen": 0}, "screen"),
    ({"on": ["u1", "c1"]}, "share a seed"), ({"on": ["u1", "u2"], "off": ["u2", "c1"]}, "both sides"),
    ({"primary": "nope"}, "primary"), ({"home": "primary"}, "one group"), ({"bogus": 1}, "unknown"),
    ({"sigma_floor": 0.04}, "below inconclusive_sigma"), ({"rule8_eps": 0.5}, "rule8_eps"),
    ({"stage_a_power": {"source": "x", "rows": []}}, "stage_a_power")])
def test_check_refuses_malformed(over, needle):
    probs = _probs("B", None, **{"_over": over})
    assert any(needle in m for m in probs), probs


def test_check_refuses_missing_r_but_accepts_null():
    p = _params("B")
    del p["r"]
    spec = _spec(["u1", "u2", "c1", "c2"])
    assert "missing 'r'" in X.check(p, models=spec.models, anchor=spec.anchor, s={})


def test_registered():
    assert RU.RULES["x63_kvk"].x35_record is False and RU.RULES["x63_kvk"].decide is X.decide
    assert RU.RULES["x63_kvk"].verdict_of(_run("B", B_GAIN, HOME_FLAT)) == X.ADOPT
