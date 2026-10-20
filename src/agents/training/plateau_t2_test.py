"""The plateau's TIER 2 rule (``agents.training.plateau_t2``): the panel's Δ (RISING / FLAT / UNDECIDED, INCOMPLETE,
INCONCLUSIVE, NOT_DECLARED) and the §8.3 two-tier signature, each on constructed inputs."""
from __future__ import annotations

from agents.training import cycle_monitor as CM
from agents.training import plateau_t1 as T1
from agents.training import plateau_t2 as T2

D = 10_000_000


def cells(score_pairs: dict, n: int) -> dict:
    """member → a pentanomial of ``n`` pairs of two independent games at win probability ``s``."""
    out = {}
    for m, s in score_pairs.items():
        ww, ll = round(n * s * s), round(n * (1 - s) * (1 - s))
        out[m] = {"counts": [ll, 0, n - ww - ll, 0, ww], "aborted_share": 0.0}
    return out


def test_panel_flat_rising_undecided():
    mem = ["a", "b"]
    flat = T2.panel_verdict(cells({"a": 0.5, "b": 0.5}, 1000), cells({"a": 0.5, "b": 0.5}, 1000), mem, 1000)
    assert flat["verdict"] == T2.FLAT and flat["upper95"] < 14
    rising = T2.panel_verdict(cells({"a": 0.6, "b": 0.6}, 1000), cells({"a": 0.5, "b": 0.5}, 1000), mem, 1000)
    assert rising["verdict"] == T2.RISING and rising["lower95"] > 0
    small = T2.panel_verdict(cells({"a": 0.51, "b": 0.51}, 500), cells({"a": 0.5, "b": 0.5}, 500), mem, 500)
    assert small["verdict"] == T2.UNDECIDED


def test_panel_incomplete_inconclusive_not_declared():
    mem = ["a", "b"]
    assert T2.panel_verdict({}, {}, [], 500)["verdict"] == T2.NOT_DECLARED
    short = T2.panel_verdict(cells({"a": 0.5}, 500), cells({"a": 0.5, "b": 0.5}, 500), mem, 500)
    assert short["verdict"] == T2.INCOMPLETE and short["short"] == ["b"]
    few = T2.panel_verdict(cells({"a": 0.5, "b": 0.5}, 400), cells({"a": 0.5, "b": 0.5}, 400), mem, 500)
    assert few["verdict"] == T2.INCOMPLETE
    ab = cells({"a": 0.5, "b": 0.5}, 500)
    ab["a"]["aborted_share"] = 0.3
    assert T2.panel_verdict(ab, cells({"a": 0.5, "b": 0.5}, 500), mem, 500)["verdict"] == T2.INCONCLUSIVE


def chk(g, t1, mon, panel=T2.NOT_DECLARED):
    return {"grid": g, "tier1": t1, "monitor": mon, "panel": panel}


def test_the_signatures():
    sig = T2.signature
    assert sig([chk(5 * D, T1.GAIN, CM.NONE)], D)["status"] == T2.TRANSITIVE_IMPROVEMENT
    assert sig([chk(5 * D, T1.GAIN, CM.CONFIRMED)], D)["status"] == T2.TREADMILL
    assert sig([chk(5 * D, T1.FLAT, CM.CONFIRMED)], D)["status"] == T2.TREADMILL
    two_flat = [chk(5 * D, T1.FLAT, CM.NONE, T2.FLAT), chk(6 * D, T1.FLAT, CM.NONE, T2.FLAT)]
    assert sig(two_flat, D)["status"] == T2.TRUE_PLATEAU
    no_panel = [chk(5 * D, T1.FLAT, CM.NONE), chk(6 * D, T1.FLAT, CM.NONE)]
    assert sig(no_panel, D)["status"] == T2.PLATEAU_BUT_PANEL
    # SUSPECTED blocks a plateau call (the safe direction); a gap of 2Δ is not "consecutive"
    assert sig([chk(5 * D, T1.FLAT, CM.NONE, T2.FLAT), chk(6 * D, T1.FLAT, CM.SUSPECTED, T2.FLAT)], D)["status"] \
        == T2.CONTINUE
    assert sig([chk(4 * D, T1.FLAT, CM.NONE, T2.FLAT), chk(6 * D, T1.FLAT, CM.NONE, T2.FLAT)], D)["status"] \
        == T2.CONTINUE
    # the newest check with BOTH verdicts decides; a later half-read check does not
    assert sig(two_flat + [chk(7 * D, T1.GAIN, CM.CONTINUE)], D)["status"] == T2.TRUE_PLATEAU
    assert sig([], D)["status"] == T2.CONTINUE
