"""The plateau's TIER 2 rule (``agents.training.plateau_t2``): the panel's Δ (RISING / FLAT / UNDECIDED, INCOMPLETE,
INCONCLUSIVE, NOT_DECLARED), the size bar on a counted flag (MATERIAL / NEGLIGIBLE / INCONCLUSIVE, the 2026-10-19
amendment) and the §8.3 two-tier signature, each on constructed inputs."""
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
    assert sig([{**chk(5 * D, T1.GAIN, CM.CONFIRMED), "size": T2.MATERIAL}], D)["status"] == T2.TREADMILL
    assert sig([{**chk(5 * D, T1.FLAT, CM.CONFIRMED), "size": T2.MATERIAL}], D)["status"] == T2.TREADMILL
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


def sz(lo: float, hi: float) -> dict:
    return {"measured": True, "lower_pp": lo, "upper_pp": hi, "size_pp": (lo + hi) / 2}


def test_the_size_bar_states():
    bar = T2.SIZE_RULE.bar_pp
    assert bar == 1.7 and T2.SIZE_RULE.window_checks == CM.DEFAULT_PLAN.dense_lags
    assert T2.size_state(sz(2.0, 3.0)) == T2.MATERIAL
    assert T2.size_state(sz(0.0, 1.2)) == T2.NEGLIGIBLE
    assert T2.size_state(sz(1.2, 2.4)) == T2.SIZE_INCONCLUSIVE          # the bar inside the interval
    # rule 8: a bound within 1e-9 of the bar is not a crossing
    assert T2.size_state(sz(bar + 5e-10, 3.0)) == T2.SIZE_INCONCLUSIVE
    assert T2.size_state(sz(0.0, bar - 5e-10)) == T2.SIZE_INCONCLUSIVE
    assert T2.size_state({"measured": False, "reason": "no cycle"}) == T2.SIZE_INCONCLUSIVE
    assert T2.size_state(None) == T2.SIZE_INCONCLUSIVE


def test_a_flag_counts_only_when_material():
    cm = T2.counted_monitor
    for flag in (CM.SUSPECTED, CM.CONFIRMED):
        assert cm(flag, T2.MATERIAL) == flag
        assert cm(flag, T2.NEGLIGIBLE) == CM.NONE
        assert cm(flag, T2.SIZE_INCONCLUSIVE) == CM.INCONCLUSIVE
        assert cm(flag, None) == CM.INCONCLUSIVE                      # a flag with no size read never counts NONE
    for other in (CM.NONE, CM.INCONCLUSIVE, CM.CONTINUE, "NOT_STARTED", None):
        assert cm(other, T2.MATERIAL) == other and cm(other, T2.NEGLIGIBLE) == other


def test_the_signatures_under_the_size_bar():
    sig = T2.signature

    def c(g, t1, mon, size, panel=T2.NOT_DECLARED):
        return {**chk(g, t1, mon, panel), "size": size}

    # F-ED-27: a persistent but NEGLIGIBLE flag no longer blocks a plateau
    neg = [c(5 * D, T1.FLAT, CM.SUSPECTED, T2.NEGLIGIBLE, T2.FLAT), c(6 * D, T1.FLAT, CM.SUSPECTED, T2.NEGLIGIBLE,
                                                                       T2.FLAT)]
    assert sig(neg, D)["status"] == T2.TRUE_PLATEAU
    assert sig([dict(x, panel=T2.NOT_DECLARED) for x in neg], D)["status"] == T2.PLATEAU_BUT_PANEL
    # a MATERIAL flag, or one whose size cannot be told from the bar, still blocks it
    for size in (T2.MATERIAL, T2.SIZE_INCONCLUSIVE, None):
        mixed = [neg[0], c(6 * D, T1.FLAT, CM.SUSPECTED, size, T2.FLAT)]
        assert sig(mixed, D)["status"] == T2.CONTINUE, size
    # a CONFIRMED treadmill needs a MATERIAL size; a NEGLIGIBLE one reads as NONE
    assert sig([c(5 * D, T1.FLAT, CM.CONFIRMED, T2.MATERIAL)], D)["status"] == T2.TREADMILL
    assert sig([c(5 * D, T1.GAIN, CM.CONFIRMED, T2.NEGLIGIBLE)], D)["status"] == T2.TRANSITIVE_IMPROVEMENT
    assert sig([c(5 * D, T1.GAIN, CM.SUSPECTED, T2.MATERIAL)], D)["status"] == T2.CONTINUE
