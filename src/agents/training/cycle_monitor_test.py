"""The cycle monitor's RULE (``agents.training.cycle_monitor``) on synthetic matrices: the hybrid row plan, the edge
orientation, the Nash LP, NONE on a transitive run, SUSPECTED on a planted treadmill hole (the lag curve and the
triangle both see it), CONFIRMED only when the TOP-UP games show it again, the rounding band, the abort rule, and
determinism (the same rows give the same JSON). Small bootstrap sizes keep it fast; the registered sizes are
pinned by the rule string."""
from __future__ import annotations

import math

import numpy as np
import pytest

from agents.training import cycle_monitor as CM

D = 10_000_000
FAST = CM.MonitorRule(n_boot=200, posterior_draws=20)


def pent(p: float, n: int, rng: np.random.Generator) -> list:
    """``n`` mirrored pairs of two independent games at win probability ``p`` (pentanomial, the newer side)."""
    wins = (rng.random((n, 2)) < p).sum(1)
    c = [0.0] * 5
    for v in wins:
        c[2 * int(v)] += 1
    return c


def matrix(elo, *, planted=None, pairs=500, seed=3, plan=CM.DEFAULT_PLAN):
    rng = np.random.default_rng(seed)
    steps = [(k + 1) * D for k in range(len(elo))]
    edges = []
    for rp in CM.plan_rows(steps, plan):
        for _lag, s in rp.edges:
            i, j = steps.index(rp.newest), steps.index(s)
            d = elo[i] - elo[j] + (planted or {}).get((i, j), 0.0)
            edges.append((f"n{i}", rp.newest, f"n{j}", s, pent(1 / (1 + 10 ** (-d / 400)), pairs, rng)))
    return CM.build_matrix(edges), f"n{len(elo) - 1}"


CLIMB = [300 * math.log(1 + k) for k in range(14)]


def test_the_hybrid_row_plan():
    steps = [k * D + 7 for k in range(1, 21)] + [25 * D + 3]
    rows = {r.grid: r for r in CM.plan_rows(steps)}
    assert min(rows) == 2 * D and 20 * D in rows
    assert [lag for lag, _s in rows[10 * D].opponents] == [1, 2, 3, 4, 5, 6, 7, 8]
    assert [lag for lag, _s in rows[20 * D].opponents] == [1, 2, 3, 4, 5, 6, 7, 8, 16]
    assert rows[20 * D].newest == 20 * D + 7 and dict(rows[20 * D].opponents)[16] == 4 * D + 7
    # a lag whose node is missing is listed as None, never substituted; it is not a playable edge
    assert rows[25 * D].newest == 25 * D + 3 and dict(rows[25 * D].opponents)[1] is None
    assert [s for _lag, s in rows[25 * D].edges] == [20 * D + 7, 19 * D + 7, 18 * D + 7, 17 * D + 7, 9 * D + 7]
    assert CM.MonitorPlan().lags == (1, 2, 3, 4, 5, 6, 7, 8, 16, 32, 64)


def test_build_matrix_orients_newer_first_and_pools():
    mx = CM.build_matrix([("old", 1, "new", 2, [1, 0, 0, 0, 3]), ("new", 2, "old", 1, [2, 0, 0, 0, 0])])
    assert mx.keys == ["old", "new"] and mx.ei.tolist() == [1] and mx.ej.tolist() == [0]
    assert mx.counts.tolist() == [[5, 0, 0, 0, 1]]           # the first edge reversed (3 LL for new) + 2 LL
    with pytest.raises(CM.MonitorError):
        CM.build_matrix([("a", 1, "a", 1, [1, 0, 0, 0, 0])])


def test_the_nash_lp_on_known_games():
    rps = np.array([[0, 0.2, -0.2], [-0.2, 0, 0.2], [0.2, -0.2, 0]])
    assert np.allclose(CM.nash_lp(rps), [1 / 3] * 3, atol=1e-7)
    trans = np.array([[0, -0.1, -0.2], [0.1, 0, -0.1], [0.2, 0.1, 0]])
    assert np.allclose(CM.nash_lp(trans), [0, 0, 1], atol=1e-9)


def test_a_transitive_climb_reads_none_and_is_deterministic():
    mx, new = matrix(CLIMB)
    r1 = CM.signals(mx, new, FAST, seed=11)
    r2 = CM.signals(mx, new, FAST, seed=11)
    assert r1.to_json() == r2.to_json()
    assert r1.verdict == CM.NONE, r1.to_json()["signals"]
    assert r1.detail["nash"]["support"] == 1 and r1.detail["nash"]["weights"] == {new: 1.0}
    curve = [c["score"] for c in r1.detail["lag"]["curve"]]
    assert curve == sorted(curve)                             # the newest beats older nodes more: monotone


def test_a_planted_hole_is_suspected_by_the_lag_curve_and_a_triangle():
    mx, new = matrix(CLIMB, planted={(13, 9): -120.0})
    r = CM.signals(mx, new, FAST, seed=11)
    assert r.verdict == CM.SUSPECTED
    assert r.signals["lag"] == CM.FLAG and r.signals["triangles"] == CM.FLAG
    lag = next(it for it in r.flagged_items if it["signal"] == "lag")
    assert lag["long"] == "n9"
    edges = CM.topup_edges(r, FAST)
    assert ("n13", "n9") in edges and len(edges) == len(set(edges))


def _topup(edges, elo, planted, seed=5):
    rng = np.random.default_rng(seed)
    out = {}
    for a, b in edges:
        i, j = int(a[1:]), int(b[1:])
        d = elo[i] - elo[j] + planted.get((i, j), 0.0)
        out[(a, b)] = pent(1 / (1 + 10 ** (-d / 400)), FAST.topup_pairs, rng)
    return out


def test_confirmation_reads_the_topup_games_only():
    planted = {(13, 9): -120.0}
    mx, new = matrix(CLIMB, planted=planted)
    r = CM.signals(mx, new, FAST, seed=11)
    edges = CM.topup_edges(r, FAST)
    assert CM.confirm(r, {}, FAST) == {"complete": False}
    assert CM.final_verdict(r, None) == (CM.SUSPECTED, False)
    real = CM.confirm(r, _topup(edges, CLIMB, planted), FAST)
    assert real["complete"] and real["confirmed"]
    assert CM.final_verdict(r, real) == (CM.CONFIRMED, True)
    # the hole was a fluke: fresh games are transitive, so the check stays SUSPECTED (final), never CONFIRMED
    fluke = CM.confirm(r, _topup(edges, CLIMB, {}), FAST)
    assert fluke["complete"] and not fluke["confirmed"]
    assert CM.final_verdict(r, fluke) == (CM.SUSPECTED, True)


def test_the_rounding_band():
    rule = CM.REGISTERED
    assert rule.mc_band == pytest.approx(math.sqrt(0.05 * 0.95 / 2000))
    assert CM._boot_state(0.05, rule) == CM.BAND
    assert CM._boot_state(0.05 - rule.mc_band * 0.99, rule) == CM.BAND
    assert CM._boot_state(0.05 - rule.mc_band * 1.01, rule) == CM.FLAG
    assert CM._boot_state(0.05 + rule.mc_band * 1.01, rule) == CM.CLEAR


def _row(i, a, sa, b, sb, pc, kind=CM.REQUEST_KIND_ROW, rid="monitor_row:r:1:d1", aborted=0):
    return {"row_id": f"w:{i}", "player": {"id": a, "sha256": a, "step": sa}, "opponent": {"id": b, "sha256": b,
            "step": sb}, "pairs": {"counts": list(pc), "n_pairs": sum(pc)}, "request": {"id": rid, "kind": kind},
            "regime": {"regime_id": "r1"}, "counts": {"w": 1, "l": 1, "d": 0, "aborted": aborted}}


def test_evaluate_rows_aborts_and_scope():
    pc = [100, 0, 300, 0, 100]
    rows = [_row(0, "c", 3, "b", 2, pc), _row(1, "c", 3, "a", 1, pc), _row(2, "b", 2, "a", 1, pc)]
    res, conf, verdict, final = CM.evaluate_rows(rows, "c", "monitor_topup:r:1:d1", FAST)
    assert final and verdict in (CM.NONE, CM.INCONCLUSIVE) and conf is None
    bad = [_row(0, "c", 3, "b", 2, pc, aborted=3000), rows[1], rows[2]]
    assert CM.evaluate_rows(bad, "c", "x", FAST)[2] == CM.INCONCLUSIVE
    with pytest.raises(CM.MonitorError):
        CM.evaluate_rows([dict(rows[0], regime={"regime_id": "r2"}), rows[1]], "c", "x", FAST)


def test_the_registered_rule_string_names_every_constant():
    s = CM.REGISTERED.rule_string()
    for frag in ("alpha=0.05", "n_boot=2000", "pairs=500", "topup=500x3", "support_w=0.01", "abort=0.25"):
        assert frag in s
