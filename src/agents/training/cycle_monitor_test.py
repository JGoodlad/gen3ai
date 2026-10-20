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


# ------------------------------------------------------------------------------------------------ the SIZE (F-ED-27)
def diffuse(tau_pp: float, among, seed: int = 21) -> dict:
    """An independent cyclic perturbation of SD ``tau_pp`` (pp of score at ½) on every planned edge whose two nodes
    are both in ``among``, in Elo (the planted-shift unit of :func:`matrix`)."""
    rng = np.random.default_rng(seed)
    steps = [(k + 1) * D for k in range(len(CLIMB))]
    out = {}
    for rp in CM.plan_rows(steps):
        for _lag, s in rp.edges:
            i, j = steps.index(rp.newest), steps.index(s)
            if i in among and j in among:
                out[(i, j)] = float(rng.normal(0.0, tau_pp * 4 / 100)) * 400 / math.log(10)
    return out


WINDOW_MIN = (len(CLIMB) - 8) * D          # the last 8 checks' nodes: n5..n13 (node k sits at (k + 1)·Δ)


def test_the_size_reads_the_recent_window_only():
    """F-ED-27's regression: a strong cycle among OLD nodes makes the accumulated (a) flag, while the recent
    window's size is small and its interval sits below the 1.7 pp bar."""
    mx, new = matrix(CLIMB, planted=diffuse(8.0, range(0, 5)))
    r = CM.signals(mx, new, FAST, seed=11)
    assert r.signals["hodge"] == CM.FLAG and r.detail["hodge"]["p_boot"] < 0.01
    s = CM.cyclic_size(mx, WINDOW_MIN, seed=5, n_boot=400)
    assert s["measured"] and s["nodes"] == 9 and s["edges"] == 36 and s["df"] == 28
    assert s["upper_pp"] < 1.65, s                          # below the 1.7 pp bar by a margin (rule 8)


def test_the_size_is_unbiased_on_the_targets_scale_and_material_when_large():
    sizes = []
    for seed in range(3):
        mx, _new = matrix(CLIMB, planted=diffuse(4.0, range(len(CLIMB)), seed=31 + seed), seed=3 + seed)
        s = CM.cyclic_size(mx, WINDOW_MIN, seed=7, n_boot=400)
        assert s["lower_pp"] > 2.5, s                       # MATERIAL with a margin
        assert s["lower_pp"] <= s["size_pp"] <= s["upper_pp"]
        sizes.append(s["size_pp"])
    assert 3.0 < float(np.mean(sizes)) < 5.0, sizes          # τ = 4 pp planted (the window's realised SD varies)


def test_the_size_of_a_transitive_climb_is_negligible_and_deterministic():
    mx, _new = matrix(CLIMB)
    a = CM.cyclic_size(mx, WINDOW_MIN, seed=9, n_boot=400)
    assert a == CM.cyclic_size(mx, WINDOW_MIN, seed=9, n_boot=400)
    assert a["upper_pp"] < 1.65 and 0.9 < a["phi"] < 1.1, a   # below the bar by a margin (rule 8)
    # a window with no cycle is not measured (never a silent zero)
    tiny = CM.cyclic_size(mx, (len(CLIMB) - 1) * D, seed=9, n_boot=50)
    assert tiny["measured"] is False and "no cycle" in tiny["reason"]


def test_the_noncentrality_bounds_invert_the_cdf():
    from scipy.stats import ncx2

    lo = CM._ncp_bound(80.0, 28, 0.95)
    hi = CM._ncp_bound(80.0, 28, 0.05)
    assert 0 < lo < 80 - 28 < hi
    assert abs(float(ncx2.cdf(80.0, 28, lo)) - 0.95) < 1e-8 and abs(float(ncx2.cdf(80.0, 28, hi)) - 0.05) < 1e-8
    assert CM._ncp_bound(28.0, 28, 0.95) == 0.0 and CM._ncp_bound(10.0, 28, 0.05) == 0.0
