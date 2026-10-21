"""AM-X47b-1 / -2: G-C pooled over seed pairs (``main.x35.gc``) and its wiring into every X35 reader. Each test fails
on revert: without the module nothing imports; with the readers back on a bare triple, the bare-triple cases read a
verdict instead of INCONCLUSIVE."""
from __future__ import annotations

import math

import numpy as np
import pytest

from main.x35 import d1 as D1
from main.x35 import gc as GC
from main.x35 import kit as K
from main.x35 import readout as R
from main.x35 import x47 as X

PROD = [3501, 3502, 3503]
TOY = [3601, 3602, 3603]


def _pair(values, battles):
    return np.asarray(values, dtype=np.float64), np.asarray(battles, dtype=object)


def test_side_seeds_are_the_kits():
    assert GC.SIDE_SEEDS == {"production": K.PROD_SEEDS, "toy": K.TOY_SEEDS}


def test_the_point_is_the_equal_weight_mean_of_per_pair_means_by_hand():
    a = _pair([1.0, 0.0, 1.0, 1.0, 1.0, 0.5], ["x", "x", "y", "z", "w", "v"])     # mean 0.75 over 6 turns
    b = _pair([0.0, 0.5, 0.25, 0.25, 0.25], ["x", "y", "z", "w", "v"])           # mean 0.25 over 5 turns
    est, lo, hi = GC.crossed_bootstrap([a, b])
    assert est == pytest.approx(0.5, abs=1e-15)                    # (0.75 + 0.25) / 2, NOT the turn-pooled 5.75 / 11
    assert lo <= est <= hi


def test_constant_zero_pairs_give_a_zero_interval():
    p = _pair([0.0] * 6, ["a", "a", "b", "b", "c", "c"])
    assert GC.crossed_bootstrap([p, p, p]) == (0.0, 0.0, 0.0)


def test_the_interval_carries_the_seed_to_seed_spread():
    """Three pairs that disagree (means 0.0, 0.1, 0.2) with tiny within-pair noise: the crossed interval spans the
    seed spread; a bootstrap of the concatenated turns clustered by (seed, battle), F-AMX-1's sketch, is several times
    narrower (measured ~6.5x here)."""
    rng = np.random.default_rng(0)
    battles = [f"b{k}" for k in range(40) for _ in range(5)]
    pairs = [_pair(m + rng.normal(0, 1e-3, len(battles)), battles) for m in (0.0, 0.1, 0.2)]
    est, lo, hi = GC.crossed_bootstrap(pairs)
    assert est == pytest.approx(0.1, abs=1e-3)
    assert lo < 0.05 and hi > 0.15
    cat = _pair(np.concatenate([p[0] for p in pairs]), [f"{s}:{b}" for s in range(3) for b in battles])
    _, lo1, hi1 = GC.crossed_bootstrap([cat])
    assert (hi1 - lo1) < (hi - lo) / 4


def test_one_battle_draw_is_shared_across_pairs():
    """Pairs that agree battle by battle: a shared draw keeps the pooled mean equal to each pair's, so the interval
    equals the single pair's (an independent draw per pair would shrink it by ~sqrt(3))."""
    battles = [f"b{k}" for k in range(30)]
    v = np.linspace(-1, 1, 30)
    one = GC.crossed_bootstrap([_pair(v, battles)])
    three = GC.crossed_bootstrap([_pair(v, battles)] * 3)
    assert three[1] == pytest.approx(one[1], abs=1e-12) and three[2] == pytest.approx(one[2], abs=1e-12)


def _rows():
    """Two decisive turns and one non-decisive one, actions 0 / 1, 4 seeds (+1 / −1 outcomes)."""
    win, loss = [1.0] * 4, [-1.0] * 4
    return [{"id": "t0", "ok": True, "outcomes": {"0": win, "1": loss}},       # 0 near-best, 1 dominated
            {"id": "t1", "ok": True, "outcomes": {"0": win, "1": loss}},
            {"id": "t2", "ok": True, "outcomes": {"0": win, "1": win}},        # nothing dominated: not decisive
            {"id": "t3", "ok": False}]


def test_b2_counts_starved_decisive_turns_and_applies_rule_8_per_pair():
    index = {"t0": 0, "t1": 1, "t2": 2, "t3": 3}
    battles = ["A", "B", "C", "D"]
    p_off = np.array([[0.9, 0.1], [0.9, 0.1], [0.5, 0.5], [0.5, 0.5]])
    p_on = np.array([[0.005, 0.995], [0.9, 0.1], [0.001, 0.999], [0.5, 0.5]])   # t0 starved; t2 is not decisive
    d, cl, ex = GC.b2_deltas(_rows(), index, battles, p_on, p_off)
    assert list(d) == [1.0, 0.0] and list(cl) == ["A", "B"] and ex == 0
    p_edge = p_on.copy()
    p_edge[1] = [0.01 + 5e-7, 1 - 0.01 - 5e-7]                                 # within 1e-6 of the 1 % line
    d, cl, ex = GC.b2_deltas(_rows(), index, battles, p_edge, p_off)
    assert list(cl) == ["A"] and ex == 1


def test_b1_is_the_entropy_difference_in_nats():
    p_on = np.array([[0.5, 0.5, 0.0]])
    p_off = np.array([[1.0, 0.0, 0.0]])
    d, _ = GC.b1_deltas(p_on, p_off, ["A"])
    assert d[0] == pytest.approx(math.log(2), abs=1e-15)


def test_pool_refuses_a_wrong_side_seed_or_truth():
    rec = GC.constructed_record("toy", TOY, [0.0] * 3, [0.0] * 3)
    assert rec["schema"] == GC.GC_SCHEMA and rec["seeds"] == TOY and rec["truth"]["continuation"] == GC.TRUTH_BY_SIDE["toy"]
    with pytest.raises(R.ReadoutInputError, match="not all production seeds"):
        GC.constructed_record("production", TOY, [0.0] * 3, [0.0] * 3)
    arrays = {s: {"on": "a", "off": "b", "b1": _pair([0.0, 0.0], ["x", "y"]), "b2": _pair([0.0, 0.0], ["x", "y"])}
              for s in TOY}
    with pytest.raises(R.ReadoutInputError, match="AM-X47b-2"):
        GC.pool_arrays("toy", GC.TRUTH_BY_SIDE["production"], arrays)


def test_check_record_refuses_a_bare_triple_other_seeds_and_another_sides_truth():
    rec = GC.constructed_record("production", PROD, [0.0] * 3, [0.0] * 3)
    b1, b2 = GC.check_record(rec, PROD, PROD)
    assert b1[1] <= b1[0] <= b1[2] and b2[1] <= b2[0] <= b2[2]
    with pytest.raises(R.ReadoutInputError, match="seed-pooled record"):
        GC.check_record({"b1": [0, -1, 1], "b2": [0, -1, 1]}, PROD, PROD)
    with pytest.raises(R.ReadoutInputError, match="R1 has"):
        GC.check_record(rec, [3501, 3502, 3504], PROD)
    with pytest.raises(R.ReadoutInputError, match="R1 has"):
        GC.check_record(GC.constructed_record("production", PROD[:2], [0.0] * 2, [0.0] * 2), PROD, PROD)
    wrong = dict(rec, truth={"continuation": GC.TRUTH_BY_SIDE["toy"]})
    with pytest.raises(R.ReadoutInputError, match="AM-X47b-2"):
        GC.check_record(wrong, PROD, PROD)


def _a1_doc(g_c):
    return {"arm": "A1", "look": 1,
            "r1": {"on": {"3501": 60.0, "3502": 60.5, "3503": 61.0}, "off": {"3501": 50.0, "3502": 50.5, "3503": 51.0}},
            "r2": {"on": [200, 400], "off": [200, 400]},
            "g_f": {"h_pp": [[51.0, 50.5, 51.5], [50.8, 51.2, 50.9], [51.1, 50.7, 51.3]]},
            "g_c": g_c, "g_k": {ts: {"on": [250, 500], "off": [250, 500]} for ts in R.G_K_TEAMSETS}}


def test_readout_decide_takes_the_record_and_refuses_a_bare_triple():
    ok = GC.constructed_record("production", PROD, [0.0] * 3, [0.0] * 3)
    v = R.read(_a1_doc(ok))
    g = v.as_dict()["guards"]["G-C"]
    assert g["status"] == R.PASS and g["detail"]["side"] == "production" and len(g["detail"]["per_pair"]) == 3
    assert v.verdict == R.ADOPT
    bare = R.read(_a1_doc({"b1": [0.0, -0.01, 0.01], "b2": [0.0, -0.01, 0.01]}))
    assert bare.as_dict()["guards"]["G-C"]["status"] == R.INCONCLUSIVE and bare.verdict == R.INCONCLUSIVE
    rise = R.read(_a1_doc(GC.constructed_record("production", PROD, [0.0] * 3, [0.2] * 3)))   # B2 detected rise
    assert rise.as_dict()["guards"]["G-C"]["status"] == R.FAIL and rise.verdict == R.TRADE
    toy_truth = R.read(_a1_doc(GC.constructed_record("toy", TOY, [0.0] * 3, [0.0] * 3)))    # the toy's record
    assert toy_truth.as_dict()["guards"]["G-C"]["status"] == R.INCONCLUSIVE


def test_x47_decide_needs_the_toy_record_over_its_seeds():
    off = {"3601": 50.0, "3602": 50.4, "3603": 49.6}
    on = {"3601": 60.0, "3602": 60.4, "3603": 59.6}
    doc = {"look": 1, "c0_final": off, "x47_s_eq": on, "x47_final": on,
           "g_f": {"h_pp": [[50.0 + a - b for b in (0.5, -0.5, 0.0)] for a in (1.0, -1.0, 0.0)]}}
    assert X.read(dict(doc, g_c=GC.constructed_record("toy", TOY, [0.0] * 3, [0.0] * 3))).verdict == R.ADOPT
    assert X.read(dict(doc, g_c={"b1": [0.0, -0.01, 0.01], "b2": [0.0, -0.01, 0.01]})).verdict == R.INCONCLUSIVE
    assert X.read(dict(doc, g_c=GC.constructed_record("production", PROD, [0.0] * 3, [0.0] * 3))).verdict \
        == R.INCONCLUSIVE


def test_decide_d1_reads_g_c_over_h_seeds():
    hc = {str(3501 + i): {k: [70, 140] for k in D1.PANELS} for i in range(3)}
    hoff = {str(3501 + i): {k: [56, 140] for k in D1.PANELS} for i in range(3)}
    doc = {"arm": "X38", "look": 1, "h": {"on": hc, "off": hoff},
           "p": {"on": {"3501": 50.1, "3502": 50.2, "3503": 50.0}, "off": {"3501": 50.0, "3502": 50.1, "3503": 49.9}},
           "r2": {"on": [250, 400], "off": [264, 400]},
           "g_f": {"h_pp": [[51.0, 50.5, 51.5], [50.8, 51.2, 50.9], [51.1, 50.7, 51.3]]}}
    good = D1.read_d1(dict(doc, g_c=GC.constructed_record("production", PROD, [0.0] * 3, [0.0] * 3)))
    bare = D1.read_d1(dict(doc, g_c={"b1": [0.0, -0.01, 0.01], "b2": [0.0, -0.01, 0.01]}))
    assert good.verdict == D1.ADOPT and bare.verdict == D1.INCONCLUSIVE


def test_am_x47c_2_each_sides_committed_truth_carries_its_sides_stamp():
    """F-5: production's B2 truth is the P* refresh (X26@280M), not READ A's 150M rows, and each committed file is
    stamped with its side's continuation; the 150M truth is refused for a production G-C."""
    from main.policy_spectrum.truth import load_rows
    from utils.paths import repo_path

    assert GC.TRUTH_BY_SIDE["production"] == "X26@280M greedy (both sides)"
    for side, rel in GC.TRUTH_ROWS_REL.items():
        rows = load_rows(repo_path(rel))
        assert len(rows) == 306 and all(r.get("ok") for r in rows)
        assert {r["continuation"] for r in rows} == {GC.TRUTH_BY_SIDE[side]}
    arrays = {s: {"on": "a", "off": "b", "b1": _pair([0.1, 0.2], ["x", "y"]), "b2": _pair([0.0, 1.0], ["x", "y"])}
              for s in PROD}
    with pytest.raises(R.ReadoutInputError, match="AM-X47c-2"):
        GC.pool_arrays("production", "X26@150M greedy (both sides)", arrays)
