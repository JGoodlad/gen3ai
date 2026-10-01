"""Pure unit tests of the RND variants' pre-registered comparisons (synthetic arrays; no model, no
bank, no ``models/``)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from main.ridealong_read import variants as VA
from main.ridealong_read.meters import conditional_auroc


# ── p-values and Holm ───────────────────────────────────────────────────────────────────────────

def test_holm_hand_checked_example():
    # p sorted: d .005 (x4 = .02), a .01 (x3 = .03), c .03 (x2 = .06), b .04 (x1 = .04 -> cummax .06)
    h = VA.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.005})
    assert [h[k]["p_holm"] for k in "abcd"] == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert [h[k]["reject"] for k in "abcd"] == [True, False, False, True]
    assert [h[k]["p"] for k in "abcd"] == [0.01, 0.04, 0.03, 0.005]
    assert list(h) == ["a", "b", "c", "d"]                      # caller's order is kept


def test_holm_untestable_member_keeps_the_family_size():
    h = VA.holm({"a": 0.02, "b": None})
    assert h["b"]["p"] is None and h["b"]["p_holm"] == 1.0 and not h["b"]["reject"]
    assert h["a"]["p_holm"] == pytest.approx(0.04) and h["a"]["reject"]   # m = 2, not 1


def test_bootstrap_p_values():
    d = np.array([-1.0, 1.0, 2.0, 3.0])
    # #(d <= 0) = 1, #(d >= 0) = 3 -> 2 * min(2, 4) / 5
    assert VA.p_two_sided(d) == pytest.approx(0.8)
    assert VA.p_greater(d) == pytest.approx(2 / 5)
    assert VA.p_less(d) == pytest.approx(4 / 5)
    assert VA.p_two_sided(np.ones(999)) == pytest.approx(2 / 1000)
    assert VA.p_two_sided(np.zeros(9)) == 1.0                   # capped
    assert VA.p_two_sided([np.nan, np.nan]) is None
    assert VA.p_greater(np.array([0.4, 0.6, 0.7]), null=0.5) == pytest.approx(2 / 4)


# ── paired resampling ───────────────────────────────────────────────────────────────────────────

def _bank(n_battles=60, per=8, seed=0):
    rng = np.random.default_rng(seed)
    cl = np.repeat(np.arange(n_battles), per)
    return rng, cl


def test_paired_boot_same_resamples_for_both_scores():
    rng, cl = _bank()
    x = rng.normal(size=len(cl))
    seen = []

    def fn(W):
        seen.append(W.copy())
        return np.stack([(W * x).sum(1), (W * (x + 3.0)).sum(1) - 3.0 * W.sum(1)], axis=1)

    D = VA.paired_boot(cl, fn, n_boot=100)
    # Both columns were computed on ONE weight matrix: their difference is exactly 0 per resample.
    np.testing.assert_allclose(D[:, 0], D[:, 1], atol=1e-9)
    # And a second call (another checkpoint, the same bank rows) sees the same resamples.
    D2 = VA.paired_boot(cl, fn, n_boot=100)
    np.testing.assert_array_equal(D, D2)
    np.testing.assert_array_equal(seen[0], seen[len(seen) // 2])


def test_compare_coverage_identical_variant_is_not_detected_and_a_better_one_is():
    rng, cl = _bank(80, 10)
    n = len(cl)
    lab_b = rng.uniform(size=n) < 0.2
    base = rng.normal(size=n)
    errs = {"rnd": base, "rndv_same": base * 2.0 + 1.0,          # a monotone copy: same ranks
            "rndv_good": base + 3.0 * lab_b}
    out = VA.compare_coverage(errs, {"x": lab_b, "y": rng.uniform(size=n) < 0.3}, cl, n_boot=200)
    vs = out["contrasts"]["x"]["vs_base"]
    assert vs["rndv_same"]["diff"] == 0.0 and vs["rndv_same"]["diff_ci"] == [0.0, 0.0]
    assert vs["rndv_same"]["p"] == 1.0
    assert out["verdicts"]["rndv_same"] == "NOT_DETECTED"
    assert out["verdicts"]["rndv_good"] == "COVERS_BETTER"
    assert out["holm_family"].endswith("= 4")


def test_coverage_verdict_table():
    assert VA.coverage_verdict(True, False) == "COVERS_BETTER"
    assert VA.coverage_verdict(False, True) == "COVERS_WORSE"
    assert VA.coverage_verdict(True, True) == "MIXED"
    assert VA.coverage_verdict(False, False) == "NOT_DETECTED"


def test_cond_aurocs_equal_meters_conditional_auroc():
    rng = np.random.default_rng(3)
    ref = rng.uniform(size=500)
    lab = rng.uniform(size=500) < 0.3
    s1, s2 = rng.normal(size=500), lab + rng.normal(size=500)
    got = VA._cond_aurocs([s1, s2], lab, ref, 5)
    assert got[0] == pytest.approx(conditional_auroc(s1, lab, ref))
    assert got[1] == pytest.approx(conditional_auroc(s2, lab, ref))


def test_compare_v_error_ranks_and_beats_base():
    rng, cl = _bank(100, 6)
    n = len(cl)
    ref = rng.uniform(size=n)
    err = rng.uniform(size=n)
    noise = rng.normal(size=n)
    zs = {"rnd": noise, "rndv_same": noise.copy(), "rndv_good": noise + 4.0 * (err > 0.5)}
    out = VA.compare_v_error(zs, err, ref, cl, n_boot=100)
    assert out["vs_base"]["rndv_same"]["diff"] == 0.0
    assert out["vs_base"]["rndv_same"]["verdict"] == "NOT_DETECTED"
    assert out["vs_base"]["rndv_good"]["verdict"] == "BEATS_BASE"
    assert out["best"] == "rndv_good"
    assert out["ranking_descriptive"][0]["key"] == "rndv_good"


def test_ranked_vs_base_worse_and_base_stands():
    dist = {"rnd": np.full(50, 0.7), "rndv_bad": np.full(50, 0.5)}
    out = VA.ranked_vs_base({"rnd": 0.7, "rndv_bad": 0.5}, dist)
    assert out["vs_base"]["rndv_bad"]["verdict"] == "WORSE_THAN_BASE"
    assert out["best"] == "BASE STANDS"


# ── quantiles, saturation, drift ────────────────────────────────────────────────────────────────

def test_weighted_quantile_unit_and_integer_weights():
    rng = np.random.default_rng(4)
    x = np.round(rng.normal(size=301), 1)                       # ties
    sq = VA.SortedQuantiles(x)
    qs = (0.1, 0.25, 0.5, 0.75, 0.9)
    np.testing.assert_array_equal(sq.point(qs), np.quantile(x, qs, method="inverted_cdf"))
    w = rng.integers(0, 4, len(x))
    rep = np.repeat(x, w)
    np.testing.assert_array_equal(sq(w[None, :].astype(float), qs)[0],
                                  np.quantile(rep, qs, method="inverted_cdf"))


def test_saturation_rule_and_stream_homogenised_guard():
    rng, cl = _bank(60, 10)
    wide = rng.lognormal(0.0, 1.0, len(cl))
    narrow = 1.0 + 0.05 * rng.normal(size=len(cl))              # rel_spread collapses
    keys = ["rnd", "rndv_fast", "rndv_decay", "rndv_small"]
    first = {k: VA.spread_point_and_dist(wide, cl, n_boot=200) for k in keys}
    last = {k: VA.spread_point_and_dist(narrow if k != "rndv_small" else wide, cl, n_boot=200)
            for k in keys}
    f = {k: (v[0]["rel_spread"], v[1]) for k, v in first.items()}
    la = {k: (v[0]["rel_spread"], v[1]) for k, v in last.items()}
    out = VA.saturation_compare(f, la)
    assert out["keys"]["rndv_small"]["verdict"] == "NOT_DETECTED"
    assert out["keys"]["rndv_small"]["r"] == pytest.approx(1.0)
    assert out["stream_homogenised"] is True
    for k in ("rnd", "rndv_fast", "rndv_decay"):
        assert out["keys"][k]["verdict_raw"] == "SATURATED"
        assert out["keys"][k]["verdict"] == "STREAM_HOMOGENISED"
    # Without decay saturating, the guard does not fire: base and fast read SATURATED.
    la["rndv_decay"] = f["rndv_decay"]
    out = VA.saturation_compare(f, la)
    assert out["stream_homogenised"] is False and out["keys"]["rnd"]["verdict"] == "SATURATED"


def test_drift_compare_inflation_and_prediction_overlap():
    rng, cl = _bank(50, 8)
    a = rng.lognormal(size=len(cl))
    out = VA.drift_compare(a, {"B": (4.0 * a, a.copy())}, cl)
    p = out["pairs"]["B"]
    assert p["inflation"] == pytest.approx(4.0) and p["inflation_ci"] == [4.0, 4.0]
    assert p["verdict"] == "DRIFT_CONFIRMED" and p["ci_overlaps_offline_prediction"] is True
    assert p["renormalised"]["inflation"] == pytest.approx(1.0)
    assert p["frac_above_a_p90"] > 0.5


# ── identification ──────────────────────────────────────────────────────────────────────────────

def test_identification_arithmetic():
    A = VA.identification_arith(0.25, 0.5, 1.0, 1.0)
    assert float(A["f_i"]) == 0.25 and float(A["f_ii"]) == 0.5
    assert float(A["R"]) == 2.0 and float(A["R0"]) == 1.0 and float(A["rho"]) == 2.0
    assert float(A["S"]) == pytest.approx(math.log(0.5) / math.log(0.25))   # = 0.5
    full = VA.identification_arith(0.1, 0.2, 1.0, 2.0)                     # f_ii == f_i
    assert float(full["S"]) == pytest.approx(1.0) and float(full["rho"]) == pytest.approx(1.0)
    none = VA.identification_arith(0.1, 1.0, 1.0, 1.0)                     # nothing transfers
    assert float(none["S"]) == pytest.approx(0.0)
    assert np.isnan(VA.identification_arith(1.0, 1.0, 1.0, 1.0)["S"])     # 0 / 0: undefined


def test_identification_verdict_thresholds():
    assert VA.identification_verdict(0.71, [0.9, 1.0]) == "INDETERMINATE"
    assert VA.identification_verdict(None, [0.9, 1.0]) == "INDETERMINATE"
    assert VA.identification_verdict(0.7, [0.51, 0.9]) == "IDENTIFIES"      # 0.7 is judged
    assert VA.identification_verdict(0.3, [0.1, 0.49]) == "SELECTIVE"
    assert VA.identification_verdict(0.3, [0.4, 0.6]) == "UNDECIDED"
    assert VA.identification_verdict(0.3, [0.5, 0.6]) == "UNDECIDED"        # strict bound
    assert VA.identification_verdict(0.3, None) == "UNDECIDED"


def test_identification_read_end_to_end():
    rng, cl = _bank(64, 4)
    n = len(cl)
    base = rng.lognormal(size=n)
    errs = {"rnd": {"i": 0.1 * base, "ii": 0.1 * base * 1.02, "i0": base, "ii0": base * 1.02},
            "rndv_small": {"i": 0.1 * base, "ii": base, "i0": base, "ii0": base},
            "rndv_slow": {"i": 0.9 * base, "ii": base, "i0": base, "ii0": base}}
    out = VA.identification_read(errs, cl, {"rnd": "arch"}, n_boot=200)
    k = out["keys"]
    assert k["rnd"]["S"] == pytest.approx(1.0) and k["rnd"]["verdict"] == "IDENTIFIES"
    assert k["rndv_small"]["S"] == pytest.approx(0.0) and k["rndv_small"]["verdict"] == "SELECTIVE"
    assert k["rndv_slow"]["verdict"] == "INDETERMINATE"
    assert k["rnd"]["arch"] == "arch" and k["rndv_small"]["arch"] is None
    assert out["holm_S_gt_half"]["rnd"]["reject"] and out["holm_S_lt_half"]["rndv_small"]["reject"]


def test_probe_indices_fixed_distinct_shuffled():
    a, b = VA.probe_indices(10000), VA.probe_indices(10000)
    np.testing.assert_array_equal(a, b)
    assert len(a) == VA.N_PROBE and len(set(a.tolist())) == len(a)
    assert not np.all(np.diff(a) > 0)                            # a shuffled order, not sorted
    small = VA.probe_indices(100)
    assert sorted(small.tolist()) == list(range(100))           # fewer rows than N_PROBE: all of them


def test_chimera_v1_deterministic_and_every_block_is_a_real_block():
    torch = pytest.importorskip("torch")
    from agents.model.ridealong_heads import block_chimera

    edges = (0, 5, 9, 12, 20, 31)
    x = torch.from_numpy(np.random.default_rng(5).normal(size=(37, 31)).astype(np.float32))
    c1, c2 = block_chimera(x, edges), block_chimera(x, edges)
    assert torch.equal(c1, c2)
    donors = VA.chimera_donors(37, edges)
    xn, cn = x.numpy(), c1.numpy()
    for k in range(len(edges) - 1):
        lo, hi = edges[k], edges[k + 1]
        np.testing.assert_array_equal(cn[:, lo:hi], xn[donors[k], lo:hi])
        # and each block is a REAL block of SOME row
        for j in range(37):
            assert (np.abs(xn[:, lo:hi] - cn[j, lo:hi]).sum(1) == 0).any()
    np.testing.assert_array_equal(donors[0], np.arange(37))
    assert (donors[1:] != np.arange(37)[None, :]).all()          # every other block is another row's


def test_context_block_index_and_the_chimera_mask():
    pytest.importorskip("torch")
    from main.ridealong_read.variants_probe import build_probes, context_block_index

    layout = {"parts": {"a": {"start": 0}, "b": {"start": 6}, "context": {"start": 10},
                        "d": {"start": 14}}}
    from agents.model.ridealong_heads import obs_block_edges

    edges = obs_block_edges(layout, 20)
    assert context_block_index(layout, edges) == 2
    rng = np.random.default_rng(6)
    rows = rng.normal(size=(50, 20)).astype(np.float32)
    masks = rng.uniform(size=(50, 11)) < 0.5
    battles = np.repeat(np.arange(10), 5).astype(str)
    p = build_probes(rows, masks, battles, [f"id{i}" for i in range(50)], layout, "test")
    donors = VA.chimera_donors(50, edges)
    np.testing.assert_array_equal(p["masks_ii"], masks[p["idx"]][donors[2]])
    np.testing.assert_array_equal(p["rows_ii"][:, 10:14], rows[p["idx"]][donors[2], 10:14])
    assert p["provenance"]["context_block_index"] == 2 and p["provenance"]["n"] == 50


# ── series, steps, provenance ───────────────────────────────────────────────────────────────────

def test_parse_step():
    assert VA.parse_step("models/r/checkpoints/checkpoint_11877888_steps.zip") == 11877888
    assert VA.parse_step("models/r/final_model.zip") is None


def _ident(s):
    return {"keys": {"rnd": {"m_i": 1.0, "m_ii": 2.0, "R": 2.0, "rho": 1.5, "f_i": 0.5,
                             "f_ii": 0.75, "S": s, "verdict": "UNDECIDED"}}}


def test_series_aggregation_and_cross_checkpoint():
    rng, cl = _bank(30, 5)
    carries = []
    for i, step in enumerate((2000000, 4000000, 6000000)):
        c = VA.Carry(label=f"c{i}", path=f"r/checkpoints/checkpoint_{step}_steps.zip",
                     heads="trained", keys=["rnd"], ident=_ident(0.1 * (i + 1)))
        pt, dist = VA.spread_point_and_dist(rng.lognormal(size=len(cl)), cl, n_boot=50)
        c.spread = {"rnd": (pt["rel_spread"], dist)}
        carries.append(c)
    s = VA.identification_series(carries)
    assert [r["label"] for r in s["rnd"]] == ["c0", "c1", "c2"]
    assert [r["step"] for r in s["rnd"]] == [2000000, 4000000, 6000000]
    assert [r["S"] for r in s["rnd"]] == pytest.approx([0.1, 0.2, 0.3])
    body = VA.cross_checkpoint(carries, None)
    assert body["c_saturation"]["first_vs_last"]["first"] == "c0"
    assert body["c_saturation"]["first_vs_last"]["last"] == "c2"
    assert [p["first"] for p in body["c_saturation"]["consecutive"]] == ["c0", "c1"]
    assert body["d_feat_drift"] == {"skipped": "needs >= 2 checkpoints with feat"}
    assert [c["path"] for c in body["checkpoints"]][0].endswith("checkpoint_2000000_steps.zip")
    assert "skipped" in VA.cross_checkpoint(carries[:1], None)["c_saturation"]


def test_declarations_carry_every_variant_and_base_arch():
    pytest.importorskip("torch")
    from agents.model.ridealong_heads import RND_VARIANTS

    d = VA.declarations(RND_VARIANTS)
    assert list(d) == ["rnd"] + [f"rndv_{n}" for n in RND_VARIANTS]
    assert d["rnd"]["predictor_vs_target"] == ("obs→256→256→64 predictor vs obs→256→64 target, "
                                               "same ReLU-MLP family, predictor one layer deeper")
    assert d["rndv_fast"]["lr_mult"] == 10.0 and d["rndv_decay"]["decay_half_life_updates"] == 10.0
    assert all(set(v) == {"input", "lr_mult", "decay_half_life_updates", "predictor_vs_target", "why"}
               for v in d.values())
