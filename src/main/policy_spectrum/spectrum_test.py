"""The rank-mass spectrum math (unit, synthetic)."""

from __future__ import annotations

import numpy as np
import pytest

from main.policy_spectrum import spectrum as S


def _dec(battle, cats, kind="free", phase="midgame", opp="bot"):
    return {"battle": battle, "n_legal": len(cats), "cats": {str(k): v for k, v in cats.items()},
            "kind": kind, "phase": phase, "legal_bucket": "4-6", "opp_class": opp,
            "opp_name": opp, "source": "s", "outcome": "win"}


def test_masked_probs_zero_on_illegal_and_normalised():
    lg = np.array([[1.0, 2.0, 3.0, 50.0]], dtype=np.float32)
    m = np.array([[1, 1, 1, 0]])
    p = S.masked_probs(lg, m)
    assert p[0, 3] == 0.0
    assert p.sum() == pytest.approx(1.0)
    assert p[0, 2] > p[0, 1] > p[0, 0]


def test_spectrum_is_the_sorted_mass_and_entropy():
    probs = np.zeros((2, 11))
    probs[0, [1, 6, 7]] = [0.2, 0.7, 0.1]
    probs[1, [6, 7]] = [0.5, 0.5]
    decs = [_dec("a", {1: "switch", 6: "attack", 7: "status"}), _dec("b", {6: "attack", 7: "setup"})]
    out = S.aggregate(S.Derived(probs, decs), np.arange(2), ci=False)
    assert out["spectrum"][:3] == pytest.approx([(0.7 + 0.5) / 2, (0.2 + 0.5) / 2, 0.1 / 2])
    h0 = -(0.7 * np.log(0.7) + 0.2 * np.log(0.2) + 0.1 * np.log(0.1))
    assert out["entropy"] == pytest.approx((h0 + np.log(2)) / 2, abs=1e-6)
    assert out["entropy_norm"] == pytest.approx((h0 / np.log(3) + 1.0) / 2, abs=1e-6)
    assert out["top1_ge_0.9"] == 0.0


def test_category_block_and_rank_by_category():
    probs = np.zeros((3, 11))
    probs[0, [1, 6, 7]] = [0.2, 0.795, 0.005]      # status legal, starved by mass
    probs[1, [6, 7]] = [0.4, 0.6]                  # status is the top choice
    probs[2, [6, 8]] = [0.9, 0.1]                  # status not legal
    decs = [_dec("a", {1: "switch", 6: "attack", 7: "status"}),
            _dec("b", {6: "attack", 7: "status"}), _dec("c", {6: "attack", 8: "setup"})]
    D = S.Derived(probs, decs)
    blk = S.category_block(D, np.arange(3))
    st = blk["status"]
    assert st["n"] == 2
    assert st["mass"] == pytest.approx((0.005 + 0.6) / 2)
    assert st["share_lt_1pct"] == pytest.approx(0.5)
    assert st["share_top1"] == pytest.approx(0.5)
    assert st["best_rank"] == pytest.approx((3 + 1) / 2)
    assert blk["hazard"] == {"n": 0}
    rc = S.rank_by_category(D, np.arange(3))
    assert rc["rank1"]["attack"]["share"] == pytest.approx(2 / 3)
    assert rc["rank1"]["status"]["mass"] == pytest.approx(0.6)
    assert rc["rank3"]["n"] == 1                   # only one decision has 3 legal actions


def test_bootstrap_is_deterministic_and_clusters_battles():
    rng = np.random.default_rng(0)
    v = rng.random(200)
    cl = np.repeat(np.arange(20), 10)
    a, b = S._boot_ci(v, cl), S._boot_ci(v, cl)
    assert a == b and a[0] < v.mean() < a[1]
    assert S._boot_ci(v, np.zeros(200, dtype=int)) is None     # one cluster: no interval


def test_read_and_paired_delta_on_identical_policies():
    rng = np.random.default_rng(1)
    decs, probs = [], np.zeros((60, 11))
    for i in range(60):
        legal = rng.choice(11, size=4, replace=False)
        cats = {int(k): ("switch" if k < 6 else "attack") for k in legal}
        decs.append(_dec(f"b{i // 6}", cats, kind="free" if i % 5 else "forced_switch"))
        p = rng.random(4)
        probs[i, legal] = p / p.sum()
    r1, r2 = S.read(probs, decs), S.read(probs, decs)
    assert r1 == r2
    assert r1["strata"]["all"]["all"]["n"] == 60
    assert set(r1["strata"]["kind"]) == {"free", "forced_switch"}
    d = S.paired_delta(probs, probs, decs)
    assert d["rank1"]["delta"] == 0.0 and d["entropy"]["ci"] == [0.0, 0.0]
    sharper = probs ** 3
    sharper /= sharper.sum(axis=1, keepdims=True)
    d2 = S.paired_delta(probs, sharper, decs)
    assert d2["rank1"]["delta"] > 0 and d2["entropy"]["delta"] < 0
