"""The weighted battle-clustered bootstrap equals the replicated computation (unit, synthetic)."""

from __future__ import annotations

import numpy as np
import pytest

from agents.training.instrumented_ppo.ridealong_terms import error_by_decile, rank_auroc, spearman
from main.ridealong_read import boot as BT


def _data(n=400, seed=0):
    rng = np.random.default_rng(seed)
    score = np.round(rng.normal(0, 1, n), 1)                 # rounded: exercises TIES
    err = np.abs(score * 0.3 + rng.normal(0, 1, n))
    return score, err, err > 0.8


def test_unit_weights_equal_the_plain_statistics():
    score, err, lab = _data()
    W = np.ones((1, len(score)))
    r = BT.SortedMidranks(score)(W)
    assert BT._auroc_w(r, lab[None, :], W)[0] == pytest.approx(rank_auroc(score, lab))
    re_ = BT.SortedMidranks(err)(W)
    assert BT._pearson_w(r, re_, W)[0] == pytest.approx(spearman(score, err))
    dec = error_by_decile(score, err)
    ratio = BT._decile_ratio_w(np.argsort(score, kind="stable"), err, W)[0]
    assert ratio == pytest.approx(dec[-1] / dec[0])            # N divisible by 10: exact


def test_integer_weights_equal_explicit_replication():
    score, err, lab = _data(seed=1)
    w = np.random.default_rng(2).integers(0, 4, len(score)).astype(float)
    W = w[None, :]
    idx = np.repeat(np.arange(len(score)), w.astype(int))
    r = BT.SortedMidranks(score)(W)
    assert BT._auroc_w(r, lab[None, :], W)[0] == pytest.approx(rank_auroc(score[idx], lab[idx]))
    re_ = BT.SortedMidranks(err)(W)
    assert BT._pearson_w(r, re_, W)[0] == pytest.approx(spearman(score[idx], err[idx]))


def test_boot_is_deterministic_and_brackets_the_point():
    score, err, lab = _data(n=600, seed=3)
    cl = np.arange(600) // 6
    a = BT.boot_rank_stats(score, cl, label=lab, err=err, n_boot=200)
    b = BT.boot_rank_stats(score, cl, label=lab, err=err, n_boot=200)
    np.testing.assert_array_equal(a["auroc"], b["auroc"])
    lo, hi = BT.ci(a["auroc"])
    assert lo <= rank_auroc(score, lab) <= hi
    lo, hi = BT.ci(a["spearman"])
    assert lo <= spearman(score, err) <= hi
    assert len(a["decile_ratio"]) == 200


def test_weight_chunks_are_battle_draw_counts():
    cl = np.array(["a", "a", "b", "c"])
    Ws = list(BT.weight_chunks(cl, n_boot=7, chunk=3))
    W = np.concatenate(Ws)
    assert W.shape == (7, 4) and (W[:, 0] == W[:, 1]).all()
    assert (W[:, [0, 2, 3]].sum(1) == 3).all()               # 3 battles drawn per resample


def test_ci_refuses_a_mostly_degenerate_distribution():
    assert BT.ci(np.array([np.nan] * 8 + [0.1, 0.2])) is None
    assert BT.ci(np.array([0.5] * 10)) == [0.5, 0.5]
