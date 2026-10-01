"""Unit tests of the RND input-choice measurement (synthetic data, CPU, seconds)."""

from __future__ import annotations

import numpy as np
import pytest

from main.ridealong_read import rnd_choice as R


def _decisions():
    out = []
    for b in range(8):
        src = "N0@36M" if b < 4 else "K2@90M"
        team = f"t{b % 2}" if b < 6 else "t_new"
        oc = "exploiter" if b == 7 else "bot"
        for n in range(5):
            out.append({"battle": f"b{b}", "source": src, "team": team, "opp_class": oc})
    return out


def test_split_is_battle_level_and_labels_heldout_only():
    sp = R.split(_decisions())
    assert sp["train"].sum() == 20 and sp["heldout"].sum() == 20
    assert not (sp["unseen_team"] & sp["train"]).any() and not (sp["exploiter"] & sp["train"]).any()
    assert sp["unseen_team"].sum() == 10 and sp["exploiter"].sum() == 5


def test_split_refuses_a_battle_straddling_the_split():
    d = _decisions()
    d[0] = dict(d[0], source="K2@90M")                  # battle b0 now has rows on both sides
    with pytest.raises(ValueError, match="straddle"):
        R.split(d)


def test_train_and_read_is_deterministic_and_learns_the_training_rows():
    rng = np.random.default_rng(0)
    x_train = rng.normal(0, 1, (512, 16)).astype(np.float32)
    x_far = rng.normal(4, 1, (256, 16)).astype(np.float32)

    def read(rnd):
        return (R.rnd_errors(rnd, x_train), R.rnd_errors(rnd, x_far))

    a = R.train_and_read(x_train, read, epochs_read=(1, 8), batch=64, lr=1e-3)
    b = R.train_and_read(x_train, read, epochs_read=(1, 8), batch=64, lr=1e-3)
    np.testing.assert_array_equal(a["8"][0], b["8"][0])
    assert a["train_loss"]["8"] < a["train_loss"]["1"]
    tr, far = a["8"]
    assert far.mean() > tr.mean()                   # rows it never saw read as more novel


def test_drift_reads_zero_drift_and_inflation():
    rng = np.random.default_rng(1)
    d = _decisions()
    sp = R.split(d)
    battles = np.array([x["battle"] for x in d])
    e = rng.uniform(0.5, 1.5, len(d))
    same = R.drift_reads(e, e.copy(), sp, battles)
    assert same["heldout"]["inflation_mean"] == 1.0
    assert same["heldout"]["drift_auroc"]["auroc"] == pytest.approx(0.5)
    up = R.drift_reads(e, e * 3, sp, battles)
    assert up["train"]["inflation_mean"] == pytest.approx(3.0)
    assert up["train"]["frac_above_a_p90"] > 0.5


def test_novelty_reads_flags_unseen_rows():
    d = _decisions()
    sp = R.split(d)
    battles = np.array([x["battle"] for x in d])
    err = np.where(sp["unseen_team"], 2.0, 1.0)
    out = R.novelty_reads(err, sp, battles)
    assert out["unseen_team_vs_seen"]["auroc"] == 1.0
    assert out["mean_err"]["heldout_unseen_team"] == 2.0


def test_renormalised_removes_a_pure_rescale():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, (256, 8)).astype(np.float32)
    res = R.train_and_read(x, lambda rnd: None, epochs_read=(2,), batch=64)
    rnd = res["_rnd"]
    shifted = (x * 3.0 + 5.0).astype(np.float32)          # the same states, a rescaled representation
    ea = R.rnd_errors(rnd, x)
    assert R.rnd_errors(rnd, shifted).mean() > 2 * ea.mean()
    np.testing.assert_allclose(R.rnd_errors(R.renormalised(rnd, shifted), shifted), ea, rtol=1e-3, atol=1e-6)
    assert float(rnd.obs_count.item()) == 256.0            # the original is untouched
