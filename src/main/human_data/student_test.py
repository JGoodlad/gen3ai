"""X59-S's label sources and students (``teacher.py``, ``student.py``): matched noise, restricted sampling, the set
loss, the arms' targets and their VOID guards, and one tiny train → export → reload on synthetic rows."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from main.human_data import rowset as RS
from main.human_data import student as ST
from main.human_data import teacher as TE


def test_matched_uniform_is_a_pure_function_of_the_rowset_and_row():
    rid = np.array([3, 1, 2], np.int64)
    a = TE.matched_uniform("abc", rid)
    assert np.array_equal(a, TE.matched_uniform("abc", rid))
    assert np.array_equal(a[[1, 2, 0]], TE.matched_uniform("abc", rid[[1, 2, 0]]))   # order-free
    assert not np.array_equal(a, TE.matched_uniform("abd", rid))
    assert ((a >= 0) & (a < 1)).all()


def test_restricted_sampling_never_leaves_the_allowed_set():
    rng = np.random.default_rng(0)
    lg = rng.normal(size=(2000, 11)).astype(np.float32)
    allowed = (rng.random((2000, 11)) < 0.4).astype(np.int8)
    allowed[np.arange(2000), rng.integers(0, 11, 2000)] = 1
    p = TE.masked_probs(lg, allowed)
    assert np.allclose(p.sum(1), 1.0) and (p[allowed == 0] == 0).all()
    u = np.concatenate([rng.random(1998), [0.0, 1.0 - 1e-17]])
    a = TE.inverse_cdf(p, u)
    assert (allowed[np.arange(2000), a] == 1).all()
    # the empirical frequencies follow the renormalised policy
    one = np.tile(p[:1], (20000, 1))
    f = np.bincount(TE.inverse_cdf(one, rng.random(20000)), minlength=11) / 20000
    assert np.abs(f - p[0]).max() < 0.02


def _z(n=6):
    mask = np.zeros((n, 11), np.int8)
    mask[:, [0, 1, 6, 7, 8]] = 1
    cons = mask.copy()
    cons[:, 1] = 0
    ms = np.zeros((n, 11), np.int8)
    ms[:, [6, 7, 8]] = 1
    act = np.array([6, 0, -1, 7, 8, -1], np.int8)[:n]
    hidden = act < 0
    return {"mask": mask, "cons": cons, "moveset": ms, "act": act, "hidden": hidden}


def test_targets_per_arm_and_the_void_guards():
    z = _z()
    t = ST.targets("human", z, None)
    assert t[0].tolist() == [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0] and (t[2] == z["moveset"][2]).all()
    lab = np.array([7, 0, -1, 6, 6, -1], np.int8)
    tp = ST.targets("pstar", z, lab)
    assert tp[0, 7] == 1 and (tp[2] == t[2]).all() and (tp[5] == t[5]).all()   # hidden rows: the SAME set label
    bad = lab.copy()
    bad[1] = 1                                         # outside the consistent set: the log rules it out
    with pytest.raises(ST.StudentError, match="VOID"):
        ST.targets("pstar", z, bad)
    bad = lab.copy()
    bad[2] = 6                                         # a P* label on a hidden row: the arms would differ there
    with pytest.raises(ST.StudentError, match="hidden"):
        ST.targets("pstar", z, bad)
    with pytest.raises(ST.StudentError):
        ST.targets("pstar", z, None)


def test_the_set_loss_is_the_cross_entropy_on_a_single_label():
    import torch as th

    lg = th.randn(5, 11)
    legal = th.ones(5, 11)
    legal[:, 10] = 0
    a = th.tensor([0, 3, 6, 7, 9])
    t = th.nn.functional.one_hot(a, 11).float()
    ce = -th.log_softmax(lg.masked_fill(legal == 0, float("-inf")), -1).gather(1, a.view(-1, 1)).squeeze(1)
    assert th.allclose(ST.set_nll_t(lg, legal, t), ce, atol=1e-6)
    s = th.zeros(5, 11)
    s[:, :10] = 1                                      # the whole legal set: zero loss
    assert th.allclose(ST.set_nll_t(lg, legal, s), th.zeros(5), atol=1e-6)


def test_the_plan_is_keyed_by_the_seed_alone():
    a = ST.plan([16384] * 12 + [700], 5901, 3)
    assert a == ST.plan([16384] * 12 + [700], 5901, 3) and a != ST.plan([16384] * 12 + [700], 5902, 3)
    assert sum(nb for *_, nb in a) == 3 * sum(-(-n // ST.BATCH) for n in ([16384 * 8, 16384 * 4 + 700]))


def test_lr_warms_up_then_decays():
    assert ST.lr_at(0, 10000) < ST.lr_at(ST.WARMUP - 1, 10000) == pytest.approx(ST.LR_MAX)
    assert ST.lr_at(10000, 10000) == pytest.approx(ST.LR_MIN)


def _tiny_rowset(d: Path, n: int, seed: int) -> None:
    from agents.battle.core_obs import obs_dim

    rng = np.random.default_rng(seed)
    d.mkdir(parents=True)
    z = _z(6)

    def arr(k):
        idx = rng.integers(0, 6, k)
        return {"rows": np.zeros((k, obs_dim()), np.float32), "mask": z["mask"][idx],
                "act": z["act"][idx], "hidden": z["hidden"][idx], "cons": z["cons"][idx], "moveset": z["moveset"][idx],
                "rid": rng.integers(0, 2 ** 62, k), "bkey": np.zeros(k, np.int64), "acct": np.zeros(k, np.int64),
                "seat": np.zeros(k, np.int8), "k": np.zeros(k, np.int16), "turn": np.zeros(k, np.int16),
                "forced": np.zeros(k, bool), "rating": np.zeros(k, np.int16), "quit_dist": np.zeros(k, np.int16),
                "end": np.zeros(k, np.int8)}
    tr = RS._write_shard(d, "train_0000", arr(n))
    va = RS._write_shard(d, "val", arr(64))
    (d / "rows_manifest.json").write_text(json.dumps({"rowset": "t.r1", "train_shards": [tr], "val": va,
                                                      "train_rows": n, "val_rows": 64}))


def test_train_export_reload_on_synthetic_rows(tmp_path, restore_torch_globals):
    rows = tmp_path / "rows"
    _tiny_rowset(rows, 300, 1)
    out = tmp_path / "s_h"
    out.mkdir()
    (out / "rows_path.txt").write_text(str(rows) + "\n")
    assert ST.train("human", rows, None, 5901, out, epochs=1, budget_min=0, threads=1) == 0
    side = ST.export(out, {"own": (rows, "human")})
    assert side["steps"] == 1 and side["anchors_bare_load_bit_equal"] and np.isfinite(side["yardsticks"]["own"])
    other = tmp_path / "s_s"
    other.mkdir()
    (other / "rows_path.txt").write_text(str(rows) + "\n")
    assert ST.train("selfplay", rows, None, 5901, other, epochs=1, budget_min=0, threads=1) == 0
    import torch as th

    a = th.load(out / "state.pt", weights_only=False)
    b = th.load(other / "state.pt", weights_only=False)
    assert a["init_sha256"] == b["init_sha256"]        # one seed, one initial network, in every arm
    with pytest.raises(ST.StudentError, match="another arm"):
        ST.train("human", rows, None, 5901, other, epochs=1, budget_min=0, threads=1)
