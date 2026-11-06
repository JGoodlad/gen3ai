"""B2's fine-tune: the fold split (PREREG §9: "the inner-validation split never containing a read panel's deal"), the
plan, the loss, the pick, the bank refusal. The end-to-end train → pick → export chain is ``export_test``'s."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from main.x65 import bank as BK
from main.x65 import student as ST

N_ACT = BK.N_ACT


def _deal(panel: int, i: int) -> str:
    return f"{panel:02d}{i:018x}"


def _bank(tmp: Path, n_deals: int = 40, rows_per: int = 30, obs: np.ndarray = None) -> Path:
    """A constructed bank: three panels, ``n_deals`` deals each, every row an anchor row, every other row a label."""
    rng = np.random.default_rng(0)
    shards = []
    for pi, panel in enumerate(BK.U_PANELS):
        n = n_deals * rows_per
        mask = np.zeros((n, N_ACT), np.int8)
        mask[:, :4] = 1
        pp = np.zeros((n, N_ACT))
        pp[:, :4] = rng.dirichlet(np.ones(4), n)
        label = (np.arange(n) % 2 == 0)
        pp[~label] = 0
        deals = np.array([_deal(pi, i // rows_per) for i in range(n)], "U20")
        if obs is None:
            o = rng.normal(size=(n, 8)).astype(np.float32)
        else:
            o = obs[np.arange(n) % len(obs)]
            mask = np.zeros((n, N_ACT), np.int8)
            from agents.model.compile_parity_fixture import FIXTURE_PATH

            fm = np.load(FIXTURE_PATH)["action_mask"]
            mask[:] = fm[np.arange(n) % len(fm)]
            legal = mask.astype(bool)
            pp = np.where(legal, rng.random((n, N_ACT)) + 0.1, 0.0)
            pp = pp / pp.sum(1, keepdims=True)
            pp[~label] = 0
        arr = {"obs": o, "mask": mask, "label": label, "pi_prime": pp,
               "inner_val": np.array([BK.inner_val(str(d)) for d in deals]), "deal": deals,
               "played": np.zeros(n, np.int8), "pi_argmax": np.zeros(n, np.int8), "turn": np.full(n, 12, np.int16),
               "arm": np.ones(n, np.int8), "leak_attr": np.zeros(n, np.int16), "leak_rewr": np.zeros(n, np.int16),
               "leak_resid": np.zeros(n, bool)}
        name = f"pstar_s__{panel}__300"
        np.savez(tmp / f"{name}.npz", **arr)
        shards.append({"name": name, "label": "pstar_s", "panel": panel, "unit_seed": 300, "rows": n,
                       "labels": int(label.sum()), "sha256": BK.sha256_file(tmp / f"{name}.npz")})
    man = {"schema": BK.MANIFEST_SCHEMA, "void": None, "engine": BK.E4, "dirty": False, "shards": shards,
           "rows": int(sum(s["rows"] for s in shards)), "label_rows": int(sum(s["labels"] for s in shards))}
    (tmp / "bank_manifest.json").write_text(json.dumps(man))
    return tmp


@pytest.mark.parametrize("fold", list(ST.FOLDS))
def test_no_fold_ever_trains_or_validates_on_a_read_panel_deal(tmp_path, fold):
    bank = _bank(tmp_path)
    z, ix = ST.load_fold(bank, fold)
    read = set(BK.load_rows(bank, [fold], ("deal",))["deal"].tolist())
    for k, v in ix.items():
        assert len(v) and not (set(z["deal"][v].tolist()) & read), k
        assert BK.U_PANELS.index(fold) not in set(z["panel"][v].tolist()), k
    assert set(z["deal"][ix["val_anchor"]].tolist()) == {d for d in z["deal"].tolist() if BK.inner_val(d)}
    assert not (set(z["deal"][ix["train_anchor"]].tolist()) & set(z["deal"][ix["val_anchor"]].tolist()))


def test_the_split_refuses_a_read_panel_row_or_deal(tmp_path):
    bank = _bank(tmp_path)
    z = BK.load_rows(bank, BK.U_PANELS, ST.FIELDS)            # every panel loaded: the read panel's rows are in
    with pytest.raises(ST.StudentError, match="VOID"):
        ST.split(z, "forum72", [])
    z = BK.load_rows(bank, ST.train_panels("forum72"), ST.FIELDS)
    with pytest.raises(ST.StudentError, match="VOID"):
        ST.split(z, "forum72", [str(z["deal"][0])])           # a training row whose deal is a read-panel deal


def test_the_plan_draws_without_replacement_per_epoch_and_checkpoints_every_quarter():
    per, cks = ST.plan(1000, 3000, 6511)
    assert per == 4 and cks == list(range(1, 17))
    per, cks = ST.plan(58_000, 160_000, 6511)
    assert len(cks) == 16 and cks[-1] == per * ST.EPOCHS and len(set(cks)) == 16
    seen_l, seen_a = [], []
    for s in range(per):
        lp, ap = ST.batch_at(s, per, 58_000, 160_000, 6511)
        assert len(ap) == ST.BATCH_ANCHOR
        seen_l.extend(lp.tolist())
        seen_a.extend(ap.tolist())
    assert sorted(seen_l) == list(range(58_000))               # every label row once in an epoch
    assert len(set(seen_a)) == len(seen_a)                     # anchor rows without replacement within the epoch
    a = ST.batch_at(per + 3, per, 58_000, 160_000, 6511)
    assert np.array_equal(a[0], ST.batch_at(per + 3, per, 58_000, 160_000, 6511)[0])
    assert not np.array_equal(a[0], ST.batch_at(3, per, 58_000, 160_000, 6511)[0])   # a fresh permutation an epoch


def test_the_warm_up_then_constant_learning_rate():
    assert ST.lr_at(0, 1e-5) == pytest.approx(1e-5 / 200)
    assert ST.lr_at(199, 3e-5) == pytest.approx(3e-5)
    assert ST.lr_at(5000, 3e-5) == 3e-5


def test_the_kl_is_over_the_legal_actions_and_zero_at_the_target():
    import torch as th

    m = th.tensor([[1, 1, 0, 1.0]])
    t = th.tensor([[0.2, 0.5, 0.0, 0.3]])
    lp = ST.masked_log_softmax_t(th.log(th.tensor([[0.2, 0.5, 0.7, 0.3]])), m)
    assert float(ST.kl_t(t, lp, m)) == pytest.approx(0.0, abs=1e-6)
    lp2 = ST.masked_log_softmax_t(th.zeros(1, 4), m)
    want = sum(p * np.log(p / (1 / 3)) for p in (0.2, 0.5, 0.3))
    assert float(ST.kl_t(t, lp2, m)) == pytest.approx(want, rel=1e-5)


def test_the_pick_is_the_lowest_l_with_ties_to_the_lower_lr_then_the_earlier_checkpoint():
    rows = [{"lr": 3e-5, "checkpoint": 2, "L": 0.5}, {"lr": 1e-5, "checkpoint": 9, "L": 0.5},
            {"lr": 1e-5, "checkpoint": 4, "L": 0.5}, {"lr": 1e-5, "checkpoint": 1, "L": 0.6}]
    assert ST.choose(rows) == {"lr": 1e-5, "checkpoint": 4, "L": 0.5}
    assert ST.choose(rows[:1] + [{"lr": 1e-5, "checkpoint": 0, "L": 0.50001}])["lr"] == 3e-5


def test_the_trainer_refuses_an_unregistered_bank(tmp_path):
    bank = _bank(tmp_path)
    with pytest.raises(BK.BankError, match="not the registered"):
        ST.train(bank, "0" * 64, "forum72", 1e-5, tmp_path / "nope.zip", tmp_path / "out")
    with pytest.raises(ST.StudentError, match="registered"):
        ST.train(bank, BK.sha256_file(bank / "bank_manifest.json"), "forum72", 2e-5, tmp_path / "x.zip",
                 tmp_path / "out")
