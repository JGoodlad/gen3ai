"""X65-R2's learnability probe (DESCRIPTIVE): the probe's knobs never reach the read panel, a probe arm is never a
student, the meters equal a hand count, the read checkpoint is chosen on ``val`` alone, and (slow) a probe arm and its
control train, dump and report end to end."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from main.x65 import bank as BK
from main.x65 import learnability as LB
from main.x65 import student as ST
from main.x65.student_test import _bank


def test_check_probe_refuses_the_read_panel_and_normalises(tmp_path):
    base = {"name": "a", "beta": 0.1, "epochs": 2, "evals_per_epoch": 2, "train_panels": None, "target_model": None}
    pr = LB.check_probe(base, "heldout72_hpfix", 1e-4)
    assert pr["train_panels"] == ["forum72", "p72"] and pr["target_sha256"] is None
    assert LB.check_probe(dict(base, train_panels=["p72", "forum72"]), "heldout72_hpfix", 1e-4)["train_panels"] == \
        ["forum72", "p72"]
    with pytest.raises(ST.StudentError, match="never the read panel"):
        LB.check_probe(dict(base, train_panels=["heldout72_hpfix"]), "heldout72_hpfix", 1e-4)
    with pytest.raises(ST.StudentError, match="learning rate"):
        LB.check_probe(base, "p72", float("nan"))
    with pytest.raises(ST.StudentError, match="beta"):
        LB.check_probe(dict(base, beta=-1), "p72", 1e-4)
    tm = tmp_path / "t.zip"
    tm.write_bytes(b"x")
    assert LB.check_probe(dict(base, target_model=str(tm)), "p72", 1e-4)["target_sha256"] == BK.sha256_file(tm)


def test_restrict_keeps_only_the_chosen_training_panel(tmp_path):
    bank = _bank(tmp_path)
    z, ix = ST.load_fold(bank, "heldout72_hpfix")
    r = LB.restrict(z, ix, ["forum72"])
    for k, v in r.items():
        assert len(v) and set(z["panel"][v].tolist()) == {BK.U_PANELS.index("forum72")}, k
        assert set(v.tolist()) == {i for i in ix[k].tolist() if z["panel"][i] == BK.U_PANELS.index("forum72")}


def test_pick_refuses_a_probe_arm(tmp_path):
    import torch as th

    for lr in ST.LRS:
        (tmp_path / f"lr{lr:g}").mkdir()
        th.save({"probe": {"name": "x"}, "done": True}, tmp_path / f"lr{lr:g}" / "state.pt")
    with pytest.raises(ST.StudentError, match="probe"):
        ST.pick(tmp_path)


def test_the_meters_equal_a_hand_count():
    pick = np.array([1, 1, 2, 0, 0, 3, 0, 0])
    base = np.array([0, 0, 0, 0, 0, 0, 0, 0])            # rows 0, 1, 2, 5 are departures
    a = np.array([1, 0, 2, 0, 1, 0, 0, 2])               # recovers 0 and 2; leaves base on rows 4 and 7 (both keeps)
    m = LB.meters(a, pick, base)
    assert m["departures"] == 4 and m["D"] == 0.5 and m["K"] == 0.5
    assert m["delta"] == 0.0                              # agrees on rows 0, 2, 3, 6; base on rows 3, 4, 6, 7
    assert m["student_dep_rate"] == 0.5 and m["precision"] == 0.5
    deal = np.array(list("aabbccdd"))
    est, lo, hi = LB.ratio_ci(deal, (a == pick) & (pick != base), pick != base)
    assert est == 0.5 and lo <= est <= hi


def _arm(d: Path, steps_val_read: dict[int, tuple[float, float]]) -> None:
    """A constructed arm: 40 rows a set, half departures; checkpoint ``s`` recovers a share of ``val`` / ``read``."""
    n = 40
    arr = {}
    pick = np.array([1, 0] * (n // 2), np.int8)
    for s in LB.SETS:
        arr.update({f"{s}__pick": pick, f"{s}__base": np.zeros(n, np.int8), f"{s}__init_argmax": np.zeros(n, np.int8),
                    f"{s}__deal": np.array([f"d{i // 4}" for i in range(n)]), f"{s}__key": np.array([f"k{s}{i}" for i in range(n)]),
                    f"{s}__leak_attr": (np.arange(n) % 4 == 0).astype(np.int16), f"{s}__leak_rewr": np.zeros(n, np.int16),
                    f"{s}__q_margin": np.linspace(0, 0.2, n)})
    d.mkdir(parents=True)
    np.savez(d / "rows.npz", **arr)
    (d / "probe.json").write_text(json.dumps({"kind": "trained"}))
    for step, (fv, fr) in steps_val_read.items():
        ev = {}
        for s, f in (("train", 1.0), ("val", fv), ("read", fr)):
            a = np.zeros(n, np.int8)
            a[np.flatnonzero(pick == 1)[: round(f * n / 2)]] = 1
            ev[f"{s}__argmax"], ev[f"{s}__logp"] = a, np.zeros((n, 11), np.float16)
        np.savez(d / f"eval_s{step:06d}.npz", **ev)


def test_the_read_checkpoint_is_chosen_on_val_never_on_read(tmp_path):
    _arm(tmp_path / "arm", {0: (0.0, 0.0), 10: (0.5, 0.1), 20: (0.3, 0.9), 30: (0.5, 0.0)})
    doc = LB.report({"x": tmp_path / "arm"})["arms"]["x"]
    assert doc["at"]["selected_on_val"]["step"] == 10                   # the best val, ties to the earlier
    assert doc["at"]["best_on_read_OPTIMISTIC"]["step"] == 20
    assert doc["at"]["last"]["step"] == 30
    sel = doc["at"]["selected_on_val"]
    assert sel["val"]["meters"]["D"] == 0.5 and sel["read"]["meters"]["D"] == 0.1
    assert sel["read"]["ci"]["D"][0] == pytest.approx(0.1)
    assert sel["val"]["splits"]["attribution_miss"]["n"] == 10
    assert sel["val"]["splits"]["departures_q_margin<0.03"]["n"] == 3        # departures only: rows 0, 2, 4
    assert sel["val"]["splits"]["departures_q_margin>=0.1"]["departures"] == 10
    assert doc["init_path_check"]["read"]["D"] == 0.0 and doc["init_path_check"]["read"]["K"] == 1.0


@pytest.mark.slow
def test_a_probe_arm_and_its_control_train_dump_and_report(tmp_path):
    import torch as th

    from agents.model.compile_parity_fixture import FIXTURE_PATH
    from main.policy_spectrum.reader import load_checkpoint
    from main.x65 import export as EX
    from main.x65.export_test import _fresh_zip

    fx = np.load(FIXTURE_PATH)["obs"].astype(np.float32)
    (tmp_path / "bank").mkdir()
    bank = _bank(tmp_path / "bank", n_deals=48, rows_per=24, obs=fx)
    init = _fresh_zip(tmp_path / "init")
    # the control's target: P*'s zip with its pointer head moved (a fresh network's heads start near uniform)
    sd = {k: v.detach().clone() for k, v in load_checkpoint(init).policy.state_dict().items()}
    g = th.Generator().manual_seed(3)
    for k in sd:
        if k.startswith("pointer_head.") and sd[k].dtype == th.float32:
            sd[k] += th.randn(sd[k].shape, generator=g)
    other = init.parent / "other.zip"                   # beside its model_config.json
    EX.write_zip(init, sd, other)
    sha, bsha = BK.sha256_file(init), BK.sha256_file(bank / "bank_manifest.json")
    pr = {"name": "smoke", "beta": 0.0, "epochs": 2, "evals_per_epoch": 2, "train_panels": ["forum72"],
          "target_model": None}
    out = tmp_path / "cap"
    assert ST.train(bank, bsha, "heldout72_hpfix", 3e-4, init, out, limit_steps=1, init_sha256=sha, probe=pr) == 3
    assert ST.train(bank, bsha, "heldout72_hpfix", 3e-4, init, out, init_sha256=sha, probe=pr) == 0   # resumed
    d = out / "lr0.0003"
    st = th.load(d / "state.pt", weights_only=False)
    assert st["probe"]["train_panels"] == ["forum72"] and st["probe"]["beta"] == 0.0
    assert len(list(d.glob("eval_s*.npz"))) == 1 + 4                   # step 0 and every checkpoint
    with pytest.raises(ST.StudentError, match="another bank"):          # a resume under other knobs is refused
        ST.train(bank, bsha, "heldout72_hpfix", 3e-4, init, out, init_sha256=sha, probe=dict(pr, beta=0.5))
    ctrl = tmp_path / "ctrl"
    pc = dict(pr, train_panels=None, target_model=str(other), beta=0.1)
    assert ST.train(bank, bsha, "heldout72_hpfix", 1e-4, init, ctrl, init_sha256=sha, probe=pc) == 0
    with np.load(ctrl / "lr0.0001" / "rows.npz") as z:
        assert set(z["read__panel"].tolist()) == {BK.U_PANELS.index("heldout72_hpfix")}
        assert not np.array_equal(z["read__pick"], z["read__init_argmax"])   # the control's own picks
    LB.dump(bank, bsha, "heldout72_hpfix", init, tmp_path / "r0")
    doc = LB.report({"cap": d, "ctrl": ctrl / "lr0.0001", "r0": tmp_path / "r0"})
    for n in ("cap", "ctrl"):
        chk = doc["arms"][n]["init_path_check"]
        # P*'s argmax on the trainer's forward is its step-0 dump, row for row
        with np.load(Path(doc["arms"][n]["dir"]) / "eval_s000000.npz") as e, \
                np.load(Path(doc["arms"][n]["dir"]) / "rows.npz") as z:
            for s in LB.SETS:
                assert np.array_equal(e[f"{s}__argmax"], z[f"{s}__init_argmax"]), (n, s)
        assert chk["read"]["n"] == 48 * 24 // 2
    assert doc["arms"]["r0"]["at"]["selected_on_val"]["step"] is None
