"""B3's export (PREREG §9: "bit-identity on the fixture"): a one-ULP difference in any logit is caught; the slow tier
trains a tiny fold end to end on a fresh production network, picks, exports it with this checkout standing in for the
play pin, and checks it bit for bit (and that a perturbed student is caught)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from main.x65 import bank as BK
from main.x65 import export as EX
from main.x65 import student as ST
from main.x65.student_test import _bank


def test_verify_catches_one_ulp_and_a_non_bare_load():
    rng = np.random.default_rng(0)
    tr = {"raw": rng.normal(size=(64, 11)).astype(np.float32), "play": rng.normal(size=(64, 11)).astype(np.float32)}
    pin = {"loader": "bare", "fixture_sha256": "f", "raw": tr["raw"].copy(), "play": tr["play"].copy()}
    assert EX.verify(tr, pin, "f") == []
    bad = dict(pin, raw=pin["raw"].copy())
    bad["raw"][17, 3] = np.nextafter(bad["raw"][17, 3], np.float32(np.inf))
    assert EX.verify(tr, bad, "f") == ["the raw logits differ from the trainer's"]
    assert EX.verify(tr, dict(pin, play=pin["play"].astype(np.float64)), "f")
    assert EX.verify(tr, dict(pin, loader="foreign"), "f")
    assert EX.verify(tr, pin, "g")


def test_the_zip_replaces_only_the_policy(tmp_path):
    import zipfile

    import torch as th

    src = tmp_path / "p.zip"
    with zipfile.ZipFile(src, "w") as z:
        for n in ("data", "policy.pth", "policy.optimizer.pth", "pytorch_variables.pth"):
            z.writestr(n, n.encode() * 3)
    EX.write_zip(src, {"w": th.ones(2)}, tmp_path / "s.zip")
    with zipfile.ZipFile(src) as a, zipfile.ZipFile(tmp_path / "s.zip") as b:
        assert a.namelist() == b.namelist()
        for n in a.namelist():
            assert (a.read(n) == b.read(n)) == (n != "policy.pth")


# ------------------------------------------------------------------------------------------------ slow: end to end
def _fresh_zip(d: Path) -> Path:
    from agents.model.model_version import ModelVersion
    from agents.model.snapshot import save_model_snapshot
    from main.fresh_checkpoint import build_fresh_model

    model, args, pk = build_fresh_model(7)
    d.mkdir(parents=True, exist_ok=True)
    model.save(str(d / "init"))
    version = ModelVersion.from_layout_and_policy_kwargs(
        pk["features_extractor_kwargs"]["layout"], pk, vf_coef=args.vf_coef, move_belief_coef=args.move_belief_coef,
        spread_belief_coef=args.spread_belief_coef, opp_belief_aux_coef=args.opp_belief_aux_coef,
        hp_type_belief_coef=args.hp_type_belief_coef, item_belief_coef=args.item_belief_coef)
    save_model_snapshot(str(d), version, git_hash="test")
    return d / "init.zip"


@pytest.mark.slow
def test_a_fold_trains_picks_and_exports_bit_identically(tmp_path):
    import subprocess

    import torch as th

    from agents.model.compile_parity_fixture import FIXTURE_PATH
    from utils.paths import repo_path

    fx = np.load(FIXTURE_PATH)["obs"].astype(np.float32)
    (tmp_path / "bank").mkdir()
    bank = _bank(tmp_path / "bank", n_deals=48, rows_per=24, obs=fx)
    init = _fresh_zip(tmp_path / "init")
    sha = BK.sha256_file(init)
    bsha = BK.sha256_file(bank / "bank_manifest.json")
    out = tmp_path / "s_f"
    for lr in ST.LRS:
        assert ST.train(bank, bsha, "forum72", lr, init, out, limit_steps=3, init_sha256=sha) == 3   # a chunk
        assert ST.train(bank, bsha, "forum72", lr, init, out, init_sha256=sha) == 0                   # resumed
    st = th.load(out / "lr1e-05" / "state.pt", weights_only=False)
    assert len([h for h in st["hist"] if h["checkpoint"] is not None]) == 16
    heads = [n for n in st["frozen"] + st["trainable"] if ".win_head." in n or ".value_cls" in n]
    assert heads and set(heads) <= set(st["frozen"])            # the value head's own parameters never train
    assert any(n.startswith("pointer_head.") for n in st["trainable"])
    doc = ST.pick(out)
    assert len(doc["candidates"]) == 32 and doc["student"] == "S_F"
    head = subprocess.run(["git", "-C", str(repo_path()), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=False).stdout.strip()
    ex = EX.write(out, init, repo_path(), init_sha256=sha, pin=head)       # this checkout stands in for the pin
    assert ex["bit_identical"] and ex["pin_loader"] == "bare"
    # a student whose weights are not the picked ones (its pointer head moved by 1e-3) is caught
    sd = th.load(out / doc["picked"]["path"], weights_only=True)
    k = next(n for n in sd if n.startswith("pointer_head.") and n.endswith("weight") and sd[n].dtype == th.float32)
    sd[k] = sd[k] + 1e-3
    EX.write_zip(init, sd, out / "model" / "bad.zip")
    tr = EX.trainer_logits(init, th.load(out / doc["picked"]["path"], weights_only=True), EX.fixture())
    pn = EX.pin_logits(out / "model" / "bad.zip", repo_path(), tmp_path / "bad.npz")
    assert EX.verify(tr, pn, EX.fixture()["sha256"])


@pytest.mark.slow
def test_an_x66_fold_trains_on_an_honest_bank_picks_on_val_and_exports_bit_identically(tmp_path):
    import json
    import subprocess

    import torch as th

    from agents.model.compile_parity_fixture import FIXTURE_PATH
    from utils.paths import repo_path

    fx = np.load(FIXTURE_PATH)["obs"].astype(np.float32)
    (tmp_path / "bank").mkdir()
    bank = _bank(tmp_path / "bank", n_deals=48, rows_per=24, obs=fx, source="sprime_honest")
    init = _fresh_zip(tmp_path / "init")
    sha, bsha = BK.sha256_file(init), BK.sha256_file(bank / "bank_manifest.json")
    out = tmp_path / "s_p"
    assert ST.train(bank, bsha, "p72", 1e-4, init, out, limit_steps=3, init_sha256=sha, recipe="x66") == 3
    assert ST.train(bank, bsha, "p72", 1e-4, init, out, init_sha256=sha, recipe="x66") == 0          # resumed
    d = out / "lr0.0001"
    st = th.load(d / "state.pt", weights_only=False)
    assert st["recipe"] == "x66" and st["seed"] == 6613 and st.get("probe") is None
    assert len([h for h in st["hist"] if h["checkpoint"] is not None]) == 24
    assert len(list(d.glob("eval_s*.npz"))) == 25                       # step 0 and every checkpoint, scored
    with np.load(d / "rows.npz") as z:
        assert set(z["read__panel"].tolist()) == {BK.U_PANELS.index("p72")}
    with pytest.raises(ST.StudentError, match="another bank"):          # an x65 resume of an x66 state is refused
        ST.train(bank, bsha, "p72", 1e-4, init, out, init_sha256=sha, probe={
            "name": "x", "beta": 0.1, "epochs": 12, "evals_per_epoch": 2, "train_panels": None,
            "target_model": None})
    doc = ST.pick(out, "x66")
    assert doc["student"] == "S_P" and doc["picked"]["step"] > 0
    head = subprocess.run(["git", "-C", str(repo_path()), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=False).stdout.strip()
    ex = EX.write(out, init, repo_path(), init_sha256=sha, pin=head)       # this checkout stands in for the pin
    assert ex["bit_identical"] and ex["recipe"] == "x66" and Path(ex["zip"]).name == "x66_S_P.zip"
    side = json.loads(Path(ex["zip"]).with_suffix(".json").read_text())
    assert side["x65_student"]["recipe"] == "x66"
