"""The belief-only fine-tune: disjoint shards, a panel bank never trains, only the three deltas move, and the export
is the input checkpoint with exactly those tensors replaced."""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pytest
import torch as th

from main.offpool_belief import BANK_SCHEMA, OffPoolBeliefError
from main.offpool_belief import bank as BK
from main.offpool_belief import finetune as FT


# ------------------------------------------------------------------------------------------------------- shards
def test_two_shards_of_one_seed_are_disjoint_and_a_shard_is_a_function_of_its_arguments() -> None:
    a, b = FT.draw(12000, 500, 0, 5), FT.draw(12000, 300, 500, 5)
    assert not set(a.tolist()) & set(b.tolist())
    assert len(set(a.tolist())) == 500
    np.testing.assert_array_equal(a, FT.draw(12000, 500, 0, 5))
    assert not np.array_equal(a, FT.draw(12000, 500, 0, 6))
    with pytest.raises(OffPoolBeliefError):
        FT.draw(100, 60, 50, 5)


def _fake_bank(d: Path, rows: Dict[str, np.ndarray], *, team_ids: Tuple[str, ...], offset: int, schema: str = FT.TRAIN_SCHEMA,
               corpus: str = "c", seed: int = 1) -> Path:
    from utils.hashing import sha256_file

    d.mkdir(parents=True)
    n = len(rows["obs"])
    full = {k: rows.get(k, np.zeros(n, np.int32)) for k in BK.ROW_KEYS}
    np.savez_compressed(d / BK.ROWS_FILE, **full)
    man = {"schema": schema, "corpus": {"corpus": corpus}, "draw": {"seed": seed, "offset": offset},
           "team_ids": list(team_ids), "pilot": {"sha256": "p"}, "n_rows": n, "obs_dim": int(rows["obs"].shape[1]),
           "rows_sha256": sha256_file(d / BK.ROWS_FILE)}
    (d / BK.MANIFEST_FILE).write_text(json.dumps(man))
    return d


def _tiny() -> Dict[str, np.ndarray]:
    return {"obs": np.zeros((3, 4), np.float32), "mask": np.ones((3, 10), bool)}


def test_a_panel_scoring_bank_never_trains(tmp_path: Path) -> None:
    d = _fake_bank(tmp_path / "panel", _tiny(), team_ids=("a",), offset=0, schema=BANK_SCHEMA)
    with pytest.raises(OffPoolBeliefError, match="held-out material"):
        FT.load_trainbank(d)


def test_a_tampered_rows_file_is_refused(tmp_path: Path) -> None:
    d = _fake_bank(tmp_path / "t", _tiny(), team_ids=("a",), offset=0)
    np.savez_compressed(d / BK.ROWS_FILE, **{k: np.ones(3, np.int32) for k in BK.ROW_KEYS})
    with pytest.raises(OffPoolBeliefError, match="does not match"):
        FT.load_trainbank(d)


def test_shards_must_be_disjoint_and_of_one_corpus(tmp_path: Path) -> None:
    a = _fake_bank(tmp_path / "a", _tiny(), team_ids=("t1", "t2"), offset=0)
    b = _fake_bank(tmp_path / "b", _tiny(), team_ids=("t3",), offset=2)
    v = _fake_bank(tmp_path / "v", _tiny(), team_ids=("t4",), offset=3)
    train, val, ident = FT.gather([a, b], v)
    assert len(train["obs"]) == 6 and len(val["obs"]) == 3 and [i["role"] for i in ident] == ["train", "train", "validation"]
    dup = _fake_bank(tmp_path / "dup", _tiny(), team_ids=("t2",), offset=9)
    with pytest.raises(OffPoolBeliefError, match="disjoint"):
        FT.gather([a, b], dup)
    other = _fake_bank(tmp_path / "other", _tiny(), team_ids=("t9",), offset=9, corpus="d")
    with pytest.raises(OffPoolBeliefError, match="another corpus"):
        FT.gather([a], other)


# ----------------------------------------------------------------------------------------------- the loss, the freeze
@pytest.fixture(scope="module")
def learner() -> Tuple[Any, Dict[str, np.ndarray]]:
    """The golden learner with its golden buffer's own rows and labels. A fresh network's deltas are zero, which
    hides every upstream gradient, so the trained tensors that are all zero get a small seeded value first."""
    from agents.training import learner_golden as LG

    th.manual_seed(0)
    model = LG.build_learner(trunk_precision="fp32")
    LG.load_buffer_into(model)
    rb = model.rollout_buffer
    flat = lambda x: np.ascontiguousarray(x.reshape((-1,) + x.shape[2:]))  # noqa: E731
    rows = {k: flat(rb.observations[k]) for k in FT.LABEL_KEYS}
    rows["obs"] = flat(rb.observations["observation"]).astype(np.float32)
    rows["mask"] = rb.action_masks.reshape(-1, rb.action_masks.shape[-1]) > 0.5
    model.policy.set_training_mode(False)
    g = th.Generator().manual_seed(3)
    for name, p in model.policy.features_extractor.named_parameters():
        if name.startswith(FT.TRAINED) and not bool(p.detach().abs().sum() > 0):
            with th.no_grad():
                p.copy_(0.02 * th.randn(p.shape, generator=g))
    return model, rows


def test_only_the_three_deltas_take_gradient_and_every_one_of_them_does(learner: Any) -> None:
    model, rows = learner
    params = FT.trained_parameters(model)
    assert params and all(n.startswith(FT.TRAINED) for n in params)
    grad = {n for n, p in model.policy.named_parameters() if p.requires_grad}
    assert grad == {"features_extractor." + n for n in params}
    loss = FT.belief_loss(model, rows, np.arange(len(rows["obs"])), "cpu")
    assert loss is not None and bool(th.isfinite(loss))
    loss.backward()
    dead = [n for n, p in params.items() if p.grad is None or not bool(p.grad.abs().sum() > 0)]
    assert dead == []
    assert all(sum(n.endswith(b) for n in params) == 1 for b in FT.OUTPUT_BIASES)


def test_the_prior_line_is_the_legs_with_their_output_layers_zeroed_and_it_restores_every_bit(learner: Any) -> None:
    model, rows = learner
    params = FT.trained_parameters(model)
    before = {n: p.detach().clone() for n, p in params.items()}
    live = FT.leg_losses(model, rows, "cpu", 32)
    prior = FT.prior_losses(model, rows, "cpu", 32)
    assert set(live) == set(prior) == {*FT.GATES, "total"}
    assert all(th.equal(p.detach(), before[n]) for n, p in params.items())           # restored bit for bit
    assert live == FT.leg_losses(model, rows, "cpu", 32)
    assert all(abs(live[g] - prior[g]) > 1e-6 for g in FT.GATES)                      # every leg's delta was live
    assert live["total"] == pytest.approx(sum(live[g] for g in FT.GATES))
    with th.no_grad():                                                                # the same read, done by hand
        for n, p in params.items():
            if any(o in n for o in FT.OUTPUT_LAYERS):
                p.zero_()
    assert FT.leg_losses(model, rows, "cpu", 32) == prior


def test_steps_lower_the_loss_and_leave_every_frozen_tensor_bit_identical(learner: Any) -> None:
    model, rows = learner
    params = FT.trained_parameters(model)
    frozen = {n: p.detach().clone() for n, p in model.policy.named_parameters() if not p.requires_grad}
    before = {n: p.detach().clone() for n, p in params.items()}
    sel = np.arange(len(rows["obs"]))
    opt = th.optim.Adam(list(params.values()), lr=1e-3)
    first = FT.mean_loss(model, rows, "cpu", 32)
    for _ in range(6):
        loss = FT.belief_loss(model, rows, sel, "cpu")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    assert FT.mean_loss(model, rows, "cpu", 32) < first - 1e-4
    assert all(th.equal(p, frozen[n]) for n, p in model.policy.named_parameters() if n in frozen)
    assert all(not th.equal(p.detach(), before[n]) for n, p in params.items())


# ---------------------------------------------------------------------------------------------- slow: end to end
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
def test_a_run_trains_picks_on_validation_and_exports_only_the_trained_tensors(learner: Any, tmp_path: Path,
                                                                               monkeypatch: Any) -> None:
    from main.offpool_belief import panels as PN

    monkeypatch.setattr(PN, "refuse_inside_repo", lambda *a, **k: None)
    _model, rows = learner
    init = _fresh_zip(tmp_path / "run")
    n = len(rows["obs"])
    half = {k: v[:n // 2] for k, v in rows.items()}, {k: v[n // 2:] for k, v in rows.items()}
    a = _fake_bank(tmp_path / "a", half[0], team_ids=("t1",), offset=0)
    v = _fake_bank(tmp_path / "v", half[1], team_ids=("t2",), offset=1)
    recipe = dict(FT.RECIPE, epochs=2, batch=16)
    plan = FT.plan(str(init), [a], v, tmp_path / "ft", recipe)
    assert plan["train_rows"] == n // 2 and plan["total_steps"] == 2 * int(np.ceil(n / 2 / 16))
    assert not (tmp_path / "ft").exists()
    early = FT.plan(str(init), [a, tmp_path / "later"], v, tmp_path / "ft", recipe, banks_optional=True)
    assert early["banks"] == "not built" and early["train_rows"] is None and early["trained_parameters"] > 0
    with pytest.raises(OffPoolBeliefError):
        FT.plan(str(init), [a, tmp_path / "later"], v, tmp_path / "ft", recipe)
    with pytest.raises(OffPoolBeliefError, match="named twice"):
        FT.plan(str(init), [a], a, tmp_path / "ft", recipe)
    doc = FT.run(str(init), [a], v, tmp_path / "ft", recipe=recipe, log=lambda _s: None)
    assert [c["epoch"] for c in doc["curve"]] == [0, 1, 2]
    assert all(c["val_loss"] == pytest.approx(sum(c["val_legs"][g] for g in FT.GATES)) for c in doc["curve"])
    assert set(doc["val_prior"]) == {*FT.GATES, "total"}
    assert doc["picked_epoch"] == min(range(3), key=lambda e: (doc["curve"][e]["val_loss"], e))
    out = tmp_path / "ft" / FT.EXPORT_FILE
    assert (tmp_path / "ft" / "model_config.json").is_file() and json.loads((tmp_path / "ft" / FT.RESULT_FILE).read_text())
    with zipfile.ZipFile(init) as zi, zipfile.ZipFile(out) as zo:
        assert zi.namelist() == zo.namelist()
        assert all(zi.read(m) == zo.read(m) for m in zi.namelist() if m != "policy.pth")
        s0 = th.load(io.BytesIO(zi.read("policy.pth")), map_location="cpu", weights_only=True)
        s1 = th.load(io.BytesIO(zo.read("policy.pth")), map_location="cpu", weights_only=True)
    changed = {k for k in s0 if not th.equal(s0[k], s1[k])}
    assert all(any(("features_extractor." + pre) in k for pre in FT.TRAINED) for k in changed)
    assert (doc["picked_epoch"] == 0) == (not changed)
    with pytest.raises(OffPoolBeliefError, match="already holds"):
        FT.run(str(init), [a], v, tmp_path / "ft", recipe=recipe, log=lambda _s: None)


# ------------------------------------------------------------------------------------------ the shard's refusals
def test_a_shard_holding_a_held_out_panel_team_is_refused_before_any_game(tmp_path: Path, monkeypatch: Any) -> None:
    import utils.team_corpora as TC
    from main import heldout_panels as HP
    from main.offpool_belief import panels as PN

    monkeypatch.setattr(PN, "refuse_inside_repo", lambda *a, **k: None)
    clean = TC.load_corpus("n1_proc_v1", "ALL")
    panel_team = HP.panel_team_texts("forum72")[0][1]

    def fake(name: str, tier: str = "") -> Any:
        texts = (panel_team,) + clean.texts[:7]
        return TC.Corpus(name, tier, tuple(f"t{i}" for i in range(8)), texts, "s", "d", "m")
    assert FT.resolve_shard("n1_proc_v1", "ALL", tmp_path / "ok", teams=8, offset=0, seed=1)[1]
    monkeypatch.setattr(TC, "load_corpus", fake)
    with pytest.raises(OffPoolBeliefError, match="hit a held-out panel"):
        FT.resolve_shard("n1_proc_v1", "ALL", tmp_path / "bad", teams=8, offset=0, seed=1)
    monkeypatch.undo()
    with pytest.raises(OffPoolBeliefError, match="corpus"):
        FT.resolve_shard("hr_spec_v1", "", tmp_path / "h", teams=8, offset=0, seed=1)
