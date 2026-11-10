"""The human-replay clone (``main.human_data.clone``): the plateau schedule and its stop, the layout guard, the
battle-disjoint held-out split, the held-out read, and one tiny train → export → register → ``init`` / ``opp`` on
synthetic rows.

TIERS: the pure rules and one single-step train are routine; the chain through the project's save path, the registry
and the trainer's two consumers builds the production policy five times, so it is `slow`."""
from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from main.human_data import clone as CL
from main.human_data import rowset as RS
from main.human_data import student as ST
from main.human_data.student_test import _z

CODE = "c" * 64


def _rowset(d: Path, train_bkeys, val_bkeys, layout: "str | None" = CODE) -> Path:
    """A rowset whose train shard and validation shard carry the given battle keys, one row each; zero rows."""
    from agents.battle.core_obs import obs_dim

    d.mkdir(parents=True)
    z = _z(6)

    def arr(bkeys):
        k = len(bkeys)
        idx = np.arange(k) % 6
        return {"rows": np.zeros((k, obs_dim()), np.float32), "mask": z["mask"][idx], "act": z["act"][idx],
                "hidden": z["hidden"][idx], "cons": z["cons"][idx], "moveset": z["moveset"][idx],
                "rid": np.arange(k, dtype=np.int64), "bkey": np.asarray(bkeys, np.int64),
                "acct": np.zeros(k, np.int64), "seat": np.zeros(k, np.int8), "k": np.zeros(k, np.int16),
                "turn": np.zeros(k, np.int16), "forced": np.zeros(k, bool), "rating": np.zeros(k, np.int16),
                "quit_dist": np.zeros(k, np.int16), "end": np.zeros(k, np.int8)}
    tr = RS._write_shard(d, "train_0000", arr(train_bkeys))
    va = RS._write_shard(d, "val", arr(val_bkeys))
    man = {"rowset": "t.r1", "train_shards": [tr], "val": va, "train_rows": len(train_bkeys),
           "val_rows": len(val_bkeys)}
    if layout is not None:
        man["layout_hash"] = layout
    (d / "rows_manifest.json").write_text(json.dumps(man))
    return d


# ── the schedule ───────────────────────────────────────────────────────────────────────────────────────────────

def test_the_registered_recipe_is_the_default_and_any_override_is_a_smoke():
    r = CL.REGISTERED
    assert (r.max_epochs, r.evals_per_epoch, r.patience, r.min_delta, r.warmup) == (40, 4, 4, 1e-3, 1000)
    assert r.lr_ladder == (3e-4, 1.5e-4, 7.5e-5, 3.75e-5) and not r.smoke and not r.as_json()["smoke"]
    for f in dataclasses.fields(CL.Recipe):
        if f.name != "lr_ladder":
            assert dataclasses.replace(r, **{f.name: 3}).smoke, f.name
    assert CL.default_id("hr_spec_v1.r2", 5901, False) == "hclone_hr_spec_v1_r2_s5901"
    assert CL.default_id("hr_spec_v1.r2", 5901, True).endswith("_s5901_smoke")


def test_the_plateau_schedule_steps_the_ladder_and_stops_on_the_last_rung():
    """THE STOP RULE. An improvement is at least ``min_delta`` under the reference; ``patience`` reads without one
    step the ladder; a plateau on the last rung is CONVERGED. Fails on revert of any branch of ``advance``."""
    r = dataclasses.replace(CL.REGISTERED, patience=2, lr_ladder=(4e-4, 2e-4))
    s = {"rung": 0, "bad": 0, "ref": None}
    assert CL.advance(s, 1.50, r) is None and s == {"rung": 0, "bad": 0, "ref": 1.50}
    assert CL.advance(s, 1.40, r) is None and s["ref"] == 1.40 and s["bad"] == 0          # an improvement
    assert CL.advance(s, 1.3995, r) is None and s["ref"] == 1.40 and s["bad"] == 1        # under min_delta: not one
    assert CL.advance(s, 1.3980, r) is None and s["ref"] == 1.3980 and s["bad"] == 0      # the count resets
    assert CL.advance(s, 1.3985, r) is None and s["bad"] == 1
    assert CL.advance(s, 1.3990, r) is None and s == {"rung": 1, "bad": 0, "ref": 1.3980}  # the ladder steps
    assert CL.advance(s, 1.41, r) is None and s["bad"] == 1
    assert CL.advance(s, 1.3975, r) == "CONVERGED" and s["rung"] == 1                     # the last rung plateaued


def test_the_learning_rate_warms_up_then_sits_on_the_ladder():
    r = CL.REGISTERED
    assert CL.lr_at(0, 0, r) == pytest.approx(3e-4 / 1000) and CL.lr_at(999, 0, r) == pytest.approx(3e-4)
    assert CL.lr_at(50_000, 0, r) == pytest.approx(3e-4) and CL.lr_at(50_000, 3, r) == pytest.approx(3.75e-5)


# ── the data guards ────────────────────────────────────────────────────────────────────────────────────────────

def test_rows_encoded_under_another_layout_are_refused_unless_the_hash_is_named():
    """THE LAYOUT GUARD. Fails on revert (a clone would train on rows the live encoder never produces)."""
    rows = "f" * 64
    assert CL.check_layout({"layout_hash": CODE}, None, CODE) == {
        "rowset_layout_hash": CODE, "code_layout_hash": CODE, "match": True, "waived": False}
    with pytest.raises(CL.CloneError, match="--accept-layout ffffffffffff"):
        CL.check_layout({"layout_hash": rows}, None, CODE)
    for bad in ("f" * 11, "c" * 12, "e" * 64):                # too short; the code's hash; another hash
        with pytest.raises(CL.CloneError, match="encoded under layout"):
            CL.check_layout({"layout_hash": rows}, bad, CODE)
    got = CL.check_layout({"layout_hash": rows}, "f" * 12, CODE)
    assert got["waived"] and not got["match"] and got["accepted_prefix"] == "f" * 12
    with pytest.raises(CL.CloneError, match="no layout_hash"):
        CL.check_layout({}, "f" * 12, CODE)


def test_the_code_layout_hash_is_the_one_a_rows_manifest_records(monkeypatch):
    """Fails if ``code_layout_hash`` drifts from ``rowset._code_identity``'s (the guard would refuse fresh rows)."""
    from main.policy_spectrum import bank
    from utils.bridge import sim_bridge_bin

    monkeypatch.setattr(sim_bridge_bin, "resolve_live_reader_bin", lambda: __file__)
    assert RS._code_identity()["layout_hash"] == CL.code_layout_hash()
    assert len(CL.code_layout_hash()) == 64 and bank.encoder_identity()


def test_the_split_drops_every_training_row_of_a_validation_battle(tmp_path):
    """THE BATTLE-DISJOINT SPLIT (X59-S REVIEW G: the rowset validates by account, so a validation battle's other
    seat trains). Fails on revert of the ``isin`` drop, and of the STOP / REPORT halving by battle."""
    rows = _rowset(tmp_path / "rows", train_bkeys=[10, 11, 12, 10, 13, 12, 14], val_bkeys=[12, 10, 30, 12, 31, 10])
    man, sha = RS.load_manifest(rows)
    cache = tmp_path / "split.json"
    sp = CL.split_plan(rows, man, sha, CL.REGISTERED, cache)
    assert sp["kept"] == [3] and sp["train_rows"] == 3 and sp["dropped_rows"] == 4 and sp["val_battles"] == 4
    assert (sp["stop_battles"], sp["report_battles"]) == (2, 2)
    assert json.loads(cache.read_text()) == sp and CL.split_plan(rows, man, sha, CL.REGISTERED, cache) == sp
    vb = np.array([10, 12, 30, 31])
    kept = CL._train_shard(rows, man["train_shards"][0], vb)
    assert len(kept["target"]) == 3 and kept["rows"].shape[0] == 3
    S, R, got_vb = CL.held_out(rows, man, CL.REGISTERED)
    assert got_vb.tolist() == [10, 12, 30, 31]
    assert len(S["target"]) == 3 and len(R["target"]) == 3          # {10, 30} and {12, 31}: 2 + 1 and 2 + 1 rows
    cut = dataclasses.replace(CL.REGISTERED, held_out_rows=1)
    assert [len(h["target"]) for h in CL.held_out(rows, man, cut)[:2]] == [1, 1]
    assert CL.split_plan(rows, man, sha, cut, cache)["held_out_rows"] == 1   # a stale cache is not reused


def test_a_split_that_leaves_nothing_is_refused(tmp_path):
    rows = _rowset(tmp_path / "rows", train_bkeys=[10, 10], val_bkeys=[10, 11])
    man, sha = RS.load_manifest(rows)
    with pytest.raises(CL.CloneError, match="nothing to train on"):
        CL.split_plan(rows, man, sha, CL.REGISTERED)
    one = _rowset(tmp_path / "one", train_bkeys=[1], val_bkeys=[10, 10])
    with pytest.raises(CL.CloneError, match="fewer than two battles"):
        CL.held_out(one, RS.load_manifest(one)[0], CL.REGISTERED)


def test_the_held_out_read_is_the_set_nll_and_top1_on_single_label_rows(monkeypatch):
    """Top-1 counts only rows whose target is ONE action; a hidden-move row (a set) is in the NLL alone. Fails on
    revert of the ``single`` restriction (a set row would count as a hit whenever the argmax is in the set)."""
    import torch as th

    from main.x50_r1 import hom

    z = _z(6)
    V = {"rows": np.zeros((6, 4), np.float32), "mask": z["mask"], "target": ST.targets("human", z, None)}
    lg = th.full((6, 11), -5.0)
    lg[:, 6] = 5.0                                   # the policy always picks action 6
    monkeypatch.setattr(hom, "policy_logits_t", lambda pol, x, m: lg[:len(x)])
    r = CL.read_held_out(None, V, "cpu", batch=4)
    # act = [6, 0, hidden, 7, 8, hidden]: four single-label rows, one hit; the two hidden rows' sets hold 6
    assert r["rows"] == 6 and r["top1_rows"] == 4 and r["top1"] == 0.25
    want = ST.set_nll_t(lg, th.from_numpy(z["mask"]).float(), th.from_numpy(V["target"]).float()).double().mean()
    assert r["nll"] == pytest.approx(float(want), abs=1e-12) and r["uniform_nll"] == ST.uniform_nll(V)


# ── the trainer ────────────────────────────────────────────────────────────────────────────────────────────────

def _tiny(tmp_path, monkeypatch, n=40):
    monkeypatch.setattr(CL, "code_layout_hash", lambda: CODE)
    rows = _rowset(tmp_path / "rows", train_bkeys=list(range(100, 100 + n)) + [10, 12],
                   val_bkeys=[10, 12, 30, 31, 10, 30])
    out = tmp_path / "clone"
    out.mkdir()
    (out / "rows_path.txt").write_text(str(rows) + "\n")
    return rows, out


SMOKE = dataclasses.replace(CL.REGISTERED, max_epochs=2, evals_per_epoch=1, patience=1, warmup=1)


def test_a_tiny_train_stops_on_the_recipe_and_keeps_the_best_weights(tmp_path, monkeypatch, restore_torch_globals):
    """ROUTINE half of the chain below (one build): two one-step epochs run to the cap, so the stop is BUDGET
    (recorded as not converged) and the state carries the reads, the best weights and the split."""
    import torch as th

    rows, out = _tiny(tmp_path, monkeypatch)
    assert CL.train(rows, 5901, out, SMOKE, threads=1) == 0
    st = th.load(out / "state.pt", weights_only=False)
    assert st["done"] and st["step"] == 2 and st["recipe"]["smoke"] and st["arm"] == "human"
    assert st["stop"]["reason"] in ("BUDGET", "CONVERGED") and st["stop"]["converged"] == (
        st["stop"]["reason"] == "CONVERGED")
    assert st["hist_steps"][0] == 0 and len(st["hist"]) == len(st["top1_hist"]) == len(st["lr_hist"]) >= 2
    assert st["best"]["nll"] == min(st["hist"]) and st["best"]["step"] in st["hist_steps"]
    assert json.loads((out / "split.json").read_text())["dropped_rows"] == 2


def test_train_refuses_rows_of_another_layout_before_building_anything(tmp_path, monkeypatch):
    rows, out = _tiny(tmp_path, monkeypatch)
    monkeypatch.setattr(CL, "code_layout_hash", lambda: "d" * 64)
    with pytest.raises(CL.CloneError, match="encoded under layout"):
        CL.train(rows, 5901, out, SMOKE, threads=1)
    assert not (out / "state.pt").exists()
    assert CL.main(["train", "--rows", str(rows), "--seed", "1", "--out", str(out)]) == 1


@pytest.mark.slow
def test_train_export_register_then_init_and_opp_on_synthetic_rows(tmp_path, monkeypatch, restore_torch_globals):
    """THE CHAIN, end to end on CPU: a tiny clone is trained, exported through the project's save path, registered
    (as a smoke model, in a temp store and registry), and consumed by both levers: ``init`` copies exactly its trained
    tensors into a fresh learner of ANOTHER seed (the rest keeps that seed's init), and ``opp`` hands its zip to the
    stable-opponent resolver, which loads it as a frozen opponent of the run's own architecture."""
    import torch as th

    from agents.model.model_version import ModelVersion
    from agents.model.snapshot import load_foreign_opponent
    from agents.training.fixed_opponent_pool import resolve_stable_opponents
    from main.fresh_checkpoint import build_fresh_model
    from main.policy_spectrum.reader import inference_globals
    from main.train import human_data_setup as HS
    from utils import human_models as HM

    rows, out = _tiny(tmp_path, monkeypatch)
    monkeypatch.setenv("GEN3AI_ARCHIVE_DIR", str(tmp_path / "archive"))
    reg = tmp_path / "registry.json"
    reg.write_text(json.dumps({"models": {}}))
    monkeypatch.setattr(HM, "registry_path", lambda: reg)
    with pytest.raises(CL.CloneError, match="has not stopped"):
        th.save({"schema": "gen3_human_clone_state_v1", "done": False}, out / "state.pt")
        CL.export(out)
    (out / "state.pt").unlink()
    assert CL.train(rows, 5901, out, SMOKE, threads=1) == 0
    done = th.load(out / "state.pt", weights_only=False)
    assert CL.train(rows, 5901, out, SMOKE, threads=1) == 0                     # done: nothing retrained
    assert th.load(out / "state.pt", weights_only=False)["train_s"] == done["train_s"]
    with pytest.raises(CL.CloneError, match="other rows / seed / init / recipe"):
        CL.train(rows, 5901, out, dataclasses.replace(SMOKE, patience=2), threads=1)
    with pytest.raises(CL.CloneError, match="VOID: the clone learned nothing"):
        CL.export(out)                                 # two steps at 3e-4 are under the learned-nothing margin
    (out / "state.pt").unlink()
    (out / "split.json").unlink()
    fast = dataclasses.replace(SMOKE, max_epochs=8, lr_ladder=(1e-2, 5e-3))
    assert CL.train(rows, 5901, out, fast, threads=1) == 0
    man = CL.export(out)
    mid = "hclone_t_r1_s5901_smoke"
    assert man["id"] == mid and man["smoke"] and man["human_data"] is True and man["rowset"] == "t.r1"
    assert set(man["held_out"]) == {"stop", "report"} and man["held_out"]["report"]["rows"] == 2
    assert 0 < len(man["trained_keys"]) < man["n_tensors"]            # the critic's tensors got no gradient
    got = CL.register(out)
    assert got["status"] == "smoke" and HM.stub(mid)["rowset"] == "t.r1"
    with pytest.raises(CL.CloneError, match="never overwritten"):
        CL.register(out)
    with pytest.raises(HM.HumanModelError, match="SMOKE model"):
        HM.load(mid)
    hm = HM.load(mid, debug=True)
    clone_sd = HM.read_policy_state(hm["zip_path"])

    def args(clause):
        return SimpleNamespace(human_data=clause, model=None, team_mix=None, debug=True, self_play=True,
                               stable_opponents=None, stable_opponent_selfplay_share=0.2)

    with inference_globals(1, "cpu"):
        model, fargs, pk = build_fresh_model(5902)
        version = ModelVersion.from_layout_and_policy_kwargs(
            pk["features_extractor_kwargs"]["layout"], pk, vf_coef=fargs.vf_coef,
            move_belief_coef=fargs.move_belief_coef, spread_belief_coef=fargs.spread_belief_coef,
            opp_belief_aux_coef=fargs.opp_belief_aux_coef, hp_type_belief_coef=fargs.hp_type_belief_coef,
            item_belief_coef=fargs.item_belief_coef)
        before = {k: v.clone() for k, v in model.policy.state_dict().items()}
        a = args(f"init:{mid}@1")
        HS.resolve(a, emit=lambda _l: None)
        applied = HS.apply_init(model, a, version, emit=lambda _l: None)
        after = model.policy.state_dict()
        trained = set(man["trained_keys"])
        assert applied["tensors"] == len(trained) and applied["of_tensors"] == len(after)
        for k, v in after.items():
            assert th.equal(v, clone_sd[k] if k in trained else before[k]), k
        assert any(not th.equal(after[k], before[k]) for k in trained)
        o = args(f"opp:{mid}@0.25")
        HS.resolve(o, emit=lambda _l: None)
        assert o.stable_opponents == f"{hm['zip_path']}:{mid}" and o.stable_opponent_selfplay_share == 0.25
        with pytest.raises(HM.HumanModelError):
            resolve_stable_opponents(o.stable_opponents, version)                 # --exploiter's call: no allowance
        entries = resolve_stable_opponents(o.stable_opponents, version, allow_human_model=HS.allowed_stable_zip(o))
        assert [e.label for e in entries] == [f"ext_{mid}"] and entries[0].zip_path == hm["zip_path"]
        load_foreign_opponent(entries[0].zip_path, current_version=version, device="cpu",
                              config_path=entries[0].config_path)
