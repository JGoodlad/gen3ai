"""The readers end to end on a slice of the committed Lane S bank: fresh `blob` and `fixed_mass`
checkpoints saved, loaded back through THE strict loader, read on CPU (integration: the Rust core
re-encodes the bank; `models/` is never written)."""
from __future__ import annotations

import json
import math

import pytest

pytestmark = [pytest.mark.sim, pytest.mark.integration]


def _save(dst, arm: str, seed: int, perturb: bool):
    from agents.model.parity_probe import PERTURB_SCALE, perturb_
    from agents.model.snapshot import arch_toggles_from_model, current_model_version
    from agents.observation.state_encoder import load_mappings
    from agents.training.rust_eval import parity as PAR
    from main.fresh_checkpoint import build_fresh_model
    from main.train.production_args import production_args

    dst.mkdir(parents=True)
    args = production_args()
    args.belief_tokens = arm
    with PAR.declared_torch_state(1):
        model, _, _ = build_fresh_model(seed, args=args)
        if perturb:
            perturb_(model.policy, seed=seed + 100, scale=PERTURB_SCALE)
        path = dst / "snapshot_000000001000.zip"
        model.save(str(path))
    (dst / "model_config.json").write_text(
        current_model_version(load_mappings(), **arch_toggles_from_model(model)).to_json())
    return path


@pytest.fixture(scope="module")
def bank_slice():
    """One battle per source cycle of the committed bank (9 battles), re-encoded with its truths."""
    from main.belief_roles.bank_rows import bank_rows_of
    from main.policy_spectrum import bank as B
    from main.policy_spectrum.bank_test import BANK_V1

    bank = B.load_bank(BANK_V1)
    seen, keep = set(), set()
    for b in bank.battles:
        if b.source["label"] not in seen:
            seen.add(b.source["label"])
            keep.add(b.battle_id)
    sub = B.Bank(bank.manifest, [b for b in bank.battles if b.battle_id in keep],
                 [d for d in bank.decisions if d["battle"] in keep])
    return bank_rows_of(sub, workers=2)


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory):
    root = tmp_path_factory.mktemp("belief_roles")
    return {"blob": _save(root / "run_blob", "blob", 11, perturb=True),
            "fixed_mass": _save(root / "run_fm", "fixed_mass", 12, perturb=True),
            "fixed_mass_cold": _save(root / "run_fm_cold", "fixed_mass", 13, perturb=False)}


def _read(br, path, label, **kw):
    from main.belief_roles.__main__ import read_one
    from main.belief_roles.roles import derive

    return read_one(br, derive(), path, label, threads=2, commit="test", reencode_s=0.0, **kw)


def test_both_arms_read_on_the_bank(bank_slice, ckpts):
    br = bank_slice
    assert br.n > 100 and (br.event >= 0).any() and (br.event >= 1000).any()
    reads = {k: _read(br, p, k) for k, p in ckpts.items()}
    assert reads["blob"]["arm"] == "blob"
    assert reads["fixed_mass"]["arm"] == "fixed_mass" == reads["fixed_mass_cold"]["arm"]
    for k, r in reads.items():
        json.dumps(r)                                                  # serialisable, no NaN objects
        pr = r["per_run"]
        for m in ("intent_logloss", "intent_miss_rate", "presence_brier", "presence_bce",
                  "other_mean_err", "roles_r1_weighted_abs_delta"):
            assert pr[m] is not None and math.isfinite(pr[m]), (k, m, pr[m])
        assert r["label_mismatch"] == 0                                # the truth counts k inside V
        assert r["off_pool"]["n_rows"] == 0 and r["per_run_off_pool"]["intent_logloss"] is None
        it = r["on_pool"]["intent"]["all"]
        assert it["n_covered"] + it["n_miss"] == it["n_labeled"] > 0
        # Amendment 3(b): a fixed_mass read with no blob set has NO adoption-gate value
        if r["arm"] == "fixed_mass":
            assert r["eset_reference"] == {"mode": "none"}
            assert pr["intent_logloss_conditional"] is None
    # the Smogon prior column is checkpoint-independent
    a, b = reads["blob"]["on_pool"]["prior"], reads["fixed_mass"]["on_pool"]["prior"]
    assert a["presence"]["brier"] == b["presence"]["brier"]
    assert a["roles_r1_r2"]["weighted_abs_delta"] == b["roles_r1_r2"]["weighted_abs_delta"]
    # a COLD fixed_mass checkpoint (δ_θ zero-init) reads exactly the prior's presence; a perturbed one does not
    cold = reads["fixed_mass_cold"]["on_pool"]
    assert abs(cold["arm"]["presence"]["brier"] - cold["prior"]["presence"]["brier"]) < 1e-6
    hot = reads["fixed_mass"]["on_pool"]
    assert abs(hot["arm"]["presence"]["brier"] - hot["prior"]["presence"]["brier"]) > 1e-4
    # the switch side covers every switch-in in both arms: blob through BeliefHead's content over V,
    # fixed_mass (U4's flat pointer) through the hypothesis slots + OTHER_species' tail
    for k in reads:
        assert reads[k]["on_pool"]["intent"]["switch"]["n_miss"] == 0, k
        assert reads[k]["on_pool"]["intent"]["miss_breakdown"]["switch"] == 0, k


def test_the_conditional_metric_on_the_blob_set(bank_slice, ckpts, tmp_path):
    """Amendment 3(b) end to end: the blob read writes its named set E_row, the fixed_mass read is
    scored on it (the look's blob set, here of one run). On its OWN set the blob's conditional loss is its as-built loss on the same rows (all
    its mass sits on E_row); the fixed_mass arm is scored on a subset of the blob's in-set rows (its own
    rule-8 rows excluded), gives every in-set event positive mass, and reports its outside mass."""
    from main.belief_roles.eset import ERow

    br = bank_slice
    blob = _read(br, ckpts["blob"], "blob", erow_out=tmp_path / "blob.erow.npz")
    ref = ERow.load(tmp_path / "blob.erow.npz")
    assert ref.n == br.n and ref.meta["checkpoint_sha256"] == blob["checkpoint"]["sha256"]
    fm = _read(br, ckpts["fixed_mass"], "fm", references=[ref])
    assert blob["eset_reference"]["mode"] == "own"
    assert fm["eset_reference"]["mode"] == "all_blob_mean"
    assert [x["checkpoint_sha256"] for x in fm["eset_reference"]["references"]] == [blob["checkpoint"]["sha256"]]
    bi = blob["on_pool"]["intent"]["all"]
    bc = blob["on_pool"]["intent_conditional"]["per_reference"][0]
    fcm = fm["on_pool"]["intent_conditional"]
    assert fcm["available"] and fcm["n_references"] == 1
    fc = fcm["per_reference"][0]
    assert fm["per_run"]["intent_logloss_conditional"] == fc["all"]["logloss"] == fcm["mean"]["logloss"]
    assert bc["all"]["n_scored"] == bi["n_covered"] and bc["all"]["n_rows"] == bi["n_labeled"]
    # on its own set the blob's renormalisation can only RAISE its probabilities: its α keeps SWITCH
    # finite on rows where no β slot is legal (mass that names no event; F-X5-AM3-2), which the
    # as-built read charged to it
    assert bc["all"]["logloss"] <= bi["logloss"] + 1e-12
    assert abs(bc["coverage"]["outside_freq"] - bi["miss_rate"]) < 1e-12
    assert fc["all"]["n_zero_event_mass"] == 0
    assert 0 < fc["all"]["n_in_set"] <= bc["all"]["n_in_set"]
    assert math.isfinite(fm["per_run"]["intent_logloss_conditional"])
    assert 0.0 < fc["coverage"]["mean_outside_mass"] < 1.0
