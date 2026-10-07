"""The readers end to end on a slice of the committed Lane S bank: fresh X5 checkpoints saved, loaded back
through THE strict loader, read on CPU (integration: the Rust core re-encodes the bank; `models/` is never
written); the adoption-gate metric scored on a BANKED blob named set (the X5 A/B's look-3 read, sliced to
the same rows); a PRE-BREAK checkpoint refused with the loader's reason."""
from __future__ import annotations

import json
import math

import pytest

pytestmark = [pytest.mark.sim, pytest.mark.integration]

#: A blob run's named set as the X5 A/B's look-3 read banked it (the full bank v1, 20,712 rows).
BANKED_BLOB_EROW = ("designs", "research_state", "measurements", "x5ab_look3_2026-10-07", "purpose",
                    "blob_s1001.erow.npz")


def _save(dst, seed: int, perturb: bool):
    from agents.model.parity_probe import PERTURB_SCALE, perturb_
    from agents.model.snapshot import arch_toggles_from_model, current_model_version
    from agents.observation.state_encoder import load_mappings
    from agents.training.rust_eval import parity as PAR
    from main.fresh_checkpoint import build_fresh_model
    from main.train.production_args import production_args

    dst.mkdir(parents=True)
    with PAR.declared_torch_state(1):
        model, _, _ = build_fresh_model(seed, args=production_args())
        if perturb:
            perturb_(model.policy, seed=seed + 100, scale=PERTURB_SCALE)
        path = dst / "snapshot_000000001000.zip"
        model.save(str(path))
    (dst / "model_config.json").write_text(
        current_model_version(load_mappings(), **arch_toggles_from_model(model)).to_json())
    return path


@pytest.fixture(scope="module")
def bank_slice():
    """One battle per source cycle of the committed bank (9 battles), re-encoded with its truths, and the
    slice's row indices in the FULL bank (a banked E_row is per full-bank row)."""
    from main.belief_roles.bank_rows import bank_rows_of
    from main.policy_spectrum import bank as B
    from main.policy_spectrum.bank_test import BANK_V1

    bank = B.load_bank(BANK_V1)
    seen, keep = set(), set()
    for b in bank.battles:
        if b.source["label"] not in seen:
            seen.add(b.source["label"])
            keep.add(b.battle_id)
    idx = [i for i, d in enumerate(bank.decisions) if d["battle"] in keep]
    sub = B.Bank(bank.manifest, [b for b in bank.battles if b.battle_id in keep],
                 [bank.decisions[i] for i in idx])
    return bank_rows_of(sub, workers=2), idx


@pytest.fixture(scope="module")
def ckpts(tmp_path_factory):
    root = tmp_path_factory.mktemp("belief_roles")
    return {"x5": _save(root / "run_x5", 12, perturb=True),
            "x5_cold": _save(root / "run_x5_cold", 13, perturb=False)}


def _read(br, path, label, **kw):
    from main.belief_roles.__main__ import read_one
    from main.belief_roles.roles import derive

    return read_one(br, derive(), path, label, threads=2, commit="test", reencode_s=0.0, **kw)


def test_an_x5_checkpoint_reads_on_the_bank(bank_slice, ckpts):
    br, _ = bank_slice
    assert br.n > 100 and (br.event >= 0).any() and (br.event >= 1000).any()
    reads = {k: _read(br, p, k) for k, p in ckpts.items()}
    for k, r in reads.items():
        assert r["arm"] == "fixed_mass"                                # the X5 arm's name in every read
        json.dumps(r)                                                  # serialisable, no NaN objects
        pr = r["per_run"]
        for m in ("intent_logloss", "intent_miss_rate", "presence_brier", "presence_bce",
                  "other_mean_err", "roles_r1_weighted_abs_delta"):
            assert pr[m] is not None and math.isfinite(pr[m]), (k, m, pr[m])
        assert r["label_mismatch"] == 0                                # the truth counts k inside V
        assert r["off_pool"]["n_rows"] == 0 and r["per_run_off_pool"]["intent_logloss"] is None
        it = r["on_pool"]["intent"]["all"]
        assert it["n_covered"] + it["n_miss"] == it["n_labeled"] > 0
        # Amendment 3(b): a read with no blob set has NO adoption-gate value
        assert r["eset_reference"] == {"mode": "none"}
        assert pr["intent_logloss_conditional"] is None
        # the switch side covers every switch-in: the hypothesis slots + OTHER_species' tail
        assert r["on_pool"]["intent"]["switch"]["n_miss"] == 0, k
        assert r["on_pool"]["intent"]["miss_breakdown"]["switch"] == 0, k
    # the Smogon prior column is checkpoint-independent
    a, b = reads["x5"]["on_pool"]["prior"], reads["x5_cold"]["on_pool"]["prior"]
    assert a["presence"]["brier"] == b["presence"]["brier"]
    assert a["roles_r1_r2"]["weighted_abs_delta"] == b["roles_r1_r2"]["weighted_abs_delta"]
    # a COLD checkpoint (δ_θ zero-init) reads exactly the prior's presence; a perturbed one does not
    cold = reads["x5_cold"]["on_pool"]
    assert abs(cold["arm"]["presence"]["brier"] - cold["prior"]["presence"]["brier"]) < 1e-6
    hot = reads["x5"]["on_pool"]
    assert abs(hot["arm"]["presence"]["brier"] - hot["prior"]["presence"]["brier"]) > 1e-4


def test_the_conditional_metric_on_a_banked_blob_set(bank_slice, ckpts):
    """Amendment 3(b) end to end on a BANKED blob named set (the X5 A/B's look-3 read of blob_s1001; the
    blob read arm itself was deleted at the version break), sliced to the test bank's rows: the X5 read is
    scored on it, records the reference, gives every in-set event positive mass and reports its outside
    mass."""
    import numpy as np

    from main.belief_roles.eset import ERow
    from utils.paths import repo_path

    br, idx = bank_slice
    full = ERow.load(repo_path(*BANKED_BLOB_EROW))
    assert full.meta["bank_sha256"] == br.bank.manifest["content_sha256"]    # the same bank v1
    ii = np.asarray(idx)
    ref = ERow(seat_events=full.seat_events[ii], switch_ok=full.switch_ok[ii], tie=full.tie[ii],
               meta=dict(full.meta))
    assert ref.n == br.n
    fm = _read(br, ckpts["x5"], "x5", references=[ref])
    er = fm["eset_reference"]
    assert er["mode"] == "all_blob_mean"
    assert [x["checkpoint_sha256"] for x in er["references"]] == [full.meta["checkpoint_sha256"]]
    assert er["references"][0]["erow_sha256"] == ref.content_sha256()
    fcm = fm["on_pool"]["intent_conditional"]
    assert fcm["available"] and fcm["n_references"] == 1
    fc = fcm["per_reference"][0]
    assert fc["reference_label"] == "blob_s1001"
    assert fm["per_run"]["intent_logloss_conditional"] == fc["all"]["logloss"] == fcm["mean"]["logloss"]
    assert fc["all"]["n_zero_event_mass"] == 0
    assert 0 < fc["all"]["n_scored"] <= fc["all"]["n_in_set"] <= fc["all"]["n_rows"]
    assert math.isfinite(fm["per_run"]["intent_logloss_conditional"])
    assert 0.0 < fc["coverage"]["mean_outside_mass"] < 1.0


def test_a_pre_break_run_is_refused_by_the_read_with_the_loaders_reason(bank_slice, ckpts, tmp_path):
    """A REAL checkpoint zip whose run config records a pre-break generation (here: the same weights under
    a v143 blob config) is refused by the read itself — `forward.ReadRefused` carrying `version_break`'s
    reason and the last pre-break commit — before any forward."""
    import shutil

    from agents.model.model_version.version_break import LAST_BLOB_COMMIT
    from main.belief_roles.forward import ReadRefused

    br, _ = bank_slice
    run = tmp_path / "run_pre_break"
    run.mkdir()
    z = run / "snapshot_000000001000.zip"
    shutil.copy(ckpts["x5_cold"], z)
    cfg = json.loads((ckpts["x5_cold"].parent / "model_config.json").read_text())
    cfg.update(config_version=143, belief_tokens="blob")
    (run / "model_config.json").write_text(json.dumps(cfg))
    with pytest.raises(ReadRefused, match=f"(?s)belief_tokens='blob'.*DELETED.*{LAST_BLOB_COMMIT[:12]}"):
        _read(br, z, "pre_break")
