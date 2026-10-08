"""The ride-along reader end to end on a handful of banked battles (integration: the Rust core
re-encodes the bank; the checkpoint is a FRESH production one saved under the test's tmp dir).

Why fresh: every archived checkpoint in ``models/`` predates the X5 version break (config v144) and is
refused by this code (it runs PINNED to its own commit), so no archived checkpoint is readable at HEAD
until a post-break run exists. A fresh production checkpoint carries no ride-along heads, so the reader
attaches FRESH ones — the ``fresh-untrained`` branch below. Teeth: the heads' provenance is stated, the
forward's V equals the policy-spectrum path's critic read on the same rows, feature-RND's drift of a
checkpoint against ITSELF is exactly zero, fresh ``fast`` / ``decay`` RND variants score bit-identically
to base (they ARE base at init) and fresh heads are their own identification floor, and a bare run
directory / an output under ``models/`` are refused.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.sim, pytest.mark.integration]

N_BATTLES = 12


def B_DIR() -> Path:
    from main.ridealong_read.reader import bank_dir
    return bank_dir()


@pytest.fixture(scope="module")
def ckpt(tmp_path_factory) -> Path:
    """A fresh production (X5) checkpoint + its run-level ``model_config.json``."""
    from agents.model.snapshot import arch_toggles_from_model, current_model_version
    from agents.observation.state_encoder import load_mappings
    from agents.training.rust_eval import offline as PAR
    from main.fresh_checkpoint import build_fresh_model
    from main.train.production_args import production_args

    run = tmp_path_factory.mktemp("ridealong_read") / "run_x5"
    (run / "checkpoints").mkdir(parents=True)
    with PAR.declared_torch_state(1):
        model, _, _ = build_fresh_model(7, args=production_args())
        p = run / "checkpoints" / "final_model.zip"
        model.save(str(p))
    (run / "model_config.json").write_text(
        current_model_version(load_mappings(), **arch_toggles_from_model(model)).to_json())
    return p


def test_reader_smoke_end_to_end(ckpt, tmp_path):
    from main.ridealong_read.__main__ import main

    out = tmp_path / "out"
    rc = main(["--checkpoint", f"{ckpt}=K2final", "--out", str(out), "--archive", str(tmp_path / "arch"),
               "--limit-battles", str(N_BATTLES), "--threads", "2", "--rnd-epochs", "1",
               "--drift-pair", str(ckpt), str(ckpt)])
    assert rc == 0
    r = json.loads((out / "K2final.json").read_text())
    assert r["schema"] == "gen3_ridealong_read_v1"
    assert r["heads"] in ("trained", "fresh-untrained")
    # Gate ①'s byte check belongs to the bank's recording encoder (`policy_spectrum_integration_test`'s rule): while
    # this checkout's encoder identity equals the bank's, every re-encoded row must be the recorded one; after a
    # deliberate encoder change (the X5 version break's OBS-FACTS append grew the row 2761 -> 2845) the read must
    # STATE that it ran on a drifted re-encoding, never claim the recorded observations.
    from main.policy_spectrum import bank as B
    same_encoder = B.encoder_identity() == B.load_bank(B_DIR()).manifest["encoder_identity"]
    assert r["reencode"]["recorded_rows_checked"] > 0
    assert r["reencode"]["obs_as_recorded"] is same_encoder
    # a fresh checkpoint is no truth continuation's greedy policy: meter (iv) has no own truth
    assert r["own_truth_continuation"] is None
    assert r["bank"]["battles"] == N_BATTLES
    u = r["i_uncertainty_vs_v_error"]["all"]
    assert {"ens_std", "ens_logit_std", "rnd_z", "ref_v_entropy"} <= set(u)
    assert r["i_uncertainty_vs_v_error"]["rows"]["used"] > 0
    rv = r["rnd_variants"]
    if r["heads"] == "fresh-untrained":                  # a pre-v127 checkpoint: every variant
        assert rv["keys"] == ["rnd", "rndv_fast", "rndv_decay", "rndv_small", "rndv_feat"]
        assert {f"{k}_z" for k in rv["keys"]} <= set(u)
        # fresh fast / decay ARE base (deep copies of its predictor): (a) and (b) read exactly 0
        aw = rv["a_v_error_beyond_v_uncertainty"]["auroc_within_ref_quintiles"]
        late = rv["b_coverage"]["contrasts"]["b2_late_game_vs_rest"]
        assert late["n_pos"] > 0
        for k in ("rndv_fast", "rndv_decay"):
            assert aw[k] == aw["rnd"]                    # None on a read with no |V - z| > 0.5 row
            assert late["vs_base"][k]["diff"] == 0.0 and late["vs_base"][k]["diff_ci"] == [0.0, 0.0]
            assert rv["b_coverage"]["verdicts"][k] == "NOT_DETECTED"
        # fresh heads ARE their own predictor-reset floor: f_i = f_ii = 1, nothing to judge
        for k, row in rv["e_identification"]["keys"].items():
            assert row["f_i"] == 1.0 and row["f_ii"] == 1.0 and row["verdict"] == "INDETERMINATE"
        assert rv["e_identification"]["floor"]["decay_anchor_matches_rebuilt_init"] is True
        assert rv["e_identification"]["probe"]["heads_block_edges_match"] is True
    series = json.loads((out / "rnd_variants_series.json").read_text())
    assert series["checkpoints"][0]["label"] == "K2final"
    assert "skipped" in series["c_saturation"]
    rc_ = json.loads((out / "rnd_input_choice.json").read_text())
    through = rc_["feature_rnd_drift"]["1"]["through_B"][str(ckpt)]["drift"]["heldout"]
    assert through["inflation_mean"] == 1.0            # a checkpoint against itself: no drift
    z = np.load(tmp_path / "arch" / "K2final.rows.npz")
    assert z["v"].shape[0] == r["bank"]["decisions"] and np.isfinite(z["adv_std"]).any()
    if r["heads"] == "fresh-untrained":
        np.testing.assert_array_equal(z["rndv_fast_err"], z["rnd_err"])
        np.testing.assert_array_equal(z["rndv_decay_err"], z["rnd_err"])
        assert np.isfinite(z["rndv_feat_err"]).all() and np.isfinite(z["rndv_small_z"]).all()


def test_v_matches_the_policy_spectrum_critic_read(ckpt):
    from main.policy_spectrum.bank import load_bank
    from main.policy_spectrum.qhat import logits_and_values
    from main.policy_spectrum.reader import inference_globals, load_checkpoint, reencode
    from main.ridealong_read import reader as RD

    bank = RD.limit_bank(load_bank(RD.bank_dir()), 3)
    rows, masks, _ = reencode(bank, workers=1)
    with inference_globals(2):
        model = load_checkpoint(ckpt)
        heads, prov = RD.attach_heads(model.policy, rows.shape[1], rows)
        cols = RD.forward_columns(model, heads, rows, masks, fit_err_stats=True)
        ref = logits_and_values(model, rows, masks)
    np.testing.assert_allclose(cols.v, ref[:, 11], atol=1e-5)
    np.testing.assert_allclose(cols.logits, ref[:, :11], atol=1e-4)
    pooled = RD.pooled_only(ckpt, rows, masks, threads=2)
    np.testing.assert_allclose(pooled, cols.pooled, atol=1e-5)
    assert np.allclose(cols.pi.sum(1), 1.0, atol=1e-5) and (cols.pi[~masks] == 0).all()


def test_refusals(ckpt, tmp_path):
    from main.ridealong_read.__main__ import main
    from utils.paths import main_models_dir

    with pytest.raises(SystemExit, match="REFUSED"):
        main(["--checkpoint", str(ckpt.parent), "--out", str(tmp_path)])
    md = main_models_dir()
    if md is None:
        pytest.skip("no models/ archive (main_models_dir() is None) — the models/ refusal needs one")
    with pytest.raises(SystemExit, match="REFUSED"):
        main(["--checkpoint", str(ckpt), "--out", str(md / "ridealong_read_should_not_exist")])
    assert not (md / "ridealong_read_should_not_exist").exists()
