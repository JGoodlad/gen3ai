"""The ride-along reader end to end on a handful of banked battles (integration: the Rust core
re-encodes the bank, a real checkpoint is loaded from the MAIN checkout's ``models/``).

Skips cleanly when there is no ``models/`` archive or the checkpoint is absent (a CI box, a fresh
clone). Teeth: the heads' provenance is stated, the forward's V equals the policy-spectrum path's
critic read on the same rows, feature-RND's drift of a checkpoint against ITSELF is exactly zero,
and a bare run directory / an output under ``models/`` are refused.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.sim, pytest.mark.integration]

CKPT_REL = "ai_v14_07_g0p_k2/final_model.zip"
N_BATTLES = 12


@pytest.fixture(scope="module")
def ckpt() -> Path:
    from utils.paths import main_models_dir

    md = main_models_dir()
    if md is None:
        pytest.skip("no models/ archive (main_models_dir() is None)")
    p = md / CKPT_REL
    if not p.exists():
        pytest.skip(f"checkpoint {p} is absent")
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
    assert r["reencode"]["obs_as_recorded"] is True
    assert r["own_truth_continuation"] == "K2final"
    assert r["bank"]["battles"] == N_BATTLES
    u = r["i_uncertainty_vs_v_error"]["all"]
    assert set(u) == {"ens_std", "ens_logit_std", "rnd_z", "ref_v_entropy"}
    assert r["i_uncertainty_vs_v_error"]["rows"]["used"] > 0
    rc_ = json.loads((out / "rnd_input_choice.json").read_text())
    through = rc_["feature_rnd_drift"]["1"]["through_B"][str(ckpt)]["drift"]["heldout"]
    assert through["inflation_mean"] == 1.0            # a checkpoint against itself: no drift
    z = np.load(tmp_path / "arch" / "K2final.rows.npz")
    assert z["v"].shape[0] == r["bank"]["decisions"] and np.isfinite(z["adv_std"]).any()


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
    assert md is not None
    with pytest.raises(SystemExit, match="REFUSED"):
        main(["--checkpoint", str(ckpt), "--out", str(md / "ridealong_read_should_not_exist")])
    assert not (md / "ridealong_read_should_not_exist").exists()
