"""Unit tests of the reader's per-row ``Columns`` with RND variants (no model, no bank): the npz
flattening, the key accessors, and the uncertainty scores meter (i) reads per variant."""

from __future__ import annotations

import numpy as np

from main.ridealong_read import reader as RD


def _cols(n=7, variants=("fast", "feat")) -> RD.Columns:
    rng = np.random.default_rng(0)
    f = {k: rng.normal(size=(n, 11)).astype(np.float32) if k in ("logits", "pi", "adv_mean",
                                                                  "adv_std", "adv_raw_std")
         else rng.normal(size=(n, 128)).astype(np.float32) if k == "pooled"
         else rng.normal(size=(n, 3)).astype(np.float32) if k == "opp_mean"
         else rng.uniform(size=n).astype(np.float32)
         for k in RD.ARRAY_FIELDS}
    rndv = {f"rndv_{v}_{s}": rng.normal(size=n).astype(np.float32) for v in variants
            for s in ("err", "z")}
    return RD.Columns(**f, rndv=rndv)


def test_baseline_spec_carries_every_declared_variant():
    from agents.model.ridealong_heads import RND_VARIANTS, RideAlongSpec

    assert tuple(RD.BASELINE_SPEC_KW["rnd_variants"]) == RND_VARIANTS
    assert RideAlongSpec(**RD.BASELINE_SPEC_KW).rnd_variants == RND_VARIANTS


def test_columns_keys_and_accessors():
    c = _cols()
    assert c.variant_names() == ["fast", "feat"]
    assert c.novelty_keys() == ["rnd", "rndv_fast", "rndv_feat"]
    assert c.err("rnd") is c.rnd_err and c.z("rndv_feat") is c.rndv["rndv_feat_z"]
    nob = _cols(variants=())
    nob.rnd_err = np.full(7, np.nan, dtype=np.float32)
    assert nob.novelty_keys() == []


def test_npz_flattens_variants(tmp_path):
    c = _cols()
    np.savez_compressed(tmp_path / "x.rows.npz", ids_sha256=np.array("s"), **c.arrays())
    z = np.load(tmp_path / "x.rows.npz")
    for k in ("rndv_fast_err", "rndv_fast_z", "rndv_feat_err", "rndv_feat_z", "rnd_err", "v"):
        assert k in z.files
    assert "rndv" not in z.files
    np.testing.assert_array_equal(z["rndv_feat_err"], c.rndv["rndv_feat_err"])


def test_uncertainty_scores_add_each_variant_z():
    s = RD.uncertainty_scores(_cols())
    assert {"rndv_fast_z", "rndv_feat_z", "rnd_z", "ens_std", "ens_logit_std",
            "ref_v_entropy"} == set(s)


def test_run_checkpoints_sorted_by_step_and_refuses_none(tmp_path):
    import pytest

    from main.ridealong_read.__main__ import _ckpt, run_checkpoints

    ck = tmp_path / "run" / "checkpoints"
    ck.mkdir(parents=True)
    for s in (10000000, 2000000, 900000):
        (ck / f"checkpoint_{s}_steps.zip").write_bytes(b"")
    (ck / "checkpoint_2000000_steps.json").write_text("{}")          # sidecars are not checkpoints
    (tmp_path / "run" / "snapshots").mkdir()
    got = run_checkpoints(tmp_path / "run")
    assert [p.name for p in got] == ["checkpoint_900000_steps.zip", "checkpoint_2000000_steps.zip",
                                     "checkpoint_10000000_steps.zip"]        # numeric, not lexical
    assert _ckpt(str(got[0]))[1] == "run__checkpoint_900000_steps"           # each named explicitly
    with pytest.raises(SystemExit, match="no checkpoints"):
        run_checkpoints(tmp_path / "run" / "snapshots")
