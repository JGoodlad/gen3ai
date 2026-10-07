"""The CLI's refusals: an output under `models/`, a bare run directory as a checkpoint, a PRE-BREAK
(blob / pre-break fixed_mass) checkpoint — refused with the loader's typed reason, never a traceback."""
from __future__ import annotations

import pytest


def test_models_dir_is_refused(tmp_path, monkeypatch):
    from main.belief_roles.__main__ import main

    models = tmp_path / "models"
    models.mkdir()
    monkeypatch.setenv("GEN3AI_MODELS_DIR", str(models))
    with pytest.raises(SystemExit, match="REFUSED"):
        main(["read", "--out", str(models / "x"), "--ckpt", "a.zip=a"])


def test_a_bare_run_directory_is_refused(tmp_path):
    from main.belief_roles.forward import load_strict

    with pytest.raises(ValueError, match="bare run directory"):
        load_strict(tmp_path)


@pytest.mark.parametrize("cfg", [
    {"config_version": 143, "belief_tokens": "blob"},
    {"config_version": 130},                                  # below v136: blob was the only past
    {"config_version": 143, "belief_tokens": "fixed_mass"},  # a PRE-BREAK X5 checkpoint
])
def test_a_pre_break_checkpoint_is_refused_with_the_pinned_fix_before_the_bank_loads(tmp_path, cfg):
    """The X5 version break (config v144) deleted the blob read arm; a pre-break checkpoint is read PINNED
    to its own commit. The refusal is the loader's (``version_break``) and it comes BEFORE the bank's
    re-encode (the zip here is not even a zip, and no bank exists at ``--bank``): a typed SystemExit naming
    the last pre-break commit, never a traceback."""
    import json

    from agents.model.model_version.version_break import LAST_BLOB_COMMIT
    from main.belief_roles.__main__ import main
    from main.belief_roles.forward import ReadRefused, load_strict

    run = tmp_path / "run"
    (run / "checkpoints").mkdir(parents=True)
    (run / "model_config.json").write_text(json.dumps(cfg))
    z = run / "checkpoints" / "snapshot_000000001000.zip"
    z.write_bytes(b"not a zip")
    with pytest.raises(SystemExit, match=f"(?s)REFUSED.*PRE-GENERATION.*{LAST_BLOB_COMMIT[:12]}"):
        main(["read", "--out", str(tmp_path / "out"), "--bank", str(tmp_path / "no_bank"),
              "--ckpt", f"{z}=a"])
    with pytest.raises(ReadRefused, match=LAST_BLOB_COMMIT[:12]):
        load_strict(z)


def test_a_reference_must_be_a_banked_blob_set_not_a_checkpoint_label(tmp_path):
    from main.belief_roles.__main__ import main

    with pytest.raises(SystemExit, match="BANKED"):
        main(["read", "--out", str(tmp_path / "out"), "--ckpt", "a.zip=a", "--ckpt", "b.zip=b",
              "--reference", "a=b"])
