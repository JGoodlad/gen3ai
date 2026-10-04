"""The CLI's refusals: an output under `models/`, a bare run directory as a checkpoint."""
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
