"""K9(b)'s weight-dump cap is PER PROCESS, not per lineage (T61, 2026-10-19).

THE BUG: `_dump_artifacts` (`rust_rollout.consistency`) counts its policy-weight dumps on the model
(``_behaviour_weight_dumps``, cap `_DUMP_WEIGHTS_MAX` = 3), and the attribute was not in
`_excluded_save_params`. Every checkpoint saved it and every resume restored it, so after the lineage's
third violation no restart ever dumped weights again (X26: none after 203M;
`designs/research_state/measurements/x26_a3_stop_316M_2026-10-18/` F1).

Each test fails on revert:
  * a save no longer writes the counter (revert the `hparams._excluded_save_params` entry);
  * a checkpoint that ALREADY carries counter = 3 resumes with the count back at zero, and the resumed
    model's next violation dumps its weights (revert `strict_load.RESET_ON_LOAD` / its use in
    `never_unpickled`).
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

from agents.training.rust_rollout import consistency as C


def _plant_counter(src: Path, dst: Path, value: int) -> Path:
    """A copy of ``src`` whose sb3 ``data`` member carries the counter as an OLD save wrote it (a plain
    JSON int, which sb3 restores verbatim)."""
    with zipfile.ZipFile(src) as zin, zipfile.ZipFile(dst, "w") as zout:
        for item in zin.infolist():
            raw = zin.read(item.filename)
            if item.filename == "data":
                data = json.loads(raw)
                data["_behaviour_weight_dumps"] = value
                raw = json.dumps(data).encode()
            zout.writestr(item, raw)
    return dst


def _violate(model: Any, dump_dir: Path, n_updates: int) -> Dict[str, Any]:
    """One violation's dump: a single judged row far over the gate's bar."""
    one = np.zeros(1, dtype=np.int64)
    details = {"t": one, "e": one, "action": one, "logp_stored": np.zeros(1), "logp_recomputed": np.ones(1),
               "obs": {}}
    return C._dump_artifacts(model, str(dump_dir), details, np.array([1.0]), n_updates)


@pytest.fixture(scope="module")
def world(tmp_path_factory: Any) -> Dict[str, Any]:
    from agents.training.opponent_inference_load_test import RIDEALONG_OFF, _build, _save

    root = tmp_path_factory.mktemp("wdump")
    model, version, _ = _build(RIDEALONG_OFF, 5)
    model._behaviour_weight_dumps = C._DUMP_WEIGHTS_MAX            # this process has used its cap
    zip_path = _save(model, version, root / "run")
    legacy = _plant_counter(zip_path, root / "run" / "legacy.zip", C._DUMP_WEIGHTS_MAX)
    return {"model": model, "version": version, "zip": zip_path, "legacy": legacy, "root": root}


def test_a_save_writes_no_weight_dump_counter(world: Dict[str, Any]) -> None:
    data = json.loads(zipfile.ZipFile(world["zip"]).read("data"))
    assert "_behaviour_weight_dumps" not in data


def test_the_saving_process_is_capped(world: Dict[str, Any], tmp_path: Path) -> None:
    """Precondition: the cap holds WITHIN a process (a capped model dumps rows but no weights)."""
    out = _violate(world["model"], tmp_path, 7)
    assert "rows_npz" in out and "policy_pt" not in out, out


def test_a_resume_of_a_checkpoint_carrying_counter_3_still_dumps_weights(world: Dict[str, Any],
                                                                         tmp_path: Path) -> None:
    from agents.model.snapshot import load_checkpoint_strict, load_model_snapshot

    data = json.loads(zipfile.ZipFile(world["legacy"]).read("data"))
    assert data["_behaviour_weight_dumps"] == C._DUMP_WEIGHTS_MAX   # precondition: the old save's counter
    resumed = load_model_snapshot(str(world["legacy"]), env=None, current_version=world["version"], device="cpu")
    assert resumed.__dict__.get("_behaviour_weight_dumps", 0) == 0
    out = _violate(resumed, tmp_path, 11)
    assert out.get("policy_pt") == "behaviour_violation_u11_policy.pt", out
    assert (tmp_path / out["policy_pt"]).is_file()
    assert resumed._behaviour_weight_dumps == 1
    # a read-only strict load of the same old checkpoint loads cleanly, the counter ignored
    assert load_checkpoint_strict(str(world["legacy"]), device="cpu").__dict__.get("_behaviour_weight_dumps", 0) == 0
