from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import stable_baselines3
from sb3_contrib import MaskablePPO

from agents.model.model_version import ModelVersion, ModelVersionError
from utils.git import get_git_hash


def save_model_snapshot(
    model_dir: str,
    version: ModelVersion,
    git_hash: Optional[str] = None,
    current_lr: Optional[float] = None,
    current_epochs: Optional[int] = None,
) -> None:
    """Write model_config.json and metadata.json into model_dir.

    Does NOT call model.save() — the caller is responsible for the .zip file.
    Safe to call multiple times; files are overwritten in place.
    """
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, "model_config.json"), "w") as f:
        f.write(version.to_json())

    if git_hash is None:
        git_hash = get_git_hash()

    metadata = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash,
        "python_version": sys.version,
        "sb3_version": stable_baselines3.__version__,
    }
    if current_lr is not None:
        metadata["current_lr"] = current_lr
    if current_epochs is not None:
        metadata["current_epochs"] = current_epochs
    with open(os.path.join(model_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)


def write_checkpoint_sidecar(
    checkpoint_path: str,
    current_lr: float,
    current_epochs: int,
) -> None:
    """Write lr/epochs alongside a checkpoint .zip as a small JSON sidecar.

    checkpoint_path: full path to the .zip (with or without extension).
    Sidecar lands at the same path with .zip replaced by .json.
    """
    with open(_sidecar_path(checkpoint_path), "w") as f:
        json.dump({"current_lr": current_lr, "current_epochs": current_epochs}, f, indent=2)


def read_checkpoint_sidecar(checkpoint_path: str) -> dict:
    """Read the sidecar JSON for a checkpoint. Returns {} if not found."""
    path = _sidecar_path(checkpoint_path)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _sidecar_path(checkpoint_path: str) -> str:
    if checkpoint_path.endswith(".zip"):
        return checkpoint_path[:-4] + ".json"
    return checkpoint_path + ".json"


def load_model_snapshot(
    model_path: str,
    env,
    current_version: ModelVersion,
    device: str = "auto",
    tensorboard_log: Optional[str] = None,
) -> MaskablePPO:
    """Load a model with a compatibility check against the current architecture.

    Args:
        model_path:      Path to the .zip (with or without extension), or a directory
                         containing final_model.zip or best_model.zip.
        env:             VecEnv to attach to the loaded model.
        current_version: ModelVersion reflecting current code; checked against saved config.
        device:          Passed to MaskablePPO.load().
        tensorboard_log: Passed to MaskablePPO.load().

    Raises:
        ModelVersionError:  If saved config is incompatible with current_version.
        FileNotFoundError:  If no .zip can be found at the resolved path.
    """
    zip_path, config_dir = _resolve_paths(model_path)

    config_path = os.path.join(config_dir, "model_config.json")
    if os.path.exists(config_path):
        saved_version = ModelVersion.from_json_file(config_path)
        current_version.check_compatible(saved_version)
    else:
        print(
            f"[ModelVersion] WARNING: No model_config.json found at {config_dir!r}. "
            "Skipping compatibility check (legacy model)."
        )

    kwargs: dict = {"env": env, "device": device}
    if tensorboard_log:
        kwargs["tensorboard_log"] = tensorboard_log

    return MaskablePPO.load(zip_path, **kwargs)


def _resolve_paths(model_path: str) -> tuple[str, str]:
    """Return (zip_path, config_dir) for the given model_path.

    Tries: exact path, path+'.zip', path/final_model.zip, path/best_model.zip.
    """
    candidates = [
        model_path,
        model_path + ".zip",
        os.path.join(model_path, "final_model.zip"),
        os.path.join(model_path, "best_model.zip"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate, os.path.dirname(os.path.abspath(candidate))

    raise FileNotFoundError(
        f"Cannot find model zip at any of: {candidates}"
    )
