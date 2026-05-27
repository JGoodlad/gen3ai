from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional

import stable_baselines3
from sb3_contrib import MaskablePPO

from agents.model.model_version import ModelVersion, ModelVersionError
from agents.training.instrumented_ppo import InstrumentedMaskablePPO
from utils.git import get_git_hash


def save_model_snapshot(
    model_dir: str,
    version: ModelVersion,
    git_hash: Optional[str] = None,
    current_lr: Optional[float] = None,
    current_epochs: Optional[int] = None,
    hparams: Optional[dict] = None,
) -> None:
    """Write model_config.json and metadata.json into model_dir.

    Does NOT call model.save() — the caller is responsible for the .zip file.
    Safe to call multiple times; files are overwritten in place.
    Preserves any existing snapshot_history and evals in the metadata file.
    """
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, "model_config.json"), "w") as f:
        f.write(version.to_json())

    if git_hash is None:
        import os as _os
        git_hash = _os.environ.get("LAUNCHER_GIT_HASH") or get_git_hash()

    # Preserve snapshot_history accumulated by other writers.
    meta_path = os.path.join(model_dir, "metadata.json")
    existing_history = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            existing = json.load(f)
            existing_history = existing.get("snapshot_history", {})

    metadata = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash,
        "python_version": sys.version,
        "sb3_version": stable_baselines3.__version__,
    }
    if hparams:
        metadata.update(hparams)
    if current_lr is not None:
        metadata["current_lr"] = current_lr
    if current_epochs is not None:
        metadata["current_epochs"] = current_epochs
    if existing_history:
        metadata["snapshot_history"] = existing_history
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


def record_eval_results(model_dir: str, step: int, metrics: dict) -> None:
    """Attach eval results to the most recently-saved checkpoint in snapshot_history.

    If no checkpoint has been recorded yet, logs a warning and skips the write
    rather than creating a top-level 'evals' key in an inconsistent location.
    """
    meta_path = os.path.join(model_dir, "metadata.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    history = meta.get("snapshot_history", {})
    latest = _latest_checkpoint(history)
    if latest is None:
        print("[EVAL] Warning: no checkpoint in snapshot_history yet; eval results not persisted.")
        return

    history[latest]["evals"] = {
        **metrics,
        "step": step,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
    meta["snapshot_history"] = history
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def _latest_checkpoint(history: dict) -> str | None:
    """Return the checkpoint name with the highest step number, or None."""
    best_name, best_step = None, -1
    for name in history:
        try:
            step = int(name.split("_")[1])
        except (IndexError, ValueError):
            continue
        if step > best_step:
            best_name, best_step = name, step
    return best_name


def record_snapshot_in_history(
    model_dir: str,
    checkpoint_name: str,
    lr: float,
    n_epochs: int,
    hparams: Optional[dict] = None,
    git_hash: Optional[str] = None,
) -> None:
    """Append or update a checkpoint entry in snapshot_history within metadata.json.

    checkpoint_name: basename of the checkpoint zip, e.g. 'checkpoint_50000000_steps.zip'.
    Creates metadata.json if it doesn't exist yet (history-only file until the next
    save_model_snapshot call fills in the rest of the fields).
    """
    meta_path = os.path.join(model_dir, "metadata.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    history = meta.get("snapshot_history", {})
    entry = {"lr": lr, "n_epochs": n_epochs}
    if hparams:
        entry.update(hparams)
    entry["git_hash"] = git_hash or get_git_hash()
    history[checkpoint_name] = entry
    meta["snapshot_history"] = history
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def record_checkpoint(
    model_dir: str,
    checkpoint_path: str,
    lr: float,
    n_epochs: int,
    hparams: Optional[dict] = None,
    git_hash: Optional[str] = None,
) -> None:
    """Write per-checkpoint metadata file and append to run-level snapshot_history.

    checkpoint_path: full path to the .zip (with or without extension).
    Call this whenever a checkpoint .zip is saved.
    """
    resolved_hash = git_hash or get_git_hash()
    write_checkpoint_metadata(checkpoint_path, lr, n_epochs, hparams=hparams, git_hash=resolved_hash)
    name = os.path.basename(checkpoint_path)
    if not name.endswith(".zip"):
        name += ".zip"
    record_snapshot_in_history(model_dir, name, lr, n_epochs, hparams=hparams, git_hash=resolved_hash)


def write_checkpoint_metadata(
    checkpoint_path: str,
    current_lr: float,
    current_epochs: int,
    hparams: Optional[dict] = None,
    git_hash: Optional[str] = None,
) -> None:
    """Write lr/epochs (and optional hparams) alongside a checkpoint .zip.

    checkpoint_path: full path to the .zip (with or without extension).
    Metadata file lands at the same path with .zip replaced by .json.
    """
    data = dict(hparams) if hparams else {}
    data["current_lr"] = current_lr
    data["current_epochs"] = current_epochs
    import os as _os
    data["git_hash"] = git_hash or _os.environ.get("LAUNCHER_GIT_HASH") or get_git_hash()
    with open(_checkpoint_metadata_path(checkpoint_path), "w") as f:
        json.dump(data, f, indent=2)


def read_checkpoint_metadata(checkpoint_path: str) -> dict:
    """Read the per-checkpoint metadata JSON. Returns {} if not found."""
    path = _checkpoint_metadata_path(checkpoint_path)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def _checkpoint_metadata_path(checkpoint_path: str) -> str:
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
        device:          Passed to InstrumentedMaskablePPO.load().
        tensorboard_log: Passed to InstrumentedMaskablePPO.load().

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

    return InstrumentedMaskablePPO.load(zip_path, **kwargs)


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
