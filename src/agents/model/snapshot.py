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
    Preserves any existing snapshot_history and the top-level `latest_eval` block
    (so a checkpoint saved after an eval doesn't erase the eval results).
    """
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, "model_config.json"), "w") as f:
        f.write(version.to_json())

    if git_hash is None:
        import os as _os
        git_hash = _os.environ.get("LAUNCHER_GIT_HASH") or get_git_hash()

    # Preserve state accumulated by other writers (this rebuilds metadata from
    # scratch, so anything not carried forward here is dropped).
    meta_path = os.path.join(model_dir, "metadata.json")
    existing_history = {}
    existing_latest_eval = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            existing = json.load(f)
            existing_history = existing.get("snapshot_history", {})
            existing_latest_eval = existing.get("latest_eval")

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
    if existing_latest_eval is not None:
        metadata["latest_eval"] = existing_latest_eval
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)


def record_eval_results(model_dir: str, step: int, metrics: dict) -> None:
    """Write the most recent eval to metadata.json as a top-level `latest_eval`.

    Stored at the TOP LEVEL (not nested under a checkpoint) and labeled by the
    snapshot `step` it evaluated. This is robust to the subprocess-eval timing:
    the eval is for a frozen snapshot and can finish AFTER a newer checkpoint is
    saved (so it must not bind to "the latest checkpoint"), and an early eval can
    land BEFORE any checkpoint exists. Always writes — never skipped for lack of a
    checkpoint. `save_model_snapshot` carries this block forward across checkpoints;
    `read_latest_eval_block` / the TUI-resume path read it back.
    """
    meta_path = os.path.join(model_dir, "metadata.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)

    meta["latest_eval"] = {
        **metrics,
        "step": step,
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
    }
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


def _build_snapshot_entry(
    lr: float,
    n_epochs: int,
    hparams: Optional[dict] = None,
    git_hash: Optional[str] = None,
    handoff_lr: Optional[float] = None,
) -> dict:
    """Build the canonical per-checkpoint metadata dict.

    Single source of truth for BOTH the per-checkpoint sidecar
    (``write_checkpoint_metadata``) and the run-level ``snapshot_history`` entry
    (``record_snapshot_in_history``), so the per-model summary and the history can
    never drift apart.

    The entry is the **union** of the two historical schemas — it carries both
    naming conventions for the same values so no consumer keyed on either name
    loses data:
        ``lr`` / ``n_epochs``                 — snapshot_history convention
        ``current_lr`` / ``current_epochs``   — original sidecar convention
    All four (plus ``git_hash``) are assigned after the hparams are merged in, so
    they always win over a colliding ``hparams`` key.
    """
    entry = dict(hparams) if hparams else {}
    entry["lr"] = entry["current_lr"] = lr
    entry["n_epochs"] = entry["current_epochs"] = n_epochs
    entry["git_hash"] = git_hash or os.environ.get("LAUNCHER_GIT_HASH") or get_git_hash()
    if handoff_lr is not None:
        entry["handoff_lr"] = handoff_lr
    return entry


def record_snapshot_in_history(
    model_dir: str,
    checkpoint_name: str,
    lr: float,
    n_epochs: int,
    hparams: Optional[dict] = None,
    git_hash: Optional[str] = None,
    handoff_lr: Optional[float] = None,
) -> None:
    """Append or update a checkpoint entry in snapshot_history within metadata.json.

    checkpoint_name: basename of the checkpoint zip, e.g. 'checkpoint_50000000_steps.zip'.
    The entry is built by ``_build_snapshot_entry`` — identical in shape to the
    per-checkpoint sidecar. Creates metadata.json if it doesn't exist yet
    (history-only file until the next save_model_snapshot call fills in the rest).
    """
    meta_path = os.path.join(model_dir, "metadata.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    history = meta.get("snapshot_history", {})
    history[checkpoint_name] = _build_snapshot_entry(lr, n_epochs, hparams, git_hash, handoff_lr)
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
    handoff_lr: Optional[float] = None,
) -> None:
    """Write per-checkpoint metadata file and append to run-level snapshot_history.

    checkpoint_path: full path to the .zip (with or without extension).
    Call this whenever a checkpoint .zip is saved.

    handoff_lr: Phase 1 → Phase 2 LR for TwoPhaseLRCallback. ``None`` while
    still in Phase 1; the float starting LR of the cosine once Phase 2 has
    begun. Persisted so launcher restarts reproduce the cosine.
    """
    resolved_hash = git_hash or get_git_hash()
    write_checkpoint_metadata(
        checkpoint_path,
        lr,
        n_epochs,
        hparams=hparams,
        git_hash=resolved_hash,
        handoff_lr=handoff_lr,
    )
    name = os.path.basename(checkpoint_path)
    if not name.endswith(".zip"):
        name += ".zip"
    record_snapshot_in_history(
        model_dir,
        name,
        lr,
        n_epochs,
        hparams=hparams,
        git_hash=resolved_hash,
        handoff_lr=handoff_lr,
    )


def write_checkpoint_metadata(
    checkpoint_path: str,
    lr: float,
    n_epochs: int,
    hparams: Optional[dict] = None,
    git_hash: Optional[str] = None,
    handoff_lr: Optional[float] = None,
) -> None:
    """Write the per-checkpoint metadata sidecar alongside a checkpoint .zip.

    The sidecar's schema is identical to a run-level ``snapshot_history`` entry —
    both are built by ``_build_snapshot_entry``, which emits the union of both
    naming conventions (lr/current_lr, n_epochs/current_epochs) plus git_hash,
    optional handoff_lr and hparams — so the per-model summary mirrors the history
    exactly and drops no field.

    checkpoint_path: full path to the .zip (with or without extension). The sidecar
    lands at the same path with .zip replaced by .json.

    handoff_lr: see ``record_checkpoint``.
    """
    entry = _build_snapshot_entry(lr, n_epochs, hparams, git_hash, handoff_lr)
    with open(_checkpoint_metadata_path(checkpoint_path), "w") as f:
        json.dump(entry, f, indent=2)


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


def current_model_version(mappings) -> ModelVersion:
    """Build a ``ModelVersion`` reflecting the CURRENT code's architecture for ``mappings``.

    Single source for the ``from_layout_and_policy_kwargs`` construction otherwise
    repeated inline (train_rl_agent's self-play pool setup; the eval worker's sentinel
    version check). Returns the ``current_version`` to pass to ``load_model_snapshot`` /
    ``SnapshotPool`` so a stale-arch snapshot fails with a clean ``ModelVersionError``
    instead of loading mismatched weights.

    Imports are function-local to avoid an import cycle (state_encoder/features_extractor
    pull in model code).
    """
    from agents.observation.state_encoder import Gen3ObservationEncoder
    from agents.model.features_extractor import Gen3FeaturesExtractor, NET_ARCH

    enc = Gen3ObservationEncoder(mappings)
    ext_kwargs = enc.get_features_extractor_kwargs()
    policy_kwargs = {
        "features_extractor_class": Gen3FeaturesExtractor,
        "features_extractor_kwargs": ext_kwargs,
        "net_arch": NET_ARCH,
    }
    return ModelVersion.from_layout_and_policy_kwargs(ext_kwargs["layout"], policy_kwargs)


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
