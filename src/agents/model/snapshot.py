from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional, Tuple

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
    cli_args: Optional[dict] = None,
    original_command: Optional[str] = None,
) -> None:
    """Write model_config.json and metadata.json into model_dir.

    Does NOT call model.save() — the caller is responsible for the .zip file.
    Safe to call multiple times; files are overwritten in place.
    Preserves any existing snapshot_history and the top-level `latest_eval` block
    (so a checkpoint saved after an eval doesn't erase the eval results).

    Run provenance — `cli_args` (the full argparse namespace, the LATEST process's) and
    `launcher_command` (read from the `LAUNCHER_COMMAND` env the launcher sets) are recorded and
    carried forward across the many overwriting saves, so the exact invocation survives on every
    run, including launcher-managed ones (which don't write a `command.txt`).

    `original_command` is the **immutable** original invocation that CREATED the model — the
    launcher command under a launcher, else this process's `sys.argv`. Unlike `cli_args` (which
    is overwritten with the resuming process's args on every restart), it is written ONCE at model
    creation and then preserved verbatim across all subsequent saves/restarts: the existing value
    always wins. The caller may pass it explicitly; otherwise it is derived here.
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
    existing_cli_args = None
    existing_launcher_command = None
    existing_original_command = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            existing = json.load(f)
            existing_history = existing.get("snapshot_history", {})
            existing_latest_eval = existing.get("latest_eval")
            existing_cli_args = existing.get("cli_args")
            existing_launcher_command = existing.get("launcher_command")
            existing_original_command = existing.get("original_command")

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
    # Run provenance (carried forward like snapshot_history / latest_eval): the full CLI
    # namespace and the launcher's own invocation (the latter from the env it sets).
    cli = cli_args if cli_args is not None else existing_cli_args
    if cli is not None:
        metadata["cli_args"] = cli
    launcher_command = os.environ.get("LAUNCHER_COMMAND") or existing_launcher_command
    if launcher_command:
        metadata["launcher_command"] = launcher_command
    # The original invocation that created the model — IMMUTABLE: existing value wins, so a resume
    # never overwrites it. First write (model creation) derives it from the launcher command (if
    # any) else this process's argv.
    original = existing_original_command or original_command or (
        os.environ.get("LAUNCHER_COMMAND") or " ".join(sys.argv)
    )
    if original:
        metadata["original_command"] = original
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


def append_eval_result_row(
    model_dir: str,
    step: int,
    n_games: int,
    bot_win_rates: dict,
    sentinels: "list[dict] | None" = None,
    bot_td_tails: "dict | None" = None,
    bot_counts: "dict | None" = None,
) -> None:
    """Append one eval cycle's pairwise win-records to ``<model_dir>/eval_results.jsonl``.

    This is the **append-only source of truth** for the ELO / skill-rating fit
    (``agents.training.elo``): each line is the full tournament-matrix row a cycle already
    produced — the trainee (frozen at ``step``) vs every bot and every pool sentinel,
    ``n_games`` each. Append-only (one line per cycle) so it survives launcher restarts,
    unlike the overwritten top-level ``metadata.json:latest_eval``. ``bot_win_rates`` maps
    bot name → trainee win rate; ``sentinels`` is ``[{"step": int, "win_rate": float}, …]``
    (empty/omitted on the non-self-play bot-only path).

    ``bot_counts`` (optional) maps bot name → ``(n_won, n_finished)`` — the EXACT win/loss
    record. Recovering counts from ``win_rate * n_games`` is exact only at full coverage; under
    battle-level work-stealing a crashed shard makes an opponent's win_rate ride over fewer games
    than ``n_games``, so the exact counts are the fidelity-preserving record. Written as the
    additive sibling ``counts`` ({name: [n_won, n_finished]}); old readers ignore it, and it is
    what a future Glicko-2 / TrueSkill backfill (``agents.training.rating``) consumes for an exact
    ladder. Omitted when not supplied, so existing rows/readers stay byte-identical.

    Best-effort: never raise into the eval path — a failed append must not break eval.
    """
    try:
        row = {
            "step": int(step),
            "n_games": int(n_games),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "bots": {k: float(v) for k, v in bot_win_rates.items()},
            "sentinels": [
                {"step": int(s["step"]), "win_rate": float(s["win_rate"])}
                for s in (sentinels or [])
            ],
        }
        # #4 — per-bot TD-residual tail history (append-only, restart-safe). Optional sibling of
        # `bots`; omitted when no captured battles produced residuals, so old rows stay identical.
        if bot_td_tails:
            row["td_resid_tails"] = {k: float(v) for k, v in bot_td_tails.items()}
        # Exact per-opponent W/L (the rating-fidelity record; see docstring). Optional, additive.
        if bot_counts:
            row["counts"] = {k: [int(c[0]), int(c[1])] for k, c in bot_counts.items()}
        with open(os.path.join(model_dir, "eval_results.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"⚠️ [ELO] failed to append eval_results.jsonl row at step {step}: {e}")


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
    eval_block: Optional[dict] = None,
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

    ``eval_block`` — when present, the most-recent eval+pool stats known when the
    checkpoint was saved, stamped under a ``"latest_eval"`` key (mirroring the
    top-level block's name; see ``record_checkpoint``).
    """
    entry = dict(hparams) if hparams else {}
    entry["lr"] = entry["current_lr"] = lr
    entry["n_epochs"] = entry["current_epochs"] = n_epochs
    entry["git_hash"] = git_hash or os.environ.get("LAUNCHER_GIT_HASH") or get_git_hash()
    if handoff_lr is not None:
        entry["handoff_lr"] = handoff_lr
    if eval_block:
        entry["latest_eval"] = eval_block
    return entry


def record_snapshot_in_history(
    model_dir: str,
    checkpoint_name: str,
    lr: float,
    n_epochs: int,
    hparams: Optional[dict] = None,
    git_hash: Optional[str] = None,
    handoff_lr: Optional[float] = None,
    eval_block: Optional[dict] = None,
) -> None:
    """Append or update a checkpoint entry in snapshot_history within metadata.json.

    checkpoint_name: basename of the checkpoint zip, e.g. 'checkpoint_50000000_steps.zip'.
    The entry is built by ``_build_snapshot_entry`` — identical in shape to the
    per-checkpoint sidecar. Creates metadata.json if it doesn't exist yet
    (history-only file until the next save_model_snapshot call fills in the rest).

    eval_block: see ``record_checkpoint`` / ``_build_snapshot_entry``.
    """
    meta_path = os.path.join(model_dir, "metadata.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    history = meta.get("snapshot_history", {})
    history[checkpoint_name] = _build_snapshot_entry(
        lr, n_epochs, hparams, git_hash, handoff_lr, eval_block
    )
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

    Each checkpoint is also stamped (sidecar + snapshot_history) with the
    most-recent eval+pool stats known when it was saved — the current top-level
    ``latest_eval`` block (``_read_latest_eval``), under a ``"latest_eval"`` key.
    The block keeps its own ``step`` (the snapshot it actually evaluated), so
    storing it under a possibly-newer checkpoint never mislabels which weights were
    measured; it is the per-checkpoint "latest eval as of this checkpoint" view.
    The canonical, timing-robust record stays the top-level ``latest_eval``.
    """
    resolved_hash = git_hash or get_git_hash()
    eval_block = _read_latest_eval(model_dir)
    write_checkpoint_metadata(
        checkpoint_path,
        lr,
        n_epochs,
        hparams=hparams,
        git_hash=resolved_hash,
        handoff_lr=handoff_lr,
        eval_block=eval_block,
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
        eval_block=eval_block,
    )


def write_checkpoint_metadata(
    checkpoint_path: str,
    lr: float,
    n_epochs: int,
    hparams: Optional[dict] = None,
    git_hash: Optional[str] = None,
    handoff_lr: Optional[float] = None,
    eval_block: Optional[dict] = None,
) -> None:
    """Write the per-checkpoint metadata sidecar alongside a checkpoint .zip.

    The sidecar's schema is identical to a run-level ``snapshot_history`` entry —
    both are built by ``_build_snapshot_entry``, which emits the union of both
    naming conventions (lr/current_lr, n_epochs/current_epochs) plus git_hash,
    optional handoff_lr, hparams and the ``latest_eval`` stamp — so the per-model
    summary mirrors the history exactly and drops no field.

    checkpoint_path: full path to the .zip (with or without extension). The sidecar
    lands at the same path with .zip replaced by .json.

    handoff_lr / eval_block: see ``record_checkpoint``.
    """
    entry = _build_snapshot_entry(lr, n_epochs, hparams, git_hash, handoff_lr, eval_block)
    with open(_checkpoint_metadata_path(checkpoint_path), "w") as f:
        json.dump(entry, f, indent=2)


def _read_latest_eval(model_dir: str) -> Optional[dict]:
    """Return the current top-level ``latest_eval`` block from ``<model_dir>/metadata.json``.

    ``None`` if the file is missing/corrupt or no eval has run yet. Used by
    ``record_checkpoint`` to stamp each checkpoint with the most-recent eval+pool
    stats as of save time (see its docstring); a thin local reader so ``snapshot``
    has no import dependency on the eval callbacks (which import from here).
    """
    meta_path = os.path.join(model_dir, "metadata.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return None
    block = meta.get("latest_eval")
    return block if isinstance(block, dict) else None


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
    enforce_vf_coef: Optional[float] = None,
    enforce_reward_config=None,
    enforce_value_tail_weight: Optional[float] = None,
    enforce_value_dist: Optional[Tuple[float, float]] = None,
) -> MaskablePPO:
    """Load a model with a compatibility check against the current architecture.

    Args:
        model_path:      Path to the .zip (with or without extension), or a directory
                         containing final_model.zip or best_model.zip.
        env:             VecEnv to attach to the loaded model.
        current_version: ModelVersion reflecting current code; checked against saved config.
        device:          Passed to InstrumentedMaskablePPO.load().
        tensorboard_log: Passed to InstrumentedMaskablePPO.load().
        enforce_vf_coef: TRAINING-RESUME ONLY. When set, the saved config's vf_coef must
                         match this value or a ModelVersionError is raised — vf_coef is fixed
                         for a run's lifetime. Left None on every frozen-snapshot load (eval
                         sentinels, self-play pool opponents, distill teacher, the roundtrip
                         smoke), where the value-loss coefficient is irrelevant to the forward.

    Raises:
        ModelVersionError:  If saved config is incompatible with current_version, or (when
                            enforce_vf_coef is set) its vf_coef differs.
        FileNotFoundError:  If no .zip can be found at the resolved path.
    """
    zip_path, config_dir = _resolve_paths(model_path)

    # model_config.json is run-LEVEL — it sits at the run root, but a checkpoint .zip may
    # live one level down in <run>/checkpoints/. Search the zip's dir then its parent
    # (mirroring load_foreign_opponent) so the arch check survives the relocation.
    config_path = os.path.join(config_dir, "model_config.json")
    if not os.path.exists(config_path):
        parent_config = os.path.join(os.path.dirname(config_dir), "model_config.json")
        if os.path.exists(parent_config):
            config_path = parent_config
    arch_validated = False
    if os.path.exists(config_path):
        saved_version = ModelVersion.from_json_file(config_path)
        current_version.check_compatible(saved_version)
        if enforce_vf_coef is not None:
            saved_version.check_vf_coef(enforce_vf_coef)
        if enforce_reward_config is not None:
            saved_version.check_reward_config(enforce_reward_config)
        if enforce_value_tail_weight is not None:
            saved_version.check_value_tail_weight(enforce_value_tail_weight)
        if enforce_value_dist is not None:
            saved_version.check_value_dist(*enforce_value_dist)
        arch_validated = True
    else:
        print(
            f"[ModelVersion] WARNING: No model_config.json found at {config_dir!r}. "
            "Skipping compatibility check (legacy model)."
        )

    kwargs: dict = {"env": env, "device": device}
    if tensorboard_log:
        kwargs["tensorboard_log"] = tensorboard_log

    # Tolerate a benign TRAINING-ONLY obs-key difference between the saved policy and the live env.
    # The model forward reads ONLY obs["observation"] (everything else — belief_*, win_target/win_mask,
    # win_margin — is a privileged label consumed by the aux loss, never by the network), so an env that
    # declares a training-only key the saved policy predates (e.g. `win_margin`, added mid-run) is safe
    # to resume. But SB3's `check_for_correct_spaces` compares the FULL Dict obs space and FATALs on the
    # extra key. We override the saved spaces with the env's so that check passes — SAFE because
    # `check_compatible` above already pinned the REAL obs (`total_dim`/`arch_signature`) + action space.
    # Gated on `arch_validated` (skip the legacy no-config path, where the strict check is the only guard)
    # and on `env` being attached (frozen-opponent loads pass env=None and want the strict check).
    if env is not None and arch_validated:
        kwargs["custom_objects"] = {
            "observation_space": env.observation_space,
            "action_space": env.action_space,
        }

    return InstrumentedMaskablePPO.load(zip_path, **kwargs)


def load_foreign_opponent(
    model_path: str,
    current_version: ModelVersion,
    device: str = "cpu",
    config_path: Optional[str] = None,
) -> "tuple[MaskablePPO, ModelVersion]":
    """Load a frozen model from ANOTHER run as an inference-only OPPONENT ("stable opponent").

    Unlike ``load_model_snapshot`` — which checks the saved config against the LIVE trainee via
    ``check_compatible`` (a hard FATAL on any ``_WEIGHT_FIELD`` / ``use_popart`` mismatch) — a stable
    cross-run opponent is validated for OBSERVATION-FAMILY compatibility ONLY
    (``ModelVersion.check_opponent_compatible`` = same ``arch_signature``): it never shares weights
    with the trainee and never reads its value head, so ``use_popart`` / ``vf_coef`` / reward-config
    are irrelevant to its forward. Loaded with ``env=None`` (no optimizer, and SB3 skips
    ``check_for_correct_spaces``) for inference only — the opponent builds its own obs via the live
    ``Gen3ObservationEncoder`` and calls ``model.policy.get_distribution``.

    Args:
        model_path:      Path to the opponent's ``.zip`` (with or without extension), or a directory
                         containing ``final_model.zip`` / ``best_model.zip``.
        current_version: ModelVersion reflecting the CURRENT run's code (``current_model_version``);
                         the opponent's ``arch_signature`` must equal it.
        device:          Passed to ``InstrumentedMaskablePPO.load`` (default ``"cpu"`` — an opponent
                         forward is cheap and decouples from the training GPU).
        config_path:     Explicit path to the opponent's ``model_config.json``. When ``None``,
                         it is searched next to the zip then in the parent dir (so a
                         ``best_model/best_model.zip`` finds the run-level config). The resolver
                         in ``agents.training.fixed_opponent_pool`` passes it explicitly.

    Returns:
        ``(model, foreign_version)`` — the loaded model and its parsed ``ModelVersion``.

    Raises:
        ModelVersionError:  if the opponent's ``arch_signature`` (observation family) differs from
                            ``current_version`` — surfaced by the caller as a startup FATAL.
        FileNotFoundError:  if no ``.zip`` resolves, or the sibling ``model_config.json`` is missing
                            (provenance is REQUIRED — never silently load a foreign model blind).
    """
    zip_path, config_dir = _resolve_paths(model_path)

    if config_path is None:
        for d in (config_dir, os.path.dirname(config_dir)):
            cand = os.path.join(d, "model_config.json")
            if os.path.exists(cand):
                config_path = cand
                break
    if config_path is None or not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Stable opponent at {zip_path!r} has no model_config.json (looked next to it in "
            f"{config_dir!r} and its parent). A stable opponent must carry its architecture "
            "provenance so its observation-family compatibility can be verified; refusing to "
            "load it blind."
        )
    foreign_version = ModelVersion.from_json_file(config_path)
    current_version.check_opponent_compatible(foreign_version)

    return InstrumentedMaskablePPO.load(zip_path, env=None, device=device), foreign_version


def current_model_version(
    mappings,
    *,
    attend_unrevealed_opponents: bool = False,
    opp_belief_cls_k: int = 0,
    opp_belief_slots: bool = False,
    value_active_readout: bool = False,
    use_popart: bool = False,
    opp_belief_aux_coef: float = 0.0,
    move_belief_mode: str = "off",
    move_belief_coef: float = 0.0,
    opp_belief_latent: bool = False,
    opp_belief_latent_coef: float = 0.0,
    damage_op: bool = False,
    damage_reattend: bool = False,
    damage_outgoing: bool = False,
    move_candidate_floor: float = 0.0,
    move_latent: bool = False,
    move_belief_latent_coef: float = 0.0,
    spread_belief: bool = False,
    spread_belief_coef: float = 0.0,
    move_prior_fusion: bool = False,
    move_belief_prefuse: bool = False,
    mask_incoming_damage_obs: bool = False,
    mask_active_move_scalars_obs: bool = False,
    mask_move_effects_obs: bool = False,
    win_prob_mode: str = "none",
    win_prob_coef: float = 1.0,
    value_dist_mode: str = "none",
    value_dist_bins: int = 0,
    value_dist_vmin: float = 0.0,
    value_dist_vmax: float = 0.0,
    value_dist_coef: float = 1.0,
    damage_topk_k: int = 0,
    damage_refine_rounds: int = 0,
    damage_matrices_outgoing: bool = False,
    damage_matrices_incoming: bool = False,
    threat_refine_outgoing: bool = False,
    threat_unrevealed_outgoing: bool = False,
    threat_prob_outspeed: bool = False,
    threat_status_refine: bool = False,
    vf_coef: float = 0.5,
    reward_config=None,
    value_tail_weight: float = 0.0,
) -> ModelVersion:
    """Build a ``ModelVersion`` reflecting the CURRENT RUN's architecture for ``mappings``.

    Single source for the ``from_layout_and_policy_kwargs`` construction otherwise
    repeated inline (train_rl_agent's self-play pool setup; the eval worker's sentinel
    version check). Returns the ``current_version`` to pass to ``load_model_snapshot`` /
    ``SnapshotPool`` so a stale-arch snapshot fails with a clean ``ModelVersionError``
    instead of loading mismatched weights.

    **The architecture TOGGLES must be passed in.** They default off (the encoder's
    ``get_features_extractor_kwargs`` carries no CLI toggles), but a run that enables any of them
    (e.g. ``--opp-belief-aux-coef>0`` → ``opp_belief_slots``, ``--use-popart``,
    ``--attend-unrevealed-opponents``) MUST thread its real values here — otherwise the gate compares
    a toggle-OFF "current" version against the run's own toggle-ON snapshots and FATALs on every
    pool/eval/distill load it is meant to protect.

    Imports are function-local to avoid an import cycle (state_encoder/features_extractor
    pull in model code).
    """
    from agents.observation.state_encoder import Gen3ObservationEncoder
    from agents.model.features_extractor import Gen3FeaturesExtractor, NET_ARCH

    enc = Gen3ObservationEncoder(mappings)
    ext_kwargs = enc.get_features_extractor_kwargs()
    ext_kwargs["attend_unrevealed_opponents"] = attend_unrevealed_opponents
    ext_kwargs["opp_belief_cls_k"] = opp_belief_cls_k
    ext_kwargs["opp_belief_slots"] = opp_belief_slots
    ext_kwargs["value_active_readout"] = value_active_readout
    ext_kwargs["move_belief_mode"] = move_belief_mode
    ext_kwargs["opp_belief_latent"] = opp_belief_latent
    ext_kwargs["damage_op"] = damage_op
    ext_kwargs["damage_reattend"] = damage_reattend
    ext_kwargs["damage_outgoing"] = damage_outgoing
    ext_kwargs["move_candidate_floor"] = move_candidate_floor
    ext_kwargs["move_latent"] = move_latent
    ext_kwargs["spread_belief"] = spread_belief
    ext_kwargs["move_prior_fusion"] = move_prior_fusion
    ext_kwargs["move_belief_prefuse"] = move_belief_prefuse
    ext_kwargs["mask_incoming_damage_obs"] = mask_incoming_damage_obs
    ext_kwargs["mask_active_move_scalars_obs"] = mask_active_move_scalars_obs
    ext_kwargs["mask_move_effects_obs"] = mask_move_effects_obs
    ext_kwargs["win_prob_mode"] = win_prob_mode
    ext_kwargs["value_dist_mode"] = value_dist_mode
    ext_kwargs["value_dist_bins"] = value_dist_bins
    ext_kwargs["value_dist_vmin"] = value_dist_vmin
    ext_kwargs["value_dist_vmax"] = value_dist_vmax
    ext_kwargs["damage_topk_k"] = damage_topk_k
    ext_kwargs["damage_refine_rounds"] = damage_refine_rounds
    ext_kwargs["damage_matrices_outgoing"] = damage_matrices_outgoing
    ext_kwargs["damage_matrices_incoming"] = damage_matrices_incoming
    ext_kwargs["threat_refine_outgoing"] = threat_refine_outgoing
    ext_kwargs["threat_unrevealed_outgoing"] = threat_unrevealed_outgoing
    ext_kwargs["threat_prob_outspeed"] = threat_prob_outspeed
    ext_kwargs["threat_status_refine"] = threat_status_refine
    policy_kwargs = {
        "features_extractor_class": Gen3FeaturesExtractor,
        "features_extractor_kwargs": ext_kwargs,
        "net_arch": NET_ARCH,
        "use_popart": use_popart,
    }
    return ModelVersion.from_layout_and_policy_kwargs(
        ext_kwargs["layout"], policy_kwargs, vf_coef=vf_coef, reward_config=reward_config,
        value_tail_weight=value_tail_weight, opp_belief_aux_coef=opp_belief_aux_coef,
        move_belief_coef=move_belief_coef, opp_belief_latent_coef=opp_belief_latent_coef,
        win_prob_coef=win_prob_coef, move_belief_latent_coef=move_belief_latent_coef,
        spread_belief_coef=spread_belief_coef, value_dist_coef=value_dist_coef,
    )


def arch_toggles_from_model(model) -> dict:
    """Extract THIS run's architecture TOGGLES from a live model, JSON-serializable for a worker
    subprocess's cfg. The eval/self-play workers run in separate processes and rebuild a
    ``current_model_version`` to gate sentinel/foreign snapshot loads; without the run's real toggles
    that gate is toggle-OFF and FATALs on the run's own belief-ON / popart / attend-unrevealed
    snapshots. Pass this dict through the worker cfg and splat it into ``current_model_version``."""
    fe = model.policy.features_extractor
    return {
        "attend_unrevealed_opponents": bool(getattr(fe, "attend_unrevealed_opponents", False)),
        "opp_belief_cls_k": int(getattr(fe, "opp_belief_cls_k", 0)),
        "opp_belief_slots": bool(getattr(fe, "opp_belief_slots", False)),
        "value_active_readout": bool(getattr(fe, "value_active_readout", False)),
        "move_belief_mode": str(getattr(fe, "move_belief_mode", "off")),
        "opp_belief_latent": bool(getattr(fe, "opp_belief_latent", False)),
        "damage_op": bool(getattr(fe, "damage_op_enabled", False)),
        "damage_reattend": bool(getattr(fe, "damage_reattend_enabled", False)),
        "damage_outgoing": bool(getattr(fe, "damage_outgoing", False)),
        "move_candidate_floor": float(getattr(fe, "move_candidate_floor", 0.0)),
        "move_latent": bool(getattr(fe, "move_latent", False)),
        "spread_belief": bool(getattr(fe, "spread_belief_enabled", False)),
        "move_prior_fusion": bool(getattr(fe, "move_prior_fusion", False)),
        "move_belief_prefuse": bool(getattr(fe, "move_belief_prefuse", False)),
        "mask_incoming_damage_obs": bool(getattr(fe, "mask_incoming_damage_obs", False)),
        "mask_active_move_scalars_obs": bool(getattr(fe, "mask_active_move_scalars_obs", False)),
        "mask_move_effects_obs": bool(getattr(fe, "mask_move_effects_obs", False)),
        "win_prob_mode": str(getattr(fe, "win_prob_mode", "none")),
        # v29 value-dist head: only the check_compatible-gated structural toggles (mode + atom count) —
        # the support (vmin/vmax) is resume-only-checked on the trainer, never by a worker's load gate.
        "value_dist_mode": str(getattr(fe, "value_dist_mode", "none")),
        "value_dist_bins": int(getattr(fe, "value_dist_bins", 0)),
        # gen3_unified_topk_incoming_v1 (v30): the top-K incoming block's K (0 = off) — STRUCTURAL int,
        # gated in check_compatible (it scales the projection widths), so it must reach the worker's gate.
        "damage_topk_k": int(getattr(fe, "damage_topk_k", 0)),
        # gen3_iterative_damage_v1 (v31): the iterative-refinement round count (0 = off) — STRUCTURAL int,
        # gated in check_compatible (0↔N a state_dict change, N↔M a forward change), so it must reach the gate.
        "damage_refine_rounds": int(getattr(fe, "damage_refine_rounds", 0)),
        # gen3_per_move_matrices_v1 (v32): the outgoing per-move damage matrix — STRUCTURAL bool (widens the
        # op out_dim), gated in check_compatible, so it must reach the worker's gate.
        "damage_matrices_outgoing": bool(getattr(fe, "damage_matrices_outgoing", False)),
        # gen3_per_move_matrices_v1 (v33): the incoming per-move damage matrix — STRUCTURAL bool, gated.
        "damage_matrices_incoming": bool(getattr(fe, "damage_matrices_incoming", False)),
        # gen3_bidir_threat_trunk_v1 (v36): the bidirectional in-trunk threat field — threat_refine_outgoing
        # is STRUCTURAL (adds outgoing_proj), the other two forward-behavior; all version-gated, so all must
        # reach the worker's gate (else a threat-ON self-play snapshot FATALs the toggle-OFF "current").
        "threat_refine_outgoing": bool(getattr(fe, "threat_refine_outgoing", False)),
        "threat_unrevealed_outgoing": bool(getattr(fe, "threat_unrevealed_outgoing", False)),
        "threat_prob_outspeed": bool(getattr(fe, "threat_prob_outspeed", False)),
        # gen3_status_trunk_v1 (v37): STRUCTURAL bool (adds status projections), gated → must reach the gate.
        "threat_status_refine": bool(getattr(fe, "threat_status_refine", False)),
        "use_popart": getattr(model.policy, "popart", None) is not None,
    }


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
