from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Optional, Tuple

import stable_baselines3
from sb3_contrib import MaskablePPO

from agents.model.damage_tables import _PRIOR_FLOOR, sanitize_historical_move_floor
from agents.model.model_version import ModelVersion, ModelVersionError
from agents.model.team_signature import TEAM_SIGNATURE_DIM as _TEAM_SIG_DIM
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
    existing_matchup_history = []
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            existing = json.load(f)
            existing_history = existing.get("snapshot_history", {})
            existing_latest_eval = existing.get("latest_eval")
            existing_cli_args = existing.get("cli_args")
            existing_launcher_command = existing.get("launcher_command")
            existing_original_command = existing.get("original_command")
            existing_matchup_history = existing.get("matchup_history", [])

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
    # MATCHUP ERA HISTORY (append-only): `cli_args` records only the LATEST process's declared
    # matchup — a manual resume that changes the matchup (a new --trainee-team / --exploiter /
    # --bot-weights) would silently overwrite what earlier eras trained against. Each era's
    # declared spec is appended here ONCE (on hash change), so the run's full training-regime
    # timeline survives every restart. The current era's `_matchup_spec{,_hash}` ride in via
    # `cli_args` (stamped by train_rl_agent); saves without cli_args (the per-checkpoint path)
    # preserve the history untouched.
    matchup_history = list(existing_matchup_history)
    _m_hash = (cli or {}).get("_matchup_spec_hash")
    if _m_hash and (not matchup_history or matchup_history[-1].get("hash") != _m_hash):
        matchup_history.append({
            "hash": _m_hash,
            "spec": (cli or {}).get("_matchup_spec"),
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        })
    if matchup_history:
        metadata["matchup_history"] = matchup_history
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
    externals: "dict | None" = None,
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

    ``externals`` (optional) maps a stable/exploiter opponent label (``ext_*``) →
    ``{"win_rate": float, "counts": [n_won, n_finished]}`` — the per-cycle vs-target record
    (e.g. the exploiter VERDICT metric), which previously lived only in the OVERWRITTEN
    ``latest_eval`` block + TensorBoard. Kept in its own sibling (never inside ``bots``) so the
    ELO fit's ladder is untouched. Each row is also stamped with the run's CURRENT declared
    ``matchup_hash`` (the measurement-regime tag, read from the run metadata) so rows from
    different regimes/eras — e.g. the OOD-eval era vs post-fix — are distinguishable IN-FILE
    instead of by out-of-band dates. Both additive; old readers ignore them.

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
        if externals:
            row["externals"] = {
                k: {"win_rate": float(v.get("win_rate", 0.0)),
                    **({"counts": [int(v["counts"][0]), int(v["counts"][1])]}
                       if v.get("counts") else {})}
                for k, v in externals.items()}
        m_hash = _read_matchup_hash(model_dir)
        if m_hash:
            row["matchup_hash"] = m_hash
        with open(os.path.join(model_dir, "eval_results.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")
    except (OSError, ValueError, KeyError, TypeError) as e:
        print(f"⚠️ [ELO] failed to append eval_results.jsonl row at step {step}: {e}")


def _read_matchup_hash(model_dir: str) -> "str | None":
    """The CURRENT-era declared-matchup hash (the measurement-regime tag) from the run's
    metadata — `cli_args._matchup_spec_hash` (the latest process's declaration), falling back to
    the last `matchup_history` era. `None` for pre-MatchupSpec runs / missing metadata. Cheap
    best-effort reader used to stamp eval rows / manifests / checkpoint sidecars so each record
    is self-describing about the regime it was produced under."""
    meta_path = os.path.join(model_dir, "metadata.json")
    if not os.path.exists(meta_path):
        return None
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (OSError, ValueError):
        return None
    h = (meta.get("cli_args") or {}).get("_matchup_spec_hash")
    if h:
        return h
    hist = meta.get("matchup_history") or []
    return hist[-1].get("hash") if hist else None


def read_recorded_matchup(model_path: str) -> "tuple[str | None, dict | None]":
    """The (hash, spec) a resumed checkpoint's RUN last declared — for the resume drift guard.

    `model_path` is the --model argument (a checkpoint .zip or a run dir); the run metadata is
    searched next to the zip and one level up (the `load_model_snapshot` convention, covering
    `<run>/checkpoints/<name>.zip`). Returns `(None, None)` for pre-MatchupSpec runs."""
    apath = os.path.abspath(model_path)
    dirs = [apath] if os.path.isdir(apath) else [os.path.dirname(apath)]
    dirs.append(os.path.dirname(dirs[0]))
    for d in dirs:
        meta_path = os.path.join(d, "metadata.json")
        if not os.path.exists(meta_path):
            continue
        try:
            with open(meta_path) as f:
                meta = json.load(f)
        except (OSError, ValueError):
            continue
        cli = meta.get("cli_args") or {}
        if cli.get("_matchup_spec_hash"):
            return cli["_matchup_spec_hash"], cli.get("_matchup_spec")
        hist = meta.get("matchup_history") or []
        if hist:
            return hist[-1].get("hash"), hist[-1].get("spec")
    return None, None


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
    matchup_hash: Optional[str] = None,
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
    # The declared-matchup regime tag AS OF this checkpoint (like the latest_eval stamp): each
    # checkpoint is self-describing about what it was training against, robust to a later era
    # overwriting the run-level cli_args.
    if matchup_hash:
        entry["matchup_hash"] = matchup_hash
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
    matchup_hash: Optional[str] = None,
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
        lr, n_epochs, hparams, git_hash, handoff_lr, eval_block, matchup_hash
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
    matchup_hash = _read_matchup_hash(model_dir)
    write_checkpoint_metadata(
        checkpoint_path,
        lr,
        n_epochs,
        hparams=hparams,
        git_hash=resolved_hash,
        handoff_lr=handoff_lr,
        eval_block=eval_block,
        matchup_hash=matchup_hash,
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
        matchup_hash=matchup_hash,
    )


def write_checkpoint_metadata(
    checkpoint_path: str,
    lr: float,
    n_epochs: int,
    hparams: Optional[dict] = None,
    git_hash: Optional[str] = None,
    handoff_lr: Optional[float] = None,
    eval_block: Optional[dict] = None,
    matchup_hash: Optional[str] = None,
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
    entry = _build_snapshot_entry(lr, n_epochs, hparams, git_hash, handoff_lr, eval_block,
                                  matchup_hash)
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
    enforce_value_seed_vicreg_coef: Optional[float] = None,
    enforce_reward_config=None,
    enforce_value_tail_weight: Optional[float] = None,
    enforce_value_dist: Optional[Tuple[float, float]] = None,
    enforce_belief_grad_mode: Optional[str] = None,
    allow_belief_grad_mode_change: bool = False,
    enforce_value_from_dist: Optional[bool] = None,
    allow_value_from_dist_change: bool = False,
    allow_zarch_lut_add: bool = False,
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
        gate_version = current_version
        if (allow_zarch_lut_add and saved_version.zarch_lut == "off"
                and current_version.zarch_lut != "off"):
            # gen3_zarch_lut_v1 EXPLOITER FORK: adding the per-team LUT to a LUT-less generalist
            # checkpoint is the ONLY way this feature is ever used — an exploiter always warm-forks
            # (0.84 @2M forked vs ~0.65 @20M from scratch), and no generalist carries a LUT. The
            # loaded policy is rebuilt from the ZIP's saved (LUT-off) policy_kwargs, so the state_dict
            # still matches exactly; the caller attaches the freshly-initialized LUT modules
            # post-load (`attach_zarch_lut`), the belief_grad_mode/value_from_dist migration pattern.
            # ONLY an ADD is allowed here — a mode flip or a removal still FATALs.
            import dataclasses as _dc
            gate_version = _dc.replace(current_version, zarch_lut="off", zarch_lut_teams=0)
            print("[ModelVersion] NOTE: adding the per-team LUT (--zarch-lut "
                  f"{current_version.zarch_lut}, {current_version.zarch_lut_teams} teams) to a "
                  "LUT-less checkpoint — the LUT modules start freshly initialized, everything else "
                  "warm-starts from the checkpoint.")
        gate_version.check_compatible(saved_version)
        if enforce_vf_coef is not None:
            saved_version.check_vf_coef(enforce_vf_coef)
        if enforce_value_seed_vicreg_coef is not None:
            # TRAINING-RESUME ONLY (the vf_coef pattern): the seed-VICReg coefficient is fixed for
            # a run's lifetime — silently toggling it on resume would drift the critic-readout
            # objective. Frozen-opponent loads leave this None (their forward never runs the loss).
            saved_version.check_value_seed_vicreg(enforce_value_seed_vicreg_coef)
        if enforce_reward_config is not None:
            saved_version.check_reward_config(enforce_reward_config)
        if enforce_value_tail_weight is not None:
            saved_version.check_value_tail_weight(enforce_value_tail_weight)
        if enforce_value_dist is not None:
            saved_version.check_value_dist(*enforce_value_dist)
        if enforce_belief_grad_mode is not None:
            saved_version.check_belief_grad_mode(enforce_belief_grad_mode,
                                                 allow_change=allow_belief_grad_mode_change)
        if enforce_value_from_dist is not None:
            saved_version.check_value_from_dist(enforce_value_from_dist,
                                                allow_change=allow_value_from_dist_change)
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
    _patch_historical_floor(zip_path, kwargs)

    return InstrumentedMaskablePPO.load(zip_path, **kwargs)


# Shared Inductor cache. Under `spawn` every worker re-imports and re-traces from scratch, but a
# SHARED on-disk cache turns all but the first process's CODEGEN into a hit (measured 19.1s cold ->
# 5.8s warm). This is the only compile artifact that crosses a process boundary: `torch.compile`
# returns a live Python object, so the compiled callable itself can never be handed to a spawned
# child — see `prewarm_extractor_compile`.
DEFAULT_INDUCTOR_CACHE_DIR = "/tmp/gen3ai_inductor_cache"

# A compile must beat eager by at least this much to be kept. Not a safety margin against noise so
# much as a floor on "worth the risk at all": anything under it means dynamo overhead is eating the
# fusion win, which is what a partially-traced graph looks like.
_MIN_COMPILE_SPEEDUP = 1.05
_TIMING_REPS = 12
_TIMING_WARMUP = 3

# Set once a compile has been MEASURED to pay off in this process. The validation answers "does this
# extractor's code object compile to something faster?", and `torch.compile` keys on exactly that
# code object — so the answer cannot differ for a second model in the same process. Consumers that
# load models in a LOOP (the search-teacher worker rebuilds its opponent every iteration; an eval
# worker walks several opponents) would otherwise re-pay ~15 eager forwards each time for an answer
# they already have. Deliberately process-local: a fresh process re-validates, because that is where
# a genuinely different outcome (a cold cache, a failing backend) could show up.
_COMPILE_VALIDATED = False


def _compile_warn(msg: str) -> None:
    """Report a compile problem LOUDLY.

    Falling back to eager is a ~6.5x regression on the opponent forward and a measured ~24% loss of
    end-to-end training throughput, but it is invisible — the run simply produces fewer steps per
    hour, forever, and looks healthy. So a failure goes to stderr AND (under the launcher) into the
    event stream, where it surfaces in the TUI rather than scrolling past in a worker's stdout."""
    line = f"⚠️ [CompileExtractor] {msg}"
    print(line, file=sys.stderr, flush=True)
    try:
        from main.launcher.ipc import emit
        emit(line)
    except Exception:
        pass                                          # standalone / no launcher pipe: stderr is enough


def _inductor_cache_dir() -> str:
    """Set (once) and return the shared Inductor cache dir. Deliberately NOT an import-time side
    effect — `snapshot.py` is imported by the prober, eval workers and offline tooling that never
    compile anything, and a module that mutates the environment on import is the kind of thing that
    is impossible to reason about later. Inductor reads this when it first codegens, which is always
    inside `maybe_compile_extractor`, so setting it here is early enough."""
    return os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", DEFAULT_INDUCTOR_CACHE_DIR)


class CompileExtractorError(RuntimeError):
    """Raised under `--compile-opponents-strict` when a compile does not deliver its speedup."""


def maybe_compile_extractor(model, enabled: bool, label: str = "opponent",
                            hide_cuda: bool = False, strict: bool = False) -> bool:
    """`torch.compile` a frozen model's FEATURE EXTRACTOR for CPU inference. Returns True if applied.

    WHY THE EXTRACTOR AND NOT THE OP. The 2026-06-30 attempt compiled only `DamageOperator.forward`
    inside `policy.get_distribution` and measured **0.70× (slower)** — dynamo overhead on a graph that
    still ran ~10k eager dispatches around it. Compiling the WHOLE extractor instead gives one fused
    graph: measured **6.5× at B=1 on CPU** on the literal production config (6.37 -> 0.98 ms), 1 graph
    / 0 graph breaks, values within 5.1e-7 of eager.

    THREE PROPERTIES THAT MAKE THIS CHEAP (all measured, `tmp/compile_share_probe.py`):
      * the compile is keyed on the CODE OBJECT, not the module instance — a second extractor built in
        the same process compiles in **0.00s**, so every later pool snapshot is free;
      * parameters are graph INPUTS, not baked constants — `load_state_dict` of a different checkpoint
        does NOT recompile (dynamo frame count unchanged);
      * we patch the BOUND `fe.forward`, never the module — `torch.compile(module)` would prefix every
        state_dict key with `_orig_mod.` and break resume/load.

    `hide_cuda` MUST be True in an env worker and False in the learner. Compiling even a CPU-device
    model inside a CUDA-VISIBLE process INITIALISES CUDA and takes ~252 MiB of card; ×48 workers is
    ~12 GB and is exactly the OOM that killed the June attempt. This used to be INFERRED from
    `torch.cuda.is_initialized()` as a proxy for "am I a worker" — which was correct only by accident
    of the call sites, and would have silently blinded the learner's GPU the first time anyone called
    this from the main process before CUDA was touched. It is now the caller's explicit declaration.

    NOTE ON `suppress_errors`: this deliberately does NOT set it. It used to, because ONE op
    crashed Inductor codegen (`BeliefHead.species_posterior`, now fixed) — and globally
    suppressing backend errors to work
    around it meant every OTHER compile failure also became a silent per-frame eager fallback. A
    failure here should be loud, caught, and reported. Late failures are handled by
    `_eager_fallback_on_error`, which degrades that one model instead of the whole process.
    """
    if not enabled:
        return False
    import torch

    fe = getattr(getattr(model, "policy", None), "features_extractor", None)
    if fe is None:
        return False
    if hide_cuda:
        if torch.cuda.is_initialized():
            # Refuse rather than pretend: the context already exists, so hiding the device now buys
            # nothing and the caller has mis-declared which process it is in.
            msg = (f"{label}: DISABLED — hide_cuda=True but this process has already initialised "
                   f"CUDA; refusing to compile (it would add a ~252 MiB context per worker).")
            _compile_warn(msg)
            if strict:
                raise CompileExtractorError(msg)
            return False
        os.environ["CUDA_VISIBLE_DEVICES"] = ""       # no per-worker CUDA context (the June OOM)
    cache_dir = _inductor_cache_dir()
    if hasattr(fe, "disable_observation_debugger"):
        fe.disable_observation_debugger()             # numpy asserts inside forward; dynamo can't trace

    global _COMPILE_VALIDATED
    revalidate = not _COMPILE_VALIDATED
    original = fe.forward
    warmup = _compile_warmup_obs(fe)
    try:
        eager_ms = _time_forward(fe.forward, warmup) if revalidate else 0.0
        compiled = torch.compile(fe.forward)
        # Force the compile HERE, inside the try — `torch.compile` is lazy, so without this the real
        # compilation happens on the first live decision, far outside this handler.
        with torch.no_grad():
            compiled(warmup)
        comp_ms = _time_forward(compiled, warmup) if revalidate else 0.0
    except Exception as e:                            # by default never take a run down for a perf knob
        fe.forward = original
        msg = f"{label}: DISABLED — {type(e).__name__}: {str(e)[:200]}"
        _compile_warn(msg)
        if strict:
            raise CompileExtractorError(msg) from e
        return False

    if not revalidate:
        # Already proven in this process — keep it without re-timing. STILL LOG IT: a silent success
        # is indistinguishable from "never ran" in a run log, and that is not hypothetical — the
        # eval-worker opponent compile looked missing for exactly this reason until it was verified
        # by instrumenting the call. Coverage you cannot see is coverage you will doubt.
        fe.forward = _eager_fallback_on_error(compiled, original, label)
        print(f"[CompileExtractor] {label}: ON (reused this process's validated compile)", flush=True)
        return True

    speedup = eager_ms / comp_ms if comp_ms > 0 else 0.0
    if speedup < _MIN_COMPILE_SPEEDUP:
        # Compiling can LOSE: the June attempt measured 0.70× because dynamo overhead exceeded the
        # fusion win on a fragmented graph. Measure, then keep or revert — never assume.
        fe.forward = original
        msg = (f"{label}: REVERTED to eager — compiled {comp_ms:.2f} ms vs eager {eager_ms:.2f} ms "
               f"({speedup:.2f}x) is below the {_MIN_COMPILE_SPEEDUP:.2f}x floor; the graph is "
               f"probably fragmented. Expect roughly a {1.0 / max(speedup, 1e-9):.1f}x slower "
               f"opponent forward than a healthy compile gives.")
        _compile_warn(msg)
        if strict:
            raise CompileExtractorError(msg)
        return False

    fe.forward = _eager_fallback_on_error(compiled, original, label)
    _COMPILE_VALIDATED = True
    print(f"[CompileExtractor] {label}: ON — {eager_ms:.2f} -> {comp_ms:.2f} ms "
          f"({speedup:.1f}x, cache {cache_dir})", flush=True)
    return True


def _eager_fallback_on_error(compiled, original, label: str):
    """Wrap the compiled callable so it degrades to eager instead of killing the caller.

    TWO things it guards:

    1. A LATE compile failure. `torch.compile` guards on input properties, so a shape/dtype it has
       not seen can trigger a fresh trace at CALL time — long after the load-time try/except
       returned. Opponent inference is always B=1 so that should not happen, but "should not" is not
       a guarantee worth a crashed 3-hour run. This is the targeted replacement for the old global
       `suppress_errors=True`: same never-crash property, scoped to ONE model, and it SAYS so
       instead of silently running eager while claiming to be compiled.

    2. GRAD-ENABLED calls. The compiled artifact here is built for INFERENCE. Under `requires_grad`
       dynamo hands the graph to AOTAutograd, which must also lower the BACKWARD — and Inductor's
       CPU backward codegen fails on this model's scatter/`index_add` (the HP-type belief); that is
       the documented reason the June `--compile-damage-op` integration was inference-only. Every
       frozen-opponent consumer runs under `no_grad`, but the PROBER does not: gradient saliency
       backprops through this same extractor. So route grad-enabled calls to eager. Value-identical
       either way, and it keeps `maybe_compile_extractor` safe to apply to any non-training model
       rather than only the ones we have audited for no_grad."""
    import torch

    state = {"failed": False}

    def guarded(obs):
        if state["failed"] or torch.is_grad_enabled():
            return original(obs)
        try:
            return compiled(obs)
        except Exception as e:
            state["failed"] = True
            _compile_warn(f"{label}: FELL BACK to eager (this model is now ~6x slower) — "
                          f"{type(e).__name__}: {str(e)[:200]}")
            return original(obs)

    return guarded


def _time_forward(fn, obs, reps: int = _TIMING_REPS) -> float:
    """min-of-N ms for one forward. min, not mean: contention only ever ADDS time, and this runs at
    worker startup while other workers are still spawning."""
    import time
    import torch
    with torch.no_grad():
        for _ in range(_TIMING_WARMUP):
            fn(obs)
        best = float("inf")
        for _ in range(reps):
            t0 = time.perf_counter()
            fn(obs)
            best = min(best, time.perf_counter() - t0)
    return best * 1e3


def _compile_warmup_obs(fe) -> dict:
    """A zero observation of the right width — enough to force compilation of the B=1 graph."""
    import torch
    layout = getattr(fe, "layout", None)
    dim = layout["total_dim"] if layout else fe.observation_space.shape[0]
    return {"observation": torch.zeros(1, dim)}


# Extractor kwargs that `_migrate_config` strips from `model_config.json` but which are ALSO
# pickled into every checkpoint zip's `policy_kwargs["features_extractor_kwargs"]`. SB3 rebuilds the
# extractor from the ZIP, not from the config, so migrating only the config leaves the zip carrying
# arguments `Gen3FeaturesExtractor.__init__` no longer accepts -> `TypeError: got an unexpected
# keyword argument`. That breaks every READ path (prober, ELO, offline probes, frozen pool
# opponents, eval workers) and the training-RESUME path alike.
#
# Split in two because the two migrations have different SAFETY properties, and collapsing them
# would silently undo the v71 refusal:
#   * INERT (v70) — the refine loop was unreachable in production (0 rounds, and mutually exclusive
#     with the shipped prefuse placement), so these selected nothing. Pop unconditionally.
#   * JUDGED (v71) — `move_belief_prefuse` was a pure FORWARD-BEHAVIOR toggle with a byte-identical
#     state_dict, so popping a non-supported value would let a post-ordering checkpoint load into
#     the pre-ordering forward and be quietly wrong forever, with no shape check anywhere to catch
#     it. Mirror `_migrate_config`'s v71 rule exactly: refuse loudly, never default.
# Agreement with `_migrate_config` is pinned by `dead_kwargs_sanitize_test.py` rather than by
# sharing a constant, so the two cannot drift apart unnoticed.
_DEAD_FEK_INERT = (
    "damage_refine_rounds", "threat_refine_outgoing", "threat_unrevealed_outgoing",
    "threat_status_refine", "move_belief_single_compute",
)
_DEAD_FEK_JUDGED = (("move_belief_prefuse", True), ("damage_op_prefuse", True),
                    ("damage_reattend", False),
                    # v75: the SimSiam latent-belief predictor is deleted. True is REFUSED because
                    # it put parameters in the state_dict; False pops silently (nothing built).
                    ("opp_belief_latent", False))


def sanitize_dead_extractor_kwargs(fek: dict) -> bool:
    """Drop v70/v71/v75-deleted keys from a saved `features_extractor_kwargs`. True if it changed.

    Raises `ModelVersionError` — exactly as `_migrate_config` does — when a JUDGED field records a
    value the surviving forward pass cannot reproduce.
    """
    changed = False
    for dead, supported in _DEAD_FEK_JUDGED:
        if dead in fek:
            if bool(fek[dead]) is not supported:
                raise ModelVersionError(
                    f"{dead}={fek[dead]!r} is no longer supported (gen3_tiered_pipeline_v1): the "
                    f"only supported value is {supported}.\nThis checkpoint trained under a forward "
                    "pass that no longer exists in the codebase and cannot be reproduced from HEAD."
                )
            fek.pop(dead)
            changed = True
    for dead in _DEAD_FEK_INERT:
        if dead in fek:
            fek.pop(dead)
            changed = True
    return changed


def _patch_historical_floor(zip_path: str, kwargs: dict) -> None:
    """Sanitize a saved `policy_kwargs` so an older checkpoint can be RECONSTRUCTED.

    Two independent fixes share this hook because both are "the zip records something the live
    constructor no longer accepts", and both are read from the same one zip read:

    1. `move_candidate_floor` (below).
    2. Extractor kwargs deleted at config v70/v71/v75 (`sanitize_dead_extractor_kwargs`).

    --- 1. Let a PRE-v65 checkpoint be RECONSTRUCTED, without loosening the resume gate.

    Before `gen3_unconditional_move_legality_v1`, `move_candidate_floor: 0.0` was how a config said
    "legality gate OFF" — the value was a SWITCH, not a probability. v65 gave it a validated range,
    so every checkpoint saved before that now raises inside `build_move_prior_logits` the moment SB3
    rebuilds the extractor from the saved `policy_kwargs`. That is a hard stop for *reading* old
    models: the prober, the offline probes, ELO ladders, and frozen self-play opponents drawn from
    an older pool all reconstruct an extractor and none of them are resuming training.

    Sanitising HERE is safe because it does not touch the safety property. The resume guard lives in
    `ModelVersion.check_compatible`, which compares the SAVED config against the live one and already
    special-cases `saved == 0.0` as "predates v65" — it runs before this and is unaffected. So a
    training resume still FATALs, while a read-only load succeeds. Doing it in `_migrate_config`
    instead would conflate the two and silently let a resume adopt a different prior.
    """
    try:
        from stable_baselines3.common.save_util import load_from_zip_file
        data, _, _ = load_from_zip_file(zip_path, device="cpu", print_system_info=False)
    except Exception:
        return                                  # unreadable here → let SB3's own load report it
    pk = (data or {}).get("policy_kwargs") or {}
    fek = pk.get("features_extractor_kwargs")
    if not isinstance(fek, dict):
        return
    changed = False
    if "move_candidate_floor" in fek:
        before = fek["move_candidate_floor"]
        sanitize_historical_move_floor(fek)
        changed = fek["move_candidate_floor"] != before
    # `|=` not `or`: short-circuiting would skip the dead-kwarg strip whenever the floor already
    # needed patching, which is precisely the pre-v65 checkpoints that ALSO carry the dead keys.
    changed |= sanitize_dead_extractor_kwargs(fek)
    if changed:
        kwargs.setdefault("custom_objects", {})["policy_kwargs"] = pk


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

    load_kwargs: dict = {"env": None, "device": device}
    _patch_historical_floor(zip_path, load_kwargs)
    return InstrumentedMaskablePPO.load(zip_path, **load_kwargs), foreign_version


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
    damage_op: bool = False,
    damage_outgoing: bool = False,
    move_candidate_floor: float = _PRIOR_FLOOR,
    move_latent: bool = False,
    move_belief_latent_coef: float = 0.0,
    spread_belief: bool = False,
    spread_belief_nature: bool = False,
    spread_belief_coef: float = 0.0,
    move_prior_fusion: bool = False,
    damage_candidate_k: int = 0,
    entity_topk_seats: int = 0,
    consequence_topk: int = 6,
    edge_bias_families: str = "off",
    entity_tail_seats: bool = False,
    win_prob_mode: str = "none",
    win_prob_coef: float = 1.0,
    value_dist_mode: str = "none",
    value_dist_bins: int = 0,
    value_dist_vmin: float = 0.0,
    value_dist_vmax: float = 0.0,
    value_dist_coef: float = 1.0,
    seed_quantile: bool = False,
    value_threat_inject: bool = False,
    opp_intent: bool = False,
    species_prior_fusion: bool = False,
    t0_species_prior: bool = False,
    opp_intent_grad_mode: str = "detached",
    intent_value_reduce: bool = False,
    intent_move_cell: bool = False,
    damage_topk_k: int = 0,
    damage_matrices_outgoing: bool = False,
    damage_matrices_incoming: bool = False,
    damage_matrices_outgoing_all: bool = False,
    threat_prob_outspeed: bool = False,
    hp_type_belief_coef: float = 0.0,
    hp_belief_mode: str = "composed",
    belief_grad_mode: str = "shaping",
    pubval_mode: str = "none",
    pubval_coef: float = 0.0,
    zarch_film: str = "off",
    zarch_dim: int = 0,
    zarch_lut: str = "off",
    zarch_lut_rosters=None,
    zarch_lut_init_std: float = 1.0,
    zarch_recon_coef: float = 0.0,
    zarch_vicreg_coef: float = 0.0,
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
    ext_kwargs["damage_op"] = damage_op
    ext_kwargs["damage_outgoing"] = damage_outgoing
    ext_kwargs["move_candidate_floor"] = move_candidate_floor
    ext_kwargs["move_latent"] = move_latent
    ext_kwargs["spread_belief"] = spread_belief
    ext_kwargs["spread_belief_nature"] = spread_belief_nature
    ext_kwargs["move_prior_fusion"] = move_prior_fusion
    ext_kwargs["damage_candidate_k"] = damage_candidate_k
    ext_kwargs["entity_topk_seats"] = entity_topk_seats
    ext_kwargs["consequence_topk"] = consequence_topk
    ext_kwargs["zarch_lut_init_std"] = zarch_lut_init_std
    ext_kwargs["edge_bias_families"] = edge_bias_families
    ext_kwargs["entity_tail_seats"] = entity_tail_seats
    ext_kwargs["win_prob_mode"] = win_prob_mode
    ext_kwargs["value_dist_mode"] = value_dist_mode
    ext_kwargs["value_dist_bins"] = value_dist_bins
    ext_kwargs["value_dist_vmin"] = value_dist_vmin
    ext_kwargs["value_dist_vmax"] = value_dist_vmax
    ext_kwargs["seed_quantile"] = seed_quantile
    ext_kwargs["value_threat_inject"] = value_threat_inject
    ext_kwargs["opp_intent"] = opp_intent
    ext_kwargs["species_prior_fusion"] = species_prior_fusion
    ext_kwargs["t0_species_prior"] = t0_species_prior
    ext_kwargs["opp_intent_grad_mode"] = opp_intent_grad_mode
    ext_kwargs["intent_value_reduce"] = intent_value_reduce
    ext_kwargs["intent_move_cell"] = intent_move_cell
    ext_kwargs["damage_topk_k"] = damage_topk_k
    ext_kwargs["damage_matrices_outgoing"] = damage_matrices_outgoing
    ext_kwargs["damage_matrices_incoming"] = damage_matrices_incoming
    ext_kwargs["damage_matrices_outgoing_all"] = damage_matrices_outgoing_all
    ext_kwargs["threat_prob_outspeed"] = threat_prob_outspeed
    ext_kwargs["hp_belief_mode"] = hp_belief_mode
    ext_kwargs["belief_grad_mode"] = belief_grad_mode
    ext_kwargs["pubval_mode"] = pubval_mode
    ext_kwargs["zarch_film"] = zarch_film
    ext_kwargs["zarch_dim"] = zarch_dim
    ext_kwargs["zarch_lut"] = zarch_lut
    ext_kwargs["zarch_lut_rosters"] = zarch_lut_rosters
    policy_kwargs = {
        "features_extractor_class": Gen3FeaturesExtractor,
        "features_extractor_kwargs": ext_kwargs,
        "net_arch": NET_ARCH,
        "use_popart": use_popart,
    }
    return ModelVersion.from_layout_and_policy_kwargs(
        ext_kwargs["layout"], policy_kwargs, vf_coef=vf_coef, reward_config=reward_config,
        value_tail_weight=value_tail_weight, opp_belief_aux_coef=opp_belief_aux_coef,
        move_belief_coef=move_belief_coef,
        win_prob_coef=win_prob_coef, move_belief_latent_coef=move_belief_latent_coef,
        spread_belief_coef=spread_belief_coef, value_dist_coef=value_dist_coef,
        hp_type_belief_coef=hp_type_belief_coef, pubval_coef=pubval_coef,
        zarch_recon_coef=zarch_recon_coef, zarch_vicreg_coef=zarch_vicreg_coef,
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
        "damage_op": bool(getattr(fe, "damage_op_enabled", False)),
        "damage_outgoing": bool(getattr(fe, "damage_outgoing", False)),
        "move_candidate_floor": float(getattr(fe, "move_candidate_floor", _PRIOR_FLOOR)),
        "move_latent": bool(getattr(fe, "move_latent", False)),
        "spread_belief": bool(getattr(fe, "spread_belief_enabled", False)),
        "spread_belief_nature": bool(getattr(fe, "spread_belief_nature", False)),
        "move_prior_fusion": bool(getattr(fe, "move_prior_fusion", False)),
        "damage_candidate_k": int(getattr(fe, "damage_candidate_k", 0)),
        # v51 gen3_pointer_native_v1: no pointer toggle — the pointer head is unconditional (the
        # cross-era break rides ARCH_SIGNATURE, not a kwarg).
        # v71 gen3_tiered_pipeline_v1: no prefuse/reattend toggles either — the PRE-transformer
        # placement is unconditional, so there is nothing for a worker to gate on.
        # v54 gen3_entity_move_seats_v1: STRUCTURAL int (threat_seat_proj + seat count), gated in
        # check_compatible — must reach the worker's gate.
        "entity_topk_seats": int(getattr(fe, "entity_topk_seats", 0)),
        "consequence_topk": int(getattr(fe, "consequence_topk", 6)),
        # v56 gen3_edge_bias_trunk_v1: STRUCTURAL str (per-family bias maps + attention biases),
        # gated in check_compatible — must reach the worker's gate.
        "edge_bias_families": str(getattr(fe, "edge_bias_families", "off")),
        "entity_tail_seats": bool(getattr(fe, "entity_tail_seats", False)),
        "win_prob_mode": str(getattr(fe, "win_prob_mode", "none")),
        # v43 pubval aux head (gen3_pubval_aux_v1): STRUCTURAL string like win_prob_mode, gated in
        # check_compatible, so it must reach the worker's gate (a pubval-ON run's own snapshots carry it).
        "pubval_mode": str(getattr(fe, "pubval_mode", "none")),
        # v44 z_arch/FiLM (gen3_zarch_film_v1): STRUCTURAL string + int gated in check_compatible
        # (the encoder + generator params; the dim = generator in_features), so both must reach the
        # worker's gate (a zarch-ON run's own pool/sentinel snapshots carry them).
        "zarch_film": str(getattr(fe, "zarch_film", "off")),
        "zarch_dim": int(getattr(fe, "zarch_dim", 0)),
        # v46 per-team LUT (gen3_zarch_lut_v1): STRUCTURAL string + the table height. The ROSTERS
        # themselves ride the persistent `zarch_lut_table` buffer in the state_dict, so a worker only
        # needs the shape-relevant pair to reproduce the module before load_state_dict fills it.
        "zarch_lut": str(getattr(fe, "zarch_lut", "off")),
        "zarch_lut_rosters": (
            [[0] * _TEAM_SIG_DIM for _ in range(int(getattr(fe, "zarch_lut_teams", 0)))]
            if str(getattr(fe, "zarch_lut", "off")) != "off" else None),
        # v29 value-dist head: only the check_compatible-gated structural toggles (mode + atom count) —
        # the support (vmin/vmax) is resume-only-checked on the trainer, never by a worker's load gate.
        "value_dist_mode": str(getattr(fe, "value_dist_mode", "none")),
        "value_dist_bins": int(getattr(fe, "value_dist_bins", 0)),
        # gen3_seed_quantile_v1 (v63): the per-seed quantile head is a state_dict-changing module,
        # so a frozen opponent's gate must see it (else a quantile-on run FATALs on its own sentinels).
        "seed_quantile": bool(getattr(fe, "seed_quantile", False)),
        # gen3_value_threat_inject_v1 (v64): the critic threat-injection projection is a
        # state_dict-changing module AND it flips the op's reducer on, so a frozen opponent's
        # gate must see it (else an inject-on run FATALs loading its own sentinels).
        "value_threat_inject": bool(getattr(fe, "value_threat_inject", False)),
        # gen3_opp_intent_v1 (v67): state_dict-changing heads, so a frozen opponent's gate must see them.
        "opp_intent": bool(getattr(fe, "opp_intent", False)),
        # gen3_species_prior_fusion_v1 (v68): the state_dict is identical either way, so this toggle is
        # the ONLY carrier of what the species head's output MEANS — a frozen opponent's gate must see it.
        "species_prior_fusion": bool(getattr(fe, "species_prior_fusion", False)),
        # gen3_t0_species_prior_v1 (v72): same shape of toggle — no state_dict delta, so the
        # recorded value is the only thing a resume can compare.
        "t0_species_prior": bool(getattr(fe, "t0_species_prior", None) is not None),
        "opp_intent_grad_mode": str(getattr(fe, "opp_intent_grad_mode", "detached")),
        "intent_value_reduce": bool(getattr(fe, "intent_value_reduce", None) is not None),
        # gen3_intent_move_cell_v1 (v77): widens the pointer move scorer (policy state_dict), so a
        # frozen opponent's gate must see it (else a flag-on run FATALs loading its own sentinels).
        "intent_move_cell": bool(getattr(fe, "intent_move_cell", None) is not None),
        # gen3_unified_topk_incoming_v1 (v30): the top-K incoming block's K (0 = off) — STRUCTURAL int,
        # gated in check_compatible (it scales the projection widths), so it must reach the worker's gate.
        "damage_topk_k": int(getattr(fe, "damage_topk_k", 0)),
        # gen3_per_move_matrices_v1 (v32): the outgoing per-move damage matrix — STRUCTURAL bool (widens the
        # op out_dim), gated in check_compatible, so it must reach the worker's gate.
        "damage_matrices_outgoing": bool(getattr(fe, "damage_matrices_outgoing", False)),
        # gen3_per_move_matrices_v1 (v33): the incoming per-move damage matrix — STRUCTURAL bool, gated.
        "damage_matrices_incoming": bool(getattr(fe, "damage_matrices_incoming", False)),
        # gen3_per_move_matrices_v1 (v39): the TRANSPOSED outgoing matrix (our 6 mons → opp active) —
        # STRUCTURAL bool (widens the op out_dim), gated in check_compatible, so it must reach the worker's gate.
        "damage_matrices_outgoing_all": bool(getattr(fe, "damage_matrices_outgoing_all", False)),
        # gen3_bidir_threat_trunk_v1 (v36): the uncertainty-aware P(outspeed) — a version-gated
        # forward-behavior bool, so it must reach the worker's check_compatible gate.
        "threat_prob_outspeed": bool(getattr(fe, "threat_prob_outspeed", False)),
        # gen3_hp_belief_ablation_v1 (v53): 'composed' vs 'flat' changes both the state_dict (the
        # HPTypeBelief head) and the forward, so it must reach the worker's check_compatible gate.
        "hp_belief_mode": str(getattr(fe, "hp_belief_mode", "composed")),
        # gen3_belief_grad_mode_v1 (v41): the belief-trunk-gradient knob. detach() is value-preserving so a
        # frozen opponent's forward is identical regardless — it is NOT check_compatible-gated (resume-only).
        # Threaded for the trainee's recorded config + so a worker rebuilds the SAME forward (no-op either way).
        "belief_grad_mode": str(getattr(fe, "belief_grad_mode", "shaping")),
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
