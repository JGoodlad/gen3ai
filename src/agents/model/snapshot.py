from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, cast, Dict, Optional, Tuple

import stable_baselines3
from sb3_contrib import MaskablePPO

from agents.model.damage_tables import _PRIOR_FLOOR, sanitize_historical_move_floor
from agents.model.model_version import ModelVersion, ModelVersionError
from agents.training.instrumented_ppo import InstrumentedMaskablePPO
from utils.git import get_git_hash

#: The launcher exports its chosen pin here; every git_hash written for a run reads it.
LAUNCHER_GIT_HASH_ENV = "LAUNCHER_GIT_HASH"
#: ...and WHICH of the four sources chose it ("pin_commit"/"checkpoint"/"sync_to_main"/"head").
LAUNCHER_PIN_SOURCE_ENV = "LAUNCHER_PIN_SOURCE"


class GitHashMismatchError(RuntimeError):
    """The launcher's declared pin and the imported checkout's HEAD name DIFFERENT commits.

    A producer-side GIGO guard, thrown where the wrong value would be WRITTEN rather than
    where it is later read. One of the two is lying about which code is running, and a
    checkpoint stamped with the wrong commit is unrecoverable provenance: the launcher pins a
    resume to the recorded hash, so a wrong stamp resumes on wrong code.
    """


def _hashes_agree(a: Optional[str], b: Optional[str]) -> bool:
    """Prefix-tolerant equality (one side may be a short hash). Unknowns never disagree."""
    x = (a or "").strip().lower()
    y = (b or "").strip().lower()
    if not x or not y or x == "unknown" or y == "unknown":
        return True
    return x == y or x.startswith(y) or y.startswith(x)


def resolve_git_hash(explicit: Optional[str] = None) -> str:
    """THE one resolver for "which commit is this run's code?" — run metadata AND sidecars.

    Order: an ``explicit`` argument (a caller that already knows, e.g. the round-trip smoke's
    sentinel) → ``$LAUNCHER_GIT_HASH`` (the launcher's pin) → ``get_git_hash()`` (the HEAD of
    the checkout this module was IMPORTED from, per ``utils.git``).

    🚨 **When the env pin and the imported tree disagree, this RAISES.** That combination
    means the child is running code the launcher did not pin — the 2026-09-05 defect in
    reverse — and silently picking either one is how a sidecar came to record the ambient
    main HEAD (``fff95a16``) for a run pinned to ``eb5261ff``. Not a warning: a warning in a
    training child's stdout is a line in a 1 MiB ring buffer nobody reads.
    """
    if explicit:
        return explicit
    env_hash = (os.environ.get(LAUNCHER_GIT_HASH_ENV) or "").strip()
    tree_hash = get_git_hash()
    if env_hash and not _hashes_agree(env_hash, tree_hash):
        from utils.paths import repo_root
        raise GitHashMismatchError(
            f"REFUSING to stamp a checkpoint with an ambiguous commit.\n"
            f"  ${LAUNCHER_GIT_HASH_ENV} (the launcher's pin) = {env_hash}\n"
            f"  HEAD of the imported checkout {repo_root()} = {tree_hash}\n"
            f"These name different commits, so one of them is not the code that is running. "
            f"Most likely the child's PYTHONPATH no longer points at the pinned worktree. "
            f"Fix the launch (or unset ${LAUNCHER_GIT_HASH_ENV} if you deliberately run "
            f"unpinned) — do not let the run stamp a commit it may not be executing."
        )
    return env_hash or tree_hash


def _update_pin_history(
    existing: Any,
    *,
    git_hash: str,
    pin_source: Optional[str],
    step: int,
    legacy_git_hash: Optional[str] = None,
    legacy_pin_source: Optional[str] = None,
) -> list:
    """APPEND-ONLY spans of "which commit ran which steps".

    The scalar ``git_hash`` is rewritten on every save, so on a run that restarts every 3 h it
    records the LAST code to touch the run, not the code that ran most of it (observed on
    ai_v9_171: ``eb5261ff``, then ``fff95a16`` after one resume). This list keeps the whole
    story: an entry per contiguous commit span, and an existing entry is never rewritten
    except to advance its ``last_step``.

    A LEGACY metadata.json — one with a scalar ``git_hash`` but no history — is seeded with a
    single ``derived: true`` entry for that hash, so the absence of a history is not silently
    read as "this run only ever ran one commit".
    """
    hist = [dict(e) for e in existing if isinstance(e, dict)] if isinstance(existing, list) else []
    if not hist and legacy_git_hash:
        hist.append({
            "git_hash": legacy_git_hash,
            "pin_source": legacy_pin_source,
            "first_step": int(step),
            "last_step": int(step),
            "derived": True,   # first_step is when we NOTICED, not when that commit started
        })
    if hist and hist[-1].get("git_hash") == git_hash:
        last = hist[-1]
        last["last_step"] = max(int(last.get("last_step") or 0), int(step))
        if pin_source and not last.get("pin_source"):
            last["pin_source"] = pin_source
        return hist
    hist.append({
        "git_hash": git_hash,
        "pin_source": pin_source,
        "first_step": int(step),
        "last_step": int(step),
    })
    return hist


def _read_pin_history(model_dir: str) -> Optional[list]:
    """The run-level ``pin_history`` as it stands right now, for stamping into a sidecar."""
    try:
        with open(os.path.join(model_dir, "metadata.json")) as f:
            hist = json.load(f).get("pin_history")
    except (OSError, ValueError):
        return None
    return hist if isinstance(hist, list) else None


def save_model_snapshot(
    model_dir: str,
    version: ModelVersion,
    git_hash: Optional[str] = None,
    current_lr: Optional[float] = None,
    current_epochs: Optional[int] = None,
    hparams: Optional[dict] = None,
    cli_args: Optional[dict] = None,
    original_command: Optional[str] = None,
    reward_composition: Optional[dict] = None,
    lineage: Optional[dict] = None,
    num_timesteps: Optional[int] = None,
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

    `reward_composition` is the per-class ACTIVE-term census of the run's reward
    (`agents.training.reward_manager.reward_class_composition`), carried forward like `cli_args`.
    It is recorded because `model_config.json` states the reward FLAGS while nothing stated what
    they COMPOSE TO — the gap the silent v8->v9 composition drift lived in — and because it is the
    field a launch-diff gate compares between a new generation and its reference.

    `lineage` is the **immutable** fork-ancestry block (`agents.training.lineage`): who this run
    forked from, its teachers/target, and the ancestry chain. Same contract as `original_command` —
    written ONCE at fork creation, existing value always wins, so a launcher restart (which swaps
    `--model` to the fork's OWN drifted checkpoint) can never re-point the recorded parent. The
    caller passes `None` on a same-run restart; a fresh run passes the explicit null form
    (`fork_parent: null, role: "fresh"`), because "no block" and "no parent" are different facts.

    `original_command` is the **immutable** original invocation that CREATED the model — the
    launcher command under a launcher, else this process's `sys.argv`. Unlike `cli_args` (which
    is overwritten with the resuming process's args on every restart), it is written ONCE at model
    creation and then preserved verbatim across all subsequent saves/restarts: the existing value
    always wins. The caller may pass it explicitly; otherwise it is derived here.
    """
    os.makedirs(model_dir, exist_ok=True)

    with open(os.path.join(model_dir, "model_config.json"), "w") as f:
        f.write(version.to_json())

    # ONE resolver for the run metadata and every checkpoint sidecar (`resolve_git_hash`),
    # so the two can never disagree about which commit ran — and it RAISES when the
    # launcher's pin and the imported checkout's HEAD do.
    git_hash = resolve_git_hash(git_hash)
    # WHERE that pin came from, when a launcher chose it: "pin_commit" (named on the command
    # line — e.g. every arm of a batch pinned to ONE commit), "checkpoint" (the resumed
    # checkpoint's own hash), "sync_to_main" or "head". Absent when nothing set it.
    pin_source = os.environ.get(LAUNCHER_PIN_SOURCE_ENV)

    # Preserve state accumulated by other writers (this rebuilds metadata from
    # scratch, so anything not carried forward here is dropped).
    meta_path = os.path.join(model_dir, "metadata.json")
    existing_history = {}
    existing_latest_eval = None
    existing_team_wr = None
    existing_cli_args = None
    existing_launcher_command = None
    existing_original_command = None
    existing_lineage = None
    existing_matchup_history = []
    existing_reward_composition = None
    existing_pin_history = None
    existing_git_hash = None
    existing_pin_source = None
    existing_num_timesteps = None
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            existing = json.load(f)
            existing_history = existing.get("snapshot_history", {})
            existing_latest_eval = existing.get("latest_eval")
            existing_team_wr = existing.get("team_win_rates")
            existing_cli_args = existing.get("cli_args")
            existing_launcher_command = existing.get("launcher_command")
            existing_original_command = existing.get("original_command")
            existing_lineage = existing.get("lineage")
            existing_matchup_history = existing.get("matchup_history", [])
            existing_reward_composition = existing.get("reward_composition")
            existing_pin_history = existing.get("pin_history")
            existing_git_hash = existing.get("git_hash")
            existing_pin_source = existing.get("pin_source")
            existing_num_timesteps = existing.get("num_timesteps")

    metadata: Dict[str, Any] = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash,
        "python_version": sys.version,
        "sb3_version": stable_baselines3.__version__,
    }
    if pin_source:
        metadata["pin_source"] = pin_source
    # APPEND-ONLY commit spans. The scalar `git_hash` above is "the code that saved THIS
    # checkpoint"; `pin_history` is "which code ran which steps", which the scalar destroys
    # on every restart. Same immutability contract as `lineage`: existing entries win.
    _step = num_timesteps
    if _step is None and hparams:
        _step = hparams.get("num_timesteps")
    if _step is None:
        _step = _max_recorded_step(existing_history)
    metadata["pin_history"] = _update_pin_history(
        existing_pin_history,
        git_hash=git_hash,
        pin_source=pin_source,
        step=int(_step or 0),
        legacy_git_hash=existing_git_hash,
        legacy_pin_source=existing_pin_source,
    )
    if hparams:
        metadata.update(hparams)
    # HOW FAR THIS RUN TRAINED, as a top-level key — "latest", so unlike `original_command` /
    # `lineage` / `pin_history` it is OVERWRITTEN on every save. It exists because the step count
    # was previously readable only by opening the checkpoint .zip, which the JSON-only offline
    # tools (`main.lineage`, `main.sidecar_audit`, `main.dose`) deliberately never do.
    # ABSENT rather than 0 when the save does not know it: a legacy run reads UNKNOWN, and an
    # existing value is carried forward rather than clobbered by a save that was handed no step
    # (`_max_recorded_step`'s stale-but-ordered guess feeds pin_history's spans ONLY — writing a
    # guess here would make an inferred number indistinguishable from a recorded one).
    known_step = num_timesteps
    if known_step is None and hparams:
        known_step = hparams.get("num_timesteps")
    if known_step is None:
        known_step = existing_num_timesteps
    if known_step is not None:
        metadata["num_timesteps"] = int(known_step)
    if current_lr is not None:
        metadata["current_lr"] = current_lr
    if current_epochs is not None:
        metadata["current_epochs"] = current_epochs
    # Run provenance (carried forward like snapshot_history / latest_eval): the full CLI
    # namespace and the launcher's own invocation (the latter from the env it sets).
    cli = cli_args if cli_args is not None else existing_cli_args
    if cli is not None:
        metadata["cli_args"] = cli
    # The reward COMPOSITION census — same carry-forward rule as cli_args, so the many saves that
    # don't know the reward config (periodic checkpoints, the eval best-model copy) preserve it.
    composition = reward_composition if reward_composition is not None else existing_reward_composition
    if composition is not None:
        metadata["reward_composition"] = composition
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
    # LINEAGE — IMMUTABLE, exactly like original_command above: the existing block always wins, so a
    # restart preserves the fork parent recorded at creation instead of re-deriving it from the
    # checkpoint the launcher swapped in. See agents.training.lineage.
    lineage_block = existing_lineage if existing_lineage is not None else lineage
    if lineage_block is not None:
        metadata["lineage"] = lineage_block
    if existing_history:
        metadata["snapshot_history"] = existing_history
    if existing_latest_eval is not None:
        metadata["latest_eval"] = existing_latest_eval
    if existing_team_wr is not None:
        metadata["team_win_rates"] = existing_team_wr
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


def record_team_win_rates(model_dir: str, table: dict) -> None:
    """Write the per-team win-rate table to metadata.json as a top-level `team_win_rates` block.

    The owner's recording rule (design_flywheel_tick_tock.md §6b): per-team rates ride the run's
    EXISTING metadata channel beside `latest_eval`'s per-opponent records — one artifact per run
    holding the whole competitive story — and are deliberately NOT emitted to TensorBoard.
    `save_model_snapshot` carries the block forward across checkpoints like `latest_eval`."""
    meta_path = os.path.join(model_dir, "metadata.json")
    meta = {}
    if os.path.exists(meta_path):
        with open(meta_path) as f:
            meta = json.load(f)
    meta["team_win_rates"] = table
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


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
    hodge: "dict | None" = None,
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

    ``hodge`` (optional) is the cycle's spine/width read (``agents.training.hodge``) as
    ``{"recorded": bool, "width_elo", "cyclic_fraction", …, "caveats": [...]}`` — the same two
    numbers recorded to TensorBoard, kept here so they can be replotted offline, and carrying
    ``recorded: false`` + a reason when the cycle's graph had no testable triangle. Recording
    the OMISSION is the point: a missing TB point and a suppressed one look identical in
    TensorBoard, and only one of them is a fact about the graph.

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
        if hodge:
            row["hodge"] = hodge
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
    # `meta` is a raw JSON blob, so every field off it is `Any`.
    h = (meta.get("cli_args") or {}).get("_matchup_spec_hash")
    if h:
        return h  # type: ignore[no-any-return]
    hist = meta.get("matchup_history") or []
    return cast(Optional[str], hist[-1].get("hash")) if hist else None


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


def _max_recorded_step(history: dict) -> int:
    """Highest step number named by a ``snapshot_history`` key, or 0.

    The fall-back step for ``pin_history`` when the caller passed no ``num_timesteps`` — a
    stale-but-ordered number beats 0, which would collapse every span onto the same point.
    """
    best = 0
    for name in history or {}:
        try:
            best = max(best, int(str(name).split("_")[1]))
        except (IndexError, ValueError):
            continue
    return best


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


def _step_from_checkpoint_name(checkpoint_path: str) -> Optional[int]:
    """The step encoded in ``<prefix>_<N>_steps.zip``, or ``None``.

    The LAST-RESORT source for a sidecar's ``num_timesteps`` — used only when the caller passed
    neither the value nor an hparams block carrying it. The name is written by the checkpoint
    callback as ``f"{prefix}_{self.num_timesteps}_steps.zip"``, so where the form matches it IS
    the step; where it does not (``final_model.zip``, ``best_model.zip``) this returns ``None``
    and the key is simply absent, which reads as unknown.
    """
    parts = os.path.basename(checkpoint_path).removesuffix(".json").removesuffix(".zip").split("_")
    if len(parts) >= 2 and parts[-1] == "steps" and parts[-2].isdigit():
        return int(parts[-2])
    return None


def _resolve_checkpoint_steps(checkpoint_path: str, hparams: Optional[dict],
                              explicit: Optional[int]) -> Optional[int]:
    """The step a checkpoint sits at, in preference order, or ``None`` when nobody knows.

    explicit argument -> the ``hparams`` block (every production caller routes
    ``main.train.run_io._model_hparams`` through here, which records it) -> the ``.zip``'s own
    ``..._<N>_steps`` name. ``None`` is a legitimate answer and must stay one: 0 would be a
    claim, and the readers render an absent key as UNKNOWN.
    """
    if explicit is not None:
        return int(explicit)
    if hparams and hparams.get("num_timesteps") is not None:
        try:
            return int(hparams["num_timesteps"])
        except (TypeError, ValueError):
            pass
    return _step_from_checkpoint_name(checkpoint_path)


def _build_snapshot_entry(
    lr: float,
    n_epochs: int,
    hparams: Optional[dict] = None,
    git_hash: Optional[str] = None,
    handoff_lr: Optional[float] = None,
    eval_block: Optional[dict] = None,
    matchup_hash: Optional[str] = None,
    pin_history: Optional[list] = None,
    num_timesteps: Optional[int] = None,
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
    # Same resolver the run-level metadata uses — so a sidecar and metadata.json can never
    # name different commits, which is exactly what happened before 2026-09-05 (the sidecar
    # took `git rev-parse HEAD` in the child's CWD, i.e. the un-pinned main checkout).
    entry["git_hash"] = resolve_git_hash(git_hash)
    if handoff_lr is not None:
        entry["handoff_lr"] = handoff_lr
    if eval_block:
        entry["latest_eval"] = eval_block
    # The declared-matchup regime tag AS OF this checkpoint (like the latest_eval stamp): each
    # checkpoint is self-describing about what it was training against, robust to a later era
    # overwriting the run-level cli_args.
    if matchup_hash:
        entry["matchup_hash"] = matchup_hash
    # The run's commit-span history AS OF this write, so a checkpoint carries the whole
    # provenance and not just the one hash (see `_update_pin_history`).
    if pin_history:
        entry["pin_history"] = pin_history
    # HOW FAR THE RUN HAD TRAINED when this checkpoint was written. Assigned after the hparams
    # merge like `git_hash` above, so an explicit argument always wins over a colliding hparam.
    # Absent — never 0 — when nobody knows it, so a legacy sidecar reads UNKNOWN.
    if num_timesteps is not None:
        entry["num_timesteps"] = int(num_timesteps)
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
    pin_history: Optional[list] = None,
    num_timesteps: Optional[int] = None,
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
        lr, n_epochs, hparams, git_hash, handoff_lr, eval_block, matchup_hash, pin_history,
        num_timesteps,
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
    num_timesteps: Optional[int] = None,
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
    # 🚨 THIS LINE WAS THE 2026-09-05 DEFECT: it read `git_hash or get_git_hash()`, skipping
    # $LAUNCHER_GIT_HASH entirely, and `get_git_hash()` asked git about the process CWD — the
    # UNPINNED main checkout, because the launcher spawns the child with no `cwd=`. The
    # ambient hash it returned is truthy, so it then WON the `git_hash or env or ...` chain
    # inside `_build_snapshot_entry` too: every sidecar and every snapshot_history entry
    # recorded main's HEAD while metadata.json recorded the pin.
    resolved_hash = resolve_git_hash(git_hash)
    eval_block = _read_latest_eval(model_dir)
    matchup_hash = _read_matchup_hash(model_dir)
    pin_history = _read_pin_history(model_dir)
    # THE STEP THIS CHECKPOINT IS: the explicit argument, else the hparams block every production
    # caller passes (`main.train.run_io._model_hparams` records it), else the .zip's own name.
    steps = _resolve_checkpoint_steps(checkpoint_path, hparams, num_timesteps)
    write_checkpoint_metadata(
        checkpoint_path,
        lr,
        n_epochs,
        hparams=hparams,
        git_hash=resolved_hash,
        handoff_lr=handoff_lr,
        eval_block=eval_block,
        matchup_hash=matchup_hash,
        pin_history=pin_history,
        num_timesteps=steps,
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
        pin_history=pin_history,
        num_timesteps=steps,
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
    pin_history: Optional[list] = None,
    num_timesteps: Optional[int] = None,
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
    entry = _build_snapshot_entry(
        lr, n_epochs, hparams, git_hash, handoff_lr, eval_block, matchup_hash, pin_history,
        _resolve_checkpoint_steps(checkpoint_path, hparams, num_timesteps),
    )
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
            return cast(dict, json.load(f))
    return {}


def _checkpoint_metadata_path(checkpoint_path: str) -> str:
    if checkpoint_path.endswith(".zip"):
        return checkpoint_path[:-4] + ".json"
    return checkpoint_path + ".json"


def load_model_snapshot(
    model_path: str,
    env: Any,
    current_version: ModelVersion,
    device: str = "auto",
    tensorboard_log: Optional[str] = None,
    enforce_vf_coef: Optional[float] = None,
    enforce_reward_config: Any = None,   # duck-typed, like ModelVersion.build
    enforce_value_tail_weight: Optional[float] = None,
    enforce_value_dist: Optional[Tuple[float, float]] = None,
    enforce_belief_grad_mode: Optional[str] = None,
    allow_belief_grad_mode_change: bool = False,
    enforce_value_from_dist: Optional[bool] = None,
    allow_value_from_dist_change: bool = False,
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


# Compiled-opponent machinery lives in `compile_opponents.py` (split 2026-08-16); re-exported
# here because the historical import path is this module.
from agents.model.compile_opponents import (   # noqa: F401
    COMPILE_QUORUM_ENV, DEFAULT_INDUCTOR_CACHE_DIR, CompileExtractorError, arm_compile_quorum,
    maybe_compile_extractor,
    _compile_warn, _compile_warmup_obs, _eager_fallback_on_error, _inductor_cache_dir,
    _measure_arms, _time_forward,
)



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
#
# THE RULE, stated once, because every future deletion has to answer it. When a kwarg leaves
# `Gen3FeaturesExtractor.__init__` it goes in exactly one of these two lists:
#   * INERT     — no value of it selected anything in the surviving forward (it only SIZED or
#                 INITIALISED a deleted module, it was training-only, or its branch was
#                 unreachable in production). Pop unconditionally.
#   * JUDGED    — some value of it fed a forward this codebase can no longer reproduce. Record
#                 the ONE value that the surviving code still reproduces; every other value is
#                 refused loudly. Two things put a flag here: a byte-identical state_dict across
#                 its values (nothing shape-based can catch the swap), or an ON value that named
#                 PARAMETERS (popping it hands SB3 an unplaceable state_dict).
# Forgetting BOTH lists is the failure this machinery cannot self-detect, so
# `ctor_kwarg_snapshot_test.py` pins the live kwarg set and fails red when one disappears.
_DEAD_FEK_INERT = (
    "damage_refine_rounds", "threat_refine_outgoing", "threat_unrevealed_outgoing",
    "threat_status_refine", "move_belief_single_compute",
    # v78 (gen3_flag_surface_p1_v1): the zarch family's non-module knobs. `zarch_dim` /
    # `zarch_lut_init_std` only ever SIZED or INITIALISED the deleted modules, and
    # `zarch_lut_rosters` is a lookup table, so none of them can be ON on their own — the two
    # JUDGED mode strings below carry the whole decision.
    "zarch_dim", "zarch_lut_init_std", "zarch_lut_rosters",
    # v88: pubval's training coefficient — scaled a loss for a head that no longer exists.
    "pubval_coef",
)
_DEAD_FEK_JUDGED = (("move_belief_prefuse", True), ("damage_op_prefuse", True),
                    ("damage_reattend", False),
                    # v88 (gen3_dead_flag_purge_v1): both were config_only frozen OFF since v78;
                    # each ON value widened a projection/out_dim the surviving code cannot
                    # rebuild, so ON is refused, OFF pops. pubval's mode string goes the same
                    # way (PubValHead carried parameters when != 'none').
                    ("value_active_readout", False),
                    ("damage_matrices_outgoing_all", False),
                    ("pubval_mode", "none"),
                    # v75: the SimSiam latent-belief predictor is deleted. True is REFUSED because
                    # it put parameters in the state_dict; False pops silently (nothing built).
                    ("opp_belief_latent", False),
                    # v78: the ZArchEncoder + FiLM generators, the per-team LUT Embedding, and the
                    # per-seed quantile Linear are deleted. Each ON value named PARAMETERS, so it is
                    # refused for the v75 reason rather than popped into an unplaceable state_dict.
                    ("zarch_film", "off"), ("zarch_lut", "off"), ("seed_quantile", False),
                    # ---------------------------------------------------------------------------
                    # PRE-FLOOR names (v48/v52/v66). MEASURED 2026-08-17 over the 89 runs under
                    # `models/` with a checkpoint: these five were deleted from the constructor
                    # without ever entering either list. 70 runs CARRY one; 7 (`ai_v9_01`..
                    # `ai_v9_07`) actually reached a bare `TypeError` on the resume /
                    # frozen-opponent / eval-worker paths instead of the judged refusal those paths
                    # exist to give — the other 63 were masked only because an EARLIER entry in
                    # this tuple refused them first, which is luck, not coverage. All five predate
                    # MIGRATION_FLOOR, and that is exactly WHY they were missed: the config half of
                    # the rule became the blanket floor refusal, while the zip half carries no
                    # `config_version` to fall under any floor. `ctor_kwarg_snapshot_test.py` is
                    # the tripwire that stops the next deletion repeating this.
                    #
                    # v48 (gen3_cpu_damage_deleted_v1): the three `--unified-obs` ablation masks
                    # ZEROED an obs region out of the model's view — "an ablation toggle (no
                    # weight-shape change)", so a True checkpoint's state_dict is byte-identical to
                    # a False one and nothing shape-based would catch the swap. False = the region
                    # is read live, which the surviving forward still does, so False pops.
                    ("mask_incoming_damage_obs", False),
                    ("mask_active_move_scalars_obs", False),
                    ("mask_move_effects_obs", False),
                    # v52 (gen3_typed_hp_belief_v1): the tri-state off|prior|learned is gone —
                    # `HPTypeBelief` is now UNCONDITIONAL under a move belief. 'learned' is the one
                    # value whose state_dict CARRIES that head (and the only value any archived run
                    # records), so it pops; 'off'/'prior' name a head-less forward the surviving
                    # code cannot build, and popping one would hand SB3 a state_dict missing the
                    # head's parameters.
                    ("hp_type_belief_mode", "learned"),
                    # v66: the op marginalised P(KO) over the nature posterior instead of
                    # evaluating at its mode. No parameters either way (it read the spread head's
                    # logits), so True is the v71 shape exactly — a byte-identical state_dict in
                    # front of a `DamageOperator._nature_marg_ko` kernel that no longer exists.
                    # False = mean-field, which is what the op still computes, so False pops.
                    ("spread_belief_nature_marginalize", False),
                    # v96 (gen3_critic_route_wave_v1): three deleted CRITIC routes. Each ON value
                    # built a zero-init projection module, so it named PARAMETERS the surviving
                    # extractor has no home for ⇒ refused (the v75 rule); OFF pops silently,
                    # because the route built nothing at all. This is the REACHABLE half of the
                    # judgment — `_migrate_config`'s v96 block says the same thing about the config
                    # JSON, but MIGRATION_FLOOR 96 refuses a pre-v96 config before it runs, whereas
                    # the pickled kwargs carry no `config_version` for any floor to catch. It
                    # matters immediately: gen-15's config records all three OFF (they pop), while
                    # the `ai_v9_17_tdaux_lam*` forks recorded intent_value_reduce=True and
                    # value_clock=True and are therefore refused WITH the re-read diagnosis rather
                    # than TypeError-ing inside SB3's rebuild.
                    ("intent_value_reduce", False),
                    ("value_clock", False),
                    ("value_intent", False))


def sanitize_dead_extractor_kwargs(fek: dict) -> bool:
    """Drop DELETED keys from a saved `features_extractor_kwargs`. True if it changed.

    Raises `ModelVersionError` — exactly as `_migrate_config` does — when a JUDGED field records a
    value the surviving forward pass cannot reproduce.

    This is the ZIP-side twin of `_migrate_config`'s key handling and both are needed: the config
    JSON drives the version GATE, while these kwargs are what SB3 splats into the extractor
    constructor when it rebuilds the policy. A key dropped from one and not the other either fails
    the gate for the wrong reason or TypeErrors inside the constructor.
    """
    changed = False
    for dead, supported in _DEAD_FEK_JUDGED:
        if dead in fek:
            # Compare on TYPE, not truthiness: the v78 entries are mode STRINGS, and `bool("off")`
            # is True — a truthiness test would refuse every one of them, including the OFF configs
            # this is meant to wave through.
            recorded = fek[dead]
            mismatch = (recorded != supported if isinstance(supported, str)
                        else bool(recorded) is not supported)
            if mismatch:
                raise ModelVersionError(
                    f"{dead}={recorded!r} is no longer supported: the only supported value is "
                    f"{supported!r}.\nThis checkpoint trained under a forward pass that no longer "
                    "exists in the codebase and cannot be reproduced from HEAD.\n"
                    "To re-read it, use the git_hash recorded in its own metadata.json."
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
    2. Extractor kwargs deleted from the constructor (`sanitize_dead_extractor_kwargs`).

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
    mappings: Dict[str, Any],
    *,
    attend_unrevealed_opponents: bool = False,
    opp_belief_cls_k: int = 0,
    opp_belief_slots: bool = False,
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
    value_threat_inject: bool = False,
    opp_intent: bool = False,
    species_prior_fusion: bool = False,
    t0_species_prior: bool = False,
    opp_intent_grad_mode: str = "detached",
    intent_value_reduce: bool = False,
    intent_move_cell: bool = False,
    value_entity_pool: bool = False,
    value_entity_pool_full: bool = False,
    history_events: bool = False,
    item_belief: bool = False,
    intent_threshold: bool = False,
    intent_conditional: bool = False,
    pair_outcome_cell: bool = False,
    pair_outcome_switch: bool = False,
    switch_branch_cell: bool = False,
    conditional_threat_cell: bool = False,
    pair_value_route: bool = False,
    op_drop_renders: bool = False,
    op_believed_lean: bool = False,
    value_clock: bool = False,
    value_intent: bool = False,
    damage_topk_k: int = 0,
    damage_matrices_outgoing: bool = False,
    damage_matrices_incoming: bool = False,
    threat_prob_outspeed: bool = False,
    hp_type_belief_coef: float = 0.0,
    item_belief_coef: float = 0.0,
    td_aux_coef: float = 0.0,
    hp_belief_mode: str = "composed",
    belief_grad_mode: str = "shaping",
    cf_evidential: bool = False,
    cf_twin_heads: bool = False,
    cf_shadow_critic: bool = False,
    q_winprob_mode: str = "none",
    vf_coef: float = 0.5,
    reward_config: Any = None,               # duck-typed, like ModelVersion.build
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
    ext_kwargs["edge_bias_families"] = edge_bias_families
    ext_kwargs["entity_tail_seats"] = entity_tail_seats
    ext_kwargs["win_prob_mode"] = win_prob_mode
    # gen3_cf_evidential_head_v1 (v98): a state_dict-changing head, so a frozen eval/pool opponent's
    # gate must see it — otherwise a cf-evidential run FATALs loading its OWN sentinels.
    ext_kwargs["cf_evidential"] = cf_evidential
    # gen3_cf_twin_heads_v1 (v99): the same reason, twice — the twin win-prob heads and the shadow
    # critic are state_dict-changing modules the forward never calls, so a frozen eval/pool
    # opponent's gate must see them or a twin-heads run FATALs loading its OWN sentinels.
    ext_kwargs["cf_twin_heads"] = cf_twin_heads
    ext_kwargs["cf_shadow_critic"] = cf_shadow_critic
    # gen3_q_winprob_head_v1 (v107): the per-action Q head's params are a state_dict delta whose
    # ONLY output is a stash, so nothing downstream would fail on a mismatch — a frozen eval/pool
    # opponent's gate must see the toggle or a Q-head run FATALs loading its OWN sentinels.
    ext_kwargs["q_winprob_mode"] = q_winprob_mode
    ext_kwargs["value_dist_mode"] = value_dist_mode
    ext_kwargs["value_dist_bins"] = value_dist_bins
    ext_kwargs["value_dist_vmin"] = value_dist_vmin
    ext_kwargs["value_dist_vmax"] = value_dist_vmax
    ext_kwargs["value_threat_inject"] = value_threat_inject
    ext_kwargs["opp_intent"] = opp_intent
    ext_kwargs["species_prior_fusion"] = species_prior_fusion
    ext_kwargs["t0_species_prior"] = t0_species_prior
    ext_kwargs["opp_intent_grad_mode"] = opp_intent_grad_mode
    ext_kwargs["intent_value_reduce"] = intent_value_reduce
    ext_kwargs["intent_move_cell"] = intent_move_cell
    ext_kwargs["value_entity_pool"] = value_entity_pool
    ext_kwargs["value_entity_pool_full"] = value_entity_pool_full
    ext_kwargs["history_events"] = history_events
    ext_kwargs["item_belief"] = item_belief
    ext_kwargs["intent_threshold"] = intent_threshold
    ext_kwargs["intent_conditional"] = intent_conditional
    ext_kwargs["pair_outcome_cell"] = pair_outcome_cell
    ext_kwargs["pair_outcome_switch"] = pair_outcome_switch
    ext_kwargs["switch_branch_cell"] = switch_branch_cell
    ext_kwargs["conditional_threat_cell"] = conditional_threat_cell
    ext_kwargs["pair_value_route"] = pair_value_route
    ext_kwargs["op_drop_renders"] = op_drop_renders
    ext_kwargs["op_believed_lean"] = op_believed_lean
    ext_kwargs["value_clock"] = value_clock
    ext_kwargs["value_intent"] = value_intent
    ext_kwargs["damage_topk_k"] = damage_topk_k
    ext_kwargs["damage_matrices_outgoing"] = damage_matrices_outgoing
    ext_kwargs["damage_matrices_incoming"] = damage_matrices_incoming
    ext_kwargs["threat_prob_outspeed"] = threat_prob_outspeed
    ext_kwargs["hp_belief_mode"] = hp_belief_mode
    ext_kwargs["belief_grad_mode"] = belief_grad_mode
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
        hp_type_belief_coef=hp_type_belief_coef, item_belief_coef=item_belief_coef,
        td_aux_coef=td_aux_coef,
    )


def arch_toggles_from_model(model: Any) -> dict:
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
        # gen3_cf_evidential_head_v1 (v98): state_dict-changing head, invisible to the forward —
        # the recorded toggle is the only thing a load gate can compare.
        "cf_evidential": bool(getattr(fe, "cf_evidential", False)),
        # gen3_cf_twin_heads_v1 (v99): same category — params in the state_dict, invisible to the
        # forward, so the recorded toggle is the only thing a load gate can compare.
        "cf_twin_heads": bool(getattr(fe, "cf_twin_heads", False)),
        "cf_shadow_critic": bool(getattr(fe, "cf_shadow_critic", False)),
        # gen3_q_winprob_head_v1 (v107): same category — params in the state_dict whose only
        # output is a side stash, so the recorded toggle is all a load gate can compare.
        "q_winprob_mode": str(getattr(fe, "q_winprob_mode", "none")),
        # v29 value-dist head: only the check_compatible-gated structural toggles (mode + atom count) —
        # the support (vmin/vmax) is resume-only-checked on the trainer, never by a worker's load gate.
        "value_dist_mode": str(getattr(fe, "value_dist_mode", "none")),
        "value_dist_bins": int(getattr(fe, "value_dist_bins", 0)),
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
        # gen3_unified_value_readout_v1 (v80): widens the value projection (state_dict), so a
        # frozen opponent's gate must see it.
        "value_entity_pool": bool(getattr(fe, "value_entity_pool", None) is not None),
        # gen3_unified_value_readout_v2 (v82): the full row set grows source_emb (state_dict).
        "value_entity_pool_full": bool(getattr(
            getattr(fe, "value_entity_pool", None), "full", False)),
        # gen3_event_window_v1 (v81): adds the EventSeats trunk modules (state_dict), gated.
        "history_events": bool(getattr(fe, "history_events", None) is not None),
        # gen3_item_belief_v1 (v83): adds the ItemBelief module (state_dict), gated.
        "item_belief": bool(getattr(fe, "item_belief_head", None) is not None),
        # gen3_intent_threshold_v1 (v84): the threshold operator's projections (state_dict), gated.
        "intent_threshold": bool(getattr(fe, "intent_threshold_move", None) is not None),
        # gen3_intent_conditional_v1 (v85): the mechanic-cell projection (state_dict), gated.
        "intent_conditional": bool(getattr(fe, "intent_conditional", None) is not None),
        # gen3_pair_outcome_v1 (v93): the unified outcome vector's move-cell projection
        # (state_dict + pointer-move-cell width), gated.
        "pair_outcome_cell": bool(getattr(fe, "pair_outcome_move", None) is not None),
        # gen3_pair_outcome_switch_v1 / gen3_switch_branch_v1 (v94): the Phase B projections
        # (state_dict + pointer switch/move cell widths), both gated.
        "pair_outcome_switch": bool(getattr(fe, "pair_outcome_switch", None) is not None),
        "switch_branch_cell": bool(getattr(fe, "switch_branch", None) is not None),
        # gen3_conditional_threat_v1 / gen3_pair_value_route_v1 (v95): the Phase C projections —
        # OA1 widens the pointer SWITCH cell; PV adds a zero-init injection inside CLSPool (no
        # width moves at all, so the gate is the only thing that can see it). Both read off the
        # BUILT module, never off a stored bool, so a config that lies about itself cannot survive.
        "conditional_threat_cell": bool(getattr(fe, "conditional_threat", None) is not None),
        "pair_value_route": bool(
            getattr(getattr(fe, "cls_pool", None), "pair_value_proj", None) is not None),
        # gen3_op_lean_forward_v1 (v86): out_gain shape / d3 forward math, both gated.
        "op_drop_renders": bool(getattr(getattr(fe, "damage_op", None), "drop_renders", False)),
        "op_believed_lean": bool(getattr(getattr(fe, "damage_op", None), "believed_lean", False)),
        # gen3_value_direct_routes_v1 (v87): both widen the value projection (state_dict), gated.
        "value_clock": bool(getattr(fe, "value_clock_route", None) is not None),
        "value_intent": bool(getattr(fe, "value_intent_route", None) is not None),
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
