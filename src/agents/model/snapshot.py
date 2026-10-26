from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, cast, Dict, Optional

import stable_baselines3
from sb3_contrib import MaskablePPO

from agents.model.damage_tables import _PRIOR_FLOOR, sanitize_historical_move_floor
from agents.model.model_version import ModelVersion, ModelVersionError
from agents.training.instrumented_ppo import InstrumentedMaskablePPO
from agents.training.instrumented_ppo.inference import InferenceMaskablePPO
from agents.training.instrumented_ppo.strict_load import StrictMaskablePPO
from utils.atomic_io import atomic_write_json
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
    torch_version: Optional[str] = None,
) -> list:
    """APPEND-ONLY spans of "which commit ran which steps" — and, since 2026-09-30, under WHICH
    torch (``torch_version``): a span is one (commit, torch) pair, so a run that changed
    interpreter at a restart shows the boundary. An entry written before the torch record has no
    ``torch`` key, which the launcher's interpreter guard reads as the legacy 2.5.1
    (``main.launcher.torch_runtime``).

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
    if hist and hist[-1].get("git_hash") == git_hash and (
            torch_version is None or hist[-1].get("torch") == torch_version):
        last = hist[-1]
        last["last_step"] = max(int(last.get("last_step") or 0), int(step))
        if pin_source and not last.get("pin_source"):
            last["pin_source"] = pin_source
        return hist
    entry: Dict[str, Any] = {
        "git_hash": git_hash,
        "pin_source": pin_source,
        "first_step": int(step),
        "last_step": int(step),
    }
    if torch_version is not None:
        entry["torch"] = torch_version
    hist.append(entry)
    return hist


def _read_run_json(path: str) -> Optional[dict]:
    """A run-level JSON file as a dict, or ``None`` when it is absent or unreadable."""
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


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
    Safe to call multiple times; each file is REPLACED atomically (`utils.atomic_io`: a failed or killed write leaves
    the previous file whole, never a truncated JSON that the next save would crash reading).
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
    # THE REVEAL PROVENANCE GUARD (gen3_reveal_provenance_v1): metadata.json copies the run's reveal mode and must
    # agree with the model_config.json this call is about to write (and the env core's declared mode) -- refused typed
    # BEFORE either file is touched. Only a block that states a mode is judged (`hparams` is optional and partial).
    from agents.model.oracle_reveal import check_provenance, hparams_witnesses
    _stated = hparams_witnesses(hparams)
    if any(v is not None for v in _stated.values()):
        check_provenance(config=version.oracle_reveal, witnesses=_stated, where=f"save {model_dir}")

    os.makedirs(model_dir, exist_ok=True)

    # `model_config.json` = the recorded config, PLUS one DERIVED annotation.
    #
    # 🚨 `inert_reward_flags` (gen3_frozen_phi_actor_only_v1) names the reward flags this config
    # makes unreachable — because a recorded value is not a running value and nothing said so. Since
    # the shaped-reward deletion (v122) the one remaining case is `draw_penalty` under
    # `terminal_indicator`. WRITTEN BESIDE THE VALUES, NEVER IN PLACE OF THEM: `check_reward_config`
    # compares each RECORDED value against the resuming argv's.
    #
    # It is NOT a `ModelVersion` field: it is a pure function of fields already in the file, so a
    # field would be a second copy of a derived fact, would need `check_compatible` to ignore it by
    # name, and would put a documentation string inside the weight-shape record. `to_json()` stays
    # exactly `asdict(self)` (several tests round-trip it straight back through `ModelVersion(**…)`),
    # and `_migrate_config` POPs this key on the way in as a version-INDEPENDENT sanitizer.
    _cfg = json.loads(version.to_json())
    try:
        from agents.training.reward_composition import inert_reward_flags
        _cfg["inert_reward_flags"] = inert_reward_flags(version)
    except Exception:  # noqa: BLE001 — an ANNOTATION must never be able to fail a save
        pass
    atomic_write_json(os.path.join(model_dir, "model_config.json"), _cfg, indent=2, sort_keys=True)

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
    existing_era = None
    existing_matchup_history = []
    existing_reward_composition = None
    existing_pin_history = None
    existing_git_hash = None
    existing_pin_source = None
    existing_num_timesteps = None
    existing_init_num_threads = None
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
            existing_era = existing.get("era")
            existing_matchup_history = existing.get("matchup_history", [])
            existing_reward_composition = existing.get("reward_composition")
            existing_pin_history = existing.get("pin_history")
            existing_git_hash = existing.get("git_hash")
            existing_pin_source = existing.get("pin_source")
            existing_num_timesteps = existing.get("num_timesteps")
            existing_init_num_threads = existing.get("init_num_threads")

    # WHICH TORCH ran this process (owner 2026-09-30: torch 2.8 is the default, 2.5.1 legacy).
    # Overwritten every save, like `python_version`; `pin_history` keeps the per-span story. The
    # launcher reads it to resume/fork a run under the torch it was trained on
    # (`main.launcher.torch_runtime`) — an ABSENT key means a pre-record run, i.e. torch 2.5.1.
    import torch
    metadata: Dict[str, Any] = {
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash,
        "python_version": sys.version,
        "torch_version": str(torch.__version__),
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
        torch_version=metadata["torch_version"],
    )
    if hparams:
        metadata.update(hparams)
    # `init_num_threads` — the torch thread count the run's FRESH build ran at — is IMMUTABLE like
    # `original_command`: the first recorded value wins, so a resume (whose model carries the same value
    # in its checkpoint) can never overwrite it. A FORK is a new run dir with no existing record, so it
    # records the value its checkpoint carries (the build its weights came from). ABSENT = unrecorded.
    if existing_init_num_threads is not None:
        metadata["init_num_threads"] = existing_init_num_threads
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
    # ERA — IMMUTABLE (`utils.era`), like original_command: stamped ONCE at the run's CREATION save
    # and then carried verbatim. "Creation" = the first save that finds neither an `original_command`
    # nor `cli_args` on disk (an early `record_eval_results` may already have made the file, so
    # "no file" is not the test). A run that already HAS a record but no `era` is PRE-ERA history:
    # a resume under today's code must never stamp the current era onto it.
    if existing_era is not None:
        metadata["era"] = existing_era
    elif existing_original_command is None and existing_cli_args is None:
        from utils.era import creation_era_block
        metadata["era"] = creation_era_block()
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
    atomic_write_json(meta_path, metadata, indent=2)


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
    atomic_write_json(meta_path, meta, indent=2)


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
    atomic_write_json(meta_path, meta, indent=2)


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
    sentinel_regime: "dict | None" = None,
    mirrored_pairs: "dict | None" = None,
) -> None:
    """Append one eval cycle's pairwise win-records to ``<model_dir>/eval_results.jsonl``.

    This is the **append-only source of truth** for the ELO / skill-rating fit
    (``agents.training.elo``): each line is the full tournament-matrix row a cycle already
    produced — the trainee (frozen at ``step``) vs every bot and every pool sentinel,
    ``n_games`` each. Append-only (one line per cycle) so it survives launcher restarts,
    unlike the overwritten top-level ``metadata.json:latest_eval``. ``bot_win_rates`` maps
    bot name → trainee win rate; ``sentinels`` is
    ``[{"step": int, "win_rate": float, "counts": [n_won, n_finished]}, …]`` (``counts`` optional;
    empty/omitted on the non-self-play bot-only path).

    ``sentinel_regime`` (optional) is ``{"greedy": bool, "symmetric_teams": bool}`` — the OPPONENT
    REGIME this cycle's sentinel edges were measured under. See the inline note at the write.

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

    ``mirrored_pairs`` (optional) marks a cycle played as MIRRORED TEAM PAIRS
    (``gen3_mirrored_pairs_v1``) and carries each opponent's PENTANOMIAL — ``{"schema", "bots":
    {name: [5 counts]}, "sentinels": {str(step): [5]}, "externals": {label: [5]}}`` — the pair-level
    record every interval on these games must be taken from (the pair, never the game, is the unit).
    Its PRESENCE is the row's regime stamp (``elo.load_rows`` refuses a fit across it); absent = an
    unmirrored row, so an unmirrored writer's rows stay byte-identical.

    Best-effort: never raise into the eval path — a failed append must not break eval.
    """
    try:
        row = {
            "step": int(step),
            "n_games": int(n_games),
            "evaluated_at": datetime.now(timezone.utc).isoformat(),
            "bots": {k: float(v) for k, v in bot_win_rates.items()},
            "sentinels": [
                {"step": int(s["step"]), "win_rate": float(s["win_rate"]),
                 # EXACT W/L per sentinel, for the same reason `counts` exists on the bot side:
                 # recovering it from win_rate x n_games is exact only at full shard coverage, and
                 # the snapshot ladder now CONSUMES these rows (see `sentinel_regime` below), where
                 # a rounded game count would quietly become a rounded edge. Omitted when the caller
                 # supplies none, so an older writer's rows stay byte-identical.
                 **({"counts": [int(s["counts"][0]), int(s["counts"][1])]} if s.get("counts") else {})}
                for s in (sentinels or [])
            ],
        }
        # gen3_eval_sentinel_greedy_default_v1 — THE ROW'S OWN OPPONENT REGIME. Two booleans:
        # `greedy` (the sentinel played argmax, not temperature-sampled) and `symmetric_teams` (both
        # players drew from the LADDER's own builder — full pool, sample-team bias 0.1). Recorded
        # PER ROW rather than per run because a resume can legitimately move the regime and a
        # windowed statistic must never cross that boundary unmarked (rule of evidence 15), and
        # because `agents.training.snapshot_ladder` READS these two to decide whether the cycle
        # already measured a frozen pair under the ladder's own protocol. An absent block means
        # "written before 2026-09-07" and is treated as neither.
        if sentinel_regime:
            row["sentinel_regime"] = {"greedy": bool(sentinel_regime.get("greedy", False)),
                                      "symmetric_teams": bool(sentinel_regime.get("symmetric_teams", False))}
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
        if mirrored_pairs is not None:
            row["mirrored_pairs"] = {
                "schema": str(mirrored_pairs.get("schema", "gen3_mirrored_pairs_v1")),
                **{blk: {str(k): [int(c) for c in v] for k, v in (mirrored_pairs.get(blk) or {}).items()}
                   for blk in ("bots", "sentinels", "externals")}}
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
    atomic_write_json(meta_path, meta, indent=2)


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
    # gen3_reveal_provenance_v1: the sidecar copies the run's reveal mode too; judge it against the run's own
    # model_config.json (a run root without one -- a history-only dir, a test double -- states nothing to judge).
    _cfg = _read_run_json(os.path.join(model_dir, "model_config.json"))
    if _cfg is not None and hparams:
        from agents.model.oracle_reveal import check_provenance, hparams_witnesses
        check_provenance(config=_cfg.get("oracle_reveal", "off"), witnesses=hparams_witnesses(hparams),
                         where=f"checkpoint {checkpoint_path}")
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
    atomic_write_json(_checkpoint_metadata_path(checkpoint_path), entry, indent=2)


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
    enforce_belief_grad_mode: Optional[str] = None,
    allow_belief_grad_mode_change: bool = False,
    enforce_oracle_reveal: Optional[str] = None,
    oracle_reveal_fork: Optional[Dict[str, Any]] = None,
    grow: Optional[Dict[str, Any]] = None,
) -> MaskablePPO:
    """Load a model with a compatibility check against the current architecture.

    The LEARNER's loader: the trainee's resume / fork (strict on every key, the ride-along heads
    included, and the full ``InstrumentedMaskablePPO`` whose ``_setup_model`` acquires the heads'
    optimizers at startup), the round-trip smoke and offline meters. 🚨 An OPPONENT load — the
    self-play pool, an eval sentinel — goes through ``load_opponent_snapshot`` instead: this loader
    acquires training state, which after the learner froze is a K6 FATAL
    (`gen3_opponent_inference_load_v1`).

    Args:
        model_path:      Path to the .zip (with or without extension), or a directory
                         containing final_model.zip or best_model.zip.
        env:             VecEnv to attach to the loaded model.
        current_version: ModelVersion reflecting current code; checked against saved config.
        device:          Passed to InstrumentedMaskablePPO.load().
        tensorboard_log: Passed to InstrumentedMaskablePPO.load().
        enforce_vf_coef: TRAINING-RESUME ONLY. When set, the saved config's vf_coef must
                         match this value or a ModelVersionError is raised — vf_coef is fixed
                         for a run's lifetime. Left None on every other load (the roundtrip
                         smoke, the offline meters), where the value-loss coefficient is
                         irrelevant to the forward.
        grow:            A FORK's DECLARED growth (B-GROW: the lineage stamp's ``fields`` block,
                         ``{field: {"parent": p, "child": c}}``). The gate is then
                         ``check_compatible_grown``: exactly those fields may differ, by exactly
                         that step. The zip still loads STRICTLY at the PARENT's architecture
                         (SB3 rebuilds from the zip's own policy_kwargs); the caller grows it
                         (``agents.model.growth.apply_growth``) after the optimizer realignment.

    Raises:
        ModelVersionError:  If saved config is incompatible with current_version, or (when
                            enforce_vf_coef is set) its vf_coef differs.
        FileNotFoundError:  If no .zip can be found at the resolved path.
    """
    zip_path, config_dir = _resolve_paths(model_path)
    config_path = _run_config_path(config_dir)
    arch_validated = False
    if os.path.exists(config_path):
        saved_version = ModelVersion.from_json_file(config_path)
        if grow:
            current_version.check_compatible_grown(saved_version, grow)
        else:
            current_version.check_compatible(saved_version)
        if enforce_vf_coef is not None:
            saved_version.check_vf_coef(enforce_vf_coef)
        if enforce_reward_config is not None:
            saved_version.check_reward_config(enforce_reward_config)
        if enforce_belief_grad_mode is not None:
            saved_version.check_belief_grad_mode(enforce_belief_grad_mode,
                                                 allow_change=allow_belief_grad_mode_change)
        if enforce_oracle_reveal is not None:
            # v137: the observation's meaning is fixed — a FORK may change it only under the recorded
            # DIAGNOSTIC-FORK consent (`oracle_reveal_fork`, the stamp `main.train.diagnostic_fork` decided)
            saved_version.check_oracle_reveal(enforce_oracle_reveal, diagnostic_fork=oracle_reveal_fork)
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


def _run_config_path(config_dir: str) -> str:
    """``model_config.json`` is run-LEVEL — it sits at the run root, but a checkpoint .zip may live
    one level down in <run>/checkpoints/. Search the zip's dir then its parent (mirroring
    load_foreign_opponent) so the arch check survives the relocation. Returns the zip-dir path when
    neither exists (the caller's legacy-model branch)."""
    config_path = os.path.join(config_dir, "model_config.json")
    if not os.path.exists(config_path):
        parent_config = os.path.join(os.path.dirname(config_dir), "model_config.json")
        if os.path.exists(parent_config):
            config_path = parent_config
    return config_path


def load_opponent_snapshot(model_path: str, current_version: ModelVersion,
                           device: str = "cpu") -> MaskablePPO:
    """Load one of THIS run's snapshots as an OPPONENT — the self-play pool, an eval sentinel
    (`gen3_opponent_inference_load_v1`). An opponent never trains, so the load acquires nothing:

      * the model is an ``InferenceMaskablePPO`` (policy weights only: no optimizer, no ride-along
        optimizer, no rollout buffer), so a load AFTER the learner froze cannot trip K6's freeze guard
        — the X26 FATAL at the first pool load;
      * the gate is ``check_opponent_snapshot_compatible``: ``check_compatible`` with the DECLARED
        ride-along key set ignored in either direction (F-MEM — the heads are detached and no forward
        reads them); every other mismatch is still a hard ``ModelVersionError``.

    The TRAINEE's own resume / fork stays on ``load_model_snapshot`` (strict, full learner)."""
    zip_path, config_dir = _resolve_paths(model_path)
    config_path = _run_config_path(config_dir)
    if os.path.exists(config_path):
        current_version.check_opponent_snapshot_compatible(ModelVersion.from_json_file(config_path))
    else:
        print(
            f"[ModelVersion] WARNING: No model_config.json found at {config_dir!r}. "
            "Skipping compatibility check (legacy model)."
        )
    kwargs: dict = {"env": None, "device": device}
    _patch_historical_floor(zip_path, kwargs)
    return InferenceMaskablePPO.load(zip_path, **kwargs)


# Compiled-opponent machinery lives in `compile_opponents.py` (split 2026-08-16); re-exported
# here because the historical import path is this module.
from agents.model.compile_opponents import (   # noqa: F401
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
    # v131 (deletion pass L1): the distributional value head's atom count and support. They only
    # SIZED or INITIALISED the deleted head (`value_dist_mode`, JUDGED below, carries the decision),
    # so none of them can be ON on its own.
    "value_dist_bins", "value_dist_vmin", "value_dist_vmax",
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
                    # v108 (gen3_dead_flag_purge_v2): the uncertainty-aware P(outspeed) divisor. This
                    # is the OTHER reason for the JUDGED list — not "ON named parameters" but "the
                    # state_dict is BYTE-IDENTICAL either way", so popping True would silently run a
                    # checkpoint under the physics it was not trained on and no shape check could tell.
                    ("threat_prob_outspeed", False),
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
                    ("value_intent", False),
                    # v131 (deletion pass L1): the distributional VALUE head is deleted. 'read_only' /
                    # 'shaping' built a head whose Linear is in the state_dict — PARAMETERS the surviving
                    # extractor has no home for — so ON is refused (the v75 rule); 'none' built nothing
                    # and pops. The reachable half: every v121+ checkpoint pickles `value_dist_mode`.
                    ("value_dist_mode", "none"),
                    # v132 (deletion pass L2): the privileged TRUE-TEAM value route and the DENSE
                    # AUXILIARY head are deleted. Each ON value built a module whose parameters are in
                    # the state_dict (`TrueTeamValueReadout`, `DenseAuxHead`) and the surviving
                    # extractor has no home for them, so ON is refused (the v75 rule); OFF built
                    # nothing and pops. The reachable half: every v121+ checkpoint pickles both bools
                    # (`snapshot.snapshot_extractor_kwargs` always wrote them).
                    ("value_true_team", False), ("dense_aux", False),
                    # v134 (deletion pass L4): the four counterfactual HEADS are deleted — the evidential
                    # Beta head (`CfEvidentialHead`), the twin win-prob heads B / C, the passive shadow
                    # critic and the per-action Q head (`QWinProbHead`). Each ON value built modules
                    # whose parameters are in the state_dict and the surviving extractor has no home
                    # for them, so ON is refused (the v75 rule); OFF built nothing and pops. The
                    # reachable half: every v98+ checkpoint pickles `cf_evidential` / `cf_twin_heads` /
                    # `cf_shadow_critic` and every v107+ one `q_winprob_mode`
                    # (`snapshot.snapshot_extractor_kwargs` always wrote them).
                    ("cf_evidential", False), ("cf_twin_heads", False), ("cf_shadow_critic", False),
                    ("q_winprob_mode", "none"),
                    # v135 (deletion pass P11c): the PAIR-VALUE route is deleted — its ON value built a
                    # zero-init D_MODEL projection (`PairValueInject`) inside CLSPool, a state_dict key
                    # the surviving extractor has no home for, so ON is refused (the v75 rule); OFF
                    # built nothing and pops. The reachable half: every v95+ checkpoint pickles
                    # `pair_value_route` (`snapshot.snapshot_extractor_kwargs` always wrote it).
                    ("pair_value_route", False),
                    # v144 (the X5 VERSION BREAK, gen3_x5_version_break_v1): `belief_tokens` is deleted —
                    # X5's hypothesis tokens are the only belief representation. 'fixed_mass' is the forward
                    # this code builds, so it pops; 'blob' named a forward that no longer exists (BeliefSlots'
                    # constant hidden-slot tokens, the alpha / beta heads in a state_dict) and is REFUSED with
                    # the deletion reason by `version_break.refuse_pickled_belief_tokens`, which
                    # `sanitize_dead_extractor_kwargs` runs BEFORE this table (so the refusal names the
                    # deletion and the pinned fix, not the generic sentence). Reachable: every v136-v143
                    # checkpoint pickles it.
                    ("belief_tokens", "fixed_mass"))

# POLICY kwargs (`policy_kwargs[...]`, NOT the extractor's) that left `Gen3DualHeadMaskablePolicy.__init__`
# at v131 (deletion pass L1). SB3 splats the zip's pickled `policy_kwargs` into the policy constructor, so a
# deleted name TypeErrors every load of a checkpoint that recorded it — and every v121+ checkpoint did
# (`use_popart=False`, `value_from_dist=False`). Same JUDGED rule as the extractor list: the ON value built
# PopArt buffers / routed the critic through the deleted distributional head, a forward this code cannot
# reproduce, so ON is REFUSED and OFF pops. `_migrate_config`'s `retired_levers.refuse_structural` is the
# CONFIG-side twin; the two must agree (pinned by `dead_kwargs_sanitize_test`).
_DEAD_POLICY_KWARGS_JUDGED = (("use_popart", False), ("value_from_dist", False))


def sanitize_dead_extractor_kwargs(fek: dict) -> bool:
    """Drop DELETED keys from a saved `features_extractor_kwargs`. True if it changed.

    Raises `ModelVersionError` — exactly as `_migrate_config` does — when a JUDGED field records a
    value the surviving forward pass cannot reproduce.

    This is the ZIP-side twin of `_migrate_config`'s key handling and both are needed: the config
    JSON drives the version GATE, while these kwargs are what SB3 splats into the extractor
    constructor when it rebuilds the policy. A key dropped from one and not the other either fails
    the gate for the wrong reason or TypeErrors inside the constructor.
    """
    from agents.model.model_version.retired_levers import refuse_retired_values
    from agents.model.model_version.version_break import refuse_pickled_belief_tokens
    changed = refuse_pickled_belief_tokens(fek)     # v144: blob REFUSED with its own reason; fixed_mass pops
    refuse_retired_values(fek)                      # gen3_ko_exact_closed_v1: a retired VALUE, refused with its reason
    # THE LEGACY DELETION (legacy_deletion_manifest_2026-10-12.md, U0): every landed unit's end-state levers are
    # judged by the SAME predicate the config refusal uses (`adopted_levers.lever_active`: a non-adopted value in a
    # network that BUILT the lever is refused; a vacuous / inert / adopted one pops). NOT a `_DEAD_FEK_JUDGED` row —
    # that table holds one supported value per kwarg, and the seven blocks are True in every adopted zip (F-LD-4).
    from agents.model.model_version.adopted_levers import judge_pickled_kwargs
    changed = judge_pickled_kwargs(fek) or changed
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


def sanitize_dead_policy_kwargs(pk: dict) -> bool:
    """Drop DELETED keys from a saved ``policy_kwargs`` (the policy's own, not the extractor's).
    True if it changed. Raises ``ModelVersionError`` when a recorded value is one the surviving
    policy cannot reproduce — the ZIP-side twin of `retired_levers.refuse_structural`."""
    changed = False
    for dead, supported in _DEAD_POLICY_KWARGS_JUDGED:
        if dead in pk:
            if bool(pk[dead]) is not supported:
                raise ModelVersionError(
                    f"policy kwarg {dead}={pk[dead]!r} is no longer supported: the only supported "
                    f"value is {supported!r}.\nThis checkpoint trained under a critic that no longer "
                    "exists in the codebase (PopArt / the distributional value head were deleted, "
                    "deletion pass L1) and cannot be reproduced from HEAD.\n"
                    "To re-read it, use the git_hash recorded in its own metadata.json.")
            pk.pop(dead)
            changed = True
    return changed


def _patch_historical_floor(zip_path: str, kwargs: dict) -> None:
    """Sanitize a saved `policy_kwargs` so an older checkpoint can be RECONSTRUCTED.

    Three independent fixes share this hook because all are "the zip records something the live
    constructor no longer accepts", and all are read from the same one zip read:

    1. `move_candidate_floor` (below).
    2. Extractor kwargs deleted from the constructor (`sanitize_dead_extractor_kwargs`).
    3. POLICY kwargs deleted from `Gen3DualHeadMaskablePolicy.__init__`
       (`sanitize_dead_policy_kwargs`: `use_popart`, `value_from_dist`).

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

        from agents.training.instrumented_ppo.strict_load import never_unpickled
        data, _, _ = load_from_zip_file(zip_path, device="cpu", print_system_info=False,
                                        custom_objects=never_unpickled())
    except Exception:
        return                                  # unreadable here → let SB3's own load report it
    pk = (data or {}).get("policy_kwargs") or {}
    changed = sanitize_dead_policy_kwargs(pk)
    fek = pk.get("features_extractor_kwargs")
    if not isinstance(fek, dict):
        if changed:
            kwargs.setdefault("custom_objects", {})["policy_kwargs"] = pk
        return
    if "move_candidate_floor" in fek:
        before = fek["move_candidate_floor"]
        sanitize_historical_move_floor(fek)
        changed |= fek["move_candidate_floor"] != before
    # `|=` not `or`: short-circuiting would skip the dead-kwarg strip whenever the floor already
    # needed patching, which is precisely the pre-v65 checkpoints that ALSO carry the dead keys.
    changed |= sanitize_dead_extractor_kwargs(fek)
    if changed:
        kwargs.setdefault("custom_objects", {})["policy_kwargs"] = pk


def historical_load_kwargs(zip_path: str) -> dict:
    """The extra ``MaskablePPO.load`` kwargs (a sanitized ``custom_objects``) a BARE load of
    `zip_path` needs: SB3 splats the zip's pickled ``policy_kwargs`` into the live constructors, so a
    kwarg deleted since the checkpoint was written (PopArt, the value-dist head — deletion pass L1) is
    stripped, or REFUSED when it was ON. ``{}`` when nothing needs changing. For the callers that
    cannot use ``load_foreign_opponent`` (`play.py`'s ladder session)."""
    kwargs: dict = {}
    _patch_historical_floor(zip_path, kwargs)
    return kwargs


def load_checkpoint_strict(path: str, *, device: str = "cpu",
                           custom_objects: Optional[Dict[str, Any]] = None) -> "MaskablePPO":
    """THE loader for every READER of a checkpoint that is neither the trainee's resume
    (``load_model_snapshot``) nor an opponent (``load_opponent_snapshot`` / ``load_foreign_opponent``):
    the ladder session (``play.py``), the prober, the eval worker, the offline meters, the fine-tune
    graft (`gen3_strict_checkpoint_load_v1`, P10 follow-up F1).

    A bare ``MaskablePPO.load`` retries with ``exact_match=False`` whenever the strict error names
    ``pi_features_extractor`` — any missing extractor key does, through the alias — and loads the rest
    with that submodule at FRESH INIT behind one warning (P10 F4). This builds a ``StrictMaskablePPO``,
    a plain ``MaskablePPO`` whose ``set_parameters`` is always strict (the SAME mixin the learner and the
    opponent classes carry), so such a checkpoint is a typed ``StrictLoadError``. Nothing else differs
    from a bare load (the optimizer is built, the seed applied, ``save`` works): a caller that wants a
    policy-only model uses ``load_foreign_opponent``.

    ``custom_objects`` is the caller's sanitizer's output (``historical_load_kwargs`` for a ladder
    session, the prober's ``sanitized_load_custom_objects`` for a forensic read). No module outside
    the strict classes calls ``MaskablePPO.load`` / ``PPO.load``: ``strict_checkpoint_load_gate_test``
    fails one, with an empty allowlist.

    The policy comes back in EVAL mode: a reader reads the function the rollout, the opponent pool and
    every other reader read. Only in eval mode does ``nn.MultiheadAttention`` take its fused fast path
    (first at ``hidden_opp_belief.decoder.self_attn``), a different kernel: the logits moved by up to
    4.8e-7 on 18 of 256 real rows (X59-S's CUDA pre-flight, 2026-10-26, where the anchors bare
    load failed the student's bit-equal export check). A caller that trains sets the mode itself (sb3's
    ``train`` does)."""
    refuse_deleted_pickled_kwargs(path)
    kwargs: Dict[str, Any] = {"env": None, "device": device}
    if custom_objects:
        kwargs["custom_objects"] = custom_objects
    model = StrictMaskablePPO.load(path, **kwargs)
    model.policy.set_training_mode(False)
    return model


def refuse_deleted_pickled_kwargs(zip_path: str) -> None:
    """Refuse a checkpoint whose PICKLED extractor kwargs name a deleted forward, BEFORE any loader
    splats them into the live constructor (`gen3_x5_version_break_v1`).

    `load_checkpoint_strict` takes ``custom_objects`` from its caller, and a caller whose sanitizer
    NEVER refuses (the prober's set-math `sanitized_load_custom_objects`) or that passes none at all
    would otherwise hand SB3 a blob checkpoint's ``belief_tokens='blob'`` — a bare ``TypeError`` (no
    such kwarg), or, once dropped, a state_dict with no home (the deleted ``alpha_head`` / ``belief_slots`` keys).
    The refusal is `version_break`'s typed ``ModelVersionError`` naming the deletion and the pinned
    commit. ONLY ``policy_kwargs`` is deserialized (never the rest of ``data``: an unpickled CUDA tensor
    there would create a CUDA context before the strict load's own `CudaContextOnCpuLoad` check could see
    it). A zip this cannot read is left to SB3's own load to report."""
    try:
        import json
        import zipfile

        from stable_baselines3.common.save_util import json_to_data
        path = zip_path if zip_path.endswith(".zip") or os.path.exists(zip_path) else zip_path + ".zip"
        with zipfile.ZipFile(path) as z:
            raw = json.loads(z.read("data").decode())
        data = json_to_data(json.dumps({"policy_kwargs": raw.get("policy_kwargs", {})}))
    except Exception:
        return
    fek = ((data or {}).get("policy_kwargs") or {}).get("features_extractor_kwargs")
    if isinstance(fek, dict) and "belief_tokens" in fek:
        from agents.model.model_version.version_break import refuse_pickled_belief_tokens
        refuse_pickled_belief_tokens(dict(fek))


def load_foreign_opponent(
    model_path: str,
    current_version: ModelVersion,
    device: str = "cpu",
    config_path: Optional[str] = None,
    inference_only: bool = True,
) -> "tuple[MaskablePPO, ModelVersion]":
    """Load a frozen model from ANOTHER run as an inference-only OPPONENT ("stable opponent").

    INFERENCE ONLY by default (`gen3_opponent_inference_load_v1`): the model is an
    ``InferenceMaskablePPO`` — policy weights, no optimizer, no ride-along optimizer, no rollout
    buffer — so a load inside a FROZEN learner (an exploiter target, a stable
    opponent at a restart) acquires nothing. ``inference_only=False`` is for an OFFLINE tool that
    fits the loaded model itself (the consensus warm-start's student); it builds the full learner.

    Unlike ``load_model_snapshot`` — which checks the saved config against the LIVE trainee via
    ``check_compatible`` (a hard FATAL on any ``_WEIGHT_FIELD`` mismatch) — a stable
    cross-run opponent is validated for OBSERVATION-FAMILY compatibility ONLY
    (``ModelVersion.check_opponent_compatible`` = same ``arch_signature``): it never shares weights
    with the trainee and never reads its value head, so ``vf_coef`` / reward-config
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
    cls = InferenceMaskablePPO if inference_only else InstrumentedMaskablePPO
    return cls.load(zip_path, **load_kwargs), foreign_version


def current_model_version(
    mappings: Dict[str, Any],
    *,
    attend_unrevealed_opponents: bool = False,
    opp_belief_cls_k: int = 0,
    opp_belief_slots: bool = False,
    critic: str = "shaped",
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
    opp_intent: bool = False,
    species_prior_fusion: bool = False,
    t0_species_prior: bool = False,
    opp_intent_grad_mode: str = "detached",
    intent_value_reduce: bool = False,
    value_entity_pool: bool = False,
    value_entity_pool_full: bool = False,
    history_events: bool = False,
    item_belief: bool = False,
    op_drop_renders: bool = False,
    op_believed_lean: bool = False,
    value_clock: bool = False,
    value_intent: bool = False,
    damage_topk_k: int = 0,
    damage_matrices_outgoing: bool = False,
    damage_matrices_incoming: bool = False,
    hp_type_belief_coef: float = 0.0,
    item_belief_coef: float = 0.0,
    hp_belief_mode: str = "composed",
    belief_grad_mode: str = "shaping",
    ridealong_ensemble: int = 0,
    ridealong_rnd: bool = False,
    ridealong_adv: int = 0,
    ridealong_opp: int = 0,
    ridealong_rnd_variants: str = "off",
    oracle_reveal: str = "off",
    policy_readout: str = "tower",
    token_encoding: str = "legacy",
    obs_facts: str = "off",
    mon_hazard_cost: str = "off",
    move_actor_state: str = "off",
    trunk_layers: int = 2,
    drop_progress_clock: str = "off",
    move_target_state: str = "off",
    move_set_closure: str = "off",
    tau_solve: str = "fp32_alt",
    trunk_ffn_by_kind: str = "off",
    vf_coef: float = 0.5,
    reward_config: Any = None,               # duck-typed, like ModelVersion.build
) -> ModelVersion:
    """Build a ``ModelVersion`` reflecting the CURRENT RUN's architecture for ``mappings``.

    Single source for the ``from_layout_and_policy_kwargs`` construction otherwise
    repeated inline (train_rl_agent's self-play pool setup; the eval worker's sentinel
    version check). Returns the ``current_version`` to pass to ``load_model_snapshot`` /
    ``SnapshotPool`` so a stale-arch snapshot fails with a clean ``ModelVersionError``
    instead of loading mismatched weights.

    **The architecture TOGGLES must be passed in.** They default off (the encoder's
    ``get_features_extractor_kwargs`` carries no CLI toggles), but a run that enables any of them
    (e.g. ``--opp-belief-aux-coef>0`` → ``opp_belief_slots``,
    ``--attend-unrevealed-opponents``) MUST thread its real values here — otherwise the gate compares
    a toggle-OFF "current" version against the run's own toggle-ON snapshots and FATALs on every
    pool/eval load it is meant to protect.

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
    # gen3_ridealong_heads_v1 (v126): same category — the heads' params are in the (policy's)
    # state_dict and nothing consumes their output, so the recorded toggles are all a load gate can
    # compare; a frozen eval/pool opponent's gate must see them or a baseline run FATALs loading its
    # OWN sentinels.
    ext_kwargs["ridealong_ensemble"] = int(ridealong_ensemble)
    ext_kwargs["ridealong_rnd"] = bool(ridealong_rnd)
    ext_kwargs["ridealong_adv"] = int(ridealong_adv)
    ext_kwargs["ridealong_opp"] = int(ridealong_opp)
    ext_kwargs["ridealong_rnd_variants"] = str(ridealong_rnd_variants)
    # gen3_oracle_reveal_v1 (v137): the diagnostic observation mode — resume-immutable, so not in
    # `check_compatible`, but recorded so a worker / an offline tool reads the run's own value.
    ext_kwargs["oracle_reveal"] = str(oracle_reveal)
    # gen3_policy_readout_trunk_v1 (v138, audit F2): where the policy's decision context comes from —
    # structural, so a frozen opponent's gate must see it.
    ext_kwargs["policy_readout"] = str(policy_readout)
    # gen3_static_tokens_v1 (v139): the per-Pokemon token encoding — structural, so a frozen opponent's
    # gate must see it.
    ext_kwargs["token_encoding"] = str(token_encoding)
    # gen3_obs_facts_v1 (v144, part 3): the OBS-FACTS consumer — structural (`v1` builds a module).
    ext_kwargs["obs_facts"] = str(obs_facts)
    # gen3_static_port_v1 (v147): the static encoder's two narrow facts — structural.
    ext_kwargs["mon_hazard_cost"] = str(mon_hazard_cost)
    ext_kwargs["move_actor_state"] = str(move_actor_state)
    # gen3_static_recovery_v1 (v150): the trunk depth — structural.
    ext_kwargs["trunk_layers"] = int(trunk_layers)
    # gen3_endstate_facts_v1 (v152): the surviving fact-completion levers — structural.
    ext_kwargs["drop_progress_clock"] = str(drop_progress_clock)
    # gen3_probe_facts_v1 (v153): their active's state on our move seats — structural.
    ext_kwargs["move_target_state"] = str(move_target_state)
    # gen3_move_set_closure_v1 (v154): the four-move fact — structural.
    ext_kwargs["move_set_closure"] = str(move_set_closure)
    # gen3_tau_fp32_alt_v1 (v160): the τ solve — numerics, recorded, never compared.
    ext_kwargs["tau_solve"] = str(tau_solve)
    # gen3_trunk_ffn_by_kind_v1 (v161, B-KFFN): the per-kind FFN experts — structural.
    ext_kwargs["trunk_ffn_by_kind"] = str(trunk_ffn_by_kind)
    ext_kwargs["opp_intent"] = opp_intent
    ext_kwargs["species_prior_fusion"] = species_prior_fusion
    ext_kwargs["t0_species_prior"] = t0_species_prior
    ext_kwargs["opp_intent_grad_mode"] = opp_intent_grad_mode
    ext_kwargs["intent_value_reduce"] = intent_value_reduce
    ext_kwargs["value_entity_pool"] = value_entity_pool
    ext_kwargs["value_entity_pool_full"] = value_entity_pool_full
    ext_kwargs["history_events"] = history_events
    ext_kwargs["item_belief"] = item_belief
    ext_kwargs["op_drop_renders"] = op_drop_renders
    ext_kwargs["op_believed_lean"] = op_believed_lean
    ext_kwargs["value_clock"] = value_clock
    ext_kwargs["value_intent"] = value_intent
    ext_kwargs["damage_topk_k"] = damage_topk_k
    ext_kwargs["damage_matrices_outgoing"] = damage_matrices_outgoing
    ext_kwargs["damage_matrices_incoming"] = damage_matrices_incoming
    ext_kwargs["hp_belief_mode"] = hp_belief_mode
    ext_kwargs["belief_grad_mode"] = belief_grad_mode
    policy_kwargs = {
        "features_extractor_class": Gen3FeaturesExtractor,
        "features_extractor_kwargs": ext_kwargs,
        "net_arch": NET_ARCH,
        # gen3_winprob_critic_mode_v1: the critic ROUTE, so a frozen eval/pool/sentinel opponent's
        # load gate sees it.
        "critic": critic,
    }
    return ModelVersion.from_layout_and_policy_kwargs(
        ext_kwargs["layout"], policy_kwargs, vf_coef=vf_coef, reward_config=reward_config,
        opp_belief_aux_coef=opp_belief_aux_coef,
        move_belief_coef=move_belief_coef,
        move_belief_latent_coef=move_belief_latent_coef,
        spread_belief_coef=spread_belief_coef,
        hp_type_belief_coef=hp_type_belief_coef, item_belief_coef=item_belief_coef,
    )


#: `current_model_version` keywords that are NOT architecture toggles: the mappings, the loss coefficients
#: and the reward block. Every OTHER keyword is a toggle, read by :func:`arch_toggles_from_model`.
_NON_TOGGLE_KWARGS = frozenset({
    "mappings", "opp_belief_aux_coef", "move_belief_coef", "move_belief_latent_coef", "spread_belief_coef",
    "hp_type_belief_coef", "item_belief_coef", "vf_coef", "reward_config",
})


def recorded_policy_kwargs(model: Any) -> Dict[str, Any]:
    """The policy kwargs THIS model was BUILT from, as SB3 records them (``model.policy_kwargs``: the trainer's
    own dict on a fresh run, the zip's saved one on a load). Their ``features_extractor_kwargs`` carry every
    architecture toggle the run RESOLVED (arch-arm overlays and implied fields included), which is exactly what
    the run's ``model_config.json`` and every snapshot it wrote record. A model without them is refused (typed):
    rebuilding the version any other way is the bug this replaces."""
    pk = getattr(model, "policy_kwargs", None)
    if not isinstance(pk, dict) or not isinstance(pk.get("features_extractor_kwargs"), dict):
        raise ModelVersionError(
            "this model records no policy_kwargs['features_extractor_kwargs'], so its RECORDED architecture is "
            "unknown — a compat gate will not rebuild one from the built modules (gen3_recorded_version_gate_v1)")
    return pk


def arch_toggles_from_model(model: Any) -> dict:
    """THIS run's architecture TOGGLES, JSON-serializable for a worker cfg and splattable into
    ``current_model_version`` — read from the RECORDED policy kwargs (:func:`recorded_policy_kwargs`), NEVER off
    the built modules (gen3_recorded_version_gate_v1, 2026-10-11).

    A built-module read is NOT the recorded configuration: under the end state's adoption the policy BUILT the
    seven per-action blocks and then RETIRED them (since deleted, legacy deletion U3), so a module read said False
    where the run's ``model_config.json`` and every snapshot it wrote said True — the ``--arch endstate`` seed
    ``rb_es_end_s2001`` FATALed loading its own sentinels at its first pool eval (``intent_move_cell mismatch:
    saved=True, current=False``), and ``main.h2h`` could not declare an engine for its checkpoints. The rule is
    general (any construct-then-retire module, any retired policy tower), so it stays.

    The key set is DERIVED: every ``current_model_version`` keyword but :data:`_NON_TOGGLE_KWARGS`, each
    defaulting to that keyword's default when the recorded kwargs predate it (the extractor builds the same
    default) — a new toggle cannot be forgotten here. ``critic`` is a policy kwarg."""
    import inspect

    pk = recorded_policy_kwargs(model)
    fek = pk["features_extractor_kwargs"]
    out: Dict[str, Any] = {}
    for k, p in inspect.signature(current_model_version).parameters.items():
        if k in _NON_TOGGLE_KWARGS:
            continue
        d = p.default
        v = fek.get(k, d)
        out[k] = d if v is None else (type(d)(v) if d is not None else v)
    out["critic"] = str(pk.get("critic", "shaped") or "shaped")
    return out


def recorded_model_version(model: Any, mappings: Optional[Dict[str, Any]] = None) -> ModelVersion:
    """THE one helper every compat / version comparison against a LIVE model goes through: the ``ModelVersion``
    the model's RECORDED policy kwargs build (``ModelVersion.from_layout_and_policy_kwargs`` over this tree's
    layout, as the trainer builds the version it saves), never one rebuilt from built-module reads
    (:func:`arch_toggles_from_model` docs). Its gated fields equal the run's ``model_config.json``'s."""
    from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings

    pk = recorded_policy_kwargs(model)
    layout = Gen3ObservationEncoder(mappings if mappings is not None else load_mappings()) \
        .get_features_extractor_kwargs()["layout"]
    return ModelVersion.from_layout_and_policy_kwargs(layout, pk, vf_coef=float(getattr(model, "vf_coef", 0.5)))


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
