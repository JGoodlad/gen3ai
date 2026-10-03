"""Phase 2 — THE MATCHUP: who plays whom, on which teams.

`MatchupSpec.from_args` declares the matchup ONCE and both teambuilders come from it (trainee and
opponent independent BY CONSTRUCTION — the mirror-bug class). Everything else here is the opponent
side of the same question: the heuristic roster and its `--bot-weights` vector, the cross-run
`--stable-opponents`, the single `--exploiter` target, and the self-play curriculum thresholds.

Every FATAL in here exits `FATAL_CONFIG` rather than raising: these are non-recoverable config
errors that would fail identically on every retry, so the launcher must give up instead of
restarting into them.
"""
import dataclasses
import os
import sys
from typing import Any, List, Optional

from agents.model.model_version import ModelVersionError
from agents.opponents import (
    Gen3AggressivePlayer, Gen3AggressiveV2Player, Gen3HeuristicV2Player, Gen3SetupSweepPlayer,
    Gen3SetupSweepV2Player, Gen3StallerPlayer, Gen3StallerV2Player,
)
from agents.observation.state_encoder import load_mappings
from agents.training.eval_callback import opponent_name
from agents.training.matchup_spec import MatchupSpec
from agents.training.snapshot_pool import HEURISTIC_FLOOR, SELF_PLAY_FULL, SELF_PLAY_START
from main.exit_codes import FatalConfigError, TrainExitCode
from main.launcher.ipc import emit
from main.train.run_io import _run_arch_toggles
from poke_env.player import SimpleHeuristicsPlayer
from utils.team_loader import TeamLoader


@dataclasses.dataclass
class MatchupSetup:
    """Everything downstream needs about WHO is playing and with WHAT."""

    matchup: Any
    mappings: Any
    trainee_teambuilder: Any
    opponent_teambuilder: Any
    specialist_team_str: Any
    opponent_classes: List[Any]
    bot_weight_vec: Optional[List[float]]
    fixed_opponents: List[Any]
    exploiter_entry: Any
    heuristic_floor: float
    sp_start_wr: float
    sp_full_wr: float
    promote_threshold: float


class BotWeightsRejected(FatalConfigError):
    """A malformed `--bot-weights` → ``FATAL_CONFIG`` (3), never CRASH (1)."""


def resolve_bot_weights(spec: str, roster: "List[str]") -> "List[float]":
    """`--bot-weights 'name=w,...'` → a roster-aligned weight vector (unlisted → 1.0).

    gen3_supply_guard_v2: every refusal here is a CONFIG error a restart would hit identically, so
    it raises :class:`BotWeightsRejected` (→ FATAL_CONFIG through `exit_codes.exit_code_for`). It
    used to `sys.exit(1)` (an unknown name, a token with no `=`) or let `float()` raise a bare
    ValueError — both read as CRASH, so the launcher restarted the run into the same typo until its
    rapid-crash breaker gave up. A NEGATIVE, non-finite or all-zero vector is refused too: the rust
    env core refused it with a ValueError (CRASH again), and the python env would sample from it."""
    import math
    overrides = {}
    for tok in spec.split(","):
        if not tok.strip():
            continue
        name, sep, val = tok.partition("=")
        if not sep:
            raise BotWeightsRejected(f"[Opponents] FATAL: --bot-weights token {tok!r} is not "
                                     f"name=weight")
        try:
            w = float(val)
        except ValueError:
            raise BotWeightsRejected(f"[Opponents] FATAL: --bot-weights token {tok!r}: the weight "
                                     f"is not a number") from None
        if not math.isfinite(w) or w < 0.0:
            raise BotWeightsRejected(f"[Opponents] FATAL: --bot-weights token {tok!r}: a weight "
                                     f"must be finite and >= 0")
        overrides[name.strip()] = w
    bad = set(overrides) - set(roster)
    if bad:
        raise BotWeightsRejected(f"[Opponents] FATAL: unknown --bot-weights names {sorted(bad)} "
                                 f"(valid: {sorted(set(roster))})")
    vec = [overrides.get(n, 1.0) for n in roster]
    if not sum(vec) > 0.0:
        raise BotWeightsRejected("[Opponents] FATAL: --bot-weights gives every training bot weight "
                                 "0 — there is no bot to draw")
    return vec


def _planned_run_dir(args) -> "str | None":
    """The run dir this process WILL write, as far as it is known before `run_io` resolves it:
    `--run-dir` (a launcher restart), else `<archive>/<--run-name>` (`utils.paths.run_archive_dir()`,
    the same directory `run_io._resolve_fresh_model_dir` will pick — a cwd-relative `models/<name>`
    here would disagree with it from a worktree), else None (a timestamped or exploiter-derived dir,
    which a `--model` can never already live inside, or no archive — the Directory Setup refuses that
    one itself)."""
    if getattr(args, "run_dir", None):
        return args.run_dir
    if getattr(args, "run_name", None):
        from utils.paths import RunArchiveError, run_archive_dir
        try:
            return os.path.join(str(run_archive_dir()), args.run_name)
        except RunArchiveError:
            return None
    return None


def _checkpoint_run_name(model_path: str) -> str:
    """The run a checkpoint belongs to: `<run>/checkpoints/x.zip` or `<run>/x.zip` → `<run>`."""
    d = os.path.dirname(os.path.abspath(model_path))
    if os.path.basename(d) == "checkpoints":
        d = os.path.dirname(d)
    return os.path.basename(d)


def matchup_drift_header(model_path: str, run_dir: "str | None", new_hash: str, rec_hash: str) -> str:
    """The `⚠️ [MATCHUP DRIFT]` headline, worded for what the launch IS.

    A same-run RESTART (`--model` is this run's own checkpoint) whose matchup changed is a mid-run
    curriculum change. A FORK (`--model` is another run's checkpoint) changing the matchup is the
    ordinary case — every exploiter fork of a self-play parent does — so it names the PARENT whose
    recorded matchup is being compared, instead of calling the fork "this resume". Pure → tested.
    """
    from main.train.fork_lr import is_same_run_checkpoint
    tail = ("Metrics across the change are NOT comparable (a new era lands in "
            "metadata.json:matchup_history).")
    if run_dir and is_same_run_checkpoint(model_path, run_dir):
        return (f"⚠️ [MATCHUP DRIFT] this RESTART declares matchup {new_hash} but the run last "
                f"recorded {rec_hash} — the TRAINING DISTRIBUTION IS CHANGING mid-run. " + tail)
    return (f"⚠️ [MATCHUP DRIFT] this FORK of {_checkpoint_run_name(model_path)} declares matchup "
            f"{new_hash}; the parent recorded {rec_hash} — the fork trains on a DIFFERENT "
            f"distribution than its parent did. " + tail)


def build_matchup_and_opponents(args) -> MatchupSetup:
    """Load the team pool, declare the matchup, and resolve every opponent source."""
    # Load all teams using the new TeamLoader
    loader = TeamLoader()
    sample_teams = loader.get_sample_teams()
    all_teams = loader.get_all_teams()
    
    emit(f"📦 {len(sample_teams)} sample teams (bias) / {len(all_teams)} total loaded")

    # THE MATCHUP — declared ONCE (`MatchupSpec.from_args`, designs/ai_v8/design_matchup_config.md)
    # and consumed everywhere: BOTH teambuilders come from the spec (trainee/opponent independent BY
    # CONSTRUCTION — the mirror-bug class), the eval callbacks get the trainee pin from it, the
    # Events panel echoes it, and metadata.json records it (+ spec_hash, the measurement-regime tag).
    # SPECIALIST MODE (--trainee-team) pins ONLY the trainee source; opponents keep the full pool.
    matchup = MatchupSpec.from_args(args)
    # EXPLOITER team-source guarantee: an exploiter may ONLY EVER pilot a vetted sample team (the
    # curated, tournament-proven set) — never a bulk-downloaded `other` team. FATAL otherwise (a
    # deliberate startup gate, like the stable-opponent arch check). Non-exploiter / unpinned runs
    # are unaffected; the existing TSS specialist pin IS a sample team, so it passes.
    try:
        from agents.training.matchup_spec import validate_exploiter_trainee_is_sample
        validate_exploiter_trainee_is_sample(matchup, sample_teams)
    except ValueError as _e:
        print(f"\n[Exploiter] FATAL: {_e}")
        sys.stdout.flush()
        os._exit(int(TrainExitCode.FATAL_CONFIG))
    # UNTAUGHT-SLICE guarantee (gen3_untaught_teacher_guard_v1): a pinned trainee team must not be
    # a member of the UNTAUGHT 8 — the off-slice meter's own slice. Caught by LUCK on 2026-09-20,
    # one step before ~14 GPU-h of contaminated teachers; matching is by CONTENT sha, so a renamed
    # copy is caught too. Applies to every pinned trainee, not only exploiters: today's specialist
    # is tomorrow's teacher.
    if getattr(args, "allow_untaught_teacher", False):
        print("⚠️ [Untaught] --allow-untaught-teacher: SKIPPING the untaught-slice gate — the "
              "trainee may pilot a team the off-slice meter measures. Say so wherever the number "
              "is reported.")
    else:
        try:
            from agents.training.matchup_spec import validate_trainee_not_untaught
            validate_trainee_not_untaught(matchup)
        except ValueError as _e:
            print(f"\n[Untaught] FATAL: {_e}")
            sys.stdout.flush()
            os._exit(int(TrainExitCode.FATAL_CONFIG))
    # → eval callbacks (trainee_team_str). Read from EVAL_trainee_teams (not trainee_teams); a
    # `pin_multi` source yields a LIST (eval samples
    # among them, exactly as training does), a single pin yields the raw export, else None = pool.
    _ets = matchup.eval_trainee_teams
    _specialist_team_str = (list(_ets.pin_strs) if _ets.kind == "pin_multi" and _ets.pin_strs
                            else _ets.pin_str)
    trainee_teambuilder = matchup.trainee_teams.build(all_teams, sample_teams)
    # Team-blocked episodes: hold each drawn trainee team for N consecutive episodes — the
    # per-team gradient-density counter to the measured FiLM sample starvation. Trainee side ONLY
    # (opponent draws stay per-episode); 1 = off, byte-identical. Training-only, not version-locked.
    if args.team_block_episodes > 1:
        trainee_teambuilder.set_block_episodes(args.team_block_episodes)
    opponent_teambuilder = matchup.opponent_teams.build(all_teams, sample_teams)
    for _ln in matchup.summary_lines():
        emit(_ln)
    if matchup.trainee_teams.kind == "pin_multi":
        _tt = matchup.trainee_teams
        emit(f"🎯 [MULTI-SPECIALIST] trainee pinned to {len(_tt.pin_strs)} teams (sampled uniformly): "
             f"{', '.join(os.path.basename(f) for f in _tt.pin_files)} (opponents keep the full pool)")
    elif matchup.trainee_teams.pin_str:
        # Read the TRAINING pin, not `_specialist_team_str` — that one is EVAL-derived and may be a
        # LIST. Calling `.splitlines()` on it would crash the launch at startup.
        _spec_mons = [ln.split("@")[0].split("(")[0].strip()
                      for ln in matchup.trainee_teams.pin_str.splitlines()
                      if ln.strip() and "@" in ln]
        emit(f"🎯 [SPECIALIST] trainee pinned to ONE team from {args.trainee_team}: "
             f"{', '.join(_spec_mons)} (opponents keep the full pool)")

    # RESUME MATCHUP-DRIFT GUARD: matchup flags (--trainee-team/--exploiter/--bot-weights/…) are
    # NOT resume-immutable — a mid-run curriculum change is legitimate — but it must never be
    # SILENT: a resume whose declared matchup differs from what the run last recorded overwrites
    # cli_args and changes the training distribution. Warn LOUDLY with the field diff; the new era
    # is appended to metadata `matchup_history` at the next save (save_model_snapshot), so the
    # run's full regime timeline survives. (A launcher restart forwards flags verbatim → no drift.)
    if args.model:
        from agents.model.snapshot import read_recorded_matchup
        from agents.training.matchup_spec import describe_drift
        _rec_hash, _rec_spec = read_recorded_matchup(args.model)
        if _rec_hash and _rec_hash != matchup.spec_hash():
            emit(matchup_drift_header(args.model, _planned_run_dir(args),
                                      matchup.spec_hash(), _rec_hash))
            for _d in describe_drift(_rec_spec, matchup.to_dict()):
                emit(f"   ⚠️ {_d}")

    mappings = load_mappings()

    # Training heuristic opponents — ALL eight archetype bots (both v1 and v2 of each).
    # They play differently and the extra playstyle diversity is the point. Random is NOT
    # here (it's the eval-only "is the model broken" floor).
    OPPONENT_CLASSES = [
        SimpleHeuristicsPlayer,
        Gen3HeuristicV2Player,
        Gen3StallerPlayer,
        Gen3StallerV2Player,
        Gen3AggressivePlayer,
        Gen3AggressiveV2Player,
        Gen3SetupSweepPlayer,
        Gen3SetupSweepV2Player,
    ]
    # gen3_baitbot_roster_v1: BaitBot joins the roster only when a share is declared, so the
    # default pool is byte-identical. It is appended BEFORE --bot-weights is parsed so a single
    # code path builds the weight vector and the two cannot disagree.
    _baitbot_cls = None
    if getattr(args, "bait_bot_share", 0.0) > 0:
        from agents.baitbot import make_baitbot_class
        from agents.training.eval_callback import _OPPONENT_NAMES
        _baitbot_cls = make_baitbot_class(args.bait_bot_p)
        _OPPONENT_NAMES[_baitbot_cls] = "baitbot"   # TB keys / --bot-weights use the short name
        OPPONENT_CLASSES.append(_baitbot_cls)
    print(f"[Opponents] training pool = {len(OPPONENT_CLASSES)} bots "
          f"({', '.join(opponent_name(c) for c in OPPONENT_CLASSES)})")

    # Resolve --bot-weights (name=weight) into a roster-aligned vector (unlisted → 1.0). None →
    # uniform (current behavior, byte-for-byte). Validated here so a typo fails fast at startup.
    _bot_weight_vec = None
    if args.bot_weights:
        _bot_weight_vec = resolve_bot_weights(args.bot_weights,
                                              [opponent_name(c) for c in OPPONENT_CLASSES])
        print(f"[Opponents] heuristic weights = "
              f"{ {opponent_name(c): w for c, w in zip(OPPONENT_CLASSES, _bot_weight_vec)} }")

    if _baitbot_cls is not None:
        from agents.baitbot import weight_for_share
        if _bot_weight_vec is None:
            _bot_weight_vec = [1.0] * len(OPPONENT_CLASSES)
        _others = [w for c, w in zip(OPPONENT_CLASSES, _bot_weight_vec) if c is not _baitbot_cls]
        _w = weight_for_share(args.bait_bot_share, _others)
        _bot_weight_vec[OPPONENT_CLASSES.index(_baitbot_cls)] = _w
        _realized = _w / (sum(_others) + _w)
        print(f"[Opponents] BaitBot p_bait={args.bait_bot_p} weight={_w:.4f} "
              f"-> realized share {_realized:.4f} (declared {args.bait_bot_share})")

    # Resolve + VALIDATE --stable-opponents (cross-run fixed opponents) at startup. Each foreign
    # model must share THIS run's arch_signature (= observation layout) — a mismatch is a
    # NON-RECOVERABLE config error: exit FATAL_CONFIG so the launcher gives up immediately (the
    # same path a checkpoint arch mismatch takes) and the TUI shows the fatal, instead of
    # auto-restarting into the identical failure.
    _fixed_opponents = []
    if args.stable_opponents:
        from agents.training.fixed_opponent_pool import resolve_stable_opponents
        from agents.model.snapshot import (
            current_model_version as _current_model_version, load_foreign_opponent)
        _cv_stable = _current_model_version(mappings, **_run_arch_toggles(args))
        try:
            _fixed_opponents = resolve_stable_opponents(
                args.stable_opponents, _cv_stable, default_temperature=args.stable_opponent_temp,
            )
            # Validate the WEIGHTS actually load here in the main process (resolve only reads the
            # config). A valid config + corrupt/unreadable zip would otherwise pass the gate and
            # crash every env worker → crash-restart loop. Load once on CPU and discard.
            for _e in _fixed_opponents:
                load_foreign_opponent(_e.zip_path, current_version=_cv_stable, device="cpu",
                                      config_path=_e.config_path)
        except (ModelVersionError, FileNotFoundError, ValueError) as e:
            print(f"\n[StableOpponent] FATAL: {e}")
            sys.stdout.flush()  # os._exit() skips buffer flushing — make sure the reason reaches the log
            os._exit(int(TrainExitCode.FATAL_CONFIG))
        except Exception as e:  # noqa: BLE001 — a corrupt/unreadable foreign weights zip
            print(f"\n[StableOpponent] FATAL: failed to load stable opponent weights: {e}")
            sys.stdout.flush()
            os._exit(int(TrainExitCode.FATAL_CONFIG))
        # emit() → the launcher Events panel (like the [SELFPLAY] startup lines); print()s standalone.
        # A specialist opponent shows its fold-back pin (it pilots ITS OWN team, training + eval).
        _stable_labels = ", ".join(
            e.label + (f" [pilots ITS OWN pin: {os.path.basename(e.team_file)}]" if e.team_str else "")
            for e in _fixed_opponents)
        # WHICH FILE each opponent resolved to (gen3_last_snapshot_resolution_v1) — a bare run dir
        # names a RUN, not a file, and the rung that picked it is stated rather than inferred.
        _stable_prov = [f"   {e.label}: {e.provenance()}" for e in _fixed_opponents]
        if args.self_play:
            emit(f"🐴 [STABLE] {len(_fixed_opponents)} cross-run opponent(s): {_stable_labels} — "
                 f"eval greedy; training ≤{args.stable_opponent_selfplay_share:.0%} of self-play until "
                 f"mastered (win_rate ≥ {args.stable_opponent_mastered_wr:.0%})")
        else:
            emit(f"🐴 [STABLE] {len(_fixed_opponents)} cross-run opponent(s): {_stable_labels} — "
                 "EVAL-ONLY (no --self-play, so they don't join the training mix)")
        for _line_s in _stable_prov:
            emit(_line_s)

    # EXPLOITER mode (--exploiter): resolve the single fixed target the SAME way as a stable opponent
    # (run-dir/checkpoint spec → arch-gated FixedOpponentEntry), validating its weights load here so a
    # corrupt zip FATALs once up front instead of crashing every env worker. The env factory builds one
    # RLPlayer from it per worker; the wrapper then uses it as the sole training opponent. (Mutual
    # exclusivity with --self-play is enforced at arg-parse time above.)
    _exploiter_entry = None
    if args.exploiter:
        from agents.training.fixed_opponent_pool import resolve_stable_opponents
        from agents.model.snapshot import (
            current_model_version as _current_model_version, load_foreign_opponent)
        _cv_expl = _current_model_version(mappings, **_run_arch_toggles(args))
        try:
            _resolved = resolve_stable_opponents(args.exploiter, _cv_expl,
                                                 default_temperature=args.stable_opponent_temp)
            if len(_resolved) != 1:
                raise ValueError(f"--exploiter takes exactly ONE target model, got {len(_resolved)}")
            _exploiter_entry = _resolved[0]
            load_foreign_opponent(_exploiter_entry.zip_path, current_version=_cv_expl, device="cpu",
                                  config_path=_exploiter_entry.config_path)  # validate weights load
        except (ModelVersionError, FileNotFoundError, ValueError) as e:
            print(f"\n[Exploiter] FATAL: {e}")
            sys.stdout.flush()
            os._exit(int(TrainExitCode.FATAL_CONFIG))
        except Exception as e:  # noqa: BLE001 — corrupt/unreadable foreign weights zip
            print(f"\n[Exploiter] FATAL: failed to load exploiter target weights: {e}")
            sys.stdout.flush()
            os._exit(int(TrainExitCode.FATAL_CONFIG))
        _temp_desc = f"temp {args.stable_opponent_temp:g}"
        if args.exploiter_keep_bots:
            emit(f"🥊 [EXPLOITER] training vs {_exploiter_entry.label} ({_temp_desc}) "
                 f"with the heuristic bots MIXED IN: per episode P(target)={1 - args.exploiter_bot_fraction:.0%}, "
                 f"P(bot)={args.exploiter_bot_fraction:.0%}. Goal: learn to beat the target while keeping a bot floor.")
        else:
            emit(f"🥊 [EXPLOITER] training vs {_exploiter_entry.label} as the SOLE opponent every episode "
                 f"({_temp_desc}; no self-play/pool/bots). Goal: learn to beat it.")
        emit(f"   target: {_exploiter_entry.provenance()}")
        if _exploiter_entry.team_str:
            emit(f"   target pilots ITS OWN pinned team ({os.path.basename(_exploiter_entry.team_file)}) "
                 "— the fold-back contract")

    # Opponent-parity Proposal A: the exploiter target AUTO-registers as an eval opponent, so the
    # verdict metric (eval/win_rate_vs_ext_<target>) exists without remembering to duplicate the
    # target in --stable-opponents. Dedup-guarded — the historical both-flags recipe is unchanged.
    # Training-mix side is untouched (exploiter mode excludes --self-play → the entry is eval-only).
    if _exploiter_entry is not None:
        from agents.training.fixed_opponent_pool import register_exploiter_for_eval
        _fixed_opponents, _expl_registered = register_exploiter_for_eval(
            _fixed_opponents, _exploiter_entry)
        if _expl_registered:
            emit(f"🥊 [EXPLOITER] target auto-registered for eval as {_exploiter_entry.label} "
                 f"(greedy verdict metric eval/win_rate_vs_{_exploiter_entry.label})")

    # Curriculum (transition + floor) effective values: CLI override or the module defaults.
    _heuristic_floor = args.heuristic_floor if args.heuristic_floor is not None else HEURISTIC_FLOOR
    _sp_start_wr = args.self_play_start_wr if args.self_play_start_wr is not None else SELF_PLAY_START
    _sp_full_wr = args.self_play_full_wr if args.self_play_full_wr is not None else SELF_PLAY_FULL
    if (_heuristic_floor, _sp_start_wr, _sp_full_wr) != (HEURISTIC_FLOOR, SELF_PLAY_START, SELF_PLAY_FULL):
        print(f"[Opponents] self-play curriculum: start_wr={_sp_start_wr:g} full_wr={_sp_full_wr:g} "
              f"heuristic_floor={_heuristic_floor:g} "
              f"(defaults {SELF_PLAY_START:g}/{SELF_PLAY_FULL:g}/{HEURISTIC_FLOOR:g})")

    # Promotion gate: ALREADY RESOLVED. `resolve_config` (gen3_eval_sentinel_greedy_default_v1)
    # owns the three-branch rule — explicit argv > re-derived from a typed regime > inherited from
    # the checkpoint > the regime default — and prints the one [EVAL REGIME] line naming the
    # resolved value and its source. Re-deriving it here would be a second place for the numbers to
    # disagree, and the copy that lost would be the one `metadata.json:cli_args` records.
    _promote_threshold = float(args.promote_threshold)

    return MatchupSetup(
        matchup=matchup, mappings=mappings,
        trainee_teambuilder=trainee_teambuilder, opponent_teambuilder=opponent_teambuilder,
        specialist_team_str=_specialist_team_str, opponent_classes=OPPONENT_CLASSES,
        bot_weight_vec=_bot_weight_vec, fixed_opponents=_fixed_opponents,
        exploiter_entry=_exploiter_entry, heuristic_floor=_heuristic_floor,
        sp_start_wr=_sp_start_wr, sp_full_wr=_sp_full_wr,
        promote_threshold=_promote_threshold)
