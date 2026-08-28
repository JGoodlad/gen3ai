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
from main.exit_codes import TrainExitCode
from main.launcher.ipc import emit
from main.train.run_io import _run_arch_toggles
from poke_env.player import SimpleHeuristicsPlayer
from utils.teambuilder import Gen3Teambuilder
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


def apply_distill_team_bias(args, all_teams, trainee_teambuilder):
    """gen3_exploiter_distill_v1: point `--distill-team-bias` of the trainee's episodes at the
    TEACHER TEAMS (rest = pool rehearsal), and precompute those teams' species id-sets for the
    env's per-state `distill_mask`. Returns the trainee teambuilder to use (the argument itself
    when there is no teacher), and sets `args._distill_species`.

    🚨 **THE BIAS IS KEYED ON THE TEACHERS, THE MASK ON THE COEFFICIENT** — and the split is the
    whole point of `gen3_distill_bias_at_coef0_v1`. A `--distill-coef 0` CONTROL arm exists to hold
    the team distribution constant against its treatment arm while folding no loss, so the bias
    must apply at coef 0; `ai_v9_58_R2CTRL_0827` asked for exactly that, got an effective bias of
    0.0 (the pairs were parsed only above coef 0), and its argv and metadata both said 0.4.
    `_distill_species` stays coefficient-gated in the other direction: it is what makes the env emit
    the training-only `distill_mask` obs key, so populating it at coef 0 would change the
    OBSERVATION SPACE of a run that has no distill term to read it — a difference between the arms
    where the design asks for none, and a resume-breaking change for a live control run.
    """
    args._distill_species = None
    _pairs = getattr(args, "_distill_pairs", None)
    if not _pairs:
        return trainee_teambuilder
    from poke_env.teambuilder.teambuilder import Teambuilder as _TB
    from poke_env.data.normalize import to_id_str as _to_id
    _loss_on = bool(args.distill_coef and args.distill_coef > 0)
    _species_sets, _team_strs = [], []
    for _tp, _tfs in _pairs:
        _sets = []
        for _tf in _tfs:
            with open(_tf, encoding="utf-8") as _df:
                _s = _df.read()
            _team_strs.append(_s)
            if _loss_on:
                # poke-env parks the species in `nickname` when the export has no nickname → fall back to it.
                _sets.append(frozenset(_to_id(m.species or m.nickname) for m in _TB.parse_showdown_team(_s)))
        _species_sets.append(_sets)
    if _loss_on:
        # list (per TEACHER, teacher-id = index+1) of LISTS of species-frozensets (that teacher's teams) —
        # a multi-team teacher's KL fires on ANY of its teams (the env matches `cur in sp_list`).
        args._distill_species = _species_sets
    # Bias the trainee across ALL N teacher teams (bias_prob total, split evenly); rest = pool rehearsal.
    trainee_teambuilder = Gen3Teambuilder(all_teams, bias_teams=_team_strs,
                                          bias_prob=args.distill_team_bias,
                                          team_pfsp=args.team_pfsp,
                                          team_pfsp_cap=args.team_pfsp_cap,
                                          team_pfsp_floor=args.team_pfsp_floor)
    emit(f"🧪 [DISTILL] {len(_pairs)} teacher(s) / {len(_team_strs)} team(s), "
         f"coef={args.distill_coef}"
         f"{'' if _loss_on else ' (LOSS OFF — team bias only, no teacher loaded, no distill_mask)'}"
         f" | trainee biased {args.distill_team_bias:.0%} across all "
         f"{len(_team_strs)} teacher team(s); rest = pool rehearsal")
    for _i, (_tp, _tfs) in enumerate(_pairs, start=1):
        emit(f"   [{_i}] {_tp} ← {len(_tfs)} team(s): "
             f"{', '.join(os.path.basename(_f) for _f in _tfs)}")
    return trainee_teambuilder


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
    if getattr(args, "allow_nonsample_trainee", False):
        # RESEARCH override: skip the vetted-sample gate so an exploiter can pilot whole-POOL z-near
        # teams (anchor on a sample, nearest neighbors from all 719 teams). Use for capacity studies
        # (count-vs-diversity of the FiLM cluster), NOT for a teacher you intend to distil as-is.
        print("⚠️ [Exploiter] --allow-nonsample-trainee: SKIPPING the vetted-sample gate — trainee may "
              "pilot non-sample pool teams (research/capacity mode).")
    else:
        try:
            from agents.training.matchup_spec import validate_exploiter_trainee_is_sample
            validate_exploiter_trainee_is_sample(matchup, sample_teams)
        except ValueError as _e:
            print(f"\n[Exploiter] FATAL: {_e}")
            sys.stdout.flush()
            os._exit(int(TrainExitCode.FATAL_CONFIG))
    # → eval callbacks (trainee_team_str). Read from EVAL_trainee_teams (not trainee_teams) so the
    # distillation path evals on the TAUGHT teams; a `pin_multi` source yields a LIST (eval samples
    # among them, exactly as training does), a single pin yields the raw export, else None = pool.
    _ets = matchup.eval_trainee_teams
    _specialist_team_str = (list(_ets.pin_strs) if _ets.kind == "pin_multi" and _ets.pin_strs
                            else _ets.pin_str)
    # Team-side PFSP threads ONLY into the TRAINEE builder (opponent teams aren't win-rate-sampled);
    # "off" (default) is byte-identical construction.
    trainee_teambuilder = matchup.trainee_teams.build(
        all_teams, sample_teams,
        team_pfsp=args.team_pfsp, team_pfsp_cap=args.team_pfsp_cap,
        team_pfsp_floor=args.team_pfsp_floor)
    # Team-blocked episodes: hold each drawn trainee team for N consecutive episodes — the
    # per-team gradient-density counter to the measured FiLM sample starvation. Trainee side ONLY
    # (opponent draws stay per-episode); 1 = off, byte-identical. Training-only, not version-locked.
    if args.team_block_episodes > 1:
        trainee_teambuilder.set_block_episodes(args.team_block_episodes)
    opponent_teambuilder = matchup.opponent_teams.build(all_teams, sample_teams)
    trainee_teambuilder = apply_distill_team_bias(args, all_teams, trainee_teambuilder)
    for _ln in matchup.summary_lines():
        emit(_ln)
    if matchup.trainee_teams.kind == "pin_multi":
        _tt = matchup.trainee_teams
        emit(f"🎯 [MULTI-SPECIALIST] trainee pinned to {len(_tt.pin_strs)} teams (sampled uniformly): "
             f"{', '.join(os.path.basename(f) for f in _tt.pin_files)} (opponents keep the full pool)")
    elif matchup.trainee_teams.pin_str:
        # Read the TRAINING pin, not `_specialist_team_str` — that one is EVAL-derived and may be a
        # LIST (a distillation run trains on the full pool but evals on the TAUGHT teams, so
        # `eval_trainee_teams.kind == "pin_multi"` while `trainee_teams.kind` is not). Calling
        # `.splitlines()` on it crashed every --distill-coef launch at startup.
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
            emit(f"⚠️ [MATCHUP DRIFT] this resume declares matchup {matchup.spec_hash()} but the "
                 f"run last recorded {_rec_hash} — the TRAINING DISTRIBUTION IS CHANGING mid-run. "
                 "Metrics across the change are NOT comparable (a new era lands in "
                 "metadata.json:matchup_history).")
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
        _overrides = {}
        for tok in args.bot_weights.split(","):
            if not tok.strip():
                continue
            name, sep, val = tok.partition("=")
            if not sep:
                print(f"[Opponents] ERROR: --bot-weights token '{tok}' is not name=weight")
                sys.exit(1)
            _overrides[name.strip()] = float(val)
        _valid = {opponent_name(c) for c in OPPONENT_CLASSES}
        _bad = set(_overrides) - _valid
        if _bad:
            print(f"[Opponents] ERROR: unknown --bot-weights names {sorted(_bad)} "
                  f"(valid: {sorted(_valid)})")
            sys.exit(1)
        _bot_weight_vec = [_overrides.get(opponent_name(c), 1.0) for c in OPPONENT_CLASSES]
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
        if args.self_play:
            emit(f"🐴 [STABLE] {len(_fixed_opponents)} cross-run opponent(s): {_stable_labels} — "
                 f"eval greedy; training ≤{args.stable_opponent_selfplay_share:.0%} of self-play until "
                 f"mastered (win_rate ≥ {args.stable_opponent_mastered_wr:.0%})")
        else:
            emit(f"🐴 [STABLE] {len(_fixed_opponents)} cross-run opponent(s): {_stable_labels} — "
                 "EVAL-ONLY (no --self-play, so they don't join the training mix)")

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
        # gen3_exploiter_temp_anneal_v1: when annealing the target's temperature, START it at
        # --exploiter-temp-start (so the very first episodes are already at the curriculum's hot temp,
        # before ExploiterTempAnnealCallback's first per-rollout push); else the fixed
        # --stable-opponent-temp (unchanged default).
        _expl_temp0 = (args.exploiter_temp_start if args.exploiter_temp_start is not None
                       else args.stable_opponent_temp)
        try:
            _resolved = resolve_stable_opponents(args.exploiter, _cv_expl,
                                                 default_temperature=_expl_temp0)
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
        if args.exploiter_temp_start is None:
            _temp_desc = f"temp {args.stable_opponent_temp:g}"
        elif args.exploiter_temp_mode == "ratchet":
            _temp_desc = (f"temp {args.exploiter_temp_start:g}→{args.exploiter_temp_end:g} WR-RATCHETED "
                          f"(harder when train-WR ≥ {args.exploiter_temp_ratchet_wr:.0%})")
        else:
            _temp_desc = (f"temp {args.exploiter_temp_start:g}→{args.exploiter_temp_end:g} annealed over "
                          f"{args.exploiter_temp_anneal_frac:.0%} of training")
        if args.exploiter_keep_bots:
            emit(f"🥊 [EXPLOITER] training vs {_exploiter_entry.label} ({_temp_desc}) "
                 f"with the heuristic bots MIXED IN: per episode P(target)={1 - args.exploiter_bot_fraction:.0%}, "
                 f"P(bot)={args.exploiter_bot_fraction:.0%}. Goal: learn to beat the target while keeping a bot floor.")
        else:
            emit(f"🥊 [EXPLOITER] training vs {_exploiter_entry.label} as the SOLE opponent every episode "
                 f"({_temp_desc}; no self-play/pool/bots). Goal: learn to beat it.")
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

    # Promotion gate: regime-aware default — 0.55 under greedy sentinels (the temperature handicap
    # is gone, so a genuinely-ahead trainee wins the pool by a smaller margin and 0.65 would freeze
    # the pool), else the original 0.65. An explicit --promote-threshold always wins.
    _promote_threshold = (args.promote_threshold if args.promote_threshold is not None
                          else (0.55 if args.eval_sentinel_greedy else 0.65))
    if args.eval_sentinel_greedy:
        print(f"[Opponents] eval sentinels GREEDY (best-vs-best pool/ELO signal) — "
              f"promote_threshold={_promote_threshold:g}")

    return MatchupSetup(
        matchup=matchup, mappings=mappings,
        trainee_teambuilder=trainee_teambuilder, opponent_teambuilder=opponent_teambuilder,
        specialist_team_str=_specialist_team_str, opponent_classes=OPPONENT_CLASSES,
        bot_weight_vec=_bot_weight_vec, fixed_opponents=_fixed_opponents,
        exploiter_entry=_exploiter_entry, heuristic_floor=_heuristic_floor,
        sp_start_wr=_sp_start_wr, sp_full_wr=_sp_full_wr,
        promote_threshold=_promote_threshold)
