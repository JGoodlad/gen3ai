"""THE training entry point — a thin orchestrator over the `main/train/` phase modules.

    python src/main/train_rl_agent.py --debug --steps 10000
    python -m main.train_rl_agent  …          (the launcher spawns the FILE path)

**This file keeps its path and its whole public surface.** `build_parser()` (which
`main.checkargs` inspects), `main()`, and every helper that used to live here are re-exported
below, so `from main.train_rl_agent import <anything>` resolves exactly as it did before the
2026-08-22 decomposition. The precedent is `features_extractor.py`: one file per concern, the
original kept as a hub.

THE MODULE MAP (`main/train/`, and `main/train/__init__.py` repeats it):

    constants.py        BATTLE_FORMAT / the abort drain bound
    parser/             `build_parser()` behind a hub, one module per FLAG FAMILY in `--help`
                        order; `base.py` holds `BoolFlag` / `str2bool` / `optional_float`
    compile_flags.py    the `--compile-trainer` default resolvers
    checkpoint_state.py reading a checkpoint's saved arch; the by-NAME optimizer realign
    run_io.py           the run directory, latest.txt, the TB logger, the checkpoint callback
    lifecycle.py        grad checkpointing, the trainer compile, the round-trip smoke, signals
    config.py           phase 1 — desugar / `_resolve` / validate     (mutates `args` in place)
    matchup_setup.py    phase 2 — teams, the matchup, every opponent source
    rust_env_setup.py   phase 3 — the Rust env core's `RustVecEnv` (the only env core)
    callbacks.py        phase 4 — everything that runs during `learn()`
    model_build.py      phase 5 — the resume + fresh model paths, and `learn()` itself

What is left HERE is the glue those phases hand things to each other through: the run directory,
the reward config, the vec-env, the self-play pool, and the order the five phases run in.
"""
import multiprocessing
import os as _os
import traceback

# ── BLAS THREAD PINNING — must run BEFORE torch is imported anywhere ──────────────────────────
# Every CPU helper process (the Rust env core's front end, an eval worker) would otherwise run with
# the library default of one BLAS thread per core, so N helpers spawn N×cores competing threads.
# Measured on a 16-core box, 8 neural-opponent envs run DIRECTLY (no launcher): load average 110 and
# **6 fps**, vs 231 fps with these pinned — a ~38× cliff that dwarfs every other measured throughput
# lever.
#
# `launcher/child.py` already exports these for production, so runs under the launcher were never
# affected — but `python src/main/train_rl_agent.py …` is a DOCUMENTED entry point (root CLAUDE.md
# "Training — run directly") and had no such protection. Setting them here covers both paths; child
# processes inherit them through `spawn`. `setdefault` so an explicit override still wins.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    _os.environ.setdefault(_v, "1")

try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass

# Hardened path injection for worker reliability
import sys
import os
script_path = os.path.abspath(__file__)
main_dir = os.path.dirname(script_path)
src_dir = os.path.dirname(main_dir)
root_dir = os.path.dirname(src_dir)
for d in [root_dir, src_dir, main_dir]:
    if d not in sys.path:
        sys.path.insert(0, d)

import asyncio
import json
import threading

from agents.training.snapshot_pool import SnapshotPool, heuristic_fraction
from agents.training.pool_seed import prepare_pool
from main.exit_codes import TrainExitCode, exit_code_for
from main.launcher.ipc import emit

# ── THE PHASES ────────────────────────────────────────────────────────────────────────────────
# One module per concern (see `main/train/__init__.py` for the map). Imported here rather than
# used from their packages so that every name this file ever exported still resolves from it —
# `from main.train_rl_agent import build_parser` / `_write_latest_txt` / `_TrackingCheckpointCallback`
# and the rest are all live re-exports, the same contract `features_extractor.py` keeps for its
# own phase split.
from main.train.constants import (   # noqa: F401 — re-export hub
    BATTLE_FORMAT, CLIP_RANGE_DEFAULT, _ABORT_EVAL_DRAIN_SEC,
)
from main.train.parser import (   # noqa: F401 — re-export hub
    BoolFlag, build_parser, optional_float, str2bool, _BOOL_FALSE, _BOOL_TRUE,
)
from main.train.compile_flags import (   # noqa: F401 — re-export hub
    resolve_compile_trainer_auto, resolve_compile_trainer_default,
)
from main.train.checkpoint_state import (   # noqa: F401 — re-export hub
    _load_saved_version, _read_saved_optimizer_state, _remap_optimizer_state_by_name,
    _shape_only_reset_optimizer_state, _validate_or_reset_optimizer_state,
)
from main.train.run_io import (   # noqa: F401 — re-export hub
    _HparamLogCallback, _TrackingCheckpointCallback, _attach_run_tb_logger, _model_hparams,
    _resolve_fresh_model_dir, _resolve_model_dir, _run_arch_toggles, _write_latest_txt,
)
from main.train.lifecycle import (   # noqa: F401 — re-export hub
    _apply_grad_checkpointing, _declare_compile_cache, _maybe_compile_trainer, _run_roundtrip_test,
    _setup_signal_handlers,
)
from main.train.config import resolve_config
from main.train.matchup_setup import build_matchup_and_opponents
from main.train.callbacks import build_callbacks
from main.train.fork_lr import enforce_inherited_fork_lr
from main.train.model_build import build_and_train


async def main():
    # --- Pre-flight Checks ---
    # THE TORCH FLOOR (deletion pass K1): HEAD has no torch-2.5.1 code path left; a 2.5.1 run resumes
    # pinned to its own commit. Refused FIRST, before anything is parsed or created.
    from utils.torch_floor import refusal as _torch_floor_refusal
    _why = _torch_floor_refusal()
    if _why is not None:
        print(f"\n🛑 [TorchFloor] FATAL: {_why}", file=sys.stderr, flush=True)
        sys.exit(int(TrainExitCode.FATAL_CONFIG))
    try:
        import tensorboard  # noqa: F401 — imported for its SIDE EFFECT of raising ImportError;
        # this is an availability probe, not a use. The name is deliberately never referenced.
    except ImportError:
        print("\n" + "🛑" * 30)
        print("🛑 ERROR: Tensorboard is NOT installed.")
        print("🛑 Training requires tensorboard for professional logging.")
        print("🛑 Please run: pip install tensorboard")
        print("🛑" * 30 + "\n")
        os._exit(1)

    # --- Fail-Fast Handlers ---
    def global_exception_handler(exctype, value, tb):
        print("\n" + "🛑" * 20)
        print("🛑 FATAL ERROR DETECTED - FAILING FAST")
        print("🛑" * 20)
        traceback.print_exception(exctype, value, tb)
        # CRASH (1) — the launcher restarts — unless the error is one a restart would REPLAY
        # (`exit_codes.exit_code_for`: a non-finite learner → FATAL_NONFINITE; the launcher stops).
        os._exit(exit_code_for(value)) # Force immediate termination of all threads

    sys.excepthook = global_exception_handler
    
    def asyncio_exception_handler(loop, context):
        msg = context.get("exception", context["message"])
        print(f"\n🛑 Asyncio Error: {msg}")
        os._exit(exit_code_for(context.get("exception")))
        
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(asyncio_exception_handler)

    parser = build_parser()

    args = parser.parse_args()
    if getattr(args, "trainee_teams", None) and getattr(args, "trainee_team", None):
        parser.error("--trainee-teams (multi-team pin) is mutually exclusive with --trainee-team "
                     "(single-team pin) — use one or the other.")

    _cfg = resolve_config(args, parser)
    annealing_mode, log_level = _cfg.annealing_mode, _cfg.log_level

    # --- Phase 2: teams, the matchup, and every opponent source ---
    _mu = build_matchup_and_opponents(args)
    matchup, mappings = _mu.matchup, _mu.mappings
    trainee_teambuilder, opponent_teambuilder = _mu.trainee_teambuilder, _mu.opponent_teambuilder
    _specialist_team_str, OPPONENT_CLASSES = _mu.specialist_team_str, _mu.opponent_classes
    _bot_weight_vec, _fixed_opponents = _mu.bot_weight_vec, _mu.fixed_opponents
    _exploiter_entry, _promote_threshold = _mu.exploiter_entry, _mu.promote_threshold
    _heuristic_floor, _sp_start_wr, _sp_full_wr = (
        _mu.heuristic_floor, _mu.sp_start_wr, _mu.sp_full_wr)

    # --- Directory Setup ---
    # ONE decision (`run_io._resolve_model_dir`): `--run-dir` (checked — never inside a linked
    # worktree's OWN models/), else `--run-name` / the exploiter default / the legacy date-stamp in
    # `utils.paths.run_archive_dir()` — main's models/ from a worktree, never a cwd-relative one —
    # with a guard against clobbering an existing run.
    model_dir = _resolve_model_dir(
        args.run_dir, args.run_name,
        _exploiter_entry.label if _exploiter_entry is not None else None,
        args.model)

    # FORK-LR INHERITANCE guard (gen3_fork_lr_inherit_guard_v1): a fork of a run whose LR was
    # PINNED and FROZEN inherits the NUMBER but not the FREEZE, so a live KL controller starts
    # annealing away from a rate that was chosen precisely because it should not move. Fires HERE
    # — the first moment `model_dir` is known — and BEFORE the directory is created, so a refusal
    # leaves nothing behind. `--allow-inherited-fork-lr` is the deliberate opt-in.
    enforce_inherited_fork_lr(args, model_dir)
    os.makedirs(model_dir, exist_ok=True)
    # K3: the run's OWN compile cache, declared before anything compiles or spawns (lifecycle.py).
    _declare_compile_cache(args, model_dir)
    # Full CLI namespace (JSON-safe) → persisted into metadata.json for run provenance.
    cli_args = json.loads(json.dumps(vars(args), default=str))
    # Matchup provenance (designs/ai_v8/design_matchup_config.md): the DECLARED matchup + its hash
    # ride into metadata.json beside the flags, so a run's measurement regime is auditable — two
    # eras with different hashes (e.g. the pre-fix OOD-eval era) are not metric-comparable.
    cli_args["_matchup_spec"] = matchup.to_dict()
    cli_args["_matchup_spec_hash"] = matchup.spec_hash()
    if not args.run_dir:
        with open(os.path.join(model_dir, "command.txt"), "w") as f:
            f.write(" ".join(sys.argv))

    # Per-run reward config (design §1). gamma MUST == the PPO gamma (asserted post-build below); the
    # Rust env core's collector builds its reward from the same config. Default = the single-variable run.
    from agents.training.reward_manager import (
        RewardConfig, format_reward_composition, reward_composition_block)
    # Single construction site (gamma == InstrumentedMaskablePPO(gamma=0.9999), asserted below). Every
    # reward CLI flag flows in by name → training, eval, and the version record all use ONE config.
    reward_config = RewardConfig.from_args(args)
    # STATE the reward composition rather than implying it. The v8->v9 drift was invisible because a
    # launch never said what its reward was made of; this line, and the `reward_composition` block it
    # records into metadata.json, are what a launch-diff gate compares.
    # The census PLUS the announced LINE, the class shares and the INERT-flag list — additive over
    # `reward_class_composition`, so every existing reader of this block is untouched. The line is
    # recorded because a launch PRINTED its composition and nothing kept it: a launcher rotates the
    # child log. (Since the shaped-reward deletion every run's composition is 1 TERMINAL.)
    reward_composition = reward_composition_block(reward_config)
    # `emit` prints when there is no launcher pipe, so this reaches BOTH a bare run's stdout and the
    # launcher Events panel — the composition must never be visible in only one of them.
    emit(format_reward_composition(reward_config))
    # gen3_winprob_critic_mode_v1: STATE WHICH READOUT IS THE CRITIC, for the composition line's own
    # reason. The critic changes the quantity the value function predicts, the loss that trains it,
    # what --vf-coef multiplies and what the reward stream has to be — and NOTHING in a metric would
    # say so, because every scalar keeps its name. (a shaped critic was the Python env core's;
    # it was deleted with that core — deletion pass U3 — and a shaped checkpoint is refused, D4.)
    emit(f"🎯 [CRITIC] winprob — V(s) = sigmoid(win-prob logit) in [0,1]; the value loss IS "
         f"that head's BCE against the terminal outcome, weighted by --vf-coef "
         f"{args.vf_coef:g}. Reward = the "
         f"TERMINAL WIN INDICATOR alone; gamma={args.gamma:g}; win_prob_mode="
         f"{args.win_prob_mode!r}. At victory_value 1.0 and gamma 1.0, V(s) == P(win|s) "
         f"exactly. ⚠️ A [0,1] critic cannot express 'a timeout is worse than a loss' — stall "
         f"rate and mean episode length are PRIMARY endpoints on this arm.")
    # The env: the Rust core (`--env-core rust` is the only core) — N envs in ONE core behind the
    # process (or FFI) front end; the trainee and every policy opponent forward through the inference
    # service (T2), the scripted bots play inside the core. `--debug` runs ONE env.
    n_envs = 1 if args.debug else args.n_envs
    emit(f"⚙️ Initializing {n_envs} envs (RustVecEnv — the M5 Rust env core)")

    _shutdown_event = threading.Event()

    # --- Self-Play Pool Setup ---
    # The pool is a directory the Rust env core's inference service loads its opponent slots from.
    # The heuristic-vs-pool split is NOT fixed per process anymore: every env picks its opponent
    # per-episode from a LIVE self_play_fraction that the eval callback updates each eval (see
    # `rust_env_opponents.OpponentPlan`). The initial fraction comes from the persisted win rate (summary.json)
    # so a resumed run starts at the right ramp level instead of cold-starting at 0%.
    _pool: SnapshotPool | None = None
    _opp_version = None  # ModelVersion threaded into opponent snapshot loads (set when self-play on)
    _snapshot_dir = None
    _initial_self_play_fraction = 0.0
    if args.self_play:
        from pathlib import Path as _Path
        from agents.model.snapshot import current_model_version as _current_model_version

        # gen3_fork_pool_seed_v1 — BEFORE the pool is constructed, because the starting
        # self_play_fraction is read off the pool's METADATA at construction. A genuine fork with
        # an empty pool gets its parent's pool (zips + metadata); a poolless fork is REFUSED
        # rather than silently trained against bots. See agents.training.pool_seed.
        prepare_pool(args, model_dir)
        from agents.training.pool_seed import pool_dir_for as _pool_dir_for
        _snapshot_dir = _Path(_pool_dir_for(model_dir))
        _cv = _current_model_version(mappings, **_run_arch_toggles(args))
        _opp_version = _cv
        # owns_dir=True: THIS is the pool that writes the directory (seed / promote), so it is the one
        # whose startup scan may DELETE snapshots outside the declared window (gen3_pool_cap_every_path_v1)
        # — before any env worker scans it. Every other pool over this dir is a reader.
        # device="cpu": the trainer's own pool writes / scans / seeds and never infers on a snapshot
        _pool = SnapshotPool(pool_dir=_snapshot_dir, current_version=_cv, device="cpu",
                             pfsp_scale=args.pfsp_scale, pool_spread=args.pool_spread, owns_dir=True)
        _persisted_wr = _pool.load_persisted_win_rate()
        _initial_self_play_fraction = 1.0 - heuristic_fraction(
            _persisted_wr, floor=_heuristic_floor, start=_sp_start_wr, full=_sp_full_wr)
        emit(
            f"🎮 [SELFPLAY] Pool has {len(_pool)} snapshots, win_rate_vs_bots={_persisted_wr:.2%} "
            f"→ self_play_fraction={_initial_self_play_fraction:.0%} (live, per-episode)"
        )

    # Exploiter mode is NOT self-play, so the self-play block above left _opp_version=None — but the
    # env factory still needs it to arch-gate the exploiter target's foreign load. Set it here.
    if _exploiter_entry is not None and _opp_version is None:
        from agents.model.snapshot import current_model_version as _current_model_version
        _opp_version = _current_model_version(mappings, **_run_arch_toggles(args))

    from main.train.rust_env_setup import build_rust_vec_env
    env = build_rust_vec_env(
        args, mappings=mappings, trainee_teambuilder=trainee_teambuilder,
        opponent_teambuilder=opponent_teambuilder, opponent_classes=OPPONENT_CLASSES,
        bot_weights=_bot_weight_vec, fixed_opponents=_fixed_opponents, exploiter_entry=_exploiter_entry,
        snapshot_dir=str(_snapshot_dir) if _snapshot_dir is not None else None,
        opponent_version=_opp_version, self_play_fraction=_initial_self_play_fraction, n_envs=n_envs,
        eval_trainee_team_str=_specialist_team_str)
    # Nothing steps the env before learn(): model construction / load only reads its spaces.

    def _maybe_seed_pool(model):
        """Seed the pool from the loaded weights iff self-play is active (fraction>0 → win rate
        ≥ SELF_PLAY_START) AND the pool is empty — so the seed is captured from a *competent*
        model, never random/weak. No env rebuild: the worker pools re-scan the dir on demand
        (and lazily whenever they see it empty). The eval callback also seeds when the model
        first crosses the threshold mid-run."""
        if not (args.self_play and _pool is not None):
            return
        if _initial_self_play_fraction > 0 and _pool.is_empty():
            _pool.seed(model)
            emit(f"🌱 [SELFPLAY] Seeded pool from current weights "
                 f"(win rate ≥ threshold → self_play_fraction={_initial_self_play_fraction:.0%})")

    # --- Phase 4: everything that runs during learn() ---
    _cb = build_callbacks(
        args=args, model_dir=model_dir,
        annealing_mode=annealing_mode, _pool=_pool, _fixed_opponents=_fixed_opponents,
        _bot_weight_vec=_bot_weight_vec, OPPONENT_CLASSES=OPPONENT_CLASSES,
        _specialist_team_str=_specialist_team_str, _promote_threshold=_promote_threshold,
        _heuristic_floor=_heuristic_floor, _sp_start_wr=_sp_start_wr, _sp_full_wr=_sp_full_wr)

    # --- Phase 5: the model, and the training job itself ---
    await build_and_train(
        args=args, env=env, mappings=mappings, model_dir=model_dir, cli_args=cli_args,
        log_level=log_level, n_envs=n_envs, reward_config=reward_config,
        reward_composition=reward_composition, annealing_mode=annealing_mode,
        _shutdown_event=_shutdown_event,
        _effective_max_lr=_cb.effective_max_lr,
        callbacks=_cb.callbacks, eval_callback=_cb.eval_callback, lr_callback=_cb.lr_callback,
        adaptive_ppo_callback=_cb.adaptive_ppo_callback,
        graceful_restart_callback=_cb.graceful_restart_callback,
        _maybe_seed_pool=_maybe_seed_pool)


if __name__ == "__main__":
    asyncio.run(main())
