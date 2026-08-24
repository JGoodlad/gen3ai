"""THE training entry point — a thin orchestrator over the `main/train/` phase modules.

    python src/main/train_rl_agent.py --debug --steps 10000
    python -m main.train_rl_agent  …          (the launcher spawns the FILE path)

**This file keeps its path and its whole public surface.** `build_parser()` (which
`main.checkargs` inspects), `main()`, and every helper that used to live here are re-exported
below, so `from main.train_rl_agent import <anything>` resolves exactly as it did before the
2026-08-22 decomposition. The precedent is `features_extractor.py`: one file per concern, the
original kept as a hub.

THE MODULE MAP (`main/train/`, and `main/train/__init__.py` repeats it):

    constants.py        BATTLE_FORMAT / the smoke-eval scale / the abort drain bound
    parser.py           `build_parser()` + `BoolFlag` / `str2bool` / `optional_float`
    compile_flags.py    the `--compile-opponents` / `--compile-trainer` default resolvers
    checkpoint_state.py reading a checkpoint's saved arch; the by-NAME optimizer realign
    run_io.py           the run directory, latest.txt, the TB logger, the checkpoint callback
    lifecycle.py        grad checkpointing, the trainer compile, the round-trip smoke, signals
    config.py           phase 1 — desugar / `_resolve` / validate     (mutates `args` in place)
    matchup_setup.py    phase 2 — teams, the matchup, every opponent source
    env_factory.py      phase 3 — the per-worker training-env `_init` closure
    callbacks.py        phase 4 — everything that runs during `learn()`
    model_build.py      phase 5 — the resume + fresh model paths, and `learn()` itself
    final_eval.py       the post-training win-rate evaluation

What is left HERE is the glue those phases hand things to each other through: the run directory,
the reward config, the vec-env, the self-play pool, and the order the five phases run in.
"""
import multiprocessing
import os as _os
import traceback
import functools

# ── BLAS THREAD PINNING — must run BEFORE torch is imported anywhere ──────────────────────────
# Each SubprocVecEnv worker runs a full CPU opponent forward; with the library default (one thread
# per core) N workers spawn N×cores competing threads. Measured on a 16-core box, 8 neural-opponent
# envs run DIRECTLY (no launcher): load average 110 and **6 fps**, vs 231 fps with these pinned — a
# ~38× cliff that dwarfs every other measured throughput lever.
#
# `launcher/child.py` already exports these for production, so runs under the launcher were never
# affected — but `python src/main/train_rl_agent.py …` is a DOCUMENTED entry point (root CLAUDE.md
# "Training — run directly") and had no such protection. Setting them here covers both paths; workers
# inherit them through `spawn`. `setdefault` so an explicit override still wins, and the env-worker
# `_init` pins `torch.set_num_threads(1)` independently in case one does.
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
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

from agents.model.extractor_arch import build_extractor_arch_kwargs
from agents.model.compile_opponents import arm_compile_quorum
from agents.model.compile_prewarm import prewarm_extractor_compile
from agents.training.snapshot_pool import SnapshotPool, heuristic_fraction
from agents.training.reward_manager import Gen3RewardManager
from agents.training.stall import StallConfig
from agents.training.async_vec_env import AsyncSubprocVecEnv
from main.launcher.ipc import emit

# ── THE PHASES ────────────────────────────────────────────────────────────────────────────────
# One module per concern (see `main/train/__init__.py` for the map). Imported here rather than
# used from their packages so that every name this file ever exported still resolves from it —
# `from main.train_rl_agent import build_parser` / `_write_latest_txt` / `_TrackingCheckpointCallback`
# and the rest are all live re-exports, the same contract `features_extractor.py` keeps for its
# own phase split.
from main.train.constants import (   # noqa: F401 — re-export hub
    BATTLE_FORMAT, CLIP_RANGE_DEFAULT, DEFAULT_EVAL_BATTLES, SMOKE_EVAL_BATTLES, SMOKE_STEPS,
    _ABORT_EVAL_DRAIN_SEC,
)
from main.train.parser import (   # noqa: F401 — re-export hub
    BoolFlag, build_parser, optional_float, str2bool, _BOOL_FALSE, _BOOL_TRUE,
)
from main.train.compile_flags import (   # noqa: F401 — re-export hub
    resolve_compile_opponents_preload, resolve_compile_trainer_auto,
    resolve_compile_trainer_default, _PRELOAD_WITHOUT_OPPONENTS,
)
from main.train.checkpoint_state import (   # noqa: F401 — re-export hub
    _load_saved_version, _read_saved_optimizer_state, _remap_optimizer_state_by_name,
    _shape_only_reset_optimizer_state, _validate_or_reset_optimizer_state,
)
from main.train.run_io import (   # noqa: F401 — re-export hub
    _HparamLogCallback, _TrackingCheckpointCallback, _attach_run_tb_logger, _model_hparams,
    _resolve_fresh_model_dir, _run_arch_toggles, _write_latest_txt,
)
from main.train.lifecycle import (   # noqa: F401 — re-export hub
    _apply_grad_checkpointing, _maybe_compile_trainer, _run_roundtrip_test,
    _setup_signal_handlers,
)
from main.train.config import resolve_config
from main.train.matchup_setup import build_matchup_and_opponents
from main.train.env_factory import create_training_env_random
from main.train.callbacks import build_callbacks
from main.train.model_build import attach_cf_labels, build_and_train
from main.train.final_eval import evaluate_model_random


async def main():
    # --- Pre-flight Checks ---
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
        os._exit(1) # Force immediate termination of all threads

    sys.excepthook = global_exception_handler
    
    def asyncio_exception_handler(loop, context):
        msg = context.get("exception", context["message"])
        print(f"\n🛑 Asyncio Error: {msg}")
        os._exit(1)
        
    loop = asyncio.get_event_loop()
    loop.set_exception_handler(asyncio_exception_handler)

    parser = build_parser()

    args = parser.parse_args()
    if getattr(args, "trainee_teams", None) and getattr(args, "trainee_team", None):
        parser.error("--trainee-teams (multi-team pin) is mutually exclusive with --trainee-team "
                     "(single-team pin) — use one or the other.")

    _cfg = resolve_config(args, parser)
    server_config, annealing_mode, log_level = (
        _cfg.server_config, _cfg.annealing_mode, _cfg.log_level)

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
    if args.run_dir:
        model_dir = args.run_dir                                     # launcher-managed resume
    else:
        # A memorable --run-name (models/<name>), an exploiter default (models/exploiter_vs_<target>),
        # or the legacy date-stamp — with a guard against clobbering an existing run.
        model_dir = _resolve_fresh_model_dir(
            args.run_name,
            _exploiter_entry.label if _exploiter_entry is not None else None,
            args.model)

    os.makedirs(model_dir, exist_ok=True)
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
        
    # gen3_cf_label_plumbing_v1 — the two counterfactual-factory directories under the run root.
    # `cf_records/` is WRITTEN by the env workers (opt-in); `cf_labels/` is READ from whatever an
    # out-of-process producer left there. Both are None/unused unless the flags ask for them, so a
    # default run creates neither and is FILE-identical to today.
    _cf_records_dir = os.path.join(model_dir, "cf_records") if args.cf_records else None
    # EITHER consumer wants the buffer: the evidential term reads the same label rows, so gating the
    # directory on the scalar coefficient alone would silently starve an evidential-only run.
    _cf_labels_dir = (os.path.join(model_dir, "cf_labels")
                      if (args.cf_winprob_coef > 0 or args.cf_evidential_coef > 0
                          or args.cf_twin_coef > 0 or args.cf_shadow_coef > 0) else None)
    if _cf_records_dir:
        os.makedirs(_cf_records_dir, exist_ok=True)
        emit(f"🧾 [CF] reconstruction-record tap ON → {_cf_records_dir} "
             f"(newest {args.cf_records_keep})")

    stall_cfg = StallConfig(output_dir=os.path.join(model_dir, "stalls"))
    # Per-run reward config (design §1). gamma MUST == the PPO gamma (asserted post-build below); the
    # factory passes it to every env's reward manager. Default = the single-variable run.
    from agents.training.reward_manager import (
        RewardConfig, format_reward_composition, reward_class_composition)
    # Single construction site (gamma == InstrumentedMaskablePPO(gamma=0.9999), asserted below). Every
    # reward CLI flag flows in by name → training, eval, and the version record all use ONE config.
    reward_config = RewardConfig.from_args(args)
    reward_factory = functools.partial(Gen3RewardManager, config=reward_config)
    # STATE the reward composition rather than implying it. The v8->v9 drift was invisible because a
    # launch never said what its reward was made of; this line, and the `reward_composition` block it
    # records into metadata.json, are what a launch-diff gate compares.
    reward_composition = reward_class_composition(reward_config)
    # `emit` prints when there is no launcher pipe, so this reaches BOTH a bare run's stdout and the
    # launcher Events panel — the composition must never be visible in only one of them.
    emit(format_reward_composition(reward_config))
    # Bound to the run's reward config here rather than in `model_build`, because the SHADOW
    # critic's `mc_return` labels are only this run's labels if the producer used this run's
    # reward — the digest is what the label buffer checks them against.
    _attach_cf_labels = functools.partial(
        attach_cf_labels, args=args, _cf_labels_dir=_cf_labels_dir, reward_config=reward_config)

    # Running parallel environments
    n_envs = 1 if args.debug else args.n_envs
    # --async-rollout swaps the barriered SubprocVecEnv for AsyncSubprocVecEnv (per-env in-flight
    # stepping + drain-safe env_method). Only when not --debug (DummyVecEnv has one env, no barrier).
    _async_rollout = args.async_rollout and not args.debug
    if args.debug:
        EnvClass = DummyVecEnv
    elif _async_rollout:
        EnvClass = AsyncSubprocVecEnv
    else:
        EnvClass = SubprocVecEnv

    emit(f"⚙️ Initializing {n_envs} envs ({EnvClass.__name__})"
         + (" — non-barrier async rollout" if _async_rollout else ""))

    # --compile-opponents: warm the SHARED on-disk Inductor cache in THIS process before any env
    # worker exists, so the workers all hit it warm instead of racing on a cold one (measured
    # 59.6 s -> 30.1 s wall for 16 workers). Uses the same arch table as the model build below, so
    # the cached codegen is for the graph the workers will actually run.
    #
    # gen3_forkserver_preload_v1: `--compile-opponents-preload` goes further — the graph is
    # traced ONCE in the forkserver and every worker inherits it by fork (~0.12 s/worker). This
    # was IMPOSSIBLE until the lazy poke_env __init__ (2026-08-16): forking is only safe from a
    # single-threaded process, and the extractor import used to start poke-env's global asyncio
    # loop thread — the 2026-08 attempt forked 2 of 48 workers and hung forever. The preload now
    # proves single-threadedness after its compile and RAISES otherwise (loud env-construction
    # failure, never a silent wedge); `compile_prewarm_test.py` pins the import invariant.
    if args.compile_opponents and not args.debug:
        # Publish ONE tally directory for this process tree before any worker exists. Every env
        # worker / eval worker inherits it through the environment and reports its keep-or-revert
        # verdict there, which is what lets --compile-opponents-strict be fatal on a SYSTEMIC
        # failure instead of on one worker's timing draw (three launches died that way on
        # 2026-08-24). Cleared per process, so a restart counts fresh.
        arm_compile_quorum(model_dir)
        if args.compile_opponents_preload:
            import multiprocessing as _mp
            from agents.model.extractor_arch import arch_kwargs_to_plain
            os.environ["GEN3AI_PRELOAD_ARCH"] = json.dumps(
                arch_kwargs_to_plain(build_extractor_arch_kwargs(args)))
            _mp.get_context("forkserver").set_forkserver_preload(
                ["agents.model.compile_preload"])
            emit("⚙️ forkserver preload armed — workers inherit ONE traced graph "
                 "(gen3_forkserver_preload_v1)")
        else:
            prewarm_extractor_compile(build_extractor_arch_kwargs(args), mappings)

    _shutdown_event = threading.Event()

    # --- Self-Play Pool Setup ---
    # The pool is a directory the env workers read off disk (each builds its own SnapshotPool).
    # The heuristic-vs-pool split is NOT fixed per process anymore: every env picks its opponent
    # per-episode from a LIVE self_play_fraction that the eval callback updates each eval (see
    # MaskableAgentWrapper). The initial fraction comes from the persisted win rate (summary.json)
    # so a resumed run starts at the right ramp level instead of cold-starting at 0%.
    _pool: SnapshotPool | None = None
    _opp_version = None  # ModelVersion threaded into opponent snapshot loads (set when self-play on)
    _snapshot_dir = None
    _initial_self_play_fraction = 0.0
    if args.self_play:
        from pathlib import Path as _Path
        from agents.model.snapshot import current_model_version as _current_model_version

        _snapshot_dir = _Path(args.snapshot_dir) if args.snapshot_dir else _Path(model_dir) / "snapshots"
        _cv = _current_model_version(mappings, **_run_arch_toggles(args))
        _opp_version = _cv
        _pool = SnapshotPool(pool_dir=_snapshot_dir, current_version=_cv, device=args.device,
                             pfsp_scale=args.pfsp_scale, pool_spread=args.pool_spread)
        _persisted_wr = _pool.load_persisted_win_rate()
        _initial_self_play_fraction = 1.0 - heuristic_fraction(
            _persisted_wr, floor=_heuristic_floor, start=_sp_start_wr, full=_sp_full_wr)
        emit(
            f"🎮 [SELFPLAY] Pool has {len(_pool)} snapshots, win_rate_vs_bots={_persisted_wr:.2%} "
            f"→ self_play_fraction={_initial_self_play_fraction:.0%} (live, per-episode)"
        )

    opponent_device = "cpu" if args.self_play_use_cpu else args.device
    if args.self_play:
        emit(
            f"🧠 [SELFPLAY] Opponent snapshots load on '{opponent_device}' "
            f"({'CPU — avoids per-worker CUDA contexts' if args.self_play_use_cpu else 'training device'})"
        )

    # Exploiter mode is NOT self-play, so the self-play block above left _opp_version=None — but the
    # env factory still needs it to arch-gate the exploiter target's foreign load. Set it here.
    if _exploiter_entry is not None and _opp_version is None:
        from agents.model.snapshot import current_model_version as _current_model_version
        _opp_version = _current_model_version(mappings, **_run_arch_toggles(args))

    def _make_factories():
        return [
            create_training_env_random(
                i, stall_config=stall_cfg, opponent_device=opponent_device,
                opponent_version=_opp_version,
                snapshot_dir=str(_snapshot_dir) if _snapshot_dir is not None else None,
                self_play_fraction=_initial_self_play_fraction, self_play=args.self_play,
                heuristic_weights=_bot_weight_vec,
                stable_opponents=_fixed_opponents,
                exploiter_entry=_exploiter_entry,
                cf_records_dir=_cf_records_dir,
                # What used to be closure state when the factory lived inside `main()`.
                args=args, mappings=mappings, log_level=log_level,
                trainee_teambuilder=trainee_teambuilder,
                opponent_teambuilder=opponent_teambuilder, server_config=server_config,
                OPPONENT_CLASSES=OPPONENT_CLASSES, reward_factory=reward_factory,
            )
            for i in range(n_envs)
        ]

    env_factories = _make_factories()
    env = EnvClass(env_factories)
    # NOTE: the subprocess watchdog is started LATER, just before model.learn() — nothing steps
    # the env before then (model construction/load only reads its spaces).

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

    _evaluate_model_random = functools.partial(
        evaluate_model_random, args=args, mappings=mappings,
        trainee_teambuilder=trainee_teambuilder, opponent_teambuilder=opponent_teambuilder,
        server_config=server_config)

    # --- Phase 4: everything that runs during learn() ---
    _cb = build_callbacks(
        args=args, model_dir=model_dir, server_config=server_config,
        annealing_mode=annealing_mode, _pool=_pool, _fixed_opponents=_fixed_opponents,
        _bot_weight_vec=_bot_weight_vec, OPPONENT_CLASSES=OPPONENT_CLASSES,
        _specialist_team_str=_specialist_team_str, _promote_threshold=_promote_threshold,
        _heuristic_floor=_heuristic_floor, _sp_start_wr=_sp_start_wr, _sp_full_wr=_sp_full_wr)

    # --- Phase 5: the model, and the training job itself ---
    await build_and_train(
        args=args, env=env, mappings=mappings, model_dir=model_dir, cli_args=cli_args,
        log_level=log_level, n_envs=n_envs, reward_config=reward_config,
        reward_composition=reward_composition, annealing_mode=annealing_mode,
        _async_rollout=_async_rollout, _shutdown_event=_shutdown_event,
        _run_eval=_cb.run_eval, _effective_max_lr=_cb.effective_max_lr,
        callbacks=_cb.callbacks, eval_callback=_cb.eval_callback, lr_callback=_cb.lr_callback,
        adaptive_ppo_callback=_cb.adaptive_ppo_callback,
        graceful_restart_callback=_cb.graceful_restart_callback,
        _attach_cf_labels=_attach_cf_labels, _maybe_seed_pool=_maybe_seed_pool,
        evaluate_model_random=_evaluate_model_random)


if __name__ == "__main__":
    asyncio.run(main())
