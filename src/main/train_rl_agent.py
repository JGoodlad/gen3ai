import multiprocessing
import traceback
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

# Use the git repo root (cwd-based) so TB logs always land in the main repo,
# even when the launcher pins this script to a tmp worktree.
try:
    import subprocess as _sp
    _repo_root = _sp.check_output(
        ["git", "rev-parse", "--show-toplevel"], text=True, stderr=_sp.DEVNULL
    ).strip()
except Exception:
    _repo_root = root_dir
tensorboard_dir = os.path.join(_repo_root, "tensorboard")

import asyncio
import json
import random
import argparse
import signal
import threading
import torch
from datetime import datetime
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from agents.model.features_extractor import Gen3FeaturesExtractor, NET_ARCH
from agents.model.policy import Gen3DualHeadMaskablePolicy
from agents.model.model_version import ModelVersion, ModelVersionError
from agents.model.snapshot import save_model_snapshot, load_model_snapshot, read_checkpoint_metadata, record_checkpoint
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.inference.player import RLPlayer
from utils.teambuilder import Gen3Teambuilder
from utils.team_loader import TeamLoader
from agents.training.eval_callback import (
    PerOpponentEvalCallback, opponent_name,
)
from agents.training.graceful_restart_callback import GracefulRestartCallback
from agents.training.snapshot_pool import SnapshotPool, heuristic_fraction
from agents.training.selfplay_callback import SelfPlayCallback
from agents.training.wrappers import MaskableAgentWrapper
from agents.training.gen3_env import Gen3Env
from agents.training.reward_manager import Gen3RewardManager
from agents.training.stall import StallConfig
from agents.training.watchdog import start_subprocess_watchdog, start_orphan_watchdog
from agents.training.adaptive_lr_callback import AdaptivePPOCallback, TwoPhaseLRCallback
from agents.training.instrumented_ppo import InstrumentedMaskablePPO
from agents.training.metrics_exporter_callback import MetricsExporterCallback
from utils.logging.levels import LogLevel
from main.exit_codes import TrainExitCode
from main.launcher.ipc import send_event, emit

from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from agents.opponents import (
    Gen3StallerPlayer, Gen3AggressivePlayer, Gen3SetupSweepPlayer,
    Gen3StallerV2Player, Gen3AggressiveV2Player, Gen3SetupSweepV2Player,
    Gen3HeuristicV2Player,
)
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.ps_client.server_configuration import localhost_server_configuration

BATTLE_FORMAT = "gen3ou"
CLIP_RANGE_DEFAULT = 0.15


def _model_hparams(model) -> dict:
    clip_range_vf = float(model.clip_range_vf(1.0)) if model.clip_range_vf is not None else -1.0
    opt = model.policy.optimizer
    return {
        "gamma": model.gamma,
        "gae_lambda": model.gae_lambda,
        "ent_coef": float(model.ent_coef),
        "vf_coef": float(model.vf_coef),
        "batch_size": model.batch_size,
        "n_steps": model.n_steps,
        "clip_range": float(model.clip_range(1.0)),
        "clip_range_vf": clip_range_vf,
        "optimizer": type(opt).__name__,
        "weight_decay": opt.param_groups[0].get("weight_decay", 0.0),
    }


def _write_latest_txt(model_dir: str, basename: str) -> None:
    """Atomically record the most-recent checkpoint in <model_dir>/latest.txt."""
    latest = os.path.join(model_dir, "latest.txt")
    tmp = latest + ".tmp"
    with open(tmp, "w") as f:
        f.write(basename + "\n")
    os.replace(tmp, latest)


class _HparamLogCallback(BaseCallback):
    """Logs static hyperparameters to TensorBoard once at training start."""

    def __init__(self, ent_coef: float):
        super().__init__()
        self._ent_coef = ent_coef

    def _on_training_start(self) -> None:
        self.logger.record("hparams/ent_coef", self._ent_coef)
        self.logger.record("hparams/gamma", self.model.gamma)
        self.logger.record("hparams/gae_lambda", self.model.gae_lambda)
        self.logger.record("hparams/vf_coef", float(self.model.vf_coef))
        self.logger.dump(self.num_timesteps)

    def _on_step(self) -> bool:
        return True


class _TrackingCheckpointCallback(CheckpointCallback):
    """CheckpointCallback that keeps latest.txt up to date and writes per-checkpoint metadata."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_lr_fn = None
        self._current_epochs_fn = None
        # Optional: returns the current TwoPhaseLR handoff_lr (or None).
        self._handoff_lr_fn = None

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            _write_latest_txt(
                self.save_path,
                f"{self.name_prefix}_{self.num_timesteps}_steps.zip",
            )
            if self._current_lr_fn is not None and self._current_epochs_fn is not None:
                ckpt_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.zip")
                handoff_lr = self._handoff_lr_fn() if self._handoff_lr_fn is not None else None
                record_checkpoint(
                    self.save_path,
                    ckpt_path,
                    self._current_lr_fn(),
                    self._current_epochs_fn(),
                    hparams=_model_hparams(self.model),
                    handoff_lr=handoff_lr,
                )
        return result


def _run_roundtrip_test(model, layout: dict, policy_kwargs: dict, debug: bool = False) -> None:
    """Startup smoke test: save → reload → zero forward pass → assert output shape.

    Catches serialization failures at second 5, not hour 50. Raises on any failure.
    """
    import shutil
    import tempfile
    import torch
    import numpy as np
    from agents.model.features_extractor import PROJECTION_DIM

    version = ModelVersion.from_layout_and_policy_kwargs(layout, policy_kwargs)
    total_dim = layout["total_dim"]
    tmpdir = tempfile.mkdtemp(prefix="roundtrip_")
    try:
        zip_path = os.path.join(tmpdir, "roundtrip_model")
        model.save(zip_path)
        save_model_snapshot(tmpdir, version, git_hash="roundtrip-test")
        reloaded = load_model_snapshot(
            zip_path + ".zip",
            env=model.get_env(),
            current_version=version,
            device=str(model.device),
        )
        dev = next(reloaded.policy.parameters()).device
        dummy_obs = {
            "observation": torch.zeros(1, total_dim, device=dev),
            "action_mask": torch.ones(1, 11, dtype=torch.int8, device=dev),
        }
        with torch.no_grad():
            pi_features, vf_features = reloaded.policy.features_extractor(dummy_obs)
        assert pi_features.shape == (1, PROJECTION_DIM), (
            f"Round-trip test: unexpected policy-feature shape {pi_features.shape}, expected (1, {PROJECTION_DIM})"
        )
        assert vf_features.shape == (1, PROJECTION_DIM), (
            f"Round-trip test: unexpected value-feature shape {vf_features.shape}, expected (1, {PROJECTION_DIM})"
        )
        if debug:
            print(f"[ModelVersion] Round-trip smoke test PASSED (pi+vf shape: {tuple(pi_features.shape)})")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# Wait for an in-flight subprocess eval to FINISH on a graceful restart so its
# results land before exit. A scheduled restart is self-initiated by
# GracefulRestartCallback at a rollout boundary, and the launcher won't force-kill
# until the child overruns the deadline by --restart-grace-minutes (20 min default),
# so a 10-min drain fits. The checkpoint is saved first either way, so even the
# pathological forced-SIGTERM case (child already overran → ~90s SIGKILL) is safe —
# it only risks losing the in-flight eval, never the checkpoint.
_ABORT_EVAL_DRAIN_SEC = 600.0


def _setup_signal_handlers(model, model_dir, shutdown_event, version, current_lr_fn, current_epochs_fn, handoff_lr_fn=None, eval_drain_fn=None):
    """Wire SIGINT/SIGTERM/SIGUSR1. Returns the abort_training closure so it can
    be passed to eval callbacks as their canonical "die cleanly" path.

    ``handoff_lr_fn`` is optional; when present it returns the TwoPhaseLR
    callback's current handoff_lr (or None while still in Phase 1) so the
    cosine starting LR is persisted alongside the SIGTERM checkpoint.

    ``eval_drain_fn`` is optional; when present it is called AFTER the checkpoint
    is safely saved to wait (briefly, bounded) for an in-flight subprocess eval so
    its results land before exit. Bounded so the child still exits inside the
    launcher's SIGKILL grace — the checkpoint is already safe regardless.
    """

    def _handoff() -> "float | None":
        return handoff_lr_fn() if handoff_lr_fn is not None else None

    def abort_training(reason: str) -> None:
        """Single canonical abort path — works from any thread.

        Saves a full checkpoint (with metadata + latest.txt), then exits with
        TrainExitCode.INTERRUPTED (15) so the launcher restarts the run.
        Uses os._exit() rather than sys.exit() so it terminates the whole
        process even when called from a background eval thread.
        """
        shutdown_event.set()
        print(f"\n[ABORT] {reason}")
        try:
            path = os.path.join(model_dir, "final_model_interrupted")
            model.save(path)
            _write_latest_txt(model_dir, "final_model_interrupted.zip")
            lr = current_lr_fn()
            epochs = current_epochs_fn()
            hparams = _model_hparams(model)
            save_model_snapshot(model_dir, version, current_lr=lr, current_epochs=epochs, hparams=hparams)
            record_checkpoint(model_dir, path + ".zip", lr, epochs, hparams=hparams, handoff_lr=_handoff())
            print(f"[ABORT] Checkpoint saved → {path}.zip")
        except Exception as e:
            print(f"[ABORT] Save failed: {e}")
        # Checkpoint is safe; now wait for any in-flight eval to FINISH so its results
        # land in metadata.json before we exit (bounded by _ABORT_EVAL_DRAIN_SEC, which
        # fits inside the scheduled-restart grace window).
        if eval_drain_fn is not None:
            try:
                eval_drain_fn()
            except Exception as e:
                print(f"[ABORT] eval drain failed: {e}")
        os._exit(int(TrainExitCode.INTERRUPTED))

    def _forced_checkpoint(sig, frame):
        step = model.num_timesteps
        name = f"checkpoint_forced_{step:010d}_{datetime.now().strftime('%H%M%S')}"
        ckpt = os.path.join(model_dir, name)
        model.save(ckpt)
        _write_latest_txt(model_dir, name + ".zip")
        record_checkpoint(
            model_dir,
            os.path.join(model_dir, name + ".zip"),
            current_lr_fn(),
            current_epochs_fn(),
            hparams=_model_hparams(model),
            handoff_lr=_handoff(),
        )
        print(f"\n💾 [CHECKPOINT] Forced save → {ckpt}.zip")

    signal.signal(signal.SIGINT,  lambda sig, frame: abort_training("SIGINT received"))
    signal.signal(signal.SIGTERM, lambda sig, frame: abort_training("SIGTERM received"))
    signal.signal(signal.SIGUSR1, _forced_checkpoint)
    return abort_training


async def main():
    # --- Pre-flight Checks ---
    try:
        import tensorboard
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

    parser = argparse.ArgumentParser(description="Train or Evaluate Gen 3 OU RL Agent")
    
    # --- Operational Flags ---
    parser.add_argument("--model", type=str, help="Path to existing model to load")
    parser.add_argument("--run-dir", type=str, help="Run folder to write checkpoints into (set by launcher on resume)")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and only evaluate")
    parser.add_argument("--steps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--debug", action="store_true", help="Use DummyVecEnv (1 env) for debugging")
    parser.add_argument("--n-envs", type=int, default=32, help="Number of parallel environments")
    parser.add_argument("--device", type=str, default="auto", help="Device to use (cpu, cuda, or auto)")
    parser.add_argument("--showdown-port", type=int, default=None,
                        help="Local Showdown server port (default 8000). Sets the port for the trainee, "
                             "eval, and self-play clients. Start the server on the matching port, "
                             "e.g. npm run showdown -- <port>.")
    parser.add_argument(
        "--self-play-use-cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Load self-play opponent snapshots on CPU instead of the training device. "
             "Default True: avoids one CUDA context per SubprocVecEnv worker (~300-600 MB each), "
             "which would otherwise OOM the GPU at high --n-envs. Opponent inference is batch-1 "
             "no_grad, so CPU is plenty fast. Pass --no-self-play-use-cpu to load them on --device.",
    )
    parser.add_argument("--eval-battles", type=int, default=100, help="Battles per evaluation opponent")
    parser.add_argument("--eval-concurrency", type=int, default=100, help="Concurrent battles during evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--log-level", type=str, default="periodic", choices=["quiet", "periodic", "detailed", "debug"], help="Logging verbosity level")

    # --- Hyperparameter Flags (Optimized for GPU) ---
    parser.add_argument("--batch-size", type=int, default=4096, help="PPO mini-batch size")
    parser.add_argument("--n-epochs", type=int, default=5, help="PPO optimization epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate (AdaptiveLRCallback adjusts from here)")
    parser.add_argument("--min-lr", type=float, default=1e-5, help="Hard lower bound on adaptive LR")
    parser.add_argument("--max-lr", type=float, default=None, help="Hard upper bound on adaptive LR (default: 2× --lr)")
    parser.add_argument("--anneal-lr-start-steps", type=int, default=None,
                        help="Absolute global step at which cosine LR decay begins. "
                             "Duration = --steps minus this value. Pass the same value on every resume.")
    parser.add_argument("--anneal-min-lr", type=float, default=None,
                        help="LR floor for annealing (required with --anneal-lr-start-steps). "
                             "Separate from --min-lr used by AdaptivePPO.")
    parser.add_argument("--ent-coef", type=float, default=0.02, help="Entropy coefficient (exploration bonus)")
    parser.add_argument("--clip-range", type=float, default=CLIP_RANGE_DEFAULT, help="PPO policy clip range (default 0.15)")
    parser.add_argument("--clip-range-vf", type=float, default=0.5, help="Value function clip range (None=disabled; thesis used 0.0184)")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per environment per rollout")
    parser.add_argument("--weight-decay", type=float, default=1e-5,
                        help="AdamW weight decay (L2 regularisation). Default 1e-5 is conservative for PPO.")

    # --- Self-Play Flags ---
    parser.add_argument("--use-v2-bots", "--use_v2_bots", dest="use_v2_bots", action="store_true", default=False,
                        help="Add the V2 heuristic bots (Heuristic2, StallerV2, AggressiveV2, SetupSweepV2) to the training opponent pool and to eval")
    # --- Subprocess eval ---
    parser.add_argument("--eval-workers", "--eval_workers", dest="eval_workers", type=int, default=3,
                        help="Number of parallel eval-worker subprocesses per cycle. Workers work-steal "
                             "opponents from a shared pool, so uneven per-opponent cost self-balances. "
                             "Capped at the opponent count.")
    parser.add_argument("--eval-device", "--eval_device", dest="eval_device", type=str, default="cpu",
                        help="Device for the eval-worker subprocess inference (default cpu, to decouple from the training GPU).")
    parser.add_argument("--keep-eval-snapshots", "--keep_eval_snapshots", dest="keep_eval_snapshots",
                        type=int, default=10,
                        help="Retain the N most-recent eval weight snapshots in eval_traces/step_<N>/snapshot.zip "
                             "so the prober can reload the bit-exact model that produced a cycle's traces "
                             "(~27MB each; default 10 ≈ 270MB). 0 only writes the identity manifest; the prober "
                             "then falls back to the nearest persisted checkpoint.")
    parser.add_argument("--keep-eval-trace-steps", "--keep_eval_trace_steps", dest="keep_eval_trace_steps",
                        type=int, default=20,
                        help="The trainer grooms the forensic traces it writes: after each eval cycle it "
                             "keeps only the N most-recent eval step dirs under eval_traces/ (0 = keep all). "
                             "`python -m main.prober.groom` is the manual fallback for finished runs.")
    parser.add_argument("--self-play", action="store_true", default=False, help="Enable self-play snapshot pool as training opponents")
    parser.add_argument("--snapshot-dir", type=str, default=None, help="Pool directory (default: <run_dir>/snapshots)")
    parser.add_argument("--promote-threshold", type=float, default=0.65, help="Win rate vs. pool to trigger snapshot promotion")
    parser.add_argument("--self-play-temp", type=float, default=1.0,
                        help="Sampling temperature for self-play TRAINING opponents (they sample, "
                             "not argmax, so the learner faces the policy's full action distribution). "
                             "1.0 = the policy's own distribution; >1 flatter/more random; lower → toward "
                             "greedy. Eval opponents stay deterministic regardless.")

    args = parser.parse_args()
    log_level = LogLevel[args.log_level.upper()]

    # One server config, built from --showdown-port and threaded to every Showdown client
    # (training-env players in spawn workers, eval, and self-play). Default port: 8000.
    server_config = (
        LocalhostServerConfiguration
        if args.showdown_port is None
        else localhost_server_configuration(args.showdown_port)
    )
    emit(f"🔌 Showdown server: {server_config.websocket_url}")

    annealing_mode = args.anneal_lr_start_steps is not None
    if annealing_mode:
        if args.anneal_min_lr is None:
            print("[AnnealLR] ERROR: --anneal-min-lr is required when --anneal-lr-start-steps is set")
            sys.exit(1)
        if args.anneal_lr_start_steps >= args.steps:
            print(f"[AnnealLR] ERROR: --anneal-lr-start-steps ({args.anneal_lr_start_steps:,}) "
                  f"must be less than --steps ({args.steps:,})")
            sys.exit(1)

    # Automatically enable deep traces if --debug is set
    if args.debug:
        log_level = LogLevel.DEBUG
        # A --debug smoke run is a short-lived child of the launching shell/agent and
        # uses DummyVecEnv (no SubprocVecEnv worker watchdog). If its parent dies it gets
        # orphaned, and a hung smoke (e.g. a vanished 9XXX server) then lingers for days as
        # a zombie. Exit if reparented. Started here — before team/env/server setup — so a
        # hang anywhere in startup is covered too. Real (launcher-managed) runs keep a live
        # parent and are unaffected.
        start_orphan_watchdog(label="debug-smoke")

    # Load all teams using the new TeamLoader
    loader = TeamLoader()
    sample_teams = loader.get_sample_teams()
    all_teams = loader.get_all_teams()
    
    emit(f"📦 {len(sample_teams)} sample teams (bias) / {len(all_teams)} total loaded")

    # Trainee draws from the full pool, but 50% of the time uses a sample team.
    # This exposes the agent to diverse team compositions while keeping a stable anchor.
    trainee_teambuilder = Gen3Teambuilder(all_teams, bias_teams=sample_teams, bias_prob=0.1)
    opponent_teambuilder = Gen3Teambuilder(all_teams)

    mappings = load_mappings()
    
    OPPONENT_CLASSES = [
        SimpleHeuristicsPlayer,
        Gen3StallerPlayer,
        Gen3AggressivePlayer,
        Gen3SetupSweepPlayer,
    ]
    if args.use_v2_bots:
        OPPONENT_CLASSES += [
            Gen3HeuristicV2Player,
            Gen3StallerV2Player,
            Gen3AggressiveV2Player,
            Gen3SetupSweepV2Player,
        ]
        print(f"[Opponents] --use-v2-bots: training pool = {len(OPPONENT_CLASSES)} bots "
              f"({', '.join(opponent_name(c) for c in OPPONENT_CLASSES)})")

    def create_training_env_random(idx, stall_config=None, snapshot_path=None, opponent_device="auto", opponent_version=None):
        def _init():
            try:
                ts = datetime.now().strftime('%H%M%S')
                env_username = f"RLAgent{idx}{ts}"
                opp_username = f"Opponent{idx}{ts}"

                env_log_level = log_level if idx == 0 else LogLevel.QUIET

                env = Gen3Env(
                    mappings,
                    battle_format=BATTLE_FORMAT,
                    team=trainee_teambuilder,
                    log_level=env_log_level,
                    stall_config=stall_config,
                    reward_fn=reward_factory(log_level=env_log_level),
                    server_configuration=server_config,
                    account_configuration1=AccountConfiguration(env_username, "password"),
                )

                if snapshot_path is not None:
                    # Load via the version-checked path (same as eval sentinels) so an
                    # arch-mismatched snapshot fails with a clean ModelVersionError rather
                    # than loading garbage weights. The pool writes a model_config.json
                    # alongside its snapshots, so this check is real (see snapshot_pool.py).
                    pool_model = load_model_snapshot(
                        snapshot_path, env=None,
                        current_version=opponent_version, device=opponent_device,
                    )
                    opponent = RLPlayer(
                        model=pool_model,
                        team=opponent_teambuilder,
                        battle_format=BATTLE_FORMAT,
                        server_configuration=server_config,
                        mappings=mappings,
                        account_configuration=AccountConfiguration(opp_username, "password"),
                        # Self-play opponents SAMPLE (temperature-scaled) rather than play
                        # greedily, so the learner trains against the policy's full action
                        # distribution — a richer, less exploitable signal. Eval stays argmax.
                        stochastic=True,
                        temperature=args.self_play_temp,
                        # Strict (crash-over-corruption): a stale decision context crashes the
                        # worker exactly like the trainee. A self-play opponent's default move
                        # would be garbage-in — it IS the trainee's training signal — so we
                        # never tolerate staleness; the launcher restarts from the checkpoint.
                    )
                else:
                    opponent_cls = random.choice(OPPONENT_CLASSES)
                    opponent = opponent_cls(
                        battle_format=BATTLE_FORMAT,
                        team=opponent_teambuilder,
                        server_configuration=server_config,
                        account_configuration=AccountConfiguration(opp_username, "password"),
                    )

                wrapped = MaskableAgentWrapper(env, opponent)

                # FORCE OVERRIDE: SingleAgentWrapper hardcodes 10 for gen3ou. We need 11.
                # Also ensure it propagates our Dict observation space natively.
                wrapped.action_space = env.action_space
                wrapped.observation_space = env.observation_space

                return Monitor(wrapped)
            except Exception as e:
                print(f"🛑 ERROR IN WORKER {idx}: {e}")
                traceback.print_exc()
                raise e
        return _init

    # --- Directory Setup ---
    if args.run_dir:
        model_dir = args.run_dir                                     # launcher-managed resume
    else:
        model_dir = f"models/run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    os.makedirs(model_dir, exist_ok=True)
    tb_run_name = f"MPPO_{os.path.basename(model_dir)}"
    if not args.run_dir:
        with open(os.path.join(model_dir, "command.txt"), "w") as f:
            f.write(" ".join(sys.argv))
        
    stall_cfg = StallConfig(output_dir=os.path.join(model_dir, "stalls"))
    reward_factory = Gen3RewardManager

    # Running parallel environments
    n_envs = 1 if args.debug else args.n_envs
    EnvClass = DummyVecEnv if args.debug else SubprocVecEnv

    emit(f"⚙️ Initializing {n_envs} envs ({EnvClass.__name__})")

    _shutdown_event = threading.Event()

    # --- Self-Play Pool Setup ---
    # Pool is created before envs so it can provide snapshot paths to env factories.
    # On launcher restart the pool directory already exists; _scan() restores state.
    _pool: SnapshotPool | None = None
    _opp_version = None  # ModelVersion threaded into opponent snapshot loads (set when self-play on)
    if args.self_play:
        from pathlib import Path as _Path
        from agents.model.snapshot import current_model_version as _current_model_version

        _snapshot_dir = _Path(args.snapshot_dir) if args.snapshot_dir else _Path(model_dir) / "snapshots"
        _cv = _current_model_version(mappings)
        _opp_version = _cv
        _pool = SnapshotPool(
            pool_dir=_snapshot_dir,
            current_version=_cv,
            device=args.device,
        )
        _persisted_wr = _pool.load_persisted_win_rate()
        _hfrac = heuristic_fraction(_persisted_wr)
        _n_pool_envs = 0 if _pool.is_empty() else max(1, int(round(n_envs * (1.0 - _hfrac))))
        emit(
            f"🎮 [SELFPLAY] Pool has {len(_pool)} snapshots, "
            f"win_rate_vs_bots={_persisted_wr:.2%}, "
            f"heuristic_fraction={_hfrac:.0%} → {_n_pool_envs}/{n_envs} envs use pool opponents"
        )
        if args.self_play and _pool.is_empty():
            emit("🌱 [SELFPLAY] Pool empty — will seed from the loaded weights and rebuild "
                 "self-play envs right after model load (engages this process, not next restart)")
    else:
        _n_pool_envs = 0

    opponent_device = "cpu" if args.self_play_use_cpu else args.device
    if _n_pool_envs > 0:
        emit(
            f"🧠 [SELFPLAY] Opponent snapshots load on '{opponent_device}' "
            f"({'CPU — avoids per-worker CUDA contexts' if args.self_play_use_cpu else 'training device'})"
        )

    def _make_factories():
        factories = []
        pool_entries = []
        if _pool and not _pool.is_empty() and _n_pool_envs > 0:
            for _ in range(_n_pool_envs):
                pool_entries.append(str(_pool.sample().path))
        for i in range(n_envs):
            snap = pool_entries[i] if i < len(pool_entries) else None
            factories.append(
                create_training_env_random(
                    i, stall_config=stall_cfg, snapshot_path=snap,
                    opponent_device=opponent_device, opponent_version=_opp_version,
                )
            )
        return factories

    env_factories = _make_factories()
    env = EnvClass(env_factories)
    # NOTE: the subprocess watchdog is started LATER, just before model.learn() — nothing
    # steps the env before then (model construction/load only reads its spaces), and a
    # first-self-play-process rebuild (see _maybe_engage_self_play) closes this env, which
    # would otherwise trip the watchdog's "worker died" guard.

    def _maybe_engage_self_play(model, env):
        """Seed the pool from the just-loaded weights and rebuild the training env so
        self-play opponents engage in THIS process — not only after the next restart.

        Runs only when --self-play is on AND the pool is empty (the first self-play
        process). The env was built before the model existed (the model needs the env's
        spaces), and the pool can only be seeded once the model exists — so we seed from
        the in-memory weights (the loaded checkpoint, or fresh init) and rebuild the env
        against the now-non-empty pool. On every later restart the pool already has
        snapshots, so this is a no-op and no rebuild happens."""
        nonlocal _n_pool_envs
        if not (args.self_play and _pool is not None and _pool.is_empty()):
            return env
        _pool.seed(model)
        _persisted = _pool.load_persisted_win_rate()
        _n_pool_envs = max(1, int(round(n_envs * (1.0 - heuristic_fraction(_persisted)))))
        emit(f"🌱 [SELFPLAY] Seeded pool from current weights → rebuilding "
             f"{_n_pool_envs}/{n_envs} envs with self-play opponents (engages now)")
        env.close()
        new_env = EnvClass(_make_factories())
        model.set_env(new_env)
        return new_env

    async def evaluate_model_random(model):
        ts = datetime.now().strftime('%H%M%S')
        n = args.eval_battles
        print(f"\nFinal Evaluation (Session {ts}, Battles: {n}, Concurrency: {args.eval_concurrency})...")

        rl_player = RLPlayer(
            model=model,
            team=trainee_teambuilder,
            battle_format=BATTLE_FORMAT,
            server_configuration=server_config,
            mappings=mappings,
            account_configuration=AccountConfiguration(f"RLFinal{ts}", "password"),
            max_concurrent_battles=args.eval_concurrency,
            stochastic=False,  # final eval = greedy policy
        )

        final_opponents = [
            (opponent_name(RandomPlayer), RandomPlayer(
                battle_format=BATTLE_FORMAT, team=opponent_teambuilder,
                server_configuration=server_config,
                account_configuration=AccountConfiguration(f"FinalRand{ts}", "password"),
                max_concurrent_battles=args.eval_concurrency,
            )),
            (opponent_name(SimpleHeuristicsPlayer), SimpleHeuristicsPlayer(
                battle_format=BATTLE_FORMAT, team=opponent_teambuilder,
                server_configuration=server_config,
                account_configuration=AccountConfiguration(f"FinalHeur{ts}", "password"),
                max_concurrent_battles=args.eval_concurrency,
            )),
            (opponent_name(Gen3StallerPlayer), Gen3StallerPlayer(
                battle_format=BATTLE_FORMAT, team=opponent_teambuilder,
                server_configuration=server_config,
                account_configuration=AccountConfiguration(f"FinalStall{ts}", "password"),
                max_concurrent_battles=args.eval_concurrency,
            )),
            (opponent_name(Gen3AggressivePlayer), Gen3AggressivePlayer(
                battle_format=BATTLE_FORMAT, team=opponent_teambuilder,
                server_configuration=server_config,
                account_configuration=AccountConfiguration(f"FinalAggr{ts}", "password"),
                max_concurrent_battles=args.eval_concurrency,
            )),
            (opponent_name(Gen3SetupSweepPlayer), Gen3SetupSweepPlayer(
                battle_format=BATTLE_FORMAT, team=opponent_teambuilder,
                server_configuration=server_config,
                account_configuration=AccountConfiguration(f"FinalSetup{ts}", "password"),
                max_concurrent_battles=args.eval_concurrency,
            )),
        ]
        if args.use_v2_bots:
            for _cls, _uname in [
                (Gen3HeuristicV2Player, f"FinalHeur2{ts}"),
                (Gen3StallerV2Player, f"FinalStallV2{ts}"),
                (Gen3AggressiveV2Player, f"FinalAggrV2{ts}"),
                (Gen3SetupSweepV2Player, f"FinalSetupV2{ts}"),
            ]:
                final_opponents.append((opponent_name(_cls), _cls(
                    battle_format=BATTLE_FORMAT, team=opponent_teambuilder,
                    server_configuration=server_config,
                    account_configuration=AccountConfiguration(_uname, "password"),
                    max_concurrent_battles=args.eval_concurrency,
                )))

        win_rates: dict[str, float] = {}
        for name, opponent in final_opponents:
            if rl_player.n_finished_battles > 0:
                rl_player.reset_battles()
            print(f"  vs {name} [{n} battles]...")
            start_time = datetime.now()
            await rl_player.battle_against(opponent, n_battles=n)
            duration = datetime.now() - start_time
            wr = rl_player.n_won_battles / rl_player.n_finished_battles
            win_rates[name] = wr
            print(f"  Win rate vs {name}: {wr * 100:.1f}%  [{duration}]")
            model.logger.record(f"eval_final/win_rate_vs_{name}", wr)

        aggregate = sum(win_rates.values()) / len(win_rates)
        model.logger.record("eval_final/win_rate_mean", aggregate)
        model.logger.dump(model.num_timesteps)
        print(f"\nFinal aggregate win rate: {aggregate * 100:.1f}%")

    # --- Callback Setup (Shared) ---
    checkpoint_callback = _TrackingCheckpointCallback(
        save_freq=50000,
        save_path=model_dir,
        name_prefix="checkpoint",
    )

    # --lr must lie within [--min-lr, --max-lr]. This is the user-facing contract
    # for both pure-adaptive runs and TwoPhaseLR Phase 1 — KL adaptation reads
    # args.lr as the seed for fresh runs and as the cap for resumes, so it has
    # to be a valid in-band LR. Enforced before any callback is constructed.
    _effective_max_lr = args.max_lr if args.max_lr is not None else args.lr * 2.0
    if not (args.min_lr <= args.lr <= _effective_max_lr):
        print(f"[AdaptiveLR] ERROR: --lr {args.lr:.2e} is outside "
              f"[--min-lr {args.min_lr:.2e}, --max-lr {_effective_max_lr:.2e}]")
        sys.exit(1)

    # If resuming and the checkpoint has already crossed into the cosine phase,
    # read the persisted handoff_lr so the new callback can pick up the same
    # cosine starting point. None means "still in Phase 1, KL-driven."
    resumed_handoff_lr: float | None = None
    if annealing_mode and args.model and os.path.exists(args.model):
        try:
            _meta = read_checkpoint_metadata(args.model)
            _h = _meta.get("handoff_lr")
            if isinstance(_h, (int, float)):
                resumed_handoff_lr = float(_h)
        except Exception as e:
            print(f"[TwoPhaseLR] WARNING: failed to read handoff_lr from {args.model}: {e}")

    if annealing_mode:
        lr_callback = TwoPhaseLRCallback(
            initial_lr=args.lr,
            total_steps=args.steps,
            anneal_start_steps=args.anneal_lr_start_steps,
            anneal_min_lr=args.anneal_min_lr,
            min_lr=args.min_lr,
            max_lr=args.max_lr,
            handoff_lr=resumed_handoff_lr,
        )
    else:
        lr_callback = AdaptivePPOCallback(
            initial_lr=args.lr,
            min_lr=args.min_lr,
            max_lr=args.max_lr,
        )
    # Keep alias so references below still resolve during the resume path.
    adaptive_ppo_callback = lr_callback
    checkpoint_callback._current_lr_fn = lambda: model.policy.optimizer.param_groups[0]["lr"]
    checkpoint_callback._current_epochs_fn = lambda: model.n_epochs
    # Only TwoPhaseLRCallback exposes a handoff_lr; AdaptivePPOCallback does not.
    checkpoint_callback._handoff_lr_fn = (
        (lambda: lr_callback.handoff_lr) if isinstance(lr_callback, TwoPhaseLRCallback) else None
    )
    graceful_restart_callback = GracefulRestartCallback()
    callbacks = [checkpoint_callback, lr_callback, MetricsExporterCallback(), _HparamLogCallback(args.ent_coef), graceful_restart_callback]
    eval_callback = None

    # On resume, the last eval lives in the resumed checkpoint's metadata.json (a different
    # dir from this fresh run) — point the eval callback at it so the TUI shows the most
    # recent eval immediately instead of a blank panel until the next cycle.
    _resume_meta = None
    if args.model:
        _ckpt_dir = args.model if os.path.isdir(args.model) else os.path.dirname(args.model)
        if _ckpt_dir:
            _resume_meta = os.path.join(_ckpt_dir, "metadata.json")

    if args.self_play and _pool is not None:
        # Self-play eval mirrors the bot-eval frozen-snapshot SUBPROCESS pattern
        # (non-blocking): the workers work-steal the bot roster AND up to 5 pool sentinels,
        # play a frozen snapshot, and the parent collects + promotes on a later poll. The
        # worker rebuilds opponents / teambuilders / mappings itself from the data dir, so
        # nothing live is constructed here. It runs even under --debug (fast eval cadence)
        # so a short CPU smoke against a 9XXX server exercises seed → pool eval → promotion.
        eval_callback = SelfPlayCallback(
            pool=_pool,
            model_dir=model_dir,
            server_config=server_config,
            showdown_port=args.showdown_port,
            use_v2_bots=args.use_v2_bots,
            best_model_save_path=os.path.join(model_dir, "best_model"),
            promote_threshold=args.promote_threshold,
            self_play_temp=args.self_play_temp,
            n_workers=args.eval_workers,
            eval_device=args.eval_device,
            keep_eval_snapshots=args.keep_eval_snapshots,
            keep_eval_trace_steps=args.keep_eval_trace_steps,
            resume_eval_metadata=_resume_meta,
            debug=args.debug,
        )
        callbacks.append(eval_callback)
    elif not args.debug:
        # Bot eval runs in a frozen-snapshot subprocess (non-blocking, CPU). The
        # worker rebuilds opponents/teambuilders/mappings itself from the data
        # dir, so nothing live is constructed here.
        eval_callback = PerOpponentEvalCallback(
            model_dir=model_dir,
            server_config=server_config,
            use_v2_bots=args.use_v2_bots,
            best_model_save_path=os.path.join(model_dir, "best_model"),
            n_workers=args.eval_workers,
            eval_device=args.eval_device,
            showdown_port=args.showdown_port,
            resume_eval_metadata=_resume_meta,
            keep_eval_snapshots=args.keep_eval_snapshots,
            keep_eval_trace_steps=args.keep_eval_trace_steps,
        )
        callbacks.append(eval_callback)

    if args.model:
        model_path = args.model
        if not os.path.exists(model_path) and not model_path.endswith(".zip"):
            potential_paths = [
                os.path.join("models", "goldens", model_path),
                os.path.join("models", "goldens", model_path, "final_model"),
                os.path.join("models", "goldens", model_path, "final_model.zip"),
            ]
            for p in potential_paths:
                if os.path.exists(p) or os.path.exists(p + ".zip"):
                    model_path = p
                    break

        # Build the current-code version for compatibility check
        _load_encoder = Gen3ObservationEncoder(mappings)
        _load_extractor_kwargs = _load_encoder.get_features_extractor_kwargs()
        _load_policy_kwargs = {
            "features_extractor_class": Gen3FeaturesExtractor,
            "features_extractor_kwargs": _load_extractor_kwargs,
            "net_arch": NET_ARCH,
        }
        current_version = ModelVersion.from_layout_and_policy_kwargs(
            _load_extractor_kwargs["layout"], _load_policy_kwargs
        )

        print(f"Loading existing model from {model_path}")
        try:
            model = load_model_snapshot(
                model_path,
                env=env,
                current_version=current_version,
                device=args.device,
                tensorboard_log=tensorboard_dir,
            )
        except ModelVersionError as e:
            print(f"\n[ModelVersion] FATAL: {e}")
            os._exit(1)
        model.ent_coef = args.ent_coef
        model.gae_lambda = 0.80
        # Resume-path LR setup. Phase determines whether we read from the
        # optimizer (Phase 1, KL-driven) or compute the cosine (Phase 2).
        saved_lr: float | None = None  # only set in branches that read it
        if annealing_mode:
            t = model.num_timesteps
            if lr_callback.phase(t) == 1:
                # Phase 1: KL-driven adaptation continues from the optimizer's saved
                # LR. --lr is a fresh-run seed only; on resume it's ignored so the
                # controller can keep whatever rate Phase 1 had settled on. The
                # saved LR is still clamped into [min_lr, max_lr] in case the user
                # tightened the bounds between restarts.
                saved_lr = model.policy.optimizer.param_groups[0]["lr"]
                resume_lr = saved_lr
                resume_lr_clamped = max(args.min_lr, min(resume_lr, _effective_max_lr))
                if resume_lr_clamped != resume_lr:
                    print(
                        f"[TwoPhaseLR] Clamping resume LR {resume_lr:.2e} → {resume_lr_clamped:.2e} "
                        f"to fit [{args.min_lr:.2e}, {_effective_max_lr:.2e}] (saved={saved_lr:.2e})."
                    )
                resume_lr = resume_lr_clamped
                model.lr_schedule = lambda _: resume_lr
                lr_callback._current_lr = resume_lr
                lr_detail = f"Phase 1 adaptive, saved={saved_lr:.2e} (arg --lr={args.lr:.2e} ignored on resume)"
                send_event(
                    f"▶️ Resuming TwoPhaseLR Phase 1 at LR {resume_lr:.2e}, step {t:,} "
                    f"(anneal_start={args.anneal_lr_start_steps:,})"
                )
            else:
                # Phase 2: cosine decay. handoff_lr was passed to the constructor
                # from the sidecar; fall back to the optimizer's LR if missing
                # (legacy run that pre-dates handoff_lr persistence).
                if lr_callback.handoff_lr is None:
                    lr_callback._handoff_lr = model.policy.optimizer.param_groups[0]["lr"]
                    print(
                        f"[TwoPhaseLR] No persisted handoff_lr on resume; "
                        f"using optimizer LR {lr_callback._handoff_lr:.2e} as cosine start."
                    )
                resume_lr = lr_callback._cosine_lr_at(t)
                model.lr_schedule = lambda _: resume_lr
                lr_callback._current_lr = resume_lr
                lr_detail = (
                    f"Phase 2 cosine, handoff={lr_callback.handoff_lr:.2e} → "
                    f"min={args.anneal_min_lr:.2e}"
                )
                send_event(
                    f"▶️ Resuming TwoPhaseLR Phase 2 (cosine) at LR {resume_lr:.2e}, step {t:,} "
                    f"(handoff={lr_callback.handoff_lr:.2e}, target={args.anneal_min_lr:.2e} at {args.steps:,})"
                )
        else:
            # Pure adaptive: resume LR from optimizer state. --lr is a fresh-run
            # seed only; on resume the controller keeps whatever rate it had
            # settled on. The saved LR is still clamped into [min_lr, max_lr]
            # in case the user tightened the bounds between restarts.
            saved_lr = model.policy.optimizer.param_groups[0]["lr"]
            resume_lr = saved_lr
            resume_lr_clamped = max(args.min_lr, min(resume_lr, _effective_max_lr))
            if resume_lr_clamped != resume_lr:
                print(
                    f"[AdaptiveLR] Clamping resume LR {resume_lr:.2e} → {resume_lr_clamped:.2e} "
                    f"to fit [{args.min_lr:.2e}, {_effective_max_lr:.2e}] (saved={saved_lr:.2e})."
                )
            resume_lr = resume_lr_clamped
            model.lr_schedule = lambda _: resume_lr
            adaptive_ppo_callback._current_lr = resume_lr
            lr_detail = f"saved={saved_lr:.2e} (arg --lr={args.lr:.2e} ignored on resume)"
            send_event(f"▶️ Resuming at LR {resume_lr:.2e}, epochs {args.n_epochs} (checkpoint LR={saved_lr:.2e})")
        model.n_epochs = args.n_epochs
        model.clip_range = lambda _: args.clip_range
        model.clip_range_vf = lambda _: args.clip_range_vf

        if args.eval_only:
            await evaluate_model_random(model)
            return
        else:
            remaining_steps = args.steps - model.num_timesteps
            if remaining_steps <= 0:
                print(f"Training already complete ({model.num_timesteps:,} / {args.steps:,} steps)")
                sys.exit(TrainExitCode.COMPLETE)
            print(f"Continuing Training (Steps: {remaining_steps:,} remaining of {args.steps:,}, LR: {resume_lr:.2e} ({lr_detail}))")
            _run_roundtrip_test(model, _load_extractor_kwargs["layout"], _load_policy_kwargs, debug=args.debug)
            save_model_snapshot(model_dir, current_version, hparams=_model_hparams(model))

            _abort_fn = _setup_signal_handlers(
                model, model_dir, _shutdown_event, current_version,
                lambda: model.policy.optimizer.param_groups[0]["lr"],
                lambda: model.n_epochs,
                handoff_lr_fn=(
                    (lambda: lr_callback.handoff_lr)
                    if isinstance(lr_callback, TwoPhaseLRCallback) else None
                ),
                eval_drain_fn=(
                    (lambda: eval_callback.drain(timeout=_ABORT_EVAL_DRAIN_SEC))
                    if (eval_callback is not None and hasattr(eval_callback, "drain")) else None
                ),
            )
            if eval_callback is not None:
                eval_callback.abort_fn = _abort_fn
            graceful_restart_callback.abort_fn = _abort_fn

            # First self-play process: seed the pool from these weights and rebuild the
            # env so self-play engages now. No-op on every later restart (pool non-empty).
            # Then start the worker watchdog on the FINAL env, right before rollouts begin.
            env = _maybe_engage_self_play(model, env)
            start_subprocess_watchdog(env, label="train_env", shutdown_event=_shutdown_event)

            try:
                model.learn(total_timesteps=args.steps, callback=callbacks, reset_num_timesteps=False, tb_log_name=tb_run_name)
            except Exception as e:
                print(f"Training interrupted by exception: {e}")
                final_path = os.path.join(model_dir, "final_model_exception")
                model.save(final_path)
                _write_latest_txt(model_dir, "final_model_exception.zip")

            final_path = os.path.join(model_dir, "final_model")
            model.save(final_path)
            _write_latest_txt(model_dir, "final_model.zip")
            save_model_snapshot(os.path.dirname(final_path), current_version, hparams=_model_hparams(model))
            print(f"Training complete. Model saved to {final_path}")
            best_model_dir = os.path.join(model_dir, "best_model")
            if os.path.isdir(best_model_dir):
                save_model_snapshot(best_model_dir, current_version, hparams=_model_hparams(model))
            await evaluate_model_random(model)
    else:
        print(f"Starting NEW Training (Parallel x{n_envs}, Batch: {args.batch_size}, Epochs: {args.n_epochs})")
        # model_dir and unique_id are now pre-defined earlier in main()
        
        # Initialize a dummy encoder to get the handoff kwargs
        temp_encoder = Gen3ObservationEncoder(mappings)
        extractor_kwargs = temp_encoder.get_features_extractor_kwargs()
        extractor_kwargs["log_level"] = log_level
        
        policy_kwargs = {
            "features_extractor_class": Gen3FeaturesExtractor,
            "features_extractor_kwargs": extractor_kwargs,
            "net_arch": [512, 512],
            "optimizer_class": torch.optim.AdamW,
            "optimizer_kwargs": {"weight_decay": args.weight_decay, "eps": 1e-5},
        }
        
        # --- Model Initialization ---
        total_rollout_size = args.n_steps * n_envs
        if args.batch_size > total_rollout_size:
            print(f"Note: Capping batch_size from {args.batch_size} to {total_rollout_size} to match rollout capacity.")
            args.batch_size = total_rollout_size

        model = InstrumentedMaskablePPO(
            Gen3DualHeadMaskablePolicy,
            env,
            verbose=1,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.9999,
            gae_lambda=0.80,
            clip_range=args.clip_range,
            clip_range_vf=args.clip_range_vf,
            ent_coef=args.ent_coef,
            device=args.device,
            seed=args.seed,
            tensorboard_log=tensorboard_dir,
            policy_kwargs=policy_kwargs
        )

        version = ModelVersion.from_layout_and_policy_kwargs(extractor_kwargs["layout"], policy_kwargs)
        _run_roundtrip_test(model, extractor_kwargs["layout"], policy_kwargs, debug=args.debug)
        save_model_snapshot(model_dir, version, hparams=_model_hparams(model))

        _abort_fn = _setup_signal_handlers(
            model, model_dir, _shutdown_event, version,
            lambda: model.policy.optimizer.param_groups[0]["lr"],
            lambda: model.n_epochs,
            handoff_lr_fn=(
                (lambda: lr_callback.handoff_lr)
                if isinstance(lr_callback, TwoPhaseLRCallback) else None
            ),
            eval_drain_fn=(
                (lambda: eval_callback.drain(timeout=_ABORT_EVAL_DRAIN_SEC))
                if (eval_callback is not None and hasattr(eval_callback, "drain")) else None
            ),
        )
        if eval_callback is not None:
            eval_callback.abort_fn = _abort_fn
        graceful_restart_callback.abort_fn = _abort_fn

        # First self-play process: seed the pool from these (fresh-init) weights and
        # rebuild the env so self-play engages now. No-op on every later restart.
        # Then start the worker watchdog on the FINAL env, right before rollouts begin.
        env = _maybe_engage_self_play(model, env)
        start_subprocess_watchdog(env, label="train_env", shutdown_event=_shutdown_event)

        try:
            if log_level >= LogLevel.DETAILED:
                from sb3_contrib.common.maskable.utils import is_masking_supported
                print(f"✅ [DEBUG] Masking supported for env: {is_masking_supported(env)}")
            model.learn(total_timesteps=args.steps, callback=callbacks, reset_num_timesteps=False, tb_log_name=tb_run_name)
        except Exception as e:
            print("\n" + "🛑" * 30)
            print(f"🛑 TRAINING CRASHED: {e}")
            print("🛑" * 30)
            traceback.print_exc()
            os._exit(1) # Stop immediately, do not proceed to evaluation

        final_path = os.path.join(model_dir, "final_model")
        model.save(final_path)
        _write_latest_txt(model_dir, "final_model.zip")
        _final_handoff = lr_callback.handoff_lr if isinstance(lr_callback, TwoPhaseLRCallback) else None
        record_checkpoint(model_dir, final_path + ".zip", adaptive_ppo_callback.current_lr, model.n_epochs, hparams=_model_hparams(model), handoff_lr=_final_handoff)
        save_model_snapshot(os.path.dirname(final_path), version, hparams=_model_hparams(model))
        print(f"Training complete. Model saved to {final_path}")
        best_model_dir = os.path.join(model_dir, "best_model")
        if os.path.isdir(best_model_dir):
            save_model_snapshot(best_model_dir, version, hparams=_model_hparams(model))
        await evaluate_model_random(model)

if __name__ == "__main__":
    asyncio.run(main())
