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
from datetime import datetime
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor

from agents.model.features_extractor import Gen3FeaturesExtractor, NET_ARCH
from agents.model.model_version import ModelVersion, ModelVersionError
from agents.model.snapshot import save_model_snapshot, load_model_snapshot, read_checkpoint_metadata, record_checkpoint
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.inference.player import RLPlayer
from utils.teambuilder import Gen3Teambuilder
from utils.team_loader import TeamLoader
from agents.training.eval_callback import PerOpponentEvalCallback, opponent_name
from agents.training.snapshot_pool import SnapshotPool, heuristic_fraction
from agents.training.selfplay_callback import SelfPlayCallback
from agents.training.replay_recorder import ReplayCallback
from agents.training.wrappers import MaskableAgentWrapper
from agents.training.gen3_env import Gen3Env
from agents.training.reward_manager import Gen3RewardManager
from agents.training.stall import StallConfig
from agents.training.watchdog import start_subprocess_watchdog
from agents.training.adaptive_lr_callback import AdaptivePPOCallback
from agents.training.metrics_exporter_callback import MetricsExporterCallback
from utils.logging.levels import LogLevel
from main.exit_codes import TrainExitCode
from main.launcher.ipc import send_event, emit

from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from agents.opponents import Gen3StallerPlayer, Gen3AggressivePlayer, Gen3SetupSweepPlayer
from poke_env import AccountConfiguration, LocalhostServerConfiguration

BATTLE_FORMAT = "gen3ou"
CLIP_RANGE = 0.20


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
        self.logger.dump(self.num_timesteps)

    def _on_step(self) -> bool:
        return True


class _TrackingCheckpointCallback(CheckpointCallback):
    """CheckpointCallback that keeps latest.txt up to date and writes per-checkpoint metadata."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._current_lr_fn = None
        self._current_epochs_fn = None

    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.n_calls % self.save_freq == 0:
            _write_latest_txt(
                self.save_path,
                f"{self.name_prefix}_{self.num_timesteps}_steps.zip",
            )
            if self._current_lr_fn is not None and self._current_epochs_fn is not None:
                ckpt_path = os.path.join(self.save_path, f"{self.name_prefix}_{self.num_timesteps}_steps.zip")
                record_checkpoint(self.save_path, ckpt_path, self._current_lr_fn(), self._current_epochs_fn())
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
        )
        dev = next(reloaded.policy.parameters()).device
        dummy_obs = {
            "observation": torch.zeros(1, total_dim, device=dev),
            "action_mask": torch.ones(1, 11, dtype=torch.int8, device=dev),
        }
        with torch.no_grad():
            features = reloaded.policy.features_extractor(dummy_obs)
        assert features.shape == (1, PROJECTION_DIM), (
            f"Round-trip test: unexpected output shape {features.shape}, expected (1, {PROJECTION_DIM})"
        )
        if debug:
            print(f"[ModelVersion] Round-trip smoke test PASSED (output shape: {features.shape})")
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def _setup_signal_handlers(model, model_dir, shutdown_event, version, current_lr_fn, current_epochs_fn):
    """Wire SIGINT/SIGTERM/SIGUSR1. Returns the abort_training closure so it can
    be passed to eval callbacks as their canonical "die cleanly" path."""

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
            save_model_snapshot(model_dir, version, current_lr=current_lr_fn(), current_epochs=current_epochs_fn())
            print(f"[ABORT] Checkpoint saved → {path}.zip")
        except Exception as e:
            print(f"[ABORT] Save failed: {e}")
        os._exit(int(TrainExitCode.INTERRUPTED))

    def _forced_checkpoint(sig, frame):
        step = model.num_timesteps
        name = f"checkpoint_forced_{step:010d}_{datetime.now().strftime('%H%M%S')}"
        ckpt = os.path.join(model_dir, name)
        model.save(ckpt)
        _write_latest_txt(model_dir, name + ".zip")
        record_checkpoint(model_dir, os.path.join(model_dir, name + ".zip"), current_lr_fn(), current_epochs_fn())
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
    parser.add_argument("--eval-battles", type=int, default=100, help="Battles per evaluation opponent")
    parser.add_argument("--eval-concurrency", type=int, default=100, help="Concurrent battles during evaluation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--log-level", type=str, default="periodic", choices=["quiet", "periodic", "detailed", "debug"], help="Logging verbosity level")

    # --- Hyperparameter Flags (Optimized for GPU) ---
    parser.add_argument("--batch-size", type=int, default=4096, help="PPO mini-batch size")
    parser.add_argument("--n-epochs", type=int, default=4, help="PPO optimization epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Initial learning rate (AdaptiveLRCallback adjusts from here)")
    parser.add_argument("--ent-coef", type=float, default=0.02, help="Entropy coefficient (exploration bonus)")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per environment per rollout")

    # --- Self-Play Flags ---
    parser.add_argument("--self-play", action="store_true", default=False, help="Enable self-play snapshot pool as training opponents")
    parser.add_argument("--snapshot-dir", type=str, default=None, help="Pool directory (default: <run_dir>/snapshots)")
    parser.add_argument("--promote-threshold", type=float, default=0.65, help="Win rate vs. pool to trigger snapshot promotion")

    args = parser.parse_args()
    log_level = LogLevel[args.log_level.upper()]
    
    # Automatically enable deep traces if --debug is set
    if args.debug:
        log_level = LogLevel.DEBUG

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

    def create_training_env_random(idx, stall_config=None, snapshot_path=None, device="auto"):
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
                    server_configuration=LocalhostServerConfiguration,
                    account_configuration1=AccountConfiguration(env_username, "password"),
                )

                if snapshot_path is not None:
                    from sb3_contrib import MaskablePPO as _PPO
                    pool_model = _PPO.load(snapshot_path, device=device)
                    opponent = RLPlayer(
                        model=pool_model,
                        team=opponent_teambuilder,
                        battle_format=BATTLE_FORMAT,
                        server_configuration=LocalhostServerConfiguration,
                        account_configuration=AccountConfiguration(opp_username, "password"),
                    )
                else:
                    opponent_cls = random.choice(OPPONENT_CLASSES)
                    opponent = opponent_cls(
                        battle_format=BATTLE_FORMAT,
                        team=opponent_teambuilder,
                        server_configuration=LocalhostServerConfiguration,
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
    if args.self_play:
        from pathlib import Path as _Path
        from agents.model.model_version import ModelVersion as _MV
        from agents.model.features_extractor import Gen3FeaturesExtractor as _FE, NET_ARCH as _NA
        from agents.observation.state_encoder import Gen3ObservationEncoder as _OE

        _snapshot_dir = _Path(args.snapshot_dir) if args.snapshot_dir else _Path(model_dir) / "snapshots"
        _temp_enc = _OE(mappings)
        _temp_ext_kwargs = _temp_enc.get_features_extractor_kwargs()
        _temp_policy_kwargs = {
            "features_extractor_class": _FE,
            "features_extractor_kwargs": _temp_ext_kwargs,
            "net_arch": _NA,
        }
        _cv = _MV.from_layout_and_policy_kwargs(_temp_ext_kwargs["layout"], _temp_policy_kwargs)
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
            emit("⚠️  [SELFPLAY] Pool dir exists but is empty — seeding will happen after model load")
    else:
        _n_pool_envs = 0

    def _make_factories():
        factories = []
        pool_entries = []
        if _pool and not _pool.is_empty() and _n_pool_envs > 0:
            for _ in range(_n_pool_envs):
                pool_entries.append(str(_pool.sample().path))
        for i in range(n_envs):
            snap = pool_entries[i] if i < len(pool_entries) else None
            factories.append(
                create_training_env_random(i, stall_config=stall_cfg, snapshot_path=snap, device=args.device)
            )
        return factories

    env_factories = _make_factories()
    env = EnvClass(env_factories)
    start_subprocess_watchdog(env, label="train_env", shutdown_event=_shutdown_event)
    # Note: env.seed() is deprecated in gymnasium VecEnv, use seed in reset or at init if supported.
    # But for reproducibility, we pass it to PPO.

    async def evaluate_model_random(model):
        ts = datetime.now().strftime('%H%M%S')
        n = args.eval_battles
        print(f"\nFinal Evaluation (Session {ts}, Battles: {n}, Concurrency: {args.eval_concurrency})...")

        rl_player = RLPlayer(
            model=model,
            team=trainee_teambuilder,
            battle_format=BATTLE_FORMAT,
            server_configuration=LocalhostServerConfiguration,
            mappings=mappings,
            account_configuration=AccountConfiguration(f"RLFinal{ts}", "password"),
            max_concurrent_battles=args.eval_concurrency,
        )

        final_opponents = [
            (opponent_name(RandomPlayer), RandomPlayer(
                battle_format=BATTLE_FORMAT, team=opponent_teambuilder,
                server_configuration=LocalhostServerConfiguration,
                account_configuration=AccountConfiguration(f"FinalRand{ts}", "password"),
                max_concurrent_battles=args.eval_concurrency,
            )),
            (opponent_name(SimpleHeuristicsPlayer), SimpleHeuristicsPlayer(
                battle_format=BATTLE_FORMAT, team=opponent_teambuilder,
                server_configuration=LocalhostServerConfiguration,
                account_configuration=AccountConfiguration(f"FinalHeur{ts}", "password"),
                max_concurrent_battles=args.eval_concurrency,
            )),
            (opponent_name(Gen3StallerPlayer), Gen3StallerPlayer(
                battle_format=BATTLE_FORMAT, team=opponent_teambuilder,
                server_configuration=LocalhostServerConfiguration,
                account_configuration=AccountConfiguration(f"FinalStall{ts}", "password"),
                max_concurrent_battles=args.eval_concurrency,
            )),
            (opponent_name(Gen3AggressivePlayer), Gen3AggressivePlayer(
                battle_format=BATTLE_FORMAT, team=opponent_teambuilder,
                server_configuration=LocalhostServerConfiguration,
                account_configuration=AccountConfiguration(f"FinalAggr{ts}", "password"),
                max_concurrent_battles=args.eval_concurrency,
            )),
            (opponent_name(Gen3SetupSweepPlayer), Gen3SetupSweepPlayer(
                battle_format=BATTLE_FORMAT, team=opponent_teambuilder,
                server_configuration=LocalhostServerConfiguration,
                account_configuration=AccountConfiguration(f"FinalSetup{ts}", "password"),
                max_concurrent_battles=args.eval_concurrency,
            )),
        ]

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
    
    replay_callback = ReplayCallback(
        model_dir=model_dir,
        mappings=mappings,
        trainee_teambuilder=trainee_teambuilder,
        opponent_teambuilder=opponent_teambuilder,
        save_freq=100000,
        n_replays=10,
        stall_config=stall_cfg,
        reward_fn_factory=reward_factory,
    )

    adaptive_ppo_callback = AdaptivePPOCallback(
        initial_lr=args.lr,
        initial_epochs=args.n_epochs,
    )
    checkpoint_callback._current_lr_fn = lambda: model.policy.optimizer.param_groups[0]["lr"]
    checkpoint_callback._current_epochs_fn = lambda: adaptive_ppo_callback.current_epochs
    callbacks = [checkpoint_callback, replay_callback, adaptive_ppo_callback, MetricsExporterCallback(), _HparamLogCallback(args.ent_coef)]
    eval_callback = None

    if not args.debug:
        ts_cb = datetime.now().strftime('%H%M%S')
        eval_opponents = [
            (opponent_name(RandomPlayer), RandomPlayer(
                battle_format=BATTLE_FORMAT, team=opponent_teambuilder,
                server_configuration=LocalhostServerConfiguration,
                account_configuration=AccountConfiguration(f"CbRand{ts_cb}", "password"),
                max_concurrent_battles=100,
            )),
            (opponent_name(SimpleHeuristicsPlayer), SimpleHeuristicsPlayer(
                battle_format=BATTLE_FORMAT, team=opponent_teambuilder,
                server_configuration=LocalhostServerConfiguration,
                account_configuration=AccountConfiguration(f"CbHeur{ts_cb}", "password"),
                max_concurrent_battles=100,
            )),
            (opponent_name(Gen3StallerPlayer), Gen3StallerPlayer(
                battle_format=BATTLE_FORMAT, team=opponent_teambuilder,
                server_configuration=LocalhostServerConfiguration,
                account_configuration=AccountConfiguration(f"CbStall{ts_cb}", "password"),
                max_concurrent_battles=100,
            )),
            (opponent_name(Gen3AggressivePlayer), Gen3AggressivePlayer(
                battle_format=BATTLE_FORMAT, team=opponent_teambuilder,
                server_configuration=LocalhostServerConfiguration,
                account_configuration=AccountConfiguration(f"CbAggr{ts_cb}", "password"),
                max_concurrent_battles=100,
            )),
            (opponent_name(Gen3SetupSweepPlayer), Gen3SetupSweepPlayer(
                battle_format=BATTLE_FORMAT, team=opponent_teambuilder,
                server_configuration=LocalhostServerConfiguration,
                account_configuration=AccountConfiguration(f"CbSetup{ts_cb}", "password"),
                max_concurrent_battles=100,
            )),
        ]
        if args.self_play and _pool is not None:
            eval_callback = SelfPlayCallback(
                pool=_pool,
                bot_opponents=eval_opponents,
                trainee_teambuilder=trainee_teambuilder,
                opp_teambuilder=opponent_teambuilder,
                mappings=mappings,
                promote_threshold=args.promote_threshold,
                best_model_save_path=os.path.join(model_dir, "best_model"),
            )
        else:
            eval_callback = PerOpponentEvalCallback(
                opponents=eval_opponents,
                trainee_teambuilder=trainee_teambuilder,
                mappings=mappings,
                best_model_save_path=os.path.join(model_dir, "best_model"),
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
        model.gae_lambda = 0.85
        # Resume LR from optimizer state (stored in .zip by SB3), clamped to args.lr
        # so a manual --lr override can still lower the rate.
        saved_lr = model.policy.optimizer.param_groups[0]["lr"]
        resume_lr = min(saved_lr, args.lr)
        _resume_lr_lambda = lambda _: resume_lr
        model.lr_schedule = _resume_lr_lambda
        adaptive_ppo_callback._current_lr = resume_lr
        # Resume n_epochs (not in optimizer state). Priority:
        #   1. Per-checkpoint metadata ({checkpoint}.json) — most accurate for old checkpoints
        #   2. Run-level metadata.json — reflects last SIGTERM save for launcher restarts
        #   3. args.n_epochs — fallback for first run or missing state
        # Must be read before save_model_snapshot() below overwrites the run-level file.
        _ckpt_meta = read_checkpoint_metadata(model_path)
        if "current_epochs" in _ckpt_meta:
            resume_epochs = _ckpt_meta["current_epochs"]
        else:
            _meta_path = os.path.join(model_dir, "metadata.json")
            resume_epochs = args.n_epochs
            if os.path.exists(_meta_path):
                with open(_meta_path) as _f:
                    resume_epochs = json.load(_f).get("current_epochs", args.n_epochs)
        model.n_epochs = resume_epochs
        adaptive_ppo_callback._current_epochs = resume_epochs
        model.clip_range = lambda _: CLIP_RANGE
        send_event(f"▶️  Resuming at LR {resume_lr:.2e}, epochs {resume_epochs} (checkpoint LR={saved_lr:.2e})")

        if args.eval_only:
            await evaluate_model_random(model)
            return
        else:
            remaining_steps = args.steps - model.num_timesteps
            if remaining_steps <= 0:
                print(f"Training already complete ({model.num_timesteps:,} / {args.steps:,} steps)")
                sys.exit(TrainExitCode.COMPLETE)
            print(f"Continuing Training (Steps: {remaining_steps:,} remaining of {args.steps:,}, LR: {resume_lr:.2e} (saved={saved_lr:.2e}, arg={args.lr:.2e})")
            _run_roundtrip_test(model, _load_extractor_kwargs["layout"], _load_policy_kwargs, debug=args.debug)
            save_model_snapshot(model_dir, current_version)

            _abort_fn = _setup_signal_handlers(
                model, model_dir, _shutdown_event, current_version,
                lambda: model.policy.optimizer.param_groups[0]["lr"],
                lambda: adaptive_ppo_callback.current_epochs,
            )
            if not args.debug and eval_callback is not None:
                eval_callback.abort_fn = _abort_fn

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
            save_model_snapshot(os.path.dirname(final_path), current_version)
            print(f"Training complete. Model saved to {final_path}")
            best_model_dir = os.path.join(model_dir, "best_model")
            if os.path.isdir(best_model_dir):
                save_model_snapshot(best_model_dir, current_version)
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
            "net_arch": [512, 512]
        }
        
        # --- Model Initialization ---
        total_rollout_size = args.n_steps * n_envs
        if args.batch_size > total_rollout_size:
            print(f"Note: Capping batch_size from {args.batch_size} to {total_rollout_size} to match rollout capacity.")
            args.batch_size = total_rollout_size

        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.9999,
            gae_lambda=0.85,
            clip_range=CLIP_RANGE,
            ent_coef=args.ent_coef,
            device=args.device,
            seed=args.seed,
            tensorboard_log=tensorboard_dir,
            policy_kwargs=policy_kwargs
        )

        version = ModelVersion.from_layout_and_policy_kwargs(extractor_kwargs["layout"], policy_kwargs)
        _run_roundtrip_test(model, extractor_kwargs["layout"], policy_kwargs, debug=args.debug)
        save_model_snapshot(model_dir, version)

        _abort_fn = _setup_signal_handlers(
            model, model_dir, _shutdown_event, version,
            lambda: model.policy.optimizer.param_groups[0]["lr"],
            lambda: adaptive_ppo_callback.current_epochs,
        )
        if not args.debug and eval_callback is not None:
            eval_callback.abort_fn = _abort_fn

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
        record_checkpoint(model_dir, final_path + ".zip", adaptive_ppo_callback.current_lr, adaptive_ppo_callback.current_epochs)
        save_model_snapshot(os.path.dirname(final_path), version)
        print(f"Training complete. Model saved to {final_path}")
        best_model_dir = os.path.join(model_dir, "best_model")
        if os.path.isdir(best_model_dir):
            save_model_snapshot(best_model_dir, version)
        await evaluate_model_random(model)

if __name__ == "__main__":
    asyncio.run(main())
