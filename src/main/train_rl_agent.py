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

import asyncio
import numpy as np
import argparse
from datetime import datetime
from gymnasium import spaces
from stable_baselines3 import PPO
from sb3_contrib import MaskablePPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Dict, Any, Optional

from agents.model.features_extractor import Gen3FeaturesExtractor
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings, get_observation_encoder
from agents.action.mask_generator import Gen3ActionMasker
from agents.rl_agent import RLPlayer, SingleAgentWrapper
from agents.observation.species import SpeciesEncoder
from agents.observation.moves import MovesEncoder
from agents.observation.items import ItemsEncoder
from agents.observation.abilities import AbilitiesEncoder
from agents.observation.reactive import ReactiveEncoder
from agents.observation.global_env import GlobalEnvEncoder
from utils.teambuilder import Gen3Teambuilder
from utils.team_loader import TeamLoader
from agents.training.replay_recorder import ReplayCallback
from agents.training.reward_manager import Gen3RewardManager
from agents.training.reward_function import RewardFunction
from agents.training.battle_context import BattleContext, TurnDelta
from agents.training.slot_registry import SlotRegistry
from agents.training.wrappers import MaskableAgentWrapper
from agents.action.mapper import Gen3ActionMapper
from utils.logging.levels import LogLevel

from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.environment.singles_env import SinglesEnv

BATTLE_FORMAT = "gen3ou"

STALL_THRESHOLD = 250

class Gen3Env(SinglesEnv):
    def __init__(self, mappings, reward_fn: Optional[RewardFunction] = None, log_level=LogLevel.QUIET, stalls_dir=None, *args, **kwargs):
        self.log_level = log_level
        self.stalls_dir = stalls_dir
        self._stall_logged = False
        super().__init__(*args, **kwargs)
        self.observation_encoder = get_observation_encoder(mappings)

        obs_dim = self.observation_encoder.dimension
        self.vector_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Dict({
            "observation": self.vector_space,
            "action_mask": spaces.Box(0, 1, shape=(11,), dtype=np.int8)
        })
        self.observation_spaces = {
            self.agent1.username: self.observation_space,
            self.agent2.username: self.observation_space
        }

        # Trainee reward function — injected or defaulted. Opponent battles fall
        # back to poke-env's reward_computing_helper and never touch this.
        self.reward_manager: RewardFunction = reward_fn or Gen3RewardManager(log_level=self.log_level)

        # Per-episode stable slot assignment for our team and opponent team
        self._our_slots = SlotRegistry()
        self._opp_slots = SlotRegistry()
        self._last_ctx: Optional[BattleContext] = None
        self._prev_ctx: Optional[BattleContext] = None
        self._last_action: int = -1

    def embed_battle(self, battle):
        obs = self.observation_encoder.encode(battle)

        # Only build a new context for the trainee while the battle is live.
        # Skip on terminal observations (battle.finished) and transitional wait
        # states where poke-env emits an embed_battle call before the server has
        # sent a valid request — both cases have no valid actions so we keep the
        # last context intact (MaskablePPO uses it for value estimation only).
        if battle is self.battle1 and not battle.finished:
            mask = Gen3ActionMasker.get_mask(battle).astype(np.int8)

            if mask.sum() > 0:
                for mon in battle.team.values():
                    self._our_slots.assign(mon.species)
                for mon in battle.opponent_team.values():
                    self._opp_slots.assign(mon.species)

                our_hp = np.zeros(6, dtype=np.float32)
                for mon in battle.team.values():
                    slot = self._our_slots.get(mon.species)
                    if slot is not None:
                        our_hp[slot] = mon.current_hp_fraction

                opp_hp = np.zeros(6, dtype=np.float32)
                for mon in battle.opponent_team.values():
                    slot = self._opp_slots.get(mon.species)
                    if slot is not None:
                        opp_hp[slot] = mon.current_hp_fraction

                our_active = (
                    battle.active_pokemon.species
                    if battle.active_pokemon and not battle.active_pokemon.fainted
                    else "NONE"
                )
                opp_active = (
                    battle.opponent_active_pokemon.species
                    if battle.opponent_active_pokemon
                    else "NONE"
                )
                our_fainted_count = sum(1 for m in battle.team.values() if m.fainted)
                opp_fainted_count = sum(1 for m in battle.opponent_team.values() if m.fainted)

                phase = "forced_switch" if battle.force_switch else "move_selection"
                self._last_ctx = BattleContext(
                    turn=battle.turn,
                    phase=phase,
                    mask=mask,
                    obs=obs,
                    our_slot_map=self._our_slots.snapshot(),
                    opp_slot_map=self._opp_slots.snapshot(),
                    our_hp=our_hp,
                    opp_hp=opp_hp,
                    our_active=our_active,
                    opp_active=opp_active,
                    our_fainted_count=our_fainted_count,
                    opp_fainted_count=opp_fainted_count,
                )

        return obs

    def action_masks(self) -> np.ndarray:
        if self._last_ctx is not None:
            return self._last_ctx.mask
        return np.ones(11, dtype=np.int8)


    def _save_stall_html(self, battle, suffix=""):
        """Saves a Pokémon Showdown compatible HTML replay using poke-env's built-in saver."""
        if not self.stalls_dir:
            return
            
        try:
            os.makedirs(self.stalls_dir, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            suffix_str = f"_{suffix}" if suffix else ""
            filename = f"stall_{battle.battle_tag}_{ts}{suffix_str}.html"
            path = os.path.join(self.stalls_dir, filename)
            
            # Use the official poke-env saver
            battle.save_replay(path)
            
            # Print a distinct alert so the user knows where to look
            sys.stderr.write(f"\n🛑 [STALL LOGGED] Battle {battle.battle_tag} lasted {battle.turn} turns. HTML saved to {path}\n")
            sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"Failed to save stall log: {e}\n")
            sys.stderr.flush()

    def get_action_mask(self, battle):
        if battle is self.battle1 and self._last_ctx is not None:
            return self._last_ctx.mask
        return Gen3ActionMasker.get_mask(battle).astype(np.int8)

    def action_to_order(self, action, battle, **kwargs):
        """Delegates to Gen3ActionMapper with strict turn-locked validation."""
        from poke_env.player.battle_order import BattleOrder, ForfeitBattleOrder
        if isinstance(action, BattleOrder):
            return action

        if battle is self.battle1:
            if battle.turn >= STALL_THRESHOLD:
                if not self._stall_logged:
                    self._save_stall_html(battle, suffix="STALL")
                    self._stall_logged = True
                return ForfeitBattleOrder()

            ctx = self._last_ctx
            return Gen3ActionMapper.action_to_order(
                action=action,
                battle=battle,
                mask=ctx.mask if ctx is not None else None,
                latched_turn=ctx.turn if ctx is not None else -1,
            )

        # OPPONENT (battle2/others): Unmodified default poke-env logic
        return super().action_to_order(action, battle)
        
    def calc_reward(self, battle):
        if battle is self.battle1:
            if self._prev_ctx is not None and self._last_ctx is not None:
                delta = TurnDelta.build(self._prev_ctx, self._last_ctx, self._last_action)
            else:
                delta = TurnDelta.empty()
            return self.reward_manager.process_turn_reward(battle, delta)
        return self.reward_computing_helper(
            battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def step(self, action):
        try:
            # Pre-capture the battle object
            battle = getattr(self, "_battle", None)
            if battle is None:
                battle = self.battle1 

            # Record the action ONE-SHOT before processing
            if isinstance(action, dict):
                trainee_idx = action.get(self.agent1.username, -1)
            else:
                trainee_idx = action
            
            if battle is self.battle1 and self._last_ctx is not None:
                self._prev_ctx = self._last_ctx
                self._last_action = trainee_idx
                self.reward_manager.record_action(self._last_ctx, trainee_idx)

            obs, reward, term, trunc, info = super().step(action)

            # Pass switch logs up to the main process
            if hasattr(self, "_pending_switch_log"):
                info["switch_log"] = self._pending_switch_log
                del self._pending_switch_log
                
            return obs, reward, term, trunc, info
        except Exception as e:
            print(f"🛑 ERROR IN STEP: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def reset(self, *args, **kwargs):
        # Report previous episode metrics via the manager
        self.reward_manager.report_episode(getattr(self, "battle1", None))

        # Clear move slot cache and reset trackers on reset
        self._move_slot_cache = {}
        self._last_active_name = ""
        self._our_slots.reset()
        self._opp_slots.reset()
        self._last_ctx = None
        self._prev_ctx = None
        try:
            # Disable replay saving for the new battle unless it stalls later
            if hasattr(self, "agent1"):
                self.agent1.save_replays = None

            self.reward_manager.reset()
            self._stall_logged = False
            self._last_action = -1
            return super().reset(*args, **kwargs)
        except Exception as e:
            print(f"🛑 ERROR IN RESET: {e}")
            import traceback
            traceback.print_exc()
            raise e

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
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--ent-coef", type=float, default=0.02, help="Entropy coefficient (exploration bonus)")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per environment per rollout")
    
    args = parser.parse_args()
    log_level = LogLevel[args.log_level.upper()]
    
    # Automatically enable deep traces if --debug is set
    if args.debug:
        log_level = LogLevel.DEBUG

    # Load all teams using the new TeamLoader
    loader = TeamLoader()
    sample_teams = loader.get_sample_teams()
    all_teams = loader.get_all_teams()
    
    print(f"Loaded {len(sample_teams)} sample teams for trainee and {len(all_teams)} total teams for opponents.")

    # Pre-pack teambuilders for performance and variety
    trainee_teambuilder = Gen3Teambuilder(sample_teams)
    opponent_teambuilder = Gen3Teambuilder(all_teams)

    mappings = load_mappings()
    
    def create_training_env_random(idx, stalls_dir=None):
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
                    stalls_dir=stalls_dir,
                    server_configuration=LocalhostServerConfiguration,
                    account_configuration1=AccountConfiguration(env_username, "password"),
                )
                opponent = SimpleHeuristicsPlayer(
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
    unique_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.model:
        model_dir = f"models/gen3ou_ppo_continued_{unique_id}"
    else:
        model_dir = f"models/gen3ou_ppo_new_{unique_id}"
    
    os.makedirs(model_dir, exist_ok=True)
    with open(os.path.join(model_dir, "command.txt"), "w") as f:
        f.write(" ".join(sys.argv))
        
    stalls_dir = os.path.join(model_dir, "stalls")
    os.makedirs(stalls_dir, exist_ok=True)

    # Running parallel environments
    n_envs = 1 if args.debug else args.n_envs
    EnvClass = DummyVecEnv if args.debug else SubprocVecEnv
    
    print(f"Initializing {n_envs} environments via {EnvClass.__name__}...")
    
    env_factories = [create_training_env_random(i, stalls_dir=stalls_dir) for i in range(n_envs)]
    env = EnvClass(env_factories)
    # Note: env.seed() is deprecated in gymnasium VecEnv, use seed in reset or at init if supported.
    # But for reproducibility, we pass it to PPO.

    async def evaluate_model_random(model):
        ts = datetime.now().strftime('%H%M%S')
        print(f"\nStarting Evaluation (Session {ts}, Battles: {args.eval_battles}, Concurrency: {args.eval_concurrency})...")
        
        rl_player = RLPlayer(
            model=model,
            team=trainee_teambuilder,
            battle_format=BATTLE_FORMAT,
            server_configuration=LocalhostServerConfiguration,
            mappings=mappings,
            account_configuration=AccountConfiguration(f"RLEval{ts}", "password"),
            max_concurrent_battles=args.eval_concurrency
        )
        
        random_player = RandomPlayer(
            battle_format=BATTLE_FORMAT,
            team=opponent_teambuilder,
            server_configuration=LocalhostServerConfiguration,
            account_configuration=AccountConfiguration(f"RandEval{ts}", "password"),
            max_concurrent_battles=args.eval_concurrency
        )
        
        heuristic_player = SimpleHeuristicsPlayer(
            battle_format=BATTLE_FORMAT,
            team=opponent_teambuilder,
            server_configuration=LocalhostServerConfiguration,
            account_configuration=AccountConfiguration(f"HeurEval{ts}", "password"),
            max_concurrent_battles=args.eval_concurrency
        )

        print(f"Evaluating against RandomPlayer [{args.eval_battles} battles]...")
        start_time = datetime.now()
        await rl_player.battle_against(random_player, n_battles=args.eval_battles)
        duration = datetime.now() - start_time
        print(f"Win rate vs Random: {rl_player.n_won_battles / args.eval_battles * 100:.1f}% (Time: {duration})")
        
        rl_player.reset_battles()
        print(f"Evaluating against HeuristicPlayer [{args.eval_battles} battles]...")
        start_time = datetime.now()
        await rl_player.battle_against(heuristic_player, n_battles=args.eval_battles)
        duration = datetime.now() - start_time
        print(f"Win rate vs Heuristic: {rl_player.n_won_battles / args.eval_battles * 100:.1f}% (Time: {duration})")

    # --- Callback Setup (Shared) ---
    checkpoint_callback = CheckpointCallback(
        save_freq=50000, 
        save_path=model_dir,
        name_prefix="checkpoint"
    )
    
    replay_callback = ReplayCallback(
        model_dir=model_dir,
        mappings=mappings,
        trainee_teambuilder=trainee_teambuilder,
        opponent_teambuilder=opponent_teambuilder,
        save_freq=100000,
        n_replays=10
    )
    
    callbacks = [checkpoint_callback, replay_callback]
    
    if not args.debug:
        from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
        
        def create_eval_env(idx):
            def _init():
                try:
                    ts = datetime.now().strftime('%H%M%S')
                    env = Gen3Env(
                        mappings,
                        battle_format=BATTLE_FORMAT,
                        team=trainee_teambuilder,
                        server_configuration=LocalhostServerConfiguration,
                        account_configuration1=AccountConfiguration(f"RLEval{idx}{ts}", "password"),
                    )
                    opponent = SimpleHeuristicsPlayer(
                        battle_format=BATTLE_FORMAT,
                        team=opponent_teambuilder,
                        server_configuration=LocalhostServerConfiguration,
                        account_configuration=AccountConfiguration(f"OppEval{idx}{ts}", "password"),
                    )
                    wrapped = MaskableAgentWrapper(env, opponent)
                    wrapped.action_space = env.action_space
                    wrapped.observation_space = env.observation_space
                    return Monitor(wrapped)
                except Exception as e:
                    print(f"🛑 ERROR IN EVAL WORKER {idx}: {e}")
                    traceback.print_exc()
                    raise e
            return _init
        
        eval_env = SubprocVecEnv([create_eval_env(i) for i in range(8)])
        eval_callback = MaskableEvalCallback(
            eval_env,
            best_model_save_path=os.path.join(model_dir, "best_model"),
            log_path=os.path.join(model_dir, "eval_logs"),
            eval_freq=max(1000, 500000 // args.n_envs),
            deterministic=False,
            n_eval_episodes=args.eval_battles
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

        print(f"Loading existing model from {model_path}")
        model = MaskablePPO.load(model_path, env=env, device=args.device, tensorboard_log="./tensorboard/")
        model.ent_coef = args.ent_coef # Allow overriding entropy during continuation
        
        if args.eval_only:
            await evaluate_model_random(model)
            return
        else:
            print(f"Continuing Training (Steps: {args.steps}, LR: {args.lr})")
            # model_dir and unique_id are now pre-defined earlier in main()
            
            import signal
            def signal_handler(sig, frame):
                print("\nInterrupt received, saving model...")
                final_path = os.path.join(model_dir, "final_model_interrupted")
                model.save(final_path)
                print(f"Model saved to {final_path}. Exiting.")
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            try:
                model.learn(total_timesteps=args.steps, callback=callbacks, reset_num_timesteps=False)
            except Exception as e:
                print(f"Training interrupted by exception: {e}")
                final_path = os.path.join(model_dir, "final_model_exception")
                model.save(final_path)
                
            final_path = os.path.join(model_dir, "final_model")
            model.save(final_path)
            print(f"Training complete. Model saved to {final_path}")
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
            gamma=0.99,
            ent_coef=args.ent_coef,
            device=args.device,
            seed=args.seed,
            tensorboard_log="./tensorboard/",
            policy_kwargs=policy_kwargs
        )

        import signal
        def signal_handler(sig, frame):
            print("\nInterrupt received, saving model...")
            final_path = os.path.join(model_dir, "final_model_interrupted")
            model.save(final_path)
            print(f"Model saved to {final_path}. Exiting.")
            sys.exit(0)
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        pass # Callbacks are now defined above

        try:
            if log_level >= LogLevel.DETAILED:
                from sb3_contrib.common.maskable.utils import is_masking_supported
                print(f"✅ [DEBUG] Masking supported for env: {is_masking_supported(env)}")
            model.learn(total_timesteps=args.steps, callback=callbacks, reset_num_timesteps=False)
        except Exception as e:
            print("\n" + "🛑" * 30)
            print(f"🛑 TRAINING CRASHED: {e}")
            print("🛑" * 30)
            traceback.print_exc()
            os._exit(1) # Stop immediately, do not proceed to evaluation
            final_path = os.path.join(model_dir, "final_model_exception")
            model.save(final_path)
            
        final_path = os.path.join(model_dir, "final_model")
        model.save(final_path)
        print(f"Training complete. Model saved to {final_path}")
        await evaluate_model_random(model)

if __name__ == "__main__":
    asyncio.run(main())
