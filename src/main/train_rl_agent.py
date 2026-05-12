import multiprocessing
import traceback
try:
    multiprocessing.set_start_method('spawn', force=True)
except RuntimeError:
    pass
import asyncio
import os
import sys
import json
import numpy as np
import argparse
from datetime import datetime
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Dict, Any

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
from agents.training.callbacks import ReplayCallback

from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.environment.singles_env import SinglesEnv

BATTLE_FORMAT = "gen3ou"

class Gen3Env(SinglesEnv):
    def __init__(self, mappings, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_encoder = get_observation_encoder(mappings)
        
        # Define spaces
        obs_dim = self.observation_encoder.dimension
        
        # Standard flat Box for SinglesEnv
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # Gen 3 has 11 actions: 6 switches (0-5), 4 moves (6-9), and Struggle (10)
        self.action_space = spaces.Discrete(11)
        
        # PokeEnv's SingleAgentWrapper expects the plural observation_spaces property
        self.observation_spaces = {
            self.agent1.username: self.observation_space,
            self.agent2.username: self.observation_space
        }
        
        # Subsidy tracking
        self.switch_count = 0

    def embed_battle(self, battle):
        # SNAPSHOT: Generate and freeze the mask for this specific observation
        # This is the ULTIMATE synchronization: the mask is pinned to the obs.
        if hasattr(self, "battle1") and battle is self.battle1:
            # Generate the mask once and FREEZE it
            self._last_obs_mask = Gen3ActionMasker.get_mask(battle)
            # We still keep the battle snapshot for action_to_order
            self._last_obs_battle = battle
            
        return self.observation_encoder.encode(battle)

    def get_action_mask(self, battle):
        # Use the frozen mask if available to ensure sync with observation
        if hasattr(self, "_last_obs_mask"):
            return self._last_obs_mask
        return Gen3ActionMasker.get_mask(battle)

    def action_masks(self):
        # Return the frozen mask from the last observation (fallback to current if needed)
        return self.get_action_mask(getattr(self, "battle1", None))

    def action_to_order(self, action, battle, **kwargs):
        # LOCKDOWN: Use the snapshot from the observation to avoid race conditions
        battle = getattr(self, "_last_obs_battle", battle)
        
        # FIXED SLOT MAPPING (STRICT MODE): No fallbacks allowed.
        from poke_env.player.battle_order import SingleBattleOrder
        
        mask = self.action_masks()
        team_list = list(battle.team.values())
        mon_names = [p.species for p in team_list]
        
        if action < 6:
            # 0-5 -> Team Slots 1-6
            if action < len(team_list):
                target_mon = team_list[action]
                if target_mon in battle.available_switches:
                    return SingleBattleOrder(target_mon)
            
            raise ValueError(
                f"ILLEGAL SWITCH: Action {action} (Slot {action+1}: {mon_names[action] if action < len(mon_names) else 'Empty'}).\n"
                f"Valid: {[p.species for p in battle.available_switches]}\n"
                f"Mask: {mask}"
            )
        elif action < 10:
            # 6-9 -> Move Slots 1-4 (Absolute Mapping)
            move_idx = action - 6
            active_pokemon = battle.active_pokemon
            if active_pokemon:
                mon_moves = list(active_pokemon.moves.values())[:4]
                if move_idx < len(mon_moves):
                    target_move = mon_moves[move_idx]
                    available_move_ids = [m.id for m in battle.available_moves]
                    
                    if target_move.id in available_move_ids:
                        return SingleBattleOrder(target_move)
            
            raise ValueError(
                f"ILLEGAL MOVE: Action {action} (Slot {move_idx+1}).\n"
                f"Valid: {[m.id for m in battle.available_moves]}\n"
                f"Mask: {mask}"
            )
        elif action == 10:
            # 10 -> Dedicated Struggle
            available_moves = battle.available_moves
            if len(available_moves) == 1 and available_moves[0].id == "struggle":
                return SingleBattleOrder(available_moves[0])
            
            raise ValueError(
                f"ILLEGAL STRUGGLE: Action 10 picked but struggle not available!\n"
                f"Valid: {[m.id for m in battle.available_moves]}\n"
                f"Mask: {mask}"
            )
        
        raise ValueError(f"Invalid Action Index: {action}")
        
    def calc_reward(self, battle):
        reward = self.reward_computing_helper(
            battle,
            fainted_value=2.0,
            hp_value=1.0,
            victory_value=30.0
        )
        
        # --- Switching Subsidy ---
        # Reward the first 5 switches of a battle to encourage exploration
        if hasattr(self, "_last_action"):
            action_val = self._last_action
            # Handle cases where action might be wrapped in a list or array
            if isinstance(action_val, (np.ndarray, list)) and len(action_val) > 0:
                action_val = action_val[0]
            
            if isinstance(action_val, (int, np.integer)) and action_val < 6:
                if self.switch_count < 15:
                    reward += 0.4
                    self.switch_count += 1
                
        return reward

    def step(self, action):
        try:
            self._last_action = action
            return super().step(action)
        except Exception as e:
            print(f"🛑 ERROR IN STEP: {e}")
            import traceback
            traceback.print_exc()
            raise e

    def reset(self, *args, **kwargs):
        try:
            self.switch_count = 0
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

    # --- Hyperparameter Flags (Optimized for GPU) ---
    parser.add_argument("--batch-size", type=int, default=4096, help="PPO mini-batch size")
    parser.add_argument("--n-epochs", type=int, default=4, help="PPO optimization epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--ent-coef", type=float, default=0.02, help="Entropy coefficient (exploration bonus)")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per environment per rollout")
    
    args = parser.parse_args()

    # Load all teams using the new TeamLoader
    loader = TeamLoader()
    sample_teams = loader.get_sample_teams()
    all_teams = loader.get_all_teams()
    
    print(f"Loaded {len(sample_teams)} sample teams for trainee and {len(all_teams)} total teams for opponents.")

    # Pre-pack teambuilders for performance and variety
    trainee_teambuilder = Gen3Teambuilder(sample_teams)
    opponent_teambuilder = Gen3Teambuilder(all_teams)

    mappings = load_mappings()
    
    def create_training_env_random(idx):
        def _init():
            try:
                ts = datetime.now().strftime('%H%M%S')
                env_username = f"RLAgent{idx}{ts}"
                opp_username = f"Opponent{idx}{ts}"
                
                env = Gen3Env(
                    mappings,
                    battle_format=BATTLE_FORMAT,
                    team=trainee_teambuilder,
                    log_level=40,
                    server_configuration=LocalhostServerConfiguration,
                    account_configuration1=AccountConfiguration(env_username, "password"),
                )
                opponent = SimpleHeuristicsPlayer(
                    battle_format=BATTLE_FORMAT,
                    team=opponent_teambuilder,
                    server_configuration=LocalhostServerConfiguration,
                    account_configuration=AccountConfiguration(opp_username, "password"),
                )
                from sb3_contrib.common.wrappers import ActionMasker
                wrapped = SingleAgentWrapper(env, opponent)
                
                # FORCE OVERRIDE: SingleAgentWrapper hardcodes 10 for gen3ou. We need 11.
                from gymnasium import spaces
                wrapped.action_space = spaces.Discrete(11)
                wrapped.observation_space = spaces.Dict({
                    "action_mask": spaces.Box(0, 1, shape=(11,), dtype=np.int8),
                    "observation": env.observation_space
                })
                
                from sb3_contrib.common.wrappers import ActionMasker
                return Monitor(ActionMasker(wrapped, lambda e: env.action_masks()))
            except Exception as e:
                print(f"🛑 ERROR IN WORKER {idx}: {e}")
                traceback.print_exc()
                raise e
        return _init

    # Running parallel environments
    n_envs = 1 if args.debug else args.n_envs
    EnvClass = DummyVecEnv if args.debug else SubprocVecEnv
    
    print(f"Initializing {n_envs} environments via {EnvClass.__name__}...")
    
    env_factories = [create_training_env_random(i) for i in range(n_envs)]
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
        model = PPO.load(model_path, env=env, device=args.device, tensorboard_log="./tensorboard/")
        model.ent_coef = args.ent_coef # Allow overriding entropy during continuation
        
        if args.eval_only:
            await evaluate_model_random(model)
            return
        else:
            print(f"Continuing Training (Steps: {args.steps}, LR: {args.lr})")
            unique_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_dir = f"models/gen3ou_ppo_continued_{unique_id}"
            os.makedirs(model_dir, exist_ok=True)
            
            import signal
            def signal_handler(sig, frame):
                print("\nInterrupt received, saving model...")
                final_path = os.path.join(model_dir, "final_model_interrupted")
                model.save(final_path)
                print(f"Model saved to {final_path}. Exiting.")
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)

            checkpoint_callback = CheckpointCallback(
                save_freq=50000, 
                save_path=model_dir,
                name_prefix="checkpoint"
            )
            
            try:
                model.learn(total_timesteps=args.steps, callback=checkpoint_callback, reset_num_timesteps=False)
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
        unique_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = f"models/gen3ou_ppo_new_{unique_id}"
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize a dummy encoder to get the handoff kwargs
        temp_encoder = Gen3ObservationEncoder(mappings)
        policy_kwargs = {
            "features_extractor_class": Gen3FeaturesExtractor,
            "features_extractor_kwargs": temp_encoder.get_features_extractor_kwargs(),
            "net_arch": [512, 512]
        }
        
        # --- Model Initialization ---
        total_rollout_size = args.n_steps * n_envs
        if args.batch_size > total_rollout_size:
            print(f"Note: Capping batch_size from {args.batch_size} to {total_rollout_size} to match rollout capacity.")
            args.batch_size = total_rollout_size

        from sb3_contrib import MaskablePPO
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
            save_freq=100000, # Start progressive scaling at 100k
            n_replays=3
        )
        
        callbacks = [checkpoint_callback, replay_callback]
        
        if not args.debug:
            from stable_baselines3.common.callbacks import EvalCallback
            
            # Evaluation environments: use 8 as requested
            def create_eval_env(idx):
                def _init():
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
                    from sb3_contrib.common.wrappers import ActionMasker
                    wrapped = SingleAgentWrapper(env, opponent)
                    
                    # FORCE OVERRIDE: Support 11 actions
                    from gymnasium import spaces
                    wrapped.action_space = spaces.Discrete(11)
                    wrapped.observation_space = spaces.Dict({
                        "action_mask": spaces.Box(0, 1, shape=(11,), dtype=np.int8),
                        "observation": env.observation_space
                    })
                    
                    from sb3_contrib.common.wrappers import ActionMasker
                    return ActionMasker(wrapped, lambda e: env.action_masks())
                return _init
            
            eval_env = SubprocVecEnv([create_eval_env(i) for i in range(8)])
            
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=os.path.join(model_dir, "best_model"),
                log_path=os.path.join(model_dir, "eval_logs"),
                eval_freq=max(1000, 500000 // args.n_envs), # Eval every 500k steps
                deterministic=False,
                n_eval_episodes=args.eval_battles
            )
            callbacks.append(eval_callback)

        try:
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
