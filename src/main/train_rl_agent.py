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

from agents.observation.state_encoder import Gen3ObservationEncoder
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

from poke_env.player import RandomPlayer, SimpleHeuristicsPlayer
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.environment.singles_env import SinglesEnv

BATTLE_FORMAT = "gen3ou"

import torch

class Gen3FeaturesExtractor(torch.nn.Module):
    def __init__(self, observation_space: spaces.Dict, layout: Dict[str, Any] = None, mappings: Dict[str, Any] = None):
        super().__init__()
        self.layout = layout
        self.mappings = mappings
        self._encoder = None # Lazy init for decoding
        
        # Total observation dimension is 1684
        self.species_embedding = torch.nn.Embedding(387, 32)
        
        # Projection layer
        # New dimension: 12 * (32 + 132) + 88 = 2056
        self.projection = torch.nn.Linear(2056, 256)
        self.activation = torch.nn.ReLU()
        self.features_dim = 256
        self.last_trace_time = 0
        
    def _print_deep_trace(self, x, pokemon_part, species_ids):
        import time
        if self._encoder is None and self.mappings:
            self._encoder = Gen3ObservationEncoder(self.mappings)
            
        print("\n" + "🧬" * 30)
        print(f"🧬 [DEEP TRACE - {time.strftime('%H:%M:%S')}]")
        print("=" * 60)
        
        if self._encoder:
            # Use the encoder's master description logic
            desc = self._encoder.describe_vector(x[0].cpu().numpy())
            world = desc.get('world', {})
            print(f"Turn: {world.get('turn', '???')} | Weather: {world.get('weather', 'NONE')} | Spikes: {world.get('our_spikes', 0)} (Us) / {world.get('opp_spikes', 0)} (Them)")
            
            print("\n--- OUR ACTIVE CONTEXT ---")
            ctx = desc.get('our_active', {})
            print(f"Boosts: {ctx.get('boosts', {})} | Volatiles: {ctx.get('volatiles', [])}")
            
            print("\n--- TEAM SUMMARIES ---")
            for i, mon in enumerate(desc['our_team']):
                active_str = " [Actv]" if mon.get('active') else "       "
                s = mon['stats']
                stats_str = f"{s['hp']}/{s['atk']}/{s['def']}/{s['spa']}/{s['spd']}/{s['spe']}"
                print(f"[OUR {i}] {mon['species']:12} | HP: {mon['hp']:6} | Status: {mon['status']:5}{active_str} | {stats_str}")
                print(f"  Moves: {mon.get('moves', [])}")
                
            print("-" * 30)
            for i, mon in enumerate(desc['opp_team']):
                active_str = " [Actv]" if mon.get('active') else "       "
                s = mon['stats']
                stats_str = f"{s['hp']}/{s['atk']}/{s['def']}/{s['spa']}/{s['spd']}/{s['spe']}"
                print(f"[OPP {i}] {mon['species']:12} | HP: {mon['hp']:6} | Status: {mon['status']:5}{active_str} | {stats_str}")
                print(f"  Moves: {mon.get('moves', [])}")
            
            momentum = desc.get('momentum', {})
            print(f"\n--- MOMENTUM ---")
            print(f"Fainted: {momentum.get('fainted_our', 0)} (Us) / {momentum.get('fainted_opp', 0)} (Them) | Matchups: {momentum.get('move_mults', [])}")
            
            # --- INTEGRITY CHECK ---
            warnings, is_critical = self._encoder.integrity_check(x[0].cpu().numpy())
            if warnings:
                print("\n⚠️ [INTEGRITY CHECK WARNINGS]")
                for w in warnings:
                    print(f"  - {w}")
                    
            if is_critical:
                raise ValueError(f"CRITICAL INTEGRITY FAILURE: {warnings}")
        else:
            print("Trace available but encoder/mappings missing.")
            
        print("=" * 60 + "\n")

    def forward(self, obs):
        x = obs["observation"]
        batch_size = x.shape[0]
        
        from agents.observation.constants import (
            OFFSET_OUR_TEAM, OFFSET_OPP_TEAM, OFFSET_CONTEXT, 
            POKEMON_FULL_DIM, POKEMON_VECTOR_DIM
        )
        
        # Extract pokemon parts using constants/layout
        # We handle 12 pokemon (6 our, 6 opp)
        our_team = x[:, OFFSET_OUR_TEAM : OFFSET_OPP_TEAM].reshape(batch_size, 6, POKEMON_FULL_DIM)
        opp_team = x[:, OFFSET_OPP_TEAM : OFFSET_CONTEXT].reshape(batch_size, 6, POKEMON_FULL_DIM)
        
        pokemon_part = torch.cat([our_team, opp_team], dim=1) # [B, 12, 133]
        remaining_part = x[:, OFFSET_CONTEXT:] # [B, 88] (62 context + 11 global + 15 reactive)
        
        # Extract IDs (first dim of each block)
        species_ids = pokemon_part[:, :, 0].long() # [B, 12]
        
        import time
        current_time = time.time()
        if current_time - self.last_trace_time > 15:
            self.last_trace_time = current_time
            self._print_deep_trace(x, pokemon_part, species_ids)
        
        # Embed species
        embedded_species = self.species_embedding(species_ids) # [B, 12, 32]
        
        # Keep everything else (stats, moves, items, etc.)
        # pokemon_part is [B, 12, 133]. Index 0 is ID. 1-132 is the rest.
        rest_of_pokemon = pokemon_part[:, :, 1:132] # [B, 12, 131] -- Wait, index 131 is HP. 1:132 is 131 dims.
        # Actually, let's just take everything except index 0.
        rest_of_pokemon = pokemon_part[:, :, 1:] # [B, 12, 132]
        
        # Combine
        pokemon_enriched = torch.cat([embedded_species, rest_of_pokemon], dim=2) # [B, 12, 164]
        pokemon_flat = pokemon_enriched.reshape(batch_size, -1) # [B, 1968]
        
        # Final combined vector (1968 + 88 = 2056)
        combined = torch.cat([pokemon_flat, remaining_part], dim=1) # [B, 2056]
        
        # Project to features_dim with activation
        return self.activation(self.projection(combined))

class MaskedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            features_extractor_class=Gen3FeaturesExtractor,
        )

    def extract_features(self, obs, features_extractor=None):
        self._mask = obs["action_mask"]
        return super().extract_features(obs, features_extractor)

    def forward(self, obs, deterministic=False):
        return super().forward(obs, deterministic)

    def get_distribution(self, obs):
        self._mask = obs["action_mask"]
        return super().get_distribution(obs)

    def evaluate_actions(self, obs, actions):
        self._mask = obs["action_mask"]
        return super().evaluate_actions(obs, actions)

    def _get_action_dist_from_latent(self, latent_pi):
        action_logits = self.action_net(latent_pi)
        # self._mask is (batch, 10). 1 for valid, 0 for invalid.
        mask = torch.where(self._mask == 1, 0.0, float("-inf"))
            
        return self.action_dist.proba_distribution(action_logits + mask)

def load_mappings():
    mappings = {}
    mapping_files = {
        "species": "data/pokemon/gen3_species.json",
        "moves": "data/pokemon/gen3_moves.json",
        "abilities": "data/pokemon/gen3_abilities.json",
        "items": "data/pokemon/gen3_items.json"
    }
    for key, path in mapping_files.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"CRITICAL: Mapping file missing: {path}. Run data generation script first!")
        
        with open(path, "r") as f:
            data = json.load(f)
            if not data:
                raise ValueError(f"CRITICAL: Mapping file is empty: {path}")
            mappings[key] = data
            
    return mappings

def get_observation_encoder(mappings):
    return Gen3ObservationEncoder(mappings)


class Gen3Env(SinglesEnv):
    def __init__(self, mappings, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_encoder = get_observation_encoder(mappings)
        
        # Define spaces
        obs_dim = self.observation_encoder.dimension
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        # Gen 3 has 10 actions: 6 switches (0-5) and 4 moves (6-9)
        self.action_space = spaces.Discrete(10)
        
        # PokeEnv will automatically wrap observation_spaces in a Dict(observation, action_mask)
        # because of its __setattr__ override. We just need to provide the raw space here.
        self.observation_spaces = {
            self.agent1.username: self.observation_space,
            self.agent2.username: self.observation_space
        }

    def embed_battle(self, battle):
        return self.observation_encoder.encode(battle)

    def get_action_mask(self, battle):
        return Gen3ActionMasker.get_mask(battle)
        
    def calc_reward(self, battle):
        return self.reward_computing_helper(
            battle,
            fainted_value=2.0,
            hp_value=1.0,
            victory_value=30.0
        )

async def main():
    parser = argparse.ArgumentParser(description="Train or Evaluate Gen 3 OU RL Agent")
    
    # --- Operational Flags ---
    parser.add_argument("--model", type=str, help="Path to existing model to load")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and only evaluate")
    parser.add_argument("--steps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--debug", action="store_true", help="Use DummyVecEnv (1 env) for debugging")
    parser.add_argument("--n-envs", type=int, default=8, help="Number of parallel environments")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use (cpu or cuda)")
    parser.add_argument("--eval-battles", type=int, default=100, help="Battles per evaluation opponent")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # --- Hyperparameter Flags (Optimized for CPU) ---
    parser.add_argument("--batch-size", type=int, default=512, help="PPO mini-batch size")
    parser.add_argument("--n-epochs", type=int, default=4, help="PPO optimization epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
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
            return Monitor(SingleAgentWrapper(env, opponent))
        return _init

    # Running parallel environments
    n_envs = 1 if args.debug else args.n_envs
    EnvClass = DummyVecEnv if args.debug else SubprocVecEnv
    
    print(f"Initializing {n_envs} environments via {EnvClass.__name__} (staggered startup)...")
    
    # Staggered initialization to avoid "Connection Reset" during massive login storm
    env_factories = [create_training_env_random(i) for i in range(n_envs)]
    def create_staggered_env(idx):
        import time
        def _init():
            time.sleep(idx * 0.1) # 0.1s delay per environment
            return env_factories[idx]()
        return _init

    env = EnvClass([create_staggered_env(i) for i in range(n_envs)])
    # Note: env.seed() is deprecated in gymnasium VecEnv, use seed in reset or at init if supported.
    # But for reproducibility, we pass it to PPO.

    async def evaluate_model_random(model):
        ts = datetime.now().strftime('%H%M%S')
        print(f"\nStarting Evaluation (Session {ts}, Battles: {args.eval_battles})...")
        
        rl_player = RLPlayer(
            model=model,
            team=trainee_teambuilder,
            battle_format=BATTLE_FORMAT,
            server_configuration=LocalhostServerConfiguration,
            mappings=mappings,
            account_configuration=AccountConfiguration(f"RLEval{ts}", "password"),
            max_concurrent_battles=50
        )
        
        random_player = RandomPlayer(
            battle_format=BATTLE_FORMAT,
            team=opponent_teambuilder,
            server_configuration=LocalhostServerConfiguration,
            account_configuration=AccountConfiguration(f"RandEval{ts}", "password"),
            max_concurrent_battles=50
        )
        
        heuristic_player = SimpleHeuristicsPlayer(
            battle_format=BATTLE_FORMAT,
            team=opponent_teambuilder,
            server_configuration=LocalhostServerConfiguration,
            account_configuration=AccountConfiguration(f"HeurEval{ts}", "password"),
            max_concurrent_battles=50
        )

        print(f"Evaluating against RandomPlayer [{args.eval_battles} battles]...")
        await rl_player.battle_against(random_player, n_battles=args.eval_battles)
        print(f"Win rate vs Random: {rl_player.n_won_battles / args.eval_battles * 100:.1f}%")
        
        rl_player.reset_battles()
        print(f"Evaluating against HeuristicPlayer [{args.eval_battles} battles]...")
        await rl_player.battle_against(heuristic_player, n_battles=args.eval_battles)
        print(f"Win rate vs Heuristic: {rl_player.n_won_battles / args.eval_battles * 100:.1f}%")

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
        
        # Initialize a dummy encoder to get the layout
        temp_encoder = Gen3ObservationEncoder(mappings)
        policy_kwargs = {
            "features_extractor_kwargs": {
                "layout": temp_encoder.get_layout(),
                "mappings": mappings
            },
            "net_arch": [512, 512]
        }
        
        model = PPO(
            MaskedActorCriticPolicy,
            env,
            verbose=1,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=0.99,
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

        from stable_baselines3.common.callbacks import EvalCallback
        
        # Define evaluation environment
        def create_eval_env():
            import time
            # Even more staggered for eval
            time.sleep(2.0) 
            ts = datetime.now().strftime('%H%M%S')
            env = Gen3Env(
                mappings,
                battle_format=BATTLE_FORMAT,
                team=trainee_teambuilder,
                server_configuration=LocalhostServerConfiguration,
                account_configuration1=AccountConfiguration(f"RLEvalEnv{ts}", "password"),
            )
            opponent = SimpleHeuristicsPlayer(
                battle_format=BATTLE_FORMAT,
                team=opponent_teambuilder,
                server_configuration=LocalhostServerConfiguration,
                account_configuration=AccountConfiguration(f"OppEvalEnv{ts}", "password"),
            )
            return SingleAgentWrapper(env, opponent)
        
        eval_env = DummyVecEnv([create_eval_env])
        
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=os.path.join(model_dir, "best_model"),
            log_path=os.path.join(model_dir, "eval_logs"),
            eval_freq=50000, # Eval every 50k steps (since we have 8 envs, this is ~6.25k iterations)
            deterministic=True,
            render=False,
            n_eval_episodes=20
        )

        checkpoint_callback = CheckpointCallback(
            save_freq=50000, 
            save_path=model_dir,
            name_prefix="checkpoint"
        )
        
        callbacks = [checkpoint_callback, eval_callback]

        try:
            model.learn(total_timesteps=args.steps, callback=callbacks, reset_num_timesteps=False)
        except Exception as e:
            print(f"Training interrupted by exception: {e}")
            traceback.print_exc()
            final_path = os.path.join(model_dir, "final_model_exception")
            model.save(final_path)
            
        final_path = os.path.join(model_dir, "final_model")
        model.save(final_path)
        print(f"Training complete. Model saved to {final_path}")
        await evaluate_model_random(model)

if __name__ == "__main__":
    asyncio.run(main())
