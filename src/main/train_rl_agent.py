import asyncio
import os
import json
import numpy as np
import torch
import argparse
import time
from datetime import datetime
from typing import Dict, Optional, List, Any

from poke_env.player import (
    Player,
    RandomPlayer,
    SimpleHeuristicsPlayer,
)
from poke_env.battle.abstract_battle import AbstractBattle
from poke_env.player.battle_order import BattleOrder, DefaultBattleOrder
from poke_env.battle.pokemon import Pokemon
from poke_env.battle.move import Move
from poke_env.data import GenData
from poke_env.ps_client import LocalhostServerConfiguration, AccountConfiguration

# RL Imports
import gymnasium as gym
from gymnasium.spaces import Box, Discrete
from poke_env.environment.singles_env import SinglesEnv
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

from utils.teambuilder import Gen3Teambuilder

# --- Configuration ---
BATTLE_FORMAT = "gen3ou"
N_FEATURES = 15 
DEFAULT_TEAM_NAME = "Big 5 + Starmie (Beerlover)"

class Gen3FeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(observation_space, features_dim=N_FEATURES)

    def forward(self, obs):
        return obs["observation"]

class MaskedActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(
            *args,
            **kwargs,
            net_arch=[128, 128],
            features_extractor_class=Gen3FeaturesExtractor,
        )

    def forward(self, obs, deterministic=False):
        self._mask = obs["action_mask"]
        return super().forward(obs, deterministic)

    def evaluate_actions(self, obs, actions):
        self._mask = obs["action_mask"]
        return super().evaluate_actions(obs, actions)

    def _get_action_dist_from_latent(self, latent_pi):
        action_logits = self.action_net(latent_pi)
        mask = torch.where(self._mask == 1, 0, float("-inf"))
        return self.action_dist.proba_distribution(action_logits + mask)

class Gen3Env(SinglesEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.observation_spaces = {
            agent: self.describe_embedding()
            for agent in self.possible_agents
        }

    def embed_battle(self, battle: AbstractBattle):
        moves_base_power = np.zeros(4)
        moves_dmg_multiplier = np.ones(4)
        for i, move in enumerate(battle.available_moves):
            moves_base_power[i] = move.base_power / 100
            if battle.opponent_active_pokemon is not None:
                moves_dmg_multiplier[i] = move.type.damage_multiplier(
                    battle.opponent_active_pokemon.type_1,
                    battle.opponent_active_pokemon.type_2,
                    type_chart=GenData.from_gen(3).type_chart,
                )

        fainted_mon_team = len([mon for mon in battle.team.values() if mon.fainted]) / 6
        fainted_mon_opponent = (
            len([mon for mon in battle.opponent_team.values() if mon.fainted]) / 6
        )

        our_hp = battle.active_pokemon.current_hp_fraction if battle.active_pokemon else 0.0
        opp_hp = (
            battle.opponent_active_pokemon.current_hp_fraction 
            if battle.opponent_active_pokemon else 0.0
        )

        our_spikes = battle.side_conditions.get("spikes", 0) / 3
        opp_spikes = battle.opponent_side_conditions.get("spikes", 0) / 3

        return np.concatenate(
            [
                moves_base_power,
                moves_dmg_multiplier,
                [fainted_mon_team, fainted_mon_opponent],
                [our_hp, opp_hp],
                [our_spikes, opp_spikes],
                [1.0 if battle.active_pokemon and battle.active_pokemon.status else 0.0]
            ],
            dtype=np.float32,
        )

    def calc_reward(self, battle) -> float:
        return self.reward_computing_helper(
            battle,
            fainted_value=2.0,
            hp_value=1.0,
            status_value=0.5,
            victory_value=30.0,
        )

    def describe_embedding(self):
        return Box(-1, 4, shape=(N_FEATURES,), dtype=np.float32)

class RLPlayer(Player):
    def __init__(self, model, team, *args, **kwargs):
        super().__init__(team=team, *args, **kwargs)
        self.model = model

    def choose_move(self, battle: AbstractBattle) -> BattleOrder:
        if battle.wait:
            return DefaultBattleOrder()
        
        obs = Gen3Env.embed_battle(self, battle) 
        mask = np.array(SinglesEnv.get_action_mask(battle))
        
        # Manually apply masking for the prediction
        with torch.no_grad():
            obs_dict = {
                "observation": torch.as_tensor(obs, device=self.model.device).unsqueeze(0),
                "action_mask": torch.as_tensor(mask, device=self.model.device).unsqueeze(0),
            }
            
            # This follows our MaskedActorCriticPolicy logic
            features = self.model.policy.extract_features(obs_dict)
            latent_pi, _ = self.model.policy.mlp_extractor(features)
            action_logits = self.model.policy.action_net(latent_pi)
            
            m = torch.where(obs_dict["action_mask"] == 1, 0, float("-inf"))
            probs = torch.softmax(action_logits + m, dim=1)
            action = torch.argmax(probs, dim=1).cpu().numpy()
        
        return SinglesEnv.action_to_order(action[0], battle)

def create_training_env(team_text):
    def _init():
        env = Gen3Env(
            battle_format=BATTLE_FORMAT,
            team=Gen3Teambuilder(team_text),
            log_level=40,
            server_configuration=LocalhostServerConfiguration,
        )
        opponent = SimpleHeuristicsPlayer(
            battle_format=BATTLE_FORMAT,
            team=Gen3Teambuilder(team_text),
            server_configuration=LocalhostServerConfiguration,
        )
        return Monitor(SingleAgentWrapper(env, opponent))
    return _init

async def evaluate_model(model, team_text):
    print("\nStarting Evaluation...")
    rl_player = RLPlayer(
        model=model,
        team=Gen3Teambuilder(team_text),
        battle_format=BATTLE_FORMAT,
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=10
    )
    
    random_player = RandomPlayer(
        battle_format=BATTLE_FORMAT,
        team=Gen3Teambuilder(team_text),
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=10
    )
    
    heuristic_player = SimpleHeuristicsPlayer(
        battle_format=BATTLE_FORMAT,
        team=Gen3Teambuilder(team_text),
        server_configuration=LocalhostServerConfiguration,
        max_concurrent_battles=10
    )

    print("Evaluating against RandomPlayer (100 battles)...")
    await rl_player.battle_against(random_player, n_battles=100)
    print(f"Win rate vs Random: {rl_player.n_won_battles}%")

    rl_player.reset_battles()
    print("Evaluating against SimpleHeuristicsPlayer (100 battles)...")
    await rl_player.battle_against(heuristic_player, n_battles=100)
    print(f"Win rate vs Heuristic: {rl_player.n_won_battles}%")

async def main():
    parser = argparse.ArgumentParser(description="Train or Evaluate Gen 3 OU RL Agent")
    parser.add_argument("--model", type=str, help="Path to existing model to load")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and only evaluate the model")
    parser.add_argument("--steps", type=int, default=100000, help="Number of training steps")
    args = parser.parse_args()

    # Load all teams
    with open("data/teams/teams.json", "r") as f:
        teams_meta = json.load(f)
    
    all_team_texts = [open(os.path.join("data", t["file"])).read() for t in teams_meta]

    def get_random_team():
        return np.random.choice(all_team_texts)

    def create_training_env_random(idx):
        def _init():
            ts = datetime.now().strftime('%H%M%S')
            env_username = f"RLAgent_{idx}_{ts}"
            opp_username = f"Opponent_{idx}_{ts}"
            
            env = Gen3Env(
                battle_format=BATTLE_FORMAT,
                team=Gen3Teambuilder(get_random_team()),
                log_level=40,
                server_configuration=LocalhostServerConfiguration,
                account_configuration1=AccountConfiguration(env_username, "password"),
            )
            opponent = SimpleHeuristicsPlayer(
                battle_format=BATTLE_FORMAT,
                team=Gen3Teambuilder(get_random_team()),
                server_configuration=LocalhostServerConfiguration,
                account_configuration=AccountConfiguration(opp_username, "password"),
            )
            return Monitor(SingleAgentWrapper(env, opponent))
        return _init

    # Running parallel environments
    n_envs = 4
    print(f"Initializing {n_envs} parallel environments via SubprocVecEnv...")
    env = SubprocVecEnv([create_training_env_random(i) for i in range(n_envs)])

    async def evaluate_model_random(model):
        print("\nStarting Random-vs-Random Evaluation...")
        rl_player = RLPlayer(
            model=model,
            team=Gen3Teambuilder(get_random_team()),
            battle_format=BATTLE_FORMAT,
            server_configuration=LocalhostServerConfiguration,
            max_concurrent_battles=10
        )
        
        random_player = RandomPlayer(
            battle_format=BATTLE_FORMAT,
            team=Gen3Teambuilder(get_random_team()),
            server_configuration=LocalhostServerConfiguration,
            max_concurrent_battles=10
        )
        
        heuristic_player = SimpleHeuristicsPlayer(
            battle_format=BATTLE_FORMAT,
            team=Gen3Teambuilder(get_random_team()),
            server_configuration=LocalhostServerConfiguration,
            max_concurrent_battles=10
        )

        print("Evaluating (Random Team) against RandomPlayer (Random Team) [100 battles]...")
        await rl_player.battle_against(random_player, n_battles=100)
        print(f"Win rate vs Random: {rl_player.n_won_battles}%")

        rl_player.reset_battles()
        print("Evaluating (Random Team) against HeuristicPlayer (Random Team) [100 battles]...")
        await rl_player.battle_against(heuristic_player, n_battles=100)
        print(f"Win rate vs Heuristic: {rl_player.n_won_battles}%")

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
        model = PPO.load(model_path, env=env, device="cpu")
        
        if args.eval_only:
            await evaluate_model_random(model)
            return
        else:
            print(f"Continuing training for {args.steps} additional steps...")
            unique_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_dir = f"models/gen3ou_ppo_random_continued_{unique_id}"
            os.makedirs(model_dir, exist_ok=True)
            
            checkpoint_callback = CheckpointCallback(
                save_freq=10000, 
                save_path=model_dir,
                name_prefix="checkpoint"
            )
            
            model.learn(total_timesteps=args.steps, callback=checkpoint_callback, reset_num_timesteps=False)
            
            final_path = os.path.join(model_dir, "final_model")
            model.save(final_path)
            print(f"Continued training complete. Model saved to {final_path}")
            await evaluate_model_random(model)
    else:
        print(f"Starting NEW Generalist RL training (Random vs Random, Parallel x4)")
        model = PPO(
            MaskedActorCriticPolicy,
            env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            device="cpu",
        )

        unique_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_dir = f"models/gen3ou_ppo_generalist_{unique_id}"
        os.makedirs(model_dir, exist_ok=True)

        checkpoint_callback = CheckpointCallback(
            save_freq=10000, 
            save_path=model_dir,
            name_prefix="checkpoint"
        )

        model.learn(total_timesteps=args.steps, callback=checkpoint_callback)
        final_path = os.path.join(model_dir, "final_model")
        model.save(final_path)
        print(f"Training complete. Model saved to {final_path}")
        await evaluate_model_random(model)

if __name__ == "__main__":
    asyncio.run(main())
