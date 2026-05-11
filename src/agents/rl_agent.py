import numpy as np
import torch
import asyncio
from poke_env.player import Player
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
from poke_env.environment.singles_env import SinglesEnv
from gymnasium import spaces
import gymnasium as gym

class RLPlayer(Player):
    def __init__(self, model, team, battle_format, server_configuration, mappings=None, account_configuration=None, max_concurrent_battles=10):
        super().__init__(
            battle_format=battle_format,
            team=team,
            server_configuration=server_configuration,
            account_configuration=account_configuration,
            max_concurrent_battles=max_concurrent_battles,
        )
        self.model = model
        self.mappings = mappings
        self.observation_encoder = None

    def choose_move(self, battle):
        if self.observation_encoder is None:
            from main.train_rl_agent import get_observation_encoder
            self.observation_encoder = get_observation_encoder(self.mappings)
            
        obs = self.observation_encoder.encode(battle)
        
        # Convert to 10-dim binary mask
        from agents.action.mask_generator import Gen3ActionMasker
        mask = Gen3ActionMasker.get_mask(battle)
        
        # Ensure batch dimension for SB3
        obs_batched = np.expand_dims(obs, axis=0)
        mask_batched = np.expand_dims(mask, axis=0)
        
        # Predict action
        action, _ = self.model.predict(
            {"observation": obs_batched, "action_mask": mask_batched}, 
            deterministic=True
        )
        
        return SinglesEnv.action_to_order(action[0], battle)

# We re-export SingleAgentWrapper from poke_env to maintain compatibility with train_rl_agent.py imports
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
