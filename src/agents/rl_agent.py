import asyncio
import os
import traceback
from typing import Dict, Any
from poke_env.player import Player
from poke_env.environment.singles_env import SinglesEnv
import numpy as np
import torch

class RLPlayer(Player):
    def __init__(self, model, team, battle_format, server_configuration, mappings=None, account_configuration=None, max_concurrent_battles=10, **kwargs):
        super().__init__(
            battle_format=battle_format,
            team=team,
            server_configuration=server_configuration,
            account_configuration=account_configuration,
            max_concurrent_battles=max_concurrent_battles,
            **kwargs
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
        
        # Manually apply mask to logits for deterministic legal action
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs_batched).to(self.model.device)
            mask_tensor = torch.as_tensor(mask_batched).to(self.model.device)
            dist = self.model.policy.get_distribution({"observation": obs_tensor, "action_mask": mask_tensor})
            logits = dist.distribution.logits
            masked_logits = logits + (mask_tensor - 1.0) * 1e9
            idx = torch.argmax(masked_logits, dim=1).item()
        
        # Verify legality (Strict Mode)
        if mask[idx] == 0:
            raise ValueError(f"STRICT MODE FAILURE: Main player picked illegal action {idx}. Mask: {mask}")

        # Absolute Team Slot Mapping
        if idx < 6:
            team_list = list(battle.team.values())
            if idx < len(team_list):
                return SinglesEnv.action_to_order(team_list[idx], battle)
        else:
            move_idx = idx - 6
            if move_idx < len(battle.available_moves):
                return SinglesEnv.action_to_order(battle.available_moves[move_idx], battle)
        
        # Final fallback to standard (should never happen in strict mode)
        return SinglesEnv.action_to_order(idx, battle)

# We re-export SingleAgentWrapper from poke_env to maintain compatibility with train_rl_agent.py imports
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
