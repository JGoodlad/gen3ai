import asyncio
import os
import traceback
from typing import Dict, Any
from poke_env.player import Player
from poke_env.environment.singles_env import SinglesEnv
from agents.action.mapper import Gen3ActionMapper
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

    def _predict_best_action(self, battle):
        """Shared logic for masked action prediction."""
        if self.observation_encoder is None:
            from main.train_rl_agent import get_observation_encoder
            self.observation_encoder = get_observation_encoder(self.mappings)
            
        obs = self.observation_encoder.encode(battle)
        from agents.action.mask_generator import Gen3ActionMasker
        mask = Gen3ActionMasker.get_mask(battle)
        
        obs_batched = np.expand_dims(obs, axis=0)
        mask_batched = np.expand_dims(mask, axis=0)
        
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs_batched).to(self.model.device)
            mask_tensor = torch.as_tensor(mask_batched).to(self.model.device)
            dist = self.model.policy.get_distribution({"observation": obs_tensor, "action_mask": mask_tensor})
            logits = dist.distribution.logits
            masked_logits = logits + (mask_tensor - 1.0) * 1e9
            idx = torch.argmax(masked_logits, dim=1).item()
            probs = torch.softmax(masked_logits, dim=1)[0].cpu().numpy()
            
        if mask[idx] == 0:
            raise ValueError(f"STRICT MODE FAILURE: Illegal action {idx}. Mask: {mask}")
            
        return idx, probs, mask

    def action_to_order(self, action_idx, battle):
        """Delegates to the centralized Gen3ActionMapper."""
        # RLPlayer (Inference) uses the same strict mapping as the trainee
        return Gen3ActionMapper.action_to_order(
            action=action_idx,
            battle=battle
        )
    def choose_move(self, battle):
        idx, _, _ = self._predict_best_action(battle)
        return self.action_to_order(idx, battle)

# We re-export SingleAgentWrapper from poke_env to maintain compatibility with train_rl_agent.py imports
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
