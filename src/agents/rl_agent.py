import asyncio
import os
import traceback
from typing import Dict, Any
from poke_env.player import Player
from poke_env.environment.singles_env import SinglesEnv
from agents.action.mapper import Gen3ActionMapper
import numpy as np
import torch

class Gen3Player(Player):
    """Base class for all Gen 3 players that handles embedding and mapping."""
    def __init__(self, observation_encoder=None, mappings=None, **kwargs):
        super().__init__(**kwargs)
        self.observation_encoder = observation_encoder
        self.mappings = mappings

    def embed_battle(self, battle) -> Dict[str, Any]:
        """Centralized method to get the observation dict (obs + mask)."""
        if self.observation_encoder is None:
            from agents.observation.state_encoder import get_observation_encoder
            # mappings should be loaded if not provided
            if self.mappings is None:
                from agents.observation.state_encoder import load_mappings
                self.mappings = load_mappings()
            self.observation_encoder = get_observation_encoder(self.mappings)
        return self.observation_encoder.get_observation(battle)

    def action_to_order(self, action_idx, battle):
        """Delegates to the centralized Gen3ActionMapper."""
        return Gen3ActionMapper.action_to_order(
            action=action_idx,
            battle=battle
        )

class RLPlayer(Gen3Player):
    def __init__(self, model, team, battle_format, server_configuration, mappings=None, account_configuration=None, max_concurrent_battles=10, **kwargs):
        super().__init__(
            observation_encoder=None,
            mappings=mappings,
            battle_format=battle_format,
            team=team,
            server_configuration=server_configuration,
            account_configuration=account_configuration,
            max_concurrent_battles=max_concurrent_battles,
            **kwargs
        )
        self.model = model

    def _predict_best_action(self, battle, stochastic=False):
        """Shared logic for masked action prediction."""
        obs_dict = self.embed_battle(battle)
        obs = obs_dict["observation"]
        mask = obs_dict["action_mask"]
        
        obs_batched = np.expand_dims(obs, axis=0)
        mask_batched = np.expand_dims(mask, axis=0)
        
        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs_batched).to(self.model.device)
            mask_tensor = torch.as_tensor(mask_batched).to(self.model.device)
            dist = self.model.policy.get_distribution({"observation": obs_tensor, "action_mask": mask_tensor})
            
            # Apply mask to logits for both stochastic and deterministic paths
            logits = dist.distribution.logits
            masked_logits = logits + (mask_tensor - 1.0) * 1e9
            
            if stochastic:
                # Create a new categorical distribution using the masked logits
                idx = torch.distributions.Categorical(logits=masked_logits).sample().item()
            else:
                idx = torch.argmax(masked_logits, dim=1).item()
            
            # Ensure probabilities reported in logs also reflect the mask
            probs = torch.softmax(masked_logits, dim=1)[0].cpu().numpy()
            
        if mask[idx] == 0:
            raise ValueError(f"STRICT MODE FAILURE: Illegal action {idx}. Mask: {mask}")
            
        return idx, probs, mask

    def choose_move(self, battle):
        idx, _, _ = self._predict_best_action(battle)
        return self.action_to_order(idx, battle)

# We re-export SingleAgentWrapper from poke_env to maintain compatibility with train_rl_agent.py imports
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
