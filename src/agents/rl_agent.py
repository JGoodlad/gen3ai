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
        """
        Maps an 11-action discrete index to a poke_env BattleOrder.
        Uses Strict Alphabetical Move Sorting to ensure stability across workers.
        """
        from poke_env.player.battle_order import SingleBattleOrder
        
        # 0-5: Switches (Team Slots 1-6)
        team_list = list(battle.team.values())
        if action_idx < 6:
            if action_idx < len(team_list):
                target_mon = team_list[action_idx]
                if target_mon in battle.available_switches:
                    return SingleBattleOrder(target_mon)
            return self.choose_random_move(battle)

        # 6-9: Moves (Slots 1-4)
        elif action_idx < 10:
            move_idx = action_idx - 6
            active_pokemon = battle.active_pokemon
            if active_pokemon:
                # SORT moves by ID to ensure stable mapping (Matches Encoder and Masker)
                mon_moves = sorted(active_pokemon.moves.values(), key=lambda m: m.id)[:4]
                if move_idx < len(mon_moves):
                    target_move = mon_moves[move_idx]
                    if target_move in battle.available_moves:
                        return SingleBattleOrder(target_move)
            return self.choose_random_move(battle)

        # 10: Struggle
        elif action_idx == 10:
            available_moves = battle.available_moves
            if len(available_moves) == 1 and available_moves[0].id == "struggle":
                return SingleBattleOrder(available_moves[0])
            return self.choose_random_move(battle)

        return self.choose_random_move(battle)

    def choose_move(self, battle):
        idx, _, _ = self._predict_best_action(battle)
        return self.action_to_order(idx, battle)

# We re-export SingleAgentWrapper from poke_env to maintain compatibility with train_rl_agent.py imports
from poke_env.environment.single_agent_wrapper import SingleAgentWrapper
