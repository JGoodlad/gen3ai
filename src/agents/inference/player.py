import numpy as np
import torch
from typing import Dict, Any
from poke_env.player import Player

from agents.action.mapper import Gen3ActionMapper


class Gen3Player(Player):
    """Base player: embeds battle state and maps actions using Gen 3 logic."""

    def __init__(self, observation_encoder=None, mappings=None, **kwargs):
        super().__init__(**kwargs)
        self.observation_encoder = observation_encoder
        self.mappings = mappings

    def embed_battle(self, battle) -> Dict[str, Any]:
        if self.observation_encoder is None:
            from agents.observation.state_encoder import load_mappings, get_observation_encoder
            if self.mappings is None:
                self.mappings = load_mappings()
            self.observation_encoder = get_observation_encoder(self.mappings)
        return self.observation_encoder.get_observation(battle)

    def action_to_order(self, action_idx, battle):
        ctx = getattr(battle, "_gen3_decision_context", None)
        if ctx is None or ctx.get("turn") != battle.turn:
            raise RuntimeError(
                f"action_to_order() called at turn {battle.turn} but decision context is "
                f"{'missing' if ctx is None else f'from turn {ctx.get(\"turn\")}'}. "
                "embed_battle() must be called before action_to_order()."
            )
        return Gen3ActionMapper.action_to_order(action=action_idx, battle=battle)


class RLPlayer(Gen3Player):
    """Runs a trained SB3 model to choose moves during evaluation."""

    def __init__(self, model, team, battle_format, server_configuration,
                 mappings=None, account_configuration=None,
                 max_concurrent_battles=10, **kwargs):
        super().__init__(
            observation_encoder=None,
            mappings=mappings,
            battle_format=battle_format,
            team=team,
            server_configuration=server_configuration,
            account_configuration=account_configuration,
            max_concurrent_battles=max_concurrent_battles,
            **kwargs,
        )
        self.model = model

    def _predict_best_action(self, battle, stochastic=False):
        obs_dict = self.embed_battle(battle)
        obs = obs_dict["observation"]
        mask = obs_dict["action_mask"]

        obs_batched = np.expand_dims(obs, axis=0)
        mask_batched = np.expand_dims(mask, axis=0)

        with torch.no_grad():
            obs_tensor = torch.as_tensor(obs_batched).to(self.model.device)
            mask_tensor = torch.as_tensor(mask_batched).to(self.model.device)
            dist = self.model.policy.get_distribution(
                {"observation": obs_tensor, "action_mask": mask_tensor}
            )
            logits = dist.distribution.logits
            masked_logits = logits + (mask_tensor - 1.0) * 1e9

            if stochastic:
                idx = torch.distributions.Categorical(logits=masked_logits).sample().item()
            else:
                idx = torch.argmax(masked_logits, dim=1).item()

            probs = torch.softmax(masked_logits, dim=1)[0].cpu().numpy()

        if mask[idx] == 0:
            raise ValueError(f"Illegal action {idx} selected. Mask: {mask}")

        return idx, probs, mask

    def choose_move(self, battle):
        idx, _, _ = self._predict_best_action(battle)
        return self.action_to_order(idx, battle)
