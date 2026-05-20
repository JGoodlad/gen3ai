import numpy as np
import torch
from typing import Dict, Any, Optional
from poke_env.player import Player
from poke_env.player.battle_order import ForfeitBattleOrder

from agents.action.mapper import Gen3ActionMapper
from agents.observation.turn_delta_encoder import TURN_DELTA_DIM
from agents.training.stall import StallConfig, StallLogger


class Gen3Player(Player):
    """Base player: embeds battle state and maps actions using Gen 3 logic."""

    def __init__(self, observation_encoder=None, mappings=None,
                 stall_config: Optional[StallConfig] = None, **kwargs):
        super().__init__(**kwargs)
        self.observation_encoder = observation_encoder
        self.mappings = mappings
        self._stall_config = stall_config or StallConfig()
        self._stall_loggers: dict[str, StallLogger] = {}

    def _handle_stall(self, battle, suffix: str) -> Optional[ForfeitBattleOrder]:
        """Returns ForfeitBattleOrder if the battle exceeds the stall threshold, else None.
        Each battle gets its own StallLogger, so concurrent battles don't interfere."""
        tag = battle.battle_tag
        if tag not in self._stall_loggers:
            self._stall_loggers[tag] = StallLogger(self._stall_config)
        stall_logger = self._stall_loggers[tag]
        if battle.turn >= stall_logger.threshold:
            stall_logger.log_once(battle, suffix=suffix)
            return ForfeitBattleOrder()
        return None

    def _battle_finished_callback(self, battle) -> None:
        super()._battle_finished_callback(battle)
        self._stall_loggers.pop(battle.battle_tag, None)

    def embed_battle(self, battle) -> Dict[str, Any]:
        if self.observation_encoder is None:
            from agents.observation.state_encoder import load_mappings, get_observation_encoder
            if self.mappings is None:
                self.mappings = load_mappings()
            self.observation_encoder = get_observation_encoder(self.mappings)
        result = self.observation_encoder.get_observation(battle)
        # Inference players have no episode history: all-ones prev_mask, zeros TurnDelta
        result["observation"] = np.concatenate([
            result["observation"],
            np.ones(11, dtype=np.float32),
            np.zeros(TURN_DELTA_DIM, dtype=np.float32),
        ])
        return result

    def action_to_order(self, action_idx, battle):
        ctx = getattr(battle, "_gen3_decision_context", None)
        if ctx is None or ctx.get("turn") != battle.turn:
            ctx_desc = "missing" if ctx is None else f"from turn {ctx.get('turn')}"
            raise RuntimeError(
                f"action_to_order() called at turn {battle.turn} but decision context is "
                f"{ctx_desc}. embed_battle() must be called before action_to_order()."
            )
        return Gen3ActionMapper.action_to_order(action=action_idx, battle=battle)


class RLPlayer(Gen3Player):
    """Runs a trained SB3 model to choose moves during evaluation."""

    def __init__(self, model, team, battle_format, server_configuration,
                 mappings=None, account_configuration=None,
                 stall_config: Optional[StallConfig] = None,
                 max_concurrent_battles=10, **kwargs):
        super().__init__(
            observation_encoder=None,
            mappings=mappings,
            stall_config=stall_config,
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
        forfeit = self._handle_stall(battle, "INFERENCE_STALL")
        if forfeit:
            return forfeit
        idx, _, _ = self._predict_best_action(battle)
        return self.action_to_order(idx, battle)
