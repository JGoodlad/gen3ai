import gymnasium as gym
import numpy as np
import os
import json
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from datetime import datetime

from poke_env.environment.singles_env import SinglesEnv
from poke_env.player.player import Player
from poke_env.player.battle_order import ForfeitBattleOrder, BattleOrder
from poke_env.ps_client import AccountConfiguration, LocalhostServerConfiguration

from agents.training.reward_manager import Gen3RewardManager
from agents.training.battle_context import BattleContext, TurnDelta
from agents.action.mask_generator import Gen3ActionMasker
from agents.action.mapper import Gen3ActionMapper
from utils.logging.levels import LogLevel

class Gen3SinglesEnv(SinglesEnv, gym.Env):
    """
    A premium single-agent Gymnasium environment for Gen 3 OU.
    Inherits from SinglesEnv to properly override action mapping.
    """
    def __init__(
        self,
        opponent: Union[Player, Callable[[], Player]],
        observation_encoder: Any,
        reward_manager: Gen3RewardManager,
        log_level: LogLevel = LogLevel.QUIET,
        stalls_dir: Optional[str] = None,
        **poke_env_kwargs
    ):
        # We need to pass account_configurations and other kwargs to SinglesEnv
        super().__init__(log_level=log_level, **poke_env_kwargs)
        
        # Handle Opponent (Instance or Factory)
        if callable(opponent):
            self.opponent = opponent()
        else:
            self.opponent = opponent
        self.observation_encoder = observation_encoder
        self.reward_manager = reward_manager
        self.log_level = log_level
        self.stalls_dir = stalls_dir
        
        # Override the action space for single-agent gym use
        self.action_space = gym.spaces.Discrete(11)
        obs_dim = self.observation_encoder.dimension
        self.observation_space = gym.spaces.Dict({
            "observation": gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32),
            "action_mask": gym.spaces.Box(0, 1, shape=(11,), dtype=np.int8)
        })
        
        # Internal State
        self.battle: Optional[Any] = None
        self.opponent_battle: Optional[Any] = None
        self._last_obs: Optional[np.ndarray] = None
        self._last_mask: Optional[np.ndarray] = None
        self._seen_opponent_slots: Dict[str, int] = {}
        self._step_count = 0
        self._episode_count = 0
        self._last_ctx: Optional[BattleContext] = None
        self._prev_ctx: Optional[BattleContext] = None
        self._last_action: int = -1
        
        # Forensic History
        self._forensic_history: List[Dict[str, Any]] = []

    def action_to_order(self, action: Any, battle: Any, **kwargs) -> BattleOrder:
        """Overrides SinglesEnv mapping with Gen 3 specific logic."""
        # Handle cases where action might be a BattleOrder already (from heuristic opponents)
        if isinstance(action, BattleOrder):
            return action
        return Gen3ActionMapper.action_to_order(int(action), battle)

    def order_to_action(self, order: BattleOrder, battle: Any, **kwargs) -> int:
        """Overrides SinglesEnv mapping with Gen 3 specific logic."""
        return int(Gen3ActionMapper.order_to_action(order, battle))

    def _get_observation(self, battle) -> Dict[str, Any]:
        """Delegates observation and mask generation to the encoder."""
        obs_dict = self.observation_encoder.get_observation(battle)
        mask = obs_dict["action_mask"]
        self._last_mask = mask

        if not battle.finished and mask.sum() > 0:
            our_hp = np.zeros(6, dtype=np.float32)
            opp_hp = np.zeros(6, dtype=np.float32)
            slot_map: Dict[str, int] = {}
            opp_slot_map: Dict[str, int] = {}
            for i, mon in enumerate(battle.team.values()):
                if i < 6:
                    our_hp[i] = mon.current_hp_fraction
                    slot_map[mon.species] = i
            for i, mon in enumerate(battle.opponent_team.values()):
                if i < 6:
                    opp_hp[i] = mon.current_hp_fraction
                    opp_slot_map[mon.species] = i

            our_active = (
                battle.active_pokemon.species
                if battle.active_pokemon and not battle.active_pokemon.fainted
                else "NONE"
            )
            opp_active = (
                battle.opponent_active_pokemon.species
                if battle.opponent_active_pokemon
                else "NONE"
            )
            self._last_ctx = BattleContext(
                turn=battle.turn,
                phase="forced_switch" if battle.force_switch else "move_selection",
                mask=mask,
                obs=obs_dict["observation"],
                our_slot_map=slot_map,
                opp_slot_map=opp_slot_map,
                our_hp=our_hp,
                opp_hp=opp_hp,
                our_active=our_active,
                opp_active=opp_active,
                our_fainted_count=sum(1 for m in battle.team.values() if m.fainted),
                opp_fainted_count=sum(1 for m in battle.opponent_team.values() if m.fainted),
            )

        return obs_dict

    def reset(self, seed=None, options=None):
        self._episode_count += 1
        self.reward_manager.reset()
        self._last_ctx = None
        self._prev_ctx = None
        self._last_action = -1

        # Reset via SinglesEnv (PettingZoo API)
        obs_dict, info_dict = super().reset(seed=seed, options=options)

        self.battle = self.battle1
        self.opponent_battle = self.battle2

        # Sync the opponent with their battle
        self.opponent.reset_battles()
        self.opponent._battles[self.opponent_battle.battle_tag] = self.opponent_battle

        return self._get_observation(self.battle), self._get_info()

    def step(self, action: int):
        self._step_count += 1
        if self._last_ctx is not None:
            self._prev_ctx = self._last_ctx
            self._last_action = action
            self.reward_manager.record_action(self._last_ctx, action)

        # Get Opponent Action
        if self.opponent_battle.wait:
            opp_action = 0
        else:
            opp_order = self.opponent.choose_move(self.opponent_battle)
            # Use our overriden order_to_action
            opp_action = self.order_to_action(opp_order, self.opponent_battle)
            
        # Step via SinglesEnv (PettingZoo API)
        actions = {
            self.agent1.username: np.int64(action),
            self.agent2.username: np.int64(opp_action)
        }
        
        obs_dict, reward_dict, term_dict, trunc_dict, info_dict = super().step(actions)
        
        # Internal Looping for "Wait" turns
        while self.battle.wait and not self.battle.finished:
            if self.log_level >= LogLevel.DEBUG:
                print(f"  [ENV] Trainee waiting at turn {self.battle.turn}. Auto-stepping...")
            
            if not self.opponent_battle.wait:
                opp_order = self.opponent.choose_move(self.opponent_battle)
                opp_action = self.order_to_action(opp_order, self.opponent_battle)
            else:
                opp_action = 0
                
            actions = {
                self.agent1.username: np.int64(0),
                self.agent2.username: np.int64(opp_action)
            }
            obs_dict, reward_dict, term_dict, trunc_dict, info_dict = super().step(actions)

        # Calculate Reward & Log Forensics (Delegated to Hub)
        if self._prev_ctx is not None and self._last_ctx is not None:
            delta = TurnDelta.build(self._prev_ctx, self._last_ctx, self._last_action)
        else:
            delta = TurnDelta.empty()
        final_reward = self.reward_manager.process_turn_reward(self.battle, delta)
        
        terminated = term_dict[self.agent1.username]
        truncated = trunc_dict[self.agent1.username]
        
        obs = self._get_observation(self.battle)
        return obs, final_reward, terminated, truncated, self._get_info()

    def _get_info(self) -> Dict[str, Any]:
        info: Dict[str, Any] = {}
        if self.battle:
            info["turn"] = self.battle.turn
            info["battle_tag"] = self.battle.battle_tag
        return info
