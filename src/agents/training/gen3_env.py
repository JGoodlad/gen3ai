import sys
import os
import numpy as np
from datetime import datetime
from gymnasium import spaces
from typing import Optional

from poke_env.environment.singles_env import SinglesEnv
from poke_env.player.battle_order import BattleOrder, ForfeitBattleOrder

from agents.observation.state_encoder import get_observation_encoder
from agents.action.mask_generator import Gen3ActionMasker
from agents.action.mapper import Gen3ActionMapper
from agents.training.reward_manager import Gen3RewardManager
from agents.training.reward_function import RewardFunction
from agents.training.battle_context import BattleContext, TurnDelta
from agents.training.slot_registry import SlotRegistry
from utils.logging.levels import LogLevel

STALL_THRESHOLD = 250


class Gen3Env(SinglesEnv):
    def __init__(self, mappings, reward_fn: Optional[RewardFunction] = None, log_level=LogLevel.QUIET, stalls_dir=None, *args, **kwargs):
        self.log_level = log_level
        self.stalls_dir = stalls_dir
        self._stall_logged = False
        super().__init__(*args, **kwargs)
        self.observation_encoder = get_observation_encoder(mappings)

        obs_dim = self.observation_encoder.dimension
        self.vector_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(11)
        self.observation_space = spaces.Dict({
            "observation": self.vector_space,
            "action_mask": spaces.Box(0, 1, shape=(11,), dtype=np.int8)
        })
        self.observation_spaces = {
            self.agent1.username: self.observation_space,
            self.agent2.username: self.observation_space
        }

        self.reward_manager: RewardFunction = reward_fn or Gen3RewardManager(log_level=self.log_level)

        self._our_slots = SlotRegistry()
        self._opp_slots = SlotRegistry()
        self._last_ctx: Optional[BattleContext] = None
        self._prev_ctx: Optional[BattleContext] = None
        self._last_action: int = -1

    def embed_battle(self, battle):
        obs = self.observation_encoder.encode(battle)

        if battle is self.battle1 and not battle.finished:
            mask = Gen3ActionMasker.get_mask(battle).astype(np.int8)

            if mask.sum() > 0:
                for mon in battle.team.values():
                    self._our_slots.assign(mon.species)
                for mon in battle.opponent_team.values():
                    self._opp_slots.assign(mon.species)

                our_hp = np.zeros(6, dtype=np.float32)
                for mon in battle.team.values():
                    slot = self._our_slots.get(mon.species)
                    if slot is not None:
                        our_hp[slot] = mon.current_hp_fraction

                opp_hp = np.zeros(6, dtype=np.float32)
                for mon in battle.opponent_team.values():
                    slot = self._opp_slots.get(mon.species)
                    if slot is not None:
                        opp_hp[slot] = mon.current_hp_fraction

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
                    obs=obs,
                    our_slot_map=self._our_slots.snapshot(),
                    opp_slot_map=self._opp_slots.snapshot(),
                    our_hp=our_hp,
                    opp_hp=opp_hp,
                    our_active=our_active,
                    opp_active=opp_active,
                    our_fainted_count=sum(1 for m in battle.team.values() if m.fainted),
                    opp_fainted_count=sum(1 for m in battle.opponent_team.values() if m.fainted),
                )

        return obs

    def action_masks(self) -> np.ndarray:
        if self._last_ctx is not None:
            return self._last_ctx.mask
        return np.ones(11, dtype=np.int8)

    def get_action_mask(self, battle):
        if battle is self.battle1 and self._last_ctx is not None:
            return self._last_ctx.mask
        return Gen3ActionMasker.get_mask(battle).astype(np.int8)

    def action_to_order(self, action, battle, **kwargs):
        if isinstance(action, BattleOrder):
            return action

        if battle is self.battle1:
            if battle.turn >= STALL_THRESHOLD:
                if not self._stall_logged:
                    self._save_stall_html(battle, suffix="STALL")
                    self._stall_logged = True
                return ForfeitBattleOrder()

            ctx = self._last_ctx
            return Gen3ActionMapper.action_to_order(
                action=action,
                battle=battle,
                mask=ctx.mask if ctx is not None else None,
                latched_turn=ctx.turn if ctx is not None else -1,
            )

        return super().action_to_order(action, battle)

    def calc_reward(self, battle):
        if battle is self.battle1:
            if self._prev_ctx is not None and self._last_ctx is not None:
                delta = TurnDelta.build(self._prev_ctx, self._last_ctx, self._last_action)
            else:
                delta = TurnDelta.empty()
            return self.reward_manager.process_turn_reward(battle, delta)
        return self.reward_computing_helper(
            battle, fainted_value=2.0, hp_value=1.0, victory_value=30.0
        )

    def step(self, action):
        try:
            battle = getattr(self, "_battle", None)
            if battle is None:
                battle = self.battle1

            if isinstance(action, dict):
                trainee_idx = action.get(self.agent1.username, -1)
            else:
                trainee_idx = action

            if battle is self.battle1 and self._last_ctx is not None:
                self._prev_ctx = self._last_ctx
                self._last_action = trainee_idx
                self.reward_manager.record_action(self._last_ctx, trainee_idx)

            obs, reward, term, trunc, info = super().step(action)

            if hasattr(self, "_pending_switch_log"):
                info["switch_log"] = self._pending_switch_log
                del self._pending_switch_log

            return obs, reward, term, trunc, info
        except Exception as e:
            import traceback
            print(f"ERROR IN STEP: {e}")
            traceback.print_exc()
            raise e

    def reset(self, *args, **kwargs):
        self.reward_manager.report_episode(getattr(self, "battle1", None))

        self._move_slot_cache = {}
        self._last_active_name = ""
        self._our_slots.reset()
        self._opp_slots.reset()
        self._last_ctx = None
        self._prev_ctx = None
        try:
            if hasattr(self, "agent1"):
                self.agent1.save_replays = None

            self.reward_manager.reset()
            self._stall_logged = False
            self._last_action = -1
            return super().reset(*args, **kwargs)
        except Exception as e:
            import traceback
            print(f"ERROR IN RESET: {e}")
            traceback.print_exc()
            raise e

    def _save_stall_html(self, battle, suffix=""):
        if not self.stalls_dir:
            return
        try:
            os.makedirs(self.stalls_dir, exist_ok=True)
            ts = datetime.now().strftime('%Y%m%d_%H%M%S')
            suffix_str = f"_{suffix}" if suffix else ""
            filename = f"stall_{battle.battle_tag}_{ts}{suffix_str}.html"
            path = os.path.join(self.stalls_dir, filename)
            battle.save_replay(path)
            sys.stderr.write(f"\n[STALL LOGGED] Battle {battle.battle_tag} lasted {battle.turn} turns. HTML saved to {path}\n")
            sys.stderr.flush()
        except Exception as e:
            sys.stderr.write(f"Failed to save stall log: {e}\n")
            sys.stderr.flush()
