import sys
import time
from typing import Optional
from utils.logging.rate_limiter import RateLimitedLogger
from utils.logging.levels import LogLevel
from utils.gen3_utils import SwitchDetection

class SinglePlayerRewardManager:
    """
    A decorator/wrapper that ensures a reward manager only processes turns
    for the primary trainee agent, preventing cross-talk with opponents.
    """
    def __init__(self, manager: 'Gen3RewardManager'):
        self.manager = manager

    def process_turn_reward(self, battle, base_reward: float, is_trainee: bool) -> float:
        if is_trainee:
            return self.manager.process_turn_reward(battle, base_reward)
        return base_reward

    def __getattr__(self, name):
        """Delegate everything else (reset, report_episode, etc) to the internal manager."""
        return getattr(self.manager, name)

class Gen3RewardManager:
    """
    Handles complex reward calculation, switch subsidies, rate-limited logging,
    and episode-level metric tracking for the Gen 3 RL environment.
    
    Note: This class assumes it is only called for a single agent (the trainee).
    Use SinglePlayerRewardManager to enforce this.
    """
    def __init__(self, log_level: LogLevel = LogLevel.QUIET):
        self.log_level = log_level
        self.switch_count = 0
        self.total_reward = 0.0
        self.logger = RateLimitedLogger(interval_seconds=1.0)
        self.episode_logger = RateLimitedLogger(interval_seconds=15.0)
        self._last_active_name = ""
        
    def reset(self):
        """Prepares for a new episode."""
        self.switch_count = 0
        self.total_reward = 0.0
        self._last_active_name = ""

    def process_turn_reward(self, battle, base_reward: float) -> float:
        """
        Calculates the final reward for a turn, including subsidies and logging.
        
        :param battle: The battle object for the current turn.
        :param base_reward: The delta from reward_computing_helper.
        :return: The final reward value to be passed to the RL agent.
        """
        reward = base_reward
        
        # 1. Rate-limited logging check
        is_log_turn = self.log_level >= LogLevel.DETAILED and self.logger.should_log()
        
        if is_log_turn:
            hp = battle.active_pokemon.current_hp_fraction if battle.active_pokemon else 0.0
            self.logger.log(
                f"  [REWARD] Turn {battle.turn} | Base: {base_reward:+.4f} | HP: {hp:.2f} | Won: {battle.won} | Lost: {battle.lost}\n",
                force=True
            )

        # 2. Logarithmic Switch Reward with Turn Decay
        current_active = battle.active_pokemon.species if battle.active_pokemon else "NONE"
        latched_mask = getattr(battle, "_latched_mask", None)
        
        has_switched, is_real, is_voluntary = SwitchDetection.get_switch_type(
            self._last_active_name, current_active, latched_mask
        )
        
        if is_real and is_voluntary is True:
            # Decay to 0 at Turn 250
            turn_decay = max(0.0, 1.0 - battle.turn / 250.0)
            subsidy = (3.75 / (self.switch_count + 1)) * turn_decay
            reward += subsidy
            self.switch_count += 1
            
            if is_log_turn:
                self.logger.log(
                    f"  [SUBSIDY] +{subsidy:.2f} | Switches: {self.switch_count} | Decay: {turn_decay:.2f}\n",
                    force=True
                )

        self._last_active_name = current_active
        
        # 3. Trainee-only tracking (Already isolated by wrapper)
        self.total_reward += reward
            
        return reward

    def report_episode(self, battle):
        """Prints the final summary of the episode to stderr."""
        if self.log_level < LogLevel.PERIODIC or self.total_reward == 0:
            return

        status = "UNKNOWN"
        if battle:
            if battle.won: status = "WIN"
            elif battle.lost: status = "LOSS"
            elif battle.finished: status = "FINISHED"

        msg = f"\n🏁 Episode Finished | Status: {status} | Total Reward: {self.total_reward:.2f} | Voluntary Switches: {self.switch_count}\n"
        self.episode_logger.log(msg, force=False)
