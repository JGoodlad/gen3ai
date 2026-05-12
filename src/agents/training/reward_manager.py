import sys
import time
from typing import Optional
from utils.logging.rate_limiter import RateLimitedLogger
from utils.logging.levels import LogLevel

class Gen3RewardManager:
    """
    Handles complex reward calculation, switch subsidies, rate-limited logging,
    and episode-level metric tracking for the Gen 3 RL environment.
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

    def process_turn_reward(self, battle, base_reward: float, is_trainee: bool) -> float:
        """
        Calculates the final reward for a turn, including subsidies and logging.
        
        :param battle: The battle object for the current turn.
        :param base_reward: The delta from reward_computing_helper.
        :param is_trainee: Whether this battle is for the trainee (only trainee gets tracked).
        :return: The final reward value to be passed to the RL agent.
        """
        reward = base_reward
        
        # 1. Rate-limited logging check
        is_log_turn = self.log_level >= LogLevel.DETAILED and is_trainee and self.logger.should_log()
        
        if is_log_turn:
            hp = battle.active_pokemon.current_hp_fraction if battle.active_pokemon else 0.0
            self.logger.log(
                f"  [REWARD] Turn {battle.turn} | Base: {base_reward:+.4f} | HP: {hp:.2f} | Won: {battle.won} | Lost: {battle.lost}\n",
                force=True
            )

        # 2. Logarithmic Switch Reward with Turn Decay
        current_active = battle.active_pokemon.name if battle.active_pokemon else "NONE"
        if current_active != self._last_active_name:
            # Check if voluntary via the latched mask on the battle object
            latched_mask = getattr(battle, "_latched_mask", None)
            is_voluntary = False
            if latched_mask is not None:
                # If any move (6-9) or Struggle (10) was legal, it was a choice
                is_voluntary = any(latched_mask[6:11] == 1)

            if is_voluntary:
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
        
        # 3. Trainee-only tracking (Prevents leakage from opponent's mirrored rewards)
        if is_trainee:
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
