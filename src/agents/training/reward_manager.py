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

    def record_action(self, battle, action: int, is_trainee: bool):
        if is_trainee:
            self.manager.record_action(battle, action)

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
        self.forced_switch_count = 0
        self.attack_count = 0
        self.total_reward = 0.0
        self.remaining_switch_pool = 7.5  # Max total points for switching
        self.last_switch_turn = -1
        self.logger = RateLimitedLogger(interval_seconds=1.0)
        self.episode_logger = RateLimitedLogger(interval_seconds=5.0)
        self._last_active_name = "NULL"
        self._last_opp_active_name = "NULL"
        self._opp_turns_active = 0
        self._prev_active_name = "NULL"  # The one BEFORE the current one
        self._last_action_idx = -1
        
    def reset(self):
        """Prepares for a new episode."""
        self.switch_count = 0
        self.forced_switch_count = 0
        self.attack_count = 0
        self.total_reward = 0.0
        self.remaining_switch_pool = 7.5
        self.last_switch_turn = -1
        self._last_active_name = "NULL"
        self._last_opp_active_name = "NULL"
        self._opp_turns_active = 0
        self._prev_active_name = "NULL"
        self._last_action_idx = -1

    def record_action(self, battle, action: int):
        """
        Records a voluntary action EXACTLY ONCE.
        Called by Gen3Env.step before the simulation proceeds.
        """
        # 1. Track Attacks
        if action >= 6:
            self.attack_count += 1
            
        # 2. Track Switches
        # Treat fainted Pokémon as NONE to prevent 'ghost' voluntary switches
        is_fainted = battle.active_pokemon.fainted if battle.active_pokemon else False
        current_active = "NONE" if is_fainted or not battle.active_pokemon else battle.active_pokemon.species
        
        latched_mask = getattr(battle, "_latched_mask", None)
        
        has_switched, is_real, is_voluntary = SwitchDetection.get_switch_type(
            self._last_active_name, current_active, latched_mask
        )
        
        # 3. Track Opponent "Freshness"
        opp_active = "NONE" if not battle.opponent_active_pokemon else battle.opponent_active_pokemon.species
        if opp_active != self._last_opp_active_name:
            self._last_opp_active_name = opp_active
            self._opp_turns_active = 0
        else:
            self._opp_turns_active += 1

        # We only apply the subsidy here so it's tied to the CHOICE, not the turn delta
        self._pending_subsidy = 0.0
        
        # 4. Repetition & Bouncing Penalties
        # Penalty for clicking the exact same move/switch slot twice in a row
        if action == self._last_action_idx and action != -1:
            self._pending_subsidy -= 0.02
            
        if is_real:
            if is_voluntary is True:
                # Bouncing Penalty: Penalize switching back to the mon we just left
                if current_active == self._prev_active_name and self._prev_active_name != "NONE":
                    self._pending_subsidy -= 0.15
                
                payout = self.remaining_switch_pool * 0.5
                attack_ratio = self.attack_count / max(1, battle.turn)
                ratio_mult = 1.0 if attack_ratio >= 0.33 else 0.5
                spam_mult = 1.0 if (battle.turn - self.last_switch_turn) > 1 else 0.0
                turn_decay = max(0.0, 1.0 - battle.turn / 250.0)
                
                # Reactive Bonus: Reward switches more if the opponent just arrived (within 2 turns)
                # This fixes "aimless" switching against a stable opponent.
                reactive_mult = 1.0 if self._opp_turns_active <= 2 else 0.25
                
                self._pending_subsidy += min(payout * ratio_mult * spam_mult * turn_decay * reactive_mult, 1.0)
                
                self.remaining_switch_pool -= payout
                self.switch_count += 1
                self.last_switch_turn = battle.turn
            elif is_voluntary is False:
                # This is a forced switch (Replacement or Roar/Whirlwind)
                self.forced_switch_count += 1

        self._prev_active_name = self._last_active_name
        self._last_active_name = current_active
        self._last_action_idx = action

    def process_turn_reward(self, battle, base_reward: float) -> float:
        """
        Calculates the final reward for a turn, including subsidies and logging.
        """
        reward = base_reward
        
        # Apply any subsidy calculated in the last record_action call
        if hasattr(self, "_pending_subsidy"):
            reward += self._pending_subsidy
            subsidy_val = self._pending_subsidy
            del self._pending_subsidy
        else:
            subsidy_val = 0.0
        
        # Rate-limited logging check
        is_log_turn = self.log_level >= LogLevel.DETAILED and self.logger.should_log()
        
        if is_log_turn:
            hp = battle.active_pokemon.current_hp_fraction if battle.active_pokemon else 0.0
            self.logger.log(
                f"  [REWARD] Turn {battle.turn} | Base: {base_reward:+.4f} | Subsidy: {subsidy_val:+.2f} | Won: {battle.won}\n",
                force=True
            )

        self.total_reward += reward
        return reward

    def report_episode(self, battle):
        """Prints the final summary of the episode to stderr."""
        if self.log_level < LogLevel.PERIODIC or self.total_reward == 0:
            return

        status = "UNKNOWN"
        our_alive = 0
        opp_alive = 0
        turns = 0
        if battle:
            if battle.won: status = "WIN"
            elif battle.lost: status = "LOSS"
            elif battle.finished: status = "FINISHED"
            
            our_alive = len([p for p in battle.team.values() if not p.fainted])
            opp_alive = len([p for p in battle.opponent_team.values() if not p.fainted])
            turns = battle.turn

        msg = (f"\n🏁 Episode Finished | Reward: {self.total_reward:6.2f} | Status: {status:4} | "
               f"Mon: {our_alive} vs {opp_alive} | Turns: {turns:3} | "
               f"Attacks: {self.attack_count:2} | Sw(Vol): {self.switch_count:2} | Sw(For): {self.forced_switch_count:2}\n")
        
        self.episode_logger.log(msg, force=False)
