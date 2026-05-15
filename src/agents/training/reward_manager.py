import sys
import time
from typing import Optional
from utils.logging.rate_limiter import RateLimitedLogger
from utils.logging.levels import LogLevel
from utils.gen3_utils import SwitchDetection
from poke_env.data import GenData

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
        self._last_reward_metadata = {}
        self._last_opp_fainted_count = 0
        
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
        self._last_reward_metadata = {}
        self._last_opp_fainted_count = 0

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

        # Track opponent fainted count for explosion detection
        self._last_opp_fainted_count = len([m for m in battle.opponent_team.values() if m.fainted])

        # We only apply the subsidy here so it's tied to the CHOICE, not the turn delta
        self._pending_subsidy = 0.0

        
        # 4. Repetition & Bouncing Penalties
        repetition_tax = 0.0
        bouncing_tax = 0.0
        
        # Penalty for clicking the exact same move/switch slot twice in a row
        if action == self._last_action_idx and action != -1:
            repetition_tax = -0.02
            self._pending_subsidy += repetition_tax
            
        if is_real:
            if is_voluntary is True:
                # Bouncing Penalty: Penalize switching back to the mon we just left
                if current_active == self._prev_active_name and self._prev_active_name != "NONE":
                    bouncing_tax = -0.15
                    self._pending_subsidy += bouncing_tax
                
                payout = self.remaining_switch_pool * 0.5
                attack_ratio = self.attack_count / max(1, battle.turn)
                ratio_mult = 1.0 if attack_ratio >= 0.33 else 0.5
                spam_mult = 1.0 if (battle.turn - self.last_switch_turn) > 1 else 0.0
                turn_decay = max(0.0, 1.0 - battle.turn / 250.0)
                
                # Reactive Bonus: Reward switches more if the opponent just arrived (within 2 turns)
                # This fixes "aimless" switching against a stable opponent.
                reactive_mult = 1.0 if self._opp_turns_active <= 2 else 0.25
                
                subsidy = min(payout * ratio_mult * spam_mult * turn_decay * reactive_mult, 1.0)
                self._pending_subsidy += subsidy
                
                self._last_reward_metadata = {
                    "type": "VOLUNTARY",
                    "payout": payout,
                    "ratio_mult": ratio_mult,
                    "spam_mult": spam_mult,
                    "reactive_mult": reactive_mult,
                    "repetition_tax": repetition_tax,
                    "bouncing_tax": bouncing_tax,
                    "subsidy": subsidy
                }
                
                self.remaining_switch_pool -= payout
                self.switch_count += 1
                self.last_switch_turn = battle.turn

                # Defensive Pivot Bonus
                pivot_bonus = 0.0
                if battle.active_pokemon and battle.opponent_active_pokemon and action < 6:
                    old_mon = battle.active_pokemon
                    team_list = list(battle.team.values())
                    if action < len(team_list):
                        new_mon = team_list[action]
                        opp_mon = battle.opponent_active_pokemon
                        
                        # Estimate threat using STAB types of the opponent
                        opp_types = [opp_mon.type_1]
                        if opp_mon.type_2: opp_types.append(opp_mon.type_2)
                        
                        type_chart = GenData.from_gen(3).type_chart
                        
                        def get_max_mult(mon):
                            return max(t.damage_multiplier(mon.type_1, mon.type_2, type_chart=type_chart) for t in opp_types)
                        
                        old_threat = get_max_mult(old_mon)
                        new_threat = get_max_mult(new_mon)
                        
                        if new_threat < old_threat:
                            pivot_bonus = 0.15 if new_threat == 0 else 0.1
                            self._pending_subsidy += pivot_bonus
                            self._last_reward_metadata["pivot_bonus"] = pivot_bonus
            elif is_voluntary is False:
                # This is a forced switch (Replacement or Roar/Whirlwind)
                self.forced_switch_count += 1
                self._last_reward_metadata = {"type": "FORCED"}
        else:
            self._last_reward_metadata = {"type": "ATTACK", "repetition_tax": repetition_tax}

        self._prev_active_name = self._last_active_name
        self._last_active_name = current_active
        self._last_action_idx = action

    def process_turn_reward(self, battle, base_reward: float) -> float:
        """
        Calculates the final reward for a turn, including subsidies and logging.
        """
        reward = base_reward
        
        # Explosion Mitigation Detection
        # Logic: If an opponent mon faints, and it was a 'nuclear' threat, 
        # check if we survived it.
        explosion_bonus = 0.0
        explosion_penalty = 0.0
        
        current_opp_fainted_count = len([m for m in battle.opponent_team.values() if m.fainted])
        if current_opp_fainted_count > self._last_opp_fainted_count:
            # Opponent fainted! 
            # We look at revealed moves of the opponent team to find the one that just fainted
            # This is an approximation since poke_env doesn't easily tell us which one just fainted
            # in the middle of a turn, but usually there's only one.
            for mon in battle.opponent_team.values():
                if mon.fainted and "explosion" in [m.id for m in mon.moves.values()]:
                     # Found a fainted mon with explosion!
                     if battle.active_pokemon and not battle.active_pokemon.fainted:
                         # We survived!
                         explosion_bonus = 1.0
                         # print(f"  ✨ [STRATEGY] Explosion Mitigated! Reward: +{explosion_bonus}")
                     elif battle.active_pokemon and battle.active_pokemon.fainted:
                         # We fainted to it
                         explosion_penalty = -0.5
                         # print(f"  💣 [STRATEGY] Fainted to Explosion. Penalty: {explosion_penalty}")
                     break
        
        reward += explosion_bonus
        reward += explosion_penalty
        
        # Apply any subsidy calculated in the last record_action call
        if hasattr(self, "_pending_subsidy"):
            reward += self._pending_subsidy
            subsidy_val = self._pending_subsidy
            del self._pending_subsidy
        else:
            subsidy_val = 0.0
        
        # Update trackers for next turn
        self._last_opp_fainted_count = current_opp_fainted_count

        is_log_turn = self.log_level >= LogLevel.DETAILED and self.logger.should_log()
        
        if is_log_turn:
            hp = battle.active_pokemon.current_hp_fraction if battle.active_pokemon else 0.0
            self.logger.log(
                f"  [REWARD] Turn {battle.turn} | Base: {base_reward:+.4f} | Subsidy: {subsidy_val:+.2f} | Won: {battle.won}\n",
                force=True
            )
            
            # Deep Diagnostic Trace
            if self.log_level >= LogLevel.DEBUG:
                m = self._last_reward_metadata
                if m.get("type") == "VOLUNTARY":
                    print(f"    🔍 [DEEP TRACE] Type: VOLUNTARY SWITCH")
                    print(f"       Pool Remaining: {self.remaining_switch_pool + m['payout']:.2f} -> Payout: {m['payout']:.2f}")
                    print(f"       Multipliers: Ratio:{m['ratio_mult']:.1f} | Spam:{m['spam_mult']:.1f} | Reactive:{m['reactive_mult']:.2f}")
                    print(f"       Taxes: Repetition:{m['repetition_tax']:.2f} | Bouncing:{m['bouncing_tax']:.2f}")
                    if m.get("pivot_bonus"):
                        print(f"       Pivot Bonus: +{m['pivot_bonus']:.2f}")
                    print(f"       Final Subsidy: {m['subsidy']:.4f}")
                elif m.get("type") == "ATTACK" and m.get("repetition_tax", 0) != 0:
                    print(f"    🔍 [DEEP TRACE] Type: ATTACK | Repetition Tax: {m['repetition_tax']:.2f}")
                elif m.get("type") == "FORCED":
                    print(f"    🔍 [DEEP TRACE] Type: FORCED SWITCH (No Subsidy)")

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
            elif battle.finished: status = "TIE/STALL"
            
            our_alive = len([p for p in battle.team.values() if not p.fainted])
            opp_alive = len([p for p in battle.opponent_team.values() if not p.fainted])
            turns = battle.turn

        msg = (f"\n🏁 Episode Finished | Reward: {self.total_reward:6.2f} | Status: {status:4} | "
               f"Mon: {our_alive} vs {opp_alive} | Turns: {turns:3} | "
               f"Attacks: {self.attack_count:2} | Sw(Vol): {self.switch_count:2} | Sw(For): {self.forced_switch_count:2}\n")
        
        self.episode_logger.log(msg, force=False)
