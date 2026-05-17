from typing import Optional
import numpy as np
from utils.logging.rate_limiter import RateLimitedLogger
from utils.logging.levels import LogLevel
from utils.gen3_utils import SwitchDetection
from poke_env.data import GenData
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from agents.training.battle_context import BattleContext, TurnDelta

FAINTED_VALUE = 2.0
HP_VALUE = 1.0
VICTORY_VALUE = 30.0
STALL_TAX_START_TURN = 200
STALL_TAX_DENOMINATOR = 10.0
STRUGGLE_LOOP_TAX = -0.5
STRUGGLE_LOOP_THRESHOLD = 3

STATUS_BONUS = 0.3        # reward for inflicting status; penalty for receiving
ROAR_BONUS = 0.2          # reward for Roar when spikes on opp side or opp had positive boosts
SE_SWITCH_BONUS = 0.2     # reward for switching in a mon with a SE damaging move vs opp active
SLEEP_SWAP_BONUS = 0.25   # reward for rotating a sleeping mon out; penalty for rotating one in


class Gen3RewardManager:
    """
    Self-contained reward calculator for the Gen 3 RL trainee agent.

    Owns the full reward pipeline:
      - record_action()       — tracks the action choice and computes switch subsidy
      - process_turn_reward() — computes the full turn reward from TurnDelta
      - compute_base_reward() — translates TurnDelta into a base scalar reward

    Satisfies the RewardFunction protocol. Must only be called for the trainee's
    battle — the env gates this at the call site.
    """
    def __init__(self, log_level: LogLevel = LogLevel.QUIET):
        self.log_level = log_level
        self.switch_count = 0
        self.forced_switch_count = 0
        self.attack_count = 0
        self.total_reward = 0.0
        self.remaining_switch_pool = 7.5
        self.last_switch_turn = -1
        self.logger = RateLimitedLogger(interval_seconds=1.0)
        self.episode_logger = RateLimitedLogger(interval_seconds=5.0)
        self._last_active_name = "NULL"
        self._last_opp_active_name = "NULL"
        self._opp_turns_active = 0
        self._prev_active_name = "NULL"
        self._last_action_idx = -1
        self._last_reward_metadata = {}
        self._consecutive_struggle = 0
        self.struggle_turns = 0
        self._prev_opp_boosts: dict = {}    # opp active boosts after last turn (for Roar check)
        self._prev_our_statused = 0
        self._prev_opp_statused = 0
        self._type_chart = GenData.from_gen(3).type_chart

    def reset(self):
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
        self._consecutive_struggle = 0
        self.struggle_turns = 0
        self._prev_opp_boosts = {}
        self._prev_our_statused = 0
        self._prev_opp_statused = 0

    def record_action(self, ctx: BattleContext, action: int) -> None:
        """
        Records the action the model chose for this turn.
        Called before the turn is processed, using the context the model saw.
        """
        if action >= 6:
            self.attack_count += 1

        current_active = ctx.our_active

        has_switched, is_real, is_voluntary = SwitchDetection.get_switch_type(
            self._last_active_name, current_active, ctx.mask
        )

        if ctx.opp_active != self._last_opp_active_name:
            self._last_opp_active_name = ctx.opp_active
            self._opp_turns_active = 0
        else:
            self._opp_turns_active += 1

        self._pending_subsidy = 0.0

        repetition_tax = 0.0
        bouncing_tax = 0.0

        if action == self._last_action_idx and action != -1:
            repetition_tax = -0.02
            self._pending_subsidy += repetition_tax

        struggle_loop_tax = 0.0
        if action == 10:  # struggle — forced by server when all PP depleted
            self.struggle_turns += 1
            self._consecutive_struggle += 1
            if self._consecutive_struggle >= STRUGGLE_LOOP_THRESHOLD:
                struggle_loop_tax = STRUGGLE_LOOP_TAX
                self._pending_subsidy += struggle_loop_tax
        else:
            self._consecutive_struggle = 0

        if is_real:
            if is_voluntary is True:
                if current_active == self._prev_active_name and self._prev_active_name != "NONE":
                    bouncing_tax = -0.15
                    self._pending_subsidy += bouncing_tax

                payout = self.remaining_switch_pool * 0.5
                attack_ratio = self.attack_count / max(1, ctx.turn)
                ratio_mult = 1.0 if attack_ratio >= 0.33 else 0.5
                spam_mult = 1.0 if (ctx.turn - self.last_switch_turn) > 1 else 0.0
                turn_decay = max(0.0, 1.0 - ctx.turn / 250.0)
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
                    "subsidy": subsidy,
                }

                self.remaining_switch_pool -= payout
                self.switch_count += 1
                self.last_switch_turn = ctx.turn

            elif is_voluntary is False:
                self.forced_switch_count += 1
                self._last_reward_metadata = {"type": "FORCED"}
        else:
            self._last_reward_metadata = {"type": "ATTACK", "repetition_tax": repetition_tax, "struggle_loop_tax": struggle_loop_tax}

        self._prev_active_name = self._last_active_name
        self._last_active_name = current_active
        self._last_action_idx = action

    def compute_base_reward(self, delta: TurnDelta, battle) -> float:
        """
        Translates TurnDelta into a base scalar reward.
        HP deltas and faint events are already captured in the delta; win/loss
        still requires the battle object since it is a terminal signal.
        """
        reward = float(delta.our_hp_delta.sum()) * HP_VALUE
        reward -= float(delta.opp_hp_delta.sum()) * HP_VALUE

        if delta.we_fainted:
            reward -= FAINTED_VALUE
        if delta.opp_fainted:
            reward += FAINTED_VALUE

        if battle.won:
            reward += VICTORY_VALUE
        elif battle.lost:
            reward -= VICTORY_VALUE
        elif battle.finished:
            reward -= VICTORY_VALUE  # tie/stall treated as a loss

        return reward

    def _compute_roar_bonus(self, delta: TurnDelta, battle) -> float:
        """Reward Roar when it actually forces a switch AND spikes are up or opp had positive boosts."""
        if delta.our_move_id != "roar" or delta.opp_switch_to is None:
            return 0.0
        has_spikes = battle.opponent_side_conditions.get(SideCondition.SPIKES, 0) > 0
        had_boosts = any(v > 0 for v in self._prev_opp_boosts.values())
        return ROAR_BONUS if (has_spikes or had_boosts) else 0.0

    def _compute_se_switch_bonus(self, delta: TurnDelta, battle) -> float:
        """Reward switching in a mon that has at least one SE damaging move vs the opponent."""
        if delta.our_switch_to is None:
            return 0.0
        our_mon = battle.active_pokemon
        opp_mon = battle.opponent_active_pokemon
        if not our_mon or not opp_mon:
            return 0.0
        for move in our_mon.moves.values():
            if move.base_power <= 0:
                continue
            mult = move.type.damage_multiplier(
                opp_mon.type_1, opp_mon.type_2, type_chart=self._type_chart
            )
            if mult >= 2.0:
                return SE_SWITCH_BONUS
        return 0.0

    def _compute_status_reward(self, delta: TurnDelta, battle) -> float:
        """One-time status reward: fires only when the status count changes (inflicted or cured).
        Counts all non-fainted mons on each side, bench included.
        Also rewards rotating a sleeping mon OUT and penalises rotating one IN."""
        our_mon = battle.active_pokemon
        opp_mon = battle.opponent_active_pokemon

        our_statused = sum(
            1 for mon in battle.team.values()
            if mon.status is not None and not mon.fainted
        )
        opp_statused = sum(
            1 for mon in battle.opponent_team.values()
            if mon.status is not None and not mon.fainted
        )
        d_our = our_statused - self._prev_our_statused
        d_opp = opp_statused - self._prev_opp_statused
        self._prev_our_statused = our_statused
        self._prev_opp_statused = opp_statused
        reward = (d_opp - d_our) * STATUS_BONUS

        # Sleep-swap signals: rotating a sleeping mon out is good (preserves it);
        # rotating one in wastes the switch turn.
        if delta.our_switch_to is not None:
            for mon in battle.team.values():
                if mon.species == delta.our_prev_active:
                    if mon.status == Status.SLP:
                        reward += SLEEP_SWAP_BONUS
                    break
            if our_mon and our_mon.status == Status.SLP:
                reward -= SLEEP_SWAP_BONUS

        return reward

    def _compute_pivot_bonus(self, delta: TurnDelta, battle) -> float:
        if not delta.our_switch_to or not delta.our_prev_active:
            return 0.0
        opp_mon = battle.opponent_active_pokemon
        if not opp_mon:
            return 0.0

        opp_types = [opp_mon.type_1]
        if opp_mon.type_2:
            opp_types.append(opp_mon.type_2)

        type_chart = GenData.from_gen(3).type_chart

        def max_threat(species: str) -> float:
            for mon in battle.team.values():
                if mon.species == species:
                    return max(
                        t.damage_multiplier(mon.type_1, mon.type_2, type_chart=type_chart)
                        for t in opp_types
                    )
            return 1.0

        old_threat = max_threat(delta.our_prev_active)
        new_threat = max_threat(delta.our_switch_to)

        if new_threat < old_threat:
            return 0.15 if new_threat == 0 else 0.1
        return 0.0

    def process_turn_reward(self, battle, delta: TurnDelta) -> float:
        """
        Computes the full reward for a completed turn from the TurnDelta.
        """
        base_reward = self.compute_base_reward(delta, battle)
        reward = base_reward

        # Explosion/self-destruct signal.
        # Targets only the mon that was active this turn (delta.opp_prev_active), not
        # all previously-fainted mons. Mutual KOs in Gen 3 are almost exclusively
        # explosion/self-destruct, so this is a reliable proxy until opp_move_id is
        # populated from the battle log.
        if delta.opp_fainted:
            for mon in battle.opponent_team.values():
                if mon.species == delta.opp_prev_active:
                    move_ids = {m.id for m in mon.moves.values()}
                    if move_ids & {"explosion", "selfdestruct"}:
                        if delta.we_fainted:
                            reward -= 3.0  # got blown up — strong penalty
                        else:
                            reward += 2.0  # survived/played around explosion
                    break

        # Defensive pivot bonus
        pivot_bonus = self._compute_pivot_bonus(delta, battle)
        if pivot_bonus > 0:
            reward += pivot_bonus
            self._last_reward_metadata["pivot_bonus"] = pivot_bonus

        # Roar bonus (forces switch when spikes are up or opp was boosted)
        roar_bonus = self._compute_roar_bonus(delta, battle)
        reward += roar_bonus

        # SE switch-in bonus (brings in a mon with a SE damaging move vs opp)
        se_switch_bonus = self._compute_se_switch_bonus(delta, battle)
        reward += se_switch_bonus

        # Symmetric status reward (+/- on inflict/receive)
        status_reward = self._compute_status_reward(delta, battle)
        reward += status_reward

        # Update opp boosts for next turn's Roar check (done after Roar bonus is computed)
        opp_mon = battle.opponent_active_pokemon
        self._prev_opp_boosts = dict(opp_mon.boosts) if opp_mon else {}

        # Switch subsidy from record_action
        if hasattr(self, "_pending_subsidy"):
            subsidy_val = self._pending_subsidy
            reward += subsidy_val
            del self._pending_subsidy
        else:
            subsidy_val = 0.0

        # Progressive stall tax (turns 200-250): ramps from -0.1 to -5.0
        if battle.turn > STALL_TAX_START_TURN:
            reward += -1.0 * (battle.turn - STALL_TAX_START_TURN) / STALL_TAX_DENOMINATOR

        if self.log_level >= LogLevel.DETAILED and self.logger.should_log():
            self.logger.log(
                f"  [REWARD] Turn {battle.turn} | Base: {base_reward:+.4f} | Subsidy: {subsidy_val:+.2f} | Won: {battle.won}\n",
                force=True
            )
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
                elif m.get("type") == "ATTACK" and (m.get("repetition_tax", 0) != 0 or m.get("struggle_loop_tax", 0) != 0):
                    print(f"    🔍 [DEEP TRACE] Type: ATTACK | Repetition Tax: {m['repetition_tax']:.2f} | Struggle Loop Tax: {m['struggle_loop_tax']:.2f}")
                elif m.get("type") == "FORCED":
                    print(f"    🔍 [DEEP TRACE] Type: FORCED SWITCH (No Subsidy)")

        self.total_reward += reward
        return reward

    def report_episode(self, battle):
        if self.log_level < LogLevel.PERIODIC or self.total_reward == 0:
            return

        status = "UNKNOWN"
        our_alive = opp_alive = turns = 0
        if battle:
            if battle.won: status = "WIN"
            elif battle.lost: status = "LOSS"
            elif battle.finished: status = "TIE/STALL"
            our_alive = len([p for p in battle.team.values() if not p.fainted])
            opp_alive = len([p for p in battle.opponent_team.values() if not p.fainted])
            turns = battle.turn

        self.episode_logger.log(
            f"\n🏁 Episode Finished | Reward: {self.total_reward:6.2f} | Status: {status:4} | "
            f"Mon: {our_alive} vs {opp_alive} | Turns: {turns:3} | "
            f"Attacks: {self.attack_count:2} | Sw(Vol): {self.switch_count:2} | Sw(For): {self.forced_switch_count:2} | "
            f"Struggle: {self.struggle_turns:2}\n",
            force=False
        )
