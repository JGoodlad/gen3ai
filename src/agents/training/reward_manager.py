from typing import Optional
import numpy as np
from utils.logging.rate_limiter import RateLimitedLogger
from utils.logging.levels import LogLevel
from poke_env.data import GenData
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from agents.training.battle_context import BattleContext, TurnDelta

FAINTED_VALUE = 2.0
HP_VALUE = 2.0
VICTORY_VALUE = 30.0
STALL_TAX_START_TURN = 125
STALL_TAX_DENOMINATOR = 30.0
STRUGGLE_LOOP_TAX = -0.5
STRUGGLE_LOOP_THRESHOLD = 3

SWITCH_BASE_BONUS = 0.5   # flat per-voluntary-switch bonus; consistent throughout the game
STATUS_BONUS = 0.3        # reward for inflicting status; penalty for receiving
ROAR_BONUS = 0.2          # reward for Roar when spikes on opp side or opp had positive boosts
SE_SWITCH_BONUS = 0.2     # reward for switching in a mon with a SE damaging move vs opp active
SLEEP_SWAP_BONUS = 0.25   # reward for rotating a sleeping mon out; penalty for rotating one in
SPIKES_LAYER_BONUS = 0.5  # per layer added to opponent's side (credit assignment bridge)
SPIKES_WASTE_PENALTY = -0.2  # wasted turn using Spikes when 3 layers already up
FAILED_ROAR_PENALTY = -0.2   # Roar used but opponent didn't switch (no target / already phazed out)
FUTILE_ATTACK_PENALTY = -0.05  # attacking move used but opponent net gained HP (Leftovers > damage)
MATCHUP_PENALTY = -0.15   # per turn we stay in while opp has a revealed SE move vs us


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
        self.last_switch_turn = -1
        self.logger = RateLimitedLogger(interval_seconds=1.0)
        self.episode_logger = RateLimitedLogger(interval_seconds=5.0)
        self._last_switched_from = "NULL"   # species we last voluntarily switched away from
        self._last_action_idx = -1
        self._last_reward_metadata = {}
        self._consecutive_struggle = 0
        self.struggle_turns = 0
        self._prev_opp_boosts: dict = {}    # opp active boosts after last turn (for Roar check)
        self._prev_opp_spikes: int = 0      # opp spikes layers after last turn (for Spikes bonus)
        self._prev_opp_se_threat: bool = False  # did opp have a revealed SE move vs us last turn
        self._prev_our_statused = 0
        self._prev_opp_statused = 0
        self._last_switch_was_roared = False
        self._type_chart = GenData.from_gen(3).type_chart

    def reset(self):
        self.switch_count = 0
        self.forced_switch_count = 0
        self.attack_count = 0
        self.total_reward = 0.0
        self.last_switch_turn = -1
        self._last_switched_from = "NULL"
        self._last_action_idx = -1
        self._last_reward_metadata = {}
        self._consecutive_struggle = 0
        self.struggle_turns = 0
        self._prev_opp_boosts = {}
        self._prev_opp_spikes = 0
        self._prev_opp_se_threat = False
        self._prev_our_statused = 0
        self._prev_opp_statused = 0
        self._last_switch_was_roared = False

    def record_action(self, ctx: BattleContext, action: int) -> None:
        """
        Records the action the model chose for this turn.
        Called before the turn is processed, using the context the model saw.

        Switch detection uses the action index and ctx.phase directly so the
        subsidy is credited in the SAME turn as the switch, not the next one.
        """
        self._pending_subsidy = 0.0
        self._last_switch_was_roared = False

        repetition_tax = 0.0
        bouncing_tax = 0.0
        struggle_loop_tax = 0.0

        if action >= 6:
            # Attack or Struggle
            self.attack_count += 1

            if action == self._last_action_idx and action != -1:
                repetition_tax = -0.02
                self._pending_subsidy += repetition_tax

            if action == 10:  # struggle — forced by server when all PP depleted
                self.struggle_turns += 1
                self._consecutive_struggle += 1
                if self._consecutive_struggle >= STRUGGLE_LOOP_THRESHOLD:
                    struggle_loop_tax = STRUGGLE_LOOP_TAX
                    self._pending_subsidy += struggle_loop_tax
            else:
                self._consecutive_struggle = 0

            self._last_reward_metadata = {
                "type": "ATTACK",
                "repetition_tax": repetition_tax,
                "struggle_loop_tax": struggle_loop_tax,
            }
        else:
            # Switch (action 0-5 = team slot index)
            self._consecutive_struggle = 0
            is_forced = ctx.phase == "forced_switch"
            active_norm = str(ctx.our_active).upper()
            has_live_active = active_norm not in ("NONE", "NULL", "NONE_P1", "NONE_P2")

            if is_forced and has_live_active:
                # Roar/Whirlwind: mon is alive but phazed out — no subsidy, skip bonuses
                self._last_switch_was_roared = True
                self.forced_switch_count += 1
                self._last_switched_from = ctx.our_active
                self._last_reward_metadata = {"type": "FORCED_ROAR"}

            elif is_forced:
                # Post-faint replacement — no subsidy
                self.forced_switch_count += 1
                self._last_reward_metadata = {"type": "FORCED_FAINT"}

            else:
                # Voluntary switch — look up the target species via slot map
                slot_to_species = {v: k for k, v in ctx.our_slot_map.items()}
                target_species = slot_to_species.get(action)

                if (target_species and
                        target_species == self._last_switched_from and
                        self._last_switched_from not in ("NULL", "NONE")):
                    bouncing_tax = -0.15
                    self._pending_subsidy += bouncing_tax

                spam_mult = 1.0 if (ctx.turn - self.last_switch_turn) > 1 else 0.0
                subsidy = SWITCH_BASE_BONUS * spam_mult
                self._pending_subsidy += subsidy

                self.switch_count += 1
                self.last_switch_turn = ctx.turn
                self._last_switched_from = ctx.our_active  # what we're switching away from
                self._last_reward_metadata = {
                    "type": "VOLUNTARY",
                    "spam_mult": spam_mult,
                    "repetition_tax": 0.0,
                    "bouncing_tax": bouncing_tax,
                    "subsidy": subsidy,
                }

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
        """Reward Roar when it forces a switch AND spikes are up or opp had positive boosts.
        Penalise Roar when it fails to force any switch at all (wasted turn)."""
        if delta.our_move_id != "roar":
            return 0.0
        if delta.opp_switch_to is None:
            return FAILED_ROAR_PENALTY
        has_spikes = battle.opponent_side_conditions.get(SideCondition.SPIKES, 0) > 0
        had_boosts = any(v > 0 for v in self._prev_opp_boosts.values())
        return ROAR_BONUS if (has_spikes or had_boosts) else 0.0

    def _compute_se_switch_bonus(self, delta: TurnDelta, battle) -> float:
        """Reward switching in a mon that threatens the opponent with a SE move.

        First checks revealed moves for confirmed SE; if none are revealed yet,
        falls back to checking whether any of our mon's own types are SE vs the
        opponent (a reliable proxy for STAB moves in Gen 3 OU).
        """
        if delta.our_switch_to is None:
            return 0.0
        our_mon = battle.active_pokemon
        opp_mon = battle.opponent_active_pokemon
        if not our_mon or not opp_mon:
            return 0.0

        # Confirmed SE via revealed move
        for move in our_mon.moves.values():
            if move.base_power <= 0:
                continue
            mult = move.type.damage_multiplier(
                opp_mon.type_1, opp_mon.type_2, type_chart=self._type_chart
            )
            if mult >= 2.0:
                return SE_SWITCH_BONUS

        # Fallback: STAB type advantage (fires when no moves have been revealed yet)
        our_types = [our_mon.type_1]
        if our_mon.type_2:
            our_types.append(our_mon.type_2)
        for t in our_types:
            mult = t.damage_multiplier(
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

        # Sleep-swap signals.
        # Bonus for rotating a sleeping mon OUT: only on voluntary switches — if the mon
        # fainted there's no preservation value, the opponent can just sleep another mon.
        # Penalty for rotating a sleeping mon IN: applies to voluntary + post-faint
        # (you still chose the replacement), but not Roar/Whirlwind.
        if delta.our_switch_to is not None:
            is_voluntary = self._last_reward_metadata.get("type") == "VOLUNTARY"
            if is_voluntary:
                for mon in battle.team.values():
                    if mon.species == delta.our_prev_active:
                        if mon.status == Status.SLP:
                            reward += SLEEP_SWAP_BONUS
                        break
            if not self._last_switch_was_roared and our_mon and our_mon.status == Status.SLP:
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

    def _compute_matchup_penalty(self, delta: TurnDelta) -> float:
        """Per-turn penalty for staying in while the opp had a revealed SE move vs us last turn.
        Uses last-turn's snapshot so we only penalise for threats known at decision time."""
        if delta.our_switch_to is not None:
            return 0.0  # we switched out — no staying-in penalty
        return MATCHUP_PENALTY if self._prev_opp_se_threat else 0.0

    def _update_opp_se_threat(self, battle) -> None:
        """Snapshot whether opp active has a revealed SE move vs our active, for next turn."""
        our_mon = battle.active_pokemon
        opp_mon = battle.opponent_active_pokemon
        if not our_mon or not opp_mon:
            self._prev_opp_se_threat = False
            return
        for move in opp_mon.moves.values():
            if move.base_power > 0:
                mult = move.type.damage_multiplier(
                    our_mon.type_1, our_mon.type_2, type_chart=self._type_chart
                )
                if mult >= 2.0:
                    self._prev_opp_se_threat = True
                    return
        self._prev_opp_se_threat = False

    def _compute_spikes_bonus(self, delta: TurnDelta, battle) -> float:
        """Reward each new spike layer added; penalise wasting a turn at layer cap."""
        curr = battle.opponent_side_conditions.get(SideCondition.SPIKES, 0)
        new_layers = curr - self._prev_opp_spikes
        self._prev_opp_spikes = curr
        if new_layers > 0:
            return new_layers * SPIKES_LAYER_BONUS
        if delta.our_move_id == "spikes" and curr == 3:
            return SPIKES_WASTE_PENALTY
        return 0.0

    def _compute_futile_attack_penalty(self, delta: TurnDelta, battle) -> float:
        """Penalise attacking moves where the opponent's total HP went up or stayed even
        (Leftovers healed as much or more than we dealt). Skips status moves, switches,
        and cases where we failed to act or the opponent used Rest."""
        if delta.our_move_id is None:
            return 0.0  # we switched
        if delta.our_failed_to_move:
            return 0.0  # paralysis / sleep — not our fault
        if delta.opp_switch_to is not None:
            return 0.0  # they switched; HP delta is noisy (fresh mon entering)
        if delta.opp_move_id == "rest":
            return 0.0  # opponent used Rest; large self-heal is expected
        move = battle.active_pokemon.moves.get(delta.our_move_id) if battle.active_pokemon else None
        if move is None or move.base_power == 0:
            return 0.0  # status or utility move — handled by other signals
        # Net HP sum across all opp slots: bench mons don't change between turns in Gen 3,
        # so the sum is dominated by the active slot. >= 0 means we made no net progress.
        if delta.opp_hp_delta.sum() >= 0:
            return FUTILE_ATTACK_PENALTY
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

        # Defensive pivot bonus — skip if we were roared/whirlwinded out (not our choice)
        if not self._last_switch_was_roared:
            pivot_bonus = self._compute_pivot_bonus(delta, battle)
            if pivot_bonus > 0:
                reward += pivot_bonus
                self._last_reward_metadata["pivot_bonus"] = pivot_bonus

        # Roar bonus (forces switch when spikes are up or opp was boosted); penalty if Roar failed
        roar_bonus = self._compute_roar_bonus(delta, battle)
        reward += roar_bonus

        # Futile attack penalty — attacking move that didn't overcome opponent's Leftovers healing
        futile_penalty = self._compute_futile_attack_penalty(delta, battle)
        reward += futile_penalty

        # Spikes setup bonus / waste penalty
        spikes_bonus = self._compute_spikes_bonus(delta, battle)
        reward += spikes_bonus

        # Matchup disadvantage penalty (known SE threat at decision time, and we stayed in)
        matchup_penalty = self._compute_matchup_penalty(delta)
        reward += matchup_penalty

        # SE switch-in bonus — skip if we were roared/whirlwinded out
        if not self._last_switch_was_roared:
            se_switch_bonus = self._compute_se_switch_bonus(delta, battle)
            reward += se_switch_bonus

        # Status reward (+/- on inflict/cure); sleep-swap skipped if roared out
        status_reward = self._compute_status_reward(delta, battle)
        reward += status_reward

        # Update end-of-turn snapshots for next turn's checks
        opp_mon = battle.opponent_active_pokemon
        self._prev_opp_boosts = dict(opp_mon.boosts) if opp_mon else {}
        self._update_opp_se_threat(battle)

        # Switch subsidy from record_action
        if hasattr(self, "_pending_subsidy"):
            subsidy_val = self._pending_subsidy
            reward += subsidy_val
            del self._pending_subsidy
        else:
            subsidy_val = 0.0

        # Progressive stall tax: starts at turn 125, ramps to -4.2/turn at turn 250
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
                    print(f"       Spam mult: {m['spam_mult']:.1f}")
                    print(f"       Taxes: Repetition:{m['repetition_tax']:.2f} | Bouncing:{m['bouncing_tax']:.2f}")
                    if m.get("pivot_bonus"):
                        print(f"       Pivot Bonus: +{m['pivot_bonus']:.2f}")
                    print(f"       Final Subsidy: {m['subsidy']:.4f}")
                elif m.get("type") == "ATTACK" and (m.get("repetition_tax", 0) != 0 or m.get("struggle_loop_tax", 0) != 0):
                    print(f"    🔍 [DEEP TRACE] Type: ATTACK | Repetition Tax: {m['repetition_tax']:.2f} | Struggle Loop Tax: {m['struggle_loop_tax']:.2f}")
                elif m.get("type") == "FORCED_FAINT":
                    print(f"    🔍 [DEEP TRACE] Type: FORCED SWITCH (post-faint, no subsidy)")
                elif m.get("type") == "FORCED_ROAR":
                    print(f"    🔍 [DEEP TRACE] Type: FORCED SWITCH (roar/whirlwind, no bonuses)")

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
