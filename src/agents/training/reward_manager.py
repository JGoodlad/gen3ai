from dataclasses import dataclass, fields
from typing import ClassVar, Optional
import numpy as np
from utils.logging.rate_limiter import RateLimitedLogger
from utils.logging.levels import LogLevel
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from agents.training.battle_context import BattleContext, TurnDelta
from agents.gen3_mechanics import (
    INVULNERABLE_MOVES as _INVULNERABLE_MOVES,
    is_status_move_immune as _is_status_move_immune,
    effective_multiplier as _effective_multiplier_fn,
)

@dataclass
class RewardBreakdown:
    """Per-component reward breakdown for a single turn. Stored on Gen3RewardManager
    as _last_breakdown after each process_turn_reward() call. Zero fields are omitted
    from to_dict() so the JSON stays compact."""

    # Base outcome
    hp_ours: float = 0.0           # our HP delta * HP_VALUE (negative = damage taken)
    hp_opp: float = 0.0            # opp HP delta * HP_VALUE (positive = damage dealt)
    faint_ours: float = 0.0
    faint_opp: float = 0.0
    win_loss: float = 0.0
    explosion: float = 0.0         # bonus/penalty for opponent self-KO via Explosion/Selfdestruct

    # Attack signals
    roar: float = 0.0
    futile_attack: float = 0.0

    # Field control
    spikes: float = 0.0

    # Positional
    matchup_penalty: float = 0.0

    # Switch: subsidy (set by record_action before the turn)
    switch_base: float = 0.0       # flat per-voluntary-switch subsidy
    switch_bouncing_tax: float = 0.0  # penalty for immediately switching back
    repetition_tax: float = 0.0    # same attack repeated consecutively
    struggle_tax: float = 0.0      # struggle loop penalty

    # Switch: pivot signals (what the opponent did on our switch turn)
    pivot_protect: float = 0.0     # opponent used Protect/Detect/Endure
    pivot_status: float = 0.0      # opponent's status move was type-immune on our switch-in
    pivot_damage: float = 0.0      # opponent's damaging move hit our switch-in less than old mon

    # Switch: offensive threat
    se_switch: float = 0.0         # our switch-in has a SE move vs opponent active

    # Switch: sleep rotation
    sleep_out: float = 0.0         # rotated a sleeping mon to bench
    sleep_in: float = 0.0          # sent in a sleeping mon (penalty)

    # Status signals
    status: float = 0.0

    # Progressive stall tax
    stall_tax: float = 0.0

    # Groups ordered by how frequently they produce non-zero values.
    # Each group's fields are listed in the order they should appear in the string.
    _GROUPS: ClassVar[tuple] = (
        ("base",   ("hp_ours", "hp_opp", "faint_ours", "faint_opp", "win_loss", "explosion")),
        ("attack", ("roar", "futile_attack", "repetition_tax", "struggle_tax")),
        ("switch", ("switch_base", "switch_bouncing_tax", "pivot_protect", "pivot_status",
                    "pivot_damage", "se_switch", "sleep_out", "sleep_in")),
        ("field",  ("spikes", "matchup_penalty", "status", "stall_tax")),
    )

    @property
    def total(self) -> float:
        return sum(getattr(self, f.name) for f in fields(self))

    def to_dict(self) -> dict:
        """Grouped, compact JSON dict.

        Each category (base/attack/switch/field) becomes a single string of
        'key=±value' pairs for non-zero fields. Empty categories are omitted.
        'total' is always present.

        Example:
            {'total': 0.06, 'base': 'hp_ours=-0.64 hp_opp=+0.20',
             'switch': 'switch_base=+0.50 se_switch=+0.20 pivot_damage=+0.10'}
        """
        result: dict = {"total": round(self.total, 4)}
        for group_name, group_fields in self._GROUPS:
            parts = []
            for fname in group_fields:
                v = getattr(self, fname)
                if v != 0.0:
                    parts.append(f"{fname}={v:+.4g}")
            if parts:
                result[group_name] = " ".join(parts)
        return result


FAINTED_VALUE = 2.0
HP_VALUE = 2.0
VICTORY_VALUE = 30.0
STALL_TAX_START_TURN = 125
STALL_TAX_PER_TURN = 0.1
STRUGGLE_LOOP_TAX = -0.5
STRUGGLE_LOOP_THRESHOLD = 3

SWITCH_BASE_BONUS = 0.5        # flat per-voluntary-switch bonus
STATUS_BONUS = 0.3             # reward for inflicting status; penalty for receiving
ROAR_BONUS = 0.2               # reward for Roar when spikes on opp side or opp had positive boosts
SE_SWITCH_BONUS = 0.2          # reward for switching in a mon with a SE damaging move vs opp active
SLEEP_SWAP_BONUS = 0.25        # reward for rotating a sleeping mon out; penalty for rotating one in
SPIKES_LAYER_BONUS = 0.5       # per layer added to opponent's side (credit assignment bridge)
SPIKES_WASTE_PENALTY = -0.2    # wasted turn using Spikes when 3 layers already up
FAILED_ROAR_PENALTY = -0.2     # Roar used but opponent didn't switch
FUTILE_ATTACK_PENALTY = -0.05  # attacking move used but opponent net gained HP (Leftovers > damage)
MATCHUP_PENALTY = -0.15        # per turn we stay in while opp has a revealed SE move vs us
PROTECT_SWITCH_BONUS = 0.10    # opponent used Protect/Detect/Endure on our switch turn
STATUS_IMMUNE_SWITCH_BONUS = 0.10  # our switch-in was immune to their status move


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
            if _effective_multiplier_fn(move.type, opp_mon) >= 2.0:
                return SE_SWITCH_BONUS

        # Fallback: STAB type advantage (fires when no moves have been revealed yet)
        our_types = [our_mon.type_1]
        if our_mon.type_2:
            our_types.append(our_mon.type_2)
        for t in our_types:
            if _effective_multiplier_fn(t, opp_mon) >= 2.0:
                return SE_SWITCH_BONUS

        return 0.0

    def _compute_status_reward(self, delta: TurnDelta, battle) -> float:
        """One-time reward when the statused-mon count changes on either side."""
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
        return (d_opp - d_our) * STATUS_BONUS

    # =========================================================
    # SWITCH REWARDS
    #
    # All switch-specific signals funnel through
    # _compute_all_switch_bonuses, which dispatches to one
    # focused sub-function per outcome type.  The subsidy set
    # by record_action() is applied separately at the end of
    # process_turn_reward (it runs before the turn, not after).
    # =========================================================

    def _compute_pivot_bonus(self, delta: TurnDelta, battle) -> tuple[float, float, float]:
        """Return (protect_bonus, status_bonus, damage_bonus) for this switch turn."""
        opp_move_id = delta.opp_move_id
        if delta.opp_switch_to is not None or opp_move_id is None:
            return (0.0, 0.0, 0.0)

        if opp_move_id in _INVULNERABLE_MOVES:
            return (self._pivot_protect_bonus(), 0.0, 0.0)

        opp_mon = battle.opponent_active_pokemon
        if not opp_mon:
            return (0.0, 0.0, 0.0)
        opp_move = opp_mon.moves.get(opp_move_id)
        if opp_move is None:
            return (0.0, 0.0, 0.0)

        if opp_move.base_power == 0:
            return (0.0, self._pivot_status_bonus(opp_move_id, battle), 0.0)
        return (0.0, 0.0, self._pivot_damage_bonus(opp_move, delta, battle))

    def _pivot_protect_bonus(self) -> float:
        """Opponent used Protect/Detect/Endure — we repositioned for free."""
        return PROTECT_SWITCH_BONUS

    def _pivot_status_bonus(self, opp_move_id: str, battle) -> float:
        """Opponent used a status move our switch-in was immune to (type or already statused)."""
        new_mon = battle.active_pokemon
        if not new_mon:
            return 0.0
        if _is_status_move_immune(opp_move_id, new_mon):
            return STATUS_IMMUNE_SWITCH_BONUS
        return 0.0

    def _pivot_damage_bonus(self, opp_move, delta: TurnDelta, battle) -> float:
        """Opponent used a damaging move — bonus if it hit our new mon less than the old one.

        Signal A: comparison of actual type effectiveness vs old mon vs new mon.
        The HP delta in compute_base_reward already penalises the raw damage taken,
        so this signal focuses purely on whether the switch improved the matchup.
        """
        new_mon = battle.active_pokemon
        if not new_mon:
            return 0.0
        prev_mon = next(
            (m for m in battle.team.values() if m.species == delta.our_prev_active), None
        )
        if prev_mon is None:
            return 0.0
        mult_vs_new = self._effective_multiplier(opp_move.type, new_mon)
        mult_vs_old = self._effective_multiplier(opp_move.type, prev_mon)
        if mult_vs_new < mult_vs_old:
            return 0.15 if mult_vs_new == 0 else 0.10
        return 0.0

    def _compute_sleep_out_bonus(self, delta: TurnDelta, battle) -> float:
        """Reward rotating a sleeping mon to the bench on a voluntary switch.
        Preserving a sleeping mon's PP/position has strategic value; post-faint
        replacements don't qualify since there's no choice involved."""
        if self._last_reward_metadata.get("type") != "VOLUNTARY":
            return 0.0
        for mon in battle.team.values():
            if mon.species == delta.our_prev_active:
                return SLEEP_SWAP_BONUS if mon.status == Status.SLP else 0.0
        return 0.0

    def _compute_sleep_in_penalty(self, delta: TurnDelta, battle) -> float:
        """Penalise sending in a sleeping mon — it can't act and wastes a slot.
        Applies to voluntary switches and post-faint replacements; skipped for roar
        since the phazer chose our slot, not us."""
        if self._last_switch_was_roared:
            return 0.0
        our_mon = battle.active_pokemon
        if our_mon and our_mon.status == Status.SLP:
            return -SLEEP_SWAP_BONUS
        return 0.0

    def _compute_matchup_penalty(self, delta: TurnDelta) -> float:
        """Per-turn penalty for staying in while the opp had a revealed SE move vs us last turn.
        Uses last-turn's snapshot so we only penalise for threats known at decision time."""
        if delta.our_switch_to is not None:
            return 0.0  # we switched out — no staying-in penalty
        return MATCHUP_PENALTY if self._prev_opp_se_threat else 0.0

    def _effective_multiplier(self, move_type, mon) -> float:
        return _effective_multiplier_fn(move_type, mon)

    def _update_opp_se_threat(self, battle) -> None:
        """Snapshot whether opp active has a revealed SE move vs our active, for next turn."""
        our_mon = battle.active_pokemon
        opp_mon = battle.opponent_active_pokemon
        if not our_mon or not opp_mon:
            self._prev_opp_se_threat = False
            return
        for move in opp_mon.moves.values():
            if move.base_power > 0:
                mult = self._effective_multiplier(move.type, our_mon)
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
        """Computes the full reward for a completed turn from the TurnDelta.

        Builds a RewardBreakdown with every named component; stores it on
        self._last_breakdown so callers (e.g. BattleRecorder) can inspect the
        per-signal contributions without touching the battle object.
        """
        bd = RewardBreakdown()

        # --- Base ---
        bd.hp_ours = float(delta.our_hp_delta.sum()) * HP_VALUE
        bd.hp_opp = -float(delta.opp_hp_delta.sum()) * HP_VALUE
        bd.faint_ours = -FAINTED_VALUE if delta.we_fainted else 0.0
        bd.faint_opp = FAINTED_VALUE if delta.opp_fainted else 0.0
        if battle.won:
            bd.win_loss = VICTORY_VALUE
        elif battle.lost or battle.finished:
            bd.win_loss = -VICTORY_VALUE

        base_reward = bd.hp_ours + bd.hp_opp + bd.faint_ours + bd.faint_opp + bd.win_loss

        # --- Explosion / self-destruct ---
        # Only checks the mon active this turn; mutual KOs in Gen 3 are almost
        # exclusively explosion, so this is a reliable proxy.
        if delta.opp_fainted:
            for mon in battle.opponent_team.values():
                if mon.species == delta.opp_prev_active:
                    move_ids = {m.id for m in mon.moves.values()}
                    if move_ids & {"explosion", "selfdestruct"}:
                        bd.explosion = -3.0 if delta.we_fainted else 2.0
                    break

        # --- Attack signals ---
        bd.roar = self._compute_roar_bonus(delta, battle)
        bd.futile_attack = self._compute_futile_attack_penalty(delta, battle)

        # --- Field control ---
        bd.spikes = self._compute_spikes_bonus(delta, battle)

        # --- Positional: penalty for staying in against a known threat ---
        bd.matchup_penalty = self._compute_matchup_penalty(delta)

        # --- Switch rewards (see SWITCH_REWARDS.md for full breakdown) ---
        # Pivot, SE, and sleep-out are skipped when phazed — roar removes our
        # choice, so those signals don't apply.
        if delta.our_switch_to is not None:
            if not self._last_switch_was_roared:
                bd.pivot_protect, bd.pivot_status, bd.pivot_damage = self._compute_pivot_bonus(delta, battle)
                bd.se_switch = self._compute_se_switch_bonus(delta, battle)
                bd.sleep_out = self._compute_sleep_out_bonus(delta, battle)
            bd.sleep_in = self._compute_sleep_in_penalty(delta, battle)

        # --- Status signals ---
        bd.status = self._compute_status_reward(delta, battle)

        # --- Subsidy / taxes (set by record_action before the turn) ---
        # Consume _pending_subsidy to keep the existing handoff contract, then
        # store the individual components in the breakdown from _last_reward_metadata.
        if hasattr(self, "_pending_subsidy"):
            del self._pending_subsidy
        meta = self._last_reward_metadata
        if meta.get("type") == "VOLUNTARY":
            bd.switch_base = meta.get("subsidy", 0.0)
            bd.switch_bouncing_tax = meta.get("bouncing_tax", 0.0)
        elif meta.get("type") == "ATTACK":
            bd.repetition_tax = meta.get("repetition_tax", 0.0)
            bd.struggle_tax = meta.get("struggle_loop_tax", 0.0)

        # --- Flat stall tax: -0.1/turn after turn 125 ---
        if battle.turn > STALL_TAX_START_TURN:
            bd.stall_tax = -STALL_TAX_PER_TURN

        # Update end-of-turn snapshots for next turn's checks
        opp_mon = battle.opponent_active_pokemon
        self._prev_opp_boosts = dict(opp_mon.boosts) if opp_mon else {}
        self._update_opp_se_threat(battle)

        self._last_breakdown = bd
        reward = bd.total
        self.total_reward += reward

        if self.log_level >= LogLevel.DETAILED and self.logger.should_log():
            subsidy_val = bd.switch_base + bd.switch_bouncing_tax + bd.repetition_tax + bd.struggle_tax
            self.logger.log(
                f"  [REWARD] Turn {battle.turn} | Base: {base_reward:+.4f} | Subsidy: {subsidy_val:+.2f} | Won: {battle.won}\n",
                force=True
            )
            if self.log_level >= LogLevel.DEBUG:
                if meta.get("type") == "VOLUNTARY":
                    print(f"    🔍 [DEEP TRACE] Type: VOLUNTARY SWITCH")
                    print(f"       Spam mult: {meta['spam_mult']:.1f}")
                    print(f"       Taxes: Repetition:{meta['repetition_tax']:.2f} | Bouncing:{meta['bouncing_tax']:.2f}")
                    print(f"       Final Subsidy: {meta['subsidy']:.4f}")
                elif meta.get("type") == "ATTACK" and (bd.repetition_tax != 0 or bd.struggle_tax != 0):
                    print(f"    🔍 [DEEP TRACE] Type: ATTACK | Repetition Tax: {bd.repetition_tax:.2f} | Struggle Loop Tax: {bd.struggle_tax:.2f}")
                elif meta.get("type") == "FORCED_FAINT":
                    print(f"    🔍 [DEEP TRACE] Type: FORCED SWITCH (post-faint, no subsidy)")
                elif meta.get("type") == "FORCED_ROAR":
                    print(f"    🔍 [DEEP TRACE] Type: FORCED SWITCH (roar/whirlwind, no bonuses)")

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
