from dataclasses import dataclass, fields
from typing import ClassVar, Optional
import numpy as np
from utils.logging.rate_limiter import RateLimitedLogger
from utils.logging.levels import LogLevel
from poke_env.battle.side_condition import SideCondition
from agents.enums import Status
from agents.training.battle_snapshot import BattleContext
from agents.training.turn_delta import TurnDelta
from agents import gen3_movedex as _movedex
from agents.gen3_mechanics import (
    INVULNERABLE_MOVES as _INVULNERABLE_MOVES,
    is_status_move_immune as _is_status_move_immune,
    effective_multiplier as _effective_multiplier_fn,
    effective_multiplier_by_types as _effective_multiplier_by_types_fn,
    STATUS_MOVE_IMMUNITY as _STATUS_MOVE_IMMUNITY,
)
from agents.enums import PokemonType as _PokemonType


def _ptype(name) -> "Optional[_PokemonType]":
    """LiveView type-id string (e.g. ``'fire'``) -> ``PokemonType`` enum — the primitive
    the mechanics helpers key on. ``None`` passes through."""
    return _PokemonType[name.upper()] if name else None


def _status_enum(name) -> "Optional[Status]":
    """LiveView status-id string (e.g. ``'slp'``) -> ``Status`` enum. ``None`` passes
    through. Only ``FRZ`` actually changes an effectiveness result (Flash Fire), but we
    convert faithfully so the LiveView path is byte-identical to the raw-battle path."""
    return Status[name.upper()] if name else None

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
    explosion_block: float = 0.0   # Ghost immune or Protect blocked opponent Explosion
    finishing_blow: float = 0.0    # damaging move secured the KO

    # Attack signals
    roar: float = 0.0
    futile_attack: float = 0.0
    futile_setup: float = 0.0      # setup move used at stat cap (+6 or -6)
    setup_low_hp: float = 0.0      # setup move chosen below 40% HP (penalty)
    boost_utilized: float = 0.0    # attacked while holding active stat boosts
    status_wasted: float = 0.0     # status-inflicting move had no effect

    # Field control
    spikes: float = 0.0

    # Positional
    matchup_penalty: float = 0.0
    dead_matchup_tax: float = 0.0  # escalating penalty for staying in a 0×-only matchup

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
    escape_threat_switch: float = 0.0  # switched out while opp had a revealed SE threat vs us

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
        ("base",   ("hp_ours", "hp_opp", "faint_ours", "faint_opp", "win_loss", "explosion", "explosion_block", "finishing_blow")),
        ("attack", ("roar", "futile_attack", "futile_setup", "setup_low_hp",
                    "boost_utilized", "status_wasted", "repetition_tax", "struggle_tax")),
        ("switch", ("switch_base", "switch_bouncing_tax", "escape_threat_switch",
                    "pivot_protect", "pivot_status",
                    "pivot_damage", "se_switch", "sleep_out", "sleep_in")),
        ("field",  ("spikes", "matchup_penalty", "dead_matchup_tax", "status", "stall_tax")),
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


FAINT_BASE = 0.5        # minimum faint penalty/reward at 0% HP
FAINT_HP_SCALE = 2.0   # scales faint cost/reward linearly with HP at time of faint
HP_VALUE = 2.0
VICTORY_VALUE = 30.0
FINISHING_BLOW_BONUS = 0.5   # extra bonus for KO'ing with a damaging move

# Progressive stall tax: starts EARLY (turn 60, was 125) and RAMPS so a passive
# 130-190 turn loop is strictly dominated by making progress. Per-turn cost grows
# linearly with how far past the start turn we are, clamped at STALL_TAX_MAX so a
# single turn can't dwarf a faint/HP swing. Cumulative over a very long game stays
# well under VICTORY_VALUE=30 (≈10 over a 190-turn game) because the ramp is gentle
# and most games end before the start turn.
STALL_TAX_START_TURN = 60
STALL_TAX_PER_TURN = 0.05      # base rate; multiplied by the ramp fraction below
STALL_TAX_RAMP_TURNS = 20      # turns-past-start over which the rate ramps up by 1×
STALL_TAX_MAX = 0.5            # per-turn clamp on the ramped stall tax
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
FUTILE_IMMUNE_PENALTY = -0.5   # flat per-turn penalty for attacking into a type immunity
                               # (our_effectiveness == 0.0). The ESCALATION on a repeated
                               # immune attack comes from the zero-effect repetition tax below.
ESCAPE_THREAT_BONUS = 0.25     # voluntarily switching out while opp has a revealed SE threat vs us
MATCHUP_PENALTY = -0.15        # per turn we stay in while opp has a revealed SE move vs us
PROTECT_SWITCH_BONUS = 0.10    # opponent used Protect/Detect/Endure on our switch turn
STATUS_IMMUNE_SWITCH_BONUS = 0.10  # our switch-in was immune to their status move

FUTILE_SETUP_PENALTY = -0.3
SETUP_LOW_HP_THRESHOLD = 0.40      # HP fraction below which setup is penalised
SETUP_LOW_HP_MAX_PENALTY = -0.10   # penalty at 0% HP; scales linearly to 0 at threshold
STATUS_WASTED_PENALTY = -0.3
BOOST_UTILIZED_SCALE = 0.03        # reward = boost_stage * scale * damage_dealt
EXPLOSION_BLOCK_BONUS = 1.0        # Ghost immune or Protect blocks opponent Explosion

# Repetition tax escalation — LINEAR and UNCAPPED (clamped only by the floor).
# A 12-30 turn spam must be catastrophic, not a rounding error, so the cost grows
# every consecutive turn instead of plateauing after the 4th repeat. The tax for the
# n-th consecutive repeat is max(-STEP * n, FLOOR). A "no-op" repeat (the move did
# nothing productive — no damage, no boost gained, no status landed, no hazard added)
# uses the much steeper ZERO_EFFECT step so capped setup (Calm Mind past +6), capped
# hazards (Spikes at 3), redundant status, Protect/Wish/Recover loops, and immune
# attacks all bite hard and fast. A legitimately-productive repeat (still dealing
# damage or still gaining a boost) only pays the gentle normal step.
REPETITION_STEP = 0.03              # normal productive-attack repeat, per consecutive turn
REPETITION_ZERO_EFFECT_STEP = 0.15  # no-op / immune / capped repeat — bites hard
REPETITION_TAX_FLOOR = -3.0         # per-turn clamp so one turn can't dwarf win/loss

# Switch-bouncing tax — ESCALATING (was a flat -0.15). A→B→A→B oscillation dodges the
# move-repetition tax because the action index alternates, so it needs its own
# escalating counter. The n-th consecutive bounce costs max(STEP * n, FLOOR).
BOUNCING_TAX_STEP = -0.15
BOUNCING_TAX_FLOOR = -2.0

# Dead-matchup tax — fires when the active Pokémon has NO damaging move with >0×
# effectiveness vs the opponent's active mon and we DID NOT switch out. This is the
# "trapped, must pivot" signal: the matchup re-ranks moves but can't lift switches
# above the collapsed "stay in and click" prior, so we make staying strictly worse
# than pivoting and escalate it every turn we refuse to leave.
DEAD_MATCHUP_TAX_STEP = -0.10
DEAD_MATCHUP_TAX_FLOOR = -2.0

BOOST_MOVES: frozenset[str] = frozenset({
    "calmmind", "dragondance", "swordsdance", "nastyplot",
    "agility", "rockpolish", "bulkup", "cosmicpower",
    "acidarmor", "barrier", "irondefense", "amnesia",
    "growth", "meditate", "sharpen", "doubleteam", "minimize",
    "harden", "withdraw", "defensecurl", "stockpile",
})

STATUS_INFLICTING_MOVES: frozenset[str] = frozenset({
    "toxic", "poisonpowder",
    "thunderwave",
    "willowisp",
    "sleeppowder", "hypnosis", "spore", "lovelykiss", "sing",
})


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
        # When True, current-board facts (spikes layers, team status counts,
        # opp-active boosts, terminal flags) are re-sourced from battle.live_view()
        # (the vetted LiveView read-model) instead of the raw poke-env battle.
        # Flip to False for the equivalence harness
        # (reward_resourcing_equivalence_fuzz_test.py), which diffs the two
        # sources turn-by-turn to prove the re-sourcing is value-neutral.
        self._read_live = True
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
        self._consecutive_attack_repeats: int = 0
        self._consecutive_bounces: int = 0          # A→B→A→B oscillation depth
        self._consecutive_dead_matchup_stays: int = 0  # turns stuck in a 0×-only matchup
        self._last_attack_had_effect: bool = True
        self._our_active_hp_before: float = 1.0
        self._opp_active_hp_before: float = 1.0
        self._our_boosts_before: np.ndarray = np.zeros(7, dtype=np.int8)
        self._last_opp_seen_by: dict[str, str] = {}
        # maps our_species → opp_species when this mon last switched in (voluntary, not roared)

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
        self._consecutive_attack_repeats = 0
        self._consecutive_bounces = 0
        self._consecutive_dead_matchup_stays = 0
        self._last_attack_had_effect = True
        self._our_active_hp_before = 1.0
        self._opp_active_hp_before = 1.0
        self._our_boosts_before = np.zeros(7, dtype=np.int8)
        self._last_opp_seen_by = {}

    def record_action(self, ctx: BattleContext, action: int) -> None:
        """
        Records the action the model chose for this turn.
        Called before the turn is processed, using the context the model saw.

        Switch detection uses the action index and ctx.phase directly so the
        subsidy is credited in the SAME turn as the switch, not the next one.
        """
        self._pending_subsidy = 0.0
        self._last_switch_was_roared = False

        # Snapshot HP and boosts at decision time for use in process_turn_reward
        our_slot = ctx.our_slot_map.get(ctx.our_active, 0)
        opp_slot = ctx.opp_slot_map.get(ctx.opp_active, 0)
        active_norm = str(ctx.our_active).upper()
        has_live_active = active_norm not in ("NONE", "NULL", "NONE_P1", "NONE_P2")
        self._our_active_hp_before = float(ctx.our_hp[our_slot]) if has_live_active else 0.0
        self._opp_active_hp_before = float(ctx.opp_hp[opp_slot])
        self._our_boosts_before = ctx.our_boosts.copy()

        repetition_tax = 0.0
        bouncing_tax = 0.0
        struggle_loop_tax = 0.0

        if action >= 6:
            # Attack or Struggle
            self.attack_count += 1

            # A move breaks any switch-oscillation streak.
            self._consecutive_bounces = 0

            if action == self._last_action_idx and action != -1:
                self._consecutive_attack_repeats += 1
                n = self._consecutive_attack_repeats   # 1, 2, 3, ... (uncapped)
                step = (REPETITION_ZERO_EFFECT_STEP if not self._last_attack_had_effect
                        else REPETITION_STEP)
                repetition_tax = max(-step * n, REPETITION_TAX_FLOOR)
                self._pending_subsidy += repetition_tax
            else:
                self._consecutive_attack_repeats = 0

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
            self._consecutive_attack_repeats = 0
            is_forced = ctx.phase == "forced_switch"

            if is_forced and has_live_active:
                # Roar/Whirlwind: mon is alive but phazed out — no subsidy, skip bonuses.
                # Phazing isn't voluntary oscillation, so it doesn't count as a bounce.
                self._consecutive_bounces = 0
                self._last_switch_was_roared = True
                self.forced_switch_count += 1
                self._last_switched_from = ctx.our_active
                self._last_reward_metadata = {"type": "FORCED_ROAR"}

            elif is_forced:
                # Post-faint replacement — no subsidy, not an oscillation
                self._consecutive_bounces = 0
                self.forced_switch_count += 1
                self._last_reward_metadata = {"type": "FORCED_FAINT"}

            else:
                # Voluntary switch — INTENT only. The subsidy / bounce-tax / escape
                # bonus and their cross-turn counters (switch_count, last_switch_turn,
                # _last_switched_from, _consecutive_bounces) are NOT computed here: a
                # pressed switch can silently fail to execute (poke-env "gap=0": the
                # opponent faints on hazard entry the same window and our switch never
                # realizes), and crediting/perturbing on the PRESS mis-attributes a
                # reward to a turn that shows no switch. We stash the intent and settle
                # it in _apply_switch_outcome() once delta.our_switch_to confirms the
                # switch actually happened.
                slot_to_species = {v: k for k, v in ctx.our_slot_map.items()}
                target_species = slot_to_species.get(action)
                self._last_reward_metadata = {
                    "type": "VOLUNTARY",
                    "decision_turn": ctx.turn,
                    "switch_from": ctx.our_active,   # mon we're leaving
                    "target_species": target_species,
                }

        self._last_action_idx = action

    def _apply_switch_outcome(self, delta: TurnDelta, bd: "RewardBreakdown") -> None:
        """Settle the voluntary-switch subsidy at OUTCOME time.

        ``record_action`` recorded that the model PRESSED a voluntary switch (intent);
        this fires only when the event-sourced ``delta`` confirms a switch actually
        happened (``our_switch_to is not None``). On a poke-env "gap=0" no-op — the
        press didn't execute because the opponent fainted on hazard entry the same
        window — we credit nothing and leave the bounce/spam counters untouched, so a
        phantom switch neither earns +0.5 nor perturbs future oscillation magnitudes.

        For a realized switch the math + counter mutations are identical to the old
        record_action path (same inputs: the intent-captured target / decision-turn /
        switched-from mon, the persistent counters, and the pre-turn opp-SE-threat
        snapshot which `_update_opp_se_threat` hasn't refreshed yet this turn)."""
        meta = self._last_reward_metadata
        if delta.our_switch_to is None:
            return  # pressed a switch but it never executed — no credit, no counters

        target = meta.get("target_species")
        decision_turn = meta.get("decision_turn", -1)

        # Bounce: switched straight back to the mon we just left. Escalates with the
        # oscillation depth so A↔B for 10-30 turns is prohibitive, not a rounding error.
        if (target and target == self._last_switched_from
                and self._last_switched_from not in ("NULL", "NONE")):
            self._consecutive_bounces += 1
            bd.switch_bouncing_tax = max(
                BOUNCING_TAX_STEP * self._consecutive_bounces, BOUNCING_TAX_FLOOR)
        else:
            self._consecutive_bounces = 0

        spam_mult = 1.0 if (decision_turn - self.last_switch_turn) > 1 else 0.0
        bd.switch_base = SWITCH_BASE_BONUS * spam_mult

        if self._prev_opp_se_threat:
            bd.escape_threat_switch = ESCAPE_THREAT_BONUS

        self.switch_count += 1
        self.last_switch_turn = decision_turn
        self._last_switched_from = meta.get("switch_from", "NULL")

    def compute_base_reward(self, delta: TurnDelta, battle) -> float:
        """
        Translates TurnDelta into a base scalar reward.
        HP deltas and faint events are already captured in the delta; win/loss
        still requires the battle object since it is a terminal signal.
        """
        reward = float(delta.our_hp_delta.sum()) * HP_VALUE
        reward -= float(delta.opp_hp_delta.sum()) * HP_VALUE

        if delta.we_fainted:
            reward -= (FAINT_BASE + FAINT_HP_SCALE * self._our_active_hp_before)
        if delta.opp_fainted:
            reward += (FAINT_BASE + FAINT_HP_SCALE * self._opp_active_hp_before)

        if battle.won:
            reward += VICTORY_VALUE
        elif battle.lost:
            reward -= VICTORY_VALUE
        elif battle.finished:
            reward -= VICTORY_VALUE  # tie/stall treated as a loss

        return reward

    def _compute_roar_bonus(self, delta: TurnDelta, battle, live) -> float:
        """Reward Roar when it forces a switch AND spikes are up or opp had positive boosts.
        Penalise Roar when it fails to force any switch at all (wasted turn)."""
        if delta.our_move_id != "roar":
            return 0.0
        if delta.opp_switch_to is None:
            return FAILED_ROAR_PENALTY
        has_spikes = self._opp_spikes(battle, live) > 0
        had_boosts = any(v > 0 for v in self._prev_opp_boosts.values())
        return ROAR_BONUS if (has_spikes or had_boosts) else 0.0

    def _compute_se_switch_bonus(self, delta: TurnDelta, battle, live=None) -> float:
        """Reward switching in a mon that threatens the opponent with a SE move.

        First checks revealed moves for confirmed SE; if none are revealed yet,
        falls back to checking whether any of our mon's own types are SE vs the
        opponent (a reliable proxy for STAB moves in Gen 3 OU).

        Current-board reads are dual-pathed: the LiveView ``LivePokemon`` (move
        power/type via ``gen3_movedex``, effectiveness via the mechanics primitive)
        when ``live`` is set, else the raw battle. The equivalence harness proves the
        two are value-identical.
        """
        if delta.our_switch_to is None:
            return 0.0
        if live is not None:
            our_mon = live.ours.active
            opp_mon = live.opp.active
        else:
            our_mon = battle.active_pokemon
            opp_mon = battle.opponent_active_pokemon
        if not our_mon or not opp_mon:
            return 0.0

        # Only award on voluntary switches; forced post-faint replacements don't count
        if self._last_reward_metadata.get("type") != "VOLUNTARY":
            return 0.0
        # Opponent must be alive at switch-in
        if opp_mon.fainted:
            return 0.0

        # Gate: only fire if the opponent has switched since this mon was last in.
        # Same matchup without opponent switching = bonus already spent for this matchup.
        our_species = our_mon.species
        opp_species = opp_mon.species
        if self._last_opp_seen_by.get(our_species) == opp_species:
            return 0.0

        if live is not None:
            # Confirmed SE via revealed move
            for mid in our_mon.move_ids:
                md = _movedex.get(mid)
                if md is None or md.base_power <= 0:
                    continue
                if self._live_eff_mult(md.type, opp_mon) >= 2.0:
                    return SE_SWITCH_BONUS
            # Fallback: STAB type advantage (no moves revealed yet)
            for t in our_mon.types:
                if self._live_eff_mult(_ptype(t), opp_mon) >= 2.0:
                    return SE_SWITCH_BONUS
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

    def _compute_status_reward(self, delta: TurnDelta, battle, live) -> tuple[float, int]:
        """One-time reward when the statused-mon count changes on either side.
        Returns (reward, d_opp) where d_opp is the delta in opponent statused count."""
        our_statused, opp_statused = self._statused_counts(battle, live)
        d_our = our_statused - self._prev_our_statused
        d_opp = opp_statused - self._prev_opp_statused
        self._prev_our_statused = our_statused
        self._prev_opp_statused = opp_statused
        return (d_opp - d_our) * STATUS_BONUS, d_opp

    # =========================================================
    # SWITCH REWARDS
    #
    # All switch-specific signals funnel through
    # _compute_all_switch_bonuses, which dispatches to one
    # focused sub-function per outcome type.  The subsidy set
    # by record_action() is applied separately at the end of
    # process_turn_reward (it runs before the turn, not after).
    # =========================================================

    def _compute_pivot_bonus(self, delta: TurnDelta, battle, live=None) -> tuple[float, float, float]:
        """Return (protect_bonus, status_bonus, damage_bonus) for this switch turn.

        Uses `delta.opp_resolved_move_id` — protocol-truth attribution when a
        damaging event is set, falling back to the inferred `delta.opp_move_id`
        for non-damaging moves (status, Roar, etc.). Avoids the stale-last_move
        class of bug that bit HP attribution.

        The opp-move presence + power read is dual-pathed: the LiveView active mon's
        revealed moves + ``gen3_movedex`` when ``live`` is set, else the raw battle.
        """
        opp_move_id = delta.opp_resolved_move_id
        if delta.opp_switch_to is not None or opp_move_id is None:
            return (0.0, 0.0, 0.0)

        if opp_move_id in _INVULNERABLE_MOVES:
            return (self._pivot_protect_bonus(), 0.0, 0.0)

        if live is not None:
            opp_mon = live.opp.active
            if not opp_mon or opp_move_id not in opp_mon.move_ids:
                return (0.0, 0.0, 0.0)
            md = _movedex.get(opp_move_id)
            base_power = md.base_power if md is not None else 0
        else:
            opp_mon = battle.opponent_active_pokemon
            if not opp_mon:
                return (0.0, 0.0, 0.0)
            opp_move = opp_mon.moves.get(opp_move_id)
            if opp_move is None:
                return (0.0, 0.0, 0.0)
            base_power = opp_move.base_power

        if base_power == 0:
            return (0.0, self._pivot_status_bonus(opp_move_id, battle, live), 0.0)
        return (0.0, 0.0, self._pivot_damage_bonus(opp_move_id, delta, battle, live))

    def _pivot_protect_bonus(self) -> float:
        """Opponent used Protect/Detect/Endure — we repositioned for free."""
        return PROTECT_SWITCH_BONUS

    def _pivot_status_bonus(self, opp_move_id: str, battle, live=None) -> float:
        """Opponent used a status move our switch-in was immune to (type or already statused)."""
        if live is not None:
            new_mon = live.ours.active
            if not new_mon:
                return 0.0
            if self._live_status_move_immune(opp_move_id, new_mon):
                return STATUS_IMMUNE_SWITCH_BONUS
            return 0.0
        new_mon = battle.active_pokemon
        if not new_mon:
            return 0.0
        if _is_status_move_immune(opp_move_id, new_mon):
            return STATUS_IMMUNE_SWITCH_BONUS
        return 0.0

    def _pivot_damage_bonus(self, opp_move_id, delta: TurnDelta, battle, live=None) -> float:
        """Opponent used a damaging move — bonus if it hit our new mon less than the old one.

        Signal A: comparison of actual type effectiveness vs old mon vs new mon.
        The HP delta in compute_base_reward already penalises the raw damage taken,
        so this signal focuses purely on whether the switch improved the matchup.

        `mult_vs_new` prefers `delta.opp_damaging_event.effectiveness` when the
        event's target matches our switch-in — it's the protocol-confirmed bucket
        (no drift if our local mechanics disagree with Showdown's). Falls back
        to a local recompute when the event is missing or hit a different
        target (e.g. opp damaged prev_active before switch-in was on field).
        `mult_vs_old` always recomputes — the protocol can't tell us what the
        multiplier *would have been* if we hadn't switched.

        Dual-pathed: the move type comes from ``gen3_movedex`` and the prev/new mons
        from the LiveView when ``live`` is set, else from the raw battle.
        """
        if live is not None:
            new_mon = live.ours.active
            if not new_mon:
                return 0.0
            prev_mon = live.ours.get(delta.our_prev_active)
            if prev_mon is None:
                return 0.0
            md = _movedex.get(opp_move_id)
            move_type = md.type if md is not None else None
            opp_event = delta.opp_damaging_event
            if opp_event is not None and opp_event.target_species == new_mon.species:
                mult_vs_new = opp_event.effectiveness
            else:
                mult_vs_new = self._live_eff_mult(move_type, new_mon)
            mult_vs_old = self._live_eff_mult(move_type, prev_mon)
            if mult_vs_new < mult_vs_old:
                return 0.15 if mult_vs_new == 0 else 0.10
            return 0.0

        new_mon = battle.active_pokemon
        if not new_mon:
            return 0.0
        prev_mon = next(
            (m for m in battle.team.values() if m.species == delta.our_prev_active), None
        )
        if prev_mon is None:
            return 0.0
        opp_move = battle.opponent_active_pokemon.moves.get(opp_move_id)
        opp_event = delta.opp_damaging_event
        if opp_event is not None and opp_event.target_species == new_mon.species:
            mult_vs_new = opp_event.effectiveness
        else:
            mult_vs_new = self._effective_multiplier(opp_move.type, new_mon)
        mult_vs_old = self._effective_multiplier(opp_move.type, prev_mon)
        if mult_vs_new < mult_vs_old:
            return 0.15 if mult_vs_new == 0 else 0.10
        return 0.0

    def _compute_sleep_out_bonus(self, delta: TurnDelta, battle, live=None) -> float:
        """Reward rotating a sleeping mon to the bench on a voluntary switch.
        Preserving a sleeping mon's PP/position has strategic value; post-faint
        replacements don't qualify since there's no choice involved."""
        if self._last_reward_metadata.get("type") != "VOLUNTARY":
            return 0.0
        if live is not None:
            prev = live.ours.get(delta.our_prev_active)
            if prev is None:
                return 0.0
            return SLEEP_SWAP_BONUS if prev.status == "slp" else 0.0
        for mon in battle.team.values():
            if mon.species == delta.our_prev_active:
                return SLEEP_SWAP_BONUS if mon.status == Status.SLP else 0.0
        return 0.0

    def _compute_sleep_in_penalty(self, delta: TurnDelta, battle, live=None) -> float:
        """Penalise sending in a sleeping mon — it can't act and wastes a slot.
        Applies to voluntary switches and post-faint replacements; skipped for roar
        since the phazer chose our slot, not us."""
        if self._last_switch_was_roared:
            return 0.0
        if live is not None:
            our_mon = live.ours.active
            if our_mon and our_mon.status == "slp":
                return -SLEEP_SWAP_BONUS
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

    def _compute_dead_matchup_tax(self, delta: TurnDelta, battle, live=None) -> float:
        """Escalating penalty for refusing to pivot out of a 0×-only matchup.

        Fires when EVERY damaging move our active Pokémon has does 0× to the
        opponent's active mon (e.g. an Electric attacker staring at a Ground type,
        a Normal attacker into a Ghost) and we chose to stay in rather than switch.
        The per-turn cost grows with how many consecutive turns we've stayed
        trapped, so a switch — which resets the counter to zero — strictly
        dominates clicking another useless move.

        Resets (and charges nothing) whenever we switch, faint, lack a live
        opponent, or have at least one >0× damaging move. Skips forced-switch
        slots entirely (we had no move choice there). Requires at least one
        revealed damaging move to judge — our own active mon's full moveset is
        populated from the request, so this is reliable for the trainee's mon.
        """
        if delta.our_switch_to is not None or delta.we_fainted:
            self._consecutive_dead_matchup_stays = 0
            return 0.0
        if delta.phase_is_forced_switch:
            return 0.0

        if live is not None:
            our_mon = live.ours.active
            opp_mon = live.opp.active
            if not our_mon or not opp_mon or opp_mon.fainted:
                self._consecutive_dead_matchup_stays = 0
                return 0.0
            damaging = [
                md for md in (_movedex.get(mid) for mid in our_mon.move_ids)
                if md is not None and md.base_power > 0
            ]
            if not damaging:
                self._consecutive_dead_matchup_stays = 0
                return 0.0
            best_mult = max(self._live_eff_mult(md.type, opp_mon) for md in damaging)
        else:
            our_mon = battle.active_pokemon
            opp_mon = battle.opponent_active_pokemon
            if not our_mon or not opp_mon or opp_mon.fainted:
                self._consecutive_dead_matchup_stays = 0
                return 0.0

            damaging = [m for m in our_mon.moves.values() if m.base_power > 0]
            if not damaging:
                self._consecutive_dead_matchup_stays = 0
                return 0.0

            best_mult = max(self._effective_multiplier(m.type, opp_mon) for m in damaging)
        if best_mult > 0.0:
            self._consecutive_dead_matchup_stays = 0
            return 0.0

        # Every damaging option is type-immune and we stayed in — escalate.
        self._consecutive_dead_matchup_stays += 1
        return max(DEAD_MATCHUP_TAX_STEP * self._consecutive_dead_matchup_stays,
                   DEAD_MATCHUP_TAX_FLOOR)

    # =========================================================
    # CURRENT-BOARD ACCESSORS (event-sourced re-sourcing, Step 5)
    #
    # These read current-board facts through the vetted LiveView
    # read-model (built once per turn in process_turn_reward) so
    # the reward manager doesn't reach into the raw poke-env battle
    # for "what is true now". When `live` is None (equivalence
    # harness with _read_live=False) they fall back to the battle —
    # the two paths are value-identical, which the harness proves.
    # =========================================================

    def _opp_spikes(self, battle, live) -> int:
        """Spikes layers on the opponent's side (0-3)."""
        if live is not None:
            return live.opp.side_conditions.get("spikes", 0)
        return battle.opponent_side_conditions.get(SideCondition.SPIKES, 0)

    def _statused_counts(self, battle, live) -> tuple[int, int]:
        """(our, opp) counts of non-fainted mons carrying a status condition."""
        if live is not None:
            our = sum(1 for m in live.ours.mons if m.status is not None and not m.fainted)
            opp = sum(1 for m in live.opp.mons if m.status is not None and not m.fainted)
            return our, opp
        our = sum(
            1 for mon in battle.team.values()
            if mon.status is not None and not mon.fainted
        )
        opp = sum(
            1 for mon in battle.opponent_team.values()
            if mon.status is not None and not mon.fainted
        )
        return our, opp

    def _opp_active_boosts(self, battle, live) -> dict:
        """Opponent active mon's current stat-stage boosts ({} if none on field)."""
        if live is not None:
            return dict(live.opp.active.boosts) if live.opp.active else {}
        opp_mon = battle.opponent_active_pokemon
        return dict(opp_mon.boosts) if opp_mon else {}

    def _terminal(self, battle, live) -> tuple:
        """(won, lost, finished) — terminal battle flags."""
        src = live if live is not None else battle
        return src.won, src.lost, src.finished

    def _effective_multiplier(self, move_type, mon) -> float:
        return _effective_multiplier_fn(move_type, mon)

    def _live_eff_mult(self, move_type, live_mon) -> float:
        """``effective_multiplier`` for a LiveView ``LivePokemon``, via the primitive
        ``effective_multiplier_by_types`` (no poke-env ``Pokemon``). Byte-identical to
        the raw-battle path: the LivePokemon carries the same types/ability/status the
        raw ``mon`` would expose, just in id-string form."""
        types = live_mon.types
        t1 = _ptype(types[0]) if types else None
        t2 = _ptype(types[1]) if len(types) > 1 else None
        return _effective_multiplier_by_types_fn(
            move_type, t1, t2, live_mon.ability, _status_enum(live_mon.status)
        )

    def _live_status_move_immune(self, move_id, live_mon) -> bool:
        """``is_status_move_immune`` for a LiveView ``LivePokemon`` — type immunity to
        the status the move inflicts, or the mon already carrying a status."""
        immune_types = _STATUS_MOVE_IMMUNITY.get(move_id, frozenset())
        mon_types = {_ptype(t) for t in live_mon.types}
        return bool(immune_types & mon_types) or live_mon.status is not None

    def _update_opp_se_threat(self, battle, live=None) -> None:
        """Snapshot whether opp active has a revealed SE move vs our active, for next turn."""
        if live is not None:
            our_mon = live.ours.active
            opp_mon = live.opp.active
            if not our_mon or not opp_mon:
                self._prev_opp_se_threat = False
                return
            for mid in opp_mon.move_ids:
                md = _movedex.get(mid)
                if md is not None and md.base_power > 0:
                    if self._live_eff_mult(md.type, our_mon) >= 2.0:
                        self._prev_opp_se_threat = True
                        return
            self._prev_opp_se_threat = False
            return
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

    def _compute_spikes_bonus(self, delta: TurnDelta, battle, live) -> float:
        """Reward each new spike layer added; penalise wasting a turn at layer cap."""
        curr = self._opp_spikes(battle, live)
        new_layers = curr - self._prev_opp_spikes
        self._prev_opp_spikes = curr
        if new_layers > 0:
            return new_layers * SPIKES_LAYER_BONUS
        if delta.our_move_id == "spikes" and curr == 3:
            return SPIKES_WASTE_PENALTY
        return 0.0

    def _compute_futile_attack_penalty(self, delta: TurnDelta, battle, live=None) -> float:
        """Penalise attacking moves where the opponent's total HP went up or stayed even
        (Leftovers healed as much or more than we dealt). Skips status moves, switches,
        and cases where we failed to act or the opponent used Rest."""
        if delta.our_move_id is None:
            return 0.0  # we switched
        if delta.our_failed_to_move:
            return 0.0  # paralysis / sleep — not our fault
        if delta.opp_switch_to is not None:
            return 0.0  # they switched; HP delta is noisy (fresh mon entering)
        # Rest detection uses opp_resolved_move_id (protocol-truth when an
        # event fired). Raw delta.opp_move_id could be a stale "rest" from
        # an earlier turn after opp switched between snapshots, mis-skipping
        # the penalty on a normal turn where they didn't actually rest.
        if delta.opp_resolved_move_id == "rest":
            return 0.0  # opponent used Rest; large self-heal is expected
        # Damaging-move gate: our move must be a revealed damaging move. Power comes
        # from gen3_movedex (LiveView path) or the raw revealed Move (battle path).
        if live is not None:
            our_mon = live.ours.active
            md = (_movedex.get(delta.our_move_id)
                  if our_mon and delta.our_move_id in our_mon.move_ids else None)
            if md is None or md.base_power == 0:
                return 0.0  # status or utility move — handled by other signals
        else:
            move = battle.active_pokemon.moves.get(delta.our_move_id) if battle.active_pokemon else None
            if move is None or move.base_power == 0:
                return 0.0  # status or utility move — handled by other signals
        # Type immunity: 0 damage by definition — use the harder penalty.
        if delta.our_effectiveness == 0.0:
            return FUTILE_IMMUNE_PENALTY
        # Net HP sum across all opp slots: bench mons don't change between turns in Gen 3,
        # so the sum is dominated by the active slot. >= 0 means we made no net progress.
        if delta.opp_hp_delta.sum() >= 0:
            return FUTILE_ATTACK_PENALTY
        return 0.0

    def _compute_futile_setup_penalty(self, delta: TurnDelta) -> float:
        """Penalise using a stat-boosting move when already at the ±6 cap."""
        if delta.our_move_id not in BOOST_MOVES:
            return 0.0
        if delta.our_failed_to_move or delta.we_fainted:
            return 0.0
        # If no boost stage changed, the move had zero mechanical effect
        if delta.our_boost_delta.sum() == 0:
            return FUTILE_SETUP_PENALTY
        return 0.0

    def _compute_setup_low_hp_penalty(self, delta: TurnDelta) -> float:
        """Penalise choosing a setup move below 40% HP."""
        if delta.our_move_id not in BOOST_MOVES:
            return 0.0
        if delta.our_failed_to_move or delta.we_fainted:
            return 0.0
        hp = self._our_active_hp_before
        if hp >= SETUP_LOW_HP_THRESHOLD:
            return 0.0
        return SETUP_LOW_HP_MAX_PENALTY * (1.0 - hp / SETUP_LOW_HP_THRESHOLD)

    def _compute_status_wasted_penalty(self, delta: TurnDelta, d_opp_statused: int) -> float:
        """Penalise status-inflicting moves that produced no status event."""
        if delta.our_move_id not in STATUS_INFLICTING_MOVES:
            return 0.0
        if delta.our_failed_to_move:
            return 0.0
        if delta.opp_switch_to is not None:
            return 0.0  # opp switched; ambiguous
        if d_opp_statused > 0:
            return 0.0  # status landed — no penalty
        return STATUS_WASTED_PENALTY

    def _compute_boost_utilized(self, delta: TurnDelta, battle, live=None) -> float:
        """Reward attacking moves that leverage active stat boosts."""
        if delta.our_move_id is None or delta.our_switch_to is not None:
            return 0.0
        if delta.our_failed_to_move:
            return 0.0
        if live is not None:
            mon = live.ours.active
            md = (_movedex.get(delta.our_move_id)
                  if mon and delta.our_move_id in mon.move_ids else None)
            if md is None or md.base_power == 0:
                return 0.0
        else:
            mon = battle.active_pokemon
            if not mon:
                return 0.0
            move = mon.moves.get(delta.our_move_id)
            if move is None or move.base_power == 0:
                return 0.0
        # Use the higher of atk (idx 0) or spa (idx 2) boost
        effective_boost = max(int(self._our_boosts_before[0]), int(self._our_boosts_before[2]))
        if effective_boost <= 0:
            return 0.0
        damage_dealt = max(0.0, -float(delta.opp_hp_delta.sum()))
        return effective_boost * BOOST_UTILIZED_SCALE * damage_dealt

    def _compute_finishing_blow_bonus(self, delta: TurnDelta, battle, live=None) -> float:
        """Extra bonus when a damaging move secures the KO."""
        if not delta.opp_fainted:
            return 0.0
        if delta.our_move_id is None or delta.our_switch_to is not None:
            return 0.0
        if delta.our_failed_to_move:
            return 0.0
        if live is not None:
            mon = live.ours.active
            md = (_movedex.get(delta.our_move_id)
                  if mon and delta.our_move_id in mon.move_ids else None)
            if md is None or md.base_power == 0:
                return 0.0
            return FINISHING_BLOW_BONUS
        mon = battle.active_pokemon
        if not mon:
            return 0.0
        move = mon.moves.get(delta.our_move_id)
        if move is None or move.base_power == 0:
            return 0.0
        return FINISHING_BLOW_BONUS

    def process_turn_reward(self, battle, delta: TurnDelta) -> float:
        """Computes the full reward for a completed turn from the TurnDelta.

        Builds a RewardBreakdown with every named component; stores it on
        self._last_breakdown so callers (e.g. BattleRecorder) can inspect the
        per-signal contributions without touching the battle object.
        """
        bd = RewardBreakdown()

        # Current-board facts are read through the vetted LiveView read-model
        # (built once per turn) rather than the raw poke-env battle. `live` is
        # None only in the equivalence harness (_read_live=False), where the
        # accessors fall back to the battle to prove the two are value-identical.
        live = battle.live_view() if self._read_live else None

        # --- Base ---
        bd.hp_ours = float(delta.our_hp_delta.sum()) * HP_VALUE
        bd.hp_opp = -float(delta.opp_hp_delta.sum()) * HP_VALUE
        bd.faint_ours = -(FAINT_BASE + FAINT_HP_SCALE * self._our_active_hp_before) if delta.we_fainted else 0.0
        bd.faint_opp = (FAINT_BASE + FAINT_HP_SCALE * self._opp_active_hp_before) if delta.opp_fainted else 0.0
        won, lost, finished = self._terminal(battle, live)
        if won:
            bd.win_loss = VICTORY_VALUE
        elif lost or finished:
            bd.win_loss = -VICTORY_VALUE

        base_reward = bd.hp_ours + bd.hp_opp + bd.faint_ours + bd.faint_opp + bd.win_loss

        # --- Explosion / self-destruct ---
        # Read the damaging event directly: the protocol's |move|<user>|Explosion|
        # is captured at parse time, before |faint| arrives and the active slot
        # advances to a switch-in. The old `for mon in opponent_team … move_ids &
        # {explosion, selfdestruct}` scan misfires for any opp mon that has
        # Explosion in their revealed moveset — including turns they used
        # something else. The event is per-turn-confirmed attribution.
        opp_event = delta.opp_damaging_event
        if opp_event is not None and opp_event.move_id in ("explosion", "selfdestruct"):
            if not delta.we_fainted:
                # Opponent used Explosion/SD but we survived — strategic win
                bd.explosion = 2.0
                # Extra bonus if we took 0 damage (Ghost immune, Protect, or
                # the event's effectiveness reported 0× directly)
                if delta.our_hp_delta.sum() == 0.0 or opp_event.effectiveness == 0.0:
                    bd.explosion_block = EXPLOSION_BLOCK_BONUS
            # When we_fainted: faint_ours already penalises the loss;
            # don't double-count with an explosion penalty on top.

        # --- Finishing blow ---
        bd.finishing_blow = self._compute_finishing_blow_bonus(delta, battle, live)

        # --- Attack signals ---
        bd.roar = self._compute_roar_bonus(delta, battle, live)
        bd.futile_attack = self._compute_futile_attack_penalty(delta, battle, live)
        bd.futile_setup = self._compute_futile_setup_penalty(delta)
        bd.setup_low_hp = self._compute_setup_low_hp_penalty(delta)
        bd.boost_utilized = self._compute_boost_utilized(delta, battle, live)

        # --- Field control ---
        bd.spikes = self._compute_spikes_bonus(delta, battle, live)

        # --- Positional: penalty for staying in against a known threat ---
        bd.matchup_penalty = self._compute_matchup_penalty(delta)
        # Escalating penalty for refusing to pivot out of a 0×-only matchup.
        bd.dead_matchup_tax = self._compute_dead_matchup_tax(delta, battle, live)

        # --- Switch rewards (see SWITCH_REWARDS.md for full breakdown) ---
        # Pivot, SE, and sleep-out are skipped when phazed — roar removes our
        # choice, so those signals don't apply.
        if delta.our_switch_to is not None:
            if not self._last_switch_was_roared:
                bd.pivot_protect, bd.pivot_status, bd.pivot_damage = self._compute_pivot_bonus(delta, battle, live)
                bd.se_switch = self._compute_se_switch_bonus(delta, battle, live)
                bd.sleep_out = self._compute_sleep_out_bonus(delta, battle, live)
                # Update per-mon opponent tracker for future se_switch gating
                if live is not None:
                    our_mon_in = live.ours.active
                    opp_mon_in = live.opp.active
                else:
                    our_mon_in = battle.active_pokemon
                    opp_mon_in = battle.opponent_active_pokemon
                if our_mon_in and opp_mon_in and not opp_mon_in.fainted:
                    self._last_opp_seen_by[our_mon_in.species] = opp_mon_in.species
            bd.sleep_in = self._compute_sleep_in_penalty(delta, battle, live)

        # --- Status signals ---
        bd.status, _d_opp_statused = self._compute_status_reward(delta, battle, live)
        bd.status_wasted = self._compute_status_wasted_penalty(delta, _d_opp_statused)

        # --- Subsidy / taxes ---
        # Attack taxes are action-keyed (a pressed move/struggle reliably resolves or
        # is |cant|), so record_action computes them. The switch subsidy is OUTCOME-keyed
        # and settled here against delta.our_switch_to (see _apply_switch_outcome).
        if hasattr(self, "_pending_subsidy"):
            del self._pending_subsidy
        meta = self._last_reward_metadata
        if meta.get("type") == "VOLUNTARY":
            self._apply_switch_outcome(delta, bd)
        elif meta.get("type") == "ATTACK":
            bd.repetition_tax = meta.get("repetition_tax", 0.0)
            bd.struggle_tax = meta.get("struggle_loop_tax", 0.0)

        # --- Progressive stall tax: starts at turn 60 and RAMPS ---
        # rate = STALL_TAX_PER_TURN * (turns past start / RAMP_TURNS), clamped at MAX.
        # Gentle near the start so a slightly-long game is barely touched, but a
        # 130-190 turn passive loop accumulates real pressure (≈10 total by turn 190).
        if battle.turn > STALL_TAX_START_TURN:
            ramp = (battle.turn - STALL_TAX_START_TURN) / STALL_TAX_RAMP_TURNS
            bd.stall_tax = -min(STALL_TAX_PER_TURN * ramp, STALL_TAX_MAX)

        # Update end-of-turn snapshots for next turn's checks
        self._prev_opp_boosts = self._opp_active_boosts(battle, live)
        self._update_opp_se_threat(battle, live)

        # Track whether our last move did anything PRODUCTIVE this turn, for the
        # escalating repetition tax. A move counts as effective if it dealt damage,
        # gained us a stat boost, landed a status, or added a hazard layer. Capped
        # setup (no boost change), capped hazards, redundant status, and immune/no-op
        # attacks all flip this to False, routing the next repeat through the steeper
        # ZERO_EFFECT step — that's how capped setup/hazards escalate.
        self._last_attack_had_effect = (
            float(delta.opp_hp_delta.sum()) < 0
            or int(delta.our_boost_delta.sum()) > 0
            or _d_opp_statused > 0
            or bd.spikes > 0
        )

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
                    realized = "realized" if bd.switch_base or bd.switch_bouncing_tax or bd.escape_threat_switch else "no-op (pressed switch didn't execute)"
                    print(f"    🔍 [DEEP TRACE] Type: VOLUNTARY SWITCH ({realized})")
                    print(f"       Base:{bd.switch_base:+.2f} | Bouncing:{bd.switch_bouncing_tax:+.2f} | Escape:{bd.escape_threat_switch:+.2f}")
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
