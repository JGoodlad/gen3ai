"""
Fuzz E2E test for reward function component invariants.

Runs real battles against the live Showdown server and validates every
named field on `RewardBreakdown` against a per-signal invariant. The
existing tests (`reward_manager_test.py`) cover individual signals with
mocked battles; this fuzz exercises them all against the real protocol,
which catches integration bugs (stale snapshots, mis-attribution from
poke-env quirks, signal interactions) that mock-based tests miss.

Architecture:
  - One `Gen3RewardManager` per battle — all per-battle state is keyed by
    `battle.battle_tag`. The previous single-valued fields silently destroyed
    each other's state under `max_concurrent_battles>1`, masking signals and
    producing fake "all clean" results (faint counters at 0 across 10 battles).
  - One `_BattleState` snapshot per turn that captures *exactly* what the
    reward manager observed at action time (HP, boosts, opp SE threat).
    This is what the invariants compare against.
  - Each invariant is a `(field_name, predicate, message)` triple. Adding
    a new one is two lines.

Coverage — one invariant per non-zero-able field on `RewardBreakdown`:
  base       : hp_ours, hp_opp, faint_ours, faint_opp, win_loss,
               explosion, explosion_block, finishing_blow
  attack     : roar (success + failed), futile_attack (general + immune),
               futile_setup, setup_low_hp, boost_utilized, status_wasted
  field      : spikes (layer add + waste)
  positional : matchup_penalty
  switch     : pivot_protect, pivot_status, pivot_damage, se_switch,
               escape_threat_switch, sleep_out, sleep_in
  meta       : NaN/inf rejection across every field

Run directly (requires: npm run showdown):
    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/training/reward_invariants_e2e_test.py [n_battles]
"""

import asyncio
import math
import os
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field, fields as dc_fields
from typing import Optional

import numpy as np

from poke_env import AccountConfiguration
from poke_env.battle.side_condition import SideCondition
from poke_env.battle.status import Status
from poke_env.player import RandomPlayer
from poke_env.player.player import Player
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.action.mapper import Gen3ActionMapper
from agents.action.mask_generator import Gen3ActionMasker
from agents.battle.gen3_battle import Gen3Battle
from agents.training.battle_snapshot import BattleContext
from agents.training.turn_delta import TurnDelta
from agents.training.reward_manager import (
    BOOST_MOVES,
    EXPLOSION_BLOCK_BONUS,
    FAILED_ROAR_PENALTY,
    MAT_HP_WEIGHT,
    MAT_ALIVE_WEIGHT,
    FINISHING_BLOW_BONUS,
    FUTILE_ATTACK_PENALTY,
    FUTILE_IMMUNE_PENALTY,
    HP_VALUE,
    MATCHUP_PENALTY,
    PROTECT_SWITCH_BONUS,
    REPETITION_TAX_FLOOR,
    ROAR_BONUS,
    SE_SWITCH_BONUS,
    SETUP_LOW_HP_THRESHOLD,
    SLEEP_SWAP_BONUS,
    SPIKES_LAYER_BONUS,
    SPIKES_WASTE_PENALTY,
    STALL_TAX_MAX,
    STALL_TAX_START_TURN,
    STATUS_BONUS,
    STATUS_IMMUNE_SWITCH_BONUS,
    STRUGGLE_LOOP_TAX,
    SWITCH_BASE_BONUS,
    VICTORY_VALUE,
    Gen3RewardManager,
    RewardBreakdown,
)
from agents.training.slot_registry import SlotRegistry
from utils.teambuilder import Gen3Teambuilder

BATTLE_FORMAT = "gen3ou"

# Team chosen to exercise every reward category in random play:
#   - Calm Mind / Dragon Dance: setup (futile_setup, setup_low_hp, boost_utilized)
#   - Roar / Whirlwind: phaze signal + matchup_penalty interactions
#   - Spikes / Rapid Spin: field-control signal
#   - Will-O-Wisp / Toxic / Thunder Wave: status (incl. status_wasted on immune)
#   - Explosion: explosion + explosion_block (Ghost immunity)
#   - Rest / Sleep Talk: sleep_in / sleep_out paths
#   - Mixed Type coverage: pivot_damage, se_switch matchups
MIXED_TEAM = """\
Suicune @ Leftovers
Ability: Pressure
EVs: 240 HP / 244 Def / 24 Spe
Bold Nature
- Surf
- Ice Beam
- Calm Mind
- Rest

Tyranitar @ Leftovers
Ability: Sand Stream
EVs: 252 HP / 40 Atk / 216 SpD
Careful Nature
- Rock Slide
- Earthquake
- Crunch
- Dragon Dance

Gengar @ Leftovers
Ability: Levitate
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Shadow Ball
- Thunderbolt
- Will-O-Wisp
- Explosion

Skarmory @ Leftovers
Ability: Keen Eye
EVs: 252 HP / 4 Def / 252 Spe
Jolly Nature
- Spikes
- Roar
- Protect
- Toxic

Blissey (F) @ Leftovers
Ability: Natural Cure
EVs: 4 HP / 252 Def / 252 SpD
Bold Nature
- Soft-Boiled
- Ice Beam
- Thunder Wave
- Aromatherapy

Metagross @ Leftovers
Ability: Clear Body
EVs: 252 HP / 236 Atk / 20 Spe
Adamant Nature
- Meteor Mash
- Earthquake
- Explosion
- Brick Break
"""


# ---------------------------------------------------------------------------
# Per-battle state
# ---------------------------------------------------------------------------

@dataclass
class _BattleState:
    """All state we need for the next turn's invariant check, captured at
    action time. Kept SEPARATE from the reward manager so we can snapshot
    the exact preconditions the invariants reference without depending on
    the manager's internal layout."""
    reward_fn: Gen3RewardManager
    prev_ctx: Optional[BattleContext] = None
    last_action: Optional[int] = None
    # event_cursor read at the PREVIOUS decision — the start of the window
    # whose delta we validate on the next turn. The reward delta is folded from
    # events_since(prev_cursor) (the production path: EpisodeTracker.build_delta
    # → TurnDelta.build_from_events), NOT the deprecated snapshot-diff
    # TurnDelta.build. The two resolve our_switch_to differently: the event fold
    # reads the real |switch| event (robust to a switch-in fainting in-window),
    # while the snapshot diff nulls our_switch_to when the switch-in faints —
    # which mis-fired switch_base/switch_bouncing_tax invariants against a
    # delta that no longer reported the switch the model was credited for.
    prev_cursor: int = 0
    # Snapshot what record_action() captured for use in invariants.
    our_active_hp_before: float = 1.0
    our_boosts_before: np.ndarray = field(
        default_factory=lambda: np.zeros(7, dtype=np.int8)
    )
    # Was opp's SE threat flag set going into THIS turn? Read after record_action.
    prev_opp_se_threat: bool = False
    # Slot registries persist for the whole battle so contexts compare correctly.
    our_slots: SlotRegistry = field(default_factory=SlotRegistry)
    opp_slots: SlotRegistry = field(default_factory=SlotRegistry)


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class InvariantStats:
    turns_validated: int = 0
    violations: list = field(default_factory=list)
    fires_by_signal: Counter = field(default_factory=Counter)


# ---------------------------------------------------------------------------
# Invariant predicates
# ---------------------------------------------------------------------------

def _check_invariants(
    bd: RewardBreakdown,
    delta: TurnDelta,
    battle,
    state: _BattleState,
    stats: InvariantStats,
    turn: int,
    battle_tag: str,
) -> None:
    """Run every invariant against (breakdown, delta, battle, snapshot)."""
    violations: list[str] = []

    def viol(msg: str) -> None:
        violations.append(f"Turn {turn} [{battle_tag}]: {msg}")

    # --- NaN/inf rejection across every field ---
    for f in dc_fields(bd):
        v = getattr(bd, f.name)
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            viol(f"{f.name} is {v}")

    # --- Structural invariants on the delta itself ---
    # opp_resolved_move_id MUST equal event.move_id when an event is set —
    # this is the contract the consumers (reward manager, encoder, recorder)
    # depend on. If it ever drifts, the migration to opp_resolved_move_id
    # would silently regress.
    opp_event = delta.opp_damaging_event
    if opp_event is not None:
        stats.fires_by_signal["opp_event_set"] += 1
        if delta.opp_resolved_move_id != opp_event.move_id:
            viol(
                f"opp_resolved_move_id={delta.opp_resolved_move_id!r} disagrees "
                f"with event.move_id={opp_event.move_id!r}"
            )
        # The event's effectiveness MUST be bucketed (0.0, 0.5, 1.0, or 2.0)
        # — anything else is a poke-env protocol parsing regression.
        if opp_event.effectiveness not in (0.0, 0.5, 1.0, 2.0):
            viol(
                f"event.effectiveness={opp_event.effectiveness} not in "
                f"bucketed set (0.0, 0.5, 1.0, 2.0)"
            )
        # When event is set, scalar opp_effectiveness should agree with
        # event.effectiveness — both are populated by _set_effectiveness
        # in lockstep at the same protocol emission.
        if (
            delta.opp_effectiveness is not None
            and delta.opp_effectiveness != opp_event.effectiveness
        ):
            viol(
                f"delta.opp_effectiveness={delta.opp_effectiveness} disagrees "
                f"with event.effectiveness={opp_event.effectiveness}"
            )

    # --- Material PBRS Φ_mat (design §2): the old hp_ours/hp_opp/faint_ours/faint_opp base spine is
    # GONE — material is now one always-on PBRS term, bd.pbrs_material = γ·Φ_mat(s′)−Φ_mat(s). It is
    # bounded by the Φ_mat range (|ΔΦ_mat| ≤ 2·(MAT_HP_WEIGHT·6 + MAT_ALIVE_WEIGHT·6)) and is 0 on the
    # first window of an episode (prev_phi None). Telescoping/clutch-fix is checked by the dedicated
    # PBRS tests; here we just bound it. ---
    _phi_bound = 2.0 * (MAT_HP_WEIGHT * 6 + MAT_ALIVE_WEIGHT * 6)
    if not (-_phi_bound - 1e-6 <= bd.pbrs_material <= _phi_bound + 1e-6):
        viol(f"pbrs_material={bd.pbrs_material:.4f} out of [±{_phi_bound:.2f}]")

    # --- Base: win_loss only on terminal states ---
    if battle.won:
        stats.fires_by_signal["win_loss_win"] += 1
        if not _almost(bd.win_loss, VICTORY_VALUE):
            viol(f"win_loss={bd.win_loss:.4f} on win != +{VICTORY_VALUE}")
    elif battle.lost or battle.finished:
        stats.fires_by_signal["win_loss_lose"] += 1
        if not _almost(bd.win_loss, -VICTORY_VALUE):
            viol(f"win_loss={bd.win_loss:.4f} on loss != -{VICTORY_VALUE}")
    elif bd.win_loss != 0.0:
        viol(f"win_loss={bd.win_loss:.4f} fired without terminal state")

    # --- explosion_block (event-driven). The +2.0 explosion LITERAL is deleted (design §2.5) —
    # bd.explosion is always 0; the survive-Explosion credit rides Φ_mat. Only the block bonus (no
    # damage / 0× immunity) remains, inside the `not we_fainted` gate. ---
    opp_event = delta.opp_damaging_event
    is_opp_explosion = (
        opp_event is not None and opp_event.move_id in ("explosion", "selfdestruct")
    )
    if bd.explosion != 0.0:
        viol(f"explosion={bd.explosion:.4f} — the literal is deleted; must always be 0")
    if is_opp_explosion and not delta.we_fainted:
        block_expected = (
            float(delta.our_hp_delta.sum()) == 0.0 or opp_event.effectiveness == 0.0
        )
        if block_expected:
            stats.fires_by_signal["explosion_block"] += 1
            if not _almost(bd.explosion_block, EXPLOSION_BLOCK_BONUS):
                viol(
                    f"explosion_block={bd.explosion_block:.4f} != "
                    f"{EXPLOSION_BLOCK_BONUS} on blocked Explosion"
                )
    elif bd.explosion_block != 0.0:
        viol(f"explosion_block={bd.explosion_block:.4f} fired without Explosion block")

    # --- Base: finishing_blow ---
    our_mon = battle.active_pokemon
    our_move = (
        our_mon.moves.get(delta.our_move_id)
        if our_mon and delta.our_move_id else None
    )
    # Suppressed on self-faint (Explosion/Self-Destruct / mutual KO): a kill that
    # costs us our own mon is not a clean finishing blow.
    is_finishing_blow = (
        delta.opp_fainted
        and not delta.we_fainted
        and delta.our_move_id is not None
        and delta.our_switch_to is None
        and not delta.our_failed_to_move
        and our_move is not None
        and our_move.base_power > 0
    )
    if is_finishing_blow:
        stats.fires_by_signal["finishing_blow"] += 1
        if not _almost(bd.finishing_blow, FINISHING_BLOW_BONUS):
            viol(f"finishing_blow={bd.finishing_blow:.4f} != {FINISHING_BLOW_BONUS}")
    elif bd.finishing_blow != 0.0:
        viol(f"finishing_blow={bd.finishing_blow:.4f} fired without damaging-KO precondition")

    # --- Attack: roar (success + failed) ---
    if delta.our_move_id == "roar":
        if delta.opp_switch_to is None:
            stats.fires_by_signal["roar_failed"] += 1
            if not _almost(bd.roar, FAILED_ROAR_PENALTY):
                viol(f"roar={bd.roar:.4f} on failed Roar != {FAILED_ROAR_PENALTY}")
        else:
            # Either ROAR_BONUS (when spikes or boosts) or 0 (Roar landed but no spike/boost)
            if bd.roar not in (0.0, ROAR_BONUS):
                viol(f"roar={bd.roar:.4f} on successful Roar must be 0 or {ROAR_BONUS}")
            if bd.roar > 0:
                stats.fires_by_signal["roar_success"] += 1
    else:
        if bd.roar != 0.0:
            viol(f"roar={bd.roar:.4f} fired on non-Roar move '{delta.our_move_id}'")

    # --- Attack: futile_attack (general + immune) ---
    if bd.futile_attack != 0.0:
        stats.fires_by_signal["futile_attack"] += 1
        if bd.futile_attack > 0:
            viol(f"futile_attack={bd.futile_attack:.4f} must be <= 0")
        if delta.our_move_id is None or delta.our_failed_to_move or delta.opp_switch_to:
            viol(f"futile_attack={bd.futile_attack:.4f} fired in skipped precondition")
        if bd.futile_attack not in (FUTILE_ATTACK_PENALTY, FUTILE_IMMUNE_PENALTY):
            viol(
                f"futile_attack={bd.futile_attack:.4f} not in known penalties "
                f"({FUTILE_ATTACK_PENALTY}, {FUTILE_IMMUNE_PENALTY})"
            )

    # --- Attack: futile_setup ---
    move_is_boost = delta.our_move_id in BOOST_MOVES
    if bd.futile_setup != 0.0:
        stats.fires_by_signal["futile_setup"] += 1
        if not move_is_boost:
            viol(f"futile_setup={bd.futile_setup:.4f} fired on non-boost '{delta.our_move_id}'")
        if delta.our_failed_to_move:
            viol(f"futile_setup={bd.futile_setup:.4f} fired with failed_to_move")
        if bd.futile_setup >= 0:
            viol(f"futile_setup={bd.futile_setup:.4f} must be negative")

    # --- Attack: setup_low_hp ---
    if bd.setup_low_hp != 0.0:
        stats.fires_by_signal["setup_low_hp"] += 1
        if bd.setup_low_hp > 0:
            viol(f"setup_low_hp={bd.setup_low_hp:.4f} must be <= 0")
        if not move_is_boost:
            viol(f"setup_low_hp={bd.setup_low_hp:.4f} fired on non-boost '{delta.our_move_id}'")
        if state.our_active_hp_before >= SETUP_LOW_HP_THRESHOLD:
            viol(
                f"setup_low_hp={bd.setup_low_hp:.4f} fired at hp={state.our_active_hp_before:.3f}"
                f" >= threshold {SETUP_LOW_HP_THRESHOLD}"
            )

    # --- Attack: boost_utilized ---
    if bd.boost_utilized != 0.0:
        stats.fires_by_signal["boost_utilized"] += 1
        if bd.boost_utilized < 0:
            viol(f"boost_utilized={bd.boost_utilized:.4f} must be >= 0")
        effective_boost = max(
            int(state.our_boosts_before[0]), int(state.our_boosts_before[2])
        )
        if effective_boost <= 0:
            viol(
                f"boost_utilized={bd.boost_utilized:.4f} with no positive boosts "
                f"(atk={state.our_boosts_before[0]}, spa={state.our_boosts_before[2]})"
            )
        damage_dealt = -float(delta.opp_hp_delta.sum())
        if damage_dealt <= 0:
            viol(f"boost_utilized={bd.boost_utilized:.4f} with no damage dealt")

    # --- Attack: status_wasted ---
    if bd.status_wasted != 0.0:
        stats.fires_by_signal["status_wasted"] += 1
        if bd.status_wasted > 0:
            viol(f"status_wasted={bd.status_wasted:.4f} must be <= 0")

    # --- Field: spikes (layer add + waste) ---
    if bd.spikes != 0.0:
        stats.fires_by_signal["spikes"] += 1
        if bd.spikes > 0 and bd.spikes % SPIKES_LAYER_BONUS != 0:
            viol(f"spikes={bd.spikes:.4f} not a multiple of {SPIKES_LAYER_BONUS}")
        if bd.spikes < 0 and not _almost(bd.spikes, SPIKES_WASTE_PENALTY):
            viol(f"spikes={bd.spikes:.4f} negative but not {SPIKES_WASTE_PENALTY}")

    # --- Positional: matchup_penalty ---
    if bd.matchup_penalty != 0.0:
        stats.fires_by_signal["matchup_penalty"] += 1
        if not _almost(bd.matchup_penalty, MATCHUP_PENALTY):
            viol(f"matchup_penalty={bd.matchup_penalty:.4f} != {MATCHUP_PENALTY}")
        if delta.our_switch_to is not None:
            viol(f"matchup_penalty fired on switch turn")

    # --- Switch: pivot signals ---
    if bd.pivot_protect != 0.0:
        stats.fires_by_signal["pivot_protect"] += 1
        if not _almost(bd.pivot_protect, PROTECT_SWITCH_BONUS):
            viol(f"pivot_protect={bd.pivot_protect:.4f} != {PROTECT_SWITCH_BONUS}")
        if delta.our_switch_to is None:
            viol(f"pivot_protect fired without a switch")

    if bd.pivot_status != 0.0:
        stats.fires_by_signal["pivot_status"] += 1
        if not _almost(bd.pivot_status, STATUS_IMMUNE_SWITCH_BONUS):
            viol(f"pivot_status={bd.pivot_status:.4f} != {STATUS_IMMUNE_SWITCH_BONUS}")
        if delta.our_switch_to is None:
            viol(f"pivot_status fired without a switch")

    if bd.pivot_damage != 0.0:
        stats.fires_by_signal["pivot_damage"] += 1
        # pivot_damage is either 0.10 (resist) or 0.15 (immune via type)
        if bd.pivot_damage not in (0.10, 0.15):
            viol(f"pivot_damage={bd.pivot_damage:.4f} not in (0.10, 0.15)")
        if delta.our_switch_to is None:
            viol(f"pivot_damage fired without a switch")

    # --- Switch: se_switch + escape_threat_switch ---
    if bd.se_switch != 0.0:
        stats.fires_by_signal["se_switch"] += 1
        if not _almost(bd.se_switch, SE_SWITCH_BONUS):
            viol(f"se_switch={bd.se_switch:.4f} != {SE_SWITCH_BONUS}")
        if delta.our_switch_to is None:
            viol(f"se_switch fired without a switch")

    if bd.escape_threat_switch != 0.0:
        stats.fires_by_signal["escape_threat_switch"] += 1
        if bd.escape_threat_switch < 0:
            viol(f"escape_threat_switch={bd.escape_threat_switch:.4f} must be >= 0")

    # --- Switch: sleep_out / sleep_in ---
    if bd.sleep_out != 0.0:
        stats.fires_by_signal["sleep_out"] += 1
        if not _almost(bd.sleep_out, SLEEP_SWAP_BONUS):
            viol(f"sleep_out={bd.sleep_out:.4f} != {SLEEP_SWAP_BONUS}")

    if bd.sleep_in != 0.0:
        stats.fires_by_signal["sleep_in"] += 1
        if not _almost(bd.sleep_in, -SLEEP_SWAP_BONUS):
            viol(f"sleep_in={bd.sleep_in:.4f} != {-SLEEP_SWAP_BONUS}")
        active = battle.active_pokemon
        if not (active and active.status == Status.SLP):
            viol(f"sleep_in fired without sleeping active")

    # --- Switch subsidy: only fires on a REALIZED voluntary switch (hard invariant) ---
    # The reward manager settles the switch subsidy at OUTCOME time
    # (_apply_switch_outcome, gated on delta.our_switch_to), so a pressed-but-unrealized
    # switch — the poke-env "gap=0" case where the opponent faints on hazard entry the
    # same window and our switch silently no-ops — earns nothing and perturbs no
    # counters. switch_base / switch_bouncing_tax can therefore ONLY be non-zero when
    # the event-fold confirms the switch; firing without one is a real regression.
    if bd.switch_base != 0.0:
        stats.fires_by_signal["switch_base"] += 1
        if delta.our_switch_to is None:
            viol(f"switch_base={bd.switch_base:.4f} fired without a realized switch")
        # Starts at SWITCH_BASE_BONUS, reducible to 0 by spam_mult=0 (rapid alternating
        # switches). Range: [0, SWITCH_BASE_BONUS].
        if not (0.0 <= bd.switch_base <= SWITCH_BASE_BONUS):
            viol(f"switch_base={bd.switch_base:.4f} out of [0, {SWITCH_BASE_BONUS}]")

    if bd.switch_bouncing_tax != 0.0:
        stats.fires_by_signal["switch_bouncing_tax"] += 1
        if delta.our_switch_to is None:
            viol(f"switch_bouncing_tax={bd.switch_bouncing_tax:.4f} fired without a realized switch")
        if bd.switch_bouncing_tax > 0:
            viol(f"switch_bouncing_tax={bd.switch_bouncing_tax:.4f} must be <= 0")

    # --- Attack taxes: repetition / struggle ---
    if bd.repetition_tax != 0.0:
        stats.fires_by_signal["repetition_tax"] += 1
        if bd.repetition_tax > 0:
            viol(f"repetition_tax={bd.repetition_tax:.4f} must be <= 0")
        # Escalating + uncapped: any negative value down to the per-turn floor is legal.
        if bd.repetition_tax < REPETITION_TAX_FLOOR - 1e-6:
            viol(
                f"repetition_tax={bd.repetition_tax:.4f} below floor "
                f"{REPETITION_TAX_FLOOR}"
            )

    if bd.struggle_tax != 0.0:
        stats.fires_by_signal["struggle_tax"] += 1
        if not _almost(bd.struggle_tax, STRUGGLE_LOOP_TAX):
            viol(f"struggle_tax={bd.struggle_tax:.4f} != {STRUGGLE_LOOP_TAX}")

    # --- Status signal: changes in statused count ---
    # bd.status = (d_opp_statused - d_our_statused) * STATUS_BONUS
    # Just sanity-check the magnitude is bounded.
    if bd.status != 0.0:
        stats.fires_by_signal["status"] += 1
        # Max plausible: status applied to all 6 of opp's mons in one turn (impossible
        # in Gen 3 singles), so |status| <= 6 * STATUS_BONUS. Realistically ≤ 1 fires.
        if abs(bd.status) > 6 * STATUS_BONUS + 1e-6:
            viol(
                f"status={bd.status:.4f} magnitude exceeds 6 * STATUS_BONUS "
                f"({6 * STATUS_BONUS})"
            )

    # --- Stall tax: ramped, only fires after STALL_TAX_START_TURN, clamped at MAX ---
    if bd.stall_tax != 0.0:
        stats.fires_by_signal["stall_tax"] += 1
        if bd.stall_tax > 0:
            viol(f"stall_tax={bd.stall_tax:.4f} must be <= 0")
        if bd.stall_tax < -STALL_TAX_MAX - 1e-6:
            viol(f"stall_tax={bd.stall_tax:.4f} below clamp {-STALL_TAX_MAX}")
        if battle.turn <= STALL_TAX_START_TURN:
            viol(
                f"stall_tax fired at turn {battle.turn} <= "
                f"STALL_TAX_START_TURN ({STALL_TAX_START_TURN})"
            )

    stats.turns_validated += 1
    stats.violations.extend(violations)

    if violations:
        for v in violations:
            print(f"  [INVARIANT VIOLATION] {v}", flush=True)


def _almost(a: float, b: float, tol: float = 1e-6) -> bool:
    return abs(a - b) <= tol


# ---------------------------------------------------------------------------
# Fuzz player
# ---------------------------------------------------------------------------

class RewardInvariantFuzzPlayer(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats = InvariantStats()
        # All per-battle state lives in a single dict, keyed by battle_tag.
        # The previous single-valued instance fields silently destroyed
        # each other's state under max_concurrent_battles > 1, masking
        # signals and producing fake "all clean" results.
        self._per_battle: dict[str, _BattleState] = {}

    def _get_state(self, battle) -> _BattleState:
        tag = battle.battle_tag
        if tag not in self._per_battle:
            self._per_battle[tag] = _BattleState(
                reward_fn=Gen3RewardManager(),
            )
        return self._per_battle[tag]

    def _validate_turn(self, battle, state: _BattleState, curr_ctx: BattleContext) -> None:
        """Run one turn's reward computation + invariant check, given a fresh
        curr_ctx. Pulled out so the final-turn check (via
        _battle_finished_callback) can call the same path.

        The delta is folded from the per-decision event window
        (events_since(prev_cursor)) via TurnDelta.build_from_events — the SAME
        path the trainee uses in production (EpisodeTracker.build_delta →
        Gen3Env.calc_reward), so the harness validates the exact delta training
        computes rather than the deprecated snapshot-diff TurnDelta.build. The
        switch subsidy is settled at outcome time against this delta, so the
        switch-subsidy invariant below is a hard check (no gap=0 tolerance).
        """
        events = battle.events_since(state.prev_cursor)
        delta = TurnDelta.build_from_events(
            state.prev_ctx, curr_ctx, state.last_action, events
        )
        state.reward_fn.process_turn_reward(battle, delta)
        bd = state.reward_fn._last_breakdown
        _check_invariants(
            bd=bd,
            delta=delta,
            battle=battle,
            state=state,
            stats=self.stats,
            turn=battle.turn,
            battle_tag=battle.battle_tag,
        )
        if self.stats.violations:
            print(
                f"\n[FUZZ FATAL] Invariant violation detected — aborting.",
                flush=True,
            )
            os._exit(1)

    def _battle_finished_callback(self, battle) -> None:
        """Run the final-turn invariant check, then clean up per-battle state.

        Without this, the last turn (where win_loss fires) is never validated
        — poke-env stops calling choose_move once battle.finished is True.
        """
        tag = battle.battle_tag
        state = self._per_battle.get(tag)
        if state is None or state.prev_ctx is None or state.last_action is None:
            self._per_battle.pop(tag, None)
            return
        try:
            mask = np.ones(11, dtype=np.int8)  # mask is unused by from_battle's path
            curr_ctx = BattleContext.from_battle(
                battle, mask, state.our_slots, state.opp_slots
            )
            self._validate_turn(battle, state, curr_ctx)
        except Exception as e:
            print(
                f"\n[FUZZ NOTICE] {tag}: final-turn validation skipped: {e}",
                flush=True,
            )
        finally:
            self._per_battle.pop(tag, None)

    def choose_move(self, battle):
        try:
            state = self._get_state(battle)
            mask = Gen3ActionMasker.get_mask(battle)

            # event_cursor at THIS decision point — the start of the window whose
            # delta we'll validate next turn. Read before choosing, mirroring
            # EpisodeTracker.record() which captures _last_cursor at record time.
            cursor_now = battle.event_cursor

            # Build context with the persistent slot registries — this is
            # how slot maps stay consistent across turns.
            curr_ctx = BattleContext.from_battle(
                battle, mask, state.our_slots, state.opp_slots
            )

            if state.prev_ctx is not None and state.last_action is not None:
                self._validate_turn(battle, state, curr_ctx)

            if battle.finished:
                # Final-turn invariant validation happens in
                # _battle_finished_callback (poke-env doesn't call
                # choose_move again after the battle ends, so win_loss
                # would otherwise never be exercised).
                valid = np.where(mask == 1)[0]
                return Gen3ActionMapper.action_to_order(
                    int(np.random.choice(valid)), battle
                )

            valid = np.where(mask == 1)[0]
            choice = int(np.random.choice(valid))

            # record_action snapshots the at-decision HP/boosts that some
            # invariants need to reference (faint_ours magnitude formula,
            # setup_low_hp threshold check, boost_utilized precondition).
            # Capture the same snapshot in our state BEFORE record_action
            # mutates the reward manager — otherwise faint formula checks
            # would see the next turn's values.
            state.reward_fn.record_action(curr_ctx, choice)
            state.our_active_hp_before = state.reward_fn._our_active_hp_before
            state.our_boosts_before = state.reward_fn._our_boosts_before.copy()
            state.prev_opp_se_threat = state.reward_fn._prev_opp_se_threat

            state.prev_ctx = curr_ctx
            state.last_action = choice
            state.prev_cursor = cursor_now

            return Gen3ActionMapper.action_to_order(choice, battle)

        except SystemExit:
            raise
        except Exception as e:
            print(
                f"\n[FUZZ FATAL] Battle {battle.battle_tag}, "
                f"Turn {battle.turn}: {e}",
                flush=True,
            )
            traceback.print_exc()
            os._exit(1)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

async def main(n_battles: int = 30) -> None:
    ts = int(time.time()) % 100000
    print(f"Reward Invariants Fuzz Test — gen3ou — {n_battles} battles", flush=True)

    tb = Gen3Teambuilder(MIXED_TEAM)

    fuzz = RewardInvariantFuzzPlayer(
        battle_format=BATTLE_FORMAT,
        team=tb,
        # The reward manager now reads current-board facts through battle.live_view(),
        # which only Gen3Battle provides — match production so the e2e doesn't crash.
        battle_class=Gen3Battle,
        server_configuration=LocalhostServerConfiguration,
        account_configuration=AccountConfiguration(f"RInv{ts}p", "password"),
        max_concurrent_battles=5,
    )
    opp = RandomPlayer(
        battle_format=BATTLE_FORMAT,
        team=tb,
        server_configuration=LocalhostServerConfiguration,
        account_configuration=AccountConfiguration(f"RInv{ts}o", "password"),
        max_concurrent_battles=5,
    )

    await fuzz.battle_against(opp, n_battles=n_battles)

    s = fuzz.stats
    print(f"\n{'=' * 60}", flush=True)
    print(f"Reward Invariants Fuzz Test Results", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"Turns validated   : {s.turns_validated}", flush=True)
    print(f"Violations        : {len(s.violations)}", flush=True)
    print(f"\nSignal fire counts:", flush=True)
    for signal in sorted(s.fires_by_signal):
        print(f"  {signal:<25} {s.fires_by_signal[signal]:5d}", flush=True)

    # Highlight signals that never fired — coverage holes worth knowing about.
    # These are the dataclass field names mapped to which counter they bump.
    expected_signals = {
        "faint_ours", "faint_opp", "win_loss_win", "win_loss_lose",
        "explosion", "explosion_block", "finishing_blow",
        "roar_failed", "roar_success", "futile_attack",
        "futile_setup", "setup_low_hp", "boost_utilized", "status_wasted",
        "spikes", "matchup_penalty",
        "pivot_protect", "pivot_status", "pivot_damage",
        "se_switch", "escape_threat_switch", "sleep_out", "sleep_in",
        "switch_base", "switch_bouncing_tax",
        "repetition_tax", "struggle_tax",
        "status", "stall_tax",
    }
    never_fired = expected_signals - set(s.fires_by_signal)
    if never_fired:
        print(f"\nSignals that never fired this run ({len(never_fired)}):", flush=True)
        for sig in sorted(never_fired):
            print(f"  {sig}", flush=True)
        print("  (these may need targeted scenarios for coverage)", flush=True)

    if s.violations:
        print(f"\nVIOLATIONS ({len(s.violations)}):", flush=True)
        for v in s.violations[:20]:
            print(f"  {v}", flush=True)
        if len(s.violations) > 20:
            print(f"  ... and {len(s.violations) - 20} more", flush=True)
        print(f"\nFAIL — {len(s.violations)} invariant violation(s) found.", flush=True)
        sys.exit(1)
    else:
        print(
            f"\nPASS — All {s.turns_validated} turns validated with no violations.",
            flush=True,
        )


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    asyncio.run(main(n))
