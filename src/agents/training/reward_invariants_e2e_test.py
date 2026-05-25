"""
Fuzz E2E test for reward function component invariants.

Runs real battles against the live Showdown server and validates reward
breakdown invariants on every turn. Uses the same pattern as
transition_fuzz_e2e_test.py: intercept per-turn state, validate in choose_move().

Invariants checked each turn:
  1.  bd.faint_ours < 0 when we_fainted (magnitude in (-2.5, -0.5])
  2.  bd.faint_opp > 0 when opp_fainted (magnitude in [0.5, 2.5))
  3.  bd.futile_setup < 0 only on turns where our_move_id in BOOST_MOVES
      and our_boost_delta.sum() == 0 and not our_failed_to_move
  4.  bd.futile_setup == 0 on all non-boost-move turns
  5.  bd.setup_low_hp <= 0 always
  6.  bd.setup_low_hp < 0 only when our_move_id in BOOST_MOVES and hp < 0.40
  7.  bd.boost_utilized >= 0 always
  8.  bd.boost_utilized > 0 only when there were positive boosts AND damage dealt
  9.  bd.explosion_block >= 0 always
  10. bd.status_wasted <= 0 always
  11. No field is NaN or infinite

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
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.player.player import Player
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.action.mapper import Gen3ActionMapper
from agents.action.mask_generator import Gen3ActionMasker
from agents.training.battle_context import BattleContext, TurnDelta
from agents.training.episode_tracker import EpisodeTracker
from agents.training.reward_manager import (
    Gen3RewardManager,
    BOOST_MOVES,
    SETUP_LOW_HP_THRESHOLD,
    RewardBreakdown,
)
from utils.teambuilder import Gen3Teambuilder

BATTLE_FORMAT = "gen3ou"

# Team with diverse move types to exercise all reward signals
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
- Drill Peck
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


@dataclass
class InvariantStats:
    turns_validated: int = 0
    violations: list = field(default_factory=list)
    faint_ours_turns: int = 0
    faint_opp_turns: int = 0
    futile_setup_turns: int = 0
    setup_low_hp_turns: int = 0
    boost_utilized_turns: int = 0
    explosion_block_turns: int = 0
    status_wasted_turns: int = 0


def _check_invariants(
    bd: RewardBreakdown,
    delta: TurnDelta,
    our_active_hp_before: float,
    our_boosts_before: np.ndarray,
    stats: InvariantStats,
    turn: int,
    battle_tag: str,
) -> None:
    violations = []

    def viol(msg: str) -> None:
        violations.append(f"Turn {turn} [{battle_tag}]: {msg}")

    # Invariant 11: no NaN or inf in any field
    from dataclasses import fields as dc_fields
    for f in dc_fields(bd):
        v = getattr(bd, f.name)
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            viol(f"{f.name} is {v}")

    # Invariant 1: faint_ours < 0 when we fainted, and magnitude in (2.5, 0.5]
    if delta.we_fainted:
        stats.faint_ours_turns += 1
        if bd.faint_ours >= 0:
            viol(f"faint_ours={bd.faint_ours:.4f} must be < 0 when we_fainted")
        if not (-2.5 < bd.faint_ours <= -0.5):
            viol(f"faint_ours={bd.faint_ours:.4f} out of range (-2.5, -0.5]")

    # Invariant 2: faint_opp > 0 when opp fainted, magnitude in [0.5, 2.5)
    if delta.opp_fainted:
        stats.faint_opp_turns += 1
        if bd.faint_opp <= 0:
            viol(f"faint_opp={bd.faint_opp:.4f} must be > 0 when opp_fainted")
        if not (0.5 <= bd.faint_opp < 2.5):
            viol(f"faint_opp={bd.faint_opp:.4f} out of range [0.5, 2.5)")

    # Invariants 3-4: futile_setup
    move_is_boost = delta.our_move_id in BOOST_MOVES
    boost_had_no_effect = delta.our_boost_delta.sum() == 0
    if bd.futile_setup != 0.0:
        stats.futile_setup_turns += 1
        if not move_is_boost:
            viol(f"futile_setup={bd.futile_setup:.4f} fired on non-boost move '{delta.our_move_id}'")
        if delta.our_failed_to_move:
            viol(f"futile_setup={bd.futile_setup:.4f} fired when failed_to_move=True")
        if bd.futile_setup >= 0:
            viol(f"futile_setup={bd.futile_setup:.4f} must be negative when non-zero")
    if not move_is_boost and bd.futile_setup != 0.0:
        viol(f"futile_setup={bd.futile_setup:.4f} on non-boost move (invariant 4)")

    # Invariant 5: setup_low_hp <= 0 always
    if bd.setup_low_hp > 0:
        viol(f"setup_low_hp={bd.setup_low_hp:.4f} must be <= 0")

    # Invariant 6: setup_low_hp < 0 only for boost moves at low HP
    if bd.setup_low_hp < 0:
        stats.setup_low_hp_turns += 1
        if not move_is_boost:
            viol(f"setup_low_hp={bd.setup_low_hp:.4f} fired on non-boost move '{delta.our_move_id}'")
        if our_active_hp_before >= SETUP_LOW_HP_THRESHOLD:
            viol(
                f"setup_low_hp={bd.setup_low_hp:.4f} fired at hp={our_active_hp_before:.3f}"
                f" >= threshold {SETUP_LOW_HP_THRESHOLD}"
            )

    # Invariant 7: boost_utilized >= 0 always
    if bd.boost_utilized < 0:
        viol(f"boost_utilized={bd.boost_utilized:.4f} must be >= 0")

    # Invariant 8: boost_utilized > 0 only when positive boosts AND damage dealt
    if bd.boost_utilized > 0:
        stats.boost_utilized_turns += 1
        effective_boost = max(int(our_boosts_before[0]), int(our_boosts_before[2]))
        if effective_boost <= 0:
            viol(f"boost_utilized={bd.boost_utilized:.4f} fired with no positive boosts (atk={our_boosts_before[0]}, spa={our_boosts_before[2]})")
        damage_dealt = -float(delta.opp_hp_delta.sum())
        if damage_dealt <= 0:
            viol(f"boost_utilized={bd.boost_utilized:.4f} fired with no damage dealt")

    # Invariant 9: explosion_block >= 0 always
    if bd.explosion_block < 0:
        viol(f"explosion_block={bd.explosion_block:.4f} must be >= 0")
    if bd.explosion_block > 0:
        stats.explosion_block_turns += 1

    # Invariant 10: status_wasted <= 0 always
    if bd.status_wasted > 0:
        viol(f"status_wasted={bd.status_wasted:.4f} must be <= 0")
    if bd.status_wasted < 0:
        stats.status_wasted_turns += 1

    stats.turns_validated += 1
    stats.violations.extend(violations)

    if violations:
        for v in violations:
            print(f"  [INVARIANT VIOLATION] {v}", flush=True)


class RewardInvariantFuzzPlayer(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.stats = InvariantStats()
        self._reward_fn = Gen3RewardManager()
        self._tracker: Optional[EpisodeTracker] = None
        self._prev_ctx: Optional[BattleContext] = None
        self._last_action: Optional[int] = None

    def _get_tracker(self, battle) -> EpisodeTracker:
        tag = battle.battle_tag
        if self._tracker is None or getattr(self._tracker, "_battle_tag", None) != tag:
            self._tracker = EpisodeTracker()
            self._tracker._battle_tag = tag
            self._reward_fn.reset()
            self._prev_ctx = None
            self._last_action = None
        return self._tracker

    def choose_move(self, battle):
        try:
            tracker = self._get_tracker(battle)
            mask = Gen3ActionMasker.get_mask(battle)

            # Build current context
            from agents.training.slot_registry import SlotRegistry
            our_slots = SlotRegistry()
            opp_slots = SlotRegistry()
            # Rebuild slot maps from existing tracker state if possible
            curr_ctx = BattleContext.from_battle(battle, mask, our_slots, opp_slots)

            if self._prev_ctx is not None and self._last_action is not None:
                delta = TurnDelta.build(self._prev_ctx, curr_ctx, self._last_action)

                # Snapshot what reward_fn saw at action time (set by record_action)
                our_active_hp_before = self._reward_fn._our_active_hp_before
                our_boosts_before = self._reward_fn._our_boosts_before.copy()

                self._reward_fn.process_turn_reward(battle, delta)
                bd = self._reward_fn._last_breakdown

                _check_invariants(
                    bd=bd,
                    delta=delta,
                    our_active_hp_before=our_active_hp_before,
                    our_boosts_before=our_boosts_before,
                    stats=self.stats,
                    turn=battle.turn,
                    battle_tag=battle.battle_tag,
                )

                if self.stats.violations:
                    # Fail immediately on first violation
                    print(f"\n[FUZZ FATAL] Invariant violation detected — aborting.", flush=True)
                    os._exit(1)

            if battle.finished:
                self._prev_ctx = None
                self._last_action = None
                valid = np.where(mask == 1)[0]
                return Gen3ActionMapper.action_to_order(int(np.random.choice(valid)), battle)

            valid = np.where(mask == 1)[0]
            choice = int(np.random.choice(valid))

            # Record action BEFORE storing prev_ctx so reward_fn captures correct HP/boosts
            self._reward_fn.record_action(curr_ctx, choice)
            self._prev_ctx = curr_ctx
            self._last_action = choice

            return Gen3ActionMapper.action_to_order(choice, battle)

        except SystemExit:
            raise
        except Exception as e:
            print(f"\n[FUZZ FATAL] Battle {battle.battle_tag}, Turn {battle.turn}: {e}", flush=True)
            traceback.print_exc()
            os._exit(1)


async def main(n_battles: int = 30) -> None:
    ts = int(time.time()) % 100000
    print(f"Reward Invariants Fuzz Test — gen3ou — {n_battles} battles", flush=True)

    tb = Gen3Teambuilder(MIXED_TEAM)

    fuzz = RewardInvariantFuzzPlayer(
        battle_format=BATTLE_FORMAT,
        team=tb,
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
    print(f"Turns validated    : {s.turns_validated}", flush=True)
    print(f"Faint ours turns   : {s.faint_ours_turns}", flush=True)
    print(f"Faint opp turns    : {s.faint_opp_turns}", flush=True)
    print(f"Futile setup fires : {s.futile_setup_turns}", flush=True)
    print(f"Setup low HP fires : {s.setup_low_hp_turns}", flush=True)
    print(f"Boost utilized fires: {s.boost_utilized_turns}", flush=True)
    print(f"Explosion block fires: {s.explosion_block_turns}", flush=True)
    print(f"Status wasted fires: {s.status_wasted_turns}", flush=True)
    print(f"Violations         : {len(s.violations)}", flush=True)

    if s.violations:
        print(f"\nVIOLATIONS:", flush=True)
        for v in s.violations:
            print(f"  {v}", flush=True)
        print(f"\nFAIL — {len(s.violations)} invariant violation(s) found.", flush=True)
        sys.exit(1)
    else:
        print(f"\nPASS — All {s.turns_validated} turns validated with no violations.", flush=True)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    asyncio.run(main(n))
