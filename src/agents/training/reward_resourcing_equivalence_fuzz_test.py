"""
Equivalence harness (design §8.3) for the Step-5 reward re-sourcing.

Runs real gen3ou battles in-process via the local BattleStream bridge (no server)
and, at every decision, computes the full ``RewardBreakdown`` TWICE from the SAME
``(battle, delta)``:

  * ``mgr_live``   — reads current-board facts through ``battle.live_view()``
                     (``_read_live=True``, the new path).
  * ``mgr_battle`` — reads the same facts from the raw poke-env ``battle``
                     (``_read_live=False``, the old path).

Both managers receive identical ``record_action`` and ``process_turn_reward`` calls,
so their internal state evolves in lockstep. Any per-field difference in the two
``RewardBreakdown``s is a re-sourcing regression. This is the gate that justifies
the migration: the partial re-source is value-neutral iff the diff rate is 0.

What's re-sourced (the only fields that can possibly differ): the spikes layer
count (``spikes`` / ``roar``), per-side status counts (``status`` / ``status_wasted``),
the opp-active-boosts snapshot (feeds next-turn ``roar``), and the terminal flags
(``win_loss``). Everything else is byte-identical because both managers share the
same ``TurnDelta`` and the type/effectiveness helpers still read the raw battle.

Run directly (no server needed — runs via the local bridge):
    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/training/reward_resourcing_equivalence_fuzz_test.py [n_battles]
"""

import asyncio
import dataclasses
import os
import sys
import time
import traceback
from collections import Counter
from typing import Optional

import numpy as np

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer
from poke_env.player.player import Player
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.action.mapper import Gen3ActionMapper
from agents.action.mask_generator import Gen3ActionMasker
from agents.battle.gen3_battle import Gen3Battle
from agents.training.battle_context import BattleContext, TurnDelta
from agents.training.reward_manager import Gen3RewardManager
from agents.training.slot_registry import SlotRegistry
from utils.bridge.local_battle_runner import run_local_battles
from utils.teambuilder import Gen3Teambuilder

BATTLE_FORMAT = "gen3ou"

# A deliberately broad team so random play exercises every re-sourced signal:
#   - Spikes / Roar          -> spikes-layer + roar reads
#   - Will-O-Wisp/Toxic/T-Wave -> status-count reads
#   - Calm Mind / Dragon Dance -> opp-active boosts snapshot
#   - Explosion              -> terminal / faint interplay
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


class _BattleState:
    """Per-battle pair of reward managers + the context/slot bookkeeping needed to
    rebuild each turn's TurnDelta. Both managers are driven identically."""

    def __init__(self):
        self.mgr_live = Gen3RewardManager()           # _read_live=True (default)
        self.mgr_battle = Gen3RewardManager()
        self.mgr_battle._read_live = False             # reads the raw battle
        self.prev_ctx: Optional[BattleContext] = None
        self.last_action: Optional[int] = None
        self.our_slots = SlotRegistry()
        self.opp_slots = SlotRegistry()


class RewardEquivalencePlayer(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.turns_compared = 0
        self.field_diffs: Counter = Counter()
        self.violations: list[str] = []
        self._per_battle: dict[str, _BattleState] = {}

    def _get_state(self, battle) -> _BattleState:
        tag = battle.battle_tag
        if tag not in self._per_battle:
            self._per_battle[tag] = _BattleState()
        return self._per_battle[tag]

    def _compare_turn(self, battle, state: _BattleState, curr_ctx: BattleContext) -> None:
        """Run BOTH managers on the same (battle, delta) and diff the breakdowns."""
        delta = TurnDelta.build(state.prev_ctx, curr_ctx, state.last_action)
        state.mgr_live.process_turn_reward(battle, delta)
        state.mgr_battle.process_turn_reward(battle, delta)
        bl = state.mgr_live._last_breakdown
        bb = state.mgr_battle._last_breakdown
        self.turns_compared += 1
        for f in dataclasses.fields(bl):
            a = float(getattr(bl, f.name))
            b = float(getattr(bb, f.name))
            if abs(a - b) > 1e-9:
                self.field_diffs[f.name] += 1
                msg = (
                    f"Turn {battle.turn} [{battle.battle_tag}]: "
                    f"{f.name} live={a:.6f} battle={b:.6f}"
                )
                self.violations.append(msg)
                print(f"  [REWARD DIFF] {msg}", flush=True)

    def _battle_finished_callback(self, battle) -> None:
        """Compare the final (terminal) turn — win_loss only fires here — then
        drop per-battle state. poke-env stops calling choose_move once finished."""
        tag = battle.battle_tag
        state = self._per_battle.get(tag)
        if state is None or state.prev_ctx is None or state.last_action is None:
            self._per_battle.pop(tag, None)
            return
        try:
            mask = np.ones(11, dtype=np.int8)
            curr_ctx = BattleContext.from_battle(
                battle, mask, state.our_slots, state.opp_slots
            )
            self._compare_turn(battle, state, curr_ctx)
        except Exception as e:  # pragma: no cover - diagnostic only
            print(f"  [NOTICE] {tag}: final-turn compare skipped: {e}", flush=True)
        finally:
            self._per_battle.pop(tag, None)

    def choose_move(self, battle):
        try:
            state = self._get_state(battle)
            mask = Gen3ActionMasker.get_mask(battle)
            curr_ctx = BattleContext.from_battle(
                battle, mask, state.our_slots, state.opp_slots
            )

            if state.prev_ctx is not None and state.last_action is not None:
                self._compare_turn(battle, state, curr_ctx)

            valid = np.where(mask == 1)[0]
            choice = int(np.random.choice(valid))

            if not battle.finished:
                # Drive BOTH managers identically so their state stays in lockstep.
                state.mgr_live.record_action(curr_ctx, choice)
                state.mgr_battle.record_action(curr_ctx, choice)
                state.prev_ctx = curr_ctx
                state.last_action = choice

            return Gen3ActionMapper.action_to_order(choice, battle)

        except Exception as e:
            print(
                f"\n[FUZZ FATAL] Battle {battle.battle_tag}, Turn {battle.turn}: {e}",
                flush=True,
            )
            traceback.print_exc()
            os._exit(1)


async def main(n_battles: int = 30) -> None:
    ts = int(time.time()) % 100000
    print(
        f"Reward Re-sourcing Equivalence Harness — gen3ou — {n_battles} battles",
        flush=True,
    )
    tb = Gen3Teambuilder(MIXED_TEAM)

    equiv = RewardEquivalencePlayer(
        battle_format=BATTLE_FORMAT,
        team=tb,
        # Reward manager reads battle.live_view() -> needs Gen3Battle (production parity).
        battle_class=Gen3Battle,
        server_configuration=LocalhostServerConfiguration,
        start_listening=False,
        account_configuration=AccountConfiguration(f"REq{ts}p", "password"),
        max_concurrent_battles=5,
    )
    opp = RandomPlayer(
        battle_format=BATTLE_FORMAT,
        team=tb,
        server_configuration=LocalhostServerConfiguration,
        start_listening=False,
        account_configuration=AccountConfiguration(f"REq{ts}o", "password"),
        max_concurrent_battles=5,
    )

    await run_local_battles(equiv, opp, n_battles)

    print(f"\n{'=' * 60}", flush=True)
    print("Reward Re-sourcing Equivalence Results", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"Turns compared    : {equiv.turns_compared}", flush=True)
    total_diffs = sum(equiv.field_diffs.values())
    rate = (total_diffs / equiv.turns_compared) if equiv.turns_compared else 0.0
    print(f"Field diffs        : {total_diffs}", flush=True)
    print(f"Diff rate          : {rate:.4%}", flush=True)
    if equiv.field_diffs:
        print("Per-field diffs:", flush=True)
        for name, count in equiv.field_diffs.most_common():
            print(f"  {name:<22} {count:5d}", flush=True)

    if total_diffs:
        print(
            f"\nFAIL — {total_diffs} reward-breakdown diff(s) between live and "
            f"battle sourcing. The re-source is NOT value-neutral.",
            flush=True,
        )
        for v in equiv.violations[:20]:
            print(f"  {v}", flush=True)
        if len(equiv.violations) > 20:
            print(f"  ... and {len(equiv.violations) - 20} more", flush=True)
        sys.exit(1)
    else:
        print(
            f"\nPASS — all {equiv.turns_compared} turns identical across both "
            f"sources. Partial re-source is value-neutral.",
            flush=True,
        )


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    asyncio.run(main(n))
