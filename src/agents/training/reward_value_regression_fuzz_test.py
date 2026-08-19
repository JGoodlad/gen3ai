"""Single-path reward-value regression (replaces the reward re-sourcing equivalence harness).

The reward manager now has ONE current-board source — ``battle.live_view()``, built once per
``process_turn_reward`` and threaded to every helper — so the old live-vs-raw equivalence diff
is moot (the ``_read_live`` dual path was retired). This harness keeps the coverage by running
real gen3ou battles in-process via the local BattleStream bridge (no server) and, at every
decision, recording the full ``RewardBreakdown`` and asserting it is STABLE:

  * **finite** — no NaN / inf in any field;
  * **self-consistent** — ``breakdown.total`` equals the sum of its fields; and
  * **deterministic** — a second ``Gen3RewardManager`` driven with the identical
    ``record_action`` / ``process_turn_reward`` sequence produces a byte-identical breakdown,
    catching any hidden global state or non-determinism in the reward computation.

The deltas are folded from the event log (``TurnDelta.build_from_events`` over the
``events_since`` window) — the production path. Per-field activation counts are reported so a
"0 violations" run is trustworthy (it actually exercised the switch / status / spikes / etc.
signals). On ANY violation: print the offending field + both values, then ``os._exit(1)``.

Run directly (no server needed — runs via the local bridge):
    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/training/reward_value_regression_fuzz_test.py [n_battles]
"""

import asyncio
import dataclasses
import math
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
from agents.training.battle_snapshot import BattleContext
from agents.training.turn_delta import TurnDelta
from agents.training.reward_manager import Gen3RewardManager, RewardConfig
from agents.training.slot_registry import SlotRegistry
from utils.bridge.local_battle_runner import run_local_battles
from utils.teambuilder import Gen3Teambuilder

BATTLE_FORMAT = "gen3ou"

# A deliberately broad team so random play exercises every reward signal:
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
    """Per-battle pair of reward managers driven IDENTICALLY (determinism check) plus the
    context/slot bookkeeping needed to fold each turn's TurnDelta from the event log."""

    # `--no-all-shaping-pbrs`, STATED rather than inherited. This harness's trustworthiness rests
    # on its per-field activation counts ("it actually exercised the switch / status / spikes
    # signals"), and the DEFAULT composition (since 2026-08-18) zeroes every BIAS term but
    # `no_progress_tax` — so a default manager here would report a clean run over a class it never
    # touched. The additive-BIAS regime is a strict superset of the fields the default emits, so
    # this is the configuration that maximises what finiteness / self-consistency / determinism
    # actually cover.
    _CONFIG = RewardConfig(all_shaping_pbrs=False)

    def __init__(self):
        self.mgr_a = Gen3RewardManager(config=self._CONFIG)
        self.mgr_b = Gen3RewardManager(config=self._CONFIG)
        self.prev_ctx: Optional[BattleContext] = None
        self.last_action: Optional[int] = None
        self.prev_cursor: int = 0   # event_cursor when prev_ctx was latched
        self.our_slots = SlotRegistry()
        self.opp_slots = SlotRegistry()


class RewardValueRegressionPlayer(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.turns_compared = 0
        self.field_activations: Counter = Counter()
        self.violations: list[str] = []
        self.reward_sum = 0.0
        self.reward_min = math.inf
        self.reward_max = -math.inf
        self._per_battle: dict[str, _BattleState] = {}

    def _get_state(self, battle) -> _BattleState:
        tag = battle.battle_tag
        if tag not in self._per_battle:
            self._per_battle[tag] = _BattleState()
        return self._per_battle[tag]

    def _fail(self, battle, msg: str) -> None:
        self.violations.append(msg)
        print(f"\n[REWARD REGRESSION] {msg}", flush=True)
        traceback.print_stack()
        os._exit(1)

    def _check_turn(self, battle, state: _BattleState, curr_ctx: BattleContext) -> None:
        events = battle.events_since(state.prev_cursor)
        delta = TurnDelta.build_from_events(
            state.prev_ctx, curr_ctx, state.last_action, events
        )
        state.mgr_a.process_turn_reward(battle, delta)
        state.mgr_b.process_turn_reward(battle, delta)
        ba = state.mgr_a._last_breakdown
        bb = state.mgr_b._last_breakdown
        self.turns_compared += 1

        total = 0.0
        for f in dataclasses.fields(ba):
            a = float(getattr(ba, f.name))
            b = float(getattr(bb, f.name))
            # finite
            if not math.isfinite(a):
                self._fail(battle, f"non-finite {f.name}={a} "
                                   f"[{battle.battle_tag}] turn {battle.turn}")
            # deterministic across two identically-driven managers
            if a != b:
                self._fail(battle, f"non-deterministic {f.name}: a={a:.9g} b={b:.9g} "
                                   f"[{battle.battle_tag}] turn {battle.turn}")
            if a != 0.0:
                self.field_activations[f.name] += 1
            total += a
        # self-consistency: breakdown.total == sum of fields
        if abs(ba.total - total) > 1e-9:
            self._fail(battle, f"total {ba.total:.9g} != sum-of-fields {total:.9g} "
                               f"[{battle.battle_tag}] turn {battle.turn}")
        self.reward_sum += ba.total
        self.reward_min = min(self.reward_min, ba.total)
        self.reward_max = max(self.reward_max, ba.total)

    def _battle_finished_callback(self, battle) -> None:
        """Settle the final (terminal) turn — win_loss fires here — then drop per-battle state."""
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
            self._check_turn(battle, state, curr_ctx)
        except Exception as e:  # pragma: no cover - diagnostic only
            print(f"  [NOTICE] {tag}: final-turn check skipped: {e}", flush=True)
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
                self._check_turn(battle, state, curr_ctx)

            valid = np.where(mask == 1)[0]
            choice = int(np.random.choice(valid))

            if not battle.finished:
                # Drive BOTH managers identically so their internal state stays in lockstep.
                state.mgr_a.record_action(curr_ctx, choice)
                state.mgr_b.record_action(curr_ctx, choice)
                state.prev_ctx = curr_ctx
                state.last_action = choice
                state.prev_cursor = battle.event_cursor

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
        f"Reward-Value Regression — gen3ou — {n_battles} battles "
        f"(finite + self-consistent + deterministic)",
        flush=True,
    )
    tb = Gen3Teambuilder(MIXED_TEAM)

    fuzz = RewardValueRegressionPlayer(
        battle_format=BATTLE_FORMAT,
        team=tb,
        # Reward manager reads battle.live_view() -> needs Gen3Battle (production parity).
        battle_class=Gen3Battle,
        server_configuration=LocalhostServerConfiguration,
        start_listening=False,
        account_configuration=AccountConfiguration(f"RVr{ts}p", "password"),
        max_concurrent_battles=5,
    )
    opp = RandomPlayer(
        battle_format=BATTLE_FORMAT,
        team=tb,
        battle_class=Gen3Battle,
        server_configuration=LocalhostServerConfiguration,
        start_listening=False,
        account_configuration=AccountConfiguration(f"RVr{ts}o", "password"),
        max_concurrent_battles=5,
    )

    await run_local_battles(fuzz, opp, n_battles)

    print(f"\n{'=' * 60}", flush=True)
    print("Reward-Value Regression Results", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"Turns checked      : {fuzz.turns_compared}", flush=True)
    print(f"Violations         : {len(fuzz.violations)}", flush=True)
    if fuzz.turns_compared:
        mean = fuzz.reward_sum / fuzz.turns_compared
        print(f"Reward total/turn  : mean {mean:+.4f}  min {fuzz.reward_min:+.4f}  "
              f"max {fuzz.reward_max:+.4f}", flush=True)
    print("Field activations (non-zero turns):", flush=True)
    for name, count in fuzz.field_activations.most_common():
        print(f"  {name:<22} {count:5d}", flush=True)

    ok = (not fuzz.violations) and fuzz.turns_compared > 0
    if not fuzz.turns_compared:
        print("\nNO TURNS CHECKED — are teams loading / the bridge built?", flush=True)
    print(
        "\nPASS — every reward breakdown finite, self-consistent, and deterministic."
        if ok else "\nFAIL",
        flush=True,
    )
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    asyncio.run(main(n))
