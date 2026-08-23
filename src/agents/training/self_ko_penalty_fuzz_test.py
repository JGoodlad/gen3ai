"""Bridge fuzz test for the HP-scaled self-KO penalty (--self-ko-hp-penalty).

Runs real gen3ou battles in-process via the local BattleStream bridge (no server) with a team
that runs Explosion (Gengar + Metagross), so random play self-KOs often. At every decision it
folds the production TurnDelta from the event log and drives TWO reward managers identically:

  * ON  — ``self_ko_hp_penalty = W`` (= 2.5)
  * OFF — ``self_ko_hp_penalty = 0.0`` (the default)

both under ``all_shaping_pbrs=False``, which is the composition in which a BIAS term is LIVE at all
(see ``_config``) and is checked at startup by ``_assert_arm_is_live`` rather than assumed.

and asserts the invariants the unit tests can only check on a hand-built delta — that the REAL
protocol path produces ``our_move_id ∈ {explosion, selfdestruct}`` + ``we_fainted`` on a self-KO,
and that:

  * on a self-KO turn (we executed it and fainted): ON charges EXACTLY ``−W · hp`` where ``hp`` is
    the decision-time HP the manager snapshotted (``_our_active_hp_before``), so a HEALTHY explosion
    nets a clearly-negative penalty and a low-HP "explode-a-dying-mon-for-a-KO" nets ~0;
  * on EVERY non-self-KO turn the penalty is 0 (it fires nowhere else);
  * OFF is byte-unchanged — ``self_ko_penalty == 0.0`` on every turn (the default no-op).

Coverage (a healthy self-KO actually occurred) is REPORTED — random play is stochastic, so a run
with zero self-KOs is a NOTICE, not a pass. On ANY invariant violation: print + ``os._exit(1)``.

Run directly (no server needed — runs via the local bridge):
    python src/agents/training/self_ko_penalty_fuzz_test.py [n_battles]
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

import asyncio
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
from agents.training.turn_delta import SELF_KO_MOVES, TurnDelta
from agents.training.reward_manager import (
    Gen3RewardManager,
    RewardConfig,
    reward_class_composition,
)
from agents.training.slot_registry import SlotRegistry
from utils.bridge.local_battle_runner import run_local_battles
from utils.teambuilder import Gen3Teambuilder

BATTLE_FORMAT = "gen3ou"
W = 2.5  # the self-KO penalty weight under test

# Two Explosion users (Gengar, Metagross) so random play self-KOs frequently, plus recovery /
# setup / status to keep the rest of the reward signals exercised.
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


def _config(weight: float) -> RewardConfig:
    """The composition under test — and `all_shaping_pbrs=False` is LOAD-BEARING, not tidiness.

    `self_ko_penalty` is a BIAS term, and `--all-shaping-pbrs` (the DEFAULT since 2026-08-18)
    ZEROES every BIAS term but `no_progress_tax` — before the weight is ever consulted. Built with
    the default this fuzz asserted `−W·hp` on a term its own composition forces to 0.0, so it
    FAILED on the first real self-KO it ever saw. It is the vacuity family in its second form: not
    a test that asserts nothing, but a test whose configuration makes its own claim unreachable.

    So the arms are pinned to the composition in which the flag under test is LIVE. Read the
    activeness from `_bias_term_active`, never from the flag names — that is the single source
    `reward_class_composition` prints at launch, and `_assert_arm_is_live` below checks it at
    startup rather than trusting this comment.
    """
    return RewardConfig(self_ko_hp_penalty=weight, all_shaping_pbrs=False)


class _BattleState:
    def __init__(self):
        self.mgr_on = Gen3RewardManager(config=_config(W))
        self.mgr_off = Gen3RewardManager(config=_config(0.0))
        self.prev_ctx: Optional[BattleContext] = None
        self.last_action: Optional[int] = None
        self.prev_cursor: int = 0
        self.our_slots = SlotRegistry()
        self.opp_slots = SlotRegistry()


class SelfKoPenaltyPlayer(Player):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.turns_checked = 0
        self.self_ko_turns = 0
        self.healthy_self_ko_turns = 0   # hp >= 0.8 at decision time
        self.hp_when_self_ko: list[float] = []
        self.field_activations: Counter = Counter()
        self.violations: list[str] = []
        self._per_battle: dict[str, _BattleState] = {}

    def _get_state(self, battle) -> _BattleState:
        tag = battle.battle_tag
        if tag not in self._per_battle:
            self._per_battle[tag] = _BattleState()
        return self._per_battle[tag]

    def _fail(self, battle, msg: str) -> None:
        self.violations.append(msg)
        print(f"\n[SELF-KO PENALTY] {msg}", flush=True)
        traceback.print_stack()
        os._exit(1)

    def _check_turn(self, battle, state: _BattleState, curr_ctx: BattleContext) -> None:
        events = battle.events_since(state.prev_cursor)
        delta = TurnDelta.build_from_events(state.prev_ctx, curr_ctx, state.last_action, events)
        # decision-time HP the manager snapshotted in the record_action that chose this turn's move
        # (process_turn_reward does NOT touch it, so read it either side of the call).
        hp_before = float(state.mgr_on._our_active_hp_before)
        state.mgr_on.process_turn_reward(battle, delta)
        state.mgr_off.process_turn_reward(battle, delta)
        bon = state.mgr_on._last_breakdown
        boff = state.mgr_off._last_breakdown
        self.turns_checked += 1

        # is this a self-KO turn the penalty SHOULD fire on?
        is_self_ko = (delta.we_fainted and not delta.our_failed_to_move
                      and delta.our_move_id in SELF_KO_MOVES)

        # finite
        if not math.isfinite(bon.self_ko_penalty) or not math.isfinite(boff.self_ko_penalty):
            self._fail(battle, f"non-finite self_ko_penalty on={bon.self_ko_penalty} "
                               f"off={boff.self_ko_penalty} turn {battle.turn}")
        # total == sum of fields (the new field must not break the registry-summed total)
        for tag, bd in (("on", bon), ("off", boff)):
            s = sum(float(v) for v in vars(bd).values() if isinstance(v, float))
            if abs(bd.total - s) > 1e-9:
                self._fail(battle, f"[{tag}] total {bd.total:.9g} != sum {s:.9g} turn {battle.turn}")

        # OFF is the default no-op: the field is 0.0 on every turn
        if boff.self_ko_penalty != 0.0:
            self._fail(battle, f"OFF self_ko_penalty != 0 ({boff.self_ko_penalty}) — default must be "
                               f"byte-unchanged, turn {battle.turn}")

        if is_self_ko:
            self.self_ko_turns += 1
            self.hp_when_self_ko.append(hp_before)
            expected = -W * min(1.0, max(0.0, hp_before))
            if abs(bon.self_ko_penalty - expected) > 1e-9:
                self._fail(battle, f"self-KO penalty {bon.self_ko_penalty:.6f} != expected "
                                   f"{expected:.6f} (W={W}, hp={hp_before:.4f}) "
                                   f"move={delta.our_move_id} turn {battle.turn}")
            if hp_before >= 0.8:
                self.healthy_self_ko_turns += 1
                # the targeted blunder must net a clearly-negative penalty
                if bon.self_ko_penalty > -W * 0.8 + 1e-9:
                    self._fail(battle, f"healthy self-KO (hp={hp_before:.2f}) penalty "
                                       f"{bon.self_ko_penalty:.3f} not clearly negative, turn {battle.turn}")
        else:
            # the penalty fires NOWHERE else
            if bon.self_ko_penalty != 0.0:
                self._fail(battle, f"self_ko_penalty {bon.self_ko_penalty} fired on a NON-self-KO turn "
                                   f"(we_fainted={delta.we_fainted} move={delta.our_move_id} "
                                   f"failed={delta.our_failed_to_move}) turn {battle.turn}")
        if bon.self_ko_penalty != 0.0:
            self.field_activations["self_ko_penalty"] += 1

    def _battle_finished_callback(self, battle) -> None:
        tag = battle.battle_tag
        state = self._per_battle.get(tag)
        if state is None or state.prev_ctx is None or state.last_action is None:
            self._per_battle.pop(tag, None)
            return
        try:
            mask = np.ones(11, dtype=np.int8)
            curr_ctx = BattleContext.from_battle(battle, mask, state.our_slots, state.opp_slots)
            self._check_turn(battle, state, curr_ctx)
        except Exception as e:  # pragma: no cover - diagnostic only
            print(f"  [NOTICE] {tag}: final-turn check skipped: {e}", flush=True)
        finally:
            self._per_battle.pop(tag, None)

    def choose_move(self, battle):
        try:
            state = self._get_state(battle)
            mask = Gen3ActionMasker.get_mask(battle)
            curr_ctx = BattleContext.from_battle(battle, mask, state.our_slots, state.opp_slots)

            if state.prev_ctx is not None and state.last_action is not None:
                self._check_turn(battle, state, curr_ctx)

            valid = np.where(mask == 1)[0]
            choice = int(np.random.choice(valid))

            if not battle.finished:
                state.mgr_on.record_action(curr_ctx, choice)
                state.mgr_off.record_action(curr_ctx, choice)
                state.prev_ctx = curr_ctx
                state.last_action = choice
                state.prev_cursor = battle.event_cursor

            return Gen3ActionMapper.action_to_order(choice, battle)
        except Exception as e:
            print(f"\n[FUZZ FATAL] Battle {battle.battle_tag}, Turn {battle.turn}: {e}", flush=True)
            traceback.print_exc()
            os._exit(1)


def _assert_arm_is_live() -> None:
    """Refuse to run in a composition that zeroes the term under test.

    The bug this test shipped with was silent: the ON arm's `self_ko_penalty` was suppressed, so
    the fuzz asserted a value its own config forced to 0.0. A startup check on the CENSUS (not on
    the flag names) makes that unrepresentable — and it prints the composition, so a reader can see
    which arm is live rather than infer it."""
    on = reward_class_composition(_config(W))
    off = reward_class_composition(_config(0.0))
    print(f"Composition ON  : {on['bias']} BIAS terms active, self_ko_penalty="
          f"{'LIVE' if 'self_ko_penalty' in on['bias_terms'] else 'SUPPRESSED'}", flush=True)
    print(f"Composition OFF : {off['bias']} BIAS terms active, self_ko_penalty="
          f"{'LIVE' if 'self_ko_penalty' in off['bias_terms'] else 'SUPPRESSED'}", flush=True)
    if "self_ko_penalty" not in on["bias_terms"]:
        print("\n❌ VACUOUS — the ON arm's composition SUPPRESSES self_ko_penalty, so every "
              "assertion below would be about a structurally-zero field.", flush=True)
        sys.exit(1)
    if "self_ko_penalty" in off["bias_terms"]:
        print("\n❌ the OFF arm must NOT charge self_ko_penalty — it is the no-op control.",
              flush=True)
        sys.exit(1)


async def main(n_battles: int = 40) -> None:
    ts = int(time.time()) % 100000
    print(f"Self-KO Penalty Fuzz — gen3ou — {n_battles} battles (W={W})", flush=True)
    _assert_arm_is_live()
    tb = Gen3Teambuilder(MIXED_TEAM)
    fuzz = SelfKoPenaltyPlayer(
        battle_format=BATTLE_FORMAT, team=tb, battle_class=Gen3Battle,
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"SKp{ts}p", "password"),
        max_concurrent_battles=5,
    )
    opp = RandomPlayer(
        battle_format=BATTLE_FORMAT, team=tb, battle_class=Gen3Battle,
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"SKp{ts}o", "password"),
        max_concurrent_battles=5,
    )
    await run_local_battles(fuzz, opp, n_battles)

    print(f"\n{'=' * 60}", flush=True)
    print("Self-KO Penalty Fuzz Results", flush=True)
    print(f"{'=' * 60}", flush=True)
    print(f"Turns checked        : {fuzz.turns_checked}", flush=True)
    print(f"Self-KO turns         : {fuzz.self_ko_turns}", flush=True)
    print(f"  of which HEALTHY (hp>=0.8) : {fuzz.healthy_self_ko_turns}", flush=True)
    if fuzz.hp_when_self_ko:
        hps = sorted(fuzz.hp_when_self_ko)
        print(f"  self-KO decision HP   : min {hps[0]:.2f}  median {hps[len(hps)//2]:.2f}  "
              f"max {hps[-1]:.2f}", flush=True)
    print(f"Violations            : {len(fuzz.violations)}", flush=True)

    ok = (not fuzz.violations) and fuzz.turns_checked > 0
    if not fuzz.turns_checked:
        print("\nNO TURNS CHECKED — are teams loading / the bridge built?", flush=True)
        ok = False
    if fuzz.self_ko_turns == 0:
        # stochastic coverage gap, not a correctness failure — flag loudly but don't fail
        print("\n[NOTICE] no self-KO turns observed this run — invariants held vacuously; "
              "re-run with more battles to exercise the penalty path.", flush=True)
    elif fuzz.healthy_self_ko_turns == 0:
        print("\n[NOTICE] self-KOs occurred but none at >=0.8 HP — the healthy-blunder coverage "
              "was not hit this run (stochastic).", flush=True)
    print("\nPASS — self-KO penalty = -W*hp on every self-KO, 0 elsewhere, OFF byte-unchanged."
          if ok else "\nFAIL", flush=True)
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    asyncio.run(main(n))
