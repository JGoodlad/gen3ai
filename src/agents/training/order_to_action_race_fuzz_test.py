"""Opponent order_to_action stale-race fuzz — bridge-backed, no live server.

Guards the fix for the residual stale-decision race at ``SingleAgentWrapper``'s env-level
serialize (the production crash in ``crashes/restart_err_*_fa1fe3.txt``): the opponent's order can
go stale between ``choose_move`` returning it and ``self.env.order_to_action`` re-reading the live
battle — the battle finished or flipped to a wait/default-only request under it — and a *strict*
serialize then raises ``ValueError: ... not in valid orders ['/choose default']``, killing the
``SubprocVecEnv`` worker. The fix catches that and falls back to the default order.

The single-threaded bridge can't reproduce the cross-thread *timing* (that's the e2e racing fuzz's
job), but it produces real Showdown battle states, against which this validates the fix's two
load-bearing branches with the REAL ``SinglesEnv.order_to_action`` — the exact function that raised
in production:

  1. **The crash condition is real.** A stale order — one not in ``battle.valid_orders`` (a switch to
     your own active mon is never legal) — DOES raise ``ValueError`` under a strict serialize, on
     real states. This is the regression the fix guards.
  2. **The fallback is universally safe.** ``order_to_action(DefaultBattleOrder(), battle,
     strict=True)`` resolves on EVERY real decision state (incl. force-switch) — so the wrapper's
     except-branch can never itself crash.

It then runs the *exact* wrapper handling (``try`` strict serialize → ``except ValueError`` →
default) on the stale order and asserts it yields a valid action — no raise — at every decision of
N real battles.

Run directly (needs deps/pokemon-showdown bridge + data/ mappings; no server):
    export PYTHONPATH=$PYTHONPATH:src
    python src/agents/training/order_to_action_race_fuzz_test.py [n_battles]
"""

from __future__ import annotations

import asyncio
import sys
import time

from poke_env.player import SimpleHeuristicsPlayer
from poke_env import AccountConfiguration, LocalhostServerConfiguration
from poke_env.environment.singles_env import SinglesEnv
from poke_env.player.battle_order import SingleBattleOrder, DefaultBattleOrder

from agents.inference.player import RLPlayer
from agents.observation.state_encoder import load_mappings
from agents.training.selfplay_opponent_fuzz_test import _build_model
from utils.teambuilder import Gen3Teambuilder
from utils.team_loader import TeamLoader
from utils.bridge.local_battle_runner import run_local_battles

BATTLE_FORMAT = "gen3ou"


def _wrapper_resolve(order, battle):
    """Mirror SingleAgentWrapper's opponent-order handling exactly: serialize strictly, and on the
    stale-race ValueError fall back to the default order (which must always resolve)."""
    try:
        return SinglesEnv.order_to_action(order, battle, fake=False, strict=True)
    except ValueError:
        return SinglesEnv.order_to_action(DefaultBattleOrder(), battle, fake=False, strict=True)


def _a_stale_order(battle):
    """Build an order that is NOT currently valid for this real battle state, standing in for an
    order gone stale under the race. A switch to your own active mon is never a legal switch;
    otherwise any team mon whose switch isn't currently offered. None if neither can be formed."""
    valid = {str(o) for o in battle.valid_orders}
    active = battle.active_pokemon
    if active is not None:
        cand = SingleBattleOrder(active)
        if str(cand) not in valid:
            return cand
    for mon in battle.team.values():
        cand = SingleBattleOrder(mon)
        if str(cand) not in valid:
            return cand
    return None


class _AuditingRLPlayer(RLPlayer):
    """Plays normally, but at every decision audits the fix's two branches against the real battle
    via the real SinglesEnv.order_to_action."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decisions = 0
        self.stale_checks = 0
        self.violations: list[str] = []

    def _tag(self, battle):
        return f"{battle.battle_tag} t{getattr(battle, 'turn', '?')}"

    def choose_move(self, battle):
        order = super().choose_move(battle)
        self.decisions += 1
        # (1) the default fallback must ALWAYS resolve on this real state.
        try:
            SinglesEnv.order_to_action(DefaultBattleOrder(), battle, fake=False, strict=True)
        except Exception as e:  # noqa: BLE001 - any raise here is a fix-breaking violation
            self.violations.append(f"{self._tag(battle)}: default fallback raised {e!r}")
        # (2) a stale order must raise under strict serialize, and the wrapper-style resolve must
        #     then recover with a valid action (never raise).
        stale = _a_stale_order(battle)
        if stale is not None:
            self.stale_checks += 1
            raised = False
            try:
                SinglesEnv.order_to_action(stale, battle, fake=False, strict=True)
            except ValueError:
                raised = True
            if not raised:
                self.violations.append(f"{self._tag(battle)}: stale order {stale} did NOT raise")
            try:
                _wrapper_resolve(stale, battle)
            except Exception as e:  # noqa: BLE001
                self.violations.append(f"{self._tag(battle)}: wrapper resolve raised {e!r}")
        return order


async def main(n_battles: int = 12) -> None:
    print(f"Order→action stale-race fuzz — gen3ou — {n_battles} battles")
    mappings = load_mappings()
    teams = TeamLoader().get_sample_teams()
    ts = int(time.time()) % 100000
    model, _v, _ek = _build_model(mappings)

    agent = _AuditingRLPlayer(
        model=model, team=Gen3Teambuilder(teams), battle_format=BATTLE_FORMAT,
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"OAfuzz{ts}", "password"),
        max_concurrent_battles=5, stochastic=True,
    )
    heur = SimpleHeuristicsPlayer(
        battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(teams),
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"OAheur{ts}", "password"),
        max_concurrent_battles=5,
    )
    await run_local_battles(agent, heur, n_battles, battle_format=BATTLE_FORMAT)

    fails: list[str] = []
    if agent.n_finished_battles != n_battles:
        fails.append(f"only {agent.n_finished_battles}/{n_battles} battles finished")
    if agent.stale_checks == 0:
        fails.append("no stale order was ever constructed — the audit was vacuous")
    if agent.violations:
        fails.append(f"{len(agent.violations)} violation(s):")
        fails.extend("    " + v for v in agent.violations[:10])

    print(f"  decisions: {agent.decisions}, stale-order checks: {agent.stale_checks}, "
          f"battles: {agent.n_finished_battles}/{n_battles}, violations: {len(agent.violations)}")
    print("=" * 60)
    if fails:
        print("❌ ORDER→ACTION STALE-RACE FUZZ FAILED")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("✅ ORDER→ACTION STALE-RACE FUZZ PASSED — stale order raises, default always recovers")
    sys.exit(0)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    asyncio.run(main(n))
