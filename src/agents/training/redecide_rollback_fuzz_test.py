"""Re-decide rollback fuzz test — bridge-backed, no live server.

Guards the fix for the self-play stale-decision race: when the live battle advances under the
opponent during its model forward, ``RLPlayer.choose_move`` RE-DECIDES on the now-current
request instead of crashing. Each attempt's ``embed_battle()`` records its would-be decision
into the tracker's rolling turn-history, so the re-decide MUST roll the superseded attempt back
out (``EpisodeTracker.snapshot``/``restore``) — otherwise every race would leave a **phantom
turn** in the opponent's turn-history observation.

Unit tests cover the rollback mechanism in isolation (``episode_tracker_test`` /
``player_test``). This is the end-to-end guard: it plays REAL battles in-process via the local
BattleStream bridge and forces the re-decide path on EVERY decision by injecting a one-shot
``StaleDecisionError`` before each serialize, then validates — per decision, in ``choose_move`` —
that the committed history grew by **at most one transition**. Without the rollback the injected
re-decide would double-commit (Δ_n_transitions == 2); with it, exactly the clean amount.

``_n_transitions`` is the load-bearing invariant because it is MONOTONIC (never bounded by the
history deque's maxlen), so the phantom is caught even deep in a long game where ``len(_history)``
has saturated at its cap and a length delta would read 0 either way.

What it asserts:
  1. **No phantom turn.** For every opponent decision, ``_n_transitions`` and ``len(_history)``
     each grow by ≤ 1 — the injected re-decide commits the same as a single clean decision.
  2. **The race path was actually exercised.** The injection fired (``injected > 0``) and the
     opponent re-decided (``_n_redecides > 0``) — the test isn't vacuously green.
  3. **The re-decide always yields a valid order.** All battles finish (a crash or an illegal
     action would abort one), proving the opponent never killed the "worker".

Run directly (needs deps/pokemon-showdown bridge + data/ mappings; no server):
    python src/agents/training/redecide_rollback_fuzz_test.py [n_battles]
    (in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)
"""

from __future__ import annotations

import asyncio
import sys
import time

from poke_env.player import SimpleHeuristicsPlayer
from poke_env import AccountConfiguration, LocalhostServerConfiguration

from agents.inference.player import RLPlayer
from agents.action.mapper import StaleDecisionError
from agents.observation.state_encoder import load_mappings
from agents.training.selfplay_opponent_fuzz_test import _build_model
from utils.teambuilder import Gen3Teambuilder
from utils.team_loader import TeamLoader
from utils.bridge.local_battle_runner import run_local_battles

BATTLE_FORMAT = "gen3ou"


class _InjectingRLPlayer(RLPlayer):
    """RLPlayer that forces the stale re-decide path on every decision and audits, per
    decision, that the rollback left no phantom turn in the rolling history."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decisions = 0          # choose_move calls that reached a serialize
        self.injected = 0           # one-shot stales actually raised (re-decides forced)
        self.violations: list[str] = []

    def choose_move(self, battle):
        tracker = self._get_tracker(battle)
        t0, h0 = tracker._n_transitions, len(tracker._history)
        self._arm_injection = True                      # one-shot stale for THIS decision
        order = super().choose_move(battle)
        d_trans = tracker._n_transitions - t0
        d_hist = len(tracker._history) - h0
        if d_trans > 1 or d_hist > 1:
            # A double-commit: the superseded attempt's record() was NOT rolled back.
            self.violations.append(
                f"{battle.battle_tag} turn={getattr(battle, 'turn', '?')}: "
                f"Δ_n_transitions={d_trans} Δlen_history={d_hist} (phantom turn from re-decide)"
            )
        return order

    def action_to_order(self, action_idx, battle):
        if getattr(self, "_arm_injection", False):
            self._arm_injection = False                 # disarm → the re-decide's retry is real
            self.injected += 1
            self.decisions += 1
            raise StaleDecisionError("injected one-shot stale (fuzz)")
        return super().action_to_order(action_idx, battle)


async def main(n_battles: int = 12) -> None:
    print(f"Re-decide Rollback Fuzz — gen3ou — {n_battles} battles "
          f"(one-shot stale injected on every decision)")
    mappings = load_mappings()
    teams = TeamLoader().get_sample_teams()
    ts = int(time.time()) % 100000

    model, _version, _ek = _build_model(mappings)

    agent = _InjectingRLPlayer(
        model=model, team=Gen3Teambuilder(teams), battle_format=BATTLE_FORMAT,
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"RDfuzz{ts}", "password"),
        max_concurrent_battles=5, stochastic=True, temperature=1.0,
    )
    heur = SimpleHeuristicsPlayer(
        battle_format=BATTLE_FORMAT, team=Gen3Teambuilder(teams),
        server_configuration=LocalhostServerConfiguration, start_listening=False,
        account_configuration=AccountConfiguration(f"RDheur{ts}", "password"),
        max_concurrent_battles=5,
    )

    await run_local_battles(agent, heur, n_battles, battle_format=BATTLE_FORMAT)

    fails: list[str] = []
    if agent.n_finished_battles != n_battles:
        fails.append(
            f"only {agent.n_finished_battles}/{n_battles} battles finished — a re-decide "
            f"crashed or returned an illegal order (the opponent must always return valid)"
        )
    if agent.injected == 0:
        fails.append("no stale was ever injected — the re-decide path was not exercised (vacuous)")
    if agent._n_redecides == 0:
        fails.append(
            f"injected {agent.injected} stales but _n_redecides==0 — choose_move did not "
            f"re-decide (the injection did not reach the re-decide loop)"
        )
    if agent.violations:
        fails.append(f"{len(agent.violations)} phantom-turn violation(s):")
        fails.extend(f"    {v}" for v in agent.violations[:10])

    print(f"  decisions injected: {agent.injected}, re-decides: {agent._n_redecides}, "
          f"battles: {agent.n_finished_battles}/{n_battles}, "
          f"phantom violations: {len(agent.violations)}")

    print(f"\n{'=' * 60}")
    if fails:
        print("❌ RE-DECIDE ROLLBACK FUZZ FAILED")
        for f in fails:
            print(f"  - {f}")
        sys.exit(1)
    print("✅ RE-DECIDE ROLLBACK FUZZ PASSED — re-decide leaves no phantom turn, always valid")
    sys.exit(0)


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 12
    asyncio.run(main(n))
