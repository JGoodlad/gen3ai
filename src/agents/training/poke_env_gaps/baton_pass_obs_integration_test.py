"""END-TO-END gate for the Baton Pass carry-over: a REAL battle, read at the OBSERVATION.

`src/poke_env/battle/baton_pass_carryover_test.py` pins the parser with hand-fed protocol lines.
This one refuses to trust that the hand-fed lines are what the sim emits: it plays a scripted
battle through the in-process bridge (Celebi Calm Mind ×2 + Substitute, then Baton Pass into
Charizard) and asserts the passed state survives all the way to the bytes the model is trained on
— poke-env's `Pokemon`, our `LiveView`, and the observation's active-context boosts block.

The defect it guards (2026-08-23) was invisible to every existing gate precisely because those
three agreed with each other: they were all reading the same wrong number from poke-env, while the
SIM had `spa +2 / spd +2`. Only an oracle OUTSIDE that chain can catch it — here, the recorded
protocol, which names the pass and therefore says what the boosts must be.

Run directly (no server) or under pytest as a `sim` test:
    python src/agents/training/poke_env_gaps/baton_pass_obs_integration_test.py
"""
from __future__ import annotations

import asyncio
import sys
import time
from typing import Any, Dict, List

import pytest

from agents.battle.gen3_battle import Gen3Battle
from agents.observation.constants import BOOSTS_DIM, OFFSET_CONTEXT
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from poke_env import AccountConfiguration
from poke_env.player.player import Player
from utils.bridge.local_battle_runner import run_local_battles
from utils.teambuilder import Gen3Teambuilder

pytestmark = pytest.mark.sim

PASSER_TEAM = """\
Celebi @ Leftovers
Ability: Natural Cure
EVs: 252 HP / 252 SpA / 4 Spe
Modest Nature
- Calm Mind
- Baton Pass
- Substitute
- Recover

Charizard @ Petaya Berry
Ability: Blaze
EVs: 4 HP / 252 SpA / 252 Spe
Timid Nature
- Fire Blast
- Hidden Power [Grass]
- Substitute
- Focus Punch

Blissey @ Leftovers
Ability: Natural Cure
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Soft-Boiled
- Seismic Toss
- Toxic
- Protect
"""

# A punchbag that only ever heals, so the pass is never disrupted and the battle is quiet.
PUNCHBAG_TEAM = """\
Blissey @ Leftovers
Ability: Natural Cure
EVs: 252 HP / 252 Def / 4 SpD
Bold Nature
- Soft-Boiled
- Protect
- Seismic Toss
- Toxic

Snorlax @ Leftovers
Ability: Immunity
EVs: 252 HP / 252 Def / 4 SpD
Impish Nature
- Rest
- Body Slam
- Protect
- Curse

Skarmory @ Leftovers
Ability: Keen Eye
EVs: 252 HP / 252 Def / 4 Spe
Impish Nature
- Rest
- Drill Peck
- Protect
- Roar
"""

_STAT_ORDER = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")


def decode_obs_boosts(vec, offset: int = OFFSET_CONTEXT) -> Dict[str, int]:
    """Read the active-context BOOSTS block back out of a real observation vector."""
    block = vec[offset : offset + BOOSTS_DIM]
    return {
        _STAT_ORDER[i]: round(float(block[2 * i] * 6 - block[2 * i + 1] * 6))
        for i in range(7)
        if abs(block[2 * i]) + abs(block[2 * i + 1]) > 1e-6
    }


class _ScriptedPasser(Player):
    """Substitute, Calm Mind twice, Baton Pass into Charizard, then attack."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self._encoder = Gen3ObservationEncoder(mappings=load_mappings())
        self._setup_moves_used = 0
        self.passed = False
        self.rows: List[Dict[str, Any]] = []
        self.saw_baton_pass_line = False

    async def _handle_battle_message(self, split_messages):
        for msg in split_messages[1:]:
            if len(msg) >= 2 and msg[1] == "switch":
                if any("baton pass" in part.lower() for part in msg[5:]):
                    self.saw_baton_pass_line = True
        await super()._handle_battle_message(split_messages)

    def _snapshot(self, battle) -> None:
        active = battle.active_pokemon
        live = battle.strict_view().live.ours.active
        self.rows.append(
            {
                "turn": battle.turn,
                "species": active.species if active else None,
                "pokeenv": {k: v for k, v in (active.boosts or {}).items() if v},
                "liveview": dict(live.boosts) if live else {},
                "liveview_volatiles": sorted(live.volatiles) if live else [],
                "obs": decode_obs_boosts(self._encoder.encode(battle)),
                "after_pass": self.passed,
            }
        )

    def choose_move(self, battle):
        if battle.force_switch:
            for mon in battle.available_switches:
                if mon.species == "charizard":
                    self.passed = True
                    return self.create_order(mon)
            return self.choose_random_move(battle)

        self._snapshot(battle)
        active = battle.active_pokemon
        by_id = {m.id: m for m in battle.available_moves}
        if active is not None and active.species == "celebi" and not self.passed:
            if self._setup_moves_used == 0 and "substitute" in by_id:
                self._setup_moves_used += 1
                return self.create_order(by_id["substitute"])
            if self._setup_moves_used < 3 and "calmmind" in by_id:
                self._setup_moves_used += 1
                return self.create_order(by_id["calmmind"])
            if "batonpass" in by_id:
                return self.create_order(by_id["batonpass"])
        if "fireblast" in by_id:
            return self.create_order(by_id["fireblast"])
        return self.choose_random_move(battle)


class _Punchbag(Player):
    def choose_move(self, battle):
        for move in battle.available_moves:
            if move.id == "softboiled":
                return self.create_order(move)
        return self.choose_random_move(battle)


async def _play() -> _ScriptedPasser:
    tag = f"{int(time.time() * 1000) % 1_000_000}"
    passer = _ScriptedPasser(
        account_configuration=AccountConfiguration(f"BPa{tag}", "password"),
        battle_format="gen3ou",
        team=Gen3Teambuilder(PASSER_TEAM),
        battle_class=Gen3Battle,
        start_listening=False,
        max_concurrent_battles=1,
        log_level=40,
    )
    punchbag = _Punchbag(
        account_configuration=AccountConfiguration(f"BPb{tag}", "password"),
        battle_format="gen3ou",
        team=Gen3Teambuilder(PUNCHBAG_TEAM),
        battle_class=Gen3Battle,
        start_listening=False,
        max_concurrent_battles=1,
        log_level=40,
    )
    await run_local_battles(
        passer, punchbag, 1, battle_format="gen3ou", seed=[1, 2, 3, 4]
    )
    return passer


def test_baton_passed_boosts_reach_the_observation():
    passer = asyncio.run(_play())

    assert passer.saw_baton_pass_line, (
        "the scripted battle never produced a `[from] Baton Pass` switch — the scenario "
        "broke, so a pass on the assertions below would mean nothing"
    )
    before = [r for r in passer.rows if r["species"] == "celebi" and not r["after_pass"]]
    after = [r for r in passer.rows if r["species"] == "charizard" and r["after_pass"]]
    assert before and after, f"scenario did not run to completion: {passer.rows[:6]}"

    passed_stages = before[-1]["obs"]
    assert passed_stages.get("spa", 0) >= 2, (
        f"the passer never reached +2 Special Attack; saw {passed_stages}"
    )

    entrant = after[0]
    for layer in ("pokeenv", "liveview", "obs"):
        assert entrant[layer].get("spa", 0) == passed_stages["spa"], (
            f"Baton Pass carry-over lost at the {layer} layer: entrant={entrant[layer]}, "
            f"passer={passed_stages}. Showdown never re-emits the stages, so the client is "
            f"the only thing that can carry them."
        )
    assert "substitute" in entrant["liveview_volatiles"], (
        f"the passed Substitute did not reach the entrant: {entrant['liveview_volatiles']}"
    )


if __name__ == "__main__":
    try:
        test_baton_passed_boosts_reach_the_observation()
    except AssertionError as exc:  # pragma: no cover - script path
        print(f"FAIL: {exc}")
        sys.exit(1)
    print("PASS — Baton Pass boosts + Substitute reached the observation")
