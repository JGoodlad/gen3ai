"""Integration tests for the local BattleStream bridge — require Node + deps/pokemon-showdown.

Marked `integration`: they spawn the real `local_sim_bridge.js` (no live Showdown
server needed). Two checks:
  1. A full battle driven through the real poke-env pipeline via `run_local_battles`
     completes, resolves player roles, and produces a winner.
  2. The bridge is reproducible: the same seed + same teams + same (default) choices
     yields a byte-identical protocol stream.
"""

import asyncio
import base64
import json
from pathlib import Path

import pytest

from poke_env import AccountConfiguration
from poke_env.player import RandomPlayer

from utils.bridge.local_battle_runner import run_local_battles
from utils.team_loader.loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

_BRIDGE = str(Path(__file__).parent / "local_sim_bridge.js")


def _teams():
    loader = TeamLoader()
    return loader.get_sample_teams() or loader.get_all_teams()


@pytest.mark.integration
def test_local_battles_complete():
    teams = _teams()

    async def go():
        p1 = RandomPlayer(
            battle_format="gen3ou", team=Gen3Teambuilder(teams),
            account_configuration=AccountConfiguration("LocBridgeP1", None),
            start_listening=False, start_timer_on_battle_start=False,
        )
        p2 = RandomPlayer(
            battle_format="gen3ou", team=Gen3Teambuilder(teams),
            account_configuration=AccountConfiguration("LocBridgeP2", None),
            start_listening=False, start_timer_on_battle_start=False,
        )
        await run_local_battles(p1, p2, 3)
        return p1, p2

    p1, p2 = asyncio.run(go())

    assert p1.n_finished_battles == 3
    assert p2.n_finished_battles == 3
    assert p1.n_won_battles + p2.n_won_battles == 3  # every battle had a winner
    for battle in p1._battles.values():
        assert battle.finished
        assert battle.player_role == "p1"   # role resolved from |player| vs username
        assert battle.turn > 0


async def _drive_default(packed1: str, packed2: str, seed):
    """Drive one bridge battle answering every request with `default`; return the
    protocol stream split per side. The global p1-vs-p2 interleaving is a legit
    async race (two side readers), so we compare each side's own sequence — which
    is deterministic given seed + teams + choices."""
    proc = await asyncio.create_subprocess_exec(
        "node", _BRIDGE,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    start = {"formatid": "gen3ou", "seed": seed,
             "p1": {"name": "P1", "team": packed1},
             "p2": {"name": "P2", "team": packed2}}
    proc.stdin.write((f"START {json.dumps(start)}\n").encode())
    await proc.stdin.drain()

    by_side = {"p1": [], "p2": []}
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        text = line.decode().rstrip("\n")
        if text == "__END__":
            break
        assert not text.startswith("__ERR__"), base64.b64decode(text[8:]).decode()
        side, b64 = text.split(" ", 1)
        chunk = base64.b64decode(b64).decode("utf-8")
        by_side[side].append(chunk)
        # Answer real requests with `default`; skip "wait" requests (the sim issues
        # these to the non-acting side during a forced switch — choosing then errors).
        if "|request|" in chunk and '"wait":true' not in chunk:
            proc.stdin.write((f"CHOOSE {side} default\n").encode())
            await proc.stdin.drain()
    await proc.wait()
    return by_side


@pytest.mark.integration
def test_bridge_seed_reproducible():
    teams = _teams()
    packed1 = Gen3Teambuilder(teams[0]).yield_team()
    packed2 = Gen3Teambuilder(teams[1] if len(teams) > 1 else teams[0]).yield_team()
    seed = [1234, 5678, 9012, 3456]

    run1 = asyncio.run(_drive_default(packed1, packed2, seed))
    run2 = asyncio.run(_drive_default(packed1, packed2, seed))

    def strip_timestamps(chunks):
        # |t:|<unix-time> is a real-time stamp (poke-env ignores it), not battle state.
        return [
            "\n".join(l for l in c.split("\n") if not l.startswith("|t:|"))
            for c in chunks
        ]

    # Each side's own protocol sequence is deterministic for a fixed seed+teams.
    assert strip_timestamps(run1["p1"]) == strip_timestamps(run2["p1"])
    assert strip_timestamps(run1["p2"]) == strip_timestamps(run2["p2"])
    assert any("|win|" in c or "|tie|" in c for c in run1["p1"])
