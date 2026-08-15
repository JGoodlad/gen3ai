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

# gen3 test tiers (MEASURED 2026-08-14): 13.0 s / 5 tests
pytestmark = pytest.mark.sim

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


@pytest.mark.integration
def test_concurrent_local_battles_complete():
    """`concurrency>1` overlaps battles (each its own subprocess) but serializes each battle's
    team→creation, mirroring the server. Every battle must still finish with a VALID own-team
    (a 6-mon `_teambuilder_team`) — a corrupted/missing team would mean the shared
    `_current_packed_team` raced, which the start-lock must prevent."""
    teams = _teams()

    async def go():
        p1 = RandomPlayer(
            battle_format="gen3ou", team=Gen3Teambuilder(teams),
            account_configuration=AccountConfiguration("LocConcP1", None),
            start_listening=False, start_timer_on_battle_start=False,
        )
        p2 = RandomPlayer(
            battle_format="gen3ou", team=Gen3Teambuilder(teams),
            account_configuration=AccountConfiguration("LocConcP2", None),
            start_listening=False, start_timer_on_battle_start=False,
        )
        await run_local_battles(p1, p2, 12, concurrency=4)
        return p1, p2

    p1, p2 = asyncio.run(go())

    assert p1.n_finished_battles == 12
    assert p2.n_finished_battles == 12
    assert len(p1._battles) == 12  # 12 distinct, uniquely-tagged battles ran concurrently
    for battle in p1._battles.values():
        assert battle.finished
        assert battle.turn > 0
        assert battle.player_role == "p1"
        # Each battle's own-team was read under the start-lock → a full, valid team (no race).
        assert battle._teambuilder_team is not None
        assert len(battle._teambuilder_team) == 6


class _IsolationProbe(RandomPlayer):
    """A RandomPlayer that asserts the two concurrency-safety invariants eval relies on:

    1. ``choose_move`` is ATOMIC — set a shared scratch attr, then (after the real decision)
       check it's still ours. If a concurrent battle's ``choose_move`` ran mid-call it would
       clobber it. (This is why per-decision ``self._`` state on the real model/bots is safe.)
    2. PER-BATTLE state doesn't cross-contaminate — record each battle's turns keyed by tag;
       each battle must see a monotonic-nondecreasing turn sequence (no other battle's turns).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.seen: dict = {}
        self._scratch = None

    def choose_move(self, battle):
        self._scratch = battle.battle_tag
        order = super().choose_move(battle)
        assert self._scratch == battle.battle_tag, "choose_move was interleaved mid-call!"
        turns = self.seen.setdefault(battle.battle_tag, [])
        assert not turns or battle.turn >= turns[-1], (
            f"per-battle state cross-contaminated: {battle.battle_tag} went "
            f"{turns[-1]} -> {battle.turn}"
        )
        turns.append(battle.turn)
        return order


@pytest.mark.integration
def test_concurrent_per_battle_state_isolation():
    """Concurrent battles on the SAME two players must not corrupt per-battle state — the exact
    property the real eval (per-tag trackers / reward / forensic + atomic choose_move) depends on,
    and that the server eval already relies on. Probe players fail loudly on any violation."""
    teams = _teams()

    async def go():
        p1 = _IsolationProbe(
            battle_format="gen3ou", team=Gen3Teambuilder(teams),
            account_configuration=AccountConfiguration("IsoP1", None), start_listening=False,
        )
        p2 = _IsolationProbe(
            battle_format="gen3ou", team=Gen3Teambuilder(teams),
            account_configuration=AccountConfiguration("IsoP2", None), start_listening=False,
        )
        await run_local_battles(p1, p2, 10, concurrency=5)
        return p1, p2

    p1, p2 = asyncio.run(go())
    # Each side tracked exactly its 10 battles, each a self-consistent monotonic turn sequence.
    assert len(p1.seen) == 10 and len(p2.seen) == 10
    assert all(len(turns) > 0 for turns in p1.seen.values())


@pytest.mark.integration
def test_repeated_calls_use_unique_tags():
    """Regression: calling ``run_local_battles`` repeatedly with the SAME players (the
    chunked / time-budget fuzz pattern) must give every battle a unique tag.

    A per-call ``battle-{fmt}-{index+1}`` scheme reused ``battle-{fmt}-1`` on every call.
    poke-env's ``_create_battle`` then returned the *previous* battle's object for that
    tag (it was still in ``player._battles``), so the new battle was parsed into an object
    that already held a full 6-mon team — and the new battle's first ``|switch|`` of a
    different species overflowed it: ``ValueError: p2's team already has 6 pokemons:
    cannot add p2: Tyranitar to ...`` out of ``get_pokemon``. The fix (a process-global
    battle counter) makes every tag unique, like a real server's room ids.
    """
    teams = _teams()

    async def go():
        p1 = RandomPlayer(
            battle_format="gen3ou", team=Gen3Teambuilder(teams),
            account_configuration=AccountConfiguration("LocBridgeChunkP1", None),
            start_listening=False, start_timer_on_battle_start=False,
        )
        p2 = RandomPlayer(
            battle_format="gen3ou", team=Gen3Teambuilder(teams),
            account_configuration=AccountConfiguration("LocBridgeChunkP2", None),
            start_listening=False, start_timer_on_battle_start=False,
        )
        # Three separate calls reusing the same players — pre-fix the 2nd call reused
        # tag battle-gen3ou-1 and crashed (stale 6-mon team + a differing lead).
        for _ in range(3):
            await run_local_battles(p1, p2, 2)
        return p1, p2

    p1, p2 = asyncio.run(go())

    # 6 distinct battles completed (no stale-battle reuse, no overflow crash).
    assert p1.n_finished_battles == 6
    assert p2.n_finished_battles == 6
    # The core invariant: every battle got a unique tag (no collisions across calls).
    assert len(p1._battles) == 6
    assert len(set(p1._battles.keys())) == 6
    assert all(b.finished for b in p1._battles.values())


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
        if text.startswith("__RECON__"):
            continue  # the battle's reconstruction record — not a side chunk
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
