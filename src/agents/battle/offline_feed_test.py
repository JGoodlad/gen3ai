"""The offline feed's dispatch mirror == the live ``Player`` dispatch, on a real bridge battle.

``offline_feed.feed_chunk`` claims to route a per-side chunk exactly as
``Player._handle_battle_message`` does. This plays ONE reproducible gen3ou battle for real over the
rust bridge (two seeded random players whose battle class is ``Gen3Battle``, pinned teams, fixed
sim seed, concurrency 1), captures every per-side chunk, re-feeds each viewer's chunks into a fresh
``Gen3Battle`` offline, and asserts the two event logs are identical, event by event, every field
including ``raw`` — plus the conservation report, the turn and the result.
"""

from __future__ import annotations

import asyncio

import pytest
from poke_env import AccountConfiguration
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

from agents.battle.gen3_battle import Gen3Battle
from agents.battle.offline_feed import feed_chunk, new_battle, player_names

pytestmark = [pytest.mark.sim, pytest.mark.integration]

_FIELDS = ("seq", "turn", "kind", "side", "actor_species", "target_species", "value", "raw")


def play_live(key: int, tag: str = "Of"):
    """One reproducible battle over the rust bridge: ``(battle_p1, battle_p2, chunks)``."""
    from agents.training.obs_roundtrip_fuzz_test import SeededRandomPlayer
    from utils.bridge.local_battle_runner import run_local_battles
    from utils.team_loader import TeamLoader

    pool = TeamLoader().get_all_teams()
    t1, t2 = pool[key % len(pool)], pool[(key + 1) % len(pool)]
    common = dict(battle_format="gen3ou", server_configuration=LocalhostServerConfiguration,
                  start_listening=False, max_concurrent_battles=1, battle_class=Gen3Battle)
    p1 = SeededRandomPlayer(rng_seed=1000 + key, team=t1,
                            account_configuration=AccountConfiguration(f"{tag}a{key}", "pw"),
                            **common)
    p2 = SeededRandomPlayer(rng_seed=2000 + key, team=t2,
                            account_configuration=AccountConfiguration(f"{tag}b{key}", "pw"),
                            **common)
    sink: list = []
    asyncio.run(run_local_battles(p1, p2, 1, seed=[11 + key, 22 + key, 33 + key, 44 + key],
                                  impl="rust", chunk_sink=sink))
    (b1,) = list(p1.battles.values())
    (b2,) = list(p2.battles.values())
    return b1, b2, sink


def replay_offline(chunks, viewer: str) -> Gen3Battle:
    lines = [ln for side, c in chunks if side == viewer for ln in c.split("\n")]
    b = new_battle(viewer, player_names(lines))
    # The bridge TRANSPORT (not the sim) prepends the room's `|init|battle` line to a side's first
    # chunk (`local_battle_runner._frame`); it is a CONTROL line, so it moves the conservation
    # counts and nothing else. Feed it too, so the comparison covers every line the live battle saw.
    feed_chunk(b, "|init|battle")
    for side, c in chunks:
        if side == viewer:
            feed_chunk(b, c)
    return b


@pytest.mark.parametrize("key", [3])
def test_the_offline_feed_rebuilds_the_live_event_log_exactly(key):
    b1, b2, chunks = play_live(key)
    for viewer, live in (("p1", b1), ("p2", b2)):
        off = replay_offline(chunks, viewer)
        assert len(live.events) > 100, "a real battle — the gate would otherwise be vacuous"
        assert len(off.events) == len(live.events), viewer
        for a, b in zip(live.events, off.events):
            for f in _FIELDS:
                assert getattr(a, f) == getattr(b, f), (viewer, f, a, b)
        assert off.turn == live.turn and off.won == live.won and off.finished == live.finished
        assert off.assert_conservation() == live.assert_conservation()
