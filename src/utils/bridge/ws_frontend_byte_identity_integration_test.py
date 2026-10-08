"""THE GATE: the protocol a client received through the front end == the Node bridge's own bytes.

This is the port's byte-differential shape, one layer out. A seeded battle is played through the
front end; the exact command stream it produced is then replayed into the Node
`local_sim_bridge.js`; the per-side protocol TEXT is compared byte for byte. The machinery is
:mod:`utils.bridge.ws_frontend_replay`, which carries the two normalizations and why each is
legitimate.

WHY THIS IS THE TEST THAT MATTERS. The front end sits between the sim and an opponent we do not
control, and both 2026-09-14 de-risks landed on the same warning: the expensive failure of a shim
is not a crash, it is *a slightly different game than the server would have run*, which no win
rate can distinguish from a real result. The only honest answer is a differential, and this is it.

TWO ARMS. The node-vs-node arm isolates the FRONT END (framing, relay, the `rqid` splice) and runs
wherever Node does — no cargo build, so it stays in the routine gate. The rust-vs-node arm adds
the engine the front end actually ships with (`--impl rust`) and SKIPS without a pre-built binary,
exactly as `bridge_impl_parity_test` does, so the unit suite never triggers a build.
"""

from __future__ import annotations

import asyncio
import os

import pytest

from utils.bridge.sim_bridge_bin import _ENV_OVERRIDE
from utils.bridge.ws_frontend import ShowdownFrontEnd
from utils.bridge.ws_frontend_replay import BattleCapture, check_capture
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

pytestmark = pytest.mark.sim

def _prebuilt_rust_available() -> bool:
    override = os.environ.get(_ENV_OVERRIDE)
    if override and os.path.exists(override):
        return True
    from utils.bridge.sim_bridge_bin import expected_bin_path

    return expected_bin_path("sim_bridge").is_file()


async def _one_seeded_battle(impl: str) -> BattleCapture:
    """Play ONE seeded gen3ou battle through the front end and return its repro record."""
    from websockets.asyncio.server import serve

    front = ShowdownFrontEnd(impl=impl, seed_base=20260914, validate_teams=False, capture=True)
    server = await serve(front.handler, "127.0.0.1", 0, max_size=None, ping_interval=None)
    port = server.sockets[0].getsockname()[1]
    uri = f"ws://127.0.0.1:{port}/showdown/websocket"
    teams = [t.strip() for t in TeamLoader().get_all_teams()][:8]
    # Two Rust-stack clients (`main.live`, seeded random policies; two vendored-poke-env `RandomPlayer`s until P6 of
    # the poke-env retirement). What is compared is the SERVER side — the protocol the front end relayed vs the Node
    # bridge's own bytes for the same command stream — so which client drove the battle does not enter the verdict.
    import random

    from main.live.client import ClientConfig, LiveClient
    from main.live.policy import RandomPolicy

    def client(name: str, seed: int) -> LiveClient:
        rng = random.Random(seed)
        packed = [Gen3Teambuilder(t).yield_team() for t in teams]
        return LiveClient(ClientConfig(uri=uri, username=name, battle_format="gen3ou", auth="local"),
                          policy=RandomPolicy(seed), team_fn=lambda: rng.choice(packed))

    a, b = client("WsfeByteA", 5), client("WsfeByteB", 6)
    acc = None
    try:
        await b.connect()
        await a.connect()
        acc = asyncio.ensure_future(b.accept(a.name, 1))
        await asyncio.wait_for(a.challenge(b.name or "WsfeByteB", 1), timeout=300)
        await asyncio.wait_for(acc, timeout=60)
        await front.drain()
        (battle,) = front.captured_battles
        return BattleCapture(tag=battle.tag, seed=battle.seed,
                             commands=list(battle.commands), chunks=list(battle.capture))
    finally:
        if acc is not None and not acc.done():
            acc.cancel()
            await asyncio.gather(acc, return_exceptions=True)
        await a.close()
        await b.close()
        await front.close()
        server.close()
        await server.wait_closed()


async def test_the_front_ends_protocol_is_byte_identical_to_the_node_bridges():
    capture = await _one_seeded_battle("node")
    assert capture.is_seeded, "an unseeded battle cannot be replayed, so it cannot be compared"
    assert len(capture.chunks) > 10, "a battle that relayed almost nothing proves nothing"
    verdict = await check_capture(capture)
    assert verdict is None, verdict


async def test_the_rust_backed_front_end_is_byte_identical_to_the_node_bridge():
    if not _prebuilt_rust_available():
        pytest.skip(f"no pre-built rust sim_bridge (set {_ENV_OVERRIDE} or build src/rust_sim); "
                    "skipping to avoid a cargo build in the routine suite")
    capture = await _one_seeded_battle("rust")
    verdict = await check_capture(capture)
    assert verdict is None, verdict


async def test_an_unseeded_capture_is_refused_rather_than_compared():
    """An inconclusive comparison must never be able to read as a pass.

    Without a pinned seed the replay child mints its OWN dice, so the two streams are different
    battles and every diff is spurious. The gate raises instead of returning a divergence.
    """
    capture = BattleCapture(tag="battle-gen3ou-0", seed=None,
                            commands=['START {"formatid":"gen3ou"}'], chunks=[("p1", "|turn|1")])
    with pytest.raises(ValueError, match="WITHOUT a pinned seed"):
        await check_capture(capture)
