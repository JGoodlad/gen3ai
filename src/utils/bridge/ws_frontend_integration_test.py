"""The front end END TO END: two websocket CLIENTS over a REAL websocket (our Rust-stack `main.live` clients since P6
of the poke-env retirement; vendored-poke-env `Player`s before it).

WHAT THIS COVERS THAT THE UNIT FILE CANNOT. `ws_frontend_test.py` drives the protocol through a
fake socket, so it pins the SHAPES. This file pins that the shapes compose: a real `websockets`
server, two real clients that opened a real connection, the client handshake
(`/trn` → `/utm` → `/challenge`/`/accept`), and two complete gen3ou battles resolved by a real
bridge child. Every integration defect the de-risks recorded — a challenge that never becomes a
battle, a `|request|` that never triggers a decision, a room frame poke-env cannot key a battle
off — presents as a HANG, and a hang is only caught by actually playing.

🚨 **MARKED `sim`, NOT `integration`.** `pytest.ini` defines the axes by what a test NEEDS, and
this one plays real battles through the bridge — which is exactly `sim`'s definition, while
`integration` is explicitly "plays NO battles". Two battles is seconds, so it stays out of `slow`
and inside the routine gate, which is where a transport this many opponents depend on belongs.

The bridge child is `node` so the routine gate never triggers a cargo build; the rust arm is the
byte-identity file's job.
"""

from __future__ import annotations

import asyncio

import pytest


from utils.bridge.ws_frontend import ShowdownFrontEnd
from utils.team_loader import TeamLoader
from utils.teambuilder import Gen3Teambuilder

pytestmark = pytest.mark.sim

_AUTH = "https://play.pokemonshowdown.com/action.php?"


async def _serve(front: ShowdownFrontEnd):
    """Bind an EPHEMERAL port and return (server, uri).

    Port 0 rather than a fixed 9XXX: the suite runs under `-n 2`, and a fixed port turns a
    parallel run into an `EADDRINUSE` flake that reads like a protocol failure.
    """
    from websockets.asyncio.server import serve

    server = await serve(front.handler, "127.0.0.1", 0, max_size=None, ping_interval=None)
    port = server.sockets[0].getsockname()[1]
    return server, f"ws://127.0.0.1:{port}/showdown/websocket"


async def _series(uri: str, teams, n: int):
    """Two Rust-stack clients (`main.live`, seeded random policies) play ``n`` battles over the websocket: one
    challenges, the other accepts. Until P6 of the poke-env retirement these were two vendored-poke-env
    `RandomPlayer`s; the front end's poke-env DIALECT is still exercised end to end by the Metamon anchor reads
    (`main/anchors/anchors_integration_test.py`, upstream poke-env in its own interpreter)."""
    import random

    from main.live.client import ClientConfig, LiveClient
    from main.live.policy import RandomPolicy

    def client(name: str, seed: int) -> LiveClient:
        rng = random.Random(seed)
        packed = [Gen3Teambuilder(t).yield_team() for t in teams]
        return LiveClient(ClientConfig(uri=uri, username=name, battle_format="gen3ou", auth="local"),
                          policy=RandomPolicy(seed), team_fn=lambda: rng.choice(packed))

    a, b = client("WsfeItA", 11), client("WsfeItB", 22)
    acc = None
    try:
        await b.connect()
        await a.connect()
        acc = asyncio.ensure_future(b.accept(a.name, n))
        res_a = await asyncio.wait_for(a.challenge(b.name or "WsfeItB", n), timeout=300)
        res_b = await asyncio.wait_for(acc, timeout=60)
    finally:
        if acc is not None and not acc.done():
            acc.cancel()
            await asyncio.gather(acc, return_exceptions=True)
        await a.close()
        await b.close()
    return res_a, res_b


@pytest.fixture(scope="module")
def pool_teams():
    return [t.strip() for t in TeamLoader().get_all_teams()][:12]


async def test_two_clients_complete_a_two_battle_series_over_the_websocket(pool_teams):
    front = ShowdownFrontEnd(impl="node", seed_base=20260914, validate_teams=False)
    server, uri = await _serve(front)
    try:
        res_a, res_b = await _series(uri, pool_teams, 2)
    finally:
        await front.close()
        server.close()
        await server.wait_closed()

    assert len(res_a) == 2 and len(res_b) == 2
    # Both sides must agree on every result — a front end that mis-routed a `|win|` would show up
    # here as two clients that each think they won, which no amount of "it ran" would reveal.
    for ra, rb in zip(res_a, res_b):
        assert ra.winner == rb.winner, (ra, rb)
        assert not (ra.won and rb.won)
        assert ra.turns > 0, "a battle that finished on turn 0 never actually played"


async def test_a_battle_is_backed_by_exactly_one_bridge_child_that_is_reaped(pool_teams):
    """One battle = one child, and the child is gone when the battle is. A leaked child is the
    failure a long series would only reveal as a box that has run out of processes."""
    front = ShowdownFrontEnd(impl="node", seed_base=7, validate_teams=False, capture=True)
    server, uri = await _serve(front)
    try:
        await _series(uri, pool_teams, 1)
        await front.drain()
        (battle,) = front.captured_battles
        assert battle.finished
        assert battle.proc.returncode is not None, "the bridge child outlived its battle"
        # The capture is the byte-identity gate's input; assert it is actually populated here so
        # a broken recorder fails in the cheap test rather than in the differential.
        assert battle.commands and battle.commands[0].startswith("START ")
        assert any(chunk.startswith("|request|") for _, chunk in battle.capture)
    finally:
        await front.close()
        server.close()
        await server.wait_closed()


async def test_a_rejected_team_never_becomes_a_battle(pool_teams):
    """The metamon de-risk's H1 shape, made loud: a team the validator refuses must stop at the
    popup instead of starting a battle Showdown would have rejected — which is a HANG, not an
    error, and burned a 120-game attempt at game 8."""
    front = ShowdownFrontEnd(impl="node", validate_teams=True)
    front._team_cache[("gen3ou", "Airmure (Skarmory)  |||keeneye|protect|Calm|||||")] = (
        False, ['The Pokemon "airmureskarmory" does not exist.'])
    server, _uri = await _serve(front)
    from utils.bridge.ws_frontend import _Conn

    class _WS:
        def __init__(self):
            self.sent = []

        async def send(self, text):
            self.sent.append(text)

    try:
        conn = _Conn(ws=_WS(), guest_n=1)
        await front._cmd_trn(conn, "Alpha,0,x")
        await front._cmd_utm(conn, "Airmure (Skarmory)  |||keeneye|protect|Calm|||||")
        assert conn.team is None
        other = _Conn(ws=_WS(), guest_n=2)
        await front._cmd_trn(other, "Beta,0,x")
        await front._cmd_challenge(conn, "Beta, gen3ou")
        assert front.challenges == {} and front.battles == {}
    finally:
        await front.close()
        server.close()
        await server.wait_closed()
