"""Unit tests for BattleStreamClient — no Node, no subprocess.

Verifies the transport-shim contract: outbound choice routing, no-op websocket
ceremony, no-auth login, no `.websocket` attribute, and that `feed` drives
poke-env's `_handle_message`.
"""

import asyncio

from poke_env.concurrency import POKE_LOOP
from poke_env.ps_client.account_configuration import AccountConfiguration

from utils.bridge.battle_stream_client import BattleStreamClient


def _run(coro):
    """Run a coroutine on POKE_LOOP (the loop the client is bound to)."""
    return asyncio.run_coroutine_threadsafe(coro, POKE_LOOP).result()


class _FakeStdin:
    def __init__(self):
        self.writes = []

    def write(self, data: bytes):
        self.writes.append(data)

    async def drain(self):
        return None


class _FakeProc:
    def __init__(self):
        self.stdin = _FakeStdin()


async def _noop_battle_message(_split_messages):
    return None


async def _noop_challenge(_split_message):
    return None


def _make_client(side="p1", on_battle_message=_noop_battle_message):
    return BattleStreamClient(
        AccountConfiguration(f"Test{side}", None),
        side=side,
        on_battle_message=on_battle_message,
        on_update_challenges=_noop_challenge,
        on_challenge_request=_noop_challenge,
        loop=POKE_LOOP,
    )


def test_no_websocket_attribute():
    client = _make_client()
    # poke-env's `hasattr(ps_client, "websocket")` guards must read False so the
    # /leave and VGC-team-sheet branches skip.
    assert not hasattr(client, "websocket")


def test_logged_in_without_handshake():
    client = _make_client()
    assert client.logged_in.is_set()
    # log_in is a no-op that keeps us logged in (no auth POST against a local sim).
    _run(client.log_in(["", "challstr", "x"]))
    assert client.logged_in.is_set()


def test_choose_routes_to_bridge():
    client = _make_client(side="p1")
    proc = _FakeProc()
    tag = "battle-gen3ou-1"
    client._procs[tag] = proc
    _run(client.send_message("/choose move 1", tag))
    assert proc.stdin.writes == [b"CHOOSE p1 move 1\n"]


def test_choose_switch_routes_with_side():
    client = _make_client(side="p2")
    proc = _FakeProc()
    tag = "battle-gen3ou-7"
    client._procs[tag] = proc
    _run(client.send_message("/choose switch Pikachu", tag))
    assert proc.stdin.writes == [b"CHOOSE p2 switch Pikachu\n"]


def test_forfeit_routes_to_forcelose():
    client = _make_client(side="p1")
    proc = _FakeProc()
    tag = "battle-gen3ou-1"
    client._procs[tag] = proc
    _run(client.send_message("/forfeit", tag))
    assert proc.stdin.writes == [b"FORCELOSE p1\n"]


def test_control_messages_are_noops():
    client = _make_client(side="p1")
    proc = _FakeProc()
    tag = "battle-gen3ou-1"
    client._procs[tag] = proc
    for ctrl in ("/timer on", "/leave battle-gen3ou-1", "/utm null",
                 "/challenge foo, gen3ou", "/trn x", "/search gen3ou"):
        _run(client.send_message(ctrl, tag))
    assert proc.stdin.writes == []  # nothing sent to the bridge


def test_choose_with_unregistered_room_is_safe():
    # No proc registered for the room -> silently drop (battle already torn down).
    client = _make_client()
    _run(client.send_message("/choose move 1", "battle-gen3ou-999"))  # no raise


def test_feed_invokes_handle_message():
    seen = []

    async def capture(split_messages):
        seen.append(split_messages)

    client = _make_client(on_battle_message=capture)
    framed = ">battle-gen3ou-1\n|init|battle\n|gen|3"
    _run(client.feed(framed))
    # _handle_message split the room block and routed it to the battle callback.
    assert len(seen) == 1
    assert seen[0][0] == [">battle-gen3ou-1"]
    assert seen[0][1] == ["", "init", "battle"]
