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
    """Models the `StreamWriter` surface `_write_raw` uses (incl. `is_closing`)."""

    def __init__(self, closing=False, raise_on_drain=None):
        self.writes = []
        self._closing = closing
        self._raise_on_drain = raise_on_drain

    def write(self, data: bytes):
        self.writes.append(data)

    def is_closing(self):
        return self._closing

    async def drain(self):
        if self._raise_on_drain is not None:
            raise self._raise_on_drain
        return None


class _FakeProc:
    def __init__(self, returncode=None, stdin=None):
        self.stdin = _FakeStdin() if stdin is None else stdin
        # asyncio subprocesses expose `returncode is None` while alive.
        self.returncode = returncode


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


def test_write_to_an_exited_child_is_dropped_not_raised():
    """A choice answered after the child exited must NOT raise into poke-env's handler.

    The teardown race: `close()` kills the child, but a `|request|` the reader already buffered
    still reaches poke-env, which answers it here. Pre-fix that produced one "Unhandled exception
    in _handle_message / ConnectionResetError" traceback per env — noise that also masks real
    errors. A crash is still surfaced loudly, but by `BridgeSession`'s EOF latch, not by this write.
    """
    client = _make_client(side="p1")
    proc = _FakeProc(returncode=0)  # child already exited
    tag = "battle-gen3ou-9"
    client._procs[tag] = proc
    _run(client.send_message("/choose move 1", tag))
    assert proc.stdin.writes == [], "a write to an exited child must be dropped"

    # Same for a transport that is closing but has not reaped yet.
    closing = _FakeProc(stdin=_FakeStdin(closing=True))
    client._procs[tag] = closing
    _run(client.send_message("/choose move 1", tag))
    assert closing.stdin.writes == []


def test_write_that_races_the_childs_death_swallows_connection_reset():
    """The child can die BETWEEN the liveness check and the drain — that must not raise either."""
    client = _make_client(side="p1")
    proc = _FakeProc(stdin=_FakeStdin(raise_on_drain=ConnectionResetError("Connection lost")))
    tag = "battle-gen3ou-10"
    client._procs[tag] = proc
    _run(client.send_message("/choose move 1", tag))  # must not raise
    assert proc.stdin.writes == [b"CHOOSE p1 move 1\n"], "the write was attempted, then swallowed"


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
