"""The server this tool starts — and the two ports it must never touch.

**8001 carries the live training run.** Dropping it crashes every poke-env websocket at once, and
8000 is the shared dev server one digit away. The refusal is in CODE rather than in a document,
because a document cannot fail a test, and it is tested on EVERY path that can name a port —
including the one where the caller supplies a whole URI, which is the same mistake with better
camouflage.

Nothing here starts a Showdown server: :func:`ShowdownServer.argv` is pure, and the lifecycle is
exercised for real by the integration test.
"""
from __future__ import annotations

import socket

import pytest

from main.anchors.server import (
    RESERVED_PORTS,
    ServerError,
    ShowdownServer,
    pick_port,
    port_of,
    refuse_reserved,
    server_uri,
)


@pytest.mark.parametrize("port", sorted(RESERVED_PORTS))
def test_the_reserved_ports_are_refused_and_the_message_says_which(port: int) -> None:
    with pytest.raises(ServerError) as excinfo:
        refuse_reserved(port)
    message = str(excinfo.value)
    assert str(port) in message
    assert RESERVED_PORTS[port] in message


@pytest.mark.parametrize("port", sorted(RESERVED_PORTS))
def test_a_reserved_port_reached_through_a_uri_is_refused_too(port: int) -> None:
    """The `--server-uri` seam must not be a way around the guard."""
    uri = f"ws://localhost:{port}/showdown/websocket"
    assert port_of(uri) == port
    with pytest.raises(ServerError):
        refuse_reserved(port_of(uri))


def test_constructing_the_server_on_a_reserved_port_raises_before_anything_starts() -> None:
    for port in RESERVED_PORTS:
        with pytest.raises(ServerError):
            ShowdownServer(port)


@pytest.mark.parametrize("uri, expect", [
    ("ws://localhost:9500/showdown/websocket", 9500),
    ("wss://sim3.psim.us/showdown/websocket", None),
    ("ws://127.0.0.1:9599", 9599),
    ("nonsense", None),
])
def test_port_of_reads_the_port_or_honestly_says_none(uri: str, expect) -> None:
    assert port_of(uri) == expect


def test_pick_port_stays_inside_the_configured_range_and_skips_the_reserved_ones() -> None:
    port = pick_port([9500, 9599])
    assert 9500 <= port <= 9599
    assert port not in RESERVED_PORTS


def test_pick_port_would_never_hand_back_a_reserved_port_even_if_the_range_covered_it() -> None:
    """A config that widened the range must not be able to reach 8001."""
    port = pick_port([8000, 8010])
    assert port not in RESERVED_PORTS


def test_pick_port_does_not_return_a_port_something_is_listening_on() -> None:
    """Two anchor reads sharing one server would silently MIX two measurements."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        taken = held.getsockname()[1]
        assert pick_port([taken, taken + 5]) != taken


def test_pick_port_refuses_rather_than_reusing_when_the_range_is_full() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        taken = held.getsockname()[1]
        with pytest.raises(ServerError) as excinfo:
            pick_port([taken, taken])
        assert "no free port" in str(excinfo.value)


def test_the_argv_starts_OUR_pinned_submodule_with_the_port_positional() -> None:
    """Showdown has no `--port` flag; the port is positional. `--no-security` is what lets guest
    accounts battle locally, and it is why the server writes no battle logs (de-risk H6)."""
    argv = ShowdownServer(9501, node="node").argv()
    assert argv[0] == "node"
    assert argv[1].endswith("deps/pokemon-showdown/pokemon-showdown")
    assert argv[2:] == ["start", "--no-security", "9501"]


def test_server_uri_is_the_shape_both_clients_dial() -> None:
    assert server_uri(9501) == "ws://localhost:9501/showdown/websocket"


def test_stop_is_idempotent_on_a_server_that_was_never_started() -> None:
    """`stop()` runs in a `finally` on every path, including the ones that failed before start."""
    srv = ShowdownServer(9502)
    srv.stop()
    srv.stop()
    assert srv.pid is None
