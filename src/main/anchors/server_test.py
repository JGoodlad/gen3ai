"""The servers this tool starts — and the two ports it must never touch.

🚨 **There are TWO transports and the default is NOT Node.** `FrontEndServer` runs
`utils.bridge.ws_frontend --impl rust` in its own subprocess and no Node server exists for the
life of the read; `ShowdownServer` is the explicit `--server node` opt-out. Both inherit ONE
lifecycle (`ManagedServer`): start, PID-only stop, and a readiness that means ANSWERING.

**8001 carries the live training run.** Dropping it crashes every poke-env websocket at once, and
8000 is the shared dev server one digit away. The refusal is in CODE rather than in a document,
because a document cannot fail a test, and it is tested on EVERY path that can name a port —
including the one where the caller supplies a whole URI, which is the same mistake with better
camouflage.

Nothing here starts a Showdown server: :func:`ShowdownServer.argv` is pure, and the lifecycle is
exercised for real by the integration test.
"""
from __future__ import annotations

import os
import socket

import pytest

from main.anchors.server import (
    RESERVED_PORTS,
    FrontEndServer,
    ManagedServer,
    ServerError,
    ShowdownServer,
    build_server,
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


@pytest.mark.parametrize("factory", [ShowdownServer, FrontEndServer,
                                     lambda p: build_server("rust", p),
                                     lambda p: build_server("node", p)])
def test_constructing_a_server_on_a_reserved_port_raises_before_anything_starts(factory) -> None:
    """Both transports, and the factory both go through: the guard is on the BASE class, so a
    third transport cannot arrive without it."""
    for port in RESERVED_PORTS:
        with pytest.raises(ServerError):
            factory(port)


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


# ------------------------------------------------------------------- the Rust front end (default)
def test_the_front_end_argv_runs_the_in_repo_module_with_the_rust_bridge() -> None:
    """🚨 NO NODE. The whole point of `--server rust`: the transport is our own module and each
    battle is backed by a `sim_bridge` child, so nothing named `node` appears in the command."""
    argv = FrontEndServer(9601, python="/usr/bin/python3").argv()
    assert argv[:5] == ["/usr/bin/python3", "-m", "utils.bridge.ws_frontend", "--port", "9601"]
    assert argv[argv.index("--impl") + 1] == "rust"
    assert "node" not in " ".join(argv)


def test_the_front_end_child_gets_src_on_its_pythonpath_first() -> None:
    """`python -m` in a WORKTREE: an editable install would otherwise resolve the module out of
    the MAIN checkout, and the read would be about a tree nobody edited."""
    from utils.paths import src_root

    env = FrontEndServer(9601).env()
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(src_root())


def test_the_front_end_announces_itself_and_that_is_part_of_readiness() -> None:
    """A TCP answer is not the front end's own claim to be serving. Its READY line is, and
    `wait_ready` requires BOTH — a client that dials a half-started server hangs."""
    assert FrontEndServer(9601).ready_marker == "[ws_frontend] READY"
    assert ShowdownServer(9601).ready_marker is None


def test_a_missing_ready_line_keeps_the_server_unready(tmp_path) -> None:
    srv = FrontEndServer(9601, log_path=tmp_path / "ws_frontend.log")
    (tmp_path / "ws_frontend.log").write_text("2026-09-20 INFO ws_frontend starting\n")
    assert srv._announced_ready() is False
    (tmp_path / "ws_frontend.log").write_text("[ws_frontend] READY ws://127.0.0.1:9601/x\n")
    assert srv._announced_ready() is True


def test_a_server_that_announces_itself_is_never_TCP_PROBED(tmp_path, monkeypatch) -> None:
    """🚨 A bare TCP connect against a `websockets` server logs `opening handshake failed` with a
    three-deep traceback — a FABRICATED ERROR in the transport's own log, in the one place the
    front end's validation reads ("0 ERRORs over 200 battles"). Measured 2026-09-20, one per
    start, and this is the guard against it coming back."""
    srv = FrontEndServer(9601, log_path=tmp_path / "ws_frontend.log")
    probed = []
    monkeypatch.setattr(srv, "_answers_tcp", lambda: probed.append(1) or True)
    (tmp_path / "ws_frontend.log").write_text("[ws_frontend] READY ws://127.0.0.1:9601/x\n")
    srv.wait_ready()
    assert probed == []


def test_a_server_that_announces_NOTHING_is_still_TCP_PROBED(tmp_path, monkeypatch) -> None:
    """The Node server prints no marker we parse, so the connect is all the evidence there is."""
    srv = ShowdownServer(9601, log_path=tmp_path / "showdown.log")
    probed = []
    monkeypatch.setattr(srv, "_answers_tcp", lambda: bool(probed.append(1)) or True)
    srv.wait_ready()
    assert probed == [1]


def test_the_two_transports_write_differently_named_logs(tmp_path) -> None:
    """A read's output directory says which transport served it before anything is parsed."""
    assert build_server("rust", 9601, out_dir=tmp_path).log_path == tmp_path / "ws_frontend.log"
    assert build_server("node", 9601, out_dir=tmp_path).log_path == tmp_path / "showdown.log"


def test_the_version_stamp_names_the_transport_and_for_rust_the_bridge_binary(
        monkeypatch) -> None:
    """A win rate that did not say which stack produced it cannot be compared with one that did —
    and `POKESIM_SIM_BRIDGE_BIN` is how two different ports end up serving two reads."""
    monkeypatch.setenv("POKESIM_SIM_BRIDGE_BIN", "/tmp/sim_bridge_under_test")
    version = FrontEndServer(9601).version()
    assert version.startswith("ws_frontend@") and "/tmp/sim_bridge_under_test" in version
    assert ShowdownServer(9601).version().startswith("showdown:")


def test_an_unknown_server_kind_is_refused_by_name() -> None:
    with pytest.raises(ServerError) as excinfo:
        build_server("wasm", 9601)
    assert "wasm" in str(excinfo.value)


def test_build_server_returns_the_shared_lifecycle_either_way() -> None:
    for kind in ("rust", "node"):
        srv = build_server(kind, 9601)
        assert isinstance(srv, ManagedServer)
        srv.stop()          # idempotent on a server that never started, on BOTH transports
        assert srv.pid is None
