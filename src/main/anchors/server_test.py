"""The servers this tool starts — and the two ports it must never touch.

🚨 **There are TWO transports and the default is NOT Node.** `FrontEndServer` runs
`utils.bridge.ws_frontend --impl rust` in its own subprocess and no Node server exists for the
life of the read; `ShowdownServer` is the explicit `--server node` opt-out. Both inherit ONE
lifecycle (`ManagedServer`): start, PID-only stop, and a readiness that means ANSWERING.

**8001 is the reserved training-tools server and 8000 the shared dev server** (training itself is
in-process on the Rust core and connects to neither). The refusal is in CODE rather than in a document,
because a document cannot fail a test, and it is tested on EVERY path that can name a port —
including the one where the caller supplies a whole URI, which is the same mistake with better
camouflage.

Nothing here starts a Showdown server: :func:`ShowdownServer.argv` is pure, and the lifecycle is
exercised for real by the integration test.
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
import threading

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
    release_port,
    server_uri,
)
from utils import ports
from utils.contention import scale_timeout


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


def test_pick_port_stays_inside_the_configured_range_and_skips_the_reserved_ones(tmp_path) -> None:
    port = pick_port([9500, 9599], tmp_path)
    assert 9500 <= port <= 9599
    assert port not in RESERVED_PORTS


def test_pick_port_would_never_hand_back_a_reserved_port_even_if_the_range_covered_it(tmp_path) -> None:
    """A config that widened the range must not be able to reach 8001 (nor leave a lock file for it)."""
    port = pick_port([8000, 8010], tmp_path)
    assert port not in RESERVED_PORTS
    assert not {f"{p}.lock" for p in RESERVED_PORTS} & {p.name for p in tmp_path.iterdir()}


def test_pick_port_does_not_return_a_port_something_is_listening_on(tmp_path) -> None:
    """Two anchor reads sharing one server would silently MIX two measurements."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        taken = held.getsockname()[1]
        assert pick_port([taken, taken + 5], tmp_path) != taken


def test_pick_port_refuses_rather_than_reusing_when_the_range_is_full(tmp_path) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as held:
        held.bind(("127.0.0.1", 0))
        held.listen(1)
        taken = held.getsockname()[1]
        with pytest.raises(ServerError) as excinfo:
            pick_port([taken, taken], tmp_path)
        assert "no free" in str(excinfo.value)


# -- the claim: a pick is not released until the server it is for has had its chance to bind ---------------------
def test_two_concurrent_pickers_never_get_the_same_port(tmp_path, monkeypatch) -> None:
    """Bind-and-release gave the SAME port to every picker that probed before any server bound it (the
    `peer_patience` pair both took 9500 under `-n 2`; D1 look 1's two supervisors both took 9540). Every probe
    here passes, as it does in that window, so only the CLAIM can tell the pickers apart."""
    monkeypatch.setattr(ports, "bindable", lambda *a, **k: True)
    n, lo = 8, 9680
    barrier, got, errors = threading.Barrier(n), [], []

    def picker() -> None:
        try:
            barrier.wait(timeout=30)
            got.append(pick_port([lo, lo + n + 1], tmp_path))
        except Exception as exc:  # noqa: BLE001 - reported below, never swallowed
            errors.append(exc)

    threads = [threading.Thread(target=picker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=60)
    assert not errors, errors
    assert len(got) == n and len(set(got)) == n, f"two pickers took one port: {sorted(got)}"
    assert all(lo <= p <= lo + n + 1 for p in got)


def test_a_claimed_port_is_refused_to_the_next_picker_until_it_is_released(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(ports, "bindable", lambda *a, **k: True)       # the server has not bound it yet
    first = pick_port([9690, 9691], tmp_path)
    second = pick_port([9690, 9691], tmp_path)
    assert {first, second} == {9690, 9691}
    with pytest.raises(ServerError, match="no free, unclaimed port"):
        pick_port([9690, 9691], tmp_path)
    release_port(first)
    assert pick_port([9690, 9691], tmp_path) == first


def test_a_pick_in_another_process_holds_until_that_process_ends(tmp_path) -> None:
    """Two SESSIONS, not two threads: a child picks the only port of a one-port range and keeps living; the claim
    outlives the pick (nothing has bound the port, so a bind probe alone calls it free) and dies with the process."""
    lo = ports.first_bindable(9670, 9679)
    assert lo is not None, "no bindable port in 9670-9679: the box is holding the whole range"
    code = (f"import sys; from main.anchors.server import pick_port; print(pick_port([{lo}, {lo}], sys.argv[1]), "
            "flush=True); sys.stdin.read()")
    child = subprocess.Popen([sys.executable, "-c", code, str(tmp_path)], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, text=True)
    try:
        assert int(child.stdout.readline()) == lo
        with pytest.raises(ServerError, match="no free, unclaimed port"):
            pick_port([lo, lo], tmp_path)
    finally:
        child.stdin.close()
        child.wait(timeout=scale_timeout(30))
    assert pick_port([lo, lo], tmp_path) == lo, "the claim must end with the process that took it"


def test_a_pick_cannot_take_a_port_a_kakuna_lane_has_claimed(tmp_path) -> None:
    """The lanes claim through `kakuna.claim_port`; a standalone read and a test session claim in the SAME directory
    by default, so a port the lane holds (and has not bound yet) is passed over."""
    from main.x35 import kakuna

    assert kakuna.PORT_LOCK_DIR == ports.PORT_LOCK_DIR
    import main.anchors.server as server
    assert server.PORT_LOCK_DIR == ports.PORT_LOCK_DIR
    lo = ports.first_bindable(9660, 9669)
    assert lo is not None
    held = ports.claim(lo, lo, tmp_path, host=kakuna.PORT_HOST, reuse_addr=False)
    assert held is not None
    try:
        with pytest.raises(ServerError, match="no free, unclaimed port"):
            pick_port([lo, lo], tmp_path)
    finally:
        held[1].close()
    assert pick_port([lo, lo], tmp_path) == lo


def test_an_unwritable_lock_dir_fails_closed_rather_than_picking_unclaimed(tmp_path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(ServerError, match="cannot take a port claim"):
        pick_port([9660, 9669], blocker / "locks")


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


# ------------------------------------------------- P3: the front end IN this process
@pytest.mark.parametrize("port", sorted(RESERVED_PORTS))
def test_the_in_process_front_end_refuses_the_reserved_ports_too(port: int) -> None:
    from main.anchors.server import InProcessFrontEnd

    with pytest.raises(ServerError, match=str(port)):
        InProcessFrontEnd(port)


async def test_the_in_process_front_end_serves_a_real_websocket_logs_and_closes(tmp_path, claimed_port) -> None:
    """It binds a 95XX port, greets a real client the way a Showdown server does, writes its log
    under the read's directory with the READY line, and on exit stops listening — nothing to kill
    by PID, because the server is this process."""
    from websockets.asyncio.client import connect

    from main.anchors.server import InProcessFrontEnd, in_process_version

    port = claimed_port()
    log = tmp_path / "ws_frontend.log"
    async with InProcessFrontEnd(port, log_path=log) as srv:
        assert srv.pid == os.getpid() and srv.front is not None
        async with connect(srv.uri) as ws:
            first = await ws.recv()
            assert str(first).startswith("|updateuser| Guest ")
    assert "[ws_frontend] READY" in log.read_text()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        assert s.connect_ex(("127.0.0.1", port)) != 0, "still listening after the context closed"
    assert in_process_version().startswith("ws_frontend[in-process]@")


def test_the_engine_stamp_is_the_trees_build_stamp_keys_and_empty_when_stale_builds_are_allowed(monkeypatch):
    """REVIEW F F-M2: a Rust-served cell names the engine it played on (``main.reads.engine``): this tree's port and
    env stamp keys, which every Rust artifact's stamp check proved it was built from; never under the stale opt-out."""
    from main.anchors import server as SV
    from utils.rust_env import stamp as ST

    monkeypatch.delenv(ST.ALLOW_STALE_ENV, raising=False)
    port, n_port = ST.source_hash(scope="port")
    env, n_env = ST.source_hash(scope="env")
    assert SV.engine_stamp() == f"port={port}/{n_port};env={env}/{n_env}"
    monkeypatch.setenv(ST.ALLOW_STALE_ENV, "1")
    assert SV.engine_stamp() == ""
