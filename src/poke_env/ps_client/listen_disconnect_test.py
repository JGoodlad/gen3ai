"""Unit tests for PSClient.listen()'s disconnect-signalling on exit.

Companion to ``environment/async_queue_disconnect_test.py`` (which guards the consumer
side — that a set ``_disconnected`` makes ``_AsyncQueue.get`` raise instead of hang).
This guards the PRODUCER side: which way of exiting ``listen()`` sets ``_disconnected``.

The contract:

* A close we did NOT request — an abnormal drop (ping timeout / server died / no close
  frame) OR a *clean* peer/server-initiated close (e.g. ``npm run stop``) — sets
  ``_disconnected`` so a blocked env.step/reset fails loudly and the process exits
  (→ launcher restarts). The clean-server-close case is the one that used to hang
  indefinitely, since a clean ``ConnectionClosedOK`` looks identical to our own close.
* A close WE requested (``stop_listening`` → ``_stop_listening`` sets ``_closing``, or a
  loop-cancellation) is an intentional teardown and must NOT signal — terminating the
  connection on purpose is not an error and a spurious signal would crash a healthy run.

These drive the real ``listen()`` on POKE_LOOP with ``websockets.connect`` mocked to a
fake socket whose ``recv()`` raises the close we want — no server, no real socket.
"""

import asyncio

import pytest
from unittest.mock import patch

from websockets.exceptions import ConnectionClosedError, ConnectionClosedOK

from poke_env.concurrency import POKE_LOOP
from poke_env.ps_client import ps_client as ps_client_mod
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.ps_client import PSClient
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration


class _FakeSocket:
    """Stand-in for a websockets ClientConnection: recv() raises a preset close."""

    def __init__(self, recv_exc):
        self._recv_exc = recv_exc
        self.closed = False

    async def recv(self):
        raise self._recv_exc

    async def close(self):
        self.closed = True


class _FakeConnectCM:
    """Async context manager returned by the mocked ws.connect(...)."""

    def __init__(self, socket):
        self._socket = socket

    async def __aenter__(self):
        return self._socket

    async def __aexit__(self, exc_type, exc, tb):
        return False  # don't suppress — let abnormal drops propagate to listen()


def _fake_connect(recv_exc):
    def _connect(*args, **kwargs):
        return _FakeConnectCM(_FakeSocket(recv_exc))

    return _connect


def _make_client(username: str) -> PSClient:
    return PSClient(
        account_configuration=AccountConfiguration(username, None),
        server_configuration=LocalhostServerConfiguration,
        start_listening=False,  # no auto-connect — we drive listen() ourselves
        ping_interval=None,
        ping_timeout=None,
    )


def _run(coro):
    return asyncio.run_coroutine_threadsafe(coro, POKE_LOOP).result(timeout=5)


def _run_listen(client: PSClient, recv_exc) -> None:
    # The patch stays active for the whole run because .result() blocks here until
    # listen() (running on POKE_LOOP) returns.
    with patch.object(ps_client_mod.ws, "connect", _fake_connect(recv_exc)):
        _run(client.listen())


def test_peer_clean_close_signals_disconnect():
    """A clean ConnectionClosedOK we did NOT initiate (server stopped) must signal —
    this is the case that used to hang env.step/reset forever."""
    client = _make_client("guard-peer")
    assert not client._disconnected.is_set()
    _run_listen(client, ConnectionClosedOK(None, None))
    assert client._disconnected.is_set()


def test_abnormal_drop_signals_disconnect():
    """An abnormal drop (no close frame / ping timeout) still signals (unchanged)."""
    client = _make_client("guard-abnormal")
    _run_listen(client, ConnectionClosedError(None, None))
    assert client._disconnected.is_set()


def test_our_own_close_does_not_signal():
    """When WE initiated the close (_closing set), a clean exit must NOT signal —
    intentional teardown is not an error and must not trigger a crash-restart."""
    client = _make_client("guard-self")
    client._closing = True
    _run_listen(client, ConnectionClosedOK(None, None))
    assert not client._disconnected.is_set()


def test_stop_listening_marks_closing_and_closes_socket():
    """_stop_listening must set _closing (so listen()'s exit stays silent) and actually
    close the socket — the producer half of the intentional-teardown contract."""
    client = _make_client("guard-stop")
    sock = _FakeSocket(ConnectionClosedOK(None, None))
    client.websocket = sock
    assert not client._closing
    _run(client._stop_listening())
    assert client._closing
    assert sock.closed
