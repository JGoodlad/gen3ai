"""Unit tests for _AsyncQueue's connection-loss handling.

Regression guard for the mid-battle-disconnect hang: PSClient.listen() swallows an
abnormal websocket drop (logs + returns), leaving the env's blocking get parked on a
message that will never arrive — the training process then hangs forever and the
launcher (which only watches for process EXIT) never notices it died.

The fix makes a dropped connection deterministic: listen() sets ps_client._disconnected,
and _AsyncQueue.get/race_get race the get against that event, raising ShowdownException
instead of blocking. These tests drive _AsyncQueue directly on the real POKE_LOOP (no
websocket, no server) and assert: a disconnect raises, a real item still wins (even when
the disconnect fires too), an unrelated event still returns None, and timeout semantics
are preserved.
"""

import asyncio
import time

import pytest

from poke_env.concurrency import POKE_LOOP, create_in_poke_loop
from poke_env.environment.env import _AsyncQueue
from poke_env.exceptions import ShowdownException


def _new_event() -> asyncio.Event:
    return create_in_poke_loop(asyncio.Event, POKE_LOOP)


def _set(event: asyncio.Event) -> None:
    """Set a POKE_LOOP-bound event from the test (main) thread, and wait for it."""
    POKE_LOOP.call_soon_threadsafe(event.set)
    deadline = time.monotonic() + 1.0
    while not event.is_set() and time.monotonic() < deadline:
        time.sleep(0.005)
    assert event.is_set()


def test_get_raises_on_disconnect():
    disconnected = _new_event()
    q: _AsyncQueue = _AsyncQueue(POKE_LOOP, maxsize=1, disconnected=disconnected)
    _set(disconnected)
    with pytest.raises(ShowdownException):
        q.get(timeout=2.0)


def test_race_get_raises_on_disconnect():
    disconnected = _new_event()
    other = _new_event()  # an unrelated _waiting/_trying_again-style event
    q: _AsyncQueue = _AsyncQueue(POKE_LOOP, maxsize=1, disconnected=disconnected)
    _set(disconnected)
    with pytest.raises(ShowdownException):
        q.race_get(other)


def test_get_returns_item_when_present():
    disconnected = _new_event()
    q: _AsyncQueue = _AsyncQueue(POKE_LOOP, maxsize=1, disconnected=disconnected)
    q.put("battle")
    assert q.get(timeout=2.0) == "battle"


def test_real_item_wins_over_simultaneous_disconnect():
    """A delivered message must NOT be dropped just because the socket also died —
    a battle waiting in the queue is real state we should consume, not discard."""
    disconnected = _new_event()
    q: _AsyncQueue = _AsyncQueue(POKE_LOOP, maxsize=1, disconnected=disconnected)
    q.put("battle")
    _set(disconnected)
    assert q.get(timeout=2.0) == "battle"


def test_race_get_returns_none_on_unrelated_event():
    """An unrelated event firing (no item, no disconnect) keeps the old contract:
    race_get returns None (the agent simply isn't to move)."""
    disconnected = _new_event()
    other = _new_event()
    q: _AsyncQueue = _AsyncQueue(POKE_LOOP, maxsize=1, disconnected=disconnected)
    _set(other)
    assert q.race_get(other) is None


def test_race_get_prefers_queued_battle_over_stale_event():
    """A real queued battle is authoritative and must NEVER be masked by an already-set (stale)
    _waiting/_trying_again event. This is the mutual-Arena-Trap force-switch DEADLOCK fix
    (crashes/restart_err_*_7f93c7.txt): a force-switch was delivered to the queue, but a stale
    _trying_again (set by an earlier trapped-switch rejection, not yet cleared) won the
    asyncio.wait race before the equally-ready queue.get ran, so race_get returned None — the env
    marked the agent not-to-move and the request was stranded in the queue → silent stall.
    With both the event set AND a battle queued, race_get must return the BATTLE, not None."""
    disconnected = _new_event()
    stale = _new_event()
    q: _AsyncQueue = _AsyncQueue(POKE_LOOP, maxsize=1, disconnected=disconnected)
    _set(stale)              # event already set and never cleared (the stale-event precondition)
    q.put("forceSwitch")     # a real server request is sitting in the queue
    assert q.race_get(stale) == "forceSwitch"   # battle wins over the stale event, not stranded


def test_race_get_prefers_queued_battle_even_with_disconnect():
    """Symmetry with the disconnect guard: a delivered request still wins over a simultaneous
    stale event AND a disconnect — real queued state is consumed, never discarded."""
    disconnected = _new_event()
    stale = _new_event()
    q: _AsyncQueue = _AsyncQueue(POKE_LOOP, maxsize=1, disconnected=disconnected)
    q.put("forceSwitch")
    _set(stale)
    _set(disconnected)
    assert q.race_get(stale) == "forceSwitch"


def test_get_timeout_preserved():
    """With no item and no disconnect, get(timeout) still raises TimeoutError —
    the reset/challenge path relies on this."""
    disconnected = _new_event()
    q: _AsyncQueue = _AsyncQueue(POKE_LOOP, maxsize=1, disconnected=disconnected)
    with pytest.raises(asyncio.TimeoutError):
        q.get(timeout=0.2)


def test_no_disconnect_event_behaves_as_before():
    """disconnected=None (non-env callers) keeps plain get/race_get semantics."""
    q: _AsyncQueue = _AsyncQueue(POKE_LOOP, maxsize=1)
    q.put("x")
    assert q.get(timeout=2.0) == "x"
    with pytest.raises(asyncio.TimeoutError):
        q.get(timeout=0.2)


def test_race_get_crashes_on_silent_stall(monkeypatch):
    """The wedged-battle case: no item arrives, the socket does NOT close, and the passed
    event never fires (a force-switch parsed onto the battle but never delivered to the env).
    race_get must raise ShowdownException — a HARD CRASH the launcher restarts from the last
    checkpoint — never block forever (wedges the whole SubprocVecEnv) nor return None (which
    would let step() reuse the stale battle and feed a fabricated transition to backprop)."""
    monkeypatch.setattr("poke_env.environment.env._RACE_GET_TIMEOUT_S", 0.2)
    disconnected = _new_event()
    never_fires = _new_event()
    q: _AsyncQueue = _AsyncQueue(POKE_LOOP, maxsize=1, disconnected=disconnected)
    with pytest.raises(ShowdownException):
        q.race_get(never_fires)


def test_silent_stall_crash_carries_race_trace(monkeypatch):
    """When GEN3_RACE_TRACE is on, the silent-stall crash message must carry the wedged
    battle's cross-thread interleaving so the launcher's per-crash file is enough to root-cause
    the deadlock (no live process needed)."""
    from utils import race_trace

    monkeypatch.setattr("poke_env.environment.env._RACE_GET_TIMEOUT_S", 0.2)
    monkeypatch.setattr(race_trace, "ENABLED", True)
    monkeypatch.setattr(race_trace, "_traces", {})
    race_trace.trace("battle-gen3ou-wedged", "RX |request|forceSwitch never delivered")
    disconnected = _new_event()
    never_fires = _new_event()
    q: _AsyncQueue = _AsyncQueue(POKE_LOOP, maxsize=1, disconnected=disconnected)
    with pytest.raises(ShowdownException) as exc:
        q.race_get(never_fires)
    msg = str(exc.value)
    assert "RACE TRACE" in msg
    assert "battle-gen3ou-wedged" in msg
    assert "never delivered" in msg
