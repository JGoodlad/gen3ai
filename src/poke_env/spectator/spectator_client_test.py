"""Unit tests for the stuck-room reaper in BattleSpectator.

Pure/mock tests — no server, no websocket. The reap decision is a pure function
of the battle's timestamps; the eviction path is exercised with a fake client.
"""

import asyncio
import time

import pytest

from poke_env.ps_client.server_configuration import LocalhostServerConfiguration
from poke_env.spectator.spectated_battle import SpectatedBattle
from poke_env.spectator.spectator_client import BattleSpectator, reap_reason


STALE = 600.0
MAX_WATCH = 3600.0


def _battle_aged(joined_ago: float, idle_ago: float) -> SpectatedBattle:
    """A battle whose joined_at / last_activity are back-dated by the given seconds."""
    b = SpectatedBattle("battle-gen3ou-1")
    now = time.time()
    b._joined_at = now - joined_ago
    b._last_activity = now - idle_ago
    return b


# ── reap_reason (pure decision) ────────────────────────────────────────────

def test_healthy_room_not_reaped():
    b = _battle_aged(joined_ago=120.0, idle_ago=5.0)
    assert reap_reason(b, time.time(), STALE, MAX_WATCH) is None


def test_stale_room_reaped():
    b = _battle_aged(joined_ago=700.0, idle_ago=650.0)
    reason = reap_reason(b, time.time(), STALE, MAX_WATCH)
    assert reason is not None and reason.startswith("idle")


def test_never_ending_room_reaped_even_if_chatty():
    # Still receiving messages (idle small) but watched way past the absolute cap.
    b = _battle_aged(joined_ago=MAX_WATCH + 100.0, idle_ago=2.0)
    reason = reap_reason(b, time.time(), STALE, MAX_WATCH)
    assert reason is not None and reason.startswith("watched")


def test_boundary_is_inclusive():
    # Fixed reference time so the >= boundary is exact, not subject to two
    # time.time() calls racing.
    now = 1_000_000.0
    b = SpectatedBattle("battle-gen3ou-1")
    b._joined_at = now - STALE
    b._last_activity = now - STALE
    assert reap_reason(b, now, STALE, MAX_WATCH) is not None


# ── eviction path ──────────────────────────────────────────────────────────

class _FakeClient:
    def __init__(self) -> None:
        self.sent: list = []

    async def send_message(self, msg: str) -> None:
        self.sent.append(msg)


def _spectator() -> BattleSpectator:
    return BattleSpectator(server_configuration=LocalhostServerConfiguration)


def test_abandon_frees_slot_leaves_room_and_counts():
    async def run():
        spec = _spectator()
        spec._client = _FakeClient()
        tag = "battle-gen3ou-42"
        battle = SpectatedBattle(tag)
        spec._active[tag] = battle

        await spec._abandon_battle(tag, battle, "idle 700s")

        assert spec._client.sent == [f"/leave {tag}"]
        assert tag not in spec._active          # slot freed
        assert tag in spec._finished_tags       # late messages ignored
        assert spec.abandoned_count == 1

    asyncio.run(run())


def test_abandoned_room_ignores_late_messages():
    # After abandoning, a stray server message for that tag must not resurrect the room.
    async def run():
        spec = _spectator()
        spec._client = _FakeClient()
        spec._finished_tags = set()
        tag = "battle-gen3ou-7"
        battle = SpectatedBattle(tag)
        spec._active[tag] = battle
        await spec._abandon_battle(tag, battle, "watched 3700s")

        await spec._handle_battle_message([[f">{tag}"], ["", "turn", "9"]])
        assert tag not in spec._active

    asyncio.run(run())


def test_reaper_loop_evicts_stale_then_healthy_survives():
    async def run():
        spec = _spectator()
        spec._client = _FakeClient()
        spec._reaper_interval = 0.01
        spec._stale_timeout = STALE
        spec._max_watch_time = MAX_WATCH

        spec._active["battle-gen3ou-stale"] = _battle_aged(700.0, 650.0)
        healthy = _battle_aged(30.0, 2.0)
        spec._active["battle-gen3ou-live"] = healthy

        task = asyncio.ensure_future(spec._reaper_loop())
        await asyncio.sleep(0.05)
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        assert "battle-gen3ou-stale" not in spec._active
        assert "battle-gen3ou-live" in spec._active
        assert spec.abandoned_count == 1

    asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
