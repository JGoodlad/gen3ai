import asyncio
import json
import logging
from asyncio import Queue
from collections.abc import AsyncGenerator
from typing import Dict, List, Optional, Set

from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.ps_client import PSClient
from poke_env.ps_client.server_configuration import ServerConfiguration
from poke_env.spectator.spectated_battle import SpectatedBattle


class BattleSpectator:
    """
    Connects to a Showdown server as a guest, discovers active battles for a given
    format, and yields completed SpectatedBattle objects one at a time.

    Designed to run on POKE_LOOP (the poke-env background event loop).
    Use asyncio.run_coroutine_threadsafe(spectator.watch(...), POKE_LOOP) from the
    main thread rather than asyncio.run().

    Rate limiting:
      - At most max_concurrent rooms watched simultaneously (default 10)
      - At least join_interval seconds between /join commands (default 10 s)
      - Roomlist re-queried every poll_interval seconds (default 30 s)

    Reconnection:
      - Detects dropped connections and reconnects automatically.
      - _seen is preserved across reconnects (won't re-join old rooms).
      - Reconnect delay is 10 s by default.
    """

    MAX_CONCURRENT: int = 10
    JOIN_INTERVAL: float = 10.0
    POLL_INTERVAL: float = 30.0
    RECONNECT_DELAY: float = 10.0

    def __init__(
        self,
        *,
        server_configuration: ServerConfiguration,
        max_concurrent: int = MAX_CONCURRENT,
        join_interval: float = JOIN_INTERVAL,
        poll_interval: float = POLL_INTERVAL,
        reconnect_delay: float = RECONNECT_DELAY,
        log_level: Optional[int] = None,
    ) -> None:
        self._server_configuration = server_configuration
        self._max_concurrent = max_concurrent
        self._join_interval = join_interval
        self._poll_interval = poll_interval
        self._reconnect_delay = reconnect_delay

        self._logger = logging.getLogger(__name__)
        if log_level is not None:
            self._logger.setLevel(log_level)

        self._client: Optional[PSClient] = None
        self._active: Dict[str, SpectatedBattle] = {}
        self._seen: Set[str] = set()       # preserved across reconnects
        self._pending: Optional[Queue] = None
        self._done: Optional[Queue] = None
        self._format_id: str = ""
        self._total_joined: int = 0        # preserved across reconnects

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def watch(self, format_id: str) -> AsyncGenerator[SpectatedBattle, None]:
        """
        Async generator that yields SpectatedBattle objects as battles complete.
        Reconnects automatically on connection drop. Runs until cancelled.
        """
        while True:
            try:
                async for battle in self._watch_once(format_id):
                    yield battle
                # _watch_once returned cleanly — shouldn't happen in normal operation
                return
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._logger.warning(
                    "Spectator connection lost (%s: %s) — reconnecting in %.0fs",
                    type(e).__name__, e, self._reconnect_delay,
                )
                self._active.clear()
                await asyncio.sleep(self._reconnect_delay)

    # ------------------------------------------------------------------
    # Internal: single connection session
    # ------------------------------------------------------------------

    async def _watch_once(self, format_id: str) -> AsyncGenerator[SpectatedBattle, None]:
        """
        Single connection attempt. Yields battles until the WebSocket drops.
        Raises on disconnect so watch() can reconnect.
        """
        self._active = {}
        self._pending = Queue()
        self._done = Queue()
        self._format_id = format_id

        self._client = PSClient(
            AccountConfiguration("", None),  # guest — server assigns "Guest XXXXX"
            server_configuration=self._server_configuration,
            on_battle_message=self._handle_battle_message,
            on_query_response=self._on_query_response,
            log_level=None,
        )

        await self._client.wait_for_login(wait_for=15)
        await self._client.send_message(f"/query roomlist {format_id}")

        join_task = asyncio.ensure_future(self._join_loop())
        poll_task = asyncio.ensure_future(self._poll_loop())
        try:
            while True:
                # Detect dropped connection: listen() coroutine has exited
                listen_future = getattr(self._client, "_listening_coroutine", None)
                if listen_future is not None and listen_future.done():
                    raise ConnectionError("WebSocket listener exited — connection dropped")

                # Detect background task failure (send on dead socket, etc.)
                for task in (join_task, poll_task):
                    if task.done() and not task.cancelled():
                        exc = task.exception()
                        if exc is not None:
                            raise exc

                try:
                    battle = await asyncio.wait_for(self._done.get(), timeout=5.0)
                    yield battle
                except asyncio.TimeoutError:
                    continue  # loop back to check connection health
        finally:
            join_task.cancel()
            poll_task.cancel()
            for task in (join_task, poll_task):
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

    # ------------------------------------------------------------------
    # Internal loops
    # ------------------------------------------------------------------

    async def _join_loop(self) -> None:
        """Drain _pending one room at a time, gated by max_concurrent and join_interval."""
        while True:
            room_id = await self._pending.get()
            # Wait until a slot is available
            while len(self._active) >= self._max_concurrent:
                await asyncio.sleep(1.0)
            self._logger.info("Joining %s", room_id)
            await self._client.send_message(f"/join {room_id}")  # raises if connection dead
            self._total_joined += 1
            await asyncio.sleep(self._join_interval)

    async def _poll_loop(self) -> None:
        """Re-query the roomlist every poll_interval seconds to find new battles."""
        while True:
            await asyncio.sleep(self._poll_interval)
            await self._client.send_message(f"/query roomlist {self._format_id}")  # raises if dead

    # ------------------------------------------------------------------
    # PSClient callbacks
    # ------------------------------------------------------------------

    async def _on_query_response(self, split_message: List[str]) -> None:
        """Handle |queryresponse|roomlist|{...} — enqueue unseen room IDs."""
        if len(split_message) < 4 or split_message[2] != "roomlist":
            return
        try:
            data = json.loads(split_message[3])
        except (json.JSONDecodeError, IndexError):
            self._logger.warning("Failed to parse roomlist JSON")
            return

        rooms = data.get("rooms", {})
        new_count = 0
        for room_id in rooms:
            if room_id not in self._seen and room_id not in self._active:
                self._seen.add(room_id)
                await self._pending.put(room_id)
                new_count += 1
        if new_count:
            self._logger.info("Queued %d new %s rooms", new_count, self._format_id)

    async def _handle_battle_message(self, split_messages: List[List[str]]) -> None:
        """Route incoming message lines to the right SpectatedBattle."""
        if not split_messages:
            return

        # split_messages[0][0] is ">battle-gen3ou-123456"
        raw_tag = split_messages[0][0]
        battle_tag = raw_tag.lstrip(">")
        if not battle_tag.startswith("battle-"):
            return

        if battle_tag not in self._active:
            self._active[battle_tag] = SpectatedBattle(battle_tag)

        battle = self._active[battle_tag]

        for parts in split_messages[1:]:
            if len(parts) < 2:
                continue
            msg_type = parts[1]
            if msg_type == "win":
                winner = parts[2] if len(parts) > 2 else None
                battle.finish(winner)
                await self._finish_battle(battle_tag, battle)
                return
            elif msg_type == "tie":
                battle.finish(None)
                await self._finish_battle(battle_tag, battle)
                return
            else:
                battle.add_lines([parts])

    async def _finish_battle(self, battle_tag: str, battle: SpectatedBattle) -> None:
        await self._client.send_message(f"/leave {battle_tag}")
        del self._active[battle_tag]
        await self._done.put(battle)
        self._logger.info(
            "Finished %s — winner: %s", battle_tag, battle.winner or "tie"
        )

    # ------------------------------------------------------------------
    # Status properties (safe to read from any thread — benign races for display)
    # ------------------------------------------------------------------

    @property
    def active_battles(self) -> Dict[str, SpectatedBattle]:
        """Snapshot of currently-watched battles. Keys are battle tags."""
        return dict(self._active)

    @property
    def pending_count(self) -> int:
        """Number of room IDs waiting to be joined."""
        return self._pending.qsize() if self._pending is not None else 0

    @property
    def seen_count(self) -> int:
        """Total distinct rooms seen across all roomlist queries."""
        return len(self._seen)

    @property
    def total_joined(self) -> int:
        """Total /join commands sent since watch() was called."""
        return self._total_joined
