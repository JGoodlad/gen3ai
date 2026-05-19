import asyncio
import json
import logging
from asyncio import Queue
from collections.abc import AsyncGenerator
from typing import Dict, List, Optional, Set

from poke_env.concurrency import POKE_LOOP, create_in_poke_loop
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
    """

    MAX_CONCURRENT: int = 10
    JOIN_INTERVAL: float = 10.0
    POLL_INTERVAL: float = 30.0

    def __init__(
        self,
        *,
        server_configuration: ServerConfiguration,
        max_concurrent: int = MAX_CONCURRENT,
        join_interval: float = JOIN_INTERVAL,
        poll_interval: float = POLL_INTERVAL,
        log_level: Optional[int] = None,
    ) -> None:
        self._server_configuration = server_configuration
        self._max_concurrent = max_concurrent
        self._join_interval = join_interval
        self._poll_interval = poll_interval

        self._logger = logging.getLogger(__name__)
        if log_level is not None:
            self._logger.setLevel(log_level)

        # Created fresh on each call to watch() so the object can be reused.
        self._client: Optional[PSClient] = None
        self._active: Dict[str, SpectatedBattle] = {}
        self._seen: Set[str] = set()
        self._pending: Optional[Queue] = None   # room IDs waiting to be joined
        self._done: Optional[Queue] = None       # finished SpectatedBattle objects
        self._format_id: str = ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def watch(self, format_id: str) -> AsyncGenerator[SpectatedBattle, None]:
        """
        Async generator that yields SpectatedBattle objects as battles complete.
        Runs indefinitely until the calling task is cancelled.

        Must be awaited on POKE_LOOP:
            fut = asyncio.run_coroutine_threadsafe(
                _consume(spectator.watch("gen3ou")), POKE_LOOP
            )
            fut.result()
        """
        self._active = {}
        self._seen = set()
        self._pending = Queue()
        self._done = Queue()
        self._format_id = format_id

        self._client = PSClient(
            AccountConfiguration("", None),  # connect as guest — server assigns "Guest XXXXX"
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
                battle = await self._done.get()
                yield battle
        finally:
            join_task.cancel()
            poll_task.cancel()

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
            await self._client.send_message(f"/join {room_id}")
            await asyncio.sleep(self._join_interval)

    async def _poll_loop(self) -> None:
        """Re-query the roomlist every poll_interval seconds to find new battles."""
        while True:
            await asyncio.sleep(self._poll_interval)
            await self._client.send_message(f"/query roomlist {self._format_id}")

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
