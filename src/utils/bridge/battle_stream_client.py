"""A poke-env transport that talks to a local BattleStream bridge subprocess
instead of a websocket Showdown server.

`BattleStreamClient` subclasses the (vendored) `PSClient` *from outside* poke_env
so it can reuse poke-env's own message splitting / battle routing
(`_handle_message`) while replacing the transport. It modifies no file under
`src/poke_env/`.

How it plugs in:
- The `LocalBattleRunner` constructs one client per `Player`, wired to that
  player's existing battle callbacks, and assigns it to `player.ps_client`
  (the player is built with `start_listening=False`, so no websocket is opened).
- Inbound: the runner reads protocol chunks from the bridge, frames them with the
  room header poke-env expects, and `await client.feed(framed)`. `feed` drives
  `_handle_message` to completion — including any `choose_move` + choice send a
  `|request|` in that chunk triggers. No background listen loop, no queue: the
  battle is driven sequentially and deterministically by the runner.
- Outbound: poke-env emits choices via `ps_client.send_message("/choose ...",
  battle_tag)`. We translate those onto the bridge's stdin (`CHOOSE <side>
  <choice>`); all websocket ceremony (`/utm`, `/timer on`, `/leave`, `/challenge`,
  ...) is a no-op.

There is intentionally NO `self.websocket` attribute, so poke-env's
`hasattr(self.ps_client, "websocket")` guards (VGC team sheets, `/leave`) skip
correctly.
"""

from __future__ import annotations

import asyncio
from typing import Dict, List, Optional

from poke_env.concurrency import POKE_LOOP
from poke_env.ps_client.account_configuration import AccountConfiguration
from poke_env.ps_client.ps_client import PSClient
from poke_env.ps_client.server_configuration import LocalhostServerConfiguration

_CHOOSE_PREFIX = "/choose "
_TEAM_PREFIX = "/team "


class BattleStreamClient(PSClient):
    """PSClient that relays protocol to/from a local BattleStream bridge process."""

    def __init__(
        self,
        account_configuration: AccountConfiguration,
        *,
        side: str,
        on_battle_message,
        on_update_challenges,
        on_challenge_request,
        log_level: Optional[int] = None,
        loop: asyncio.AbstractEventLoop = POKE_LOOP,
    ):
        super().__init__(
            account_configuration,
            log_level=log_level,
            on_battle_message=on_battle_message,
            on_update_challenges=on_update_challenges,
            on_challenge_request=on_challenge_request,
            server_configuration=LocalhostServerConfiguration,
            start_listening=False,
            loop=loop,
        )
        # "p1" or "p2" — which sim side this client controls.
        self._side = side
        # battle_tag -> live bridge subprocess that owns that battle. Populated by
        # the runner before it feeds a battle's first chunk; one client can hold
        # several concurrent battles, each on its own subprocess.
        self._procs: Dict[str, asyncio.subprocess.Process] = {}
        # No auth handshake against a local sim.
        self._logged_in.set()

    # --- inbound -----------------------------------------------------------
    async def feed(self, framed_text: str) -> None:
        """Process one framed protocol block (room header already prepended).

        Awaits full handling, so any choose_move + choice send triggered by a
        ``|request|`` in this block has completed by the time it returns.
        """
        await self._handle_message(framed_text)

    async def listen(self):  # pragma: no cover - never started (start_listening=False)
        return

    async def log_in(self, split_message: List[str]):
        # No authentication against an in-process sim.
        self._logged_in.set()

    # --- outbound ----------------------------------------------------------
    async def send_message(
        self, message: str, room: str = "", message_2: Optional[str] = None
    ):
        if message.startswith(_CHOOSE_PREFIX):
            await self._write_choice(room, message[len(_CHOOSE_PREFIX):])
        elif message == "/forfeit":
            await self._write_raw(room, f"FORCELOSE {self._side}")
        elif message.startswith(_TEAM_PREFIX):
            # Teampreview ("/team 123456" -> "team 123456"). gen3ou has none, but
            # keep the path correct for other gens.
            await self._write_choice(room, message[1:])
        # else: /utm, /timer on, /leave, /challenge, /accept, /trn, /search,
        # /avatar, team sheets — websocket ceremony with no local equivalent.

    async def _write_choice(self, room: str, choice: str) -> None:
        await self._write_raw(room, f"CHOOSE {self._side} {choice}")

    async def _write_raw(self, room: str, command: str) -> None:
        proc = self._procs.get(room)
        if proc is None or proc.stdin is None:
            return
        proc.stdin.write((command + "\n").encode())
        await proc.stdin.drain()
