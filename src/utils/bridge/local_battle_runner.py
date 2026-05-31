"""Run poke-env battles against a local BattleStream bridge — no websocket server.

`run_local_battles(player1, player2, n_battles)` is a drop-in replacement for
`player1.battle_against(player2, n_battles=...)` that needs no `npm run showdown`,
no usernames, no port, no matchmaking. Each battle runs in its own throwaway
`local_sim_bridge.js` subprocess (an in-process Showdown `BattleStream`), and the
protocol stream is fed through the *unmodified* poke-env parsing pipeline
(`_handle_battle_message` → `parse_message`/`parse_request` → `choose_move`).

The runner owns the coordination that the websocket challenge handshake normally
does: it picks a deterministic battle tag, fabricates the `>battle-…`/`|init|`
room framing the sim does not emit, and routes each side's protocol to the right
`Player`'s `BattleStreamClient`.

Everything runs on `POKE_LOOP` (the loop poke-env's async machinery lives on), so
`choose_move`, the subprocess I/O, and the choice send-back all share one loop and
stay deterministic — each protocol chunk is fully processed (including any choice
it triggers) before the next is read.
"""

from __future__ import annotations

import asyncio
import base64
import json
from pathlib import Path
from typing import List, Optional

from poke_env.concurrency import POKE_LOOP, handle_threaded_coroutines
from poke_env.player.player import Player
from poke_env.teambuilder.teambuilder import Teambuilder

from utils.bridge.battle_stream_client import BattleStreamClient

_BRIDGE_JS = str(Path(__file__).parent / "local_sim_bridge.js")
_PER_BATTLE_TIMEOUT = 180.0  # generous; a fast in-process battle is seconds


async def run_local_battles(
    player1: Player,
    player2: Player,
    n_battles: int,
    *,
    battle_format: Optional[str] = None,
    seed: Optional[List[int]] = None,
) -> None:
    """Play ``n_battles`` between two players via the local sim bridge.

    ``player1`` is sim side p1, ``player2`` is p2. ``seed`` is an optional
    ``[s0,s1,s2,s3]`` Gen-5 PRNG seed for reproducible battles (note: teams must
    also be fixed for full determinism).
    """
    runner = _LocalBattleRunner(player1, player2, battle_format or player1.format, seed)
    await handle_threaded_coroutines(runner.run(n_battles), POKE_LOOP)


class _LocalBattleRunner:
    def __init__(
        self,
        player1: Player,
        player2: Player,
        battle_format: str,
        seed: Optional[List[int]],
    ):
        self.p1 = player1
        self.p2 = player2
        self.fmt = battle_format
        self.seed = seed
        self.c1: Optional[BattleStreamClient] = None
        self.c2: Optional[BattleStreamClient] = None

    async def run(self, n_battles: int) -> None:
        # Attach bridge transports (on POKE_LOOP). Players must have been built
        # with start_listening=False so no websocket was ever opened.
        self.c1 = self._attach(self.p1, "p1")
        self.c2 = self._attach(self.p2, "p2")
        for i in range(n_battles):
            await asyncio.wait_for(self._one_battle(i), timeout=_PER_BATTLE_TIMEOUT)

    def _attach(self, player: Player, side: str) -> BattleStreamClient:
        client = BattleStreamClient(
            player.ps_client._account_configuration,
            side=side,
            on_battle_message=player._handle_battle_message,
            on_update_challenges=player._update_challenges,
            on_challenge_request=player._handle_challenge_request,
            loop=POKE_LOOP,
        )
        player.ps_client = client
        return client

    async def _one_battle(self, index: int) -> None:
        tag = f"battle-{self.fmt}-{index + 1}"
        # get_next_team() yields a packed team AND sets player._current_packed_team,
        # which _create_battle reads. Sequential play makes that assignment safe.
        team1 = self.p1.get_next_team()
        team2 = self.p2.get_next_team()

        proc = await asyncio.create_subprocess_exec(
            "node",
            _BRIDGE_JS,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        self.c1._procs[tag] = proc
        self.c2._procs[tag] = proc
        stderr_buf: List[bytes] = []
        stderr_task = asyncio.ensure_future(self._drain_stderr(proc, stderr_buf))

        try:
            start = {
                "formatid": self.fmt,
                "p1": {"name": self.p1.username, "team": team1},
                "p2": {"name": self.p2.username, "team": team2},
            }
            if self.seed:
                start["seed"] = self.seed
            proc.stdin.write((f"START {json.dumps(start)}\n").encode())
            await proc.stdin.drain()

            await self._demux(proc, tag, stderr_buf)
        finally:
            self.c1._procs.pop(tag, None)
            self.c2._procs.pop(tag, None)
            await self._teardown(proc, stderr_task)

    async def _demux(self, proc, tag: str, stderr_buf: List[bytes]) -> None:
        """Read framed side-chunks from the bridge and feed the right client."""
        inited = {"p1": False, "p2": False}
        while True:
            line = await proc.stdout.readline()
            if not line:
                break
            text = line.decode().rstrip("\n")
            if text == "__END__":
                break
            if text.startswith("__ERR__"):
                msg = base64.b64decode(text[len("__ERR__ "):]).decode("utf-8")
                raise RuntimeError(f"local_sim_bridge error: {msg}")
            side, b64 = text.split(" ", 1)
            chunk = base64.b64decode(b64).decode("utf-8")
            client = self.c1 if side == "p1" else self.c2
            framed = self._frame(tag, side, chunk, inited)
            await client.feed(framed)

    @staticmethod
    def _frame(tag: str, side: str, chunk: str, inited: dict) -> str:
        # The sim does not emit the server room header; poke-env's _handle_message
        # keys the battle off ">battle-…" + "|init|battle". Prepend them, and add
        # |init| only to the first chunk per side (so _create_battle fires once).
        header = f">{tag}\n"
        if not inited[side]:
            inited[side] = True
            header += "|init|battle\n"
        return header + chunk

    @staticmethod
    async def _drain_stderr(proc, buf: List[bytes]) -> None:
        try:
            while True:
                line = await proc.stderr.readline()
                if not line:
                    break
                buf.append(line)
        except asyncio.CancelledError:  # pragma: no cover
            pass

    @staticmethod
    async def _teardown(proc, stderr_task) -> None:
        if proc.returncode is None:
            try:
                if proc.stdin and not proc.stdin.is_closing():
                    proc.stdin.write(b"END\n")
                    await proc.stdin.drain()
            except (BrokenPipeError, ConnectionResetError):
                pass
            try:
                await asyncio.wait_for(proc.wait(), timeout=5.0)
            except asyncio.TimeoutError:  # pragma: no cover
                proc.kill()
                await proc.wait()
        stderr_task.cancel()
