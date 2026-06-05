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
import itertools
import json
from pathlib import Path
from typing import List, Optional

from poke_env.concurrency import POKE_LOOP, handle_threaded_coroutines
from poke_env.player.player import Player
from poke_env.teambuilder.teambuilder import Teambuilder

from utils.bridge.battle_stream_client import BattleStreamClient

_BRIDGE_JS = str(Path(__file__).parent / "local_sim_bridge.js")
_PER_BATTLE_TIMEOUT = 180.0  # generous; a fast in-process battle is seconds

# Process-global, monotonically increasing battle number — mirrors how a real Showdown
# server hands out a unique room id per battle. The tag MUST be unique across the whole
# process, not just within one ``run_local_battles`` call: the same ``Player`` objects are
# reused across calls (their ``_battles`` dict persists), and poke-env's ``_create_battle``
# returns the *existing* battle for a tag it has already seen
# (``player.py``: ``if battle_tag in self._battles: return self._battles[battle_tag]``).
# When a chunked / time-budget fuzz loop calls ``run_local_battles`` repeatedly, a per-call
# ``battle-{fmt}-{index+1}`` scheme reuses ``battle-{fmt}-1`` every chunk, so the *new*
# battle is parsed into the *previous* battle's object — which already holds a full 6-mon
# team. The new battle's first ``|switch|`` of a different species then overflows that team,
# raising ``ValueError: <side>'s team already has 6 pokemons: cannot add ...`` from
# ``get_pokemon``. A global counter makes every tag unique, so each battle always gets a
# fresh object (single-call behaviour is unchanged — tags were already unique within a call).
_BATTLE_SEQ = itertools.count(1)


async def run_local_battles(
    player1: Player,
    player2: Player,
    n_battles: int,
    *,
    battle_format: Optional[str] = None,
    seed: Optional[List[int]] = None,
    concurrency: int = 1,
) -> None:
    """Play ``n_battles`` between two players via the local sim bridge.

    ``player1`` is sim side p1, ``player2`` is p2. ``seed`` is an optional
    ``[s0,s1,s2,s3]`` Gen-5 PRNG seed for reproducible battles (note: teams must
    also be fixed for full determinism).

    ``concurrency`` > 1 plays up to that many battles at once (each its own bridge
    subprocess), mirroring poke-env's server ``battle_against``: the per-battle
    *start* (``get_next_team`` → battle created) is serialized so the shared
    ``player._current_packed_team`` can't be overwritten before ``_create_battle``
    reads it, but battle *play* overlaps. ``concurrency == 1`` is the unchanged
    sequential path. Don't set it above ~10 here — each concurrent battle is a Node
    process. (Eval runs serially — ``_EVAL_SUBPROCESS_CONCURRENCY`` is 1; only the
    integration tests exercise concurrency > 1.)
    """
    runner = _LocalBattleRunner(player1, player2, battle_format or player1.format, seed)
    await handle_threaded_coroutines(runner.run(n_battles, concurrency), POKE_LOOP)


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

    async def run(self, n_battles: int, concurrency: int = 1) -> None:
        # Attach bridge transports (on POKE_LOOP). Players must have been built
        # with start_listening=False so no websocket was ever opened.
        self.c1 = self._attach(self.p1, "p1")
        self.c2 = self._attach(self.p2, "p2")
        if concurrency <= 1:
            # Unchanged sequential path — what all the fuzz suites exercise.
            for i in range(n_battles):
                await asyncio.wait_for(self._one_battle(i), timeout=_PER_BATTLE_TIMEOUT)
            return
        # Bounded-concurrency path. A single ``start_lock`` serializes each battle's team→creation
        # critical section (released the instant both battle objects exist — see ``_one_battle``),
        # exactly like the server's per-battle semaphore; the semaphore caps how many overlap.
        start_lock = asyncio.Lock()
        sem = asyncio.Semaphore(concurrency)

        async def _guarded(i: int) -> None:
            async with sem:
                await asyncio.wait_for(
                    self._one_battle(i, start_lock), timeout=_PER_BATTLE_TIMEOUT
                )

        await asyncio.gather(*(_guarded(i) for i in range(n_battles)))

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

    async def _one_battle(self, index: int, start_lock=None) -> None:
        # Unique across the whole process (see ``_BATTLE_SEQ`` above) — never reuse a tag,
        # or poke-env hands back the prior battle's object for it and its team overflows.
        tag = f"battle-{self.fmt}-{next(_BATTLE_SEQ)}"

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

        # Serialize the team→creation critical section under ``start_lock`` (concurrent path only).
        # ``get_next_team`` sets the shared ``player._current_packed_team`` that ``_create_battle``
        # reads; holding the lock until BOTH battle objects exist (released by ``_demux``) stops a
        # concurrent battle from overwriting it first — exactly the server's semaphore behaviour.
        locked = False

        def _release_start() -> None:
            nonlocal locked
            if locked:
                locked = False
                start_lock.release()

        try:
            if start_lock is not None:
                await start_lock.acquire()
                locked = True
            # get_next_team() yields a packed team AND sets player._current_packed_team.
            team1 = self.p1.get_next_team()
            team2 = self.p2.get_next_team()
            start = {
                "formatid": self.fmt,
                "p1": {"name": self.p1.username, "team": team1},
                "p2": {"name": self.p2.username, "team": team2},
            }
            if self.seed:
                start["seed"] = self.seed
            proc.stdin.write((f"START {json.dumps(start)}\n").encode())
            await proc.stdin.drain()

            await self._demux(
                proc, tag, stderr_buf,
                started_cb=(_release_start if start_lock is not None else None),
            )
        finally:
            _release_start()  # safety: release if the battle ended before both were created
            self.c1._procs.pop(tag, None)
            self.c2._procs.pop(tag, None)
            await self._teardown(proc, stderr_task)

    async def _demux(self, proc, tag: str, stderr_buf: List[bytes], started_cb=None) -> None:
        """Read framed side-chunks from the bridge and feed the right client.

        ``started_cb`` (concurrent path): called ONCE, the moment both players hold a battle object
        for ``tag`` — i.e. both ``_create_battle`` calls have read the team — so the runner can let
        the next battle's start proceed. ``None`` on the sequential path (no-op, unchanged)."""
        inited = {"p1": False, "p2": False}
        started = False
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
            if (
                started_cb is not None
                and not started
                and tag in self.p1._battles
                and tag in self.p2._battles
            ):
                started = True
                started_cb()

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
