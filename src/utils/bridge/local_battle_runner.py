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
import sys
from pathlib import Path
from typing import List, Optional

from poke_env.concurrency import POKE_LOOP, handle_threaded_coroutines
from poke_env.player.player import Player

from utils.bridge.battle_stream_client import BattleStreamClient
from utils.bridge.seed_spec import validate_seed_spec
from utils.bridge.sim_bridge_bin import bridge_spawn_argv
from utils.bridge import reconstruction
from contextlib import suppress

from utils.contention import ProgressDeadline, describe_contention, scale_timeout

_BRIDGE_JS = str(Path(__file__).parent / "local_sim_bridge.js")
_PER_BATTLE_TIMEOUT = 180.0  # TOTAL backstop (livelock only) — the real detector is the idle gap

# gen3_battle_progress_deadline_v1 — the max plausible gap between two protocol chunks of a
# HEALTHY battle. This is the bound that actually decides, and it is sized to a property of the
# protocol (one decision: a model forward + the sim's reply), not to how long a whole battle takes.
# Deliberately generous: a contended eval forward is a few hundred ms, so 30 s is ~100x headroom
# and still catches a true wedge in half a minute.
_BATTLE_IDLE_BUDGET = 30.0

# gen3_contention_robust_timeouts_v1 — how long to wait for the bridge child to REAP after it has
# been told to exit (`END`). A cooperative exit is milliseconds of work, so 5 s is already ~1000x
# headroom on an idle box — but it is still a wall-clock bound on a subprocess, and at load ~50 on
# 16 cores this one fired and killed a measurement arm outright (2026-08-31). Scaled at CALL time
# like every other bound here; there is no incremental progress to observe on a `wait()`, so
# `scale_timeout` is the right tool rather than a `ProgressDeadline`.
_TEARDOWN_REAP_TIMEOUT = 5.0


def _per_battle_timeout() -> float:
    """The per-battle TOTAL backstop, stretched to the CPU share actually available.

    Kept as a livelock guard, NOT as the primary detector — see ``_await_battle``. Read at CALL
    time, not import time, because a run that starts while a trainer is spinning up would
    otherwise bake in the idle-box factor for its whole life. Callers that override
    ``_PER_BATTLE_TIMEOUT`` (the parity test) still get their value scaled.
    """
    return scale_timeout(_PER_BATTLE_TIMEOUT)


def _teardown_reap_timeout() -> float:
    """The post-``END`` child-reap bound, stretched to the CPU share actually available.

    Read at CALL time for the same reason as :func:`_per_battle_timeout`: a runner constructed on
    an idle box and torn down beside a trainer must see the load it is ACTUALLY running under, not
    the one it started under.
    """
    return scale_timeout(_TEARDOWN_REAP_TIMEOUT)


async def _await_battle(coro, clients, what: str) -> None:
    """Await one battle under an IDLE bound rather than a total-duration cap.

    WHY THIS IS NOT A `wait_for`. A duration cap on a bridge battle measures the box as much as
    the code, and contention scaling does not rescue it: the factor is `loadavg / cpus` (~1.4 at
    load 22), while the actual slowdown of a starved subprocess is a multiple of that. MEASURED
    2026-08-14, against the parity test's then-20 s cap, on a box saturated by a `cargo build
    --release` (16 threads, nice 0): **8 of 12 battles "timed out" plus 1 transport error, and the
    test FAILED** — none of them wedged, all still producing protocol chunks. The same test passed
    on the warm tree with no code change. A cap cannot tell slow from stuck; an idle gap can.

    HONEST SCOPE — this is insurance against SATURATION, not a fix for the steady state, and the
    measurement says so. Beside the ordinary `--nice 10` trainer a real battle stays FAST: an A/B
    of the two bounds over 10 real battles each, at **load 45 with contention scaling pinned OFF
    and a 1.5 s budget**, timed out **0 of 10 in BOTH arms** — the battles simply finish in under
    1.5 s, so neither bound is anywhere near firing. (At the shipped 20 s parity budget the
    headroom is >10x.) The regime where the cap breaks is a core hog like `cargo build`, which is
    not reproducible here without starving the live trainer. So the evidence for this change is
    the recorded incident above plus the MECHANISM tests in `local_battle_runner_test.py`, not a
    before/after rate — and the tests are written to say that rather than imply a rate exists.

    So the detector is "no sign of life for `_BATTLE_IDLE_BUDGET`", where a sign of life is a
    protocol chunk arriving at either side's `BattleStreamClient` (`feed` bumps `progress_count`).
    That is roughly load-INVARIANT: contention stretches the gaps a little, a wedge stops them
    entirely. `_PER_BATTLE_TIMEOUT` is retained as the `total_budget_s` backstop, because an idle
    bound alone never expires against a component that chatters forever without converging
    (`node_reject_bound_integration_test`'s pre-fix wedge emits `|error|` frames indefinitely).

    Raises `ProgressTimeout`, which subclasses `TimeoutError` — so every existing
    `except (asyncio.TimeoutError, TimeoutError)` handler keeps counting it as a timeout, and a
    timeout still never becomes a semantic outcome.
    """
    task = asyncio.ensure_future(coro)
    deadline = ProgressDeadline(_BATTLE_IDLE_BUDGET,
                                total_budget_s=_PER_BATTLE_TIMEOUT, what=what)
    seen = sum(c.progress_count for c in clients if c is not None)
    try:
        while True:
            done, _ = await asyncio.wait({task}, timeout=_PROGRESS_POLL_S)
            if done:
                return task.result()
            now = sum(c.progress_count for c in clients if c is not None)
            if now != seen:
                seen = now
                deadline.progress()
            deadline.check()          # raises ProgressTimeout only on a genuine wedge
    finally:
        if not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task


_PROGRESS_POLL_S = 0.5

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
# gen3_bridge_flush_on_exit_v1, part 2: the asyncio StreamReader's DEFAULT readline limit is
# 64 KiB — and a `__RECON__` line for a LONG battle (the 1000-turn runaway cap; a full command
# log + both packed teams, base64) exceeds it. Pre-drain-fix those lines were silently CUT by
# the child's early exit, so the reader never saw one whole; the drain-aware exit delivers them
# complete, and readline then raises LimitOverrunError -> ValueError -> the BATTLE crashes
# (observed as intermittent local_sim_bridge_integration_test failures — long-battle dice).
# 16 MiB is a ceiling, not an allocation: asyncio buffers lazily, and the largest observed
# recon is ~2 orders of magnitude below it. Shared by the persistent training transport
# (bridge_session imports it) so both readers accept whatever the drain-aware child delivers.
# Training's 250-turn stall cap kept ITS recon lines under 64 KiB, which is why the persistent
# path never tripped this — the bound was luck by cap, now it is explicit.
BRIDGE_STREAM_LIMIT = 16 * 1024 * 1024


_BATTLE_SEQ = itertools.count(1)


async def run_local_battles(
    player1: Player,
    player2: Player,
    n_battles: int,
    *,
    battle_format: Optional[str] = None,
    seed: Optional[List[int]] = None,
    concurrency: int = 1,
    start_extra: "Optional[dict]" = None,
    chunk_sink: "Optional[list]" = None,
    impl: str = "node",
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
    process. (Eval runs serially by default — ``_EVAL_SUBPROCESS_CONCURRENCY`` is 1 — but
    ``--eval-concurrency-per-worker`` raises it for latency-hiding; integration tests also
    exercise concurrency > 1.)

    ``start_extra`` merges extra fields into the bridge ``START`` json — e.g.
    ``{"resumeReseed": {"turn": T, "seed": [...]}}`` for the counterfactual Monte-Carlo reseed
    (swap the sim PRNG at the start of turn ``T``). ``None`` is the unchanged default.

    ``chunk_sink`` (a list) accumulates every ``(side, chunk)`` the bridge emits — the per-side
    protocol text — for a caller that wants the move-by-move trajectory (the counterfactual narrator).
    ``None`` (default) captures nothing. Only use it on a SINGLE battle (it isn't side-deduped across
    concurrent battles).

    ``impl`` selects the bridge child binary — ``"node"`` (default, ``local_sim_bridge.js``) or
    ``"rust"`` (the byte-compatible ``src/rust_sim`` binary). The Rust binary emits no
    ``__RECON__``, so ``start_extra``'s ``resumeReseed`` + the reconstruction join degrade to
    no-ops under ``rust`` — callers that need the forensic/counterfactual layer must use ``node``.
    """
    runner = _LocalBattleRunner(player1, player2, battle_format or player1.format, seed, start_extra,
                                chunk_sink, impl)
    await handle_threaded_coroutines(runner.run(n_battles, concurrency), POKE_LOOP)


class _LocalBattleRunner:
    def __init__(
        self,
        player1: Player,
        player2: Player,
        battle_format: str,
        seed: Optional[List[int]],
        start_extra: Optional[dict] = None,
        chunk_sink: Optional[list] = None,
        impl: str = "node",
    ):
        self.p1 = player1
        self.p2 = player2
        self.fmt = battle_format
        # PRODUCER-SIDE guard (gen3_bridge_seed_forms_v1): a seed the child can't parse used
        # to be dropped silently, running some OTHER dice stream under a "seeded" label.
        # Throw here, at the caller, instead. `None` (the default) is legitimate — the child
        # mints a fresh seed and reports it in __RECON__ (see utils.bridge.seed_spec).
        validate_seed_spec(seed)
        if start_extra and isinstance(start_extra.get("resumeReseed"), dict):
            validate_seed_spec(start_extra["resumeReseed"].get("seed"),
                               what="resumeReseed.seed")
        self.seed = seed
        self.start_extra = start_extra
        self.chunk_sink = chunk_sink
        # Which bridge child to spawn per battle: "node" or "rust". Resolve to an argv list
        # once (the rust path may build the binary), reused for every _one_battle spawn.
        self._spawn_argv = bridge_spawn_argv(impl)
        self.c1: Optional[BattleStreamClient] = None
        self.c2: Optional[BattleStreamClient] = None

    async def run(self, n_battles: int, concurrency: int = 1) -> None:
        # Attach bridge transports (on POKE_LOOP). Players must have been built
        # with start_listening=False so no websocket was ever opened.
        self.c1 = self._attach(self.p1, "p1")
        self.c2 = self._attach(self.p2, "p2")
        if concurrency <= 1:
            # Sequential path — what all the fuzz suites, the parity test and (at the default
            # --eval-concurrency-per-worker 1) eval itself exercise. Bounded by the IDLE gap, so a
            # merely-slow battle beside a training run finishes instead of being scored a timeout.
            for i in range(n_battles):
                await _await_battle(self._one_battle(i), (self.c1, self.c2),
                                    f"bridge battle {i}")
            return
        # Bounded-concurrency path. A single ``start_lock`` serializes each battle's team→creation
        # critical section (released the instant both battle objects exist — see ``_one_battle``),
        # exactly like the server's per-battle semaphore; the semaphore caps how many overlap.
        start_lock = asyncio.Lock()
        sem = asyncio.Semaphore(concurrency)

        async def _guarded(i: int) -> None:
            async with sem:
                # DELIBERATELY still a total-duration cap. The progress counter lives on the
                # CLIENT, and under concurrency several battles share one client — so a lively
                # neighbour would mask a wedged battle's silence, which is worse than no idle
                # bound at all. Per-battle attribution would need per-`battle_tag` counting in
                # `feed`; not built, because every default path (fuzz, parity, eval at
                # --eval-concurrency-per-worker 1) is sequential.
                await asyncio.wait_for(
                    self._one_battle(i, start_lock), timeout=_per_battle_timeout()
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
            *self._spawn_argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            limit=BRIDGE_STREAM_LIMIT,   # long-battle __RECON__ lines exceed the 64 KiB default
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
            if self.start_extra:
                start.update(self.start_extra)
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
            if text.startswith("__RECON__"):
                # The battle's full-information reconstruction record (referee view:
                # seed + both teams + command log). It goes ONLY to the bridge-layer
                # registry — never to a player/client — keyed by the battle tag so
                # the forensic trace writer can join it (see reconstruction.py).
                self._offer_recon(tag, text)
                continue
            side, b64 = text.split(" ", 1)
            chunk = base64.b64decode(b64).decode("utf-8")
            if self.chunk_sink is not None:
                self.chunk_sink.append((side, chunk))
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
    def _offer_recon(tag: str, text: str) -> None:
        """Decode a ``__RECON__`` frame and offer it to the reconstruction registry.
        Best-effort: a malformed record must never cost the battle."""
        try:
            raw = json.loads(base64.b64decode(text[len("__RECON__ "):]).decode("utf-8"))
            reconstruction.offer_record(tag, raw)
        except Exception as e:  # noqa: BLE001 — capture is telemetry, the battle is not
            import sys
            sys.stderr.write(f"[bridge] failed to capture reconstruction for {tag}: {e}\n")

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
            reap_budget = _teardown_reap_timeout()
            try:
                await asyncio.wait_for(proc.wait(), timeout=reap_budget)
            except asyncio.TimeoutError:  # pragma: no cover
                # Self-diagnosing, per the project timeout rule: a bare kill here is invisible, and
                # a killed child loses whatever it was still flushing. Say WHY the wait expired.
                sys.stderr.write(
                    f"[bridge] child did not exit within {reap_budget:.1f}s of END "
                    f"(base {_TEARDOWN_REAP_TIMEOUT:.1f}s x contention scale) — killing it. "
                    f"{describe_contention()}\n"
                )
                proc.kill()
                await proc.wait()
        stderr_task.cancel()
