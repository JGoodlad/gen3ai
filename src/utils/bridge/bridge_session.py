"""In-process Showdown transport for a poke-env ``PokeEnv`` — the RL-training analogue
of ``run_local_battles``.

``run_local_battles`` is a *synchronous driver*: it owns the battle loop and pulls
decisions from a self-contained ``choose_move``. That shape can't host SB3, whose action
arrives from *outside* the env via ``step()``. ``PokeEnv`` already solves that inversion of
control with its two ``_EnvPlayer`` agents and their ``battle_queue`` / ``order_queue``
handshake — the websocket is only the byte transport underneath. ``BridgeSession`` swaps
that transport: it replaces the ``/challenge`` handshake and the per-client websocket listen
loop with a local ``BattleStream`` subprocess and a background reader, leaving every higher
layer (obs, reward, mask, the wrappers, the reward manager) untouched.

The reader mirrors poke-env's own websocket ``listen()`` (``ps_client.py``): it reads one
protocol chunk, frames it with the room header poke-env expects, and **fires** the feed as a
task rather than awaiting it. That is mandatory, not an optimization — ``_EnvPlayer._choose_move``
blocks on ``order_queue`` awaiting SB3's action, so awaiting the feed inline would stall the
reader and deadlock the *other* side (whose request would never be delivered to ``reset()``).
Firing it is safe because ``_handle_message`` serializes same-battle handling under a
per-battle ``asyncio.Lock``, so chunks for one battle can never interleave and stay strictly
ordered (FIFO-fair lock). The two sides use independent clients (``c1``/``c2``), so one side
blocking on its order never blocks the other — exactly the two-independent-listen-loops
topology websockets give for free.

Two child-lifecycle modes:

- **persistent** (default) — ONE long-lived bridge child per env, reused across every episode.
  The first ``reset()`` spawns it and starts a single long-lived reader; each later ``reset()``
  sends a fresh ``START`` to the same child (the JS rebuilds a clean ``BattleStream`` per
  ``START``). This kills the per-episode Node-spawn cost — the reset/startup overhead the
  issue-#907 author flagged as the real bottleneck. A single reader owns stdout; an
  ``__END__``→``_battle_ended`` Event gates the next ``START`` so a battle's tail can't race the
  next battle's tag swap. **Two lifecycle rules:** (1) a child that *dies* mid-run **crashes** the
  env (no in-place recovery — resuming risks a corrupted PPO transition; the launcher restarts
  from checkpoint); (2) a *healthy* child is **recycled** every ``recycle_every`` battles — a
  backstop only, since RSS is measured flat (~229 MB, ~0 growth) and the launcher's 3h restart
  already owns the lifecycle.
- **spawn-per-battle** — a fresh child + reader per battle, torn down at ``__END__`` (identical
  to ``run_local_battles``' lifecycle, just driven as a background task). Kept as a fallback / A-B
  baseline.

Everything runs on the env's *own* event loop (``PokeEnv`` builds ``self._loop`` per env — not
the global ``POKE_LOOP``), so the clients, the reader, and the choice send-back all share that
loop, matching the queues ``_EnvPlayer`` created against it.
"""

from __future__ import annotations

import asyncio
import base64
import itertools
import json
import os
import signal
import time
from pathlib import Path
from typing import Dict, List, Optional

from poke_env.player.player import Player

from utils.bridge import bridge_trace
from utils.bridge.battle_stream_client import BattleStreamClient
from utils.bridge.sim_bridge_bin import bridge_spawn_argv

_BRIDGE_JS = str(Path(__file__).parent / "local_sim_bridge.js")

# Process-global, monotonically increasing battle number — every battle tag is unique across
# the whole worker process, mirroring how a real Showdown server hands out room ids and how
# ``local_battle_runner`` does it. A reused tag makes poke-env's ``_create_battle`` hand back
# the *previous* battle's object (it caches by tag), overflowing its team on the next switch
# (see ``project_bridge_unique_battle_tags``). One counter, never reset.
_BATTLE_SEQ = itertools.count(1)


def attach_bridge_transport(
    env, *, battle_format: str, seed: Optional[List[int]] = None, persistent: bool = True,
    recycle_every: int = 5000, impl: str = "node",
) -> "BridgeSession":
    """Swap a freshly-built ``PokeEnv``'s websocket transport for the local bridge.

    The env MUST have been built with ``start_listening=False`` so its two ``_EnvPlayer``
    agents never opened a websocket. ``persistent=True`` (default) reuses ONE long-lived bridge
    child across every episode (no Node spawn per ``reset()``); ``persistent=False`` falls back
    to a fresh child per battle.

    ``impl`` selects the child binary: ``"node"`` (default, the Node ``local_sim_bridge.js``) or
    ``"rust"`` (the byte-compatible ``src/rust_sim`` binary, resolved/built by
    ``sim_bridge_bin.resolve_sim_bridge_bin``). The two speak the identical protocol, so only the
    executable spawned differs.

    ``recycle_every`` (persistent only) replaces a HEALTHY child with a fresh one after that many
    battles. It is a **backstop, not a routine need**: a child's RSS was measured FLAT — ~189 MB
    fresh, a one-time ~+36 MB V8 warmup, then ~229 MB with ~0 growth over thousands of battles
    (V8 GC fully reclaims the per-battle ``BattleStream``). At production scale a child plays only
    ~2150 battles in the launcher's 3h restart window, so the default 5000 **never fires within a
    normal launcher run** (the 3h restart owns the lifecycle); it only caps marathon / no-launcher
    direct runs. ``0`` disables it. Returns the live ``BridgeSession`` (also stashed on
    ``env._bridge_session`` so it isn't GC'd and ``close()`` can reach it).
    """
    session = BridgeSession(
        env.agent1, env.agent2, battle_format, seed=seed, persistent=persistent,
        recycle_every=recycle_every, impl=impl,
    ).attach()
    # Keep a reference for the env's lifetime + tear the child down on env.close().
    env._bridge_session = session
    _wrap_close(env, session)
    return session


def _wrap_close(env, session: "BridgeSession") -> None:
    original_close = env.close

    def _close(*args, **kwargs):
        try:
            session.close()
        finally:
            return original_close(*args, **kwargs)

    env.close = _close


class BridgeSession:
    """Per-env bridge transport for a ``PokeEnv``'s two ``_EnvPlayer`` agents."""

    # Generous guard: a healthy in-process battle is seconds; if a reused child never emits
    # __END__ for the previous battle we fail loudly (crash → launcher restart) over hanging.
    _BATTLE_END_TIMEOUT = 180.0

    def __init__(
        self,
        agent1: Player,
        agent2: Player,
        battle_format: str,
        *,
        seed: Optional[List[int]] = None,
        persistent: bool = True,
        recycle_every: int = 5000,
        impl: str = "node",
    ):
        self.a1 = agent1
        self.a2 = agent2
        self.fmt = battle_format
        self.seed = seed
        self._persistent = persistent
        # Which sim-bridge child to spawn: "node" (default) or "rust". Resolved to an argv
        # list once (the rust path may build the binary) so the hot spawn stays cheap.
        self._impl = impl
        self._spawn_argv = bridge_spawn_argv(impl)
        # Recycle a HEALTHY persistent child after this many battles (0 = never) — a BACKSTOP for
        # marathon / no-launcher runs, not a routine need: child RSS is measured flat (plateaus
        # ~229 MB), and the launcher's 3h restart (~2150 battles/child) already owns the lifecycle,
        # so the 5000 default never fires under it. A DEAD child is never recycled — it crashes.
        self._recycle_every = max(0, recycle_every)
        self._battles_on_child = 0
        self._n_recycles = 0
        # Latched by the reader when the child's stdout hits EOF (the child exited) — distinguishes
        # a real crash from an intentional recycle/close cancel, robust to returncode-reap timing.
        self._child_crashed = False
        # Latched by the reader when DISPATCH itself fails — an ``__ERR__`` frame, or a malformed
        # line. The child usually survives such a frame (both bridges report-and-continue), so
        # ``_child_crashed`` alone would stay False while the reader task is dead: no further frame
        # is ever consumed and the next ``reset()`` waits forever. Latching the reason here turns
        # that silent hang into the loud crash the no-in-place-recovery rule wants.
        self._child_error: Optional[str] = None
        # Set synchronously by ``close()`` so ``_dispatch`` stops feeding poke-env before the
        # child is killed — otherwise a buffered request gets answered into a dead pipe.
        self._closing = False
        # _EnvPlayer was built with loop=env._loop; the queues it created live there, so the
        # clients/reader/choice-writes must all run on the same loop.
        self.loop = agent1.ps_client.loop
        self.c1: Optional[BattleStreamClient] = None
        self.c2: Optional[BattleStreamClient] = None
        self._proc: Optional[asyncio.subprocess.Process] = None
        self._reader: Optional[asyncio.Future] = None
        self._stderr_task: Optional[asyncio.Future] = None
        self._stderr_buf: List[bytes] = []
        self._inited: Dict[str, bool] = {"p1": False, "p2": False}
        self._tag: Optional[str] = None
        # Persistent mode only: the long-lived child + reader are started on the first battle,
        # and ``_battle_ended`` is set by the reader when it sees ``__END__`` so the next battle's
        # START (and tag swap) can't race the prior battle's tail.
        self._started = False
        self._battle_ended: Optional[asyncio.Event] = None
        # Diagnostics (persistent mode): seconds the LAST reset spent waiting for the previous
        # battle's ``__END__`` before sending the next START. A slow child (descheduled under load,
        # node GC) shows up here — the prime suspect for an outlier-slow episode. Cheap to record.
        self._last_end_wait_s: float = 0.0
        # Latest battle's reconstruction record as (battle_tag, base64-json), overwritten every
        # episode. Single-slot + undecoded on purpose — see _dispatch's __RECON__ branch.
        self.last_recon: Optional[tuple] = None

    def attach(self) -> "BridgeSession":
        self.c1 = self._make_client(self.a1, "p1")
        self.c2 = self._make_client(self.a2, "p2")
        self.a1.ps_client = self.c1
        self.a2.ps_client = self.c2
        # PokeEnv.reset() kicks off a battle with ``agent1.battle_against(agent2, n_battles=1)``
        # (the /challenge handshake). Intercept that single seam — the rest of reset()
        # (battle_queue.get() for both sides) is transport-agnostic and unchanged.
        self.a1.battle_against = self._battle_against  # type: ignore[assignment]
        return self

    def _make_client(self, player: Player, side: str) -> BattleStreamClient:
        return BattleStreamClient(
            player.ps_client._account_configuration,
            side=side,
            on_battle_message=player._handle_battle_message,
            on_update_challenges=player._update_challenges,
            on_challenge_request=player._handle_challenge_request,
            loop=self.loop,
        )

    # --- battle start (replaces battle_against / the challenge handshake) --------------
    async def _battle_against(self, *opponents: Player, n_battles: int = 1) -> None:
        """Drop-in for ``_EnvPlayer.battle_against`` — scheduled by ``PokeEnv.reset`` on
        ``env._loop``. Starts a bridge battle and returns: the reader fills each agent's
        ``battle_queue`` in the background, which ``reset()`` awaits with ``battle_queue.get()``
        — exactly how the websocket challenge coroutines behave. Persistent mode reuses one
        long-lived child across episodes; spawn-per-battle mode spawns a fresh one each time."""
        if self._persistent:
            await self._start_persistent_battle()
        else:
            await self._start_spawned_battle()

    async def _spawn_child(self) -> asyncio.subprocess.Process:
        # _spawn_argv is ["node", local_sim_bridge.js] or [<rust sim_bridge binary>] — both
        # speak the identical stdin/stdout protocol, so the framing/demux below is unchanged.
        return await asyncio.create_subprocess_exec(
            *self._spawn_argv,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

    async def _send_start(self, proc, *, persistent: bool) -> str:
        """Assign a fresh unique tag, register it on both clients, and write START with both
        teams. The tag is set BEFORE the write, so the reader frames the new battle's chunks
        (which the child only emits after receiving START) with the new tag."""
        self._inited = {"p1": False, "p2": False}
        old_tag = self._tag
        tag = f"battle-{self.fmt}-{next(_BATTLE_SEQ)}"
        self._tag = tag
        self.c1._procs[tag] = proc
        self.c2._procs[tag] = proc
        if old_tag is not None:  # drop the prior battle's registration (persistent reuse)
            self.c1._procs.pop(old_tag, None)
            self.c2._procs.pop(old_tag, None)
        # get_next_team() yields a packed team AND sets player._current_packed_team (read by
        # _create_battle for team tracking). Both agents draw from the env's teambuilder, same
        # as run_local_battles.
        team1 = self.a1.get_next_team()
        team2 = self.a2.get_next_team()
        start = {
            "formatid": self.fmt,
            "p1": {"name": self.a1.username, "team": team1},
            "p2": {"name": self.a2.username, "team": team2},
        }
        if persistent:
            start["persistent"] = True
        if self.seed:
            start["seed"] = self.seed
        proc.stdin.write((f"START {json.dumps(start)}\n").encode())
        await proc.stdin.drain()
        return tag

    # --- spawn-per-battle mode --------------------------------------------------------
    async def _start_spawned_battle(self) -> None:
        proc = await self._spawn_child()
        self._proc = proc
        self._stderr_buf = []
        self._stderr_task = asyncio.ensure_future(self._drain_stderr(proc, self._stderr_buf))
        tag = await self._send_start(proc, persistent=False)
        self._reader = asyncio.ensure_future(self._spawned_read_loop(proc, tag))

    async def _spawned_read_loop(self, proc, tag: str) -> None:
        """Read one battle's chunks; tear the child down at ``__END__``."""
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode().rstrip("\n")
                if text == "__END__":
                    break
                self._dispatch(text)
        finally:
            self.c1._procs.pop(tag, None)
            self.c2._procs.pop(tag, None)
            await self._teardown(proc)

    # --- persistent mode (one long-lived child + reader across episodes) --------------
    async def _spawn_persistent_child(self) -> None:
        """Spawn a fresh long-lived child + its single reader, with a fresh ``_battle_ended``.
        Used on the first battle AND on a clean recycle. The reader captures its OWN event so a
        recycled child's old reader can't spuriously set the new child's event."""
        self._child_crashed = False
        self._child_error = None
        self._proc = await self._spawn_child()
        self._battle_ended = asyncio.Event()
        self._stderr_buf = []
        self._stderr_task = asyncio.ensure_future(self._drain_stderr(self._proc, self._stderr_buf))
        self._reader = asyncio.ensure_future(
            self._persistent_read_loop(self._proc, self._battle_ended)
        )

    def _child_alive(self) -> bool:
        return (
            self._proc is not None
            and self._proc.returncode is None
            and self._reader is not None
            and not self._reader.done()
        )

    async def _start_persistent_battle(self) -> None:
        if not self._started:
            await self._spawn_persistent_child()
            self._started = True
            self._battles_on_child = 0
        else:
            # The previous battle MUST have emitted __END__ before we reuse the child — else a
            # new START would race the prior battle's still-closing streams. The reader also sets
            # the event in its finally if the child died, so this returns promptly either way.
            _t0 = time.perf_counter()
            try:
                await asyncio.wait_for(
                    self._battle_ended.wait(), timeout=self._BATTLE_END_TIMEOUT
                )
            except asyncio.TimeoutError:
                raise RuntimeError(
                    "bridge child did not finish the previous battle (no __END__ within "
                    f"{self._BATTLE_END_TIMEOUT}s) — crashing (no in-place recovery)"
                )
            self._last_end_wait_s = time.perf_counter() - _t0
            # A dispatch failure retires the reader while the child stays alive, so check it
            # BEFORE the liveness test — otherwise the real reason (the __ERR__ text) is lost
            # behind a generic "exited unexpectedly".
            if self._child_error is not None:
                raise RuntimeError(
                    "bridge child reported an error mid-run — crashing (no in-place recovery): "
                    f"{self._child_error}"
                )
            # A child that exited unexpectedly means lost / inconsistent battle state. We do NOT
            # recover in place: resuming risks feeding PPO a corrupted transition, so we CRASH and
            # let the launcher restart from the last checkpoint (the project's crash-over-corruption
            # rule, same as the trainee's stale-decision path).
            if self._child_crashed or not self._child_alive():
                stderr = b"".join(self._stderr_buf[-20:]).decode("utf-8", "replace")
                raise RuntimeError(
                    "bridge child exited unexpectedly mid-run — crashing (no in-place recovery). "
                    f"child stderr tail:\n{stderr}"
                )
            # HEALTHY child + recycle cap reached → cleanly swap in a fresh one. This bounds any
            # V8-heap growth over a long run; it is NOT crash recovery (the child is alive and the
            # prior battle ended cleanly, so no state is lost).
            if self._recycle_every and self._battles_on_child >= self._recycle_every:
                await self._recycle_child()
                self._n_recycles += 1
                self._battles_on_child = 0
        self._battle_ended.clear()
        self._battles_on_child += 1
        await self._send_start(self._proc, persistent=True)

    async def _recycle_child(self) -> None:
        """Replace a HEALTHY, idle (just-finished-a-battle) child with a fresh one to bound heap
        growth. Spawns the new child FIRST (so ``self._*`` point to it), then cancels the old
        reader/stderr and tears the old child down explicitly. Only ever called between battles on
        a live child — never to mask a crash."""
        old_proc, old_reader, old_stderr = self._proc, self._reader, self._stderr_task
        if self._tag is not None:  # drop the last battle's tag before the proc swap
            self.c1._procs.pop(self._tag, None)
            self.c2._procs.pop(self._tag, None)
            self._tag = None
        await self._spawn_persistent_child()
        for t in (old_reader, old_stderr):
            if t is not None and not t.done():
                t.cancel()
        await self._teardown(old_proc)

    async def _persistent_read_loop(self, proc, ended_event: asyncio.Event) -> None:
        """One reader for ONE child's lifetime. Frames each chunk with the CURRENT tag (swapped
        between battles by ``_send_start`` before any new chunk can arrive) and signals ``__END__``
        via ``ended_event`` (captured, not ``self._battle_ended``, so a recycled child's old reader
        can't set the new child's event). Does NOT tear the proc down — that's owned by
        ``_recycle_child`` / ``close()`` (intentional) or the OS on the crash exit; the reader only
        latches ``_child_crashed`` on stdout EOF so ``_start_persistent_battle`` raises."""
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    # stdout EOF = the child process exited. On a recycle/close we CANCEL this
                    # reader (raises at readline, skips here), so reaching this is a real crash.
                    self._child_crashed = True
                    break
                text = line.decode().rstrip("\n")
                if text == "__END__":
                    ended_event.set()
                    continue
                try:
                    self._dispatch(text)
                except Exception as e:
                    # An ``__ERR__`` frame (or a malformed line) must not silently retire the
                    # reader — latch it so the next battle's start raises with the reason.
                    self._child_error = f"{type(e).__name__}: {e}"
                    break
        finally:
            # Unblock any reset() waiting on this child's end — so a crash surfaces as the explicit
            # raise in _start_persistent_battle, not a 180s wait timeout. (Set _child_crashed FIRST,
            # above, so the woken waiter sees it.)
            ended_event.set()

    # --- inbound dispatch (mirrors ps_client.listen) ----------------------------------
    def _dispatch(self, text: str) -> None:
        """Frame one bridge stdout line and FIRE its feed as a task (never awaited) — see the
        module docstring. The per-battle lock inside ``_handle_message`` keeps same-battle
        handling ordered, and the two clients are independent, so one side's blocked
        ``_choose_move`` cannot stall reads of the other side's request."""
        if self._closing:
            # Teardown in progress: the child is being killed, so feeding this frame would only
            # make poke-env answer into a dead pipe. Drop it (no state depends on a post-close
            # frame — the env is going away).
            return
        if text.startswith("__ERR__"):
            msg = base64.b64decode(text[len("__ERR__ ") :]).decode("utf-8")
            raise RuntimeError(f"local_sim_bridge error: {msg}")
        if text.startswith("__RECON__"):
            # Full-information reconstruction record (referee view) — NEVER fed to a
            # player/client. Training persists no forensic traces, so this is kept as
            # a SINGLE-SLOT raw stash (latest episode only, ~tens of KB, undecoded):
            # routing it into the bounded reconstruction registry instead would pin
            # up to 64 full records in every env worker for nothing. Tools that want
            # a training episode's record can read (tag, b64) and decode on demand.
            self.last_recon = (self._tag, text[len("__RECON__ "):])
            return
        side, b64 = text.split(" ", 1)
        chunk = base64.b64decode(b64).decode("utf-8")
        bridge_trace.inbound(side, chunk)
        client = self.c1 if side == "p1" else self.c2
        framed = self._frame(self._tag, side, chunk)
        asyncio.ensure_future(client.feed(framed))

    def _frame(self, tag: str, side: str, chunk: str) -> str:
        # The sim emits no room header; poke-env keys the battle off ">battle-…" + "|init|battle".
        # Prepend them, with |init| only on the first chunk per side so _create_battle fires once.
        header = f">{tag}\n"
        if not self._inited[side]:
            self._inited[side] = True
            header += "|init|battle\n"
        return header + chunk

    # --- teardown ---------------------------------------------------------------------
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

    async def _teardown(self, proc) -> None:
        # Tear down ONE proc. Deliberately does NOT touch ``self._stderr_task`` — during a recycle
        # the old reader's finally runs this AFTER ``_spawn_persistent_child`` reassigned that attr
        # to the NEW child, so cancelling it here would kill the live child's stderr drain. Each
        # proc's own ``_drain_stderr`` loop ends naturally when the proc exits (readline → EOF);
        # ``close()`` cancels the current one explicitly.
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

    def close(self) -> None:
        """Synchronously tear down any live bridge child (called from env.close()).

        ``close()`` runs on the MAIN thread, but the asyncio subprocess belongs to ``env._loop``'s
        thread — ``proc.kill()`` (which pokes the loop's transport) is not safe to call from here
        and can silently no-op, leaking the Node child. Kill by PID with ``os.kill`` instead: a
        plain OS syscall, thread-agnostic, and SIGKILL needs no reaping handshake."""
        # Stop DISPATCHING first. Task cancellation is asynchronous (it only schedules on the
        # loop), so between the kill and the reader actually stopping it can still feed a buffered
        # ``|request|`` to poke-env, which answers it by writing to the child we just killed —
        # the "Unhandled exception in _handle_message / ConnectionResetError" teardown noise.
        # This flag is set SYNCHRONOUSLY, so `_dispatch` goes quiet before the child dies.
        self._closing = True
        for task in (self._reader, self._stderr_task):
            if task is not None and not task.done():
                self.loop.call_soon_threadsafe(task.cancel)
        proc = self._proc
        if proc is not None and proc.returncode is None:
            try:
                os.kill(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, OSError):  # pragma: no cover
                pass
