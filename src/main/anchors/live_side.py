"""OUR side of an anchor read as a WEBSOCKET CLIENT on the Rust stack (poke-env retirement P6, ``T27``).

The core slot (:mod:`main.anchors.core_side`) needs the battle to run in OUR process, so it serves exactly one
shape: a checkpoint on a ``--server rust`` this tool started. Every other shape — ``--server node``,
``--server-uri``, and a ``bot:<name>`` our-side on any server — puts our side on the socket as an ordinary
client, and that client is P4's :class:`main.live.client.LiveClient` (``--our-transport live``):

* **the observation** is :class:`main.live.reader.LiveReader`'s frame — the ``live_reader`` binary folding the
  lines the server sent us through ``pokesim::side_reader::SideReader``, the chain ``sim_bridge``'s core
  observation mode ships to training (P4 gate (a): byte-equal rows on 318,465 decisions);
* **a checkpoint** decides through :class:`main.anchors.core_side.CorePolicy` — the core slot's own arithmetic
  (``RLPlayer._predict_best_action``'s masked logits, ``$GEN3AI_POLICY_SEED``'s generator), so a row equal to
  the core slot's gives the action the core slot gives;
* **a bot** is the Rust port of that roster bot (:class:`main.live.bot_reader.BotReader`), deciding on our
  side's own reading of the battle, its RNG streams seeded exactly as the env core's bot route seeds them
  (``opponents::stream_seed(--bot-seed, half index, k)``) and RUNNING ACROSS the half's games (one bot per
  half, as one ``Player`` per half was). A bot never forfeits at the turn limit (F-LF-5: the Python bots
  carried no stall check, and neither does the env core's bot route); the row still stamps the plan's limit.

**Rows** are this module's :class:`main.anchors.session.BattleRecord` per finished battle, with ``n_decisions``
per GAME (the core slot's convention; the legacy ``RLPlayer`` counter was cumulative over a half) and
``n_defaults`` / ``n_redecides`` 0 by construction (the client decides exactly at the requests the reader opens).

**Failures are named.** A reader refusal, an illegal policy choice or an ``[Invalid choice]`` is P4's T28
:class:`main.live.halt.LiveParseHalt`: the read RECORDS the halt marker (all live play stops until it is
root-caused, ``python -m main.live.halt``) and fails as ``live_parse_halt``; any other client failure (a login
that never completes, a rejected team, a dropped socket) fails as ``live_client_error``.
"""
from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional, Tuple

from main.anchors.session import BattleRecord, OurSideState, SeriesFailure, build_team_source

#: ``our_transport`` row stamps of this module's two kinds of side.
TRANSPORT_LIVE = "rust_live_reader"
TRANSPORT_LIVE_BOT = "rust_live_bot"

#: Each half's ENV index in the bot's stream seed (``opponents::stream_seed(seed, env, k)``): one bot per half.
HALF_ENV = {"ours_challenge": 0, "peer_challenge": 1}

#: A bot never forfeits at the trainer's turn limit (F-LF-5); the client's limit is set past any battle.
_NO_FORFEIT = 10 ** 9


def bot_streams(seed: int, half: str) -> Dict[str, int]:
    """``{"choice": s0, "protect": s1}`` — the bot's two stream seeds in ``half`` (the Python twin of the Rust
    rule, stamped in the half's report so a reader can re-seed the same bot)."""
    from agents.training.rust_env_opponents import bot_stream_seed

    env = HALF_ENV[half]
    return {"choice": bot_stream_seed(seed, env, 0), "protect": bot_stream_seed(seed, env, 1)}


class FramePolicy:
    """A checkpoint on the live reader's frame — :class:`core_side.CorePolicy`'s decision, with the
    per-decision verification (``stochastic``, argmax match, decision time) the core slot records."""

    def __init__(self, core_policy: Any, state: OurSideState) -> None:
        self.core = core_policy
        self.state = state
        self.last: Optional[Tuple[int, int]] = None

    def choose(self, frame: Any) -> int:
        import numpy as np

        t0 = time.perf_counter()
        idx, argmax = self.core.decide(frame.row, np.asarray(frame.mask, dtype=np.int8))
        self.state.decision_times.append(time.perf_counter() - t0)
        stoch = bool(self.core.stochastic)
        if stoch not in self.state.stochastic_kwargs:
            self.state.stochastic_kwargs.append(stoch)
        self.last = (int(idx), int(argmax))
        return int(idx)


def _client_cls() -> type:
    from main.live.client import LiveClient

    class AnchorClient(LiveClient):
        """:class:`LiveClient` that appends one :class:`BattleRecord` per finished battle as it finishes, so a
        series that dies halfway keeps what completed (the watchdog reads ``state.records``)."""

        def __init__(self, *a: Any, state: OurSideState, forfeit_limit: int,
                     argmax: Optional[Dict[str, List[int]]] = None, **k: Any) -> None:
            super().__init__(*a, **k)
            self.state = state
            self.forfeit_limit = int(forfeit_limit)
            self.argmax = argmax

        def _on_battle(self, room: str, lines: List[str]) -> None:
            n0 = len(self.results)
            super()._on_battle(room, lines)
            for res in self.results[n0:]:
                self._record(res)

        def _record(self, res: Any) -> None:
            n = len(self.state.records) + 1
            counts = self.argmax.get(res.battle, [0, 0]) if self.argmax is not None else None
            self.state.records.append(BattleRecord(
                battle_tag=res.battle,
                turns=int(res.turns),
                won=res.won,
                finished=True,
                # a cap forfeit reads as an ordinary loss; only the turn count separates it
                hit_forfeit_limit=int(res.turns) >= self.forfeit_limit,
                our_team=(self.state.draws[n - 1] if n <= len(self.state.draws) else None),
                n_decisions=int(res.decisions),
                n_defaults=0,
                n_redecides=0,
                t_finished=time.time(),
                our_argmax_matches=(counts[0] if counts is not None else None),
                our_argmax_decisions=(counts[1] if counts is not None else None),
            ))
            self.state.last_progress = time.time()
            rec = self.state.records[-1]
            print(f"[anchors] game {n}: {rec.result} in {rec.turns} turns "
                  f"(cap={rec.hit_forfeit_limit}, team={rec.our_team})", flush=True)

    return AnchorClient


def build_client(plan: Any, half: str, *, username: str, team_spec: Dict[str, Any], state: OurSideState) -> Any:
    """Our side's client for one half: a checkpoint (``plan.our_side == "model"``) or a Rust bot."""
    from main.live.client import ClientConfig

    teams = build_team_source(team_spec, plan.team_seed, state.draws)
    is_bot = plan.our_side.startswith("bot:")
    cfg = ClientConfig(uri=plan.server_uri, username=username, battle_format=plan.battle_format,
                       auth="local", connect_timeout_s=float(plan.connect_timeout_s),
                       forfeit_turn_limit=(_NO_FORFEIT if is_bot else int(plan.forfeit_turn_limit)),
                       battle_idle_timeout_s=float(plan.progress_timeout_s))
    if is_bot:
        from main.live.bot_reader import BotPolicy, BotReader

        name = plan.our_side.split(":", 1)[1]
        seed = int(plan.bot_seed)
        env = HALF_ENV[half]
        state.model_loader = ""
        return _client_cls()(cfg, policy=BotPolicy(), team_fn=teams.yield_team,
                             reader_factory=lambda: BotReader(name, seed=seed, env=env),
                             state=state, forfeit_limit=plan.forfeit_turn_limit)
    from main.anchors import core_side

    stochastic, temperature = core_side.our_stochastic(plan)
    model = core_side.load_policy(plan.model_zip, plan.device, plan.model_loader, state)
    policy = FramePolicy(core_side.CorePolicy(model, stochastic=stochastic, temperature=temperature), state)
    argmax: Dict[str, List[int]] = {}

    def on_decision(d: Any) -> None:
        # the stall forfeit (index None) is not a decision, exactly as the core slot counts
        if d.index is None or policy.last is None:
            return
        idx, am = policy.last
        c = argmax.setdefault(d.battle, [0, 0])
        c[0] += int(idx == am)
        c[1] += 1

    return _client_cls()(cfg, policy=policy, team_fn=teams.yield_team, on_decision=on_decision,
                         state=state, forfeit_limit=plan.forfeit_turn_limit, argmax=argmax)


async def play_half(plan: Any, half: str, n_games: int, pplan: Any, team_spec: Dict[str, Any],
                    state: OurSideState, our_name: str, peer_name: str, role: str,
                    ) -> Tuple[Any, Optional[SeriesFailure]]:
    """One half with OUR side as the live client. The same role order, watchdog and peer teardown as the core
    slot: the acceptor is online before the challenger, :func:`session.watch` turns a dead peer or a stall into
    a named failure, the peer writes its own report before anything terminates it, and our socket is closed
    before the next half logs in (hazard H10)."""
    from main.anchors.session import await_peer_exit, await_peer_ready, start_peer, stop_peer, watch
    from main.live.client import ClientError
    from main.live.halt import LiveParseHalt

    client = None
    proc = None
    failure: Optional[SeriesFailure] = None
    our_task: Optional[asyncio.Task] = None
    try:
        client = build_client(plan, half, username=our_name, team_spec=team_spec, state=state)
        if role == "acceptor":
            # the PEER accepts: its banner is the gate (minutes of model build for the 200M policy)
            proc = start_peer(pplan, nice=plan.nice)
            await await_peer_ready(proc, pplan, plan.peer_ready_timeout_s)
            await client.connect()
            print(f"[anchors] our side online as {client.name} (live client, {plan.our_transport})", flush=True)
            our_task = asyncio.create_task(client.challenge(peer_name, n_games, plan.first_game_timeout_s))
        else:
            # WE accept, so we are online before the peer starts
            await client.connect()
            print(f"[anchors] our side online as {client.name} (live client, {plan.our_transport})", flush=True)
            our_task = asyncio.create_task(client.accept(peer_name, n_games))
            proc = start_peer(pplan, nice=plan.nice)
        watchdog = asyncio.create_task(
            watch(proc, pplan, state, n_games, plan.first_game_timeout_s, plan.progress_timeout_s))
        done, _pending = await asyncio.wait({our_task, watchdog}, return_when=asyncio.FIRST_COMPLETED)
        if watchdog in done and not watchdog.cancelled():
            exc = watchdog.exception()
            if exc is not None:
                raise exc
        watchdog.cancel()
        if our_task in done:
            our_task.result()
        else:
            try:
                await asyncio.wait_for(asyncio.shield(our_task), timeout=120.0)
            except asyncio.TimeoutError:
                our_task.cancel()
    except LiveParseHalt as exc:
        # T28: a parse panic HALTS all live play — the marker is recorded here, the read fails by name.
        from main.live.halt import record_halt

        path = record_halt(reason=exc.reason, entry_point="main.anchors", battle_id=exc.battle_id,
                           offending_lines=exc.offending_lines, stream=exc.stream, exc=exc)
        failure = SeriesFailure("live_parse_halt", f"{exc.reason} (battle {exc.battle_id}; HALT marker {path})")
        print(f"[anchors] 🛑 {half} HALTED — {failure.detail}", flush=True)
    except SeriesFailure as exc:
        failure = exc
        print(f"[anchors] 🚨 {half} FAILED — {exc.cause}: {exc.detail}", flush=True)
    except ClientError as exc:
        failure = SeriesFailure("live_client_error", f"{type(exc).__name__}: {exc}")
        print(f"[anchors] 🚨 {half} FAILED — our client: {exc}", flush=True)
    except Exception as exc:                            # noqa: BLE001 - any death must be NAMED
        failure = SeriesFailure("our_side_error", f"{type(exc).__name__}: {exc}")
        print(f"[anchors] 🚨 {half} FAILED — our side raised: {exc}", flush=True)
    finally:
        if our_task is not None and not our_task.done():
            our_task.cancel()
            try:
                await our_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        await await_peer_exit(proc, timeout_s=180.0 if failure is None else 15.0)
        if client is not None:
            await client.close()
        stop_peer(proc)
    return proc, failure
