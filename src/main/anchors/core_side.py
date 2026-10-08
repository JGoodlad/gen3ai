"""OUR side of an anchor read as an IN-PROCESS slot of the Rust websocket front end (P3, ``T27``).

**What changed, and why.** Until P3 our checkpoint played an anchor read as a poke-env WEBSOCKET
CLIENT (``main.play`` → ``RLPlayer``): it parsed the Showdown protocol with the vendored poke-env,
folded it through the Python battle layer and built its observation with the PYTHON encoder — a
second implementation of the reader training does not use. With ``--server rust`` the battle runs
in OUR ``sim_bridge`` anyway, so our side does not need to be a client at all: the front end runs
IN THIS PROCESS and our slot reads the core's own ``__OBS__`` frame (``sim_bridge``'s
``gen3_bridge_core_obs_v1`` — the training reader, unchanged: the row, the mask and the choice
token per legal action). Only the OPPONENT stays on the socket, in its own process with its own
upstream poke-env (``peer_scripts/metamon_side.py``; Foul Play). This process never imports
``poke_env`` on this path — ``src/poke_env_free_entry_points_test.py`` runs it with the import
blocked.

**The slot speaks the client protocol, through a function call instead of a socket.** It sends
``/trn``, ``/utm``, ``/challenge``, ``/accept``, ``/choose <token>|<rqid>`` and ``/forfeit`` into
:meth:`ShowdownFrontEnd.client_line`, and receives exactly the frames a websocket client would, so
every server-side rule (the 18-character name, team validation, the rqid refusal) applies to it
unchanged and the peer cannot tell the difference.

**What it reproduces from ``RLPlayer``, line for line:** the decision math of
``RLPlayer._predict_best_action`` (masked logits ``logits + (mask - 1) · 1e9``; greedy = argmax,
sampled = ``Categorical(masked / T)`` on the shared torch generator, or ``$GEN3AI_POLICY_SEED``'s
private one), the trainer's forfeit (``turn >= limit`` at a decision ⇒ ``/forfeit``), and the team
draw order (one ``yield_team()`` per battle, before the challenge or the accept). What it does NOT
reproduce: ``RLPlayer``'s re-decide loop and default move — the core decides at exactly the
requests it opens, so there is nothing stale to re-decide and no empty-mask decision to default;
``n_redecides`` / ``n_defaults`` are therefore 0 by construction.

**Every protocol surprise is a NAMED failure, never a skip:** a non-wait ``|request|`` with no
core frame, a frame nobody consumed, a frame out of order, an unwritten (NaN) cell, a chosen index
the core's mask forbids, an ``|error|[Invalid choice]``, a ``|popup|`` or a ``|nametaken|``. Each
raises :class:`main.anchors.session.SeriesFailure` with cause ``core_slot_error``.
"""
from __future__ import annotations

import asyncio
import json
import os
import time
from typing import Any, Dict, Optional

import numpy as np

from main.anchors.session import BattleRecord, OurSideState, SeriesFailure, build_team_source

#: ``--our-transport`` values. ``core`` (the DEFAULT) is this module; ``poke-env`` is the legacy
#: ``main.play`` → ``RLPlayer`` websocket client, kept for ``--server node`` / ``--server-uri`` /
#: a ``bot:`` our-side (a Python roster bot) and for the transport differential.
OUR_TRANSPORTS = ("core", "poke-env")

#: What every row stamps as ``our_transport`` — the reader that built OUR observation. A REGIME
#: BOUNDARY (P3, 2026-10-07): rows without the field were all ``poke_env_rlplayer``.
TRANSPORT_CORE = "rust_core_slot"
TRANSPORT_POKE_ENV = "poke_env_rlplayer"
TRANSPORT_POKE_ENV_BOT = "poke_env_bot"
TRANSPORT_PEER = "peer"

#: The sampling generator seed (the same variable ``RLPlayer`` reads — ``gen3_policy_sample_rng_v1``).
_POLICY_SEED_ENV = "GEN3AI_POLICY_SEED"


def _policy_seed() -> Optional[int]:
    env = os.environ.get(_POLICY_SEED_ENV)
    if env is None or env == "":
        return None
    try:
        return int(env)
    except ValueError as exc:
        raise ValueError(f"${_POLICY_SEED_ENV}={env!r} is not an integer seed") from exc


def _fail(detail: str) -> SeriesFailure:
    return SeriesFailure("core_slot_error", detail)


# ------------------------------------------------------------------------------- the policy
def load_policy(path: str, device: str, model_loader: str, state: OurSideState) -> Any:
    """Load our checkpoint — ``main.play.load_policy``'s body (``play`` imports poke-env, this must
    not), with ``main.anchors``' ``auto``/``bare``/``foreign`` rule. Which loader ran is recorded in
    ``state.model_loader`` and stamped on every row."""
    from agents.model.oracle_reveal import refuse_if_revealed
    from agents.model.snapshot import historical_load_kwargs, load_checkpoint_strict
    from main.anchors.session import foreign_loader

    # The core slot reads `sim_bridge`'s core_obs frame, which never writes the reveal.
    refuse_if_revealed(path, tool="main.anchors",
                       reason="The core slot's observation is built without the reveal.")
    if model_loader == "foreign":
        state.model_loader = "foreign"
        return foreign_loader(path, device)
    try:
        model = load_checkpoint_strict(path, device=device, **historical_load_kwargs(path))
        state.model_loader = "bare"
        return model
    except Exception as exc:                          # noqa: BLE001 - the CAUSE is reported
        if model_loader != "auto":
            raise
        print(f"[anchors] bare load failed ({type(exc).__name__}: {str(exc)[:140]}) — retrying "
              "through load_foreign_opponent, which verifies the arch_signature", flush=True)
        model = foreign_loader(path, device)
        state.model_loader = "foreign"
        return model


class CorePolicy:
    """``RLPlayer._predict_best_action``'s decision on a core row — the same arithmetic, so a row
    byte-equal to the Python encoder's gives the same action."""

    def __init__(self, model: Any, *, stochastic: bool, temperature: float) -> None:
        self.model = model
        self.stochastic = bool(stochastic)
        self.temperature = float(temperature)
        self._seed = _policy_seed()
        self._gen: Any = None

    def _generator(self, device: Any) -> Any:
        if self._seed is None:
            return None
        if self._gen is None:
            import torch

            self._gen = torch.Generator(device=device)
            self._gen.manual_seed(int(self._seed))
        return self._gen

    def decide(self, row: np.ndarray, mask: np.ndarray) -> "tuple[int, int]":
        """``(chosen index, argmax index)`` for one decision."""
        import torch

        with torch.no_grad():
            obs_t = torch.as_tensor(np.expand_dims(np.array(row, dtype=np.float32), 0)).to(self.model.device)
            mask_t = torch.as_tensor(np.expand_dims(np.asarray(mask, dtype=np.int8), 0)).to(self.model.device)
            dist = self.model.policy.get_distribution({"observation": obs_t, "action_mask": mask_t})
            masked = dist.distribution.logits + (mask_t - 1.0) * 1e9
            argmax = int(torch.argmax(masked, dim=1).item())
            if not self.stochastic:
                return argmax, argmax
            sample_logits = masked / self.temperature if self.temperature != 1.0 else masked
            cat = torch.distributions.Categorical(logits=sample_logits)
            gen = self._generator(sample_logits.device)
            if gen is None:
                idx = int(cat.sample().item())
            else:
                idx = int(torch.multinomial(cat.probs, 1, True, generator=gen).item())
            return idx, argmax


# --------------------------------------------------------------------------------- the slot
def _conn_base() -> type:
    from utils.bridge.ws_frontend import _Conn

    return _Conn


class _BattleState:
    __slots__ = ("tag", "turn", "frames", "pending", "n_decisions", "argmax_matches", "ended",
                 "started_at")

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.turn = 0
        self.frames = 0                 # core frames received for our side in this battle
        self.pending: Optional[Dict[str, Any]] = None
        self.n_decisions = 0
        self.argmax_matches = 0
        self.ended = asyncio.Event()
        self.started_at = time.time()


def make_slot(front: Any, *, username: str, policy: CorePolicy, team_source: Any,
              state: OurSideState, forfeit_limit: int, battle_format: str = "gen3ou") -> Any:
    """Build the slot as a subclass of the front end's own connection record (imported lazily so
    importing this module stays cheap)."""
    base = _conn_base()

    class CoreSlot(base):  # type: ignore[misc,valid-type]
        """One user of the in-process front end whose moves come from the core's frames."""

        def __init__(self) -> None:
            super().__init__(ws=None, guest_n=front.next_guest())
            self.core_obs = True
            self.front = front
            self.username = username
            self.policy = policy
            self.teams = team_source
            self.state = state
            self.forfeit_limit = int(forfeit_limit)
            self.fmt = battle_format
            self.logged_in = asyncio.Event()
            self.challenges: "asyncio.Queue[str]" = asyncio.Queue()
            self.battles: Dict[str, _BattleState] = {}
            self.battle_started = asyncio.Event()
            self.current: Optional[_BattleState] = None
            self.failure: Optional[SeriesFailure] = None

        # -- failure is sticky and loud ------------------------------------------------------
        def _record_failure(self, exc: SeriesFailure) -> None:
            if self.failure is None:
                self.failure = exc
                self.state.error = f"{exc.cause}: {exc.detail}"
                print(f"[anchors] 🚨 core slot: {exc.detail}", flush=True)
            # wake every waiter so the driver sees it at once
            self.battle_started.set()
            for b in self.battles.values():
                b.ended.set()

        def check(self) -> None:
            if self.failure is not None:
                raise self.failure

        # -- the client half ---------------------------------------------------------------
        async def line(self, text: str) -> None:
            await self.front.client_line(self, text)

        async def login(self) -> None:
            await self.line(f"|/trn {self.username},0,")
            if not self.logged_in.is_set():
                self.check()
                raise _fail(f"login as {self.username!r} was not answered with |updateuser|")
            print(f"[anchors] our side online as {self.username} (in-process core slot)", flush=True)

        async def _set_team(self) -> None:
            packed = self.teams.yield_team()
            await self.line(f"|/utm {packed}")

        async def _await_started(self, timeout_s: float) -> _BattleState:
            try:
                await asyncio.wait_for(self.battle_started.wait(), timeout=timeout_s)
            except asyncio.TimeoutError:
                raise SeriesFailure("no_first_game", f"no battle started within {timeout_s:g}s")
            self.battle_started.clear()
            self.check()
            assert self.current is not None
            return self.current

        async def challenge_series(self, opponent: str, n: int, start_timeout_s: float) -> None:
            """``n`` battles, each challenge sent only after the previous battle ENDED (hazard
            H14's serial loop — the only mode a single slot needs)."""
            for _ in range(n):
                self.check()
                await self._set_team()
                await self.line(f"|/challenge {opponent}, {self.fmt}")
                self.check()
                battle = await self._await_started(start_timeout_s)
                await battle.ended.wait()
                self.check()

        async def accept_series(self, opponent: str, n: int) -> None:
            """``n`` battles: draw the team, wait for THIS opponent's challenge, accept, wait for
            the battle to start — poke-env's ``_accept_challenges`` order, so the draws match."""
            want = opponent.lower()
            for _ in range(n):
                self.check()
                packed = self.teams.yield_team()
                while True:
                    who = await self.challenges.get()
                    self.check()
                    if "".join(ch for ch in who.lower() if ch.isalnum()) == "".join(
                            ch for ch in want if ch.isalnum()):
                        break
                await self.line(f"|/utm {packed}")
                await self.line(f"|/accept {who}")
                self.check()
                await self._await_started(3600.0)
            # the last battle must END before the half does
            if self.current is not None:
                await self.current.ended.wait()
            self.check()

        async def leave(self) -> None:
            await self.front.drop(self)

        # -- the server half: everything a websocket client would have received ------------
        async def send(self, text: str) -> None:
            try:
                await self._receive(text)
            except SeriesFailure as exc:
                self._record_failure(exc)
            except Exception as exc:                  # noqa: BLE001 - any surprise is NAMED
                self._record_failure(_fail(f"{type(exc).__name__}: {exc}"))

        async def on_obs(self, tag: str, slot: str, payload: str) -> None:
            try:
                b = self.battles.get(tag)
                if b is None:
                    raise _fail(f"a core frame for {tag}, a battle this slot is not in")
                if b.pending is not None:
                    raise _fail(f"{tag}: core frame {b.frames} arrived while frame "
                                f"{b.pending.get('n')} was never consumed by a request")
                d = json.loads(payload)
                if d.get("n") != b.frames:
                    raise _fail(f"{tag}: core frame n={d.get('n')}, expected {b.frames}")
                b.frames += 1
                b.pending = d
            except SeriesFailure as exc:
                self._record_failure(exc)

        async def _receive(self, text: str) -> None:
            if not text.startswith(">"):
                self._global(text)
                return
            head, _, body = text.partition("\n")
            tag = head[1:]
            lines = body.split("\n")
            if lines and lines[0] == "|init|battle":
                b = _BattleState(tag)
                self.battles[tag] = b
                self.current = b
                self.battle_started.set()
                return
            b = self.battles.get(tag)
            if b is None:
                return
            for ln in lines:
                await self._battle_line(b, ln)

        def _global(self, text: str) -> None:
            for ln in text.split("\n"):
                if ln.startswith("|updateuser|"):
                    name = ln.split("|")[2].strip()
                    if name == self.username:
                        self.logged_in.set()
                elif ln.startswith("|nametaken|"):
                    raise _fail(f"the front end refused our name: {ln}")
                elif ln.startswith("|popup|"):
                    raise _fail(f"the front end answered with a popup: {ln[:300]}")
                elif ln.startswith("|pm|"):
                    f = ln.split("|")
                    if len(f) >= 6 and f[4].startswith("/challenge ") and f[5] == self.fmt:
                        sender = f[2].strip()
                        if sender != self.username:
                            self.challenges.put_nowait(sender)

        async def _battle_line(self, b: _BattleState, ln: str) -> None:
            if ln.startswith("|turn|"):
                b.turn = int(ln.split("|")[2])
            elif ln.startswith("|request|"):
                payload = ln[len("|request|"):]
                if not payload:
                    return
                req = json.loads(payload)
                if req.get("wait"):
                    if b.pending is not None:
                        raise _fail(f"{b.tag}: a core frame is pending at a WAIT request")
                    return
                await self._decide(b, req)
            elif ln.startswith("|error|[Invalid choice]"):
                raise _fail(f"{b.tag}: the server refused our choice: {ln}")
            elif ln.startswith("|win|") or ln == "|tie" or ln.startswith("|tie|"):
                self._finish(b, ln)

        async def _decide(self, b: _BattleState, req: Dict[str, Any]) -> None:
            from agents.battle.core_obs import wrap_row

            d = b.pending
            if d is None:
                raise _fail(f"{b.tag} turn {b.turn}: a live |request| with no core frame")
            b.pending = None
            rqid = req.get("rqid")
            turn = int(d.get("turn") or 0)
            if turn >= self.forfeit_limit:
                # The trainer's rule (RLPlayer._handle_stall): a decision at turn >= the limit
                # forfeits instead of moving, and is not counted as a decision.
                await self.line(f"{b.tag}|/forfeit")
                return
            row = wrap_row(d["frame"])
            if np.isnan(row).any():
                bad = np.flatnonzero(np.isnan(row))[:8].tolist()
                raise _fail(f"{b.tag}: core row has unwritten (NaN) cells {bad}")
            mask = d.get("mask")
            if not isinstance(mask, list) or len(mask) != 11 or sum(mask) == 0:
                raise _fail(f"{b.tag}: core mask {mask!r} is not a live 11-bit mask")
            t0 = time.perf_counter()
            idx, argmax = self.policy.decide(row, np.asarray(mask, dtype=np.int8))
            self.state.decision_times.append(time.perf_counter() - t0)
            if mask[idx] != 1:
                raise _fail(f"{b.tag}: chose action {idx}, which the core mask {mask} forbids")
            token = (d.get("tokens") or {}).get(str(idx))
            if not token:
                raise _fail(f"{b.tag}: no choice token for legal action {idx} in {d.get('tokens')}")
            b.n_decisions += 1
            b.argmax_matches += int(idx == argmax)
            stoch = self.policy.stochastic
            if stoch not in self.state.stochastic_kwargs:
                self.state.stochastic_kwargs.append(stoch)
            suffix = f"|{rqid}" if rqid is not None else ""
            await self.line(f"{b.tag}|/choose {token}{suffix}")

        def _finish(self, b: _BattleState, ln: str) -> None:
            if b.ended.is_set():
                return
            won: Optional[bool]
            if ln.startswith("|win|"):
                won = ln[len("|win|"):].strip() == self.username
            else:
                won = None
            n = len(self.state.records) + 1
            self.state.records.append(BattleRecord(
                battle_tag=b.tag,
                turns=b.turn,
                won=won,
                finished=True,
                # a cap forfeit reads as an ordinary loss; only the turn count separates it
                hit_forfeit_limit=b.turn >= self.forfeit_limit,
                our_team=(self.state.draws[n - 1] if n <= len(self.state.draws) else None),
                n_decisions=b.n_decisions,
                n_defaults=0,
                n_redecides=0,
                t_finished=time.time(),
                our_argmax_matches=b.argmax_matches,
                our_argmax_decisions=b.n_decisions,
            ))
            self.state.last_progress = time.time()
            rec = self.state.records[-1]
            print(f"[anchors] game {n}: {rec.result} in {rec.turns} turns "
                  f"(cap={rec.hit_forfeit_limit}, team={rec.our_team})", flush=True)
            b.ended.set()

    return CoreSlot()


def build_slot(front: Any, *, username: str, model_zip: str, device: str, model_loader: str,
               stochastic: bool, temperature: float, team_spec: Dict[str, Any],
               team_seed: Optional[int], forfeit_limit: int, state: OurSideState,
               battle_format: str = "gen3ou") -> Any:
    """Everything one half's slot needs: the policy (loaded once per half, as ``play.main`` did),
    the seeded team source (its draws logged into ``state.draws``) and the connection."""
    model = load_policy(model_zip, device, model_loader, state)
    policy = CorePolicy(model, stochastic=stochastic, temperature=temperature)
    teams = build_team_source(team_spec, team_seed, state.draws)
    return make_slot(front, username=username, policy=policy, team_source=teams, state=state,
                     forfeit_limit=forfeit_limit, battle_format=battle_format)


def our_stochastic(plan: Any) -> "tuple[bool, float]":
    """``(stochastic, temperature)`` for our half — ``runner.our_argv``'s ``--temperature`` rule
    read the way ``main.play`` reads it (``> 0`` samples)."""
    t = (float(plan.our_temperature) if plan.our_temperature is not None
         else 0.0 if plan.regime == "greedy" else 1.0)
    return (t > 0.0, t if t > 0.0 else 1.0)


def row_transport(plan: Any) -> str:
    """What a row's ``our_transport`` says for this plan."""
    if plan.our_side_is_peer:
        return TRANSPORT_PEER
    if plan.our_transport == "core":
        return TRANSPORT_CORE
    return TRANSPORT_POKE_ENV_BOT if plan.our_side_is_bot else TRANSPORT_POKE_ENV
