"""The thin WEBSOCKET client over the reader session — live play on the Rust stack (poke-env retirement P4).

It replaces poke-env's ``PSClient`` / ``Player`` for OUR side: login, ``/utm``, challenge / accept, the
request → choice cycle with the server's ``rqid`` echoed, the forfeit turn limit, ``/leave``. The OBSERVATION is
never built here: every battle line goes to :class:`main.live.reader.LiveReader` (the training chain), whose frame
the policy reads. What this module owns is the FRAMING and the RULES:

* **Framing.** A server message is one write: its battle lines are fed as ONE ``FEED`` (after the declared
  non-battle lines are dropped, :data:`main.live.reader.ROOM_SKIP`), so a decision opens exactly when a message
  ends at our ``|request|`` — the sim's ``sendUpdates`` order (the battle chunk, then each side's request as its
  own ``sideupdate``), which a real server forwards message for message.
* **T28.** Any reader refusal, a frame with no legal token, a policy that returns an illegal action, or an
  ``|error|[Invalid choice]`` (the server refused the choice we sent) raises
  :class:`main.live.halt.LiveParseHalt` carrying the battle id, the offending lines and the battle's whole received
  stream. The client stops at once: no other battle continues, no next game starts. The entry point records the
  marker and exits ``FATAL_LIVE_PARSE``.
* **Respect for players (owner 2026-10-07; ladder ruling 2026-10-09).** Outgoing traffic is a DECLARED command set
  (:data:`ALLOWED_COMMANDS`): no chat, no PM. ``/search`` and ``/cancelsearch`` (:data:`LADDER_COMMANDS`) are sent
  ONLY by a client built with ``ClientConfig.ladder`` (``main.play --mode ladder``), and :meth:`LiveClient.ladder`
  re-runs ``main.ladder_guard.check_ladder_policy`` (the T28 halt, the owner's approval token in an agent session, a
  fresh drift-gate record) before EVERY search. The ladder is CONCURRENCY 1: the next ``/search`` goes out only after
  the previous battle ended, and a second live battle room is a ``ClientError``. Accept mode answers only the named
  opponent (or, on a LOCAL server, anyone).
* **No reconnect inside a battle.** A dropped socket while a battle is live ends the run with
  :class:`ConnectionLost` (a CRASH, not a T28 halt): a rejoined battle replays its log without our own choice
  notes, so the reader's tracker rows after a rejoin could not be the training rows (``ladder_readiness.md``).
"""
from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol

from main.live.halt import LiveParseHalt
from main.live.reader import Frame, LiveReader, ReaderRefusal, is_room_line, keyword, split_room_message

logger = logging.getLogger("main.live.client")

#: Every command this client may send. Anything else is a programming error (raised, never sent).
ALLOWED_COMMANDS = frozenset({"/trn", "/utm", "/challenge", "/accept", "/cancelchallenge",
                              "/choose", "/forfeit", "/leave", "/timer"})
#: The ladder queue's two commands: allowed ONLY when ``ClientConfig.ladder`` is set (``main.play --mode ladder``).
LADDER_COMMANDS = frozenset({"/search", "/cancelsearch"})

_INVALID_CHOICE = "|error|[Invalid choice]"


def to_id(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


class ClientError(RuntimeError):
    """A non-T28 client failure (login refused, a popup we cannot act on, a timeout)."""


class TeamRejected(ClientError):
    """The server rejected the team we registered (a `|popup|` naming the rejection)."""


class ConnectionLost(ClientError):
    """The socket dropped while a battle was live (no reconnect inside a battle, by design)."""


class SearchRefused(ClientError):
    """The server answered a ladder ``/search`` with a popup instead of a match (locked, no such ladder, ...)."""


class Policy(Protocol):
    def choose(self, frame: Frame) -> int: ...


@dataclass
class Decision:
    """One decision as played: the reader's frame, the index the policy chose, the token sent."""
    battle: str
    frame: Frame
    index: Optional[int]       # None = the stall forfeit
    token: Optional[str]
    t: float = field(default_factory=time.time)


@dataclass
class BattleResult:
    battle: str
    side: Optional[str]
    winner: Optional[str]      # the winner's NAME, or None on a tie / no result
    won: Optional[bool]
    turns: int
    decisions: int
    forfeited: bool


class LiveBattle:
    """One battle room: classify every line, feed the reader, decide, send."""

    def __init__(self, room: str, *, our_name: str, packed_team: Optional[str], reader: LiveReader,
                 policy: Policy, forfeit_turn_limit: int, send: Callable[[str], None],
                 on_decision: Optional[Callable[[Decision], None]] = None,
                 on_message: Optional[Callable[[str, List[str]], None]] = None) -> None:
        self.room = room
        self.our_id = to_id(our_name)
        self.packed_team = packed_team
        self.reader = reader
        self.policy = policy
        self.forfeit_turn_limit = int(forfeit_turn_limit)
        self.send = send
        self.on_decision = on_decision
        self.on_message = on_message
        self.stream: List[str] = []        # every line received in this room, skipped ones included
        self.pending: List[str] = []       # battle lines received before our side was known
        self.side: Optional[str] = None
        self.display_name: Optional[str] = None
        self.opened = False
        self.ended = False
        self.winner: Optional[str] = None
        self.decisions = 0
        self.forfeited = False
        self.last_turn = 0

    def halt(self, reason: str, offending: List[str]) -> LiveParseHalt:
        return LiveParseHalt(reason, battle_id=self.room, offending_lines=offending, stream=self.stream)

    def on_lines(self, lines: List[str]) -> None:
        """One server message for this room."""
        self.stream.extend(lines)
        if self.on_message is not None:
            self.on_message(self.room, lines)
        if self.ended:
            return  # after |win| / |tie the battle is over: nothing more is read (or played)
        battle: List[str] = []
        for ln in lines:
            if ln == "" or is_room_line(ln):
                # "" is FRAMING, not protocol: a real server's message may end with a newline (measured on
                # Showdown master 51ad80fa); the bare `|` line the sim emits is "|" and is fed.
                continue
            if ln.startswith(_INVALID_CHOICE):
                raise self.halt("the server REFUSED the choice we sent ([Invalid choice]) — the reader's "
                                "legality disagrees with the server's", [ln])
            if self.side is None and keyword(ln) == "player":
                parts = ln.split("|")
                if len(parts) >= 4 and to_id(parts[3]) == self.our_id and parts[2] in ("p1", "p2"):
                    self.side, self.display_name = parts[2], parts[3]
            battle.append(ln)
            if keyword(ln) in ("win", "tie"):
                self.ended = True
                self.winner = ln.split("|", 2)[2] if keyword(ln) == "win" else None
                break
        if not battle:
            return
        if self.side is None:
            self.pending.extend(battle)
            return
        if not self.opened:
            try:
                self.reader.open(self.side, self.display_name or "", self.packed_team)
            except ReaderRefusal as exc:
                raise self.halt(f"the reader refused to open ({exc.kind}): {exc}", battle) from exc
            self.opened = True
            battle, self.pending = self.pending + battle, []
        try:
            frame = self.reader.feed(battle)
        except ReaderRefusal as exc:
            raise self.halt(f"the reader refused the input ({exc.kind}): {exc}", battle) from exc
        self.last_turn = self.reader.turn
        if frame is not None:
            self.decide(frame, battle)
        if self.ended:
            self.send(f"|/leave {self.room}")

    def decide(self, frame: Frame, write: List[str]) -> None:
        if frame.turn >= self.forfeit_turn_limit:
            # The trainer's stall forfeit (`StallConfig.threshold`, the Rust env core's `stall_forfeit_due`):
            # at a decision whose turn has reached the limit we forfeit instead of acting.
            self.forfeited = True
            self.send(f"{self.room}|/forfeit")
            if self.on_decision is not None:
                self.on_decision(Decision(self.room, frame, None, None))
            return
        if not frame.tokens or int(frame.mask.sum()) == 0:
            raise self.halt("a decision with no legal choice token (the reader's legality is empty)", write)
        try:
            idx = int(self.policy.choose(frame))
        except Exception as exc:  # noqa: BLE001 — a policy that cannot choose is a missing choice (T28)
            raise self.halt(f"the policy raised at a decision: {type(exc).__name__}: {exc}", write) from exc
        token = frame.tokens.get(idx)
        if token is None or not (0 <= idx < len(frame.mask)) or frame.mask[idx] != 1:
            raise self.halt(f"the policy chose index {idx}, which is not legal (mask {frame.mask.tolist()})", write)
        try:
            self.reader.choose(token)
        except ReaderRefusal as exc:
            raise self.halt(f"the reader refused the choice ({exc.kind}): {exc}", write) from exc
        suffix = f"|{frame.rqid}" if frame.rqid is not None else ""
        self.send(f"{self.room}|/choose {token}{suffix}")
        self.decisions += 1
        if self.on_decision is not None:
            self.on_decision(Decision(self.room, frame, idx, token))

    def result(self) -> BattleResult:
        won = None if self.winner is None else (to_id(self.winner) == self.our_id)
        return BattleResult(self.room, self.side, self.winner, won, self.last_turn, self.decisions, self.forfeited)


@dataclass
class ClientConfig:
    uri: str
    username: str
    battle_format: str = "gen3ou"
    password: Optional[str] = None
    #: "local" — `/trn NAME,0,` with no assertion (a local server started with `--no-security`, or our front end);
    #: "official" — an assertion from the login server (GATED: never run in P4, `main.play` refuses it).
    auth: str = "local"
    login_url: str = "https://play.pokemonshowdown.com/api/login"
    proxy: Optional[str] = None
    connect_timeout_s: float = 30.0
    forfeit_turn_limit: int = 250
    #: a battle that has received NO message for this long is a stall → ClientError (never a T28 halt)
    battle_idle_timeout_s: float = 600.0
    #: ``main.play --mode ladder``: lets this client queue (`/search`) and play ONE battle at a time
    #: (:meth:`LiveClient.ladder`). Set only for a session ``main.ladder_guard.check_ladder_policy`` cleared.
    ladder: bool = False
    #: how long one ladder ``/search`` waits for a match before it is cancelled
    ladder_search_timeout_s: float = 900.0


class LiveClient:
    """One account on one server. Battles run ONE at a time (each with its own reader process)."""

    def __init__(self, cfg: ClientConfig, *, policy: Policy, team_fn: Callable[[], Optional[str]],
                 on_decision: Optional[Callable[[Decision], None]] = None,
                 on_message: Optional[Callable[[str, List[str]], None]] = None,
                 reader_factory: Callable[[], LiveReader] = LiveReader) -> None:
        self.cfg = cfg
        self.policy = policy
        self.team_fn = team_fn
        self.on_decision = on_decision
        self.on_message = on_message
        self.reader_factory = reader_factory
        self.ws: Any = None
        self.name: Optional[str] = None
        self._challstr: Optional[str] = None
        self._logged_in = asyncio.Event()
        self._login_error: Optional[str] = None
        self._battles: Dict[str, LiveBattle] = {}
        self._battle_started: "asyncio.Queue[str]" = asyncio.Queue()
        self._battle_done: "asyncio.Queue[BattleResult]" = asyncio.Queue()
        self._challenges: "asyncio.Queue[str]" = asyncio.Queue()
        self._popups: List[str] = []
        self._searching = False  # a ladder /search is outstanding: a popup then is the server's refusal of it
        self._fatal: Optional[BaseException] = None
        self._recv_task: Optional[asyncio.Task] = None
        self._write_task: Optional[asyncio.Task] = None
        self._outq: "asyncio.Queue[str]" = asyncio.Queue()
        self._reader: Optional[LiveReader] = None
        self.results: List[BattleResult] = []
        self._last_msg = time.monotonic()

    # -- the socket ---------------------------------------------------------------------------
    async def connect(self) -> None:
        import websockets
        kw: Dict[str, Any] = {"max_size": None, "ping_interval": None}
        if self.cfg.proxy:
            kw["proxy"] = self.cfg.proxy
        self.ws = await asyncio.wait_for(websockets.connect(self.cfg.uri, **kw), self.cfg.connect_timeout_s)
        self._recv_task = asyncio.ensure_future(self._recv_loop())
        self._write_task = asyncio.ensure_future(self._writer())
        try:
            await asyncio.wait_for(self._logged_in.wait(), self.cfg.connect_timeout_s)
        except asyncio.TimeoutError as exc:
            raise ClientError(f"login as {self.cfg.username!r} on {self.cfg.uri} did not complete in "
                              f"{self.cfg.connect_timeout_s:g}s ({self._login_error or 'no answer'})") from exc
        self._raise_fatal()
        if self._login_error:
            raise ClientError(self._login_error)

    async def close(self) -> None:
        # let queued lines (a final /leave) reach the socket before it closes
        for _ in range(50):
            if self._outq.empty():
                break
            await asyncio.sleep(0.01)
        for task in (self._recv_task, self._write_task):
            if task is None:
                continue
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
        if self.ws is not None:
            try:
                await self.ws.close()
            except Exception:  # noqa: BLE001
                pass
        if self._reader is not None:
            self._reader.close()
            self._reader = None

    def _raise_fatal(self) -> None:
        if self._fatal is not None:
            raise self._fatal

    def send_raw(self, text: str) -> None:
        """Send one client line (`<room>|<command>`), refusing any command outside the declared set."""
        body = text.split("|", 1)[1] if "|" in text else text
        cmd = body.split(" ", 1)[0]
        if cmd not in ALLOWED_COMMANDS and not (self.cfg.ladder and cmd in LADDER_COMMANDS):
            raise ClientError(f"refusing to send {cmd!r}: not in the declared command set (no chat; the ladder queue "
                              "only for `main.play --mode ladder`)")
        self._outq.put_nowait(text)

    async def _writer(self) -> None:
        """The ONE task that writes the socket, in the order the lines were queued."""
        while True:
            text = await self._outq.get()
            try:
                await self.ws.send(text)
            except Exception as exc:  # noqa: BLE001
                self._set_fatal(ConnectionLost(f"send failed: {exc}"))
                return

    def _set_fatal(self, exc: BaseException) -> None:
        if self._fatal is None:
            self._fatal = exc
        for q in (self._battle_started, self._battle_done, self._challenges):
            q.put_nowait(None)  # type: ignore[arg-type]  # wake every waiter; it re-raises the fatal
        self._logged_in.set()

    async def _recv_loop(self) -> None:
        try:
            async for text in self.ws:
                self._last_msg = time.monotonic()
                try:
                    self._on_text(text if isinstance(text, str) else text.decode("utf-8"))
                except LiveParseHalt as exc:
                    self._set_fatal(exc)
                    return
                except Exception as exc:  # noqa: BLE001 — a client bug is a fatal, never swallowed
                    self._set_fatal(exc)
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            self._set_fatal(ConnectionLost(f"the socket failed: {exc}"))
            return
        if self._fatal is None:
            self._set_fatal(ConnectionLost("the server closed the socket"))

    # -- dispatch -----------------------------------------------------------------------------
    def _on_text(self, text: str) -> None:
        room, lines = split_room_message(text)
        if room and room.startswith("battle-"):
            self._on_battle(room, lines)
            return
        for ln in lines:
            self._on_global(ln)

    def _on_global(self, ln: str) -> None:
        kw = keyword(ln)
        parts = ln.split("|")
        if kw == "challstr":
            self._challstr = ln[len("|challstr|"):]
            asyncio.ensure_future(self._login())
        elif kw == "updateuser":
            name = parts[2].strip() if len(parts) > 2 else ""
            named = parts[3] if len(parts) > 3 else "0"
            if named == "1" and to_id(name) == to_id(self.cfg.username):
                self.name = name
                self._logged_in.set()
        elif kw == "nametaken":
            self._login_error = f"login refused: {ln}"
            self._logged_in.set()
        elif kw == "popup":
            self._popups.append(ln)
            logger.warning("[live] popup: %s", ln)
            if "rejected" in ln.lower() and "team" in ln.lower():
                self._battle_started.put_nowait(TeamRejected(ln))  # type: ignore[arg-type]
            elif self._searching:
                self._battle_started.put_nowait(SearchRefused(ln))  # type: ignore[arg-type]
        elif kw == "pm" and len(parts) >= 5 and parts[4].startswith("/challenge "):
            self._challenges.put_nowait(parts[2].strip())
        elif kw == "updatechallenges" and len(parts) >= 3:
            try:
                frm = json.loads(parts[2]).get("challengesFrom") or {}
            except ValueError:
                frm = {}
            for user, fmt in frm.items():
                if fmt == self.cfg.battle_format:
                    self._challenges.put_nowait(user)
        # every other global message (formats, customgroups, queryresponse, updatesearch, chat, …) is not
        # battle protocol and is skipped; nothing global is ever fed to the reader.

    def _on_battle(self, room: str, lines: List[str]) -> None:
        b = self._battles.get(room)
        if b is None:
            if not lines or lines[0] != "|init|battle":
                return  # a message for a battle we are not in (e.g. after /leave)
            if self.cfg.ladder and any(not x.ended for x in self._battles.values()):
                from main.ladder_guard import LADDER_CONCURRENCY
                raise ClientError(f"ladder concurrency is {LADDER_CONCURRENCY} (owner 2026-10-09): {room} opened "
                                  "while another battle is live (is this account also playing elsewhere?) — "
                                  "stopping rather than playing two games at once")
            if self._reader is None:
                self._reader = self.reader_factory()
            b = LiveBattle(room, our_name=self.name or self.cfg.username, packed_team=self._team,
                           reader=self._reader, policy=self.policy,
                           forfeit_turn_limit=self.cfg.forfeit_turn_limit, send=self.send_raw,
                           on_decision=self.on_decision, on_message=self.on_message)
            self._battles[room] = b
            self._battle_started.put_nowait(room)
        was_ended = b.ended
        b.on_lines(lines)
        if b.ended and not was_ended:
            res = b.result()
            self.results.append(res)
            self._battle_done.put_nowait(res)

    async def _login(self) -> None:
        try:
            if self.cfg.auth == "local":
                assertion = ""
            elif self.cfg.auth == "official":
                assertion = await asyncio.get_running_loop().run_in_executor(None, self._official_assertion)
            else:
                raise ClientError(f"unknown auth mode {self.cfg.auth!r}")
            self._outq.put_nowait(f"|/trn {self.cfg.username},0,{assertion}")
        except Exception as exc:  # noqa: BLE001
            self._login_error = f"login failed: {exc}"
            self._logged_in.set()

    def _official_assertion(self) -> str:
        """The login server's assertion for (username, password, challstr). GATED: only reachable through
        ``main.play --server official --public-acceptance``, which P4 never ran (owner 2026-10-07)."""
        import urllib.parse
        import urllib.request
        if not self.cfg.password:
            raise ClientError("--server official needs a password ($PS_PASSWORD)")
        data = urllib.parse.urlencode({"name": self.cfg.username, "pass": self.cfg.password,
                                       "challstr": self._challstr}).encode()
        req = urllib.request.Request(self.cfg.login_url, data=data, method="POST")
        if self.cfg.proxy:
            raise ClientError("the official login through a proxy needs a SOCKS-capable HTTP client; "
                              "not built in P4 (the public acceptance is a separate, gated unit)")
        with urllib.request.urlopen(req, timeout=self.cfg.connect_timeout_s) as resp:  # noqa: S310
            body = resp.read().decode()
        obj = json.loads(body[1:] if body.startswith("]") else body)
        assertion = obj.get("assertion")
        if not assertion or assertion.startswith(";"):
            raise ClientError(f"the login server refused: {str(obj)[:200]}")
        return assertion

    # -- the battle loop ----------------------------------------------------------------------
    @property
    def _team(self) -> Optional[str]:
        return getattr(self, "_current_team", None)

    async def _next(self, q: "asyncio.Queue", what: str, timeout: float) -> Any:
        try:
            item = await asyncio.wait_for(q.get(), timeout)
        except asyncio.TimeoutError as exc:
            self._raise_fatal()
            raise ClientError(f"timed out after {timeout:g}s waiting for {what}") from exc
        self._raise_fatal()
        return item

    async def _await_battle_end(self, room: str) -> BattleResult:
        while True:
            idle = self.cfg.battle_idle_timeout_s - (time.monotonic() - self._last_msg)
            try:
                res = await self._next(self._battle_done, f"the end of {room}", max(idle, 1.0))
            except ClientError:
                if time.monotonic() - self._last_msg < self.cfg.battle_idle_timeout_s:
                    continue
                raise
            if res.battle == room:
                return res

    async def _set_team(self) -> None:
        self._current_team = self.team_fn()
        self.send_raw(f"|/utm {self._current_team if self._current_team is not None else 'null'}")

    async def challenge(self, opponent: str, n: int, accept_timeout_s: float = 120.0) -> List[BattleResult]:
        out = []
        for _ in range(n):
            await self._set_team()
            self.send_raw(f"|/challenge {opponent}, {self.cfg.battle_format}")
            room = await self._next(self._battle_started, f"{opponent} to accept", accept_timeout_s)
            if isinstance(room, TeamRejected):
                raise room
            out.append(await self._await_battle_end(room))
        return out

    async def ladder(self, n: int) -> List[BattleResult]:
        """Play ``n`` ladder games, ONE at a time: `/utm`, `/search <format>`, wait for the match, play it out, repeat.

        Concurrency 1 is structural (the next search is queued only after the previous battle ended). Before EVERY
        search ``main.ladder_guard.check_ladder_policy`` runs again, so a lapsed approval token, a stale drift record
        or a halt marker stops the session at the next game boundary, and a caller that skipped ``main.play`` is
        guarded here too."""
        if not self.cfg.ladder:
            raise ClientError("LiveClient.ladder() needs ClientConfig.ladder=True (main.play --mode ladder)")
        from main.ladder_guard import check_ladder_policy
        out: List[BattleResult] = []
        for _ in range(n):
            check_ladder_policy(battle_format=self.cfg.battle_format)
            await self._set_team()
            self._searching = True
            try:
                self.send_raw(f"|/search {self.cfg.battle_format}")
                try:
                    room = await self._next(self._battle_started, "a ladder match", self.cfg.ladder_search_timeout_s)
                except ClientError:
                    self.send_raw("|/cancelsearch")  # never leave a queued search behind
                    raise
            finally:
                self._searching = False
            if isinstance(room, ClientError):  # TeamRejected / SearchRefused
                raise room
            out.append(await self._await_battle_end(room))
        return out

    async def accept(self, opponent: Optional[str], n: int, wait_timeout_s: float = 3600.0) -> List[BattleResult]:
        out = []
        while len(out) < n:
            user = await self._next(self._challenges, "a challenge", wait_timeout_s)
            if opponent is not None and to_id(user) != to_id(opponent):
                # IGNORED, not rejected: zero contact with anyone we were not told to play (owner 2026-10-07).
                logger.warning("[live] ignoring a challenge from %r (not the named opponent)", user)
                continue
            await self._set_team()
            self.send_raw(f"|/accept {user}")
            room = await self._next(self._battle_started, f"the battle with {user}", 120.0)
            if isinstance(room, TeamRejected):
                raise room
            out.append(await self._await_battle_end(room))
        return out
