"""A websocket front end over the Rust bridge — a Showdown *server* for third-party clients.

WHY THIS EXISTS. Every external gen3ou opponent we have measured (Metamon, Foul Play) is a
websocket CLIENT: it speaks the Showdown client protocol and nothing else. Playing one has so far
meant `npm run showdown -- 9XXX`, i.e. a Node server, because our serverless transport
(`bridge_session` / `local_battle_runner`) is a *library* seam — it assigns a `BattleStreamClient`
onto a poke-env `Player` **in this process**. That seam is closed to a third party for a reason
the metamon de-risk recorded as verdict (d): their player subclasses UPSTREAM poke-env, ours
subclasses the VENDORED fork, and one process resolves `import poke_env` to exactly one of them.

So the integration point cannot be an import. It has to be a SOCKET, and the opponent has to stay
in its own process with its own poke-env. This module is that socket: an asyncio websocket server
that speaks just enough of the Showdown client protocol for a poke-env-style client to log in,
challenge or accept, and play complete battles — with each battle backed by ONE `sim_bridge`
child (the `local_sim_bridge` stdin/stdout protocol, `--use-bridge rust` semantics) instead of a
Showdown server.

    python -m utils.bridge.ws_frontend --port 9601          # ws://127.0.0.1:9601/showdown/websocket

WHAT IT IS NOT. It is not a Showdown server and must never be mistaken for one — no ladder, no
rooms, no chat, no replays, no matchmaking, no battle timer. The full deferral list is
`designs/rust_sim/ws_frontend.md`, which OWNS the protocol surface; keep it in step with this file.

🚨 **THE `rqid` IS THE DECISION TRIGGER, AND THE BRIDGE DOES NOT EMIT IT.** A `|request|` is what
makes a client move (`fp/battle/protocol.py:2279`; poke-env's `_handle_battle_request`), and every
choice Foul Play sends is `"<tag>|/choose move X|<rqid>"`. The sim's own `|request|` JSON carries
NO `rqid` — measured, not assumed: `Side.emitRequest` sends the bare object and the SERVER injects
the id (`server/room-battle.ts:796-801`, `this.rqid++` → `request.rqid = this.rqid` → re-stringify,
ONE counter shared by both slots). So this front end injects it, exactly there, or a Foul Play
client never moves. The injection is a string SPLICE (`,"rqid":N` before the closing brace) rather
than a JSON round trip, so the per-side text stays byte-identical to the bridge's apart from that
one appended key — which is what makes the byte-differential gate against the Node bridge legible.

FRAMING. The bridge emits per-side protocol CHUNKS; a websocket client expects them inside a
`>battle-…` room. The room header, `|init|battle` and the `|title|` line are fabricated here (the
same fabrication `local_battle_runner._frame` does for the in-process path) and everything after
is relayed verbatim. Chunk boundaries are preserved as frame boundaries because the clients read
them: poke-env creates the battle on a frame whose SECOND line is `|init|`, and Foul Play derives
its own `p1`/`p2` from the frame carrying the opponent's `|player|` line — which the sim already
emits as its own chunk.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import itertools
import json
import logging
import os
import re
import secrets
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from utils.bridge.seed_spec import derive_seed_from_base, validate_seed_spec
from utils.bridge.sim_bridge_bin import bridge_spawn_argv
from utils.bridge.team_validator import validate_teams_locally

logger = logging.getLogger("ws_frontend")

# The asyncio StreamReader limit for a bridge child's stdout. The TWIN is
# `local_battle_runner.BRIDGE_STREAM_LIMIT`, which carries the full reasoning (a long battle's
# `__RECON__` line exceeds the 64 KiB default and readline then raises). It is restated rather
# than imported because that module imports poke-env and THIS one must not: the whole point of a
# separate-process front end is that the opponent owns `import poke_env`, so the server half has
# to be poke-env-free. `ws_frontend_test.py` asserts the two constants are equal.
BRIDGE_STREAM_LIMIT = 16 * 1024 * 1024

# Ports this process must NEVER bind. 8001 carries the live training run and 8000 is the shared
# dev server; the project rule is that an agent's own server lives in the 9XXX range.
RESERVED_PORTS = {8000: "the shared DEV server", 8001: "the live TRAINING server"}

# Showdown refuses a userid longer than this (`server/users.ts:745`) — and the refusal is a known
# HANG for a client that does not read `|nametaken|` (foul-play de-risk hazard 1). Reproduced here
# with the server's own message AND a loud server-side log, so it fails visibly either way.
MAX_USERID_LEN = 18

# Room ids are handed out process-globally, exactly as a real server does. Never reuse a tag: a
# poke-env `Player` caches its battle objects by tag and hands back the PREVIOUS battle for a
# repeat, whose team then overflows on the next switch.
_BATTLE_SEQ = itertools.count(1)

_ID_RE = re.compile(r"[^a-z0-9]+")

# The two refusal shapes `Side.emitChoiceError` can emit. The server treats `[Invalid choice]` as
# "the choice was dropped, you may choose again" and leaves `[Unavailable choice]` alone (the sim
# follows that one with a fresh `|request|`); mirrored in `_note_side_line`.
_INVALID_CHOICE_PREFIX = "|error|[Invalid choice]"


def to_id(text: str) -> str:
    """Showdown's ``toID``: lowercase, then drop everything that is not ``[a-z0-9]``."""
    return _ID_RE.sub("", text.lower())


# --------------------------------------------------------------------------------------------
# connection + battle state
# --------------------------------------------------------------------------------------------
@dataclass
class _SideRequest:
    """The server's per-slot request record (`server/room-battle.ts:29`, `:124`).

    ``is_wait`` is Showdown's tri-state: ``'cantUndo'`` (nothing to choose — the initial value and
    a `wait` request), ``False`` (a live request, choose now), ``True`` (chosen, waiting for the
    other side). Reproducing it is what lets a stale `rqid` be refused the way the server refuses
    it, rather than silently re-running a decision the turn has moved past.
    """

    rqid: int = 0
    is_wait: object = "cantUndo"
    choice: str = ""


@dataclass
class _Conn:
    """One websocket connection = one (eventually named) user."""

    ws: object
    guest_n: int
    name: str = ""
    userid: str = ""
    team: Optional[str] = None
    avatar: str = "1"

    @property
    def named(self) -> bool:
        return bool(self.userid)

    async def send(self, text: str) -> None:
        try:
            await self.ws.send(text)
        except Exception as exc:  # noqa: BLE001 — a dropped peer is not this server's error
            logger.debug("send to %s failed: %s", self.userid or "guest", exc)


@dataclass
class _Battle:
    """One battle = one bridge child + the two connections it feeds."""

    tag: str
    fmt: str
    proc: object
    conns: Dict[str, _Conn]
    names: Dict[str, str]
    seed: Optional[List[int]]
    rqid: int = 0
    req: Dict[str, _SideRequest] = field(default_factory=lambda: {"p1": _SideRequest(),
                                                                 "p2": _SideRequest()})
    finished: bool = False
    reader: object = None
    # The byte-differential gate's material, and nothing else reads it: `capture` is every
    # per-side chunk this battle RELAYED (slot, text) and `commands` is every stdin line it wrote
    # to the child (the `START` json first, then each `CHOOSE`/`FORCELOSE` in order). Replaying
    # `commands` into a Node `local_sim_bridge.js` regenerates the battle, so the two together
    # are a complete, self-contained repro. Both are off unless the server was built with
    # `capture=True`, because a long series would otherwise grow without bound.
    capture: Optional[List[Tuple[str, str]]] = None
    commands: Optional[List[str]] = None

    def slot_of(self, conn: _Conn) -> Optional[str]:
        for slot, c in self.conns.items():
            if c is conn:
                return slot
        return None


# --------------------------------------------------------------------------------------------
# the server
# --------------------------------------------------------------------------------------------
class ShowdownFrontEnd:
    """The websocket server. One instance owns every connection, challenge and battle."""

    def __init__(self, *, impl: str = "rust", battle_format: str = "gen3ou",
                 seed_base: Optional[int] = None, validate_teams: bool = True,
                 capture: bool = False, capture_dir: Optional[str] = None):
        self._spawn_argv = bridge_spawn_argv(impl)
        self.impl = impl
        self.battle_format = battle_format
        self.seed_base = seed_base
        self._validate = validate_teams
        # `capture_dir` implies `capture`: a directory is a request for the artifact, and the
        # artifact cannot be written without the in-memory record it is made of.
        self._capture_dir = capture_dir
        self._capture = capture or capture_dir is not None
        self.users: Dict[str, _Conn] = {}
        self.battles: Dict[str, _Battle] = {}
        # (from_userid, to_userid) -> (format, packed team snapshot), mirroring the server's
        # `Challenge` record: the challenger's team is FIXED at challenge time (`challenge.ready`),
        # the acceptor's is read at `/accept`.
        self.challenges: Dict[Tuple[str, str], Tuple[str, Optional[str]]] = {}
        self._guests = itertools.count(1)
        self._battle_index = itertools.count(0)
        self._team_cache: Dict[Tuple[str, str], Tuple[bool, List[str]]] = {}
        # Every battle that has FINISHED, in order — but only while `capture` is on, because a
        # long series would otherwise hold every chunk of every battle forever. `self.battles`
        # cannot serve this: a client's `/leave` retires the room (which is correct hygiene), so
        # by the time a harness looks the record is gone.
        self.captured_battles: List[_Battle] = []
        # The live pump tasks, keyed by nothing — `self.battles` cannot serve this, because a
        # client's `/leave` retires the room while the pump is still finishing its teardown.
        # `drain()` is what a harness awaits before reading `captured_battles`.
        self._pumps: set = set()
        # Every battle whose PUMP is still running. Deliberately separate from `self.battles`,
        # which is the ROOM registry a client's `/leave` retires — a battle can be out of that
        # dict while its child is still being read and reaped, and `close()` must still reach it.
        # (A child left unreaped has its pipes closed from `__del__` instead, which runs after the
        # loop is gone and raises `RuntimeError: Event loop is closed` out of a destructor.)
        self._live: Dict[str, _Battle] = {}
        self.n_battles_finished = 0

    # -- connection lifecycle ------------------------------------------------------------
    async def handler(self, ws) -> None:
        conn = _Conn(ws=ws, guest_n=next(self._guests))
        # The real server greets with `|updateuser| Guest N` BEFORE `|challstr|`. poke-env's
        # vendored fork documents the ordering explicitly (a Guest identity before `/trn` must
        # not be read as a successful login), so reproduce it rather than skipping to challstr.
        await conn.send(f"|updateuser| Guest {conn.guest_n}|0|1|{{}}")
        await conn.send(f"|challstr|4|{secrets.token_hex(32)}")
        try:
            async for raw in ws:
                for line in str(raw).split("\n"):
                    if line:
                        await self._client_line(conn, line)
        except Exception as exc:  # noqa: BLE001 — one peer dying must not kill the server
            logger.info("connection %s closed: %s", conn.userid or "guest", exc)
        finally:
            await self._drop(conn)

    async def _drop(self, conn: _Conn) -> None:
        """A client went away: forfeit its live battles, then unregister it."""
        for battle in list(self.battles.values()):
            slot = battle.slot_of(conn)
            if slot is not None and not battle.finished:
                logger.warning("%s disconnected mid-battle %s — forfeiting %s",
                               conn.userid, battle.tag, slot)
                await self._write_child(battle, f"FORCELOSE {slot}")
        for key in [k for k in self.challenges if conn.userid in k]:
            self.challenges.pop(key, None)
        if conn.userid and self.users.get(conn.userid) is conn:
            self.users.pop(conn.userid, None)

    # -- inbound protocol ----------------------------------------------------------------
    async def _client_line(self, conn: _Conn, line: str) -> None:
        room, _, rest = line.partition("|")
        if not rest:
            return
        # `/utm`'s payload is a PACKED TEAM, which is itself full of `|` — so it is read off the
        # raw remainder, never off a `split("|")`.
        if rest.startswith("/utm "):
            await self._cmd_utm(conn, rest[len("/utm "):])
            return
        fields = rest.split("|")
        cmd = fields[0]
        verb, _, arg = cmd.partition(" ")
        arg = arg.strip()

        if verb == "/trn":
            await self._cmd_trn(conn, arg)
        elif verb == "/utm":            # `/utm null`
            await self._cmd_utm(conn, arg)
        elif verb == "/challenge":
            await self._cmd_challenge(conn, arg)
        elif verb == "/accept":
            await self._cmd_accept(conn, arg)
        elif verb in ("/reject", "/cancelchallenge"):
            await self._cmd_cancel(conn, arg)
        elif verb == "/search":
            await conn.send(
                "|popup|This is the gen3ai bridge front end, not a Showdown ladder: "
                "`/search` is not implemented. Use `/challenge <user>, <format>` and "
                "`/accept <user>`.")
        elif verb in ("/choose", "/move", "/switch", "/team"):
            choice = cmd[1:] if verb != "/choose" else arg
            await self._cmd_choose(conn, room, choice, fields[1] if len(fields) > 1 else "")
        elif verb == "/forfeit":
            await self._cmd_forfeit(conn, room)
        elif verb == "/leave":
            await self._cmd_leave(conn, arg or room)
        elif verb == "/cmd":
            await self._cmd_query(conn, arg)
        elif verb == "/avatar":
            conn.avatar = arg or conn.avatar
        elif verb in ("/timer", "/undo", "/join", "/savereplay", "/noreply", "/cancelsearch"):
            # Battle TIMER: deliberately none — the trainer's forfeit is client-side
            # (`StallConfig.threshold`, 250 turns), so a server-side clock would only add a
            # second, unrecorded way to lose. `/timer on` is accepted and does nothing.
            pass
        else:
            logger.debug("ignoring client line from %s: %r", conn.userid or "guest", line[:120])

    # -- login ---------------------------------------------------------------------------
    async def _cmd_trn(self, conn: _Conn, arg: str) -> None:
        """`/trn NAME,0,ASSERTION` — a no-op login that accepts ANY assertion.

        There is no auth here by design: the front end is a local test harness, and both clients
        we care about compute their assertion against the real login server (poke-env only when a
        password is set; Foul Play always). The ONE refusal reproduced is the length limit,
        because that one is a known silent hang rather than an error.
        """
        name = arg.split(",")[0].strip()
        userid = to_id(name)
        if not userid:
            await conn.send("|nametaken||Your name must contain at least one letter.")
            return
        if len(userid) > MAX_USERID_LEN:
            # 🚨 foul-play de-risk hazard 1: a >18-char name is refused, the client is NOT logged
            # in, and a client that ignores `|nametaken|` (Foul Play does) then waits forever.
            # Send the server's own message AND say so loudly here, so the hang has a cause in
            # the log instead of being a silent stall.
            logger.error(
                "REFUSED login for %r: userid is %d characters (>%d). The real server answers "
                "`|nametaken||Your name must be 18 characters or shorter.` and does not log the "
                "user in; a client that ignores it (e.g. Foul Play) will now HANG. Shorten the "
                "username.", name, len(userid), MAX_USERID_LEN)
            await conn.send("|nametaken||Your name must be 18 characters or shorter.")
            await conn.send(f"|popup|Your name must be {MAX_USERID_LEN} characters or shorter.")
            return
        other = self.users.get(userid)
        if other is not None and other is not conn:
            await conn.send(f'|nametaken|{name}|Someone is already using the name "{other.name}".')
            logger.error("REFUSED login for %r: that userid is already connected.", name)
            return
        conn.name, conn.userid = name, userid
        self.users[userid] = conn
        await conn.send(f"|updateuser| {name}|1|{conn.avatar}|"
                        '{"blockChallenges":false,"blockPMs":false}')
        logger.info("login: %s (%s)", name, userid)

    async def _cmd_query(self, conn: _Conn, arg: str) -> None:
        """`/cmd userdetails <user>` — Foul Play's `avatar()` blocks on the answer."""
        verb, _, who = arg.partition(" ")
        if verb != "userdetails":
            return
        target = self.users.get(to_id(who.strip()))
        payload = {"id": to_id(who.strip()), "userid": to_id(who.strip()),
                   "name": target.name if target else who.strip(),
                   "avatar": target.avatar if target else "1", "group": " "}
        await conn.send("|queryresponse|userdetails|" + json.dumps(payload, separators=(",", ":")))

    # -- teams ---------------------------------------------------------------------------
    async def _cmd_utm(self, conn: _Conn, packed: str) -> None:
        packed = packed.strip()
        if not packed or packed == "null":
            conn.team = None
            return
        ok, errors = await self._validate_team(self.battle_format, packed)
        if not ok:
            # The server's own shape (metamon de-risk hazard 1 captured it verbatim): a popup,
            # and the team does NOT take effect — so the next `/challenge` or `/accept` is
            # refused rather than starting a battle with a team the sim would reject.
            conn.team = None
            await conn.send("|popup|Your team was rejected for the following reasons:||||"
                            + "||".join(f"- {e}" for e in errors))
            logger.error("team from %s REJECTED: %s", conn.userid, "; ".join(errors))
            return
        conn.team = packed

    async def _validate_team(self, fmt: str, packed: str) -> Tuple[bool, List[str]]:
        """Validate a packed team through the bridge validator (`validate_team.js`).

        `Teams.import` accepts the PACKED form as well as an export paste, so the existing
        validator needs no change. Cached per (format, packed string) because the validator is a
        Node subprocess and a series re-registers the same handful of teams every battle.
        """
        if not self._validate:
            return True, []
        key = (fmt, packed)
        cached = self._team_cache.get(key)
        if cached is None:
            result = (await asyncio.to_thread(validate_teams_locally, fmt, [packed]))[0]
            cached = (bool(result.get("valid")), list(result.get("errors") or []))
            self._team_cache[key] = cached
        return cached

    # -- challenges ----------------------------------------------------------------------
    async def _cmd_challenge(self, conn: _Conn, arg: str) -> None:
        """`/challenge <user>, <format>` — poke-env sends `", "`, Foul Play sends `","`."""
        if not conn.named:
            await conn.send("|popup|You must choose a name before challenging.")
            return
        target_raw, _, fmt = arg.partition(",")
        fmt = to_id(fmt) or self.battle_format
        target = self.users.get(to_id(target_raw))
        if target is None:
            # The real server drops a challenge aimed at a user who is not online and says so;
            # the challenger then waits forever. Keep both halves — the popup AND a loud log.
            await conn.send(f"|popup|The user '{target_raw.strip()}' was not found.")
            logger.error("challenge from %s to %r DROPPED: that user is not connected. The "
                         "challenger will now wait forever — start the acceptor first.",
                         conn.userid, target_raw.strip())
            return
        if conn.team is None:
            await conn.send("|popup|You need to select a team before challenging "
                            "(`/utm <packed team>`).")
            return
        self.challenges[(conn.userid, target.userid)] = (fmt, conn.team)
        pm = self._challenge_pm(conn.name, target.name, fmt)
        await conn.send(pm)
        await target.send(pm)
        logger.info("challenge %s -> %s (%s)", conn.userid, target.userid, fmt)

    @staticmethod
    def _challenge_pm(from_name: str, to_name: str, fmt: str) -> str:
        """The server's challenge PM (`server/ladders-challenges.ts:217-235`).

        `|pm|<fromIdentity>|<toIdentity>|/challenge <format>|<teambuilderFormat>|<message>|
        <acceptButton>|<rejectButton>` — an identity is `<groupchar><name>`, ' ' for a regular
        user. Both consumers read the SAME two fields: field 4 must start with `/challenge` and
        field 5 must equal the format (poke-env `player.py::_handle_challenge_request`; Foul Play
        `websocket_client.py::accept_challenge`, which additionally requires EXACTLY 9 fields).
        """
        return f"|pm| {from_name}| {to_name}|/challenge {fmt}|{fmt}|||"

    async def _cmd_cancel(self, conn: _Conn, arg: str) -> None:
        other = self.users.get(to_id(arg))
        for key in [k for k in self.challenges
                    if conn.userid in k and (other is None or other.userid in k)]:
            self.challenges.pop(key, None)
            a, b = key
            for uid in (a, b):
                user = self.users.get(uid)
                if user is not None:
                    await user.send(f"|pm| {self._name_of(a)}| {self._name_of(b)}|/challenge")

    def _name_of(self, userid: str) -> str:
        user = self.users.get(userid)
        return user.name if user else userid

    async def _cmd_accept(self, conn: _Conn, arg: str) -> None:
        if not conn.named:
            return
        from_id = to_id(arg)
        pending = self.challenges.pop((from_id, conn.userid), None)
        if pending is None:
            await conn.send(f"|popup|{arg} is not challenging you.")
            logger.warning("%s accepted a challenge from %s that does not exist",
                           conn.userid, from_id)
            return
        fmt, challenger_team = pending
        challenger = self.users.get(from_id)
        if challenger is None:
            await conn.send(f"|popup|The user '{arg}' left before the battle could start.")
            return
        if conn.team is None:
            await conn.send("|popup|You need to select a team before accepting "
                            "(`/utm <packed team>`).")
            return
        # The CHALLENGER is p1, exactly as on a real server — the matched-regime harness balances
        # roles on that fact, so getting it backwards would silently flip a measurement's sides.
        await self.start_battle(challenger, conn, fmt, challenger_team, conn.team)

    # -- battles -------------------------------------------------------------------------
    async def start_battle(self, p1: _Conn, p2: _Conn, fmt: str,
                           p1_team: Optional[str], p2_team: Optional[str]) -> _Battle:
        index = next(self._battle_index)
        tag = f"battle-{fmt}-{next(_BATTLE_SEQ)}"
        seed = derive_seed_from_base(self.seed_base, index) if self.seed_base is not None else None
        proc = await asyncio.create_subprocess_exec(
            *self._spawn_argv,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE, limit=BRIDGE_STREAM_LIMIT,
        )
        battle = _Battle(tag=tag, fmt=fmt, proc=proc, conns={"p1": p1, "p2": p2},
                         names={"p1": p1.name, "p2": p2.name}, seed=seed,
                         capture=[] if self._capture else None,
                         commands=[] if self._capture else None)
        self.battles[tag] = battle

        # The room framing the sim does not emit. Line 0 is the room, line 1 MUST be `|init|`
        # (poke-env creates the battle on exactly that), and field 4 of the frame is the title
        # Foul Play parses the opponent's name out of.
        init = f">{tag}\n|init|battle\n|title|{p1.name} vs. {p2.name}"
        await p1.send(init)
        await p2.send(init)

        start = {"formatid": fmt,
                 "p1": {"name": p1.name, "team": p1_team},
                 "p2": {"name": p2.name, "team": p2_team}}
        if seed is not None:
            start["seed"] = seed
        await self._write_child(battle, "START " + json.dumps(start))
        self._live[tag] = battle
        battle.reader = asyncio.ensure_future(self._pump(battle))
        self._pumps.add(battle.reader)
        logger.info("battle %s: %s (p1) vs %s (p2) seed=%s", tag, p1.name, p2.name, seed)
        return battle

    async def _write_child(self, battle: _Battle, command: str) -> None:
        if battle.commands is not None:
            battle.commands.append(command)
        proc = battle.proc
        if proc.stdin is None or proc.returncode is not None or proc.stdin.is_closing():
            return
        try:
            proc.stdin.write((command + "\n").encode())
            await proc.stdin.drain()
        except (BrokenPipeError, ConnectionResetError):
            return

    async def _pump(self, battle: _Battle) -> None:
        """Relay one battle's per-side chunks to its two clients until the child is done."""
        proc = battle.proc
        stderr_task = asyncio.ensure_future(self._drain_stderr(battle))
        try:
            while True:
                line = await proc.stdout.readline()
                if not line:
                    break
                text = line.decode().rstrip("\n")
                if text == "__END__":
                    break
                if text.startswith("__ERR__"):
                    msg = base64.b64decode(text[len("__ERR__ "):]).decode("utf-8")
                    logger.error("bridge error in %s: %s", battle.tag, msg)
                    for conn in battle.conns.values():
                        await conn.send(f"|popup|The battle crashed: {msg}")
                    break
                if text.startswith("__RECON__"):
                    # Referee-view, full-information data. It must never reach a player; the
                    # front end has no forensic sink, so it is dropped here.
                    continue
                slot, b64 = text.split(" ", 1)
                chunk = base64.b64decode(b64).decode("utf-8")
                chunk = self._process_chunk(battle, slot, chunk)
                if battle.capture is not None:
                    battle.capture.append((slot, chunk))
                await battle.conns[slot].send(f">{battle.tag}\n{chunk}")
        except Exception as exc:  # noqa: BLE001 — one battle dying must not kill the server
            logger.exception("pump for %s failed: %s", battle.tag, exc)
        finally:
            battle.finished = True
            stderr_task.cancel()
            await self._end_child(battle)
            self._write_capture(battle)
            if self._capture:
                self.captured_battles.append(battle)
            self.n_battles_finished += 1
            self._pumps.discard(battle.reader)
            self._live.pop(battle.tag, None)

    def _write_capture(self, battle: _Battle) -> None:
        """Persist this battle's repro record, if the server was asked for one.

        The record is the gate's input (`ws_frontend_replay.BattleCapture`) and is written per
        battle rather than at shutdown, so a series that crashes keeps everything that finished.
        """
        if self._capture_dir is None or battle.commands is None:
            return
        try:
            os.makedirs(self._capture_dir, exist_ok=True)
            path = os.path.join(self._capture_dir, f"{battle.tag}.json")
            with open(path, "w") as handle:
                json.dump({"tag": battle.tag, "seed": battle.seed,
                           "commands": battle.commands,
                           "chunks": [list(c) for c in (battle.capture or [])]}, handle)
        except OSError as exc:  # noqa: BLE001 — telemetry must never cost a battle
            logger.warning("could not write capture for %s: %s", battle.tag, exc)

    async def _drain_stderr(self, battle: _Battle) -> None:
        try:
            while True:
                line = await battle.proc.stderr.readline()
                if not line:
                    return
                logger.warning("[%s child stderr] %s", battle.tag,
                               line.decode(errors="replace").rstrip())
        except asyncio.CancelledError:  # pragma: no cover
            return

    async def _end_child(self, battle: _Battle) -> None:
        proc = battle.proc
        if proc.returncode is None:
            await self._write_child(battle, "END")
            try:
                await asyncio.wait_for(proc.wait(), timeout=10.0)
            except asyncio.TimeoutError:  # pragma: no cover
                logger.warning("child for %s did not exit within 10s of END — killing it",
                               battle.tag)
                proc.kill()
                await proc.wait()
        # Close the subprocess TRANSPORT explicitly. Without this the pipes are closed from
        # `BaseSubprocessTransport.__del__`, which for a short-lived server (a test, a one-shot
        # series) can run AFTER the event loop has closed and raises `RuntimeError: Event loop is
        # closed` out of a destructor — noise that reads like a real failure and buries one.
        transport = getattr(proc, "_transport", None)
        if transport is not None:
            transport.close()

    def _process_chunk(self, battle: _Battle, slot: str, chunk: str) -> str:
        """Apply the SERVER's own `sideupdate` bookkeeping to one per-side chunk.

        Two effects, both mirroring `server/room-battle.ts`'s `sideupdate` case:
        1. a `|request|` gets an `rqid` spliced in and bumps the room's single shared counter;
        2. a `|error|[Invalid choice]` reopens the slot's request so the client may choose again.
        Everything else passes through byte-for-byte.
        """
        out: List[str] = []
        for ln in chunk.split("\n"):
            out.append(self._note_side_line(battle, slot, ln))
        return "\n".join(out)

    def _note_side_line(self, battle: _Battle, slot: str, ln: str) -> str:
        state = battle.req[slot]
        if ln.startswith("|request|") and len(ln) > len("|request|"):
            payload = ln[len("|request|"):]
            battle.rqid += 1
            state.rqid = battle.rqid
            try:
                is_wait = bool(json.loads(payload).get("wait"))
            except ValueError:
                is_wait = False
            state.is_wait = "cantUndo" if is_wait else False
            state.choice = ""
            if payload.startswith("{") and payload.endswith("}"):
                # SPLICE, not a JSON round trip: everything before the closing brace stays the
                # sim's own bytes, so the byte-differential gate's only normalization is to
                # remove this one appended key.
                return f'|request|{payload[:-1]},"rqid":{battle.rqid}}}'
            return ln
        if ln.startswith(_INVALID_CHOICE_PREFIX):
            state.is_wait = "cantUndo" if "Can't undo" in ln else False
            state.choice = ""
        return ln

    async def _cmd_choose(self, conn: _Conn, room: str, choice: str, rqid: str) -> None:
        battle = self.battles.get(room.lstrip(">"))
        if battle is None:
            return
        slot = battle.slot_of(conn)
        if slot is None:
            return
        state = battle.req[slot]
        # `server/room-battle.ts::choose`, reproduced: the two refusals a client is written
        # against, and nothing else — every LEGALITY question stays the sim's.
        if state.is_wait is not False and state.is_wait is not True:
            await conn.send(f">{battle.tag}\n|error|[Invalid choice] There's nothing to choose")
            return
        all_wait = all(bool(s.is_wait) for s in battle.req.values())
        if all_wait or (rqid and rqid.strip() != str(state.rqid)):
            await conn.send(f">{battle.tag}\n|error|[Invalid choice] Sorry, too late to make a "
                            "different move; the next turn has already started")
            return
        state.is_wait = True
        state.choice = choice
        await self._write_child(battle, f"CHOOSE {slot} {choice}")

    async def _cmd_forfeit(self, conn: _Conn, room: str) -> None:
        battle = self.battles.get(room.lstrip(">"))
        if battle is None:
            return
        slot = battle.slot_of(conn)
        if slot is not None:
            await self._write_child(battle, f"FORCELOSE {slot}")

    async def _cmd_leave(self, conn: _Conn, room: str) -> None:
        tag = room.strip().lstrip(">")
        battle = self.battles.get(tag)
        await conn.send(f">{tag}\n|deinit")
        if battle is None:
            return
        if battle.finished:
            # Both sides have their result; the room can go. Keep the capture (a harness may
            # still be reading it) but stop holding the child.
            self.battles.pop(tag, None)

    # -- lifecycle -----------------------------------------------------------------------
    async def drain(self, timeout: float = 120.0) -> None:
        """Wait until every in-flight battle's pump has finished.

        🚨 **A CLIENT SEEING `|win|` IS NOT THE BATTLE BEING OVER HERE.** `battle_against` returns
        the moment poke-env parses the win, while this server is still reading the child's tail,
        reaping it and recording the capture. A harness that inspects `captured_battles` without
        draining reads an EMPTY list and concludes the recorder is broken — which is exactly what
        it looked like the first time.
        """
        pending = {t for t in self._pumps if not t.done()}
        if pending:
            await asyncio.wait(pending, timeout=timeout)

    async def close(self, drain_timeout: float = 10.0) -> None:
        """Stop every battle and reap every child. Safe to call with battles in flight."""
        await self.drain(timeout=drain_timeout)
        for battle in list(self._live.values()):
            if battle.reader is not None and not battle.reader.done():
                battle.reader.cancel()
            await self._end_child(battle)
        self._live.clear()
        self.battles.clear()


# --------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m utils.bridge.ws_frontend", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--port", type=int, required=True,
                   help="port to bind (8000/8001 are REFUSED; use the 96XX range)")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--impl", default="rust", choices=("rust", "node"),
                   help="which sim bridge child backs each battle (default rust)")
    p.add_argument("--format", dest="battle_format", default="gen3ou")
    p.add_argument("--seed-base", type=int, default=None,
                   help="derive each battle's PRNG seed from this base (reproducible series)")
    p.add_argument("--capture-dir", default=None,
                   help="write each finished battle's repro record (commands + per-side chunks) "
                        "here; pair with --seed-base to make it replayable")
    p.add_argument("--no-validate-teams", dest="validate_teams", action="store_false",
                   help="skip the Node team validator on /utm (faster; loses the loud refusal)")
    p.add_argument("--log-level", default="INFO")
    return p


async def serve_forever(args) -> int:
    from websockets.asyncio.server import serve

    if args.port in RESERVED_PORTS:
        raise SystemExit(f"refusing --port {args.port}: that is {RESERVED_PORTS[args.port]}.")
    if args.capture_dir and args.seed_base is None:
        logger.warning("--capture-dir without --seed-base: the child mints its own seed per "
                       "battle, so the records will NOT be replayable.")
    front = ShowdownFrontEnd(impl=args.impl, battle_format=args.battle_format,
                             seed_base=args.seed_base, validate_teams=args.validate_teams,
                             capture_dir=args.capture_dir)
    async with serve(front.handler, args.host, args.port, max_size=None, ping_interval=None):
        uri = f"ws://{args.host}:{args.port}/showdown/websocket"
        logger.info("ws_frontend listening — %s  (impl=%s, format=%s, seed_base=%s)",
                    uri, args.impl, args.battle_format, args.seed_base)
        print(f"[ws_frontend] READY {uri}", flush=True)
        await asyncio.Future()
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.seed_base is not None:
        validate_seed_spec(derive_seed_from_base(args.seed_base, 0))
    try:
        return asyncio.run(serve_forever(args))
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
    sys.exit(main())
