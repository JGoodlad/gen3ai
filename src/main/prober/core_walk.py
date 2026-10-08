"""WALK a recorded battle through the RUST CORE — the prober's battle reading (``gen3_core_walk_v1``).

The poke-env retirement's P5 (``T27``): the prober reads a Rust-eval core trace from the core's own
state — the chain of ``BattleVersion``s the trainee's stream folds into, the SAME reading, trackers
and rows training runs on — never by re-parsing the protocol through poke-env's ``Battle``.

One ``core_events --walk`` process replays the battle's ``*_reconstruction.json`` (the recorded
commands, the raw ``>start`` seed) through the production bridge session, and reports per side, per
DECISION: the side's ``present()`` view (poke-env's reading of the board, every rule applied in
Rust — ``agents.battle.core_view`` copies it into a ``LiveView`` with no rule of its own), its
legality, the tracker state (the slot registries and the frozen ``TurnDelta``'s projection that
closed the window), the win-indicator reward, the encoded row's 11-bit mask and the choice token
of every legal action; and, once the battle has ended, the TERMINAL view and the delta of the
window that ended it (``RewardTracker.finalize``'s ``TurnDelta``). Every transition is gated by the
binary itself: the parse chain (one side's TEXT, what a server sends) must agree with the step
chain at every version, and its encoded row must be byte-identical (``gen3_core_parse_obs_gate_v1``).

This module is the TRANSPORT and the alignment; :mod:`main.prober.core_recorder` turns a walk into
the prober's legacy summary shape. Nothing here imports poke-env (``src/poke_env_free_entry_points_test.py``).
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from agents.battle.core_view import legal_actions_from_core, live_view_from_core
from agents.battle.live_view import LegalActions, LiveView

WALK_SCHEMA = "gen3_core_walk_v1"

#: ``Player.MESSAGES_TO_IGNORE`` — the keywords poke-env's player drops before its battle sees them
#: (so they never reach the replay log). Pinned to this literal by ``core_walk_test.py`` (it was pinned to the fork's
#: own set by the poke-env oracle test in ``core_trace_integration_test.py`` until P6 slice 6d-1 deleted it).
PLAYER_IGNORED = frozenset({"t:", "expire", "uhtmlchange"})


class CoreWalkError(RuntimeError):
    """The core REFUSED the battle (a parse-vs-step disagreement, an engine fault, a bad record)."""


@dataclass(frozen=True)
class WalkDecision:
    """One decision of the walked side, as the core folded it."""

    index: int                      # 0-based, in the side's decision order
    after: int                      # the chunk index of the request it decided on
    view: LiveView
    legal: Optional[LegalActions]
    delta: Mapping[str, Any]        # the frozen TurnDelta's projection (the window this decision closed)
    our_slots: Tuple[str, ...]      # the slot registries AFTER this decision's context assigned
    opp_slots: Tuple[str, ...]
    reward: float                   # the win indicator of the transition into this decision
    mask: np.ndarray                # (11,) int8 — the encoded row's legal mask
    tokens: Mapping[int, str]       # action index -> the choice string the core's mapper emits
    choice: Optional[str]           # the token this side actually sent at this decision


@dataclass(frozen=True)
class CoreWalk:
    side: str
    decisions: Tuple[WalkDecision, ...]
    #: ``(side, text)`` per flushed chunk, both sides, in order — the bridge's own stream.
    chunks: Tuple[Tuple[str, str], ...]
    ended: bool
    winner: Optional[str]
    terminal_view: Optional[LiveView]
    terminal_delta: Optional[Mapping[str, Any]]
    terminal_reward: Optional[float]

    def side_lines(self) -> List[str]:
        """The walked side's protocol lines, in order (every chunk split on newlines)."""
        return [ln for s, c in self.chunks if s == self.side for ln in c.split("\n")]


def script(record, *, label: str = "walk") -> List[str]:
    """The ``core_events`` stdin for ``record`` (a :class:`~utils.bridge.reconstruction.ReconstructionRecord`):
    the RAW ``>start`` seed (the live ``sim_bridge`` path runs the turn-0 construction window), both
    players as they were declared, then every command the bridge processed, in order."""
    players = record.players()
    start = {"label": label, "formatid": record.format_id, "seed": record.start_options()["seed"],
             "p1": {"name": players["p1"]["name"], "team": players["p1"]["team"]},
             "p2": {"name": players["p2"]["name"], "team": players["p2"]["team"]}}
    out = ["START " + json.dumps(start)]
    for cmd in record.commands:
        side, choice = cmd[0], cmd[1]
        if side == "forcelose":
            out.append(f"FORCELOSE {choice}")
        elif side in ("p1", "p2"):
            if "\n" in choice:
                raise CoreWalkError(f"a command carries a newline: {cmd!r}")
            out.append(f"CHOOSE {side} {choice}")
        else:
            raise CoreWalkError(f"unknown reconstruction command {cmd!r}")
    out.append("END")
    return out


def run_core(record, *, timeout: float = 300.0) -> dict:
    """``core_events --walk`` over ``record``; the battle's raw result object (refused ⇒ raises)."""
    from utils.bridge.sim_bridge_bin import resolve_core_events_bin
    from utils.contention import scale_timeout

    stdin = "\n".join(script(record)) + "\n"
    p = subprocess.run([resolve_core_events_bin(), "--walk"], input=stdin, capture_output=True,
                       text=True, timeout=scale_timeout(timeout), check=False)
    if p.returncode != 0:
        raise CoreWalkError(f"core_events --walk failed (exit {p.returncode}): {p.stderr.strip()[-2000:]}")
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    if len(lines) != 1:
        raise CoreWalkError(f"core_events --walk answered {len(lines)} battles for 1")
    res = json.loads(lines[0])
    if not res.get("ok"):
        raise CoreWalkError(f"the core REFUSED the battle: {res.get('error')}")
    return res


def _tokens(raw: Mapping[str, str]) -> Dict[int, str]:
    return {int(k): v for k, v in raw.items()}


def from_result(res: Mapping[str, Any], side: str, *, battle_tag: str = "") -> CoreWalk:
    """The :class:`CoreWalk` of ``side`` in one ``core_events --walk`` result."""
    vi = {"p1": 0, "p2": 1}[side]
    chunks = tuple(("p1" if s == 0 else "p2", "\n".join(lines)) for s, lines in res["chunks"])
    decisions: List[WalkDecision] = []
    terminal: Optional[Mapping[str, Any]] = None
    for cap in res.get("trackers", [[], []])[vi]:
        if "terminal" in cap:
            terminal = cap
            continue
        if terminal is not None:
            raise CoreWalkError("a decision after the terminal record")
        if "view" not in cap or "mask" not in cap:
            raise CoreWalkError("core_events was not run with --walk (no per-decision view / mask)")
        trk = cap["trackers"]
        decisions.append(WalkDecision(
            index=len(decisions), after=int(cap["after"]),
            view=live_view_from_core(cap["view"], battle_tag=battle_tag),
            legal=legal_actions_from_core(cap["legal"], cap.get("request")),
            delta=trk["delta"], our_slots=tuple(trk["our_slots"]), opp_slots=tuple(trk["opp_slots"]),
            reward=float(cap["reward"]), mask=np.asarray(cap["mask"], dtype=np.int8),
            tokens=_tokens(cap["tokens"]), choice=cap.get("choice")))
    tview = tdelta = treward = None
    if terminal is not None:
        if "view" not in terminal:
            raise CoreWalkError("core_events was not run with --walk (no terminal view)")
        tview = live_view_from_core(terminal["view"], battle_tag=battle_tag)
        tdelta = terminal.get("delta")
        treward = float(terminal["terminal"])
    return CoreWalk(side=side, decisions=tuple(decisions), chunks=chunks, ended=bool(res.get("ended")),
                    winner=res.get("winner"), terminal_view=tview, terminal_delta=tdelta,
                    terminal_reward=treward)


def walk(record, side: str, *, battle_tag: str = "", timeout: float = 300.0) -> CoreWalk:
    """Walk ``record`` through the core and return ``side``'s decisions + terminal."""
    return from_result(run_core(record, timeout=timeout), side, battle_tag=battle_tag)


_CACHE_CAP = 64
_cache: "OrderedDict[Tuple[str, str], CoreWalk]" = OrderedDict()


def walk_cached(record, side: str) -> CoreWalk:
    """:func:`walk`, memoised in memory by the record's content (a counterfactual view walks the same
    battle once per anchored decision)."""
    key = (hashlib.sha256(json.dumps(record.to_dict(), sort_keys=True).encode()).hexdigest(), side)
    hit = _cache.get(key)
    if hit is None:
        hit = _cache[key] = walk(record, side)
        while len(_cache) > _CACHE_CAP:
            _cache.popitem(last=False)
    else:
        _cache.move_to_end(key)
    return hit


def decision_choices(record, side: str, index: int) -> Tuple[Dict[int, str], int]:
    """``({legal action index: sim choice string}, turn)`` at ``side``'s ``index``-th decision of
    ``record`` — the core's ``present::choice_tokens`` (the mapper training's rows use, held equal to
    the Python mapper by slice O), legality from the row's mask. Replaces the poke-env materializer's
    ``map_actions_at`` for the counterfactual views (P5). Raises :class:`CoreWalkError` past the end."""
    w = walk_cached(record, side)
    if not 0 <= index < len(w.decisions):
        raise CoreWalkError(f"decision {index} out of range: the core walked {len(w.decisions)} "
                            f"decisions of {side}")
    d = w.decisions[index]
    return {a: tok for a, tok in sorted(d.tokens.items()) if d.mask[a]}, d.view.turn


@dataclass(frozen=True)
class StreamDecision:
    """One decision of a side's protocol stream, read by the core's PARSE chain (``--obs-stream``)."""

    k: int
    turn: int
    mask: np.ndarray                 # (11,) int8
    tokens: Mapping[int, str]        # legal action index -> the choice string the core's mapper emits
    obs: Optional[np.ndarray]        # the encoded row (read-only float32), only where asked

    @property
    def choices(self) -> Dict[int, str]:
        """The LEGAL actions' choice strings (``action_choices`` of the poke-env materializer)."""
        return {a: t for a, t in sorted(self.tokens.items()) if self.mask[a]}


@dataclass(frozen=True)
class StreamRequest:
    """One side's stream to read: ``chunks`` (the side's text, chunk by chunk, in order), the side's
    own ``actions`` by index (replayed as the core's choice tokens), and the decisions to ENCODE
    (``None`` = every one)."""

    username: str
    packed_team: Optional[str]
    side: str
    chunks: Sequence[str]
    actions: Sequence[int] = ()
    encode_at: Optional[Sequence[int]] = None


def read_streams(requests: Sequence[StreamRequest], *, timeout: float = 300.0) -> List[List[StreamDecision]]:
    """Every request's decisions, read by ONE ``core_events --obs-stream`` process
    (``gen3_core_obs_stream_v1``): the parse chain with the trackers on — the chain the training rows
    are encoded on — over exactly the text a client of that side received. The Rust twin of
    ``obs_materializer.materialize_decisions`` (and the deleted ``materialize_branches``, P6 slice 5). A refused stream raises
    :class:`CoreWalkError` naming it."""
    from agents.battle.core_obs import wrap_row
    from utils.bridge.sim_bridge_bin import resolve_core_events_bin
    from utils.contention import scale_timeout

    if not requests:
        return []
    buf: List[str] = []
    for r in requests:
        head = {"viewer": {"p1": 0, "p2": 1}[r.side], "username": r.username, "team": r.packed_team,
                "actions": [int(a) for a in r.actions],
                "encode_at": None if r.encode_at is None else [int(k) for k in r.encode_at]}
        buf.append("STREAM " + json.dumps(head))
        for c in r.chunks:
            for ln in c.split("\n"):
                if ln in ("END",) or ln.startswith("STREAM "):
                    raise CoreWalkError(f"a protocol line collides with the stream framing: {ln!r}")
                buf.append(ln)
        buf.append("END")
    p = subprocess.run([resolve_core_events_bin(), "--obs-stream"], input="\n".join(buf) + "\n",
                       capture_output=True, text=True, timeout=scale_timeout(timeout), check=False)
    if p.returncode != 0:
        raise CoreWalkError(f"core_events --obs-stream failed (exit {p.returncode}): {p.stderr.strip()[-2000:]}")
    lines = [ln for ln in p.stdout.splitlines() if ln.strip()]
    if len(lines) != len(requests):
        raise CoreWalkError(f"core_events --obs-stream answered {len(lines)} streams for {len(requests)}")
    out: List[List[StreamDecision]] = []
    for i, ln in enumerate(lines):
        res = json.loads(ln)
        if not res.get("ok"):
            raise CoreWalkError(f"the core REFUSED stream {i} ({requests[i].side}): {res.get('error')}")
        out.append([StreamDecision(k=int(d["k"]), turn=int(d["turn"]), mask=np.asarray(d["mask"], dtype=np.int8),
                                   tokens=_tokens(d["tokens"]),
                                   obs=wrap_row(d["obs"]) if "obs" in d else None)
                    for d in res["decisions"]])
    return out


def replay_log(lines: Sequence[str]) -> Tuple[str, ...]:
    """The replay log poke-env's battle would have kept of one side's ``lines`` — what
    ``write_battle_record``'s ``*_replay.html`` carries (``battle._build_replay_events()``, ``|``-lines).

    The DISPATCH of ``Player._handle_battle_message``: a line with no ``|`` is not the battle's; a bare
    keyword and every keyword the player does not intercept reach ``parse_message`` (which logs the line
    first); ``request`` / ``error`` / ``bigerror`` and the player's ignore set never do; ``win`` / ``tie``
    are logged by ``won_by`` / ``tied`` unless a terminal result is already logged. (poke-env's
    fallback — a ``|win|`` / ``|tie`` appended to a FINISHED battle whose stream shipped none — has no
    twin here: a core battle's stream always ships its own, and the expansion's protocol cross-check
    against the stored record would show a stream that did not.)

    The log opens with ``|init|battle``: the ROOM framing every client receives first (the server's,
    and the in-process bridge's — ``utils.bridge.battle_stream_client``), which creates the battle and
    is logged like any line; the engine's per-side stream itself does not carry it."""
    out: List[str] = [] if (lines and lines[0] == "|init|battle") else ["|init|battle"]
    terminal = False
    for line in lines:
        sm = line.split("|")
        if len(sm) == 1:
            continue
        kw = sm[1]
        if kw == "":
            out.append(line)
        elif kw in PLAYER_IGNORED or kw in ("request", "error", "bigerror"):
            continue
        elif kw == "showteam":
            raise CoreWalkError("|showteam| is not a gen-3 line")
        elif kw in ("win", "tie"):
            if not terminal:
                out.append("|win|" + sm[2] if kw == "win" else "|tie")
                terminal = True
        else:
            out.append(line)
    return tuple(ln for ln in out if ln.startswith("|"))
