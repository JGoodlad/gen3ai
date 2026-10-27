"""Logs → (observation row, opponent-action label) rows through the Rust live reader (X50 rung 1, design §2, §6).

Two reads of one seat of one battle, both through ``main.live.reader.LiveReader`` (the ``live_reader`` binary, the
training chain):

* :func:`read_spectator` — a PUBLIC log (a human replay, or :mod:`main.x50_r1.spectate`'s view of our own battle):
  OPENED WITH NO TEAM, the lines fed up to each of the viewer's decision points (:mod:`main.x50_r1.points`), and at
  each point ONE synthesised spectator request (``SPECDECIDE``: legality only). The own side is public-only.
* :func:`read_full` — the viewer's OWN stream with its real requests and packed team (our traces only): the row
  training read. The control.

The LABEL of decision ``k`` is the trackers' intent label read at decision ``k + 1`` (``LABEL``; the window decision
``k`` opened — training's shift by one), converted to ``(kind, num, slot, species)`` as ``rust_env``'s
``labels/intent.rs`` does, EXCEPT that a bare Hidden Power stays num 237 and is flagged (a replay never states the
type; design §1). The battle's last decision has no label.

THE PREFIX GUARD (:func:`prefix_guard`): a decision's row must equal the row a FRESH reader makes from the lines
before that decision point ALONE (the list physically truncated there). A mismatch is a :class:`GuardError`.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from main.live.reader import Frame, LiveReader
from main.x50_r1.points import FORCED, decision_points

KIND_MOVE, KIND_SWITCH, KIND_UNKNOWN = 0, 1, 2
SWITCH_SLOT_NONE = -100
HIDDEN_POWER_NUM = 237
UNKNOWN_LABEL = (KIND_UNKNOWN, 0, SWITCH_SLOT_NONE, 0)


class GuardError(RuntimeError):
    """A row that a prefix-only read does not reproduce: a line past the decision reached it."""


class LabelError(RuntimeError):
    """A label lookup that cannot be made (a move / species with no num): never a silent UNKNOWN."""


def to_id(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def label_tuple(lab: Optional[dict]) -> Tuple[Tuple[int, int, int, int], bool]:
    """The Rust ``IntentLabel`` JSON → ``((kind, num, slot, species), is_bare_hidden_power)``."""
    from agents.gen3_data import moves, species

    if lab is None or int(lab["kind"]) == KIND_UNKNOWN:
        return UNKNOWN_LABEL, False
    kind = int(lab["kind"])
    if kind == KIND_MOVE:
        mid = lab.get("move_id")
        if not mid:
            return UNKNOWN_LABEL, False
        md = moves.get(to_id(mid))
        if md is None:
            raise LabelError(f"intent label: move {mid!r} has no num")
        num = int(md.num)
        return (KIND_MOVE, num, SWITCH_SLOT_NONE, 0), num == HIDDEN_POWER_NUM
    if kind == KIND_SWITCH:
        sp = lab.get("switch_species")
        if not sp:
            raise LabelError("intent label: a SWITCH label with no switch-in species")
        sd = species.get(to_id(sp))
        if sd is None:
            raise LabelError(f"intent label: species {sp!r} has no num")
        slot = lab.get("switch_slot")
        return (KIND_SWITCH, 0, SWITCH_SLOT_NONE if slot is None else int(slot), int(sd.num)), False
    raise LabelError(f"intent label: unknown kind {kind}")


@dataclass
class SeatRead:
    """One seat's decisions of one battle, in order."""
    rows: List[np.ndarray] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)
    pos: List[int] = field(default_factory=list)       # the decision point (lines fed before it; spectator only)
    forced: List[bool] = field(default_factory=list)
    turn: List[int] = field(default_factory=list)
    placeholder: List[bool] = field(default_factory=list)
    raw_labels: List[Optional[dict]] = field(default_factory=list)   # LABEL read AT decision k (= k - 1's label)
    tokens: List[Dict[int, str]] = field(default_factory=list)       # the frame's choice tokens

    def labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """``[n, 4]`` labels aligned to the decisions (decision k ← the label read at k + 1) and the HP flag."""
        n = len(self.rows)
        lab = np.tile(np.array(UNKNOWN_LABEL, dtype=np.int64), (n, 1))
        hp = np.zeros(n, dtype=bool)
        for k in range(n - 1):
            t, h = label_tuple(self.raw_labels[k + 1])
            lab[k] = t
            hp[k] = h
        return lab, hp


def _take(read: SeatRead, fr: Frame, *, pos: int, forced: bool, placeholder: bool, raw: Optional[dict]) -> None:
    read.rows.append(np.array(fr.row, dtype=np.float32))
    read.masks.append(np.array(fr.mask, dtype=np.int8))
    read.pos.append(pos)
    read.forced.append(forced)
    read.turn.append(fr.turn)
    read.placeholder.append(placeholder)
    read.raw_labels.append(raw)
    read.tokens.append(dict(fr.tokens))


def read_spectator(reader: LiveReader, lines: Sequence[str], seat: str, name: str,
                   points: Optional[List[Tuple[int, str]]] = None) -> SeatRead:
    """The viewer ``seat``'s decisions over a PUBLIC log (``lines`` = ``replay_scan.battle_lines``)."""
    pts = decision_points(list(lines), seat) if points is None else points
    reader.open(seat, name, None)
    out = SeatRead()
    prev = 0
    for pos, kind in pts:
        if pos > prev:
            if reader.feed(list(lines[prev:pos])) is not None:
                raise RuntimeError("a spectator reading opened a decision on a public line")
            prev = pos
        fr, ph = reader.spectator_decide(kind == FORCED)
        if fr is None:
            raise RuntimeError(f"{seat}: the {kind} point at line {pos} opened no decision")
        _, lab = reader.intent_label()
        _take(out, fr, pos=pos, forced=kind == FORCED, placeholder=ph, raw=lab)
    return out


def read_full(reader: LiveReader, own_lines: Sequence[str], seat: str, name: str, packed_team: str) -> SeatRead:
    """The viewer's decisions over its OWN stream (real requests, its packed team): the training row."""
    reader.open(seat, name, packed_team)
    out = SeatRead()
    write: List[str] = []
    n_public = 0
    for ln in own_lines:
        write.append(ln)
        if not ln.startswith("|request|"):
            n_public += 1
            continue
        fr = reader.feed(write)
        write = []
        if fr is not None:
            _, lab = reader.intent_label()
            forced = '"forceSwitch":[true]' in ln
            _take(out, fr, pos=n_public, forced=forced, placeholder=False, raw=lab)
    return out


def full_own_actions(lines: Sequence[str], seat: str, points: List[Tuple[int, str]], full: SeatRead) -> List[int]:
    """The viewer's OWN action at each decision of its FULL read (its REAL requests): (b)'s label, read off the public
    log's decision windows (``own.viewer_actions``, as :func:`read_completed` reads it) and mapped onto the real
    request's frame by ``own.action_index`` (-2 = not legal under the real request: the label-legality refusal), the
    move set being that frame's own move tokens. ``points`` are the spectator decision points, which the caller has
    checked equal the seat's real request positions."""
    from main.x50_r1 import own as O

    if len(points) != len(full.rows):
        raise GuardError(f"{seat}: {len(points)} decision points but {len(full.rows)} full-read decisions")
    acts = O.viewer_actions(lines, seat, points)
    return [O.action_index(a, tk, m, [t[len("move "):] for t in tk.values() if t.startswith("move ")])
            for a, tk, m in zip(acts, full.tokens, full.masks)]


@dataclass
class CompletedRead:
    """:func:`read_completed`'s result: the rows (a :class:`SeatRead`), the synthesised request of each decision,
    and the viewer's OWN action at each decision as its 11-way index (-1 no label, -2 not legal: a refusal)."""
    read: SeatRead
    requests: List[str]
    own_action: List[int]
    n_legal: List[int]
    active: List[Optional[str]] = field(default_factory=list)          # the own active's nickname
    choice_lock: List[Optional[str]] = field(default_factory=list)     # its Choice Band lock, if any


def decision_locks(lines: Sequence[str], seat: str, points: List[Tuple[int, str]]) -> List[Optional[str]]:
    """Per decision, the move the viewer's request was LOCKED to (``recharge`` for a recharge), read off the
    decision's own window (its continuation line), or None. The player's real request said so; it is an input of
    the request like the team, so the prefix guard receives it, never re-derives it from a truncated log."""
    from main.x50_r1 import own as O

    acts = O.viewer_actions(lines, seat, points)
    charges = O.charge_locks(lines, seat, points)
    return [a.locked or c for a, c in zip(acts, charges)]


def read_completed(reader: LiveReader, lines: Sequence[str], seat: str, name: str, packed_team: str,
                   genders: Optional[Dict[str, str]] = None,
                   points: Optional[List[Tuple[int, str]]] = None,
                   locks: Optional[List[Optional[str]]] = None) -> CompletedRead:
    """The viewer ``seat``'s decisions over a PUBLIC log with its own side COMPLETED (rung 1b step 0): the reader is
    OPENED with ``packed_team``, every public line naming an own mon is fed with its HP exact
    (:class:`main.x50_r1.own.OwnTracker`), and each decision opens on ONE synthesised REAL request (the whole
    roster, the active's moves, the public lines' legality). The label read is the production one, as in
    :func:`read_spectator`; the viewer's own action (use (b)'s label) is read off the decision's window."""
    from main.x50_r1 import own as O

    pts = decision_points(list(lines), seat) if points is None else points
    team = O.parse_team(packed_team, genders)
    trk = O.OwnTracker(seat, name, team)
    acts = O.viewer_actions(lines, seat, pts)
    lk = decision_locks(lines, seat, pts) if locks is None else locks
    reader.open(seat, name, packed_team)
    out = CompletedRead(SeatRead(), [], [], [])
    prev = 0
    carry: List[str] = []
    for k, (pos, kind) in enumerate(pts):
        write = carry + [x for ln in lines[prev:pos] for x in trk.lines(ln)] + trk.flush()
        carry = []
        prev = pos
        req, _ = trk.request(kind == FORCED, lk[k])
        fr = reader.feed(write + [req])
        if fr is None:
            raise RuntimeError(f"{seat}: the completed {kind} point at line {pos} opened no decision")
        _, lab = reader.intent_label()
        _take(out.read, fr, pos=pos, forced=kind == FORCED, placeholder=False, raw=lab)
        out.requests.append(req)
        if '"id":"struggle"' in req and trk.active:
            # the own stream's `-activate|…|move: Struggle` follows the choice (the next write's first line)
            carry = [f"|-activate|{seat}a: {trk.active}|move: Struggle"]
        moves = trk.mons[trk.active].moves if trk.active else []      # Mimic's swap included
        out.own_action.append(O.action_index(acts[k], fr.tokens, fr.mask, moves))
        out.active.append(trk.active)
        out.choice_lock.append(trk.vol.choice if kind != FORCED else None)
        out.n_legal.append(int(np.asarray(fr.mask).sum()))
    return out


def prefix_guard(reader: LiveReader, lines: Sequence[str], seat: str, name: str,
                 points: List[Tuple[int, str]], k: int, row: np.ndarray) -> None:
    """Raise :class:`GuardError` unless decision ``k``'s ``row`` is what a fresh reader makes from the lines
    BEFORE ``points[k]`` alone (the list truncated there, the same earlier decisions)."""
    cut = points[k][0]
    prefix = list(lines[:cut])
    again = read_spectator(reader, prefix, seat, name, points=points[:k + 1])
    if not np.array_equal(again.rows[k].view(np.uint32), np.asarray(row, dtype=np.float32).view(np.uint32)):
        raise GuardError(f"{seat} decision {k} (line {cut}): the row differs from the prefix-only read")


def prefix_guard_completed(reader: LiveReader, lines: Sequence[str], seat: str, name: str, packed_team: str,
                           genders: Optional[Dict[str, str]], points: List[Tuple[int, str]],
                           locks: List[Optional[str]], k: int, row: np.ndarray) -> None:
    """:func:`prefix_guard` for a COMPLETED read: the same team and request locks, the lines before
    ``points[k]`` alone. The team itself holds facts revealed later (C-GAME's declared leak, F-1b-4); the guard
    checks that nothing else past the decision reaches its row."""
    cut = points[k][0]
    again = read_completed(reader, list(lines[:cut]), seat, name, packed_team, genders, points=points[:k + 1],
                           locks=locks[:k + 1])
    if not np.array_equal(again.read.rows[k].view(np.uint32), np.asarray(row, dtype=np.float32).view(np.uint32)):
        raise GuardError(f"{seat} decision {k} (line {cut}): the completed row differs from the prefix-only read")


def guard_index(battle: str, seat: str, n: int) -> int:
    """The ONE decision of a seat the extractor guards (deterministic in the battle id and seat)."""
    h = int(hashlib.sha256(f"{battle}|{seat}".encode()).hexdigest()[:8], 16)
    return h % max(n, 1)


def player_info(lines: Sequence[str]) -> Dict[str, Dict[str, Optional[str]]]:
    """Each seat's first non-empty name and pre-battle rating (``|player|pN|name|avatar|rating``)."""
    out: Dict[str, Dict[str, Optional[str]]] = {}
    for ln in lines:
        if not ln.startswith("|player|"):
            continue
        p = ln.split("|")
        if len(p) < 4 or p[2] not in ("p1", "p2"):
            continue
        d = out.setdefault(p[2], {"name": None, "rating": None})
        if p[3] and d["name"] is None:
            d["name"] = p[3]
        if len(p) > 5 and p[5].strip().isdigit() and d["rating"] is None:
            d["rating"] = p[5].strip()
    return out
