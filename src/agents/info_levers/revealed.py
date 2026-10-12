"""WHAT THE VIEWER HAS SEEN: the ONE source of "what play has revealed" about the opponent's team, derived from the
viewer's own PROTOCOL stream (the battle LOG it was sent), not from its observation rows.

Every Python consumer of the information levers reads it (``agents.info_levers.spec``): the search world builder
(``main.value_search.worlds``: what it may keep and what it must draw), the model-input injection (which belief
channels are pinned by play), and the audits. The Rust encoder's reading parses the same stream for the observation;
``agents.info_levers.audit`` checks the two agree on every decision of real battles (the "log consistency" audit).

For each opponent mon the viewer has seen (switched / dragged in), keyed by species id:

* ``moves`` — its OWN moves play named: a ``|move|`` line (not one CALLED from another mon's set: ``[from] Assist`` /
  ``Snatch`` / ``Magic Coat`` / ``Metronome`` / ``Mirror Move``; ``Sleep Talk`` / ``Pursuit`` / a locked move are its
  own), Focus Punch's ``-singleturn`` tightening, and the move a ``cant`` line says it chose; never a move it used
  while TRANSFORMED or one it copied with Mimic, never Struggle. A gen-3 Hidden Power is named without its type:
  ``hiddenpower``.
* ``item`` — its ORIGINAL held item, once a line names it before any item TRANSFER involving it (an item it gained by
  Trick / Thief / Covet is not its original): ``[from] item: X`` on its own line, ``-enditem`` (eaten, knocked off,
  stolen), ``-activate|…|item: X``, a Recycle, or the item OUR mon received from it by Thief / Trick.
* ``ability`` — its ORIGINAL ability, once a line names it before any transfer: ``-ability`` without a source (an
  Intimidate / Pressure announce), Trace (the tracer's original is Trace, the traced mon's is the ability copied), and
  ``[from] ability: X`` where the line's form makes the owner certain (``OWNER_OF_FROM_ABILITY``).

CONSERVATIVE BY DESIGN. A fact this parser misses is only drawn by the builder and then forced back to the truth by
the protocol gate (inefficient, never wrong); a fact it CLAIMED that play did not reveal would be kept TRUE (a leak).
So a pattern whose owner is ambiguous reveals nothing and is counted in ``Reveals.ambiguous``.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

#: A move line CALLED from someone else's set (not a reveal of the user's own move).
NOT_OWN_CALLERS = frozenset({"assist", "snatch", "magiccoat", "metronome", "mirrormove", "naturepower",
                             "copycat", "mefirst"})
#: For ``[from] ability: X`` on a line of kind K: whose ability is X? "subject" (the line's first mon), "of" (the
#: ``[of]`` mon, else the subject). A kind absent here with an ``[of]`` that differs from the subject is AMBIGUOUS.
#: (Showdown's battle.ts: ``heal`` writes ``[of] source`` = the attacker for Volt / Water Absorb, so a ``-heal``'s
#: ability is the SUBJECT's; ``damage`` / ``trySetStatus`` / weather write ``[of] the holder``.)
OWNER_OF_FROM_ABILITY: Dict[str, str] = {
    "-heal": "subject", "-immune": "subject", "-curestatus": "subject", "-cureteam": "subject",
    "-formechange": "subject", "-transform": "subject", "-activate": "subject", "-boost": "subject",
    "-damage": "of", "-status": "of", "-weather": "of", "-fail": "of", "-start": "of", "-unboost": "of",
    "-sethp": "of",
}


def toid(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def _side_and_name(ident: str) -> Optional[Tuple[str, str]]:
    """``p2a: Nick`` → (``p2``, ``Nick``); None for a token that is not a mon."""
    head, sep, name = ident.partition(": ")
    if not sep or len(head) < 2 or head[0] != "p" or not head[1].isdigit():
        return None
    return head[:2], name.strip()


@dataclass
class MonReveal:
    """What play revealed about ONE opponent mon (module docstring)."""

    species: str
    moves: Set[str] = field(default_factory=set)
    item: Optional[str] = None            # the ORIGINAL item, once revealed
    ability: Optional[str] = None         # the ORIGINAL ability, once revealed
    item_moved: bool = False              # an item transfer happened before its original was revealed
    ability_moved: bool = False
    hp: Optional[str] = None              # the last displayed HP token (e.g. ``45/100``)


@dataclass
class Reveals:
    """Every opponent mon the viewer has seen, by species id, after the protocol it was sent up to a point."""

    mons: Dict[str, MonReveal] = field(default_factory=dict)
    ambiguous: Dict[str, int] = field(default_factory=dict)   # pattern → count (nothing was revealed for them)

    def seen(self) -> Set[str]:
        return set(self.mons)

    def moves(self) -> Dict[str, Set[str]]:
        """{species: its own moves play named} — the builder's ``kept`` map."""
        return {k: set(v.moves) for k, v in self.mons.items()}

    def as_json(self) -> Dict[str, dict]:
        return {k: {"moves": sorted(v.moves), "item": v.item, "ability": v.ability} for k, v in self.mons.items()}


class _Reader:
    def __init__(self, viewer: int):
        self.opp = f"p{2 - viewer}"
        self.out = Reveals()
        self.nick: Dict[Tuple[str, str], str] = {}        # (side, nickname) → species id
        self.active: Dict[str, Tuple[str, str]] = {}      # position (p2a) → (side, nickname)
        self.transformed: Set[Tuple[str, str]] = set()
        self.mimic: Dict[Tuple[str, str], Set[str]] = {}
        self.trick: List[Tuple[str, str]] = []               # the two mons of a Trick whose item lines follow

    # ---------------------------------------------------------------- helpers
    def _mon(self, tok: str) -> Optional[Tuple[str, str]]:
        sn = _side_and_name(tok)
        return sn if sn is not None and sn in self.nick else None

    def _opp(self, key: Optional[Tuple[str, str]]) -> Optional[MonReveal]:
        if key is None or key[0] != self.opp:
            return None
        return self.out.mons.get(self.nick[key])

    def _amb(self, what: str) -> None:
        self.out.ambiguous[what] = self.out.ambiguous.get(what, 0) + 1

    def _item(self, key: Optional[Tuple[str, str]], item: str) -> None:
        m = self._opp(key)
        if m is not None and m.item is None and not m.item_moved and item:
            m.item = toid(item)

    def _ability(self, key: Optional[Tuple[str, str]], ab: str) -> None:
        m = self._opp(key)
        if m is not None and m.ability is None and not m.ability_moved and ab:
            m.ability = toid(ab)

    def _end_trick(self) -> None:
        for k in self.trick:
            m = self._opp(k)
            if m is not None and m.item is None:
                m.item_moved = True                           # its later items are not its original
        self.trick = []

    def _leave(self, pos: str) -> None:
        old = self.active.pop(pos, None)
        if old is not None:
            self.transformed.discard(old)
            self.mimic.pop(old, None)

    # ---------------------------------------------------------------- one line
    def line(self, ln: str) -> None:
        p = ln.split("|")
        if len(p) < 3:
            return
        kind = p[1]
        tags = [t for t in p[3:] if t.startswith("[")]
        frm = next((t[len("[from]"):].strip() for t in tags if t.startswith("[from]")), None)
        of_tok = next((t[len("[of]"):].strip() for t in tags if t.startswith("[of]")), None)
        if kind in ("switch", "drag", "replace"):
            sn = _side_and_name(p[2])
            if sn is None or len(p) < 4:
                return
            pos = p[2].split(":", 1)[0]
            self._leave(pos)
            sid = toid(p[3].split(",")[0])
            self.nick[sn] = sid
            self.active[pos] = sn
            if sn[0] == self.opp:
                new = self.out.mons.setdefault(sid, MonReveal(species=sid))
                if len(p) > 4:
                    new.hp = p[4]
            return
        m: Optional[MonReveal]
        k: Optional[Tuple[str, str]]
        subj = self._mon(p[2])
        if kind in ("-damage", "-heal", "-sethp") and subj is not None and len(p) > 3:
            m = self._opp(subj)
            if m is not None:
                m.hp = p[3]
        if kind == "-transform" and subj is not None:
            self.transformed.add(subj)
        if kind == "move" and subj is not None and len(p) > 3:
            self._move(subj, p[3], frm)
        elif kind == "-singleturn" and subj is not None and len(p) > 3 and p[3].startswith("move: "):
            self._own_move(subj, p[3][len("move: "):])                # Focus Punch's tightening
        elif kind == "cant" and subj is not None and len(p) > 4 and p[4].strip():
            self._own_move(subj, p[4].replace("move: ", ""))          # the move it chose and could not use
        # ---- Mimic: a copied move is not its own
        if kind in ("-activate", "-start") and subj is not None and len(p) > 4 and toid(p[3]) in ("movemimic", "mimic"):
            self.mimic.setdefault(subj, set()).add(toid(p[4]))
        # ---- items
        trick_line = kind in ("-item", "-enditem") and toid(frm or "") in ("movetrick", "moveswitcheroo")
        if self.trick and not trick_line:
            self._end_trick()
        if kind == "-activate" and len(p) > 3 and toid(p[3]) in ("movetrick", "moveswitcheroo"):
            self.trick = [k for k in (subj, self._mon(of_tok) if of_tok else None) if k is not None]
        if trick_line and subj is not None and len(p) > 3:
            if kind == "-enditem":
                self._item(subj, p[3])                        # it LOST this item: its pre-trick item
            elif subj[0] != self.opp:
                for k in self.trick:                          # OUR mon received the opp partner's pre-trick item
                    if k[0] == self.opp:
                        self._item(k, p[3])
            # (an opp mon's -item here is what it RECEIVED: nothing about its original)
        elif kind == "-item" and subj is not None and len(p) > 3:
            src = toid(frm or "")
            if src in ("movethief", "movecovet", "movebestow"):
                m = self._opp(subj)
                if m is not None and m.item is None:
                    m.item_moved = True                       # it gained an item it did not start with
            elif src == "moverecycle" or frm is None:
                self._item(subj, p[3])                        # its consumed item back / its own item named
            else:
                self._amb(f"-item [from] {frm}")
        elif kind == "-enditem" and subj is not None and len(p) > 3:
            self._item(subj, p[3])                            # eaten, knocked off, stolen: its item until now
            m = self._opp(subj)
            if m is not None and m.item is None:
                m.item_moved = True
        elif kind == "-activate" and subj is not None and len(p) > 3 and p[3].startswith("item: "):
            self._item(subj, p[3][len("item: "):])
        if frm is not None and frm.startswith("item:") and kind not in ("-item", "-enditem"):
            if of_tok is None or self._mon(of_tok) == subj:
                self._item(subj, frm[len("item:"):].strip())
            else:
                self._amb(f"{kind} [from] item [of]")
        # ---- abilities
        if kind == "-ability" and subj is not None and len(p) > 3:
            if frm is not None and toid(frm) == "abilitytrace":
                self._ability(subj, "Trace")                          # the tracer's original
                self._ability(self._mon(of_tok) if of_tok else None, p[3])   # the traced mon's
                m = self._opp(subj)
                if m is not None:
                    m.ability_moved = True
            elif frm is None:
                self._ability(subj, p[3])
            elif toid(frm).startswith("move"):
                for k in (subj, self._mon(of_tok) if of_tok else None):
                    m = self._opp(k)
                    if m is not None and m.ability is None:
                        m.ability_moved = True                     # Skill Swap / Role Play
            else:
                self._amb(f"-ability [from] {frm}")
            return
        for tok in p[3:4]:
            if tok.startswith("ability: ") and subj is not None:
                self._ability(subj, tok[len("ability: "):])        # cant / -start / -activate|M|ability: X
        if frm is not None and frm.startswith("ability:"):
            ab = frm[len("ability:"):].strip()
            of_mon = self._mon(of_tok) if of_tok else None
            if of_mon is None or of_mon == subj:
                owner = subj if subj is not None else of_mon
            else:
                rule = OWNER_OF_FROM_ABILITY.get(kind)
                owner = subj if rule == "subject" else of_mon if rule == "of" else None
                if owner is None:
                    self._amb(f"{kind} [from] ability [of]")
            if kind == "-weather":
                owner = of_mon
            self._ability(owner, ab)

    def _own_move(self, key: Tuple[str, str], name: str) -> None:
        m = self._opp(key)
        if m is None or key in self.transformed:
            return
        mid = toid(name)
        if not mid or mid == "struggle" or mid in self.mimic.get(key, set()):
            return
        m.moves.add(mid)

    def _move(self, key: Tuple[str, str], name: str, frm: Optional[str]) -> None:
        if frm is not None:
            caller = toid(frm.replace("move:", ""))
            if caller in NOT_OWN_CALLERS:
                return
        self._own_move(key, name)


def from_lines(lines: Sequence[str], viewer: int) -> Reveals:
    """The reveals of the protocol lines ``viewer`` (0 = p1, 1 = p2) was sent, in order."""
    r = _Reader(viewer)
    for ln in lines:
        r.line(ln)
    if r.trick:
        r._end_trick()
    return r.out


def viewer_lines(res: dict, viewer: int, upto_chunk: Optional[int] = None) -> List[str]:
    """Every protocol line ``viewer`` was sent in a ``core_events`` result, through chunk ``upto_chunk``."""
    chunks = res["chunks"] if upto_chunk is None else res["chunks"][: upto_chunk + 1]
    return [ln for side, lines in chunks if side == viewer for ln in lines]


def decision_chunk(res: dict, viewer: int, n: int) -> int:
    """The chunk index of the viewer's ``n``-th decision (``core_events --trackers``)."""
    ent = [e for e in res["trackers"][viewer] if "after" in e]
    if len(ent) <= n:
        raise IndexError(f"the viewer has {len(ent)} decisions, the root is decision {n}")
    return int(ent[n]["after"])


def at_decision(res: dict, viewer: int, n: int) -> Reveals:
    """The reveals at the viewer's ``n``-th decision (inclusive of the request it decided on)."""
    return from_lines(viewer_lines(res, viewer, decision_chunk(res, viewer, n)), viewer)


def own_team_request(res: dict, side_idx: int) -> List[dict]:
    """Every choice request side ``side_idx`` was sent (its own view of its team), in order."""
    out = []
    for side, lines in res["chunks"]:
        if side != side_idx:
            continue
        for ln in lines:
            if ln.startswith("|request|") and ln[len("|request|"):].strip():
                out.append(json.loads(ln[len("|request|"):]))
    return out
