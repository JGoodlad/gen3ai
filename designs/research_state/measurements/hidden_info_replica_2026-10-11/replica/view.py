"""What the SEARCHED side has seen of the opponent at the root, read from its OWN protocol stream.

Spec (design_hidden_info_binding.md §13.1 item 3 + Amendment 1): the opponent species that switched in or were
dragged in; for each, every move a public line names for that mon (not only ``|move|``); items / abilities named
by ``-item`` / ``-enditem`` / ``-ability`` or a ``[from] item:`` / ``[from] ability:`` tag.

This parser reads ONLY the searched side's lines (plus the searched side's own team, which it knows). It never
reads the opponent's true team. Every rule is written out below so a reader can check it line by line.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

#: ``|move|`` lines whose ``[from]`` names one of these were NOT drawn from the user's own set (a bounced /
#: copied / called move). Every other ``[from]`` (``lockedmove``, Pursuit's switch hit, Sleep Talk) is the
#: user's own move.
NOT_OWN_SET_FROM = {"magiccoat", "snatch", "mirrormove", "metronome", "assist", "naturepower", "mimic"}

#: Pseudo-moves a ``|move|`` line can name that are never a set's move.
NOT_A_SET_MOVE = {"struggle", "recharge"}

#: ``[from] ability:`` tags where BOTH the line's subject and its ``[of]`` mon could hold the ability: these
#: abilities belong to the SUBJECT (the mon the line is about). For every other ability the ``[of]`` mon owns
#: it (Rough Skin, Static, Synchronize, Sand Stream, Liquid Ooze, Cute Charm, …: the effect lands on someone
#: else).
SUBJECT_OWNED = {
    "voltabsorb", "waterabsorb", "naturalcure", "clearbody", "hypercutter", "keeneye", "shedskin", "speedboost",
    "raindish", "immunity", "insomnia", "limber", "oblivious", "owntempo", "waterveil", "vitalspirit",
    "magmaarmor", "flashfire", "levitate", "soundproof", "wonderguard", "sturdy", "suctioncups", "stickyhold",
    "shielddust", "innerfocus", "thickfat", "marvelscale", "guts", "earlybird", "truant", "pressure",
    "intimidate", "forecast", "colorchange", "lightningrod", "damp", "whitesmoke", "cloudnine", "airlock",
}


def to_id(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


@dataclass
class SeenMon:
    name: str                       # the protocol name (nickname or species)
    species: str                    # species id (from the switch / drag details)
    species_name: str               # the details' species display name
    gender: Optional[str]           # 'M' / 'F' as shown in the details, None if not shown
    first_line: int                 # index of its first switch / drag line
    shiny: bool = False             # ', shiny' in the details (public)
    level: Optional[int] = None     # ', L<n>' in the details (public; None = 100)
    moves: List[str] = field(default_factory=list)     # revealed move ids, in reveal order ('hiddenpower' = bare)
    item: Optional[str] = None      # the ORIGINAL item, when revealed ('' = known to hold nothing)
    ability: Optional[str] = None   # the ORIGINAL ability, when revealed
    acquired_item: bool = False     # it took an item from someone (Thief / Trick): later item lines are not original


@dataclass
class Seen:
    opp: str                                            # 'p1' / 'p2'
    mons: Dict[str, SeenMon] = field(default_factory=dict)   # keyed by the protocol name
    unparsed_moves: List[str] = field(default_factory=list)  # move-ish names that are not move ids (FINDING list)
    notes: List[str] = field(default_factory=list)

    def order(self) -> List[SeenMon]:
        return sorted(self.mons.values(), key=lambda m: m.first_line)


def _ident(tok: str) -> Optional[tuple]:
    """'p2a: Zapdos' -> ('p2', 'Zapdos'); anything else None."""
    m = re.match(r"^(p[12])[a-z]?: (.*)$", tok)
    return (m.group(1), m.group(2)) if m else None


def _tags(parts: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in parts:
        m = re.match(r"^\[(\w+)\]\s*(.*)$", p)
        if m:
            out[m.group(1)] = m.group(2)
    return out


def parse(lines: List[str], viewer: str, own_abilities: Dict[str, str], move_ids: Set[str],
          ability_support) -> Seen:
    """``lines``: the searched side's protocol lines up to the root, in order. ``viewer``: 'p1' / 'p2'.
    ``own_abilities``: the viewer's own mons' abilities by protocol name (the viewer knows its team).
    ``move_ids``: the gen3 move dex ids. ``ability_support(species_id) -> set`` of abilities the Smogon prior
    allows (used only to decide who owns a ``[from] ability:`` tag)."""
    opp = "p2" if viewer == "p1" else "p1"
    seen = Seen(opp=opp)
    transformed: Set[str] = set()
    active: Optional[str] = None

    def mon(name: str) -> Optional[SeenMon]:
        return seen.mons.get(name)

    def reveal_move(name: str, mv: str) -> None:
        m = mon(name)
        if m is None or name in transformed:
            return
        mid = to_id(mv.replace("move: ", ""))
        if not mid or mid in NOT_A_SET_MOVE:
            return
        if mid.startswith("hiddenpower"):
            mid = "hiddenpower"     # the protocol shows a bare Hidden Power; its TYPE is hidden (class M)
        elif mid not in move_ids:
            seen.unparsed_moves.append(mv)
            return
        if mid not in m.moves:
            m.moves.append(mid)

    def reveal_item(name: str, item: str, original: bool = True) -> None:
        m = mon(name)
        if m is None or m.acquired_item or not original:
            return
        iid = to_id(item.replace("item: ", ""))
        if m.item is None:
            m.item = iid

    def reveal_ability(name: str, ab: str) -> None:
        m = mon(name)
        if m is None:
            return
        aid = to_id(ab.replace("ability: ", ""))
        if m.ability is None and aid:
            m.ability = aid

    def owner_of_ability(ab: str, subj: Optional[tuple], of: Optional[tuple]) -> Optional[tuple]:
        aid = to_id(ab)
        cands = []
        for who in (subj, of):
            if who is None or who in cands:
                continue
            side, name = who
            if side == viewer:
                if own_abilities.get(name) == aid:
                    cands.append(who)
            else:
                m = mon(name)
                sup = ability_support(m.species) if m is not None else set()
                if m is not None and (not sup or aid in sup):
                    cands.append(who)
        if len(cands) == 1:
            return cands[0]
        if len(cands) == 2:
            return subj if aid in SUBJECT_OWNED else of
        return None

    for i, ln in enumerate(lines):
        if not ln.startswith("|"):
            continue
        parts = ln.split("|")[1:]
        if not parts:
            continue
        kind = parts[0]
        args = [p for p in parts[1:] if not p.startswith("[")]
        tags = _tags(parts[1:])
        subj = _ident(args[0]) if args else None
        of = _ident(tags["of"]) if "of" in tags else None
        # -------- who is on the field
        if kind in ("switch", "drag", "replace") and subj is not None and subj[0] == opp and len(args) >= 2:
            name = subj[1]
            det = args[1].split(", ")
            gender = next((d for d in det[1:] if d in ("M", "F")), None)
            shiny = "shiny" in det[1:]
            level = next((int(d[1:]) for d in det[1:] if re.match(r"^L\d+$", d)), None)
            if name not in seen.mons:
                seen.mons[name] = SeenMon(name=name, species=to_id(det[0]), species_name=det[0], gender=gender,
                                          first_line=i, shiny=shiny, level=level)
            if active is not None:
                transformed.discard(active)
            active = name
            continue
        if kind == "faint" and subj is not None and subj[0] == opp:
            transformed.discard(subj[1])
            continue
        if kind == "-transform" and subj is not None and subj[0] == opp:
            transformed.add(subj[1])
            seen.notes.append(f"transform:{subj[1]}")
            continue
        # -------- moves
        if subj is not None and subj[0] == opp:
            name = subj[1]
            if kind == "move" and len(args) >= 2:
                frm = to_id(tags.get("from", "").replace("move:", ""))
                if frm not in NOT_OWN_SET_FROM:
                    reveal_move(name, args[1])
            elif kind == "cant" and len(args) >= 3 and args[2]:
                reveal_move(name, args[2])
            elif kind in ("-singleturn", "-singlemove") and len(args) >= 2:
                if to_id(args[1].replace("move: ", "")) in move_ids:
                    reveal_move(name, args[1])
            elif kind == "-activate" and len(args) >= 3 and to_id(args[1].replace("move: ", "")) == "spite":
                reveal_move(name, args[2])
            elif kind == "-start" and len(args) >= 3 and to_id(args[1].replace("move: ", "")) == "disable":
                reveal_move(name, args[2])
        # -------- items
        if kind == "-item" and subj is not None and len(args) >= 2:
            frm = to_id(tags.get("from", "").replace("move:", ""))
            if frm in ("thief", "covet"):
                # the subject stole it: it held NOTHING before (gen 3 Thief needs an empty hand)
                if subj[0] == opp and mon(subj[1]) is not None:
                    m = mon(subj[1])
                    if m.item is None and not m.acquired_item:
                        m.item = ""
                    m.acquired_item = True
            elif frm in ("trick",):
                # the subject now holds what the [of] mon held: that is the [of] mon's ORIGINAL item
                if of is not None and of[0] == opp:
                    reveal_item(of[1], args[1])
                if subj[0] == opp and mon(subj[1]) is not None:
                    mon(subj[1]).acquired_item = True
            elif subj[0] == opp:
                reveal_item(subj[1], args[1])
        elif kind == "-enditem" and subj is not None and subj[0] == opp and len(args) >= 2:
            reveal_item(subj[1], args[1])
        # positional 'item: X' / 'ability: X' args belong to the subject
        if subj is not None and subj[0] == opp:
            for a in args[1:]:
                if a.startswith("item: "):
                    reveal_item(subj[1], a)
                elif a.startswith("ability: "):
                    reveal_ability(subj[1], a)
        if tags.get("from", "").startswith("item:"):
            who = of if of is not None else subj
            if who is not None and who[0] == opp:
                reveal_item(who[1], tags["from"])
        # -------- abilities
        if kind == "-ability" and subj is not None and len(args) >= 2:
            if to_id(tags.get("from", "")) == "abilitytrace":
                if subj[0] == opp:
                    reveal_ability(subj[1], "Trace")      # its own ability is Trace; the copied one is the [of]'s
                if of is not None and of[0] == opp:
                    reveal_ability(of[1], args[1])
            elif subj[0] == opp:
                reveal_ability(subj[1], args[1])
        elif tags.get("from", "").startswith("ability:"):
            ab = tags["from"][len("ability:"):].strip()
            who = owner_of_ability(ab, subj, of)
            if who is not None and who[0] == opp:
                reveal_ability(who[1], ab)
    return seen
