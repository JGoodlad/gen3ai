"""C-GAME: ONE completed packed team per seat of a battle (X50 rung 1b step 0; design §2).

* Every own fact the log REVEALS by the battle's end is kept: species (with the log's nickname, gender and level),
  moves, the original item and ability. The reveals are ``agents.info_levers.revealed``'s (the ONE parser of "what
  play revealed"), read from the OTHER seat, i.e. what the public log showed of this seat.
* Every other fact is drawn from the Smogon prior through the ``agents.gen3_data`` facade, by the honest world
  builder's draws (``main.value_search.world_draws``: ``Prior``, ``never_seen_species``, ``draw_moves``,
  ``new_set``): a never-seen species from the teammates prior given the revealed ones; moves from the move marginals
  given the revealed moves; item, ability, nature and EVs from the Smogon tables; a Hidden Power's type from the
  species' Hidden Power prior, its IVs Showdown's fill for that type.
* The team must be LEGAL: every set and the whole team pass Showdown's ``TeamValidator('gen3ou')``
  (``main.value_search.world_legality``), a drawn set redrawn on rejection. A REVEALED fact set the validator
  rejects cannot be redrawn: :class:`CompletionError` (the seat is excluded and counted).

The draw is deterministic in ``(battle, seat, attempt)`` (:func:`seat_rng`). The caller (``run``) re-draws with the
next attempt when the team does not explain the log (an own labelled action illegal under it), up to
:data:`MAX_ATTEMPTS`.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from agents.info_levers.packed import Spread, hp_ivs, is_hp, toid

#: draws of one seat's team before the seat is excluded (design §2)
MAX_ATTEMPTS = 32
#: redraws of one SEEN mon's set before its attempt fails on legality
SEEN_REDRAWS = 25


class CompletionError(RuntimeError):
    """A seat whose REVEALED facts no legal team can hold (or a log the completion cannot read)."""


@dataclass
class OwnSeen:
    """One own mon the log showed: its log identity and what play revealed of it."""
    nick: str
    sid: str
    gender: str
    level: int
    moves: List[str] = field(default_factory=list)       # revealed (a bare ``hiddenpower`` untyped)
    item: Optional[str] = None
    ability: Optional[str] = None


@dataclass
class SeatFacts:
    seat: str
    team_size: int
    seen: List[OwnSeen]                                   # in order of first appearance (the lead first)

    @property
    def genders(self) -> Dict[str, str]:
        return {m.nick: m.gender for m in self.seen}


def seat_rng(battle: str, seat: str, attempt: int) -> np.random.Generator:
    h = hashlib.sha256(f"x50r1b|{battle}|{seat}|{attempt}".encode()).hexdigest()
    return np.random.default_rng(int(h[:16], 16))


def seat_facts(lines: Sequence[str], seat: str) -> SeatFacts:
    """What the whole public log revealed of ``seat``'s team."""
    from agents.info_levers import revealed

    size = 6
    order: List[Tuple[str, str, str, int]] = []
    seen_nick: Dict[str, int] = {}
    for ln in lines:
        p = ln.split("|")
        if len(p) > 3 and p[1] == "teamsize" and p[2] == seat and p[3].strip().isdigit():
            size = int(p[3])
        if len(p) > 3 and p[1] in ("switch", "drag") and p[2].startswith(seat):
            nick = p[2].split(": ", 1)[1].strip() if ": " in p[2] else ""
            det = [x.strip() for x in p[3].split(",")]
            sid = toid(det[0])
            if nick in seen_nick:
                if order[seen_nick[nick]][1] != sid:
                    raise CompletionError(f"{seat}: nickname {nick!r} names two species")
                continue
            gender = next((x for x in det[1:] if x in ("M", "F")), "")
            lvl = next((int(x[1:]) for x in det[1:] if x.startswith("L") and x[1:].isdigit()), 100)
            seen_nick[nick] = len(order)
            order.append((nick, sid, gender, lvl))
    if len({o[1] for o in order}) != len(order):
        raise CompletionError(f"{seat}: two own nicknames share a species")
    rev = revealed.from_lines(lines, 0 if seat == "p2" else 1)   # the OTHER seat's view of this one
    out = []
    for nick, sid, gender, lvl in order:
        m = rev.mons.get(sid)
        out.append(OwnSeen(nick=nick, sid=sid, gender=gender, level=lvl,
                           moves=sorted(m.moves) if m is not None else [],
                           item=m.item if m is not None else None,
                           ability=m.ability if m is not None else None))
    if len(out) > size:
        raise CompletionError(f"{seat}: {len(out)} own mons seen, team size {size}")
    return SeatFacts(seat=seat, team_size=size, seen=out)


class Completer:
    """The C-GAME draws for one worker process (the Smogon prior and one validator child)."""

    def __init__(self, prior=None, legality=None):
        from agents.info_levers import spec as SP
        from main.value_search import world_draws as WD
        from main.value_search.world_legality import Legality

        self.WD = WD
        self.P = prior if prior is not None else WD.Prior()
        self.L = legality if legality is not None else Legality()
        self.spec = SP.parse("off")
        self.tally: Dict[str, int] = {}

    # ---------------------------------------------------------------- one seen mon
    def _seen_set(self, rng: np.random.Generator, m: OwnSeen, ban_items: Sequence[str] = ()) -> str:
        WD, P = self.WD, self.P
        rev = [toid(x) for x in m.moves if toid(x) != "struggle"]
        if len(rev) > 4:
            raise CompletionError(f"{m.nick}: {len(rev)} revealed moves")
        if m.sid not in P.usage and not P.moves(m.sid):
            raise CompletionError(f"{m.nick}: species {m.sid!r} has no Smogon prior")
        for _ in range(SEEN_REDRAWS):
            keep: Dict[int, str] = {}
            hp_type: Optional[str] = None
            for i, mv in enumerate(rev):
                if mv == "hiddenpower":
                    hp_type = WD.MX.draw(rng, P.hidden_power(m.sid)) or "dark"
                    keep[i] = f"hiddenpower{hp_type}"
                else:
                    keep[i] = mv
            got = WD.draw_moves(rng, P.moves(m.sid), keep, 4, None, [], hp_type)
            if got is None:
                self._count("seen_moves_none")
                continue
            moves = got[0]
            items = {k: v for k, v in P.items(m.sid).items() if k not in ban_items}
            item = m.item if m.item is not None else (WD.MX.draw(rng, items) or "")
            ability = m.ability if m.ability is not None else (WD.MX.draw(rng, P.ability(m.sid)) or "")
            key = WD.MX.draw(rng, P.spreads(m.sid)) or ("hardy", (84,) * 6)
            hp = [x for x in moves if is_hp(x)]
            spread = Spread(WD._nature_name(key[0]), key[1],
                            tuple(hp_ivs(hp[0][len("hiddenpower"):])) if hp else (31,) * 6)
            entry = self._entry(m.nick, P.name(m.sid), item, ability, moves, spread, m.gender, m.level)
            if self.L.set_ok(entry):
                return entry
            self._count("seen_set_illegal")
        # the revealed facts alone: is it the reveal that no legal set holds?
        raise CompletionError(f"{m.nick}: no legal set in {SEEN_REDRAWS} draws holds the revealed facts "
                              f"({','.join(rev)}; item {m.item}; ability {m.ability})")

    @staticmethod
    def _entry(nick: str, species: str, item: str, ability: str, moves: Sequence[str], spread: Spread,
               gender: str, level: int) -> str:
        f = [""] * 12
        f[0] = nick
        f[1] = species if toid(species) != toid(nick) else ""
        f[2] = "" if item in ("", "nothing") else item
        f[3] = ability
        f[4] = ",".join(moves)
        spread.write(f)
        f[7] = gender
        f[10] = "" if level == 100 else str(level)
        f[11] = "0" if "frustration" in moves else ""
        return "|".join(f)

    def _count(self, k: str) -> None:
        self.tally[k] = self.tally.get(k, 0) + 1

    # ---------------------------------------------------------------- the team
    def team(self, facts: SeatFacts, rng: np.random.Generator, keep: Optional[Dict[str, str]] = None,
             ban_items: Optional[Dict[str, Sequence[str]]] = None) -> Tuple[str, Dict[str, str]]:
        """One completed packed team (the lead first, then the seen mons in order, then the drawn ones) and its
        sets by key (a seen mon's nickname; ``#i`` for the i-th never-seen slot).

        ``keep``: sets of an earlier attempt to reuse (a REDRAW draws only the mons whose sets the log refuted:
        an own action's legality depends on the acting mon's set alone, so redrawing that set and keeping the rest
        samples the same distribution, the prior conditioned on the log, as redrawing the whole team).
        ``ban_items``: per nickname, items the log has REFUTED (a Choice Band whose lock the mon broke without
        switching): drawing from the prior without them is that conditioning, exactly."""
        WD, P = self.WD, self.P
        from agents.gen3_data import species

        keep = keep or {}
        ban = ban_items or {}
        sets: Dict[str, str] = {}
        for m in facts.seen:
            sets[m.nick] = keep[m.nick] if m.nick in keep else self._seen_set(rng, m, ban.get(m.nick, ()))
        ents = [sets[m.nick] for m in facts.seen]
        n_new = facts.team_size - len(facts.seen)
        if n_new > 0 and all(f"#{s}" in keep for s in range(n_new)):
            for s in range(n_new):
                sets[f"#{s}"] = keep[f"#{s}"]
                ents.append(keep[f"#{s}"])
        elif n_new > 0:
            slots = list(range(n_new))
            got = WD.never_seen_species(rng, P, slots, {s: "" for s in slots}, {m.sid for m in facts.seen},
                                        self.spec)
            if got is None:
                raise CompletionError(f"{facts.seat}: no never-seen species could be drawn")
            for s in slots:
                sid = got[0][s]
                ns = WD.new_set(rng, P, sid, None, self.spec, legal=self.L.set_ok, tally=self.tally)
                if ns is None:
                    raise CompletionError(f"{facts.seat}: no legal set for a never-seen {sid}")
                f = ns[0].split("|")
                f += [""] * (12 - len(f))
                sd = species.get(sid)
                fixed = sd.gender if sd is not None else None
                f[7] = fixed if fixed in ("M", "F") else ("" if fixed == "N" else ("M" if rng.random() < 0.5 else "F"))
                if "frustration" in f[4].split(","):
                    f[11] = "0"
                sets[f"#{s}"] = "|".join(f)
                ents.append(sets[f"#{s}"])
        team = "]".join(ents)
        probs = self.L.team_problems([team])[0]
        if probs:
            self._count("team_illegal")
            raise TeamIllegal("; ".join(probs[:3]))
        return team, sets


class TeamIllegal(RuntimeError):
    """A drawn team the validator rejects as a whole (a clause): the attempt is redrawn."""
