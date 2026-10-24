"""THE OWN SIDE, COMPLETED (X50 rung 1b step 0, F-R1-1's fix: ``design_x50_rung1b_human_opponent_model.md`` §2).

Given ONE packed team for the viewer's seat (C-GAME's completed team, or our trainee's TRUE team for the S-own-true
control), this module turns a PUBLIC log into what the viewer's OWN stream would have shown it:

* :meth:`OwnTracker.line` rewrites each public line that names one of the viewer's own mons so its HP figure is
  EXACT (the log's percent x the team's max HP, the integer the percent display could have come from; ±1 HP,
  F-1b-9), as the seat's own stream writes it;
* :meth:`OwnTracker.request` synthesises the decision's ``|request|`` the way the simulator writes one: the WHOLE
  roster (details, condition, stats, moves, ability, the CURRENT item), in the simulator's slot order (a switch swaps
  the incoming mon into slot 1), the active's four moves at their tracked PP with the legality the public lines
  determine (Choice Band lock, Encore, Disable, Taunt, Torment, a locked / charging / recharging move, 0 PP,
  Struggle when nothing is left), and the trapping flags. The reader takes it as a REAL request (no spectator flag):
  the reader's own side is then the team's, which is the point of the fix.

The legality is the tracker's reading of public lines only. Where a turn's own action shows that the active was
LOCKED into its move (``[from]lockedmove``, a charge turn's release, a recharge ``cant``), the decision's request
is restricted to that move: the player's real request said so, and the action itself is in the log.

:func:`viewer_actions` reads the viewer's OWN choice at each decision (use (b)'s label) and :func:`action_index`
maps it onto the frame's action tokens. Pure text + the gen-3 data facade; no reader, no randomness.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from agents.info_levers.packed import parse_set, set_evs, set_ivs, toid, typed_moves

STATS = ("hp", "atk", "def", "spa", "spd", "spe")
#: moves whose continuation turns the log writes as ``[from]lockedmove`` (the request then lists that move alone)
LOCKING = frozenset({"outrage", "thrash", "petaldance", "rollout", "iceball", "uproar", "bide"})
#: two-turn moves (a ``[still]`` charge line + ``-prepare``, then the release next turn)
CHARGING = frozenset({"solarbeam", "razorwind", "skullbash", "skyattack", "fly", "dig", "dive", "bounce"})
RECHARGE = frozenset({"hyperbeam", "blastburn", "frenzyplant", "hydrocannon"})
#: a ``[from] X`` on an own ``|move|`` line whose move is CALLED from the user's (or another mon's) set: the caller's
#: line came first and paid the PP. ``[from] Pursuit`` (the switch-in hit) and ``[from] lockedmove`` are not calls.
CALLERS = frozenset({"sleeptalk", "assist", "metronome", "mirrormove", "naturepower", "snatch", "magiccoat",
                     "copycat", "mefirst"})
#: targets that never aim at a foe (Pressure does not deduct for them)
SELF_TARGETS = frozenset({"self", "allySide", "allyTeam", "adjacentAlly", "adjacentAllyOrSelf"})
#: line kinds whose ``p1: <name>`` token is the SIDE (the player), not a mon
SIDE_KINDS = frozenset({"-sidestart", "-sideend", "-swapsideconditions", "player", "teamsize"})
_HP_TOK = re.compile(r"^(\d+)/(\d+)((?: [a-z]+)?)$")


class OwnError(RuntimeError):
    """A public line the own-side tracker cannot place on the completed team (an unknown own mon, a bad HP token)."""


# ------------------------------------------------------------------------------------------------ the team

@dataclass
class SetInfo:
    """One packed set as the simulator would present it."""
    nick: str
    species: str                 # display name
    sid: str
    item: str
    ability: str
    moves: List[str]             # typed ids (``hiddenpowergrass``)
    stats: Dict[str, int]
    level: int
    gender: str                  # "M" / "F" / ""
    happiness: int = 255

    @property
    def details(self) -> str:
        d = self.species
        if self.level != 100:
            d += f", L{self.level}"
        if self.gender in ("M", "F"):
            d += f", {self.gender}"
        return d


def gen3_stats(sid: str, nature: str, evs: Sequence[int], ivs: Sequence[int], level: int = 100) -> Dict[str, int]:
    """The gen-3 stats of a set (the simulator's integer formula; Shedinja's HP is 1)."""
    from agents.gen3_data import natures, species

    sd = species.get(sid)
    if sd is None:
        raise OwnError(f"species {sid!r} has no data")
    mult = natures.multipliers().get(toid(nature) or "serious") or {}
    out: Dict[str, int] = {}
    for i, st in enumerate(STATS):
        b = int(sd.base_stats[st])
        core = (2 * b + int(ivs[i]) + int(evs[i]) // 4) * level // 100
        if st == "hp":
            out[st] = 1 if sid == "shedinja" else core + level + 10
        else:
            v = core + 5
            m = float(mult.get(st, 1.0))
            out[st] = v * 11 // 10 if m > 1.0 else v * 9 // 10 if m < 1.0 else v
    return out


def parse_team(packed: str, genders: Optional[Dict[str, str]] = None) -> List[SetInfo]:
    """A packed team → its sets (``genders``: nickname → the log's gender, for mons the log showed)."""
    from agents.gen3_data import species

    out = []
    for ent in (e for e in packed.split("]") if e.strip()):
        f = parse_set(ent)
        sp_raw = f[1] or f[0]
        sid = toid(sp_raw)
        sd = species.get(sid)
        if sd is None:
            raise OwnError(f"species {sp_raw!r} has no data")
        nick = f[0] or sd.name
        lvl = int(f[10]) if f[10].strip().isdigit() else 100
        g = (genders or {}).get(nick, "") or (f[7].rstrip("*") if f[7] else "") or (sd.gender or "")
        hap = int(f[11]) if f[11].strip().isdigit() else 255
        out.append(SetInfo(nick=nick, species=sd.name, sid=sid, item=toid(f[2]), ability=toid(f[3]),
                           moves=typed_moves(f), stats=gen3_stats(sid, f[5], set_evs(f), set_ivs(f), lvl),
                           level=lvl, gender=g if g in ("M", "F") else "", happiness=hap))
    return out


def happiness_power(mid: str, happiness: int) -> Optional[int]:
    """Return / Frustration's base power at ``happiness`` (the simulator writes it into the request), else None."""
    if mid == "return":
        return max(1, happiness * 10 // 25)
    if mid == "frustration":
        return max(1, (255 - happiness) * 10 // 25)
    return None


def side_move_id(mid: str, happiness: int) -> str:
    """A move as the request's ``side.pokemon[].moves`` lists it (``return102``; a typed Hidden Power as is)."""
    pw = happiness_power(mid, happiness)
    return f"{mid}{pw}" if pw is not None else mid


def _move_row(mid: str, happiness: int = 255) -> Tuple[str, str, int, str]:
    """(request display name, request id, max PP, target) of a typed move id."""
    from agents.gen3_data import moves

    if mid.startswith("hiddenpower") and mid != "hiddenpower":
        t = mid[len("hiddenpower"):]
        raw = moves.raw()["hiddenpower"]
        return f"Hidden Power {t[:1].upper() + t[1:]} 70", "hiddenpower", int(raw["pp"]) * 8 // 5, raw["target"]
    raw = moves.raw().get(mid)
    if raw is None:
        raise OwnError(f"move {mid!r} has no data")
    pp = int(raw["pp"])
    pw = happiness_power(mid, happiness)
    name = raw["name"] if pw is None else f"{raw['name']} {pw}"
    return name, mid, pp if raw.get("noPPBoosts") else pp * 8 // 5, raw["target"]


def _is_status_move(mid: str) -> bool:
    from agents.gen3_data import moves

    raw = moves.raw().get("hiddenpower" if mid.startswith("hiddenpower") else mid) or {}
    return raw.get("category") == "Status"


# ------------------------------------------------------------------------------------------------ the tracker

@dataclass
class MonState:
    s: SetInfo
    hp: int
    status: str = ""
    item: str = ""
    pp: List[int] = field(default_factory=list)
    maxpp: List[int] = field(default_factory=list)


@dataclass
class Volatile:
    """The ACTIVE mon's legality-relevant volatiles (cleared when it leaves)."""
    last_move: Optional[str] = None
    choice: Optional[str] = None
    encore: bool = False
    disabled: Optional[str] = None
    taunt: bool = False
    torment: bool = False
    trapped: bool = False


def _subject(tok: str) -> Optional[Tuple[str, str]]:
    head, sep, name = tok.partition(": ")
    if not sep or len(head) < 2 or head[0] != "p" or head[1] not in "12":
        return None
    return head[:2], name.strip()


def percent_to_hp(pct: int, maxhp: int) -> int:
    """The exact HP a percent display came from (Showdown's HP Percentage Mod: ceil(100 hp / max), 99 for any
    hp < max that rounds to 100, and 1 for any hp > 0): the MIDDLE integer of that preimage (the least-error choice; ±1 HP on a 3-wide preimage)."""
    if pct <= 0:
        return 0
    want = pct * maxhp / 100.0
    cand = [h for h in range(max(1, int(want) - maxhp // 100 - 3), min(maxhp, int(want) + 3) + 1)
            if _shown(h, maxhp) == pct]
    if not cand:
        return max(1, min(maxhp, int(round(want))))
    return cand[(len(cand) - 1) // 2]


def _shown(hp: int, maxhp: int) -> int:
    if hp <= 0:
        return 0
    p = -(-100 * hp // maxhp)
    return 99 if p == 100 and hp < maxhp else max(1, p)


class OwnTracker:
    """The viewer's own side through a public log, on a known team (module docstring)."""

    def __init__(self, seat: str, name: str, team: List[SetInfo]):
        self.seat, self.name = seat, name
        self.mons: Dict[str, MonState] = {}
        for s in team:
            rows = [_move_row(m) for m in s.moves]
            self.mons[s.nick] = MonState(s=s, hp=s.stats["hp"], item=s.item, pp=[r[2] for r in rows],
                                         maxpp=[r[2] for r in rows])
        self.order: List[str] = [s.nick for s in team]     # the simulator's slot order (slot 1 = the active)
        self.active: Optional[str] = None
        self.vol = Volatile()
        self.foe_pressure = False
        self.foe_active: Optional[str] = None               # the foe's species id
        self.foe_abilities: Dict[str, str] = {}             # foe nickname -> revealed ability
        self.foe_nick: Optional[str] = None
        self.rqid = 0
        self._pending: List[str] = []

    # -------------------------------------------------------------------- lines
    def _own(self, tok: str, kind: str = "") -> Optional[MonState]:
        sn = _subject(tok)
        if sn is None or sn[0] != self.seat or kind in SIDE_KINDS:
            return None
        m = self.mons.get(sn[1])
        if m is None:
            raise OwnError(f"{self.seat}: the log names own mon {sn[1]!r}, which the team does not hold")
        return m

    def _hp_tok(self, m: MonState, tok: str) -> str:
        if tok.startswith("0 fnt") or tok == "0":
            m.hp = 0
            return "0 fnt"
        g = _HP_TOK.match(tok)
        if g is None:
            raise OwnError(f"an own HP token {tok!r}")
        num, den, st = int(g.group(1)), int(g.group(2)), g.group(3).strip()
        mx = m.s.stats["hp"]
        m.hp = num if den == mx else percent_to_hp(num, mx) if den == 100 else round(num * mx / den)
        m.status = st
        return f"{m.hp}/{mx}" + (f" {st}" if st else "")

    def lines(self, ln: str) -> List[str]:
        """Fold one public line; return the lines the viewer's OWN stream holds in its place: the line with its own
        HP exact, preceded by the own Pressure announce a switch-in owes (gen 3 writes it ``[silent]`` to the
        holder's stream only, after the batch of switches it belongs to)."""
        kind = ln.split("|")[1] if ln.count("|") >= 1 else ""
        out: List[str] = []
        if self._pending and kind not in ("switch", "drag"):
            out.extend(self._pending)
            self._pending = []
        out.append(self.line(ln))
        return out

    def flush(self) -> List[str]:
        """The private lines still owed at a write's end (a switch-in's Pressure announce)."""
        out, self._pending = self._pending, []
        return out

    def line(self, ln: str) -> str:
        """Fold one public line; return it as the viewer's own stream would hold it (own HP exact)."""
        p = ln.split("|")
        if len(p) < 3:
            return ln
        kind = p[1]
        tags = [t for t in p[3:] if t.startswith("[")]
        frm = next((t[len("[from]"):].strip() for t in tags if t.startswith("[from]")), None)
        sn = _subject(p[2])
        if sn is not None and sn[0] != self.seat:
            self._foe_line(kind, p, sn, frm, tags)
            return ln
        m = self._own(p[2], kind) if sn is not None else None
        if m is None:
            return ln
        if kind in ("switch", "drag") and len(p) > 4:
            self._switch_in(m)
            p[4] = self._hp_tok(m, p[4])
            if m.s.ability == "pressure":
                self._pending.append(f"|-ability|{p[2]}|Pressure|[silent]")
            return "|".join(p)
        if kind in ("-damage", "-heal", "-sethp") and len(p) > 3:
            p[3] = self._hp_tok(m, p[3])
            return "|".join(p)
        if kind == "faint":
            m.hp, m.status = 0, ""
        elif kind == "-status" and len(p) > 3:
            m.status = p[3]
        elif kind == "-curestatus":
            m.status = ""
        elif kind == "-cureteam":
            for x in self.mons.values():
                x.status = ""
        elif kind == "move" and len(p) > 3:
            self._own_move(m, toid(p[3]), frm, tags)
        elif kind == "-enditem" and len(p) > 3:
            m.item = ""
            if self.vol.choice is not None and m.s.nick == self.active:
                self.vol.choice = None
        elif kind == "-item" and len(p) > 3:
            m.item = toid(p[3])
            if m.s.nick == self.active:
                self.vol.choice = None
        elif kind == "-start" and len(p) > 3 and m.s.nick == self.active:
            eff = toid(p[3])
            if eff in ("encore", "moveencore"):
                self.vol.encore = True
            elif eff in ("disable", "movedisable") and len(p) > 4:
                self.vol.disabled = toid(p[4])
            elif eff in ("taunt", "movetaunt"):
                self.vol.taunt = True
            elif eff in ("torment", "movetorment"):
                self.vol.torment = True
        elif kind == "-end" and len(p) > 3 and m.s.nick == self.active:
            eff = toid(p[3])
            if eff in ("encore", "moveencore"):
                self.vol.encore = False
            elif eff in ("disable", "movedisable"):
                self.vol.disabled = None
            elif eff in ("taunt", "movetaunt"):
                self.vol.taunt = False
            elif eff in ("partiallytrapped",) or "[partiallytrapped]" in tags:
                self.vol.trapped = False
        elif kind == "-activate" and len(p) > 3 and m.s.nick == self.active:
            eff = toid(p[3])
            if eff in ("trapped", "movemeanlook", "movespiderweb", "moveblock"):
                self.vol.trapped = True
            elif eff in ("movewrap", "movebind", "movefirespin", "moveclamp", "movewhirlpool", "movesandtomb"):
                self.vol.trapped = True
        return ln

    def _foe_line(self, kind: str, p: List[str], sn: Tuple[str, str], frm: Optional[str], tags: List[str]) -> None:
        if kind in ("switch", "drag") and len(p) > 3:
            self.foe_nick = sn[1]
            self.foe_active = toid(p[3].split(",")[0])
            self.foe_pressure = self._pressure(sn[1], self.foe_active)
            if self.active is not None:
                self.vol.trapped = False        # a foe's trap (Mean Look, a binding move) ends with its user
        elif kind == "-ability" and len(p) > 3 and frm is None:
            self.foe_abilities[sn[1]] = toid(p[3])
            if sn[1] == self.foe_nick and toid(p[3]) == "pressure":
                self.foe_pressure = True

    def _pressure(self, nick: str, sid: str) -> bool:
        """Does the foe's active hold Pressure? Gen 3 never announces it (a ``[silent]`` line the foe alone sees),
        so: its revealed ability when one is known, else the Smogon prior's majority for its species."""
        from agents.gen3_data import priors

        ab = self.foe_abilities.get(nick)
        if ab:
            return ab == "pressure"
        return float(priors.ability(sid).get("pressure", 0.0)) > 0.5

    def _switch_in(self, m: MonState) -> None:
        if self.active is not None and self.active != m.s.nick:
            i, j = self.order.index(self.active), self.order.index(m.s.nick)
            self.order[i], self.order[j] = self.order[j], self.order[i]
        elif self.active is None:
            j = self.order.index(m.s.nick)
            self.order[0], self.order[j] = self.order[j], self.order[0]
        self.active = m.s.nick
        self.vol = Volatile()

    def _own_move(self, m: MonState, mid: str, frm: Optional[str], tags: List[str]) -> None:
        if m.s.nick != self.active:
            return
        src = toid(frm.replace("move:", "")) if frm is not None else ""
        if src in CALLERS:
            return
        if src == "lockedmove":
            self.vol.last_move = mid
            return
        idx = self._slot(m, mid)
        if idx is None:
            return
        cost = 1
        target = _move_row(m.s.moves[idx])[3]
        if mid == "curse" and not self._ghost(m):
            target = "self"                     # a non-Ghost's Curse targets itself (no Pressure deduction)
        if self.foe_pressure and target not in SELF_TARGETS:
            cost = 2
        m.pp[idx] = max(0, m.pp[idx] - cost)
        self.vol.last_move = m.s.moves[idx]
        if m.item == "choiceband" and self.vol.choice is None:
            self.vol.choice = m.s.moves[idx]

    @staticmethod
    def _ghost(m: MonState) -> bool:
        from agents.gen3_data import species

        sd = species.get(m.s.sid)
        return sd is not None and "GHOST" in sd.types

    @staticmethod
    def _slot(m: MonState, mid: str) -> Optional[int]:
        for i, x in enumerate(m.s.moves):
            if x == mid or (mid == "hiddenpower" and x.startswith("hiddenpower")):
                return i
        return None

    # -------------------------------------------------------------------- the request
    def request(self, force: bool, locked: Optional[str] = None) -> Tuple[str, Dict[str, int]]:
        """The decision's ``|request|`` line, and the action map {``move:<typed id>`` / ``switch:<nick>``:
        the choice index Showdown's token names (``move N`` / ``switch N``)}. ``locked``: the move the active was
        locked into (from the turn's own action), or ``"recharge"``."""
        self.rqid += 1
        side = {"name": self.name, "id": self.seat, "pokemon": [self._mon_json(self.mons[n]) for n in self.order]}
        acts: Dict[str, int] = {}
        for i, n in enumerate(self.order):
            mm = self.mons[n]
            if n != self.active and mm.hp > 0:
                acts[f"switch:{n}"] = i + 1
        req: Dict[str, object] = {}
        if force:
            req = {"forceSwitch": [True], "side": side, "noCancel": True, "rqid": self.rqid}
            return "|request|" + json.dumps(req, separators=(",", ":"), ensure_ascii=False), acts
        a = self.mons[self.active] if self.active else None
        if a is None:
            raise OwnError(f"{self.seat}: a turn decision with no own active")
        if locked is not None:
            if locked == "recharge":
                one = {"move": "Recharge", "id": "recharge"}
                acts["move:recharge"] = 1
            else:
                i = self._slot(a, locked)
                if i is None:
                    raise OwnError(f"{self.seat}: locked into {locked!r}, which {a.s.nick} does not hold")
                name, rid, _, _ = _move_row(a.s.moves[i], a.s.happiness)
                one = {"move": name, "id": rid}
                acts[f"move:{a.s.moves[i]}"] = 1
            req = {"active": [{"moves": [one], "trapped": True}], "side": side, "rqid": self.rqid}
            return "|request|" + json.dumps(req, separators=(",", ":"), ensure_ascii=False), acts
        moves = []
        for i, mid in enumerate(a.s.moves):
            name, rid, mx, target = _move_row(mid, a.s.happiness)
            dis = self._disabled(a, i, mid)
            moves.append({"move": name, "id": rid, "pp": a.pp[i], "maxpp": a.maxpp[i], "target": target,
                          "disabled": dis})
        if all(m["disabled"] for m in moves):
            moves = [{"move": "Struggle", "id": "struggle", "target": "randomNormal", "disabled": False}]
            acts["move:struggle"] = 1
        else:
            for i, m in enumerate(moves):
                if not m["disabled"]:
                    acts[f"move:{a.s.moves[i]}"] = i + 1
        active: Dict[str, object] = {"moves": moves}
        trapped = self.vol.trapped
        if trapped:
            active["trapped"] = True
            acts = {k: v for k, v in acts.items() if not k.startswith("switch:")}
        elif self._maybe_trapped(a):
            active["maybeTrapped"] = True
        req = {"active": [active], "side": side, "rqid": self.rqid}
        return "|request|" + json.dumps(req, separators=(",", ":"), ensure_ascii=False), acts

    def _disabled(self, a: MonState, i: int, mid: str) -> bool:
        if a.pp[i] <= 0:
            return True
        v = self.vol
        if v.choice is not None and mid != v.choice:
            return True
        if v.encore and v.last_move is not None and mid != v.last_move:
            return True
        if v.disabled is not None and (mid == v.disabled or (v.disabled == "hiddenpower" and mid.startswith("hiddenpower"))):
            return True
        if v.taunt and _is_status_move(mid):
            return True
        if v.torment and v.last_move is not None and mid == v.last_move:
            return True
        return False

    def _maybe_trapped(self, a: MonState) -> bool:
        """The simulator's ``maybeTrapped``: the foe's species MAY hold a trapping ability that applies."""
        from agents.gen3_data import priors, species

        if self.foe_active is None:
            return False
        ab = self.foe_abilities.get(self.foe_nick or "")
        cands = {ab} if ab else set(priors.ability(self.foe_active))
        sd = species.get(a.s.sid)
        types = set(sd.types) if sd is not None else set()
        if "shadowtag" in cands:
            return True
        if "arenatrap" in cands and "FLYING" not in types and a.s.ability != "levitate":
            return True
        if "magnetpull" in cands and "STEEL" in types:
            return True
        return False

    def _mon_json(self, m: MonState) -> dict:
        mx = m.s.stats["hp"]
        cond = "0 fnt" if m.hp <= 0 else f"{m.hp}/{mx}" + (f" {m.status}" if m.status else "")
        return {"ident": f"{self.seat}: {m.s.nick}", "details": m.s.details, "condition": cond,
                "active": m.s.nick == self.active,
                "stats": {k: m.s.stats[k] for k in STATS[1:]},
                "moves": [side_move_id(x, m.s.happiness) for x in m.s.moves],
                "baseAbility": m.s.ability, "item": m.item, "pokeball": "pokeball"}


# ------------------------------------------------------------------------------------------------ actions

@dataclass(frozen=True)
class OwnAction:
    """The viewer's own choice at one decision: ``kind`` ``move`` (``what`` = the move id the log names, the
    CALLER for a called move) / ``switch`` (``what`` = the nickname) / ``none`` (a hidden or absent choice).
    ``locked``: the turn's move was a continuation the request restricted to it (``recharge`` for a recharge)."""
    kind: str
    what: str = ""
    locked: Optional[str] = None


def viewer_actions(lines: Sequence[str], seat: str, points: Sequence[Tuple[int, str]]) -> List[OwnAction]:
    """The viewer's own action in each decision's window (``points[k]`` to ``points[k + 1]``, the last to the end)."""
    out: List[OwnAction] = []
    for k, (pos, kind) in enumerate(points):
        end = points[k + 1][0] if k + 1 < len(points) else len(lines)
        act = OwnAction("none")
        for ln in lines[pos:end]:
            p = ln.split("|")
            if len(p) < 3:
                continue
            sn = _subject(p[2])
            if sn is None or sn[0] != seat:
                continue
            kw = p[1]
            if kw == "switch":
                act = OwnAction("switch", sn[1])
                break
            if kind == "forced":
                continue
            if kw == "move" and len(p) > 3:
                tags = [t for t in p[3:] if t.startswith("[")]
                frm = next((t[len("[from]"):].strip() for t in tags if t.startswith("[from]")), None)
                mid = toid(p[3])
                src = toid(frm.replace("move:", "")) if frm is not None else ""
                if src == "lockedmove":
                    act = OwnAction("move", mid, locked=mid)
                elif src in CALLERS:
                    continue                    # a called move: its caller's line came first
                elif mid == "struggle":
                    act = OwnAction("move", "struggle")
                else:
                    act = OwnAction("move", mid)
                break
            if kw == "cant":
                why = toid(p[3]) if len(p) > 3 else ""
                act = OwnAction("none", locked="recharge" if why == "recharge" else None)
                break
            if kw == "faint":
                break
        out.append(act)
    return out


def charge_locks(lines: Sequence[str], seat: str, points: Sequence[Tuple[int, str]]) -> List[Optional[str]]:
    """Per decision, the two-turn move the active was CHARGING when it opened: a ``-prepare`` by the own active in
    the previous turn's window that did not release in that window (Solar Beam in sun fires at once), with no own
    switch or faint after it. Its release is the decision's move; the request lists that move alone."""
    out: List[Optional[str]] = []
    pending: Optional[Tuple[str, str]] = None           # (nickname, move) charging across the boundary
    for k, (pos, kind) in enumerate(points):
        out.append(pending[1] if (pending is not None and kind == "turn") else None)
        end = points[k + 1][0] if k + 1 < len(points) else len(lines)
        nxt: Optional[Tuple[str, str]] = None
        for ln in lines[pos:end]:
            p = ln.split("|")
            if len(p) < 3:
                continue
            sn = _subject(p[2])
            if sn is None or sn[0] != seat:
                continue
            if p[1] == "-prepare" and len(p) > 3:
                nxt = (sn[1], toid(p[3]))
            elif p[1] in ("move", "-anim") and len(p) > 3 and nxt is not None and sn[1] == nxt[0] \
                    and toid(p[3]) == nxt[1] and "[still]" not in p[4:]:
                nxt = None                                # released in the same window (sun: the `-anim`)
            elif p[1] in ("switch", "drag", "faint"):
                nxt = None
        pending = nxt if kind == "turn" else None
    return out


def action_index(act: OwnAction, tokens: Dict[int, str], mask: Sequence[int], team_moves: Sequence[str]) -> int:
    """The 11-way action index of the viewer's own choice under the decision's frame (-1 = no label; -2 = the
    choice is not LEGAL under the synthesised request: the label-legality check's refusal).

    The reader's frame names each legal index by the choice string the server takes: ``move <request id>`` (a
    typed Hidden Power is ``move hiddenpowergrass``) and ``switch <nickname>``. A move the log names that is not in the
    completed set is illegal; so is a token the mask closes."""
    if act.kind == "none":
        return -1
    if act.kind == "move":
        if act.what == "struggle":
            want = "move struggle"
        else:
            hit = [m for m in team_moves if m == act.what or (act.what == "hiddenpower" and m.startswith("hiddenpower"))]
            if not hit:
                return -2
            want = "move " + hit[0]
    else:
        want = f"switch {act.what}"
    idx = [i for i, t in tokens.items() if t == want or (want.startswith("move hiddenpower")
                                                          and t.startswith("move hiddenpower"))]
    if len(idx) != 1 or not int(mask[idx[0]]):
        return -2
    return idx[0]
