"""The TURN STORY: a battle's raw protocol folded into typed per-turn events + the board after each
turn, and the events grouped into ordered BEATS (`/game`; `designs/prober/battle_view_v2.md` §1,
`designs/prober/battle_viewer_ux_2026-10-09.md` §5).

Pure: protocol lines in, plain dicts out (JSON-ready, so the session returns them verbatim). The
existing `timeline.py` keeps ONE line per action and drops everything that is not an action; this
module keeps EVERY state-bearing line — the `-resisted`, the recoil, the Leftovers, the sand chip,
the hazard set — because that is the detail a reader of one turn wants, and the Rust core emits the
full protocol for every core trace.

Sides are named from the TRAINEE's seat: ``we`` / ``opp``. HP is always reported as % of max: our
side's lines carry exact points (``202/301``) and theirs hundredths (``92/100``); the fold keeps the
max it saw so a % is exact on both. A mon is NAMED BY ITS SPECIES in every sentence — a nickname
(``Leuphorie`` for Blissey) never reaches a reader; the event keeps the protocol name in ``mon``
and adds ``species``.

What the PUBLIC protocol revealed about each mon is tracked on the board: the moves it used, the
item a line announced (``[from] item: Leftovers``, ``-item``, ``-enditem``) and the ability
(``[from] ability: …``, ``-ability``) — the spectator half of `engine/perspective.py`.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

#: Protocol keywords that carry no battle fact (framing, rules, the clock).
_STRUCTURAL = frozenset({"", "init", "gametype", "player", "gen", "tier", "rule", "teamsize",
                         "start", "turn", "upkeep", "t:", "title", "j", "l", "c", "raw",
                         "seed", "teampreview", "clearpoke", "poke", "inactive", "inactiveoff",
                         "split", "debug", "chat", "message"})

#: Side conditions shown on the board, in display words (`-sidestart` names them).
SIDE_CONDITIONS = {"spikes": "Spikes", "reflect": "Reflect", "lightscreen": "Light Screen",
                   "safeguard": "Safeguard", "mist": "Mist"}

STATUS_WORDS = {"brn": "burned", "par": "paralyzed", "slp": "asleep", "frz": "frozen",
                "psn": "poisoned", "tox": "badly poisoned"}
CANT_WORDS = {"slp": "is fast asleep", "frz": "is frozen solid", "par": "is fully paralyzed",
              "flinch": "flinched", "recharge": "must recharge", "Taunt": "can't use that under Taunt",
              "Disable": "can't use the disabled move", "Imprison": "can't use the sealed move",
              "Focus Punch": "lost its focus", "nopp": "has no PP left", "attract": "is immobilized by love",
              "truant": "is loafing around"}
STAT_WORDS = {"atk": "Attack", "def": "Defense", "spa": "Sp. Atk", "spd": "Sp. Def", "spe": "Speed",
              "accuracy": "accuracy", "evasion": "evasion"}

#: A `[from]` source in the case a reader expects (the protocol mixes ids and display names).
SOURCE_WORDS = {"Recoil": "recoil", "recoil": "recoil", "Spikes": "Spikes", "spikes": "Spikes",
                "psn": "poison", "tox": "poison", "brn": "burn", "confusion": "confusion",
                "drain": "drain", "sandstorm": "Sandstorm", "hail": "Hail"}

_POS_RE = re.compile(r"^(p[12])[a-d]?:\s*(.*)$")
_HP_RE = re.compile(r"^(\d+)/(\d+)")


def _who(ref: str) -> Tuple[Optional[str], str]:
    """``"p1a: Gengar"`` → ``("p1", "Gengar")``; a bare side ``"p2: rhtwo"`` → ``("p2", "rhtwo")``."""
    m = _POS_RE.match(ref or "")
    return (m.group(1), m.group(2).strip()) if m else (None, (ref or "").strip())


def _tags(parts: Sequence[str]) -> Dict[str, str]:
    """The ``[from] X`` / ``[of] Y`` / ``[miss]`` tags of a protocol line."""
    out: Dict[str, str] = {}
    for p in parts:
        if p.startswith("[") and "]" in p:
            k, _, v = p[1:].partition("]")
            out[k.strip()] = v.strip()
    return out


def _source(tags: Dict[str, str]) -> Optional[str]:
    """``[from] item: Leftovers`` → ``"Leftovers"``; ``[from] Recoil`` → ``"recoil"``; ``[from] psn``
    → ``"poison"``; ``[from] Spikes`` → ``"Spikes"``."""
    src = tags.get("from")
    if not src:
        return None
    for pre in ("item: ", "ability: ", "move: "):
        if src.startswith(pre):
            return src[len(pre):]
    return SOURCE_WORDS.get(src, src)


def _parse_hp(cond: str) -> Tuple[Optional[float], Optional[int], Optional[str], bool]:
    """``"134/341 tox"`` → ``(134, 341, "tox", False)``; ``"0 fnt"`` → ``(0, None, None, True)``."""
    cond = (cond or "").strip()
    if not cond:
        return None, None, None, False
    if cond.startswith("0") and "fnt" in cond:
        return 0.0, None, None, True
    m = _HP_RE.match(cond)
    if not m:
        return None, None, None, False
    rest = cond[m.end():].strip()
    return float(m.group(1)), int(m.group(2)), (rest or None), False


class _Mon:
    __slots__ = ("name", "species", "hp", "max_hp", "status", "fainted", "moves", "item", "item_gone",
                 "ability", "first_turn")

    def __init__(self, name: str, species: str) -> None:
        self.name, self.species = name, species
        self.hp: Optional[float] = None
        self.max_hp: Optional[int] = None
        self.status = ""
        self.fainted = False
        self.moves: List[str] = []
        self.item: Optional[str] = None          # an item a PUBLIC line announced
        self.item_gone = False                   # … and since consumed / removed
        self.ability: Optional[str] = None       # an ability a PUBLIC line announced
        self.first_turn: Optional[int] = None    # the turn it first appeared on the field

    def pct(self) -> Optional[float]:
        if self.fainted:
            return 0.0
        if self.hp is None or not self.max_hp:
            return None
        return round(100.0 * self.hp / self.max_hp, 1)

    def view(self, active: bool) -> dict:
        return {"name": self.name, "species": self.species, "hp_pct": self.pct(),
                "hp": self.hp, "max_hp": self.max_hp,
                "status": self.status, "fainted": self.fainted, "active": active,
                "moves": list(self.moves), "item": self.item, "item_gone": self.item_gone,
                "ability": self.ability, "first_turn": self.first_turn}


class _Side:
    def __init__(self) -> None:
        self.mons: Dict[str, _Mon] = {}
        self.active: Optional[str] = None
        self.boosts: Dict[str, int] = {}
        self.conditions: Dict[str, int] = {}
        self.volatiles: List[str] = []
        self.team_size: Optional[int] = None

    def mon(self, name: str, species: Optional[str] = None) -> _Mon:
        m = self.mons.get(name)
        if m is None:
            m = self.mons[name] = _Mon(name, species or name)
        elif species:
            m.species = species
        return m

    def view(self) -> dict:
        return {"mons": [m.view(n == self.active) for n, m in self.mons.items()],
                "active": self.active, "boosts": dict(self.boosts),
                "conditions": dict(self.conditions), "volatiles": list(self.volatiles),
                "team_size": self.team_size}


class TurnFold:
    """Folds a battle's protocol, line by line, into typed events and the board."""

    def __init__(self, trainee_side: str = "p1",
                 our_team: Sequence[str] = ()) -> None:
        self.trainee = trainee_side
        self.sides = {"we": _Side(), "opp": _Side()}
        self.weather: Optional[str] = None
        self.turn = 0
        self.players: Dict[str, Optional[str]] = {}      # username -> we / opp
        for sp in our_team:
            if sp:
                self.sides["we"].mon(str(sp), str(sp))

    def _side(self, p: Optional[str]) -> Optional[str]:
        if p is None:
            return None
        return "we" if p == self.trainee else "opp"

    def species_of(self, side: Optional[str], name: str) -> str:
        """The SPECIES behind a protocol name (a nickname resolves through the switch line that
        brought it in); the name itself when the fold has not seen it."""
        if side in self.sides and name in self.sides[side].mons:
            return self.sides[side].mons[name].species
        return name

    def _pretty(self, side: Optional[str], name: str) -> str:
        sp = self.species_of(side, name)
        return {"we": f"our {sp}", "opp": f"their {sp}"}.get(side or "", sp)

    def board(self) -> dict:
        return {"we": self.sides["we"].view(), "opp": self.sides["opp"].view(),
                "weather": self.weather}

    # --------------------------------------------------------------------- one line
    def line(self, ln: str) -> Optional[dict]:
        """Apply one protocol line; return its EVENT (or None for a structural line)."""
        parts = ln.split("|")
        if len(parts) < 2:
            return None
        kw = parts[1]
        args = parts[2:]
        tags = _tags(args)
        if kw == "teamsize" and len(args) >= 2:
            s = self._side(args[0])
            if s:
                try:
                    self.sides[s].team_size = int(args[1])
                except ValueError:
                    pass
            return None
        if kw == "player" and len(args) >= 2 and args[1]:
            self.players[args[1]] = self._side(args[0])
            return None
        if kw in _STRUCTURAL:
            return None
        if kw in ("win", "tie"):
            if kw == "tie" or not args:
                return {"kind": "end", "side": None, "winner": None, "text": "the battle ended in a tie"}
            won = self.players.get(args[0])
            text = {"we": "we won the battle", "opp": "they won the battle"}.get(won or "", f"{args[0]} won the battle")
            return {"kind": "end", "side": None, "winner": won, "text": text}
        h = getattr(self, "_" + kw.lstrip("-").replace("-", "_"), None)
        if h is not None:
            ev = h(args, tags)
            if ev is not None:
                ev.setdefault("raw", ln)
                self._reveal(ev, args, tags)
            return ev
        side, name = _who(args[0]) if args else (None, "")
        s = self._side(side)
        return {"kind": "other", "side": s, "mon": name or None,
                "species": self.species_of(s, name) if name else None,
                "text": f"{kw.lstrip('-')}: " + " · ".join(a for a in args if a), "raw": ln}

    def _reveal(self, ev: dict, args, tags) -> None:
        """Record what a `[from] item: X` / `[from] ability: Y` tag PUBLICLY revealed, on its holder
        (the `[of]` mon when the line names one, else the line's subject), and stamp the event with
        its mon's species."""
        src = tags.get("from") or ""
        if src.startswith("item: ") or src.startswith("ability: "):
            ref = tags.get("of") or (args[0] if args else "")
            p, name = _who(ref)
            s = self._side(p)
            if s and name:
                mon = self.sides[s].mon(name)
                if src.startswith("item: "):
                    mon.item = src[len("item: "):]
                else:
                    mon.ability = src[len("ability: "):]
        if ev.get("mon") and "species" not in ev:
            ev["species"] = self.species_of(ev.get("side"), ev["mon"])

    # --------------------------------------------------------------------- handlers
    def _hp_change(self, kind: str, args, tags) -> dict:
        p, name = _who(args[0])
        s = self._side(p)
        mon = self.sides[s].mon(name) if s else None
        before = mon.pct() if mon else None
        hp, mx, status, fnt = _parse_hp(args[1] if len(args) > 1 else "")
        if mon is not None:
            if fnt:
                mon.hp = 0.0
            elif hp is not None:
                mon.hp, mon.max_hp = hp, (mx or mon.max_hp)
            if status is not None and status != "fnt":
                mon.status = status
        after = (0.0 if fnt else mon.pct()) if mon else None
        delta = (round(after - before, 1) if (after is not None and before is not None) else None)
        src = _source(tags)
        who = self._pretty(s, name)
        if kind == "damage":
            amt = f" {abs(delta):g}%" if delta is not None else ""
            text = f"{who} lost{amt}" + (f" from {src}" if src else "")
        elif kind == "heal":
            amt = f" {abs(delta):g}%" if delta is not None else ""
            text = f"{who} restored{amt}" + (f" ({src})" if src else "")
        else:
            text = f"{who}'s HP was set"
        if after is not None:
            text += f" — now {after:g}%"
        return {"kind": kind, "side": s, "mon": name, "species": self.species_of(s, name),
                "pct": delta, "hp_after": after, "hp_before": before, "source": src, "fainted": fnt,
                "text": text}

    def _damage(self, args, tags):
        return self._hp_change("damage", args, tags)

    def _heal(self, args, tags):
        return self._hp_change("heal", args, tags)

    def _sethp(self, args, tags):
        return self._hp_change("sethp", args, tags)

    def _switch(self, args, tags, kind: str = "switch"):
        p, name = _who(args[0])
        s = self._side(p)
        species = (args[1].split(",")[0].strip() if len(args) > 1 else name)
        if s is None:
            return None
        side = self.sides[s]
        prev = side.active
        prev_species = self.species_of(s, prev) if prev else None
        mon = side.mon(name, species)
        if mon.first_turn is None:
            mon.first_turn = self.turn
        hp, mx, status, fnt = _parse_hp(args[2] if len(args) > 2 else "")
        if hp is not None:
            mon.hp, mon.max_hp = hp, (mx or mon.max_hp)
        if status:
            mon.status = status
        side.active = name
        side.boosts = {}
        side.volatiles = []
        verb = "was dragged in" if kind == "drag" else ("came in" if prev is None else "switched in")
        out = f" for {prev_species}" if prev and prev != name and kind != "drag" else ""
        return {"kind": kind, "side": s, "mon": name, "species": species, "replaced": prev,
                "replaced_species": prev_species,
                "hp_after": mon.pct(), "text": f"{self._pretty(s, name)} {verb}{out}"}

    def _drag(self, args, tags):
        return self._switch(args, tags, kind="drag")

    def _replace(self, args, tags):
        return self._switch(args, tags, kind="switch")

    def _move(self, args, tags):
        p, name = _who(args[0])
        s = self._side(p)
        move = args[1] if len(args) > 1 else "?"
        tp, tname = _who(args[2]) if len(args) > 2 and args[2] and not args[2].startswith("[") else (None, "")
        ts = self._side(tp)
        if s:
            mon = self.sides[s].mon(name)
            if move not in mon.moves and not tags.get("from"):
                mon.moves.append(move)
        text = f"{self._pretty(s, name)} used {move}"
        if tname and (tname != name or ts != s):
            text += f" on {self._pretty(ts, tname)}"
        if "miss" in tags:
            text += " — it missed"
        # `[still]` only suppresses the animation (a failed Protect, a charge turn); a CHARGE is the
        # `-prepare` line, which says so on its own — so `[still]` alone claims nothing here.
        if tags.get("from"):
            text += f" (via {_source(tags)})"
        return {"kind": "move", "side": s, "mon": name, "species": self.species_of(s, name), "move": move,
                "target_side": ts, "target": tname or None,
                "target_species": self.species_of(ts, tname) if tname else None,
                "missed": "miss" in tags, "via": _source(tags) if tags.get("from") else None,
                "text": text}

    def _faint(self, args, tags):
        p, name = _who(args[0])
        s = self._side(p)
        if s:
            mon = self.sides[s].mon(name)
            mon.fainted, mon.hp = True, 0.0
        return {"kind": "faint", "side": s, "mon": name, "species": self.species_of(s, name),
                "text": f"{self._pretty(s, name)} fainted"}

    def _status(self, args, tags):
        p, name = _who(args[0])
        s = self._side(p)
        st = args[1] if len(args) > 1 else ""
        if s:
            self.sides[s].mon(name).status = st
        src = _source(tags)
        return {"kind": "status", "side": s, "mon": name, "status": st, "source": src,
                "text": f"{self._pretty(s, name)} is {STATUS_WORDS.get(st, st)}" + (f" ({src})" if src else "")}

    def _curestatus(self, args, tags):
        p, name = _who(args[0])
        s = self._side(p)
        st = args[1] if len(args) > 1 else ""
        if s:
            self.sides[s].mon(name).status = ""
        return {"kind": "cure", "side": s, "mon": name, "status": st,
                "text": f"{self._pretty(s, name)} is no longer {STATUS_WORDS.get(st, st)}"}

    def _cureteam(self, args, tags):
        p, name = _who(args[0])
        s = self._side(p)
        if s:
            for m in self.sides[s].mons.values():
                m.status = ""
        return {"kind": "cure", "side": s, "mon": name, "text": f"{self._pretty(s, name)} cured its team"}

    def _boost_change(self, args, tags, sign: int):
        p, name = _who(args[0])
        s = self._side(p)
        stat = args[1] if len(args) > 1 else ""
        try:
            n = int(args[2]) if len(args) > 2 else 0
        except ValueError:
            n = 0
        stage = None
        if s and self.sides[s].active == name:
            b = self.sides[s].boosts
            b[stat] = max(-6, min(6, b.get(stat, 0) + sign * n))
            stage = b[stat]
            if not b[stat]:
                b.pop(stat)
        word = STAT_WORDS.get(stat, stat)
        verb = ("rose" if sign > 0 else "fell") + (" sharply" if n == 2 else (" drastically" if n >= 3 else ""))
        if n == 0:
            verb = "won't go any " + ("higher" if sign > 0 else "lower")
        return {"kind": "boost" if sign > 0 else "unboost", "side": s, "mon": name, "stat": stat,
                "amount": sign * n, "stage": stage, "text": f"{self._pretty(s, name)}'s {word} {verb}"}

    def _boost(self, args, tags):
        return self._boost_change(args, tags, 1)

    def _unboost(self, args, tags):
        return self._boost_change(args, tags, -1)

    def _setboost(self, args, tags):
        p, name = _who(args[0])
        s = self._side(p)
        stat = args[1] if len(args) > 1 else ""
        try:
            n = int(args[2])
        except (IndexError, ValueError):
            n = 0
        if s:
            self.sides[s].boosts[stat] = n
        return {"kind": "boost", "side": s, "mon": name, "stat": stat, "amount": n, "stage": n,
                "text": f"{self._pretty(s, name)}'s {STAT_WORDS.get(stat, stat)} was set to {n:+d}"}

    def _clearallboost(self, args, tags):
        for side in self.sides.values():
            side.boosts = {}
        return {"kind": "boost", "side": None, "text": "every stat change was removed"}

    def _clearboost(self, args, tags):
        p, name = _who(args[0])
        s = self._side(p)
        if s:
            self.sides[s].boosts = {}
        return {"kind": "boost", "side": s, "mon": name, "text": f"{self._pretty(s, name)}'s stat changes were removed"}

    def _simple(self, kind: str, args, text_fn) -> dict:
        p, name = _who(args[0]) if args else (None, "")
        s = self._side(p)
        return {"kind": kind, "side": s, "mon": name or None, "text": text_fn(self._pretty(s, name))}

    def _crit(self, args, tags):
        return self._simple("crit", args, lambda w: f"a critical hit on {w}")

    def _supereffective(self, args, tags):
        return self._simple("supereffective", args, lambda w: f"it's super effective on {w}")

    def _resisted(self, args, tags):
        return self._simple("resisted", args, lambda w: f"it's not very effective on {w}")

    def _immune(self, args, tags):
        src = _source(tags)
        return self._simple("immune", args, lambda w: f"it doesn't affect {w}" + (f" ({src})" if src else ""))

    def _miss(self, args, tags):
        tgt = _who(args[1]) if len(args) > 1 else None
        ev = self._simple("miss", args, lambda w: f"{w}'s attack missed")
        if tgt and tgt[1]:
            ev["target"] = tgt[1]
            ev["target_side"] = self._side(tgt[0])
        return ev

    def _fail(self, args, tags):
        return self._simple("fail", args, lambda w: "but it failed")

    def _cant(self, args, tags):
        reason = args[1] if len(args) > 1 else ""
        return dict(self._simple("cant", args, lambda w: f"{w} {CANT_WORDS.get(reason, 'could not move (' + reason + ')')}"),
                    reason=reason)

    def _weather(self, args, tags):
        w = args[0] if args else "none"
        upkeep = "upkeep" in tags
        prev = self.weather
        self.weather = None if w == "none" else w
        if w == "none":
            return {"kind": "weather", "side": None, "weather": None, "text": f"the {prev or 'weather'} subsided"}
        if upkeep:
            return {"kind": "weather", "side": None, "weather": w, "upkeep": True,
                    "text": f"the {w} continues"}
        src = _source(tags)
        return {"kind": "weather", "side": None, "weather": w,
                "text": f"{w} started" + (f" ({src})" if src else "")}

    def _side_cond(self, args, start: bool):
        p, _ = _who(args[0]) if args else (None, "")
        s = self._side(p)
        raw = (args[1] if len(args) > 1 else "")
        cond = raw.split(":", 1)[-1].strip()
        key = re.sub(r"[^a-z]", "", cond.lower())
        word = SIDE_CONDITIONS.get(key, cond)
        if s:
            c = self.sides[s].conditions
            if start:
                c[key] = c.get(key, 0) + 1 if key == "spikes" else 1
            else:
                c.pop(key, None)
        where = {"we": "our side", "opp": "their side"}.get(s or "", "a side")
        kind = "hazard" if key == "spikes" else "screen"
        if start:
            text = f"{word} on {where}" + (f" (layer {self.sides[s].conditions.get('spikes')})"
                                            if key == "spikes" and s else "")
        else:
            text = f"{word} ended on {where}"
        return {"kind": kind, "side": s, "condition": key, "start": start, "text": text}

    def _sidestart(self, args, tags):
        return self._side_cond(args, True)

    def _sideend(self, args, tags):
        return self._side_cond(args, False)

    def _volatile(self, args, tags, start: bool):
        p, name = _who(args[0]) if args else (None, "")
        s = self._side(p)
        eff = (args[1] if len(args) > 1 else "").split(":", 1)[-1].strip()
        if s and self.sides[s].active == name:
            vol = self.sides[s].volatiles
            if start and eff not in vol:
                vol.append(eff)
            elif not start and eff in vol:
                vol.remove(eff)
        src = _source(tags)
        text = f"{self._pretty(s, name)}: {eff} " + ("started" if start else "ended") + (f" ({src})" if src else "")
        return {"kind": "volatile", "side": s, "mon": name, "effect": eff, "start": start, "text": text}

    def _start(self, args, tags):
        return self._volatile(args, tags, True)

    def _end(self, args, tags):
        return self._volatile(args, tags, False)

    def _item(self, args, tags):
        it = args[1] if len(args) > 1 else ""
        p, name = _who(args[0]) if args else (None, "")
        s = self._side(p)
        if s and name:
            mon = self.sides[s].mon(name)
            mon.item, mon.item_gone = it, False
        return dict(self._simple("item", args, lambda w: f"{w} has {it}"), item=it)

    def _enditem(self, args, tags):
        it = args[1] if len(args) > 1 else ""
        p, name = _who(args[0]) if args else (None, "")
        s = self._side(p)
        if s and name:
            mon = self.sides[s].mon(name)
            mon.item, mon.item_gone = it, True
        verb = "ate its" if "eat" in tags else "lost its"
        return dict(self._simple("item", args, lambda w: f"{w} {verb} {it}"), item=it, consumed=True)

    def _ability(self, args, tags):
        ab = args[1] if len(args) > 1 else ""
        p, name = _who(args[0]) if args else (None, "")
        s = self._side(p)
        if s and name and ab:
            self.sides[s].mon(name).ability = ab
        return dict(self._simple("ability", args, lambda w: f"{w}'s ability: {ab}"), ability=ab)

    def _activate(self, args, tags):
        eff = (args[1] if len(args) > 1 else "").split(":", 1)[-1].strip()
        return dict(self._simple("activate", args, lambda w: f"{w}: {eff}"), effect=eff)

    def _singleturn(self, args, tags):
        eff = (args[1] if len(args) > 1 else "").split(":", 1)[-1].strip()
        return dict(self._simple("activate", args, lambda w: f"{w} used {eff}"), effect=eff)

    def _prepare(self, args, tags):
        mv = args[1] if len(args) > 1 else ""
        return self._simple("activate", args, lambda w: f"{w} is preparing {mv}")

    def _mustrecharge(self, args, tags):
        return self._simple("activate", args, lambda w: f"{w} must recharge")

    def _hitcount(self, args, tags):
        n = args[1] if len(args) > 1 else "?"
        return dict(self._simple("hitcount", args, lambda w: f"{w} was hit {n} time(s)"), hits=n)

    def _transform(self, args, tags):
        return self._simple("activate", args, lambda w: f"{w} transformed")


def fold_turns(lines: Sequence[str], *, trainee_side: str = "p1",
               our_team: Sequence[str] = ()) -> List[dict]:
    """The whole battle, turn by turn: ``[{turn, events: [...], beats: [...], board: {...}}]``.

    Turn 0 is everything before ``|turn|1`` (the leads). Each turn's events are the lines from
    ``|turn|N`` up to (not including) ``|turn|N+1`` — the same window `protocol_for_turn` slices, so a
    decision at turn N reads the events its choice led to. ``board`` is the state at the END of the
    turn (after upkeep). ``beats`` groups the events into the ordered actions of the turn
    (`engine/turn_beats.py`); every event carries the index of its beat and that beat's phase."""
    from main.prober.engine.turn_beats import build_beats, turn_summary

    fold = TurnFold(trainee_side, our_team)
    out: List[dict] = [{"turn": 0, "events": [], "replace_boards": []}]
    raw: List[List[Tuple[str, Optional[dict]]]] = [[]]
    fainted: set = set()                 # sides whose active fainted and has not been replaced yet
    for ln in lines or ():
        if ln.startswith("|turn|"):
            out[-1]["board"] = fold.board()
            try:
                n = int(ln.split("|")[2])
            except (IndexError, ValueError):
                n = out[-1]["turn"] + 1
            fold.turn = n
            out.append({"turn": n, "events": [], "replace_boards": []})
            raw.append([])
            continue
        if ln.startswith("|switch|") and fainted:
            # A FORCED replacement: the board the replacing side decided on is THIS one (mid-turn,
            # after the faint) — not the start-of-turn board — so it is kept for that decision.
            p = fold._side(_who(ln.split("|")[2] if ln.count("|") >= 2 else "")[0])
            if p in fainted:
                out[-1]["replace_boards"].append({"side": p, "board": fold.board()})
                fainted.discard(p)
        ev = fold.line(ln)
        raw[-1].append((ln, ev))
        if ev is not None:
            out[-1]["events"].append(ev)
            if ev.get("kind") == "faint" and ev.get("side"):
                fainted.add(ev["side"])
    out[-1]["board"] = fold.board()
    for t, seq in zip(out, raw):
        t["beats"] = build_beats(seq, turn=int(t["turn"]))
        t["summary"] = turn_summary(t["beats"])
    return out


__all__ = ["CANT_WORDS", "SIDE_CONDITIONS", "SOURCE_WORDS", "STATUS_WORDS", "TurnFold", "fold_turns"]
