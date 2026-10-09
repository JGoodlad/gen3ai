"""The TURN STORY: a battle's raw protocol folded into typed per-turn events + the board after each
turn (`/game`'s left column and its turn panel; `designs/prober/battle_view_v2.md` §1).

Pure: protocol lines in, plain dicts out (JSON-ready, so the session returns them verbatim). The
existing `timeline.py` keeps ONE line per action and drops everything that is not an action; this
module keeps EVERY state-bearing line — the `-resisted`, the recoil, the Leftovers, the sand chip,
the hazard set — because that is the detail a reader of one turn wants, and the Rust core now emits
the full protocol for every core trace.

Sides are named from the TRAINEE's seat: ``we`` / ``opp``. HP is always reported as % of max: our
side's lines carry exact points (``202/301``) and theirs hundredths (``92/100``); the fold keeps the
max it saw so a % is exact on both.
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
    """``[from] item: Leftovers`` → ``"Leftovers"``; ``[from] Recoil`` → ``"recoil"``; …"""
    src = tags.get("from")
    if not src:
        return None
    for pre in ("item: ", "ability: ", "move: "):
        if src.startswith(pre):
            return src[len(pre):]
    return src.lower() if src in ("Recoil", "Spikes", "psn", "brn", "tox", "confusion") else src


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
    __slots__ = ("name", "species", "hp", "max_hp", "status", "fainted", "moves")

    def __init__(self, name: str, species: str) -> None:
        self.name, self.species = name, species
        self.hp: Optional[float] = None
        self.max_hp: Optional[int] = None
        self.status = ""
        self.fainted = False
        self.moves: List[str] = []

    def pct(self) -> Optional[float]:
        if self.fainted:
            return 0.0
        if self.hp is None or not self.max_hp:
            return None
        return round(100.0 * self.hp / self.max_hp, 1)

    def view(self, active: bool) -> dict:
        return {"name": self.name, "species": self.species, "hp_pct": self.pct(),
                "status": self.status, "fainted": self.fainted, "active": active,
                "moves": list(self.moves)}


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


def _pretty(side: Optional[str], name: str) -> str:
    return {"we": f"our {name}", "opp": f"their {name}"}.get(side or "", name)


class TurnFold:
    """Folds a battle's protocol, line by line, into typed events and the board."""

    def __init__(self, trainee_side: str = "p1",
                 our_team: Sequence[str] = ()) -> None:
        self.trainee = trainee_side
        self.sides = {"we": _Side(), "opp": _Side()}
        self.weather: Optional[str] = None
        for sp in our_team:
            if sp:
                self.sides["we"].mon(str(sp), str(sp))

    def _side(self, p: Optional[str]) -> Optional[str]:
        if p is None:
            return None
        return "we" if p == self.trainee else "opp"

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
        if kw in _STRUCTURAL:
            return None
        if kw in ("win", "tie"):
            return {"kind": "end", "side": None, "text": (f"{args[0]} won the battle" if kw == "win" and args
                                                         else "the battle ended in a tie")}
        h = getattr(self, "_" + kw.lstrip("-").replace("-", "_"), None)
        if h is not None:
            ev = h(args, tags)
            if ev is not None:
                ev.setdefault("raw", ln)
            return ev
        side, name = _who(args[0]) if args else (None, "")
        return {"kind": "other", "side": self._side(side), "mon": name or None,
                "text": f"{kw.lstrip('-')}: " + " · ".join(a for a in args if a), "raw": ln}

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
        after = mon.pct() if mon else None
        delta = (round(after - before, 1) if (after is not None and before is not None) else None)
        src = _source(tags)
        who = _pretty(s, name)
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
        return {"kind": kind, "side": s, "mon": name, "pct": delta, "hp_after": after,
                "hp_before": before, "source": src, "fainted": fnt, "text": text}

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
        mon = side.mon(name, species)
        hp, mx, status, fnt = _parse_hp(args[2] if len(args) > 2 else "")
        if hp is not None:
            mon.hp, mon.max_hp = hp, (mx or mon.max_hp)
        if status:
            mon.status = status
        side.active = name
        side.boosts = {}
        side.volatiles = []
        verb = "was dragged in" if kind == "drag" else ("came in" if prev is None else "switched in")
        out = f" for {prev}" if prev and prev != name and kind != "drag" else ""
        return {"kind": kind, "side": s, "mon": name, "species": species, "replaced": prev,
                "hp_after": mon.pct(), "text": f"{_pretty(s, name)} {verb}{out}"}

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
        text = f"{_pretty(s, name)} used {move}"
        if tname and (tname != name or ts != s):
            text += f" on {_pretty(ts, tname)}"
        if "miss" in tags:
            text += " — it missed"
        if "still" in tags and "miss" not in tags:
            text += " (charging)"
        if tags.get("from"):
            text += f" (via {_source(tags)})"
        return {"kind": "move", "side": s, "mon": name, "move": move, "target_side": ts,
                "target": tname or None, "text": text}

    def _faint(self, args, tags):
        p, name = _who(args[0])
        s = self._side(p)
        if s:
            mon = self.sides[s].mon(name)
            mon.fainted, mon.hp = True, 0.0
        return {"kind": "faint", "side": s, "mon": name, "text": f"{_pretty(s, name)} fainted"}

    def _status(self, args, tags):
        p, name = _who(args[0])
        s = self._side(p)
        st = args[1] if len(args) > 1 else ""
        if s:
            self.sides[s].mon(name).status = st
        src = _source(tags)
        return {"kind": "status", "side": s, "mon": name, "status": st, "source": src,
                "text": f"{_pretty(s, name)} is {STATUS_WORDS.get(st, st)}" + (f" ({src})" if src else "")}

    def _curestatus(self, args, tags):
        p, name = _who(args[0])
        s = self._side(p)
        st = args[1] if len(args) > 1 else ""
        if s:
            self.sides[s].mon(name).status = ""
        return {"kind": "cure", "side": s, "mon": name, "status": st,
                "text": f"{_pretty(s, name)} is no longer {STATUS_WORDS.get(st, st)}"}

    def _cureteam(self, args, tags):
        p, name = _who(args[0])
        s = self._side(p)
        if s:
            for m in self.sides[s].mons.values():
                m.status = ""
        return {"kind": "cure", "side": s, "mon": name, "text": f"{_pretty(s, name)} cured its team"}

    def _boost_change(self, args, tags, sign: int):
        p, name = _who(args[0])
        s = self._side(p)
        stat = args[1] if len(args) > 1 else ""
        try:
            n = int(args[2]) if len(args) > 2 else 0
        except ValueError:
            n = 0
        if s and self.sides[s].active == name:
            b = self.sides[s].boosts
            b[stat] = max(-6, min(6, b.get(stat, 0) + sign * n))
            if not b[stat]:
                b.pop(stat)
        word = STAT_WORDS.get(stat, stat)
        verb = ("rose" if sign > 0 else "fell") + (" sharply" if n == 2 else (" drastically" if n >= 3 else ""))
        if n == 0:
            verb = "won't go any " + ("higher" if sign > 0 else "lower")
        return {"kind": "boost" if sign > 0 else "unboost", "side": s, "mon": name, "stat": stat,
                "amount": sign * n, "text": f"{_pretty(s, name)}'s {word} {verb}"}

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
        return {"kind": "boost", "side": s, "mon": name, "stat": stat, "amount": n,
                "text": f"{_pretty(s, name)}'s {STAT_WORDS.get(stat, stat)} was set to {n:+d}"}

    def _clearallboost(self, args, tags):
        for side in self.sides.values():
            side.boosts = {}
        return {"kind": "boost", "side": None, "text": "every stat change was removed"}

    def _clearboost(self, args, tags):
        p, name = _who(args[0])
        s = self._side(p)
        if s:
            self.sides[s].boosts = {}
        return {"kind": "boost", "side": s, "mon": name, "text": f"{_pretty(s, name)}'s stat changes were removed"}

    def _simple(self, kind: str, args, text_fn) -> dict:
        p, name = _who(args[0]) if args else (None, "")
        s = self._side(p)
        return {"kind": kind, "side": s, "mon": name or None, "text": text_fn(_pretty(s, name))}

    def _crit(self, args, tags):
        return self._simple("crit", args, lambda w: f"a critical hit on {w}")

    def _supereffective(self, args, tags):
        return self._simple("supereffective", args, lambda w: f"it's super effective on {w}")

    def _resisted(self, args, tags):
        return self._simple("resisted", args, lambda w: f"it's not very effective on {w}")

    def _immune(self, args, tags):
        return self._simple("immune", args, lambda w: f"it doesn't affect {w}")

    def _miss(self, args, tags):
        tgt = _who(args[1]) if len(args) > 1 else None
        ev = self._simple("miss", args, lambda w: f"{w}'s attack missed")
        if tgt and tgt[1]:
            ev["target"] = tgt[1]
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
        text = f"{_pretty(s, name)}: {eff} " + ("started" if start else "ended") + (f" ({src})" if src else "")
        return {"kind": "volatile", "side": s, "mon": name, "effect": eff, "start": start, "text": text}

    def _start(self, args, tags):
        return self._volatile(args, tags, True)

    def _end(self, args, tags):
        return self._volatile(args, tags, False)

    def _item(self, args, tags):
        it = args[1] if len(args) > 1 else ""
        return dict(self._simple("item", args, lambda w: f"{w} has {it}"), item=it)

    def _enditem(self, args, tags):
        it = args[1] if len(args) > 1 else ""
        verb = "ate its" if "eat" in tags else "lost its"
        return dict(self._simple("item", args, lambda w: f"{w} {verb} {it}"), item=it)

    def _ability(self, args, tags):
        ab = args[1] if len(args) > 1 else ""
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
        return self._simple("activate", args, lambda w: f"{w} was hit {n} time(s)")

    def _transform(self, args, tags):
        return self._simple("activate", args, lambda w: f"{w} transformed")


def fold_turns(lines: Sequence[str], *, trainee_side: str = "p1",
               our_team: Sequence[str] = ()) -> List[dict]:
    """The whole battle, turn by turn: ``[{turn, events: [...], board: {...}}]``.

    Turn 0 is everything before ``|turn|1`` (the leads). Each turn's events are the lines from
    ``|turn|N`` up to (not including) ``|turn|N+1`` — the same window `protocol_for_turn` slices, so a
    decision at turn N reads the events its choice led to. ``board`` is the state at the END of the
    turn (after upkeep)."""
    fold = TurnFold(trainee_side, our_team)
    out: List[dict] = [{"turn": 0, "events": []}]
    for ln in lines or ():
        if ln.startswith("|turn|"):
            out[-1]["board"] = fold.board()
            try:
                n = int(ln.split("|")[2])
            except (IndexError, ValueError):
                n = out[-1]["turn"] + 1
            out.append({"turn": n, "events": []})
            continue
        ev = fold.line(ln)
        if ev is not None:
            out[-1]["events"].append(ev)
    out[-1]["board"] = fold.board()
    return out


__all__ = ["CANT_WORDS", "SIDE_CONDITIONS", "STATUS_WORDS", "TurnFold", "fold_turns"]
