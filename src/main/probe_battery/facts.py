"""The battery's FACTS: ground-truth labels read off a banked decision's two views (`gen3_probe_battery_v1`).

``V`` is the viewer's own ``present()`` view (its side is the TRUTH; the opponent's side is what the viewer has seen),
``W`` the opponent's own view at the same board (its ``ours`` is the opponent's TRUE state: exact HP, stats, item,
moves, boosts, volatiles). A fact is a function of (V, W, the legality, the opponent's choice there, the winner) —
never of a model. Each fact names:

* ``family`` — the catalogue's grouping (per-mon state, side, field, speed, KO, belief, race, switch-in, aggregate,
  phazing, opponent action);
* ``kind`` — ``c`` (continuous, scored by R²) or ``b`` (binary, scored by ROC-AUC);
* ``about`` — whose mon the fact describes (``OA`` our active, ``TA`` their active, ``OB`` our first alive bench mon,
  or ``-``): the CONTROL TASK's key (a random function of that mon's species, Hewitt & Liang);
* ``sites`` — the representation sites where a decision would NEED it (the catalogue's primary read).

A fact's value is ``None`` (the row is excluded for that fact) when it is undefined there, or when it sits within a
rounding error of a decision boundary (standing rule 8): a speed TIE is excluded from "who moves first", a best-move
damage ratio within ±3 % of the target's HP is excluded from "can KO" — never decided by chance.

The damage / speed / residual physics here is a deliberately SMALL gen-3 calculator (``_dmg_max``, ``_eff_speed``,
``_residual``) on the TRUE stats: STAB, type chart, the immunity abilities, Thick Fat, Huge / Pure Power, Choice Band,
the +10 % type items, burn, screens, weather, Explosion's Defense halving; no crit, no multi-hit beyond a fixed count.
It is an approximate LABEL, documented in ``designs/prober/probe_battery.md`` §3, never a model input.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

PHAZERS = frozenset({"roar", "whirlwind"})
#: KO-ness boundary band (rule 8): a best ratio of damage to HP inside it is excluded
KO_BAND = (0.97, 1.03)
_TYPE_ITEMS = {"charcoal": "FIRE", "mysticwater": "WATER", "magnet": "ELECTRIC", "miracleseed": "GRASS",
               "nevermeltice": "ICE", "blackbelt": "FIGHTING", "poisonbarb": "POISON", "softsand": "GROUND",
               "sharpbeak": "FLYING", "twistedspoon": "PSYCHIC", "silverpowder": "BUG", "hardstone": "ROCK",
               "spelltag": "GHOST", "dragonfang": "DRAGON", "blackglasses": "DARK", "metalcoat": "STEEL",
               "silkscarf": "NORMAL"}
_IMMUNE_ABILITY = {"levitate": "GROUND", "flashfire": "FIRE", "voltabsorb": "ELECTRIC", "waterabsorb": "WATER"}
_FIXED = {"seismictoss": ("FIGHTING", 100), "nightshade": ("GHOST", 100), "dragonrage": ("DRAGON", 40),
          "sonicboom": ("NORMAL", 20)}
_MULTI = {"doublekick": 2, "bonemerang": 2, "twineedle": 2, "rockblast": 3, "pinmissile": 3, "spikecannon": 3,
          "bulletseed": 3, "iciclespear": 3, "armthrust": 3, "furyattack": 3, "cometpunch": 3, "doubleslap": 3,
          "barrage": 3, "furyswipes": 3}
_RET_BP = {"return": 102, "frustration": 102}
_SPIKES_CHIP = {0: 0.0, 1: 1 / 8, 2: 1 / 6, 3: 1 / 4}


# ------------------------------------------------------------------------------------------------ dex helpers
def _move(mid: str):
    from agents.gen3_data import moves

    return moves.get(mid)


def _type_mult(atk_type: str, def_types: Sequence[str]) -> float:
    from agents.gen3_data import type_chart

    out = 1.0
    for t in def_types:
        out *= float(type_chart.multiplier(t.upper(), atk_type.upper()))
    return out


def _stage(stat: int, n: int) -> int:
    """The gen-3 stat stage, integer arithmetic (atk/def/spa/spd/spe share the 2/2 table)."""
    n = max(-6, min(6, int(n)))
    return (stat * (2 + n)) // 2 if n >= 0 else (stat * 2) // (2 - n)


def _boost(m: Dict[str, Any], k: str) -> int:
    return int((m.get("boosts") or {}).get(k, 0) or 0)


def _alive(m: Optional[Dict[str, Any]]) -> bool:
    return m is not None and not m.get("fainted") and float(m.get("hp_fraction") or 0.0) > 0.0


def _find(side: Dict[str, Any], species: Optional[str]) -> Optional[Dict[str, Any]]:
    if species is None:
        return None
    for m in side.get("mons") or []:
        if m.get("species") == species:
            return m
    return None


def _move_ids(m: Optional[Dict[str, Any]], usable_only: bool = False) -> List[str]:
    if m is None:
        return []
    out = []
    for mv in m.get("moves") or []:
        if usable_only and int(mv.get("current_pp") or 0) <= 0:
            continue
        out.append(str(mv.get("move_id") or mv.get("id")))
    return out


def _base_id(mid: str) -> str:
    return "hiddenpower" if mid.startswith("hiddenpower") else mid


def _status(m: Optional[Dict[str, Any]]) -> Optional[str]:
    if m is None:
        return None
    s = m.get("status")
    return None if s in (None, "", "fnt") else str(s)


# --------------------------------------------------------------------------------------------- the physics
def _dmg_max(att: Dict[str, Any], mid: str, dfn: Dict[str, Any], *, weather: Optional[str], screen: Dict[str, bool]
             ) -> Optional[float]:
    """Max-roll damage of ``att``'s move ``mid`` on ``dfn`` (both TRUE mons), in HP points; None = not modelled."""
    from agents.observation.incoming_damage import gen3_damage_max, weather_damage_mult
    from agents.enums import PokemonType

    if mid in _FIXED:
        t, dmg = _FIXED[mid]
        return 0.0 if _type_mult(t, dfn["types"]) == 0 else float(dmg)
    if mid == "superfang":
        return 0.0 if _type_mult("NORMAL", dfn["types"]) == 0 else float(int(dfn["current_hp"]) // 2)
    md = _move(mid)
    if md is None:
        return None
    bp = _RET_BP.get(mid, int(md.base_power))
    if bp <= 0:
        return None
    mtype = md.type.name
    physical = md.category.name == "PHYSICAL"
    eff = _type_mult(mtype, dfn["types"])
    dab = str(dfn.get("ability") or "")
    if _IMMUNE_ABILITY.get(dab) == mtype or (dab == "wonderguard" and eff <= 1.0):
        eff = 0.0
    if dab == "thickfat" and mtype in ("FIRE", "ICE"):
        eff *= 0.5
    st, dst = att["stats"], dfn["stats"]
    if physical:
        a = _stage(int(st["atk"]), _boost(att, "atk"))
        if att.get("item") == "choiceband":
            a = a * 3 // 2
        if att.get("ability") in ("hugepower", "purepower"):
            a *= 2
        d = _stage(int(dst["def"]), _boost(dfn, "def"))
        if mid in ("explosion", "selfdestruct"):
            d = max(1, d // 2)
    else:
        a = _stage(int(st["spa"]), _boost(att, "spa"))
        d = _stage(int(dst["spd"]), _boost(dfn, "spd"))
    if _TYPE_ITEMS.get(str(att.get("item") or "")) == mtype:
        bp = bp * 11 // 10
    stab = mtype.lower() in [t.lower() for t in att.get("types") or []]
    w = weather_damage_mult(PokemonType[mtype], weather)
    burned = physical and _status(att) == "brn" and att.get("ability") != "guts"
    scr = bool(screen.get("reflect" if physical else "light_screen"))
    dmg = float(gen3_damage_max(bp, a, d, stab=stab, type_eff=eff, screen=scr, weather=w, burned=burned))
    return dmg * _MULTI.get(mid, 1)


def _eff_speed(m: Dict[str, Any], weather: Optional[str]) -> int:
    s = _stage(int(m["stats"]["spe"]), _boost(m, "spe"))
    ab = m.get("ability")
    if (ab == "swiftswim" and weather and "rain" in weather) or (ab == "chlorophyll" and weather and "sun" in weather):
        s *= 2
    if _status(m) == "par":
        s //= 4
    return s


def _residual(m: Dict[str, Any], *, weather: Optional[str], wish: bool) -> float:
    """Net end-of-turn HP change as a fraction of max HP (approximate; gen-3 order and caps ignored)."""
    hp, mx = int(m.get("current_hp") or 0), max(1, int(m.get("max_hp") or 1))
    r = 0.0
    if m.get("item") == "leftovers" and hp < mx:
        r += 1 / 16
    st = _status(m)
    if st in ("brn", "psn"):
        r -= 1 / 8
    elif st == "tox":
        r -= min(15, int(m.get("status_counter") or 0) + 1) / 16
    types = [t.lower() for t in m.get("types") or []]
    if weather and "sand" in weather and not {"rock", "ground", "steel"} & set(types) \
            and m.get("ability") != "sandveil":
        r -= 1 / 16
    if weather and "hail" in weather and "ice" not in types:
        r -= 1 / 16
    vol = m.get("volatiles") or {}
    if "leechseed" in vol:
        r -= 1 / 8
    if "curse" in vol:
        r -= 1 / 4
    if "nightmare" in vol and st == "slp":
        r -= 1 / 4
    if "ingrain" in vol:
        r += 1 / 16
    if wish:
        r += 1 / 2
    return float(max(-1.0, min(1.0, r)))


# ------------------------------------------------------------------------------------------- the context
@dataclass
class Ctx:
    d: Dict[str, Any]
    V: Dict[str, Any]
    W: Dict[str, Any]
    oa: Optional[Dict[str, Any]]           # our active, TRUE (the viewer's own side)
    ta: Optional[Dict[str, Any]]           # their active, TRUE (the opponent's own view)
    ta_seen: Optional[Dict[str, Any]]      # their active as the viewer has seen it
    ob: Optional[Dict[str, Any]]           # our first alive bench mon (obs slot order), TRUE
    weather: Optional[str]
    our_screen: Dict[str, bool]
    their_screen: Dict[str, bool]
    top_species: Tuple[str, ...]

    @property
    def opp_true(self) -> List[Dict[str, Any]]:
        return list(self.W["ours"]["mons"])

    @property
    def opp_seen(self) -> List[Dict[str, Any]]:
        return list(self.V["opp"]["mons"])


def _screens(side: Dict[str, Any]) -> Dict[str, bool]:
    sc = side.get("side_conditions") or {}
    return {k: k in sc for k in ("reflect", "light_screen", "safeguard")}


def make_ctx(d: Dict[str, Any], top_species: Tuple[str, ...] = ()) -> Ctx:
    V, W = d["V"], d["W"]
    oa = _find(V["ours"], V["ours"].get("active"))
    ta_sp = V["opp"].get("active")
    ta = _find(W["ours"], ta_sp)
    ta_seen = _find(V["opp"], ta_sp)
    ob = None
    for sp in d.get("our_slots") or []:
        m = _find(V["ours"], sp)
        if m is not None and not m.get("active") and _alive(m):
            ob = m
            break
    w = (V.get("weather") or {}).get("weather")
    return Ctx(d=d, V=V, W=W, oa=oa if _alive(oa) else None, ta=ta if _alive(ta) else None, ta_seen=ta_seen,
               ob=ob, weather=w, our_screen=_screens(V["ours"]), their_screen=_screens(V["opp"]),
               top_species=top_species)


# ------------------------------------------------------------------------------------------- the registry
@dataclass(frozen=True)
class Fact:
    name: str
    family: str
    kind: str                      # "c" | "b"
    about: str                     # "OA" | "TA" | "OB" | "-"
    sites: Tuple[str, ...]
    fn: Callable[[Ctx], Optional[float]]
    desc: str


FACTS: List[Fact] = []


def fact(name: str, family: str, kind: str, about: str, sites: Sequence[str], desc: str):
    def deco(fn: Callable[[Ctx], Optional[float]]):
        FACTS.append(Fact(name, family, kind, about, tuple(sites), fn, desc))
        return fn
    return deco


def _b(x: bool) -> float:
    return 1.0 if x else 0.0


#: DECISION sites: the tokens the pointer head scores (our mon tokens for a switch, the move seats for a move), the
#: policy state and, for the value-relevant aggregates, the critic pool. The BOARD tokens are where a side / field
#: fact is STORED, not where a decision reads it, so they are reported per slot but never a decision site.
S_OA = ("OA", "Mcat", "PI")
S_TA = ("TA", "PI")
S_OB = ("OB", "PI")
S_OURSIDE = ("OA", "OB", "Mcat", "PI")
S_THEIRSIDE = ("TA", "OA", "Mcat", "PI")
S_FIELD = ("OA", "OB", "Mcat", "PI")
S_DUEL = ("OA", "TA", "Mcat", "PI")
S_AGG = ("VF", "PI")


def _register_mon_facts() -> None:
    for who in ("OA", "TA", "OB"):
        key = who.lower()
        sites = {"OA": S_OA, "TA": S_TA, "OB": S_OB}[who]
        fact(f"hp_{who}", "mon", "c", who, sites, f"{who}'s HP fraction (true)")(
            lambda c, k=key: None if getattr(c, k) is None else float(getattr(c, k)["hp_fraction"]))
        fact(f"status_any_{who}", "mon", "b", who, sites, f"{who} has a major status")(
            lambda c, k=key: None if getattr(c, k) is None else _b(_status(getattr(c, k)) is not None))
        if who == "OB":
            continue
        for st, names in (("par", ("par",)), ("slp", ("slp",)), ("psn", ("psn", "tox")), ("brn", ("brn",))):
            fact(f"status_{st}_{who}", "mon", "b", who, sites, f"{who}'s status is {'/'.join(names)}")(
                lambda c, k=key, n=names: None if getattr(c, k) is None else _b(_status(getattr(c, k)) in n))
        for stat in ("atk", "def", "spa", "spd", "spe"):
            fact(f"boost_{stat}_{who}", "mon", "c", who, sites, f"{who}'s {stat} stage / 6")(
                lambda c, k=key, s=stat: None if getattr(c, k) is None else _boost(getattr(c, k), s) / 6.0)
        fact(f"boosted_{who}", "mon", "b", who, sites, f"{who} has a positive stage")(
            lambda c, k=key: None if getattr(c, k) is None else
            _b(any(int(v) > 0 for v in (getattr(c, k).get("boosts") or {}).values())))
        fact(f"toxic_counter_{who}", "mon", "c", who, sites, f"{who}'s Toxic counter / 8 (Toxic rows only)")(
            lambda c, k=key: None if getattr(c, k) is None or _status(getattr(c, k)) != "tox" else
            min(8, int(getattr(c, k).get("status_counter") or 0)) / 8.0)
        fact(f"sleep_turns_{who}", "mon", "c", who, sites, f"{who}'s sleep turns slept / 4 (asleep rows only)")(
            lambda c, k=key: None if getattr(c, k) is None or _status(getattr(c, k)) != "slp" else
            min(4, int(getattr(c, k).get("status_counter") or 0)) / 4.0)
        for vol in ("substitute", "leechseed", "confusion"):
            fact(f"{vol}_{who}", "mon", "b", who, sites, f"{who} has {vol}")(
                lambda c, k=key, v=vol: None if getattr(c, k) is None else
                _b(v in (getattr(c, k).get("volatiles") or {})))


_register_mon_facts()


def _locked(legal: Optional[Dict[str, Any]]) -> Optional[float]:
    if not legal or not legal.get("move_slots"):
        return None
    return _b(any(bool(s.get("disabled")) for s in legal["move_slots"]))


@fact("choice_locked_OA", "mon", "b", "OA", S_OA, "our Choice Band active is locked into one move (CB holders only)")
def _(c: Ctx) -> Optional[float]:
    if c.oa is None or c.oa.get("item") != "choiceband":
        return None
    return _locked(c.d.get("legal"))


@fact("choice_locked_TA", "mon", "b", "TA", S_TA, "their Choice Band active is locked (true CB holders only)")
def _(c: Ctx) -> Optional[float]:
    if c.ta is None or c.ta.get("item") != "choiceband":
        return None
    return _locked(c.d.get("opp_legal"))


# ----------------------------------------------------------------------------------------------- side / field
def _spikes(side: Dict[str, Any]) -> int:
    return int((side.get("side_conditions") or {}).get("spikes", 0) or 0)


fact("spikes_ours", "side", "c", "-", S_OURSIDE, "Spikes layers on our side / 3")(lambda c: _spikes(c.V["ours"]) / 3)
fact("spikes_ours_any", "side", "b", "-", S_OURSIDE, "any Spikes on our side")(lambda c: _b(_spikes(c.V["ours"]) > 0))
fact("spikes_theirs", "side", "c", "-", S_THEIRSIDE, "Spikes layers on their side / 3")(
    lambda c: _spikes(c.V["opp"]) / 3)
fact("spikes_theirs_any", "side", "b", "-", S_THEIRSIDE, "any Spikes on their side")(
    lambda c: _b(_spikes(c.V["opp"]) > 0))
for _side, _sites, _k in (("ours", S_OURSIDE, "our"), ("opp", S_THEIRSIDE, "their")):
    fact(f"screen_{_side}", "side", "b", "-", _sites, f"Reflect or Light Screen on {_k} side")(
        lambda c, s=_side: _b(any(k in (c.V[s].get("side_conditions") or {}) for k in ("reflect", "light_screen"))))
    fact(f"safeguard_{_side}", "side", "b", "-", _sites, f"Safeguard on {_k} side")(
        lambda c, s=_side: _b("safeguard" in (c.V[s].get("side_conditions") or {})))
    fact(f"screen_turns_{_side}", "side", "c", "-", _sites, f"turns left on {_k} screen / 5 (screen rows only)")(
        lambda c, s=_side: (lambda sc: None if not sc else
                            float(np.clip((min(sc) + 5 - int(c.V.get("turn") or 0)) / 5, 0, 1)))(
            [int(v) for k, v in (c.V[s].get("side_conditions") or {}).items() if k in ("reflect", "light_screen")]))
fact("wish_ours", "side", "b", "-", S_OURSIDE, "a Wish resolves on our side this turn")(
    lambda c: None if not c.d.get("wish") else _b(bool(c.d["wish"][0])))
fact("wish_theirs", "side", "b", "-", S_THEIRSIDE, "a Wish resolves on their side this turn")(
    lambda c: None if not c.d.get("wish") else _b(bool(c.d["wish"][1])))

fact("weather_any", "field", "b", "-", S_FIELD, "any weather")(lambda c: _b(bool(c.weather)))
for _w in ("sand", "rain", "sun", "hail"):
    fact(f"weather_{_w}", "field", "b", "-", S_FIELD, f"the weather is {_w}")(
        lambda c, w=_w: _b(bool(c.weather) and w in str(c.weather).lower()))
fact("weather_turns_left", "field", "c", "-", S_FIELD, "a TEMPORARY weather's turns left / 5")(
    lambda c: None if not c.weather or (c.V.get("weather") or {}).get("is_permanent") else
    float(np.clip((5 - int((c.V.get("weather") or {}).get("turns_active") or 0)) / 5, 0, 1)))
fact("turn", "field", "c", "-", S_FIELD, "the turn number / 150 (capped)")(
    lambda c: min(150, int(c.V.get("turn") or 0)) / 150)


# ----------------------------------------------------------------------------------------------- speed / KO
@fact("we_move_first", "speed", "b", "-", S_DUEL, "our active's effective Speed beats theirs (ties excluded)")
def _(c: Ctx) -> Optional[float]:
    if c.oa is None or c.ta is None:
        return None
    a, b = _eff_speed(c.oa, c.weather), _eff_speed(c.ta, c.weather)
    return None if a == b else _b(a > b)


@fact("speed_log_ratio", "speed", "c", "-", S_DUEL, "log(our effective Speed / theirs), clipped ±2")
def _(c: Ctx) -> Optional[float]:
    if c.oa is None or c.ta is None:
        return None
    a, b = max(1, _eff_speed(c.oa, c.weather)), max(1, _eff_speed(c.ta, c.weather))
    return float(np.clip(math.log(a / b), -2, 2))


def _best_ratio(att: Dict[str, Any], mids: Sequence[str], dfn: Dict[str, Any], weather: Optional[str],
                screen: Dict[str, bool]) -> Optional[float]:
    hp = int(dfn.get("current_hp") or 0)
    if hp <= 0:
        return None
    best = None
    for mid in mids:
        dm = _dmg_max(att, mid, dfn, weather=weather, screen=screen)
        if dm is None:
            continue
        best = dm if best is None else max(best, dm)
    return None if best is None else best / hp


def _our_usable(c: Ctx) -> List[str]:
    lg = c.d.get("legal") or {}
    slots = lg.get("move_slots") or []
    if slots:
        own = {_base_id(m): m for m in _move_ids(c.oa)}
        return [own.get(str(s["id"]), str(s["id"])) for s in slots
                if not s.get("disabled") and int(s.get("current_pp") or 0) > 0]
    return _move_ids(c.oa, usable_only=True)


def _ko(ratio: Optional[float]) -> Optional[float]:
    if ratio is None or KO_BAND[0] <= ratio <= KO_BAND[1]:
        return None
    return _b(ratio >= 1.0)


@fact("we_can_ko", "ko", "b", "TA", S_DUEL, "a usable move of our active max-rolls their active's HP (no crit)")
def _(c: Ctx) -> Optional[float]:
    if c.oa is None or c.ta is None or "substitute" in (c.ta.get("volatiles") or {}):
        return None
    return _ko(_best_ratio(c.oa, _our_usable(c), c.ta, c.weather, c.their_screen))


@fact("they_can_ko", "ko", "b", "OA", S_DUEL, "a move of their active max-rolls our active's HP (true set)")
def _(c: Ctx) -> Optional[float]:
    if c.oa is None or c.ta is None or "substitute" in (c.oa.get("volatiles") or {}):
        return None
    return _ko(_best_ratio(c.ta, _move_ids(c.ta, True), c.oa, c.weather, c.our_screen))


@fact("our_best_dmg", "ko", "c", "TA", S_DUEL, "our best max-roll damage / their current HP, clipped at 2")
def _(c: Ctx) -> Optional[float]:
    if c.oa is None or c.ta is None:
        return None
    r = _best_ratio(c.oa, _our_usable(c), c.ta, c.weather, c.their_screen)
    return None if r is None else float(min(2.0, r))


@fact("their_best_dmg", "ko", "c", "OA", S_DUEL, "their best max-roll damage / our current HP, clipped at 2")
def _(c: Ctx) -> Optional[float]:
    if c.oa is None or c.ta is None:
        return None
    r = _best_ratio(c.ta, _move_ids(c.ta, True), c.oa, c.weather, c.our_screen)
    return None if r is None else float(min(2.0, r))


# ------------------------------------------------------------------------------------------------- belief
def _seen_moves(m: Optional[Dict[str, Any]]) -> List[str]:
    return [_base_id(x) for x in _move_ids(m)]


_MOVE_CLASSES: Dict[str, Callable[[str], bool]] = {
    "phazer": lambda mid: mid in PHAZERS,
    "recovery": lambda mid: bool(getattr(_move(mid), "is_heal", False)) and mid not in ("wish", "rest"),
    "rest": lambda mid: mid == "rest",
    "setup": lambda mid: bool(getattr(_move(mid), "is_boost", False)) or mid == "curse",
    "status": lambda mid: getattr(_move(mid), "status_inflicted", None) is not None,
    "earthquake": lambda mid: mid == "earthquake",
    "explosion": lambda mid: mid in ("explosion", "selfdestruct"),
    "spikes": lambda mid: mid == "spikes",
    "rapidspin": lambda mid: mid == "rapidspin",
    "protect": lambda mid: bool(getattr(_move(mid), "is_protect", False)),
    "batonpass": lambda mid: mid == "batonpass",
    "substitute": lambda mid: mid == "substitute",
}


def _register_belief() -> None:
    for cls, pred in _MOVE_CLASSES.items():
        fact(f"TA_hidden_{cls}", "belief", "b", "TA", S_TA,
             f"their active's UNREVEALED moves include a {cls} move (rows where none is revealed and < 4 are)")(
            lambda c, p=pred: _hidden_class(c, p))
    for it in ("leftovers", "choiceband"):
        fact(f"TA_hidden_item_{it}", "belief", "b", "TA", S_TA, f"their active's UNREVEALED item is {it}")(
            lambda c, i=it: None if c.ta is None or (c.ta_seen is not None and c.ta_seen.get("item")) else
            _b(c.ta.get("item") == i))


def _hidden_class(c: Ctx, pred: Callable[[str], bool]) -> Optional[float]:
    if c.ta is None:
        return None
    seen = _seen_moves(c.ta_seen)
    if len(seen) >= 4 or any(pred(m) for m in seen):
        return None
    true = [_base_id(m) for m in _move_ids(c.ta)]
    return _b(any(pred(m) for m in true if m not in seen))


_register_belief()


@fact("opp_hidden_count", "belief", "c", "-", ("PI", "VF", "TA"), "opponent mons not yet revealed / 6")
def _(c: Ctx) -> Optional[float]:
    return (int(c.W["ours"].get("team_size") or 6) - len(c.opp_seen)) / 6


def _register_species(top: Sequence[str]) -> None:
    for sp in top:
        fact(f"opp_hidden_has_{sp}", "belief", "b", "-", ("PI", "VF", "TA"),
             f"the opponent's UNREVEALED mons include {sp} (rows where {sp} is not revealed and some mon is hidden)")(
            lambda c, s=sp: None if any(m["species"] == s for m in c.opp_seen) or
            len(c.opp_seen) >= int(c.W["ours"].get("team_size") or 6) else
            _b(any(m["species"] == s for m in c.opp_true)))


def _opp_action(c: Ctx) -> Optional[Tuple[str, str]]:
    ch = c.d.get("opp_choice")
    if not ch:
        return None
    verb, _, arg = str(ch).partition(" ")
    return verb, arg.strip().lower().replace(" ", "")


@fact("opp_switches_next", "opp_action", "b", "TA", S_TA, "the opponent's ACTUAL choice at this board is a switch")
def _(c: Ctx) -> Optional[float]:
    a = _opp_action(c)
    return None if a is None or c.ta is None else _b(a[0] == "switch")


@fact("opp_attacks_next", "opp_action", "b", "TA", S_TA, "the opponent's actual choice is a damaging move")
def _(c: Ctx) -> Optional[float]:
    a = _opp_action(c)
    if a is None or c.ta is None:
        return None
    if a[0] != "move":
        return 0.0
    md = _move(a[1])
    return _b(md is not None and (int(md.base_power) > 0 or a[1] in _FIXED or a[1] in _RET_BP or a[1] == "superfang"))


# ----------------------------------------------------------------------------------------- race / switch-in
def _wish(c: Ctx, i: int) -> bool:
    w = c.d.get("wish") or [False, False]
    return bool(w[i])


fact("residual_OA", "race", "c", "OA", ("OA", "VF", "PI"), "our active's net end-of-turn HP change (approx.)")(
    lambda c: None if c.oa is None else _residual(c.oa, weather=c.weather, wish=_wish(c, 0)))
fact("residual_TA", "race", "c", "TA", ("TA", "VF", "PI"), "their active's net end-of-turn HP change (approx.)")(
    lambda c: None if c.ta is None else _residual(c.ta, weather=c.weather, wish=_wish(c, 1)))
fact("residual_race", "race", "c", "-", ("OA", "TA", "VF", "PI"), "residual_OA minus residual_TA")(
    lambda c: None if c.oa is None or c.ta is None else
    _residual(c.oa, weather=c.weather, wish=_wish(c, 0)) - _residual(c.ta, weather=c.weather, wish=_wish(c, 1)))


def _grounded(m: Dict[str, Any]) -> bool:
    return "flying" not in [t.lower() for t in m.get("types") or []] and m.get("ability") != "levitate"


fact("entry_chip_OB", "switch_in", "c", "OB", S_OB, "Spikes damage our bench mon takes on entry (fraction)")(
    lambda c: None if c.ob is None else (_SPIKES_CHIP[_spikes(c.V["ours"])] if _grounded(c.ob) else 0.0))


@fact("threat_on_OB", "switch_in", "c", "OB", S_OB, "their active's best max-roll on our bench mon / its HP, cap 2")
def _(c: Ctx) -> Optional[float]:
    if c.ob is None or c.ta is None:
        return None
    r = _best_ratio(c.ta, _move_ids(c.ta, True), c.ob, c.weather, c.our_screen)
    return None if r is None else float(min(2.0, r))


@fact("OB_ko_by_TA", "switch_in", "b", "OB", S_OB, "their active can max-roll KO our bench mon (from full entry)")
def _(c: Ctx) -> Optional[float]:
    if c.ob is None or c.ta is None:
        return None
    return _ko(_best_ratio(c.ta, _move_ids(c.ta, True), c.ob, c.weather, c.our_screen))


@fact("OB_outspeeds_TA", "switch_in", "b", "OB", S_OB, "our bench mon outspeeds their active (ties excluded)")
def _(c: Ctx) -> Optional[float]:
    if c.ob is None or c.ta is None:
        return None
    a, b = _eff_speed(c.ob, c.weather), _eff_speed(c.ta, c.weather)
    return None if a == b else _b(a > b)


# ----------------------------------------------------------------------------------------------- aggregates
def _hp_sum(mons: Sequence[Dict[str, Any]], size: int) -> float:
    return sum(float(m.get("hp_fraction") or 0) for m in mons if not m.get("fainted")) + max(0, size - len(mons))


def _n_alive(mons: Sequence[Dict[str, Any]], size: int) -> int:
    return sum(1 for m in mons if _alive(m)) + max(0, size - len(mons))


fact("our_hp_total", "aggregate", "c", "-", S_AGG, "our summed HP fractions / 6")(
    lambda c: _hp_sum(c.V["ours"]["mons"], int(c.V["ours"].get("team_size") or 6)) / 6)
fact("their_hp_total", "aggregate", "c", "-", S_AGG, "their summed HP fractions / 6 (true)")(
    lambda c: _hp_sum(c.W["ours"]["mons"], int(c.W["ours"].get("team_size") or 6)) / 6)
fact("hp_total_diff", "aggregate", "c", "-", S_AGG, "our HP total minus theirs")(
    lambda c: (_hp_sum(c.V["ours"]["mons"], 6) - _hp_sum(c.W["ours"]["mons"], 6)) / 6)
fact("our_alive", "aggregate", "c", "-", S_AGG, "our mons alive / 6")(
    lambda c: _n_alive(c.V["ours"]["mons"], int(c.V["ours"].get("team_size") or 6)) / 6)
fact("their_alive", "aggregate", "c", "-", S_AGG, "their mons alive / 6")(
    lambda c: _n_alive(c.W["ours"]["mons"], int(c.W["ours"].get("team_size") or 6)) / 6)
fact("alive_diff", "aggregate", "c", "-", S_AGG, "our alive minus theirs / 6")(
    lambda c: (_n_alive(c.V["ours"]["mons"], 6) - _n_alive(c.W["ours"]["mons"], 6)) / 6)


@fact("outcome_win", "aggregate", "b", "-", S_AGG, "the viewer WINS this game (draws excluded)")
def _(c: Ctx) -> Optional[float]:
    w = c.d.get("winner")
    if w not in ("p1", "p2"):
        return None
    return _b(w == f"p{int(c.d['side']) + 1}")


# --------------------------------------------------------------------------------------------------- phazing
def _has_phazer(m: Optional[Dict[str, Any]], usable: bool = False) -> bool:
    return any(_base_id(x) in PHAZERS for x in _move_ids(m, usable))


def _boosted(m: Optional[Dict[str, Any]]) -> bool:
    return m is not None and any(int(v) > 0 for v in (m.get("boosts") or {}).values())


S_PHZ = ("PI", "OA", "TA", "VF")
fact("TA_phazer_true", "phazing", "b", "TA", S_TA, "their active HAS Roar / Whirlwind (true set)")(
    lambda c: None if c.ta is None else _b(_has_phazer(c.ta)))
fact("TA_phazer_revealed", "phazing", "b", "TA", S_TA, "their active has REVEALED Roar / Whirlwind")(
    lambda c: None if c.ta is None else _b(_has_phazer(c.ta_seen)))
fact("opp_team_phazer_true", "phazing", "b", "-", ("PI", "VF", "TA"), "an alive opponent mon has Roar / Whirlwind")(
    lambda c: _b(any(_alive(m) and _has_phazer(m) for m in c.opp_true)))
fact("opp_team_phazer_revealed", "phazing", "b", "-", ("PI", "VF", "TA"),
     "an alive opponent mon has REVEALED Roar / Whirlwind")(
    lambda c: _b(any(_alive(m) and _has_phazer(m) for m in c.opp_seen)))
fact("OA_phazer", "phazing", "b", "OA", S_OA, "our active has a usable Roar / Whirlwind")(
    lambda c: None if c.oa is None else _b(_has_phazer(c.oa, True)))
fact("our_team_phazer", "phazing", "b", "-", ("PI", "VF", "OA"), "an alive mon of ours has Roar / Whirlwind")(
    lambda c: _b(any(_alive(m) and _has_phazer(m) for m in c.V["ours"]["mons"])))
fact("phaze_value", "phazing", "b", "-", S_PHZ,
     "their active is boosted or behind a Substitute AND Spikes are on their side (phazing chips + erases)")(
    lambda c: None if c.ta is None else
    _b((_boosted(c.ta) or "substitute" in (c.ta.get("volatiles") or {})) and _spikes(c.V["opp"]) > 0))
fact("phaze_value_tool", "phazing", "b", "-", S_PHZ, "phaze_value AND our active can phaze now")(
    lambda c: None if c.ta is None or c.oa is None else
    _b((_boosted(c.ta) or "substitute" in (c.ta.get("volatiles") or {})) and _spikes(c.V["opp"]) > 0
       and _has_phazer(c.oa, True)))
fact("phaze_threat", "phazing", "b", "-", S_PHZ, "our active is boosted AND an alive opponent mon has a phazer")(
    lambda c: None if c.oa is None else _b(_boosted(c.oa) and any(_alive(m) and _has_phazer(m) for m in c.opp_true)))
fact("phaze_threat_believed", "phazing", "b", "-", S_PHZ,
     "our active is boosted AND an alive opponent mon has REVEALED a phazer")(
    lambda c: None if c.oa is None else _b(_boosted(c.oa) and any(_alive(m) and _has_phazer(m) for m in c.opp_seen)))


# ------------------------------------------------------------------------------------------------- extraction
def top_opp_species(decisions: Sequence[Dict[str, Any]], k: int = 10) -> Tuple[str, ...]:
    """The ``k`` species most often on the OPPONENT's team over the bank's games (count, then name) — a property of
    the bank, deterministic."""
    seen: Dict[Tuple[str, int], List[str]] = {}
    for d in decisions:
        key = (d["game"], int(d["side"]))
        if key not in seen:
            seen[key] = [m["species"] for m in d["W"]["ours"]["mons"]]
    cnt = Counter(sp for v in seen.values() for sp in v)
    return tuple(sp for sp, _ in sorted(cnt.items(), key=lambda t: (-t[1], t[0]))[:k])


def registry(top_species: Sequence[str]) -> List[Fact]:
    """The full fact list for a bank (the static facts + one hidden-species fact per top opponent species)."""
    base = [f for f in FACTS if not f.name.startswith("opp_hidden_has_")]
    FACTS[:] = base
    _register_species(top_species)
    return list(FACTS)


def extract(decisions: Sequence[Dict[str, Any]]) -> Tuple[List[Fact], np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """``(facts, values [N, F] float32 (NaN = excluded), valid [N, F] bool, species keys {OA, TA, OB: [N] int})``.
    The species keys index a sorted species list (−1 = no mon) — the control task's input."""
    top = top_opp_species(decisions)
    facts = registry(top)
    N, F = len(decisions), len(facts)
    vals = np.full((N, F), np.nan, dtype=np.float32)
    names: Dict[str, int] = {}
    keys = {k: np.full(N, -1, dtype=np.int64) for k in ("OA", "TA", "OB")}
    all_sp = sorted({m["species"] for d in decisions for side in (d["V"]["ours"], d["W"]["ours"])
                     for m in side["mons"]})
    names = {s: i for i, s in enumerate(all_sp)}
    for i, d in enumerate(decisions):
        c = make_ctx(d, top)
        for k, m in (("OA", c.oa), ("TA", c.ta), ("OB", c.ob)):
            if m is not None:
                keys[k][i] = names[m["species"]]
        for j, f in enumerate(facts):
            v = f.fn(c)
            if v is not None and math.isfinite(v):
                vals[i, j] = float(v)
    valid = np.isfinite(vals)
    return facts, vals, valid, keys
