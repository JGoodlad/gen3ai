"""HIDDEN POWER REPAIR for replay-derived gen3ou teams (``gen3_hp_repair_v1``).

Replay-derived team sets (Metamon's ``hl_05_26`` archive, our ``utils.ladder_corpus``) write Hidden Power
with no type and no IVs (or the default all-31 line). Blank IVs are all 31, and under the gen-3 formula
(``sim/dex.ts`` ``getHiddenPower``) that is Hidden Power DARK, almost never the player's real choice: a
gen-3 replay never reveals the type, so the fill is an ARTIFACT (``designs/ops/TECH_DEBT_BACKLOG.md``,
"The Metamon ladder-usage corpus's HIDDEN fields are artifacts"; measured 2026-09-25: 49,433 of 52,007
corpus HP users). This module re-samples that one hidden field by the owner's spec (2026-09-25):

1. **Only an UNTYPED Hidden Power whose IVs are absent or all 31 is a target.** A typed one
   (``Hidden Power [Fire]`` / ``Hidden Power Fire``) or one with any IV below 31 (the player chose a
   type through the IVs) is left byte-identical. In the archive every set carries an explicit
   ``IVs: 31 / 31 / ...`` line, which is the same default fill as no line at all.
2. **The set's SIDE.** In gen 3 a move's category follows its TYPE (``data/mods/gen3/scripts.ts`` and
   ``moves.ts`` ``hiddenpower.onModifyMove``: Fire / Water / Grass / Ice / Electric / Dark / Psychic /
   Dragon special, the rest physical), so the new type must be on the set's side. Three votes, each
   -1 / 0 / +1 (+1 = physical): the EVs (``sign(Atk EV - SpA EV)``), the nature (+1 if it raises Atk or
   lowers SpA, -1 if it raises SpA or lowers Atk) and the set's OTHER attacking moves
   (``sign(#physical - #special)`` by the gen-3 type rule; status, fixed-damage and OHKO moves do not
   vote, variable-power attacks such as Return do). The sign of the vote sum is the side; a zero sum
   falls to the set's real Atk vs SpA stat (base stats + IVs + EVs + nature, the gen-3 formula); an
   exact stat tie leaves the side open (both).
3. **The TYPE** is sampled from Smogon's per-species Hidden Power usage (``priors.hidden_power``,
   ``gen3_hidden_power_priors.json``: the chaos ``Moves`` typed ``hiddenpower<type>`` keys, 12 months)
   restricted to the side and renormalised. A species with no Smogon mass on that side falls back to
   the FORMAT-WIDE Smogon HP usage on that side (``sum over species of the chaos HP counts``). Priors
   are Smogon only (owner rule), never our pool.
4. **The IVs** are written the way Showdown's validator does (``sim/team-validator.ts`` ~1154-1164:
   ``fillStats(dex.types.get(hpType).HPivs, 31)``) — :data:`HP_IVS`, pinned to ``data/typechart.ts`` by a
   test, so the power stays 70. NOT ``utils.gen3_utils.GEN3_HP_IVS``: that table's Ghost row is
   30 Atk / 30 SpD where Showdown's is 30 Def / 30 SpD (both Ghost, power 70; FINDING 2026-10-18). The 30 Speed IV of Fire / Psychic / Rock /
   Fighting lowers the Speed stat by 1 (a speed-tie change, pinned by a test).
5. **Deterministic.** One ``random.Random`` per team, seeded by :func:`team_seed` (base seed + team id),
   drawn once per target mon in slot order over the types in alphabetical order.

The EV spread and nature are kept as written (guessed or not); only the HP move line and the IVs line
change. ``repair_team`` returns the new paste and one :class:`MonRepair` record per target mon.
"""
from __future__ import annotations

import hashlib
import math
import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from utils.showdown_id import to_id_str

from . import moves as moves_dex
from . import natures as natures_dex
from . import priors
from . import species as species_dex

REPAIR_VERSION = "gen3_hp_repair_v1"

#: Gen 3: a damaging move's category follows its TYPE (``data/mods/gen3/scripts.ts`` ``specialTypes``).
SPECIAL_TYPES = frozenset({"fire", "water", "grass", "ice", "electric", "dark", "psychic", "dragon"})
_STAT_KEYS6 = ("hp", "atk", "def", "spa", "spd", "spe")
#: ``data/typechart.ts`` ``HPivs`` per type (the stats set to 30; the rest are 31), as the validator fills them.
_HPIVS_SPARSE = {
    "Bug": ("atk", "def", "spd"), "Dark": (), "Dragon": ("atk",), "Electric": ("spa",),
    "Fighting": ("def", "spa", "spd", "spe"), "Fire": ("atk", "spa", "spe"), "Flying": ("hp", "atk", "def", "spa", "spd"),
    "Ghost": ("def", "spd"), "Grass": ("atk", "spa"), "Ground": ("spa", "spd"), "Ice": ("atk", "def"),
    "Poison": ("def", "spa", "spd"), "Psychic": ("atk", "spe"), "Rock": ("def", "spd", "spe"), "Steel": ("spd",),
    "Water": ("atk", "def", "spa"),
}
HP_IVS: Dict[str, Dict[str, int]] = {t: {k: (30 if k in low else 31) for k in _STAT_KEYS6}
                                     for t, low in _HPIVS_SPARSE.items()}
HP_TYPES = tuple(sorted(t.lower() for t in HP_IVS))               # the 16 types, alphabetical
PHYSICAL, SPECIAL = "physical", "special"
#: Gen-3 attacks with base power 0 in the dex whose damage still uses the attacking stat
#: (Showdown ``basePowerCallback`` or a random power: Magnitude, Present). Fixed-damage
#: (Seismic Toss, Night Shade, Super Fang, Counter, ...) and OHKO moves are NOT here: they do not vote.
#: Re-derived from the pinned Showdown by ``hp_repair_showdown_integration_test.py``.
VARIABLE_POWER_ATTACKS = frozenset({"flail", "frustration", "lowkick", "magnitude", "present", "return",
                                    "reversal", "spitup"})
STAT_KEYS = ("hp", "atk", "def", "spa", "spd", "spe")
_STAT_LABEL = {"hp": "HP", "atk": "Atk", "def": "Def", "spa": "SpA", "spd": "SpD", "spe": "Spe"}
_LABEL_STAT = {v.lower(): k for k, v in _STAT_LABEL.items()}

_HP_LINE = re.compile(r"^-\s*Hidden Power(?:\s*\[\s*(?P<br>[A-Za-z]*)\s*\]|\s+(?P<sp>[A-Za-z]+))?\s*$",
                      re.IGNORECASE)
_STAT_PAIR = re.compile(r"(\d+)\s*(HP|Atk|Def|SpA|SpD|Spe)", re.IGNORECASE)
_NICK = re.compile(r"^(?P<nick>.+?)\s*\((?P<species>[^()]+)\)\s*(?:\((?:M|F)\))?\s*$")


class HpRepairError(ValueError):
    """A set the repair cannot read (an unknown species / nature, a malformed stat line)."""


def team_seed(base_seed: int, team_id: str) -> int:
    """The per-team RNG seed: the first 8 bytes of ``sha256(f"{REPAIR_VERSION}|{base_seed}|{team_id}")``."""
    digest = hashlib.sha256(f"{REPAIR_VERSION}|{int(base_seed)}|{team_id}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def move_side(move_type: str) -> str:
    """The gen-3 category of a damaging move of ``move_type`` (any case)."""
    return SPECIAL if move_type.lower() in SPECIAL_TYPES else PHYSICAL


def gen3_stat(stat: str, base: int, iv: int, ev: int, level: int, nature_mult: float) -> int:
    """The gen-3 stat formula (``sim/battle.ts`` ``spreadModify`` / ``statModify``)."""
    core = (2 * base + iv + ev // 4) * level // 100
    if stat == "hp":
        return core + level + 10
    return int(math.floor((core + 5) * nature_mult))


@dataclass(frozen=True)
class SideVotes:
    """The three votes (+1 physical, -1 special, 0 none), the stat tie-break and the side they give."""
    evs: int
    nature: int
    moves: int
    atk_stat: int
    spa_stat: int
    side: Optional[str]          # PHYSICAL / SPECIAL, or None (an exact tie everywhere: both sides open)
    decided_by: str              # "votes" / "stat" / "open"


@dataclass(frozen=True)
class MonRepair:
    """One repaired set: where it is, what decided its side, the distribution sampled and the draw."""
    slot: int
    species: str
    votes: SideVotes
    prior_source: str                            # "species" / "format"
    restricted: Dict[str, float]                 # the renormalised distribution sampled from
    u: float                                     # the uniform draw
    hp_type: str                                 # capitalised, e.g. "Fire"
    ivs: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        v = self.votes
        return {"slot": self.slot, "species": self.species, "hp_type": self.hp_type,
                "side": v.side, "decided_by": v.decided_by,
                "votes": {"evs": v.evs, "nature": v.nature, "moves": v.moves},
                "atk_stat": v.atk_stat, "spa_stat": v.spa_stat,
                "prior_source": self.prior_source, "u": self.u,
                "restricted": {k: round(p, 9) for k, p in sorted(self.restricted.items())},
                "ivs": dict(self.ivs)}


@dataclass
class _Set:
    lines: List[str]
    species: str
    nature: Optional[str]
    level: int
    evs: Dict[str, int]
    ivs: Dict[str, int]
    iv_line: Optional[int]
    moves: List[Tuple[int, str]]                 # (line index, raw move name)


def _sign(x: float) -> int:
    return (x > 0) - (x < 0)


def _parse_stats(line: str, default: int) -> Dict[str, int]:
    body = line.split(":", 1)[1]
    out = {k: default for k in STAT_KEYS}
    for chunk in body.split("/"):
        chunk = chunk.strip()
        if not chunk:
            continue
        m = _STAT_PAIR.fullmatch(chunk)
        if m is None:
            raise HpRepairError(f"unreadable stat entry {chunk!r} in {line!r}")
        out[_LABEL_STAT[m.group(2).lower()]] = int(m.group(1))
    return out


def _parse_set(block: str) -> _Set:
    lines = block.split("\n")
    head = lines[0].split("@", 1)[0].strip()
    m = _NICK.match(head)
    name = m.group("species") if m and m.group("species").strip() not in ("M", "F") else re.sub(
        r"\s*\((?:M|F)\)\s*$", "", head)
    species = to_id_str(name)
    nature, level, evs, ivs, iv_line, moves = None, 100, {k: 0 for k in STAT_KEYS}, \
        {k: 31 for k in STAT_KEYS}, None, []
    for i, raw in enumerate(lines[1:], start=1):
        s = raw.strip()
        low = s.lower()
        if low.startswith("evs:"):
            evs = _parse_stats(s, 0)
        elif low.startswith("ivs:"):
            ivs, iv_line = _parse_stats(s, 31), i
        elif low.startswith("level:"):
            level = int(s.split(":", 1)[1].strip())
        elif low.endswith(" nature"):
            nature = to_id_str(s[: -len(" nature")])
        elif s.startswith("-"):
            moves.append((i, s[1:].strip()))
    return _Set(lines, species, nature, level, evs, ivs, iv_line, moves)


def _hp_move(st: _Set) -> Optional[Tuple[int, Optional[str]]]:
    """``(line index, typed type or None)`` of the set's Hidden Power, or None without one."""
    for i, mv in st.moves:
        m = _HP_LINE.match("- " + mv)
        if m:
            typed = (m.group("br") or m.group("sp") or "").strip()
            return i, (typed or None)
    return None


def is_target(st: _Set) -> bool:
    hp = _hp_move(st)
    return hp is not None and hp[1] is None and all(v == 31 for v in st.ivs.values())


def judge_side(species: str, evs: Dict[str, int], nature: Optional[str], other_moves: Sequence[str],
               ivs: Optional[Dict[str, int]] = None, level: int = 100) -> SideVotes:
    """The set's side (module docstring, step 2). ``other_moves`` are move ids, Hidden Power excluded."""
    sp = species_dex.get(species)
    if sp is None:
        raise HpRepairError(f"unknown species {species!r}")
    mult = {k: 1.0 for k in STAT_KEYS}
    if nature:
        nd = natures_dex.get(nature)
        if nd is None:
            raise HpRepairError(f"unknown nature {nature!r}")
        mult.update(nd.multipliers)
    ev_vote = _sign(evs.get("atk", 0) - evs.get("spa", 0))
    nat_vote = _sign((mult["atk"] > 1.0) + (mult["spa"] < 1.0) - (mult["spa"] > 1.0) - (mult["atk"] < 1.0))
    phys = spec = 0
    for mid in other_moves:
        md = moves_dex.get(mid)
        if md is None:
            raise HpRepairError(f"unknown move {mid!r}")
        if md.dex_category == "Status" or (md.base_power <= 0 and mid not in VARIABLE_POWER_ATTACKS):
            continue
        if move_side(md.type.name) == PHYSICAL:
            phys += 1
        else:
            spec += 1
    move_vote = _sign(phys - spec)
    iv = ivs or {k: 31 for k in STAT_KEYS}
    atk = gen3_stat("atk", sp.base_stats["atk"], iv["atk"], evs.get("atk", 0), level, mult["atk"])
    spa = gen3_stat("spa", sp.base_stats["spa"], iv["spa"], evs.get("spa", 0), level, mult["spa"])
    total = ev_vote + nat_vote + move_vote
    if total:
        side, how = (PHYSICAL if total > 0 else SPECIAL), "votes"
    elif atk != spa:
        side, how = (PHYSICAL if atk > spa else SPECIAL), "stat"
    else:
        side, how = None, "open"
    return SideVotes(ev_vote, nat_vote, move_vote, atk, spa, side, how)


def _on_side(t: str, side: Optional[str]) -> bool:
    return side is None or move_side(t) == side


def format_hp_usage() -> Dict[str, float]:
    """Format-wide Smogon HP-type usage: ``{type: sum over species of W * P(hiddenpower<type> in set)}``
    (= the chaos typed-HP counts summed; unnormalised). The fallback for a species with no mass on a side."""
    out: Dict[str, float] = {t: 0.0 for t in HP_TYPES}
    for sp, w in priors.species_usage().items():
        for mid, p in priors.moves(sp).items():
            if mid.startswith("hiddenpower") and mid[len("hiddenpower"):] in out:
                out[mid[len("hiddenpower"):]] += w * p
    return out


def side_distribution(species: str, side: Optional[str]) -> Tuple[str, Dict[str, float]]:
    """``(source, {type: p})``: Smogon's species HP usage restricted to ``side`` and renormalised, or the
    format-wide usage on that side when the species has none there."""
    for source, dist in (("species", priors.hidden_power(species)), ("format", format_hp_usage())):
        kept = {t: float(p) for t, p in dist.items() if p > 0 and _on_side(t, side)}
        tot = sum(kept.values())
        if tot > 0:
            return source, {t: p / tot for t, p in sorted(kept.items())}
    raise HpRepairError(f"no Smogon Hidden Power mass on side {side!r} for {species!r} nor format-wide")


def draw_type(dist: Dict[str, float], u: float) -> str:
    """Inverse-CDF draw over ``dist``'s types in alphabetical order (``u`` in [0, 1))."""
    acc = 0.0
    items = sorted(dist.items())
    for t, p in items:
        acc += p
        if u < acc:
            return t
    return items[-1][0]


def hp_iv_line(hp_type: str) -> str:
    ivs = HP_IVS[hp_type.capitalize()]
    return "IVs: " + " / ".join(f"{ivs[k]} {_STAT_LABEL[k]}" for k in STAT_KEYS)


def repair_team(text: str, seed: int) -> Tuple[str, List[MonRepair]]:
    """Repair every target set of a paste (module docstring). Returns ``(new paste, records)``; a paste
    with no target comes back byte-identical. Blocks are split on blank lines and re-joined as found."""
    rng = random.Random(seed)
    parts = re.split(r"(\n[ \t]*\n+)", text)
    records: List[MonRepair] = []
    slot = -1
    for pi in range(0, len(parts), 2):
        block = parts[pi]
        if not block.strip():
            continue
        slot += 1
        lead = block[: len(block) - len(block.lstrip("\n"))]
        st = _parse_set(block.lstrip("\n"))
        if not is_target(st):
            continue
        hp_line, _ = _hp_move(st)  # type: ignore[misc]
        others = [moves_dex.canonical_id(mv) for i, mv in st.moves if i != hp_line]
        votes = judge_side(st.species, st.evs, st.nature, others, st.ivs, st.level)
        source, dist = side_distribution(st.species, votes.side)
        u = rng.random()
        t = draw_type(dist, u)
        name = t.capitalize()
        lines = list(st.lines)
        indent = lines[hp_line][: len(lines[hp_line]) - len(lines[hp_line].lstrip())]
        lines[hp_line] = f"{indent}- Hidden Power [{name}]"
        if st.iv_line is not None:
            lines[st.iv_line] = hp_iv_line(name)
        else:
            first_move = min(i for i, _ in st.moves)
            lines.insert(first_move, hp_iv_line(name))
        parts[pi] = lead + "\n".join(lines)
        records.append(MonRepair(slot, st.species, votes, source, dist, u, name, dict(HP_IVS[name])))
    return "".join(parts), records


def hidden_power_type_of(block: str) -> Optional[str]:
    """The Hidden Power type a set plays (typed, else its IVs' gen-3 type), capitalised; None without HP."""
    st = _parse_set(block.strip("\n"))
    hp = _hp_move(st)
    if hp is None:
        return None
    if hp[1]:
        return hp[1].capitalize()
    order = ("Fighting", "Flying", "Poison", "Ground", "Rock", "Bug", "Ghost", "Steel",
             "Fire", "Water", "Grass", "Electric", "Psychic", "Ice", "Dragon", "Dark")
    x = sum((1 << i) * (st.ivs[k] % 2) for i, k in enumerate(("hp", "atk", "def", "spe", "spa", "spd")))
    return order[x * 15 // 63]
