"""TEAM-SET OVERLAP — how much of an anchor read's team set we (or the opponent) have already seen.

Why it exists (2026-10-18). The Kakuna reads called ``away`` (Metamon's ``competitive/gen3ou``)
"Metamon's teams", a set our models never trained on. It is not: **19 of its 21 teams are EXACT copies
of teams in our 719-team training pool**, and 8 of the 72 ``home72`` teams are exact copies of ``away``
teams. A read's team set is therefore part of its regime, and every read now states how much of it
our training pool and the opponent's known team sets already hold (``summary.json["team_overlap"]``,
printed by :func:`main.anchors.results.render`).

**What is matched — a team is a set of six MON SETS.** One mon set, at two tiers:

* ``mon`` — the normalised SPECIES and its sorted, normalised MOVES. The tier the 2026-10-18 finding
  was taken at: "is this the same Pokemon doing the same four things".
* ``strict`` — ``mon`` plus ITEM, ABILITY, NATURE and the six EVs (a missing EV reads 0, a missing
  item / ability / nature reads ``""``).

Normalisation, each rule a fact rather than a judgement:

* a name is its Showdown id (:func:`utils.showdown_id.to_id_str`); a nickname ``Nick (Species)``
  resolves to the species, a ``(M)`` / ``(F)`` tag is dropped;
* **Hidden Power carries its TYPE**: ``Hidden Power [Fire]``, ``Hidden Power Fire`` and
  ``hiddenpowerfire`` are one move; a TYPELESS ``Hidden Power`` takes the type its IVs give under the
  gen-3 formula (no ``IVs:`` line = all 31 = Dark), which is the type the simulator plays;
* **formes**: a forme is matched by its OWN id (Deoxys-Speed is not Deoxys), EXCEPT a forme whose
  base stats and types equal its base species' (the Unown letters): it is the base species
  mechanically and matches as it. Formes share their base's national-dex ``num``, so nothing here is
  keyed by ``num`` (root ``CLAUDE.md`` § Data Dependencies).

**What is counted, against each reference set** (:func:`overlap`):

* ``exact_team`` / ``exact_team_strict`` — read teams whose six mon sets equal one reference team's;
* ``same_species6`` — read teams whose six species equal one reference team's;
* ``mons_seen`` / ``mons_seen_strict`` — read MONS whose mon set appears on any reference team;
* ``max_shared_mons`` — per read team, the most mon sets it shares with any ONE reference team, as a
  histogram over 0..6, and ``teams_sharing_ge4`` (the ``heldout72`` rejection threshold).
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from utils.showdown_id import to_id_str

OVERLAP_VERSION = "gen3_team_overlap_v1"

#: The gen-3 Hidden Power type order (index = the formula's result).
HP_TYPES = ("fighting", "flying", "poison", "ground", "rock", "bug", "ghost", "steel",
            "fire", "water", "grass", "electric", "psychic", "ice", "dragon", "dark")
_STATS = ("hp", "atk", "def", "spa", "spd", "spe")
_STAT_ALIASES = {"hp": "hp", "atk": "atk", "def": "def", "spa": "spa", "spd": "spd", "spe": "spe"}
#: The bit order the Hidden Power type formula reads the IVs in (HP, Atk, Def, Spe, SpA, SpD).
_HP_BIT_ORDER = ("hp", "atk", "def", "spe", "spa", "spd")

_NICK_RE = re.compile(r"^(?P<nick>.*?)\s*\((?P<species>[^()]+)\)\s*$")
_GENDER_RE = re.compile(r"\s*\((?:M|F)\)\s*$")
_HP_MOVE_RE = re.compile(r"^hidden\s*power\s*\[?\s*([a-z]*)\s*\]?\s*$", re.IGNORECASE)

MonKey = Tuple[str, Tuple[str, ...]]
StrictKey = Tuple[str, Tuple[str, ...], str, str, str, Tuple[int, ...]]


@dataclass(frozen=True)
class MonSet:
    species: str
    moves: Tuple[str, ...]          # sorted ids, Hidden Power typed
    item: str = ""
    ability: str = ""
    nature: str = ""
    evs: Tuple[int, ...] = (0, 0, 0, 0, 0, 0)   # hp atk def spa spd spe

    @property
    def key(self) -> MonKey:
        return (self.species, self.moves)

    @property
    def strict_key(self) -> StrictKey:
        return (self.species, self.moves, self.item, self.ability, self.nature, self.evs)


@dataclass(frozen=True)
class Team:
    label: str
    mons: Tuple[MonSet, ...]

    @property
    def key(self) -> FrozenSet[MonKey]:
        return frozenset(m.key for m in self.mons)

    @property
    def strict_key(self) -> FrozenSet[StrictKey]:
        return frozenset(m.strict_key for m in self.mons)

    @property
    def species6(self) -> FrozenSet[str]:
        return frozenset(m.species for m in self.mons)


def hidden_power_type(ivs: Dict[str, int]) -> str:
    """The gen-3 Hidden Power type of a spread (missing IVs read 31)."""
    bits = sum(((ivs.get(s, 31) & 1) << i) for i, s in enumerate(_HP_BIT_ORDER))
    return HP_TYPES[bits * 15 // 63]


def _species_id(name: str) -> str:
    """The species' Showdown id; a forme mechanically identical to its base matches as the base."""
    from agents import gen3_data

    sid = to_id_str(name)
    sd = gen3_data.species.get(sid)
    if sd is not None and sd.base_species:
        base = gen3_data.species.get(to_id_str(sd.base_species))
        if base is not None and base.base_stats == sd.base_stats and base.types == sd.types:
            return base.id
    return sid


def _parse_spread(text: str) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for part in text.split("/"):
        bits = part.strip().split()
        if len(bits) >= 2 and bits[0].isdigit():
            stat = _STAT_ALIASES.get(bits[1].lower())
            if stat:
                out[stat] = int(bits[0])
    return out


def parse_mon(block: str) -> Optional[MonSet]:
    lines = [ln.strip() for ln in block.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    head, _, item = lines[0].partition("@")
    head = _GENDER_RE.sub("", head.strip())
    m = _NICK_RE.match(head)
    species = m.group("species") if m and len(m.group("species").strip()) > 1 else head
    ability = nature = ""
    evs: Dict[str, int] = {}
    ivs: Dict[str, int] = {}
    raw_moves: List[str] = []
    for ln in lines[1:]:
        low = ln.lower()
        if ln.startswith("-"):
            raw_moves.append(ln[1:].strip().split("/")[0].strip())
        elif low.startswith("ability:"):
            ability = to_id_str(ln.split(":", 1)[1])
        elif low.startswith("evs:"):
            evs = _parse_spread(ln.split(":", 1)[1])
        elif low.startswith("ivs:"):
            ivs = _parse_spread(ln.split(":", 1)[1])
        elif low.endswith(" nature"):
            nature = to_id_str(ln[: -len(" nature")])
    moves = []
    for mv in raw_moves:
        hp = _HP_MOVE_RE.match(mv)
        if hp:
            typ = hp.group(1).lower() or hidden_power_type(ivs)
            moves.append(f"hiddenpower{typ}")
        else:
            moves.append(to_id_str(mv))
    return MonSet(species=_species_id(species), moves=tuple(sorted(moves)),
                  item=to_id_str(item), ability=ability, nature=nature,
                  evs=tuple(evs.get(s, 0) for s in _STATS))


def parse_team(text: str, label: str = "") -> Team:
    """A Showdown export (any blank-line layout, trailing blank lines included) as a :class:`Team`."""
    mons = [parse_mon(b) for b in re.split(r"\n\s*\n", text.strip())]
    return Team(label=label, mons=tuple(m for m in mons if m is not None))


def parse_items(items: Iterable[Tuple[str, str]]) -> List[Team]:
    return [parse_team(text, label) for label, text in items]


class Reference:
    """A reference team set, indexed for the per-team best-overlap count."""

    def __init__(self, teams: Sequence[Team]):
        self.teams = list(teams)
        self.keys = Counter(t.key for t in self.teams)
        self.strict_keys = Counter(t.strict_key for t in self.teams)
        self.species6 = Counter(t.species6 for t in self.teams)
        self.by_key: Dict[FrozenSet[MonKey], List[str]] = defaultdict(list)
        self.mon_index: Dict[MonKey, List[int]] = defaultdict(list)
        self.strict_mons = set()
        for i, t in enumerate(self.teams):
            self.by_key[t.key].append(t.label)
            for k in set(m.key for m in t.mons):
                self.mon_index[k].append(i)
            self.strict_mons.update(m.strict_key for m in t.mons)

    def max_shared(self, team: Team) -> Tuple[int, Optional[str]]:
        hits: Counter = Counter()
        for k in set(m.key for m in team.mons):
            for i in self.mon_index.get(k, ()):
                hits[i] += 1
        if not hits:
            return 0, None
        i, c = max(hits.items(), key=lambda kv: (kv[1], -kv[0]))
        return c, self.teams[i].label


def overlap(read: Sequence[Team], ref: Reference, *, n_examples: int = 5) -> Dict[str, Any]:
    """Every count of the module docstring, for ``read`` against ``ref``."""
    best = [ref.max_shared(t) for t in read]
    hist = Counter(c for c, _ in best)
    exact = [t for t in read if ref.keys[t.key]]
    return {
        "n_read_teams": len(read),
        "n_reference_teams": len(ref.teams),
        "exact_team": len(exact),
        "exact_team_strict": sum(1 for t in read if ref.strict_keys[t.strict_key]),
        "same_species6": sum(1 for t in read if ref.species6[t.species6]),
        "n_read_mons": sum(len(t.mons) for t in read),
        "mons_seen": sum(1 for t in read for m in t.mons if m.key in ref.mon_index),
        "mons_seen_strict": sum(1 for t in read for m in t.mons if m.strict_key in ref.strict_mons),
        "max_shared_mons_hist": {str(k): hist.get(k, 0) for k in range(7)},
        "teams_sharing_ge4": sum(1 for c, _ in best if c >= 4),
        "exact_examples": [[t.label, ref.by_key[t.key][0]] for t in exact[:n_examples]],
    }


def _dir_items(directory: Path) -> List[Tuple[str, str]]:
    from main.anchors.session import load_team_texts

    return load_team_texts(directory)


def reference_sets(plan: Any, cfg: Any) -> Dict[str, Any]:
    """``{name: (source, items) | {"missing": path}}`` — our training pool, then the opponent's
    KNOWN team sets: ``opponents.metamon.reference_team_dirs`` when the config names them, else
    every ``<cache_dir>/teams/<set>/<format>`` directory that is not one of OUR installed copies
    (``gen3ai_*``). Foul Play plays what it is handed and has no team set of its own."""
    from main.anchors.session import pool_team_texts
    from utils.paths import repo_path

    out: Dict[str, Any] = {
        "training_pool": (str(repo_path("data", "teams")),
                          [(f"pool{i:03d}", t) for i, t in enumerate(pool_team_texts())]),
    }
    kinds = {getattr(plan, "opponent_kind", "")}
    our_side = str(getattr(plan, "our_side", ""))
    if our_side.startswith("metamon:"):
        kinds.add("metamon")
    if "metamon" in kinds:
        meta = cfg.metamon
        dirs: Dict[str, str] = dict(getattr(meta, "reference_team_dirs", {}) or {})
        if not dirs:
            root = Path(meta.cache_dir) / "teams"
            fmt = getattr(plan, "battle_format", "gen3ou")
            if root.is_dir():
                for d in sorted(root.iterdir()):
                    if not d.name.startswith("gen3ai_") and (d / fmt).is_dir():
                        dirs[d.name] = str(d / fmt)
        for name, path in sorted(dirs.items()):
            p = Path(path)
            out[f"metamon:{name}"] = ((str(p), _dir_items(p)) if p.is_dir()
                                      else {"missing": str(p)})
    return out


def read_team_sets(plan: Any, cfg: Any) -> Dict[str, Tuple[str, List[Tuple[str, str]]]]:
    """``{"ours": (source, items), "theirs": (source, items)}`` — what each side of the read DRAWS."""
    from main.anchors.runner import our_team_items

    out: Dict[str, Tuple[str, List[Tuple[str, str]]]] = {}
    def peer_dir(kind: str) -> Optional[Path]:
        if kind not in ("metamon", "foulplay"):
            return None
        d = Path(cfg.opponent(kind).team_dir(plan.teamset, plan.battle_format))
        return d if d.is_dir() else None

    spec = getattr(plan, "our_team_spec", None) or {}
    our_side = str(getattr(plan, "our_side", "model"))
    if getattr(plan, "our_side_is_peer", False):
        # A peer on our side draws through ITS OWN team set (runner.our_peer_plan), not our source.
        d = peer_dir("metamon" if our_side.startswith("metamon:") else our_side)
        if d is not None:
            out["ours"] = (str(d), _dir_items(d))
    elif spec:
        out["ours"] = (str(spec.get("path", spec.get("kind"))), our_team_items(spec))
    d = peer_dir(getattr(plan, "opponent_kind", ""))
    if d is not None:
        out["theirs"] = (str(d), _dir_items(d))
    return out


def read_overlap(plan: Any, cfg: Any) -> Dict[str, Any]:
    """THE required ``team_overlap`` block of a read's ``summary.json``.

    Each side's team set is measured against every reference; a side whose team TEXTS equal the
    other's is reported once (``theirs_same_as_ours``). A reference that is configured but absent on
    this box is listed under ``missing_references`` — named, never a silent zero."""
    sides = read_team_sets(plan, cfg)
    if not sides:
        raise ValueError("team_overlap: neither side of this read names a team set")
    same = ("ours" in sides and "theirs" in sides
            and sorted(t for _, t in sides["ours"][1]) == sorted(t for _, t in sides["theirs"][1]))
    measured = {k: v for k, v in sides.items() if not (same and k == "theirs")}
    parsed = {k: parse_items(items) for k, (_, items) in measured.items()}
    against: Dict[str, Any] = {}
    missing: List[str] = []
    for name, ref in reference_sets(plan, cfg).items():
        if isinstance(ref, dict):
            missing.append(f"{name}={ref['missing']}")
            continue
        source, items = ref
        index = Reference(parse_items(items))
        against[name] = {"source": source,
                         **{side: overlap(teams, index) for side, teams in parsed.items()}}
    return {
        "version": OVERLAP_VERSION,
        "read_sets": {k: {"source": s, "n_teams": len(items)} for k, (s, items) in sides.items()},
        "theirs_same_as_ours": same,
        "against": against,
        "missing_references": missing,
    }


def render_lines(block: Optional[Dict[str, Any]]) -> List[str]:
    """The printed form: one line per (reference, side)."""
    if not block:
        return ["  overlap     (not computed — this read predates gen3_team_overlap_v1)"]
    lines = []
    for name, ref in block.get("against", {}).items():
        for side in ("ours", "theirs"):
            o = ref.get(side)
            if not o:
                continue
            n = o["n_read_teams"]
            lines.append(
                f"  overlap     {side} vs {name} ({o['n_reference_teams']}): exact {o['exact_team']}/{n}"
                f" (strict {o['exact_team_strict']}) · same 6 species {o['same_species6']}/{n}"
                f" · mon sets seen {o['mons_seen']}/{o['n_read_mons']}"
                f" · ≥4 shared with one team {o['teams_sharing_ge4']}/{n}")
    if block.get("theirs_same_as_ours"):
        lines.append("  overlap     (both sides draw the same team texts — reported once, as ours)")
    for m in block.get("missing_references", []):
        lines.append(f"  overlap     ⚠️  reference not on this box: {m}")
    return lines
