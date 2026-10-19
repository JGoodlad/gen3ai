"""THE MATCHUP PLAN of the team-strength meter: which team meets which, in which unit, at which seed.

PILOT A (one checkpoint on BOTH sides, the in-process Rust core, ``main.h2h``'s mirrored pairs). Every RATED team
(the union of the registered sets, by identity) plays ``pairs_per_team`` mirrored pairs, each against a different
REFERENCE-panel team (never itself): the reference ids, sorted, shuffled by ``random.Random(f"{seed}:{team id}")``,
taken in that order (cycling when more pairs than reference teams are asked). A pair is two games on one battle seed:
game ``2k`` the player (seat p1) pilots the rated team against the reference team, game ``2k+1`` the teams are handed
over, so each team plays each seat once and team-draw luck cancels inside the pair. All pairs are ordered by
``random.Random(f"{seed}:order")`` (so an interrupted run leaves every set partly rated, never one set unrated) and
cut into UNITS of ``unit_pairs``; unit ``u`` plays at the per-game seed base ``h2h.cycle_seed(seed, key, u)`` with the
plan's key (:func:`plan_key`). A unit is the durable, resumable grain.

PILOT B (Metamon's Kakuna on both sides, the anchors' B-D9 asymmetric PAIR CELL). One unit = one ``main.anchors``
invocation of ``unit_games`` role-balanced games, Kakuna on a rated set's SUBSET (``--our-teamset``) against Kakuna on
the reference panel (``--teamset``); Metamon draws each side's team at random (``--team-seed``), and the capture says
which team each side played. Unit ``u`` of set ``s`` plays at ``--team-seed = --seed-base =`` :func:`kakuna_seed`.
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

SCHEMA = "gen3_tsr_plan_v1"


@dataclass(frozen=True)
class Pair:
    subject: str        # the RATED team (the pair's owner: its cluster in the bootstrap)
    ref: str            # the reference-panel team it meets
    index: int          # the subject's j-th pair


@dataclass(frozen=True)
class Unit:
    index: int
    seed: int
    pairs: Tuple[Pair, ...]

    @property
    def name(self) -> str:
        return f"x26_{self.index:04d}"


def plan_key(reg_digest: str) -> str:
    """The schedule key of a plan: its registration's digest (a different panel or set list is another schedule)."""
    return "tsr:" + hashlib.blake2b(f"{SCHEMA}:{reg_digest}".encode(), digest_size=8).hexdigest()


def subject_pairs(subjects: Sequence[str], refs: Sequence[str], pairs_per_team: int, seed: int) -> List[Pair]:
    refs_sorted = sorted(refs)
    if len(set(refs_sorted)) != len(refs_sorted):
        raise ValueError("the reference panel holds a team twice")
    out: List[Pair] = []
    for t in sorted(set(subjects)):
        pool = [r for r in refs_sorted if r != t]
        if not pool:
            raise ValueError(f"team {t}: no reference team other than itself")
        random.Random(f"{int(seed)}:{t}").shuffle(pool)
        for j in range(int(pairs_per_team)):
            out.append(Pair(subject=t, ref=pool[j % len(pool)], index=j))
    return out


def plan_units(subjects: Sequence[str], refs: Sequence[str], *, pairs_per_team: int, unit_pairs: int, seed: int,
               key: str) -> List[Unit]:
    from main.h2h.play import cycle_seed

    pairs = subject_pairs(subjects, refs, pairs_per_team, seed)
    random.Random(f"{int(seed)}:order").shuffle(pairs)
    units = []
    for u, lo in enumerate(range(0, len(pairs), int(unit_pairs))):
        units.append(Unit(index=u, seed=cycle_seed(int(seed), key, u), pairs=tuple(pairs[lo:lo + int(unit_pairs)])))
    return units


def kakuna_seed(seed: int, set_name: str, unit: int) -> int:
    """A unit's ``--team-seed`` (= ``--seed-base``): 31 bits of a keyed hash (Metamon seeds a Python ``random``)."""
    d = hashlib.blake2b(f"{SCHEMA}:kakuna:{int(seed)}:{set_name}:{int(unit)}".encode(), digest_size=8).digest()
    return int.from_bytes(d, "big") & 0x7FFFFFFF


def kakuna_units(sets: Sequence[str], units_per_set: int, seed: int) -> List[Tuple[str, str, int]]:
    """``(unit name, set, team seed)``, interleaved set by set (unit 0 of every set first)."""
    out = []
    for u in range(int(units_per_set)):
        for s in sets:
            out.append((f"kak_{s}_{u:03d}", s, kakuna_seed(seed, s, u)))
    return out


def digest(obj: Mapping[str, Any]) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def summary(units: Sequence[Unit]) -> Dict[str, Any]:
    per: Dict[str, int] = {}
    as_ref: Dict[str, int] = {}
    for u in units:
        for p in u.pairs:
            per[p.subject] = per.get(p.subject, 0) + 1
            as_ref[p.ref] = as_ref.get(p.ref, 0) + 1
    return {"units": len(units), "pairs": sum(len(u.pairs) for u in units), "subjects": len(per),
            "pairs_per_subject": sorted(set(per.values())), "ref_pairs_min_max": [min(as_ref.values()),
                                                                                 max(as_ref.values())]}
