"""THE MATURITY BATTERY's FROZEN STATE BANK: the k-stratified draw of `design_hidden_info_binding.md` §12.13.2 /
§12.13.7, generalised to opponent STRATA (the competent-opponent states E2 recommends carrying, §16.12), and the
LEAN subset nested inside the FULL bank.

The rule (pure; no model, no outcome): the candidates are the eligible pool of the drawing checkpoint's last three
eval cycles (a free move round, turn 2-40, >= 3 legal actions, gate ① passed) minus its top-2 ties (rule 8). The
cells are (k bin, stratum), processed k bin by k bin in :data:`K_BINS` order and, within a k bin, stratum by stratum
in the order given. Within a cell the candidates are taken in ``sha256(salt + id)`` order, skipping a candidate whose
battle already holds ``per_battle_per_bin`` state(s) in this k bin or ``per_battle`` states in all (counted over the
cells processed so far), until the cell holds its target. A cell that cannot be filled REFUSES (:class:`BankShort`);
the rule is never relaxed. A battle belongs to one stratum (its opponent), so strata never compete for a battle.

A SUPPLEMENTARY bank (the plateau bank, PREREGISTRATION §2.5 of the X26 battery) names ``exclude_battles``: every
candidate of an excluded battle (another bank's) is dropped BEFORE the draw, so the two banks share no battle.

The LEAN subset is the FIRST ``n`` states of each cell in that same selection order (:func:`nested_subset`), so a LEAN
bank is a subset of the FULL one and a later FULL read re-uses every LEAN row.

PURE: stdlib only, no ``main`` / ``agents`` import (the pin-side bank driver loads this file by path).
"""
from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

#: §12.13.1's frozen stage buckets (the same edges as Tier 1's ``t1_common``; `maturity_bank_test` checks equality)
K_BINS: tuple[str, ...] = ("1-2", "3", "4", "5", "6")
TURN_BINS: tuple[str, ...] = ("<=7", "8-14", "15-21", ">=22")
MAT_BINS: tuple[str, ...] = ("<=7", "8-9", "10-11", "12")

Cell = tuple[str, str]


class BankShort(RuntimeError):
    """A (k bin, stratum) cell could not be filled under the rule."""


@dataclass(frozen=True)
class Candidate:
    id: str
    battle: str
    k: int
    stratum: str


def k_bin(k: int) -> str:
    if not 1 <= int(k) <= 6:
        raise ValueError(f"k = {k} outside 1..6")
    return "1-2" if k <= 2 else str(int(k))


def turn_bin(t: int) -> str:
    return "<=7" if t <= 7 else "8-14" if t <= 14 else "15-21" if t <= 21 else ">=22"


def mat_bin(m: int) -> str:
    return "<=7" if m <= 7 else "8-9" if m <= 9 else "10-11" if m <= 11 else "12"


def order_key(salt: str, did: str) -> str:
    return hashlib.sha256((salt + did).encode()).hexdigest()


def draw(cands: Iterable[Candidate], targets: Mapping[Cell, int], salt: str, strata: Sequence[str],
         per_battle_per_bin: int = 1, per_battle: int = 3,
         exclude_battles: Iterable[str] = ()) -> dict[Cell, list[str]]:
    """{(k bin, stratum): [ids in SELECTION order]} under the module's rule. ``targets`` names every cell drawn;
    ``strata`` fixes their order within a k bin; a candidate of a battle in ``exclude_battles`` is never drawn."""
    ex = set(exclude_battles)
    cs = [c for c in cands if c.battle not in ex]
    ids = [c.id for c in cs]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate candidate ids")
    unknown = {s for _b, s in targets} - set(strata)
    if unknown:
        raise ValueError(f"targets name strata {sorted(unknown)} not in {list(strata)}")
    st_of_b: dict[str, str] = {}
    for c in cs:
        if st_of_b.setdefault(c.battle, c.stratum) != c.stratum:
            raise ValueError(f"battle {c.battle} in two strata")
    per_bin_bat: Counter = Counter()
    per_bat: Counter = Counter()
    out: dict[Cell, list[str]] = {}
    for b in K_BINS:
        for s in strata:
            want = targets.get((b, s))
            if want is None:
                continue
            pool = sorted((c for c in cs if k_bin(c.k) == b and c.stratum == s), key=lambda c: order_key(salt, c.id))
            got: list[str] = []
            for c in pool:
                if len(got) >= want:
                    break
                if per_bin_bat[(b, c.battle)] >= per_battle_per_bin or per_bat[c.battle] >= per_battle:
                    continue
                per_bin_bat[(b, c.battle)] += 1
                per_bat[c.battle] += 1
                got.append(c.id)
            if len(got) < want:
                raise BankShort(f"cell (k {b}, {s}) holds {len(got)} of {want} under the rule "
                                f"({len(pool)} candidates)")
            out[(b, s)] = got
    return out


def nested_subset(selected: Mapping[Cell, Sequence[str]], n: int) -> dict[Cell, list[str]]:
    """The first ``n`` ids of each cell in selection order (the LEAN bank inside the FULL one)."""
    out = {}
    for cell, xs in selected.items():
        if len(xs) < n:
            raise BankShort(f"cell {cell} holds {len(xs)} < {n}")
        out[cell] = list(xs[:n])
    return out


def ids_sha(ids: Iterable[str]) -> str:
    """sha256 of the sorted ids concatenated (E2's ``states_sha256`` convention)."""
    return hashlib.sha256("".join(sorted(ids)).encode()).hexdigest()


def flat(selected: Mapping[Cell, Sequence[str]]) -> list[str]:
    return sorted(x for xs in selected.values() for x in xs)
