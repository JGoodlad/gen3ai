"""The packed-set fields every information-lever consumer reads off a TRUE team: ids, Hidden Power typing, the spread.

A packed team is ``]``-joined entries ``name|species|item|ability|moves|nature|evs|gender|ivs|shiny|level|happiness``
(``evs`` / ``ivs`` comma lists in HP, Atk, Def, SpA, SpD, Spe order; an empty EV is 0, an empty IV 31).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

HP_TYPES = ("fighting", "flying", "poison", "ground", "rock", "bug", "ghost", "steel", "fire", "water",
            "grass", "electric", "psychic", "ice", "dragon", "dark")
STATS = ("hp", "atk", "def", "spa", "spd", "spe")       # the packed-team order


class PackedError(ValueError):
    pass


def toid(s: str) -> str:
    return "".join(ch for ch in s.lower() if ch.isalnum())


def hp_type_of(ivs: Sequence[int]) -> str:
    """Gen-3 Hidden Power type from IVs given in the packed order (HP, Atk, Def, SpA, SpD, Spe)."""
    iv = dict(zip(STATS, ivs))
    v = sum((iv[k] & 1) << i for i, k in enumerate(("hp", "atk", "def", "spe", "spa", "spd")))
    return HP_TYPES[v * 15 // 63]


def hp_ivs(t: str) -> List[int]:
    """30/31 IVs (Hidden Power at 70 power) giving type ``t``, with the most 31s, in the packed order."""
    best: Optional[List[int]] = None
    for mask in range(64):
        b = [(mask >> i) & 1 for i in range(6)]          # hp, atk, def, spe, spa, spd
        v = sum(x << i for i, x in enumerate(b))
        if HP_TYPES[v * 15 // 63] == t and (best is None or sum(b) > sum(best)):
            best = b
    if best is None:
        raise PackedError(f"no Hidden Power IVs for type {t!r}")
    iv = dict(zip(("hp", "atk", "def", "spe", "spa", "spd"), (30 + x for x in best)))
    return [iv[k] for k in STATS]


def parse_set(entry: str) -> List[str]:
    f = entry.split("|")
    return f + [""] * (12 - len(f)) if len(f) < 12 else f


def set_ivs(f: Sequence[str]) -> List[int]:
    raw = f[8].split(",") if f[8] else [""] * 6
    return [int(x) if x else 31 for x in raw]


def set_evs(f: Sequence[str]) -> List[int]:
    raw = f[6].split(",") if f[6] else [""] * 6
    return [int(x) if x else 0 for x in raw]


def is_hp(m: str) -> bool:
    return m.startswith("hiddenpower")


def species_id(f: Sequence[str]) -> str:
    return toid(f[1] or f[0])


def typed_moves(f: Sequence[str]) -> List[str]:
    """The packed set's move ids, a bare ``hiddenpower`` typed by its IVs (what the engine runs)."""
    out = []
    for m in f[4].split(","):
        mid = toid(m)
        if mid == "hiddenpower":
            mid = "hiddenpower" + hp_type_of(set_ivs(f))
        out.append(mid)
    return out


@dataclass(frozen=True)
class Spread:
    """Nature, EVs, IVs (packed order); the IVs encode the Hidden Power type."""

    nature: str
    evs: Tuple[int, ...]
    ivs: Tuple[int, ...]

    @staticmethod
    def of(f: Sequence[str]) -> "Spread":
        return Spread(f[5], tuple(set_evs(f)), tuple(set_ivs(f)))

    def key(self) -> Tuple[str, Tuple[int, ...]]:
        """What the PRIOR knows of a spread: nature + EVs (its IVs follow the Hidden Power type)."""
        return toid(self.nature), self.evs

    def same(self, other: "Spread") -> bool:
        return self.key() == other.key() and self.ivs == other.ivs

    def write(self, f: List[str]) -> None:
        f[5] = self.nature
        f[6] = ",".join(str(x) if x else "" for x in self.evs)
        f[8] = "" if all(x == 31 for x in self.ivs) else ",".join(str(x) for x in self.ivs)
