"""The packed-set fields every information-lever consumer reads off a TRUE team: ids, Hidden Power typing, the spread.

A packed team is ``]``-joined entries ``name|species|item|ability|moves|nature|evs|gender|ivs|shiny|level|happiness``
(``evs`` / ``ivs`` comma lists in HP, Atk, Def, SpA, SpD, Spe order; an empty EV is 0, an empty IV 31).

THE ONE READER (GIGO 2026-10-15): a packed token is NOT an id until it is resolved. The engine resolves a move token
through Showdown's name-variant table (``wisp`` -> ``willowisp``), so a team may spell one; a reader that compared the
raw token with an id from play (a protocol line, a prior, a log) saw a different move, and a world carrying both
spellings ran Will-O-Wisp twice. Every consumer of a packed team's species / item / ability / moves reads them through
:func:`species_id`, :func:`item_id`, :func:`ability_id` and :func:`typed_moves` (never ``toid(f[4])`` / a raw split), and
a token the gen-3 data does not know is a THROWING defect (:func:`require_known`), never a silent miss.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from agents.gen3_data import abilities, items, moves, species
from utils.showdown_id import to_id_str

HP_TYPES = ("fighting", "flying", "poison", "ground", "rock", "bug", "ghost", "steel", "fire", "water",
            "grass", "electric", "psychic", "ice", "dragon", "dark")
STATS = ("hp", "atk", "def", "spa", "spd", "spe")       # the packed-team order


class PackedError(ValueError):
    pass


def toid(s: str) -> str:
    """The Showdown id of a name (``utils.showdown_id.to_id_str``: the ONE definition)."""
    return to_id_str(s)


def move_id(m: str) -> str:
    """A packed move token as its canonical move id (``wisp`` -> ``willowisp``; ``agents.gen3_data.moves``)."""
    return moves.canonical_id(m)


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


def hp_iv_masks(t: str) -> List[List[int]]:
    """EVERY 30/31 IV pattern (packed order) giving Hidden Power type ``t`` at 70 power — the canonical one
    (:func:`hp_ivs`, the most 31s) first, then by fewer 31s; a pool set may use any of them."""
    out: List[Tuple[int, List[int]]] = []
    for mask in range(64):
        b = [(mask >> i) & 1 for i in range(6)]          # hp, atk, def, spe, spa, spd
        v = sum(x << i for i, x in enumerate(b))
        if HP_TYPES[v * 15 // 63] == t:
            iv = dict(zip(("hp", "atk", "def", "spe", "spa", "spd"), (30 + x for x in b)))
            out.append((-sum(b), [iv[k] for k in STATS]))
    out.sort(key=lambda x: (x[0], x[1][::-1]))
    canon = hp_ivs(t)
    return [canon] + [m for _, m in out if m != canon]


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


def item_id(f: Sequence[str]) -> str:
    """The packed set's item id ("" = no item)."""
    return toid(f[2])


def ability_id(f: Sequence[str]) -> str:
    """The packed set's ability id ("" = the engine's default)."""
    return toid(f[3])


def typed_moves(f: Sequence[str]) -> List[str]:
    """The packed set's move ids as the engine runs them: each token resolved through the name-variant table
    (:func:`move_id`: ``wisp`` -> ``willowisp``), a bare ``hiddenpower`` typed by the set's IVs."""
    out = []
    for m in f[4].split(","):
        mid = move_id(m)
        if mid == "hiddenpower":
            mid = "hiddenpower" + hp_type_of(set_ivs(f))
        out.append(mid)
    return out


def unknown_ids(f: Sequence[str]) -> List[str]:
    """What the gen-3 data does not know of a packed set (empty = every id canonical): the species, a non-empty item or
    ability, and each move once resolved. The data carries a name-variant table for MOVES only, so a species / item /
    ability variant (Showdown's ``cb`` for Choice Band, ``deon`` for Deoxys) is unknown here, never silently read as a
    different id."""
    out = []
    if species.get(species_id(f)) is None:
        out.append(f"species {species_id(f)!r}")
    if item_id(f) and items.get(item_id(f)) is None:
        out.append(f"item {item_id(f)!r}")
    if ability_id(f) and abilities.get(ability_id(f)) is None:
        out.append(f"ability {ability_id(f)!r}")
    for m in typed_moves(f):
        if m and m != "hiddenpower" and moves.get(m) is None:
            out.append(f"move {m!r}")
    return out


def require_known(team: str, where: str = "") -> None:
    """THROWING: :class:`PackedError` when any set of the packed ``team`` holds an id the gen-3 data does not know
    (:func:`unknown_ids`). A true team enters the search builder and the lever injector through this check."""
    bad = []
    for i, e in enumerate(x for x in team.split("]") if x.strip()):
        f = parse_set(e)
        if unknown_ids(f):
            bad.append(f"slot {i} {species_id(f)}: {', '.join(unknown_ids(f))}")
    if bad:
        raise PackedError(f"{where + ': ' if where else ''}a packed team holds ids the gen-3 data does not know "
                          f"({'; '.join(bad)})")


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
