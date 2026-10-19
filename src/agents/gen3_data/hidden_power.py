"""THE ONE Hidden Power IV table and formula (gen 3) — every Python reader of "HP type -> IVs" or "IVs -> HP type /
power" imports this module (owner 2026-10-19: "only 1 impl for the HP determinization").

* **Type -> IVs** (:data:`HP_IVS`, :func:`ivs_packed`): Showdown's validator fill, ``fillStats(dex.types.get(hpType)
  .HPivs, 31)`` (``sim/team-validator.ts``; the ``HPivs`` rows of ``deps/pokemon-showdown/data/typechart.ts``: the
  stats listed there are 30, every other stat 31). Every row gives Hidden Power at 70 power. Pinned to the submodule by
  ``hidden_power_test.py``.
* **IVs -> type / power** (:func:`type_of`, :func:`power_of`): ``Dex.getHiddenPower``'s gen 3-5 branch
  (``sim/dex.ts``): the LSB (type) or second LSB (power) of each IV, weighted 1, 2, 4, 8, 16, 32 in the order
  HP, Atk, Def, **Spe**, SpA, SpD; ``type = FORMULA_TYPES[x * 15 // 63]``, ``power = x * 40 // 63 + 30``.
* :func:`iv_patterns`: every 30/31 IV pattern giving a type (all at 70 power), the canonical fill first.

The Rust engine's reverse formula (``src/rust_sim/src/state.rs`` ``hidden_power_type`` / ``hidden_power_bp``) is the
engine's own copy of ``getHiddenPower``; ``hidden_power_test.py`` pins its type list to :data:`FORMULA_TYPES`, and the
crate's ``hidden_power_fill_tests`` pin it to the same ``typechart.ts`` rows. ``hidden_power_single_impl_test.py``
FAILS on a second table or formula anywhere else in ``src/``.
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Sequence

#: The gen-3 formula's type order (index = ``x * 15 // 63``), lowercase ids.
FORMULA_TYPES = ("fighting", "flying", "poison", "ground", "rock", "bug", "ghost", "steel", "fire", "water",
                 "grass", "electric", "psychic", "ice", "dragon", "dark")
#: The stat keys in the packed-team / Showdown ``StatsTable`` order (HP, Atk, Def, SpA, SpD, Spe).
STATS = ("hp", "atk", "def", "spa", "spd", "spe")
#: The order the formula weights the IVs 1, 2, 4, 8, 16, 32 (Speed BEFORE SpA / SpD).
FORMULA_BIT_ORDER = ("hp", "atk", "def", "spe", "spa", "spd")

#: ``typechart.ts`` ``HPivs`` per type: the stats set to 30 (the rest are 31).
_HPIVS_SPARSE = {
    "Bug": ("atk", "def", "spd"), "Dark": (), "Dragon": ("atk",), "Electric": ("spa",),
    "Fighting": ("def", "spa", "spd", "spe"), "Fire": ("atk", "spa", "spe"), "Flying": ("hp", "atk", "def", "spa", "spd"),
    "Ghost": ("def", "spd"), "Grass": ("atk", "spa"), "Ground": ("spa", "spd"), "Ice": ("atk", "def"),
    "Poison": ("def", "spa", "spd"), "Psychic": ("atk", "spe"), "Rock": ("def", "spd", "spe"), "Steel": ("spd",),
    "Water": ("atk", "def", "spa"),
}
#: Capitalised type -> ``{stat: iv}`` — the validator's fill (Hidden Power at 70 power).
HP_IVS: Dict[str, Dict[str, int]] = {t: {k: (30 if k in low else 31) for k in STATS}
                                     for t, low in _HPIVS_SPARSE.items()}


def _x(ivs: Mapping[str, int], bit: int) -> int:
    return sum(((int(ivs.get(k, 31)) >> bit) & 1) << i for i, k in enumerate(FORMULA_BIT_ORDER))


def as_mapping(ivs: "Mapping[str, int] | Sequence[int]") -> Mapping[str, int]:
    """IVs as ``{stat: iv}``: a mapping passes through (a missing stat reads 31); a sequence is in :data:`STATS` order."""
    if isinstance(ivs, Mapping):
        return ivs
    seq = list(ivs)
    if len(seq) != 6:
        raise ValueError(f"6 IVs expected in {STATS} order, got {seq!r}")
    return dict(zip(STATS, seq))


def type_of(ivs: "Mapping[str, int] | Sequence[int]") -> str:
    """The gen-3 Hidden Power type (lowercase id) of IVs (a ``{stat: iv}`` mapping, missing = 31, or 6 in
    :data:`STATS` order)."""
    return FORMULA_TYPES[_x(as_mapping(ivs), 0) * 15 // 63]


def power_of(ivs: "Mapping[str, int] | Sequence[int]") -> int:
    """The gen-3 Hidden Power base power (30..70) of IVs (as :func:`type_of`)."""
    return _x(as_mapping(ivs), 1) * 40 // 63 + 30


def _canon(t: str) -> str:
    name = t.strip().capitalize()
    if name not in HP_IVS:
        raise KeyError(f"no Hidden Power type {t!r}")
    return name


def ivs_of(t: str) -> Dict[str, int]:
    """The canonical (validator-fill) IVs of Hidden Power type ``t`` (any case), a fresh ``{stat: iv}`` dict."""
    return dict(HP_IVS[_canon(t)])


def ivs_packed(t: str) -> List[int]:
    """:func:`ivs_of` as a list in :data:`STATS` (packed-team) order."""
    iv = HP_IVS[_canon(t)]
    return [iv[k] for k in STATS]


def iv_patterns(t: str) -> List[List[int]]:
    """EVERY 30/31 IV pattern (:data:`STATS` order) giving Hidden Power type ``t`` at 70 power: the canonical fill
    (:func:`ivs_packed`) first, then the rest by most 31s (ties: the reversed list, descending-stable)."""
    canon = ivs_packed(t)
    want = _canon(t).lower()
    out = []
    for mask in range(64):
        iv = {k: 30 + ((mask >> i) & 1) for i, k in enumerate(STATS)}
        if type_of(iv) == want:
            out.append([iv[k] for k in STATS])
    out.sort(key=lambda m: (-sum(x - 30 for x in m), m[::-1]))
    return [canon] + [m for m in out if m != canon]
