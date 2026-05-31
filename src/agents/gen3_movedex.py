"""Gen 3 move reference data — a *concept module* (the ``gen3_mechanics.py`` pattern).

Static, gen3-only facts about moves, keyed by move id: base power, type, category,
accuracy, never-miss, secondary/recoil flags. This is **reference data** — not state,
not history. It never changes during a battle, so it lives apart from the ``LiveView``
(current board) and ``TurnView`` (what happened) read-models. It is the single source of
truth for "what is move X", shared by the observation encoder and (soon) the reward
function so neither re-loads or re-parses ``data/pokemon/gen3_moves.json`` itself.

Borrow-discipline (project standard): we borrow poke-env's value-enums ``PokemonType`` and
``MoveCategory`` as canonical **keys / names only** — we never call methods on them or let
them carry data. The data is ours; the enums are just the keys it's filed under.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

from agents.enums import MoveCategory, PokemonType

# Resolve the data file relative to the repo root (src/agents/ -> repo root is parents[2])
# so the dex loads regardless of the process CWD, not only from the repo root.
_MOVES_JSON = Path(__file__).resolve().parents[2] / "data" / "pokemon" / "gen3_moves.json"

# Gen 1-3 physical/special split is by TYPE, not per-move (the modern per-move category
# arrived in gen 4). These eight types are special; everything else is physical. Mirrors
# poke-env's ``Move.SPECIAL_TYPES`` for gen<=3 so our derived category agrees with the sim.
_GEN3_SPECIAL_TYPES = frozenset({
    PokemonType.FIRE, PokemonType.WATER, PokemonType.GRASS, PokemonType.ELECTRIC,
    PokemonType.ICE, PokemonType.PSYCHIC, PokemonType.DARK, PokemonType.DRAGON,
})

# The gen3 data stores Curse's type as "???" (typeless). poke-env spells that enum member
# THREE_QUESTION_MARKS, so map it explicitly rather than via name-upper.
_TYPE_ALIASES = {"???": PokemonType.THREE_QUESTION_MARKS}


def _resolve_type(name: str) -> PokemonType:
    if name in _TYPE_ALIASES:
        return _TYPE_ALIASES[name]
    return PokemonType[name.upper()]


def _derive_category(base_power: int, move_type: PokemonType) -> MoveCategory:
    """Gen3 category from base power + type: a 0-power move is STATUS; a damaging move is
    SPECIAL or PHYSICAL by its type (the gen 1-3 type-based split)."""
    if base_power <= 0:
        return MoveCategory.STATUS
    return MoveCategory.SPECIAL if move_type in _GEN3_SPECIAL_TYPES else MoveCategory.PHYSICAL


@dataclass(frozen=True)
class MoveData:
    """Immutable reference record for one gen3 move. Primitives + borrowed enums."""

    id: str
    num: int
    base_power: int
    type: PokemonType        # borrowed enum — used as a key/name, never called
    category: MoveCategory   # borrowed enum — used as a key/name, never called
    accuracy: int            # raw percent (0–100); the data stores an int
    never_miss: bool         # bypasses the accuracy/evasion check (Swift, Aerial Ace, …)
    has_secondary: bool
    has_recoil: bool

    @property
    def is_damaging(self) -> bool:
        """A move deals direct damage iff it has base power. Mirrors the reward/obs
        convention (``base_power > 0``) rather than category, so fixed-/variable-power
        moves count as damaging and a 0-BP status move does not."""
        return self.base_power > 0


def _build() -> Dict[str, MoveData]:
    if not _MOVES_JSON.exists():
        raise FileNotFoundError(f"CRITICAL: move data missing: {_MOVES_JSON}")
    raw = json.loads(_MOVES_JSON.read_text())
    if not raw:
        raise ValueError(f"CRITICAL: move data empty: {_MOVES_JSON}")
    dex: Dict[str, MoveData] = {}
    for mid, v in raw.items():
        base_power = int(v.get("basePower", 0))
        mtype = _resolve_type(str(v.get("type", "Normal")))
        dex[mid] = MoveData(
            id=mid,
            num=int(v.get("num", 0)),
            base_power=base_power,
            type=mtype,
            category=_derive_category(base_power, mtype),
            accuracy=int(v.get("accuracy", 100)),
            never_miss=bool(v.get("never_miss", False)),
            has_secondary=bool(v.get("hasSecondary", False)),
            has_recoil=bool(v.get("hasRecoil", False)),
        )
    return dex


_DEX: Optional[Dict[str, MoveData]] = None


def _dex() -> Dict[str, MoveData]:
    global _DEX
    if _DEX is None:
        _DEX = _build()
    return _DEX


def get(move_id: Optional[str]) -> Optional[MoveData]:
    """Reference record for ``move_id``, or ``None`` if it isn't a known gen3 move.

    Use this on any path that must tolerate an unrevealed / foreign move id (the obs
    encoder's "known" flag, an opponent's move before it's revealed). Decision / reward
    code that only ever sees real revealed moves should prefer :func:`move_data`, which
    raises instead of silently returning ``None``."""
    if move_id is None:
        return None
    return _dex().get(move_id)


def move_data(move_id: str) -> MoveData:
    """Reference record for a move that MUST exist (crash-don't-drop). Raises ``KeyError``
    on an unknown id — a deliberate tripwire for decision / reward code, which should
    never be handed a move that isn't in the gen3 dex."""
    md = get(move_id)
    if md is None:
        raise KeyError(f"Unknown gen3 move id: {move_id!r}")
    return md


def is_damaging(move_id: Optional[str]) -> bool:
    """True iff ``move_id`` is a known damaging move (``base_power > 0``). Unknown → False
    (an unrevealed move is not assumed to be damaging)."""
    md = get(move_id)
    return bool(md and md.is_damaging)
