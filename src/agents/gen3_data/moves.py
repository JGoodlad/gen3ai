"""Gen 3 move reference data — a *concept module* (the ``gen3_mechanics.py`` pattern).

Static, gen3-only facts about moves, keyed by move id: base power, type, category, accuracy,
never-miss, secondary/recoil flags. This is **reference data** — not state, not history. It never
changes during a battle, so it lives apart from the ``LiveView`` (current board) and ``TurnView``
(what happened) read-models. It is the single source of truth for "what is move X", shared by the
observation encoder and the reward function so neither re-loads or re-parses
``data/pokemon/gen3_moves.json`` itself.

Reached via the facade as ``gen3_data.moves`` (was ``agents.gen3_movedex``).

Borrow-discipline (project standard): we borrow poke-env's value-enums ``PokemonType`` and
``MoveCategory`` as canonical **keys / names only** — we never call methods on them or let them
carry data. The data is ours; the enums are just the keys it's filed under.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from agents.enums import MoveCategory, PokemonType

from . import _base

# Gen 1-3 physical/special split is by TYPE, not per-move (the modern per-move category arrived
# in gen 4). These eight types are special; everything else is physical. Mirrors poke-env's
# ``Move.SPECIAL_TYPES`` for gen<=3 so our derived category agrees with the sim.
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

    # --- gen3_move_effects_v1: action-aligned effect classification ---
    # Derived once in the acquisition tool from the field Showdown actually keys the
    # mechanic on (declarative or, for Belly Drum, a curated callback override) — see
    # tools/pokemon_data_extractor/sync.py:build_moves. These let the observation's
    # reactive block tell a setup move from a heal from a wasted status at the policy
    # head (where, for status moves, base_power=0 and the type multiplier are otherwise
    # identical). NOTE: Curse's setup is type-conditional (non-Ghost user only), so
    # `is_boost` is False here and the encoder resolves Curse live from the user's type.
    is_boost: bool = False        # raises the USER'S own stats (setup); incl. Belly Drum
    is_heal: bool = False         # restores the user's HP (flags.heal — incl. weather-heal/Rest/Wish)
    is_protect: bool = False      # Protect / Detect / Endure (stalling volatile)
    is_phaze: bool = False        # forces the foe to switch (Roar / Whirlwind)
    is_hazard: bool = False       # sets an entry hazard (gen3: Spikes)
    status_inflicted: Optional[str] = None  # major status this move's PURPOSE is to inflict, else None

    @property
    def is_damaging(self) -> bool:
        """A move deals direct damage iff it has base power. Mirrors the reward/obs convention
        (``base_power > 0``) rather than category, so fixed-/variable-power moves count as
        damaging and a 0-BP status move does not."""
        return self.base_power > 0


def _build(raw: Dict[str, dict]) -> Dict[str, MoveData]:
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
            is_boost=bool(v.get("isBoost", False)),
            is_heal=bool(v.get("isHeal", False)),
            is_protect=bool(v.get("isProtect", False)),
            is_phaze=bool(v.get("isPhaze", False)),
            is_hazard=bool(v.get("isHazard", False)),
            status_inflicted=v.get("status") or None,
        )
    return dex


raw = _base.singleton(lambda: _base.load_json("gen3_moves.json"))
_dex = _base.singleton(lambda: _build(raw()))


def get(move_id: Optional[str]) -> Optional[MoveData]:
    """Reference record for ``move_id``, or ``None`` if it isn't a known gen3 move.

    Use this on any path that must tolerate an unrevealed / foreign move id (the obs encoder's
    "known" flag, an opponent's move before it's revealed). Decision / reward code that only ever
    sees real revealed moves should prefer :func:`move_data`, which raises instead of silently
    returning ``None``."""
    if move_id is None:
        return None
    return _dex().get(move_id)


def move_data(move_id: str) -> MoveData:
    """Reference record for a move that MUST exist (crash-don't-drop). Raises ``KeyError`` on an
    unknown id — a deliberate tripwire for decision / reward code, which should never be handed a
    move that isn't in the gen3 dex."""
    md = get(move_id)
    if md is None:
        raise KeyError(f"Unknown gen3 move id: {move_id!r}")
    return md


def is_damaging(move_id: Optional[str]) -> bool:
    """True iff ``move_id`` is a known damaging move (``base_power > 0``). Unknown → False (an
    unrevealed move is not assumed to be damaging)."""
    md = get(move_id)
    return bool(md and md.is_damaging)
