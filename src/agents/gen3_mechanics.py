"""
Gen 3 OU mechanics — single source of truth.

All type immunities, status conditions, volatile effects, move categories, stat
boost helpers, and related functions used across reward signals, heuristic
opponents, observation encoders, and loggers live here.  Import from this module
instead of re-declaring these facts in individual files.
"""
from __future__ import annotations

import functools

import numpy as np
from poke_env.battle.effect import Effect
from poke_env.battle.pokemon_type import PokemonType
from poke_env.battle.status import Status
from poke_env.data import GenData

# ---------------------------------------------------------------------------
# Type effectiveness
# ---------------------------------------------------------------------------

# Abilities that modify incoming move-type damage in Gen 3. Maps ability name to
# {move_type: multiplier}. Multiplier 0.0 = full immunity (Levitate / Volt Absorb /
# Water Absorb / Flash Fire); 0.5 = halved damage (Thick Fat). Each entry mirrors
# the |-immune| or |-resisted| message Showdown emits for that ability, so callers
# comparing against battle.*_last_effectiveness see consistent values.
#
# Gen 3-only — Heatproof / Filter / Solid Rock are Gen 4+, and Lightning Rod doesn't
# grant immunity in Gen 3 singles. Wonder Guard depends on the full type-chart
# product (not a single type) so it's handled separately in effective_multiplier.
ABILITY_TYPE_MULTIPLIER: dict[str, dict[PokemonType, float]] = {
    "levitate":    {PokemonType.GROUND:   0.0},
    "voltabsorb":  {PokemonType.ELECTRIC: 0.0},
    "waterabsorb": {PokemonType.WATER:    0.0},
    "flashfire":   {PokemonType.FIRE:     0.0},
    "thickfat":    {PokemonType.ICE:      0.5, PokemonType.FIRE: 0.5},
}

_type_chart = GenData.from_gen(3).type_chart

# Types with no entry in the chart — attacking/defending as one of these is a no-op
# (×1). Hoisted to a module constant so the hot path never reconstructs the set literal
# `PokemonType.damage_multiplier` builds on every call.
_NULL_TYPES: frozenset[PokemonType] = frozenset(
    {PokemonType.THREE_QUESTION_MARKS, PokemonType.STELLAR}
)
_REAL_TYPES: tuple[PokemonType, ...] = tuple(t for t in PokemonType if t not in _NULL_TYPES)

# Dense, PokemonType-keyed single-type effectiveness table, precomputed ONCE at import:
#   _CHART[attacking_type][defending_type] == _type_chart[defending.name][attacking.name].
# Indexing this avoids the per-call `.name` enum-attribute access, the set-literal
# construction, and most of the enum hashing that dominated the matchup-encoder profile.
_CHART: dict[PokemonType, dict[PokemonType, float]] = {
    att: {deff: _type_chart[deff.name][att.name] for deff in _REAL_TYPES}
    for att in _REAL_TYPES
}


@functools.lru_cache(maxsize=None)
def _eff_cached(
    move_type: PokemonType,
    type_1: PokemonType,
    type_2: PokemonType | None,
    ability: str,
    frozen: bool,
) -> float:
    """Pure, memoized effectiveness primitive. Keyed on the only things the result
    depends on — attacking type, the defender's two types, its (lowercased) ability, and
    whether it's frozen (the sole status that matters, for Flash Fire). The key space is
    tiny (≤18 types² × a handful of abilities), so after warmup every matchup cell is a
    dict lookup. Byte-identical to `PokemonType.damage_multiplier` × the ability modifier
    for real types; an unknown/typeless type resolves to a neutral ×1 (every real-battle
    input is a real PokemonType, so the fallback only ever fires for the null types and for
    test mocks — never in production).
    """
    row = _CHART.get(move_type)
    if row is None or type_1 in _NULL_TYPES:
        base = 1.0
    else:
        base = row.get(type_1, 1.0)
        if type_2 is not None:
            base *= row.get(type_2, 1.0)
    if ability == "wonderguard":
        return base if base > 1.0 else 0.0
    if ability == "flashfire" and frozen:
        return base
    return base * ABILITY_TYPE_MULTIPLIER.get(ability, {}).get(move_type, 1.0)


def effective_multiplier_by_types(
    move_type: PokemonType,
    type_1: PokemonType,
    type_2: PokemonType | None = None,
    ability: str | None = None,
    status: Status | None = None,
) -> float:
    """Value-based effectiveness — the cacheable core of `effective_multiplier`.

    Use this when you already hold the defender's types/ability/status (e.g. the matchup
    encoder hoists them out of its inner loop) so the hot path never touches a poke-env
    `Pokemon` property. Normalizes the ability/status into the canonical cache key, then
    defers to `_eff_cached`.
    """
    return _eff_cached(
        move_type, type_1, type_2, (ability or "").lower(), status == Status.FRZ
    )


def effective_multiplier(move_type: PokemonType, mon) -> float:
    """Damage-type multiplier of move_type vs mon, including Gen 3 ability modifiers.

    Returns the *raw* multiplier — 0×, 0.25×, 0.5×, 1×, 2×, 4× are all possible.
    Multiplies the raw type-chart value by the mon's ability modifier from
    ABILITY_TYPE_MULTIPLIER (default 1.0 = no effect).

    Callers comparing against `battle.*_last_effectiveness` (which poke-env
    bucketizes to {0.0, 0.5, 1.0, 2.0}) must pipe this through
    `bucket_effectiveness()` first — otherwise 4× HP Grass on Water/Ground
    won't equal the 2.0 the protocol reports.

    Gen 3 quirk: Flash Fire does NOT activate when the target is frozen — the
    incoming Fire move falls through and is resisted normally (0.5× from Fire-type).
    See pokemon-showdown/data/mods/gen3/abilities.ts.

    Wonder Guard (Shedinja): only super-effective moves do damage; everything else
    is fully absorbed. Returns the raw type-chart product for SE hits, else 0.

    Thin object-based wrapper over `effective_multiplier_by_types`: reads the four
    attributes the result depends on off `mon`, then defers to the memoized primitive.
    """
    return effective_multiplier_by_types(
        move_type,
        mon.type_1,
        mon.type_2,
        getattr(mon, "ability", None),
        getattr(mon, "status", None),
    )


def bucket_effectiveness(mult: float) -> float:
    """Bucket a raw type multiplier into Showdown's reported effectiveness.

    Showdown emits one of |-immune| / |-resisted| / nothing-for-neutral /
    |-supereffective| per damaging hit, which poke-env stores as
    {0.0, 0.5, 1.0, 2.0}. Raw multipliers like 0.25× or 4× collapse into the
    resisted / super-effective buckets. Use this when comparing against
    `battle.*_last_effectiveness` so 4× SE hits match 2.0 and 0.25× double-
    resisted hits match 0.5.
    """
    if mult == 0.0:
        return 0.0
    if mult < 1.0:
        return 0.5
    if mult == 1.0:
        return 1.0
    return 2.0


# ---------------------------------------------------------------------------
# Status conditions
# ---------------------------------------------------------------------------

# Status moves → types immune to the status they inflict (Gen 3).
# stunspore is Normal-type in Gen 3 (reclassified Grass in Gen 6) — no type immunity.
# glare (Normal-type) — no type immunity.
# sleep moves (spore, sleeppowder, hypnosis, lovelykiss, yawn) — no type immunity.
STATUS_MOVE_IMMUNITY: dict[str, frozenset] = {
    "thunderwave":  frozenset({PokemonType.GROUND}),
    "toxic":        frozenset({PokemonType.STEEL, PokemonType.POISON}),
    "poisongas":    frozenset({PokemonType.STEEL, PokemonType.POISON}),
    "poisonpowder": frozenset({PokemonType.STEEL, PokemonType.POISON}),
    "willowisp":    frozenset({PokemonType.FIRE}),
}

# Volatile effects worth surfacing in logs and reward signals (Gen 3 relevant subset).
NOTABLE_EFFECTS: tuple[Effect, ...] = (
    Effect.TAUNT, Effect.CONFUSION, Effect.ENCORE, Effect.ATTRACT,
    Effect.DISABLE, Effect.SUBSTITUTE,
)


def is_status_move_immune(move_id: str, mon) -> bool:
    """True if mon's types make it immune to the status inflicted by move_id,
    or if the mon already has a status condition (can't be double-statused)."""
    immune_types = STATUS_MOVE_IMMUNITY.get(move_id, frozenset())
    mon_types = {getattr(mon, "type_1", None), getattr(mon, "type_2", None)} - {None}
    return bool(immune_types & mon_types) or getattr(mon, "status", None) is not None


def mon_status_str(mon) -> str | None:
    """Permanent status + notable volatile effects as a compact string, or None.

    Examples: "BRN", "taunt", "PAR, confusion"
    """
    if mon is None:
        return None
    parts = []
    status = getattr(mon, "status", None)
    if status is not None:
        parts.append(status.name)
    effects = getattr(mon, "effects", {})
    for eff in NOTABLE_EFFECTS:
        if eff in effects:
            parts.append(eff.name.lower())
    return ", ".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Move category sets
# ---------------------------------------------------------------------------

PHAZING_MOVES: frozenset[str] = frozenset({"roar", "whirlwind"})

INVULNERABLE_MOVES: frozenset[str] = frozenset({"protect", "detect", "endure"})

STATUS_MOVES: frozenset[str] = frozenset({
    "toxic", "willowisp", "thunderwave", "stunspore",
    "sleeppowder", "spore", "glare", "poisonpowder",
})

RECOVERY_MOVES: frozenset[str] = frozenset({
    "recover", "softboiled", "moonlight", "morningsun", "synthesis",
    "rest", "wish", "slackoff", "milkdrink",
})

HAZARD_CLEAR_MOVES: frozenset[str] = frozenset({"rapidspin"})

SETUP_MOVES: frozenset[str] = frozenset({
    "swordsdance", "calmmind", "dragondance", "nastyplot",
    "bulkup", "curse", "meditate", "sharpen",
})


# ---------------------------------------------------------------------------
# Stat boosts
# ---------------------------------------------------------------------------

# Canonical stat order for boost arrays — indices 0-6 match BOOST_STATS.
BOOST_STATS: tuple[str, ...] = ("atk", "def", "spa", "spd", "spe", "accuracy", "evasion")
BOOST_DIM: int = len(BOOST_STATS)  # 7


def boosts_array(mon) -> np.ndarray:
    """Return a (7,) int8 array of stat stages in BOOST_STATS order.

    Returns all zeros if mon is None or has no boosts.  Stat stages are clamped
    to [-6, +6] by the game engine; we trust poke-env to reflect that.
    """
    if mon is None:
        return np.zeros(BOOST_DIM, dtype=np.int8)
    boosts = getattr(mon, "boosts", {})
    return np.array([boosts.get(s, 0) for s in BOOST_STATS], dtype=np.int8)


def boosts_str(mon) -> str | None:
    """Non-zero stat stages as a compact string, e.g. 'atk:+2 spa:+1', or None."""
    if mon is None:
        return None
    boosts = getattr(mon, "boosts", {})
    parts = [f"{s}:{v:+d}" for s, v in boosts.items() if v != 0]
    return " ".join(parts) if parts else None
