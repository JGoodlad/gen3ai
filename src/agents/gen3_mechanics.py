"""
Gen 3 OU mechanics — single source of truth.

All type immunities, status conditions, volatile effects, move categories, stat
boost helpers, and related functions used across reward signals, heuristic
opponents, observation encoders, and loggers live here.  Import from this module
instead of re-declaring these facts in individual files.
"""
from __future__ import annotations

import functools


from agents.enums import PokemonType, Status
from agents.gen3_data import type_chart as _type_chart_data

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

# Gen-3 type-effectiveness chart, owned by the project and reached through the gen3_data facade
# (loaded once from data/pokemon/gen3_type_chart.json; derived from poke-env by
# tools/pokemon_data_extractor). The dense _CHART below is built from it. Byte-identical to the
# old GenData.from_gen(3).type_chart, so effectiveness is unchanged (pinned by gen3_mechanics_test).
_type_chart = _type_chart_data.chart()

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

# Gen-3 abilities that grant FULL immunity to a specific major status. Keyed by the
# Showdown ability id → the status ids it blocks. Only abilities that *prevent the
# status from applying* are listed — abilities that merely cure faster (Shed Skin) or
# wake early (Early Bird) do NOT block and are excluded. Immunity (Snorlax) is the
# OU-relevant one (blocks Toxic/poison); the rest are correctness for completeness.
ABILITY_STATUS_IMMUNITY: dict[str, frozenset[str]] = {
    "immunity":    frozenset({"psn", "tox"}),
    "limber":      frozenset({"par"}),
    "waterveil":   frozenset({"brn"}),
    "insomnia":    frozenset({"slp"}),
    "vitalspirit": frozenset({"slp"}),
    "magmaarmor":  frozenset({"frz"}),
}


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

# Gen3 Curse, the NON-GHOST branch: +1 Atk / +1 Def / −1 Spe (CurseLax / Curse-Registeel — a
# gen3ou-defining setup move). Type-CONDITIONAL (a Ghost user's Curse is a different move
# entirely: 50% max HP for a target curse), so it cannot live in a static per-move table — the
# per-move `selfBoosts` data field deliberately excludes it (its pure-setup gates are the rust
# engine's draw-free contract) and consumers resolve the branch from the USER'S live types at
# runtime (the obs encoder's `is_boost` convention; the C1 consequence-edge kernel's
# `CURSE_BOOSTS` buffer sources these stages). One source: this constant.
CURSE_NON_GHOST_BOOSTS: dict[str, int] = {"atk": 1, "def": 1, "spe": -1}


def boosts_str(mon) -> str | None:
    """Non-zero stat stages as a compact string, e.g. 'atk:+2 spa:+1', or None."""
    if mon is None:
        return None
    boosts = getattr(mon, "boosts", {})
    parts = [f"{s}:{v:+d}" for s, v in boosts.items() if v != 0]
    return " ".join(parts) if parts else None
