"""
Gen 3 OU mechanics — single source of truth.

All type immunities, status conditions, volatile effects, move categories, stat
boost helpers, and related functions used across reward signals, heuristic
opponents, observation encoders, and loggers live here.  Import from this module
instead of re-declaring these facts in individual files.
"""
from __future__ import annotations

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
# Gen 3-only — Heatproof / Filter / Solid Rock are Gen 4+, Wonder Guard is Shedinja-
# only (not OU), and Lightning Rod doesn't grant immunity in Gen 3 singles.
ABILITY_TYPE_MULTIPLIER: dict[str, dict[PokemonType, float]] = {
    "levitate":    {PokemonType.GROUND:   0.0},
    "voltabsorb":  {PokemonType.ELECTRIC: 0.0},
    "waterabsorb": {PokemonType.WATER:    0.0},
    "flashfire":   {PokemonType.FIRE:     0.0},
    "thickfat":    {PokemonType.ICE:      0.5, PokemonType.FIRE: 0.5},
}

_type_chart = GenData.from_gen(3).type_chart


def effective_multiplier(move_type: PokemonType, mon) -> float:
    """Damage-type multiplier of move_type vs mon, including Gen 3 ability modifiers.

    Returns the final multiplier Showdown reports through |-immune|/|-resisted|/
    |-supereffective| messages. Multiplies the raw type-chart value by the mon's
    ability modifier from ABILITY_TYPE_MULTIPLIER (default 1.0 = no effect).

    Gen 3 quirk: Flash Fire does NOT activate when the target is frozen — the
    incoming Fire move falls through and is resisted normally (0.5× from Fire-type).
    See pokemon-showdown/data/mods/gen3/abilities.ts.
    """
    ability = (getattr(mon, "ability", None) or "").lower()
    base = move_type.damage_multiplier(mon.type_1, mon.type_2, type_chart=_type_chart)
    if ability == "flashfire" and getattr(mon, "status", None) == Status.FRZ:
        return base
    return base * ABILITY_TYPE_MULTIPLIER.get(ability, {}).get(move_type, 1.0)


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
