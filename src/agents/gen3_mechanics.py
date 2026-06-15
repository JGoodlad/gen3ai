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


def _ability_revealed(mon) -> bool:
    """Is ``mon``'s ability confirmed? The SAME predicate the per-Pokémon ability block uses
    for its ``known`` flag (``AbilitiesEncoder``) and ``_resolve_ability_distribution``: a set
    ability whose normalized id isn't poke-env's ``"unknownability"`` sentinel. Keeping one
    predicate is what makes the reactive ``status_will_land_known`` bit consistent with the
    ability block's ``known`` bit for the same opponent."""
    ability = getattr(mon, "ability", None)
    if not ability:
        return False
    return ability.lower().replace(" ", "").replace("_", "") != "unknownability"


def status_land_estimate(
    move_id: str, status_id: str | None, mon, ability_dist
) -> tuple[float, bool]:
    """``(probability, known)`` that a dedicated status move applies ``status_id`` to ``mon``.

    ``probability`` ∈ [0,1] is the "priors first, confirmation collapses it" estimate — the same
    spirit as the matchup encoder's ability-expectation (``_expected_multiplier``).
    ``ability_dist`` is ``[(ability_id_or_None, prob), …]``: a singleton for a REVEALED ability,
    the Smogon prior for an UNREVEALED opponent, or ``[(None, 1.0)]`` when we have no info.
    Ability-INDEPENDENT certain blocks → 0.0 regardless of the distribution: type immunity
    (``STATUS_MOVE_IMMUNITY``), already carrying a status (both via :func:`is_status_move_immune`),
    and an active Substitute. Otherwise the result is ``1 − P(ability blocks it)`` — e.g. an
    unrevealed Snorlax (Immunity 0.86 / Thick Fat 0.14) reads ≈0.14 for Toxic.

    ``known`` is the prior-vs-confirmed flag, ROUTED CONSISTENTLY WITH THE ABILITY BLOCK: True
    when the value rests on confirmed information — a type-certain hard block (immune / already
    statused / Substitute, all always visible) OR the opponent's ability is revealed
    (:func:`_ability_revealed`, the same predicate the ability ``known`` bit uses). False when
    the value is a Smogon-prior estimate that a future ability reveal could move. So a fractional
    probability always has ``known=False``; the 0.0/1.0 endpoints are disambiguated by this bit
    exactly the way the ability block disambiguates a confirmed ability from a prior."""
    if status_id is None or mon is None:
        return 0.0, False
    if is_status_move_immune(move_id, mon):  # type immunity OR already statused — certain
        return 0.0, True
    effects = getattr(mon, "effects", None) or {}
    if Effect.SUBSTITUTE in effects:  # Sub blocks status — certain
        return 0.0, True
    block_mass = 0.0
    for ability, p in (ability_dist or [(None, 1.0)]):
        if ability is None:
            continue
        aid = ability.lower().replace(" ", "").replace("_", "")
        if status_id in ABILITY_STATUS_IMMUNITY.get(aid, frozenset()):
            block_mass += p
    return max(0.0, 1.0 - block_mass), _ability_revealed(mon)


def status_land_probability(move_id: str, status_id: str | None, mon, ability_dist) -> float:
    """The probability half of :func:`status_land_estimate` (see it for the full contract)."""
    return status_land_estimate(move_id, status_id, mon, ability_dist)[0]


def status_move_lands(move_id: str, status_id: str | None, mon) -> bool:
    """Server-truth bool: does this status land given ``mon``'s CURRENT (revealed) ability?
    Equivalent to :func:`status_land_probability` with a singleton distribution drawn from
    the live ``mon.ability`` (no priors) — the "confirmation" half of the priors-then-confirm
    pattern. ``None`` status (a non-status move) → False; an unrevealed/unknown ability
    contributes nothing (treated as non-immune)."""
    ability = getattr(mon, "ability", None)
    dist = [(ability, 1.0)] if ability else [(None, 1.0)]
    return status_land_probability(move_id, status_id, mon, dist) > 0.0


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

# Gen3 Protect/Detect/Endure consecutive-use success probability (gen3_protect_odds_v1).
# Showdown's gen3 format inherits the stall condition through gen4 → gen5 (NOT the base
# `data/conditions.ts` *3 rule): gen5 starts the counter at 2 and DOUBLES it each consecutive
# successful stall move, and gen4 caps it (`counterMax: 8`, "the chance does not fall below
# 1/8"). So the per-attempt success odds are 100% / 50% / 25% / 12.5% (then a 12.5% floor) —
# NOT the cartridge's unbounded halving and NOT the base *3. poke-env tracks the live counter
# as `Pokemon.protect_counter` (consecutive successful stall moves; reset to 0 on a switch,
# faint, non-stall move, or a failed roll), so k=0 is the fresh 100% case. Verified against the
# compiled sim + a 14.8k-attempt bridge measurement (protect_success_prob_fuzz_test.py).
_PROTECT_COUNTER_MAX = 8  # gen4 counterMax → the 1/8 (12.5%) floor


def protect_success_probability(protect_counter: int) -> float:
    """P(the next Protect/Detect/Endure succeeds) given a mon's current consecutive-stall
    counter ``k`` (poke-env ``protect_counter``). ``k<=0`` → 1.0 (fresh, post-reset); else
    ``1 / min(2**k, 8)`` — floored doubling (1.0 / 0.5 / 0.25 / 0.125 / 0.125 / …)."""
    k = int(protect_counter)
    if k <= 0:
        return 1.0
    return 1.0 / min(2 ** k, _PROTECT_COUNTER_MAX)

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
