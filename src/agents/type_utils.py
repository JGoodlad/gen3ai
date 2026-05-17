"""
Shared type-effectiveness utilities.

Raw PokemonType.damage_multiplier() only uses the type chart and is unaware of
ability-based immunities (Levitate, Volt Absorb, etc.). Use effective_multiplier()
everywhere type effectiveness is computed so that ability immunities are respected.
"""
from poke_env.battle.pokemon_type import PokemonType
from poke_env.data import GenData

# Abilities that grant a full type immunity in Gen 3.
ABILITY_TYPE_IMMUNITY: dict[str, PokemonType] = {
    "levitate":    PokemonType.GROUND,
    "voltabsorb":  PokemonType.ELECTRIC,
    "waterabsorb": PokemonType.WATER,
    "flashfire":   PokemonType.FIRE,
}

_type_chart = GenData.from_gen(3).type_chart


def effective_multiplier(move_type: PokemonType, mon) -> float:
    """Type effectiveness of move_type vs mon, accounting for Gen 3 ability immunities.

    Returns 0.0 when the mon's ability (Levitate, Volt Absorb, Water Absorb, Flash Fire)
    nullifies the move type entirely; otherwise delegates to the type chart.
    """
    ability = (getattr(mon, "ability", None) or "").lower()
    if ABILITY_TYPE_IMMUNITY.get(ability) == move_type:
        return 0.0
    return move_type.damage_multiplier(mon.type_1, mon.type_2, type_chart=_type_chart)
