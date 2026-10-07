"""This module defines the PokemonType class, which represents a Pokemon type.
PokemonTypes are mainly associated with Pokemons and moves.

**P1 of the poke-env retirement (T27): the definition moved OUT of the fork.** The class is OWNED by
``agents.enums`` (standard library only, so our data facade and the trainer load no ``poke_env`` module);
this module re-exports the same object, so ``poke_env.battle.pokemon_type.PokemonType is agents.enums.PokemonType`` and every
poke-env comparison against it is unchanged. The fork is retired in P6; this file goes with it.
"""

from agents.enums import PokemonType

__all__ = ["PokemonType"]
