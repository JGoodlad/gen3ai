"""This module defines the MoveCategory class, which represents a move category.

**P1 of the poke-env retirement (T27): the definition moved OUT of the fork.** The class is OWNED by
``agents.enums`` (standard library only, so our data facade and the trainer load no ``poke_env`` module);
this module re-exports the same object, so ``poke_env.battle.move_category.MoveCategory is agents.enums.MoveCategory`` and every
poke-env comparison against it is unchanged. The fork is retired in P6; this file goes with it.
"""

from agents.enums import MoveCategory

__all__ = ["MoveCategory"]
