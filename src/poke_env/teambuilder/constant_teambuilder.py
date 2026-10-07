"""``ConstantTeambuilder`` — see ``utils.team_packing``.

**P1 of the poke-env retirement (T27): the definition moved OUT of the fork** into ``utils.team_packing``
(poke-env-free); this module re-exports the same objects, so the fork's ``Player`` and every poke-env consumer see the
identical classes. Retired with the fork (P6).
"""

from utils.team_packing import ConstantTeambuilder

__all__ = ["ConstantTeambuilder"]
