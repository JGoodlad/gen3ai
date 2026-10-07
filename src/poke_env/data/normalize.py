"""``to_id_str`` — re-exported from ``utils.showdown_id`` (P1 of the poke-env retirement, ``T27``).

The definition moved OUT of the fork so our code can normalise a Showdown id without importing poke-env; this
module hands back the same function object, so the fork and our code can never disagree. Retired with the fork (P6).
"""
from utils.showdown_id import to_id_str

__all__ = ["to_id_str"]
