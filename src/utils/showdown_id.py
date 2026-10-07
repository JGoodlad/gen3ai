"""The Showdown id normaliser — ONE definition, standard library only (P1 of the poke-env retirement, ``T27``).

``to_id_str("King's Rock") == "kingsrock"``: the lower-cased alphanumerics of a name, which is exactly how
Showdown derives an id (``toID``). The battle layer, the observation encoders, the BC log reader and the data
facade's callers all key their tables by it. It used to be imported from ``poke_env.data.normalize``, which put
the whole poke-env package on the import path of every one of them; the vendored fork's
``poke_env/data/normalize.py`` now re-exports THIS function, so the two can never disagree.
"""
from __future__ import annotations

from functools import lru_cache

__all__ = ["to_id_str"]


@lru_cache(2**13)
def to_id_str(name: str) -> str:
    """Converts a full-name to its corresponding id string.

    :param name: The name to convert.
    :type name: str
    :return: The corresponding id string.
    :rtype: str
    """
    return "".join(char for char in name if char.isalnum()).lower()
