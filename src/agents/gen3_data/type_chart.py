"""Gen 3 type-effectiveness chart — a *concept module* (the ``gen3_data.moves`` pattern).

The single-type effectiveness multipliers, owned by the project (derived from poke-env's
``GenData`` by ``tools/pokemon_data_extractor`` and committed under ``data/``). Keyed
``{DEFENDING_TYPE: {ATTACKING_TYPE: multiplier}}`` with poke-env enum-name keys (e.g. ``"FIRE"``),
so it drops straight into ``gen3_mechanics._CHART`` with no conversion. Reached via the facade as
``gen3_data.type_chart``.
"""
from __future__ import annotations

from typing import Dict

from . import _base

raw = _base.singleton(lambda: _base.load_json("gen3_type_chart.json"))


def chart() -> Dict[str, Dict[str, float]]:
    """The full ``{DEFENDING_TYPE: {ATTACKING_TYPE: multiplier}}`` table."""
    return raw()


def multiplier(defending_type_name: str, attacking_type_name: str) -> float:
    """Effectiveness of ``attacking_type_name`` against ``defending_type_name`` (enum-name keys)."""
    return raw()[defending_type_name][attacking_type_name]
