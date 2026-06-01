"""Gen 3 ability reference data — a *concept module* (the ``gen3_data.moves`` pattern).

Static facts about an ability keyed by its id: ``num`` and display ``name``. Reference data,
owned by the project (derived from the pokedex + Showdown abilities source by
``tools/pokemon_data_extractor``). Reached via the facade as ``gen3_data.abilities``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from . import _base


@dataclass(frozen=True)
class AbilityData:
    """Immutable reference record for one gen3 ability."""

    id: str
    num: int
    name: str


def _build(raw: Dict[str, dict]) -> Dict[str, AbilityData]:
    return {aid: AbilityData(id=aid, num=int(v.get("num", 0)), name=v.get("name", aid))
            for aid, v in raw.items()}


raw = _base.singleton(lambda: _base.load_json("gen3_abilities.json"))
_dex = _base.singleton(lambda: _build(raw()))


def get(ability_id: Optional[str]) -> Optional[AbilityData]:
    """Reference record for ``ability_id``, or ``None`` if it isn't a known gen3 ability."""
    if ability_id is None:
        return None
    return _dex().get(ability_id)


def ability_data(ability_id: str) -> AbilityData:
    """Reference record for an ability that MUST exist (crash-don't-drop). Raises ``KeyError``."""
    ab = get(ability_id)
    if ab is None:
        raise KeyError(f"Unknown gen3 ability id: {ability_id!r}")
    return ab
