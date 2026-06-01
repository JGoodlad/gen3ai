"""Gen 3 item reference data — a *concept module* (the ``gen3_data.moves`` pattern).

Static facts about an item keyed by its id: item-dex ``num`` and display ``name``. Reference
data, owned by the project (derived from the Showdown items source by
``tools/pokemon_data_extractor``). Reached via the facade as ``gen3_data.items``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from . import _base


@dataclass(frozen=True)
class ItemData:
    """Immutable reference record for one gen3 item."""

    id: str
    num: int
    name: str


def _build(raw: Dict[str, dict]) -> Dict[str, ItemData]:
    return {iid: ItemData(id=iid, num=int(v.get("num", 0)), name=v.get("name", iid))
            for iid, v in raw.items()}


raw = _base.singleton(lambda: _base.load_json("gen3_items.json"))
_dex = _base.singleton(lambda: _build(raw()))


def get(item_id: Optional[str]) -> Optional[ItemData]:
    """Reference record for ``item_id``, or ``None`` if it isn't a known gen3 item."""
    if item_id is None:
        return None
    return _dex().get(item_id)


def item_data(item_id: str) -> ItemData:
    """Reference record for an item that MUST exist (crash-don't-drop). Raises ``KeyError``."""
    it = get(item_id)
    if it is None:
        raise KeyError(f"Unknown gen3 item id: {item_id!r}")
    return it
