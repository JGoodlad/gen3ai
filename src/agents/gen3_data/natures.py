"""Gen 3 nature reference data — a *concept module* (the ``gen3_data.moves`` pattern).

A nature's five stat multipliers (atk/def/spa/spd/spe, each 0.9 / 1.0 / 1.1), keyed by nature
name. Natures are gen-independent; the file is owned by the project (copied out of poke-env's
static data by ``tools/pokemon_data_extractor``). Reached via the facade as ``gen3_data.natures``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

from . import _base

# The five battle-relevant stat multipliers (HP is never nature-modified).
_MULT_KEYS = ("atk", "def", "spa", "spd", "spe")


@dataclass(frozen=True)
class NatureData:
    """Immutable reference record for one nature."""

    name: str
    multipliers: Dict[str, float]   # keyed by atk/def/spa/spd/spe


def _build(raw: Dict[str, dict]) -> Dict[str, NatureData]:
    return {name: NatureData(name=name,
                             multipliers={k: float(v[k]) for k in _MULT_KEYS if k in v})
            for name, v in raw.items()}


raw = _base.singleton(lambda: _base.load_json("gen3_natures.json"))
_dex = _base.singleton(lambda: _build(raw()))


def get(nature_name: Optional[str]) -> Optional[NatureData]:
    """Reference record for ``nature_name``, or ``None`` if it isn't a known nature."""
    if nature_name is None:
        return None
    return _dex().get(nature_name)


def multipliers() -> Dict[str, Dict[str, float]]:
    """``{nature_name: {atk/def/spa/spd/spe: multiplier}}`` — the form the spread encoder uses."""
    return {name: dict(nd.multipliers) for name, nd in _dex().items()}
