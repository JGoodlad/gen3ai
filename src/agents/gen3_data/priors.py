"""Gen 3 Smogon-usage priors — a *concept module* (the ``gen3_data.moves`` pattern).

Per-species probability distributions derived from aggregated Smogon usage stats by
``tools/smogon_stats_downloader/compute_priors.py``: ability priors (which ability a species is
likely running) and Hidden Power priors (which HP type). Owned by the project under ``data/``;
reached via the facade as ``gen3_data.priors``. Unlike the deterministic reference dexes these are
*probabilistic* — but the consumer doesn't care where they came from, only asks by species.
"""
from __future__ import annotations

from typing import Dict

from . import _base

ability_raw = _base.singleton(lambda: _base.load_json("gen3_ability_priors.json"))
hidden_power_raw = _base.singleton(lambda: _base.load_json("gen3_hidden_power_priors.json"))


def ability(species: str) -> Dict[str, float]:
    """``{ability_id: probability}`` for ``species`` (empty dict if the species has no entry)."""
    return ability_raw().get(species, {})


def hidden_power(species: str) -> Dict[str, float]:
    """``{hp_type: probability}`` for ``species`` (empty dict if the species has no entry)."""
    return hidden_power_raw().get(species, {})
