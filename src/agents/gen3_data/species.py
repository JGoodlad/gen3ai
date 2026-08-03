"""Gen 3 species reference data — a *concept module* (the ``gen3_data.moves`` pattern).

Static facts about a species keyed by its id: national-dex ``num`` and base stats. Reference
data, owned by the project (derived from the poke-env pokedex by ``tools/pokemon_data_extractor``
and committed under ``data/``). Reached via the facade as ``gen3_data.species``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

from . import _base

_STAT_KEYS = ("atk", "def", "hp", "spa", "spd", "spe")


@dataclass(frozen=True)
class SpeciesData:
    """Immutable reference record for one gen3 species."""

    id: str
    num: int
    name: str
    base_stats: Dict[str, int]   # keyed by atk/def/hp/spa/spd/spe
    types: Tuple[str, ...] = ()  # STAB/defensive types, UPPERCASED onto the TypeEncoder axis (1-2 entries)
    # gen3_species_formes_v1: the BASE species id for an ALTERNATE forme (Deoxys-Speed ->
    # "deoxys", Unown-B -> "unown"), else ``None`` for a base form. A forme SHARES its
    # base's national-dex ``num`` — and ``num`` is what the obs species channel and every
    # num-indexed GPU table are keyed by — so a forme is observationally its base. Any
    # ``table[sd.num] = …`` consumer MUST iterate :func:`base_form_ids` instead of
    # :func:`raw`, or the last forme row written silently shadows the base's stats/types.
    base_species: Optional[str] = None
    # An in-battle-only forme (Castform-Sunny/Rainy/Snowy under Forecast): describable, but
    # never selectable in a team.
    battle_only: bool = False


def _build(raw: Dict[str, dict]) -> Dict[str, SpeciesData]:
    dex: Dict[str, SpeciesData] = {}
    for sid, v in raw.items():
        bs = v.get("baseStats", {})
        dex[sid] = SpeciesData(
            id=sid,
            num=int(v.get("num", 0)),
            name=v.get("name", sid),
            base_stats={k: int(bs[k]) for k in _STAT_KEYS if k in bs},
            types=tuple(str(t).upper() for t in v.get("types", ())),
            base_species=v.get("baseSpecies"),
            battle_only=bool(v.get("battleOnly")),
        )
    return dex


raw = _base.singleton(lambda: _base.load_json("gen3_species.json"))
_dex = _base.singleton(lambda: _build(raw()))
_base_forms = _base.singleton(
    lambda: tuple(sid for sid, sd in _dex().items() if sd.base_species is None)
)


def get(species_id: Optional[str]) -> Optional[SpeciesData]:
    """Reference record for ``species_id``, or ``None`` if it isn't a known gen3 species."""
    if species_id is None:
        return None
    return _dex().get(species_id)


def base_form_ids() -> Tuple[str, ...]:
    """The BASE-form species ids only — one entry per national-dex ``num``.

    The iteration order every ``table[species.num] = …`` builder must use
    (`gen3_species_formes_v1`). ``raw()`` also contains alternate/cosmetic FORMES
    (Deoxys-Attack, the Unown letters, Castform's weather formes), which share their base's
    ``num``; iterating ``raw()`` into a num-indexed table is last-write-wins, so a forme
    would silently overwrite the base's stats/types at that num. Formes are still in
    ``raw()``/``get()`` — a battle genuinely fields them and the port must compute their
    stats — they just must not claim the shared num in a lookup table."""
    return _base_forms()


def species_data(species_id: str) -> SpeciesData:
    """Reference record for a species that MUST exist (crash-don't-drop). Raises ``KeyError``."""
    sd = get(species_id)
    if sd is None:
        raise KeyError(f"Unknown gen3 species id: {species_id!r}")
    return sd
