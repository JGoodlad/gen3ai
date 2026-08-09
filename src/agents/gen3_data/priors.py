"""Gen 3 Smogon-usage priors — a *concept module* (the ``gen3_data.moves`` pattern).

Per-species probability distributions derived from aggregated Smogon usage stats by
``tools/smogon_stats_downloader/compute_priors.py``: ability priors (which ability a species is
likely running) and Hidden Power priors (which HP type). Owned by the project under ``data/``;
reached via the facade as ``gen3_data.priors``. Unlike the deterministic reference dexes these are
*probabilistic* — but the consumer doesn't care where they came from, only asks by species.
"""
from __future__ import annotations

import functools
from typing import Dict, List, Tuple

from . import _base

ability_raw = _base.singleton(lambda: _base.load_json("gen3_ability_priors.json"))
hidden_power_raw = _base.singleton(lambda: _base.load_json("gen3_hidden_power_priors.json"))
move_raw = _base.singleton(lambda: _base.load_json("gen3_move_priors.json"))
item_raw = _base.singleton(lambda: _base.load_json("gen3_item_priors.json"))
spread_raw = _base.singleton(lambda: _base.load_json("gen3_spread_priors.json"))
smogon_stats_raw = _base.singleton(lambda: _base.load_json("gen3_smogon_stats.json"))

_EV_INDEX = {"hp": 0, "atk": 1, "def": 2, "spa": 3, "spd": 4, "spe": 5}


def ability(species: str) -> Dict[str, float]:
    """``{ability_id: probability}`` for ``species`` (empty dict if the species has no entry)."""
    return ability_raw().get(species, {})


def hidden_power(species: str) -> Dict[str, float]:
    """``{hp_type: probability}`` for ``species`` (empty dict if the species has no entry)."""
    return hidden_power_raw().get(species, {})


def moves(species: str) -> Dict[str, float]:
    """``{move_id: P(move in set)}`` for ``species`` (NOT sum-1; a set runs ~4 moves).

    The slot-accounting quantity: revealed moves are certain, this prior covers the rest."""
    return move_raw().get(species, {})


def items(species: str) -> Dict[str, float]:
    """``{item_id: P(item)}`` for ``species`` (sum→1 over observed items; includes ``choiceband``,
    the never-revealed item — reserved for a future worst-case item-inference channel, not yet
    consumed by the incoming-damage belief)."""
    return item_raw().get(species, {})


def spreads(species: str) -> List[list]:
    """``[[nature, [hp,atk,def,spa,spd,spe], weight], ...]`` raw usage spreads (weights sum→1)."""
    return spread_raw().get(species, [])


@functools.lru_cache(maxsize=1)
def species_usage() -> Dict[str, float]:
    """``{species_id: raw usage weight}`` — how often each species appears in gen3ou, from the
    aggregated Smogon stats (``gen3_smogon_stats.json`` → per-species ``Raw count``, the same
    weighted-sets denominator ``compute_priors._raw_count`` uses). Keys are normalized Showdown
    ids (lowercase alnum, mirroring the acquisition layer's ``_to_id``). Weights are NOT
    normalized — the consumer picks its own floor/normalization. A species absent from the
    stats simply has no entry."""
    out: Dict[str, float] = {}
    for name, sp_data in smogon_stats_raw().get("data", {}).items():
        sid = "".join(c for c in name.lower() if c.isalnum())
        w = float(sp_data.get("Raw count") or sp_data.get("usage") or 0.0)
        if sid and w > 0.0:
            out[sid] = out.get(sid, 0.0) + w
    return out


def gen3_stat(base: int, ev: int, mult: float) -> int:
    """Gen-3 non-HP stat at level 100, IV 31, EV ``ev``, nature multiplier ``mult`` —
    exact integer math (``×11//10`` / ``×9//10`` / ``×1``), not float. The single source of truth
    for the L100/IV31 stat formula (the incoming-damage encoder uses it for its no-prior fallback)."""
    pre = 2 * base + 31 + ev // 4 + 5
    if mult > 1.0:
        return pre * 11 // 10
    if mult < 1.0:
        return pre * 9 // 10
    return pre


@functools.lru_cache(maxsize=None)
def stat_distribution(species: str, stat: str) -> Tuple[Tuple[int, float], ...]:
    """Usage-weighted distribution over a species' realized **stat value** (L100, IV31, nature
    applied), derived from the Smogon spread priors. ``stat`` ∈ {``atk``, ``spa``, ``spe``}.

    Returns a tuple of ``(stat_value, weight)`` sorted by value (weights sum→1), or ``()`` if the
    species has no spread data. This is the uncertainty source for the incoming-damage **magnitude**
    (atk/spa) and the **P(outspeed)** belief (spe) — every opponent set is a different point here."""
    from . import natures as _nat, species as _sp
    if stat not in ("atk", "spa", "spe"):
        return ()
    sp = _sp.get(species)
    spr = spreads(species)
    if sp is None or not spr:
        return ()
    base = sp.base_stats.get(stat)
    if base is None:
        return ()
    evi = _EV_INDEX[stat]
    agg: Dict[int, float] = {}
    for nature, evs, w in spr:
        nd = _nat.get(str(nature).lower())
        mult = nd.multipliers.get(stat, 1.0) if nd is not None else 1.0
        val = gen3_stat(base, evs[evi], mult)
        agg[val] = agg.get(val, 0.0) + float(w)
    return tuple(sorted(agg.items()))
