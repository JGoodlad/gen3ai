"""Gen 3 Smogon-usage priors — a *concept module* (the ``gen3_data.moves`` pattern).

Per-species probability distributions derived from aggregated Smogon usage stats by
``tools/smogon_stats_downloader/compute_priors.py``: ability priors (which ability a species is
likely running) and Hidden Power priors (which HP type). Owned by the project under ``data/``;
reached via the facade as ``gen3_data.priors``. Unlike the deterministic reference dexes these are
*probabilistic* — but the consumer doesn't care where they came from, only asks by species.
"""
from __future__ import annotations

import functools
from typing import Any, Dict, List, Tuple

from . import _base
from . import format_spec as _fs

#: A set has exactly this many move SLOTS; the chaos ``Moves`` field counts every slot (an empty one
#: under the key ``""``), so ``Σ_m P(m in set) + empty-slot mass == MOVE_SLOTS`` EXACTLY.
MOVE_SLOTS = 4
#: Tolerance of the load guards below. Every invariant is exact in real arithmetic; what is left is
#: float summation order plus the JSON round trip (measured < 1e-13). The defect the move guard exists
#: for (F-X5-41: ``Moves`` over the UNWEIGHTED ``Raw count``) is off by 6-90 % on EVERY species.
_LOAD_RTOL = 1e-6


class PriorInvariantError(ValueError):
    """A committed Smogon prior broke an invariant that holds exactly when it is derived correctly
    (``tools/smogon_stats_downloader/compute_priors.py``, ``gen3_smogon_prior_denominator_v1``) — or a
    species-usage marginal not at the weighted population (``gen3_smogon_species_usage_weighted_v1``)."""


def _checked_distributions(filename: str, table: Dict[str, Any], values: Any) -> Dict[str, Any]:
    """THROWS unless every species' row is a probability distribution (entries in ``(0, 1]``, sum 1)."""
    for sp, row in table.items():
        v = [float(x) for x in values(row)]
        tot = sum(v)
        if not v or any(not (0.0 < x <= 1.0 + _LOAD_RTOL) for x in v) or abs(tot - 1.0) > _LOAD_RTOL:
            raise PriorInvariantError(f"{filename}[{sp}]: not a distribution (sum {tot!r}) — regenerate "
                                      f"it with tools/smogon_stats_downloader/compute_priors.py")
    return table


def _checked_format_legal(filename: str, kind: str, table: Dict[str, Any]) -> Dict[str, Any]:
    """THROWS unless no entry of ``table`` gives an entity the gen3ou FORMAT SPEC bans any mass
    (``gen3_format_spec_priors_v1``; ``designs/endstate/design_format_spec.md`` §5.1): a banned ability / item /
    move (or a move banned for that species) / teammate species — or a teammate row keyed BY a banned species.
    The acquisition tool (``compute_priors.py``) removes that mass and renormalises; this guard makes a file
    regenerated without the spec unloadable, so banned mass can never reach a model."""
    spec = _fs.active()
    for sp, row in table.items():
        if kind == "move":
            bad = [m for m in row if spec.is_banned("move", m, species=sp)]
        elif kind == "teammate":
            bad = [t for t in [sp, *row] if t in spec.banned_species]
        else:
            banned = spec.banned_abilities if kind == "ability" else spec.banned_items
            bad = [e for e in row if e in banned]
        if bad:
            raise PriorInvariantError(
                f"{filename}[{sp}]: {kind} {bad} is BANNED in {spec.format_id} (agents.gen3_data.format_spec) "
                f"and must carry no prior — regenerate it with tools/smogon_stats_downloader/compute_priors.py")
    return table


def _weighted_count(rec: Dict[str, Any]) -> float:
    """The species' RATING-WEIGHTED set total W (Smogon's ``p.raw.weight``) = ``Σ Abilities``.
    The ONE facade-side reading of W: the move prior's denominator (``_checked_moves``) and the
    species-usage marginal (``species_usage``) both go through it."""
    return float(sum((rec.get("Abilities") or {}).values()))


def _species_id(name: str) -> str:
    """A chaos species name as a normalized Showdown id (lowercase alnum, the acquisition layer's
    ``_to_id``)."""
    return "".join(c for c in name.lower() if c.isalnum())


def _checked_species_usage(table: Dict[str, float]) -> Dict[str, float]:
    """THROWS unless ``table`` is the species-usage marginal at the RATING-WEIGHTED population: one
    entry per chaos species, each equal to that species' W read from TWO independent weighted fields
    of its record — ``Σ Abilities`` and ``Σ Moves / MOVE_SLOTS`` (equal on a correct record, to
    1.2e-14 on the committed window). A table of the UNWEIGHTED ``Raw count`` (F-X5-47; W / Raw count
    runs 0.10-0.94 by species, so every species is >= 6 % off) — or of the latest-month ``usage``
    share — fails it on every species, so an unweighted marginal can never reach a model."""
    stats: Dict[str, Dict[str, Any]] = {}
    for name, rec in smogon_stats_raw().get("data", {}).items():
        sid = _species_id(name)
        if sid in stats:
            raise PriorInvariantError(f"gen3_smogon_stats.json: two chaos records normalize to {sid!r}")
        stats[sid] = rec
    banned = _fs.active().banned_species
    if banned & set(table):
        raise PriorInvariantError(f"species usage: banned species {sorted(banned & set(table))} carry usage — "
                                  f"the format spec gives them prior 0 (design_format_spec.md §5.1)")
    if set(table) != {s for s, r in stats.items() if _weighted_count(r) > 0.0 and s not in banned}:
        raise PriorInvariantError(
            f"species usage: the species set differs from the chaos records' "
            f"({len(table)} vs {len(stats)}) — regenerate it from gen3_smogon_stats.json (F-X5-47)")
    for sid, v in table.items():
        rec = stats[sid]
        w = _weighted_count(rec)
        w_moves = float(sum((rec.get("Moves") or {}).values())) / MOVE_SLOTS
        if not (v > 0.0) or abs(v - w) > _LOAD_RTOL * w or abs(v - w_moves) > _LOAD_RTOL * w:
            raise PriorInvariantError(
                f"species usage[{sid}] = {v!r}, but the species' RATING-WEIGHTED set total is "
                f"W = sum(Abilities) = {w!r} (sum(Moves) / {MOVE_SLOTS} = {w_moves!r}) — the marginal "
                f"is not the weighted population every other Smogon prior uses (an UNWEIGHTED Raw "
                f"count table, F-X5-47)")
    return table


def _checked_moves(table: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """THROWS unless every species' move prior is ``Moves[m] / W`` — checked through its exact
    consequence ``Σ_m P(m in set) + Moves[""] / W == MOVE_SLOTS`` against the committed chaos stats,
    each ``P`` in ``(0, 1]``. A table divided by the UNWEIGHTED ``Raw count`` (F-X5-41: Skarmory Spikes
    0.547, sums 0.41-3.75) fails it on every species, so a deflated prior can never reach a model."""
    stats = {k.lower(): v for k, v in smogon_stats_raw().get("data", {}).items()}
    for sp, row in table.items():
        rec = stats.get(sp)
        if rec is None:
            raise PriorInvariantError(f"gen3_move_priors.json[{sp}]: no chaos record in "
                                      f"gen3_smogon_stats.json to check it against")
        w = _weighted_count(rec)
        vals = [float(x) for x in row.values()]
        if w <= 0.0 or not vals or any(not (0.0 < x <= 1.0) for x in vals):
            raise PriorInvariantError(f"gen3_move_priors.json[{sp}]: a P(move in set) outside (0, 1] "
                                      f"(or no weighted total, W = {w!r})")
        tot = sum(vals) + float((rec.get("Moves") or {}).get("", 0.0)) / w
        if abs(tot - MOVE_SLOTS) > _LOAD_RTOL * MOVE_SLOTS:
            raise PriorInvariantError(
                f"gen3_move_priors.json[{sp}]: sum P(move in set) + empty-slot mass = {tot!r}, not "
                f"{MOVE_SLOTS} — the prior is not the chaos Moves over the WEIGHTED total (a "
                f"deflated table over the unweighted Raw count, F-X5-41). Regenerate it with "
                f"tools/smogon_stats_downloader/compute_priors.py")
    return table


ability_raw = _base.singleton(lambda: _checked_format_legal("gen3_ability_priors.json", "ability", _checked_distributions(
    "gen3_ability_priors.json", _base.load_json("gen3_ability_priors.json"), dict.values)))
hidden_power_raw = _base.singleton(lambda: _checked_distributions(
    "gen3_hidden_power_priors.json", _base.load_json("gen3_hidden_power_priors.json"), dict.values))
move_raw = _base.singleton(lambda: _checked_format_legal(
    "gen3_move_priors.json", "move", _checked_moves(_base.load_json("gen3_move_priors.json"))))
item_raw = _base.singleton(lambda: _checked_format_legal("gen3_item_priors.json", "item", _checked_distributions(
    "gen3_item_priors.json", _base.load_json("gen3_item_priors.json"), dict.values)))
spread_raw = _base.singleton(lambda: _checked_distributions(
    "gen3_spread_priors.json", _base.load_json("gen3_spread_priors.json"),
    lambda rows: [r[2] for r in rows]))
teammate_raw = _base.singleton(lambda: _checked_format_legal("gen3_teammate_priors.json", "teammate",
                                                              _checked_distributions(
    "gen3_teammate_priors.json", _base.load_json("gen3_teammate_priors.json"), dict.values)))
smogon_stats_raw = _base.singleton(lambda: _base.load_json("gen3_smogon_stats.json"))

_EV_INDEX = {"hp": 0, "atk": 1, "def": 2, "spa": 3, "spd": 4, "spe": 5}


def ability(species: str) -> Dict[str, float]:
    """``{ability_id: probability}`` for ``species`` (empty dict if the species has no entry)."""
    return ability_raw().get(species, {})


def hidden_power(species: str) -> Dict[str, float]:
    """``{hp_type: probability}`` for ``species`` (empty dict if the species has no entry)."""
    return hidden_power_raw().get(species, {})


def moves(species: str) -> Dict[str, float]:
    """``{move_id: P(move in set)}`` for ``species`` (NOT sum-1; a set runs ~4 moves) — the chaos
    ``Moves`` over the species' RATING-WEIGHTED total (``gen3_smogon_prior_denominator_v1``; until
    2026-10-04 over the unweighted ``Raw count``, deflated — F-X5-41), checked at load
    (``_checked_moves``).

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


def teammates(species: str) -> Dict[str, float]:
    """``{teammate_species_id: P(teammate | species)}`` (sum→1) — the Smogon chaos ``Teammates``
    co-occurrence, the ONE species×species JOINT the usage stats publish. The Smogon-based
    coupling prior for the hidden-TEAM belief: one revealed Tyranitar should reshape the
    posterior over the five unrevealed slots. Empty dict if the species has no entry."""
    return teammate_raw().get(species, {})


@functools.lru_cache(maxsize=1)
def species_usage() -> Dict[str, float]:
    """``{species_id: W}`` — how often each species appears in gen3ou, as its RATING-WEIGHTED set
    total W (``_weighted_count`` = ``Σ Abilities``, Smogon's ``p.raw.weight``) from the aggregated
    Smogon stats (``gen3_smogon_stats.json``): the same population every other Smogon prior is
    weighted by — in particular the teammate conditional the co-occurrence lift divides by this
    share (``gen3_smogon_species_usage_weighted_v1``). Until 2026-10-04 it was the UNWEIGHTED
    ``Raw count`` (every rating weight 1, pkmn/stats ``p.raw.count++``; F-X5-47 — W / Raw count runs
    0.10-0.94 by species, so low-rated play inflated e.g. Shuckle's share ×5.2), checked at build
    (``_checked_species_usage`` THROWS on a Raw-count table). Keys are normalized Showdown ids
    (lowercase alnum, mirroring the acquisition layer's ``_to_id``). Weights are NOT normalized —
    the consumer picks its own floor/normalization. A species absent from the stats simply has no
    entry."""
    out: Dict[str, float] = {}
    banned = _fs.active().banned_species          # the format spec: a banned species has usage 0 (§5.1)
    for name, sp_data in smogon_stats_raw().get("data", {}).items():
        sid = _species_id(name)
        w = _weighted_count(sp_data)
        if sid and w > 0.0 and sid not in banned:
            out[sid] = w
    return _checked_species_usage(out)


def gen3_stat(base: int, ev: int, mult: float, iv: int = 31) -> int:
    """Gen-3 non-HP stat at level 100, IV ``iv`` (31 unless given), EV ``ev``, nature multiplier
    ``mult`` — exact integer math (``×11//10`` / ``×9//10`` / ``×1``), not float. The single source
    of truth for the L100 stat formula (the incoming-damage encoder uses it for its no-prior
    fallback; the nature/EV label's throwing guard passes the set's TRUE IV)."""
    pre = 2 * base + iv + ev // 4 + 5
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
