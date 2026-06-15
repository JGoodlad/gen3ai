"""Gen 3 legal movepool — a *concept module* (the ``moves.py`` pattern).

Which moves each species can LEGALLY learn in gen 3, keyed by species id. This is **reference
data** (derived once from the learnset source, never changes during a battle), poke-env-free,
reached via the facade as ``gen3_data.learnset``.

It is distinct from ``gen3_data.priors.moves`` (the Smogon usage prior — how OFTEN a legal move is
run): this is the *hard legality gate*. The move-belief prior uses it to PRUNE impossible candidate
moves (a species can't run a move it can't learn) so the "guess unrevealed moves" path only ever
proposes learnable moves. The runtime reads only ``data/`` through here — the poke-env learnset
source is acquisition-layer only (``tools/pokemon_data_extractor/sync.py::build_learnset``).

Tolerance contract (so the gate never *wrongly* prunes): an unknown species — one with no learnset
entry — yields ``None`` from :func:`get_legal_moves` and ``True`` from :func:`is_legal`. The caller
must read that as "no legality constraint" (fall back to the usage prior), NEVER as "no legal moves"
— legality only ever PRUNES when we positively know the movepool.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Optional

from . import _base

# Parsed once: {species_id: frozenset(move_ids)}. The committed json is {species_id: [move_id, ...]}.
_dex = _base.singleton(
    lambda: {sid: frozenset(moves) for sid, moves in _base.load_json("gen3_learnset.json").items()}
)


def get_legal_moves(species: str) -> Optional[FrozenSet[str]]:
    """The legal gen3 move ids for ``species``, or ``None`` when the species has no learnset entry.

    ``None`` means "no legality constraint" (the caller should fall back to the usage prior), NOT
    "no legal moves" — never use this to zero a species' whole candidate set."""
    return _dex().get(species)


def legal_moves_for_species(species: str) -> FrozenSet[str]:
    """The legal gen3 move ids for ``species``; raises ``KeyError`` on an unknown species
    (crash-don't-drop, decision-facing — use when a missing movepool is a real bug)."""
    moves = _dex().get(species)
    if moves is None:
        raise KeyError(f"no gen3 learnset for species {species!r}")
    return moves


def is_legal(species: str, move_id: str) -> bool:
    """Whether ``species`` can legally learn ``move_id`` in gen3. An unknown species returns
    ``True`` (no movepool known → no constraint → never prune); a known species returns membership."""
    moves = _dex().get(species)
    return True if moves is None else move_id in moves


def raw() -> Dict[str, list]:
    """The parsed ``{species_id: [move_id, ...]}`` dict (used by tests / mapping assembly)."""
    return _base.load_json("gen3_learnset.json")
