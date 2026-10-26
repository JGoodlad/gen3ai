"""The categorical-id vocabulary guard (``gen3_vocab_bounds_v1``, GIGO audit 2026-10-26 C-F1).

Every species / move / item / ability id the observation carries is the dex ``num`` of an entry in
``data/pokemon/gen3_*.json``, and the model reads it with ``torch.nn.Embedding(VOCAB_*)``. A dex num at or above
its table's rows is an out-of-range index: an IndexError on CPU, a device-side assert on CUDA, or — through any
clamp — a silent collision onto another row. Nothing pinned that relation (the headroom was species 386 of
400, moves 370 of 400, items 499 of 600, abilities 76 of 100); :func:`check_vocab_bounds` is the throwing guard,
run by ``Gen3ObservationEncoder`` on every mapping it is built from, and ``vocab_bounds_test.py`` runs it on the
shipped dex. The Rust encoder's tables hold the twin guard (``encoder/data.rs``): every id the Rust encoder
writes comes from a table that refused to load a num at or above its bound.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping

from agents.observation.constants import VOCAB_ABILITIES, VOCAB_ITEMS, VOCAB_MOVES, VOCAB_SPECIES

#: ``mappings`` category -> (the vocabulary's rows, its layout key).
VOCABS = {
    "species": (VOCAB_SPECIES, "max_species"),
    "moves": (VOCAB_MOVES, "max_moves"),
    "items": (VOCAB_ITEMS, "max_items"),
    "abilities": (VOCAB_ABILITIES, "max_abilities"),
}


class VocabBoundError(ValueError):
    """A dex entry's ``num`` is not in ``[0, rows)`` of its embedding table."""


def check_vocab_bounds(mappings: Mapping[str, Any]) -> Dict[str, int]:
    """Raise :class:`VocabBoundError` naming the first entry whose ``num`` is outside ``[0, rows)`` of its
    vocabulary; returns each category's largest num (the headroom). A category absent from ``mappings`` is
    skipped (a unit test's partial mapping); a row without a ``num`` is not a dex row (``reverse``,
    ``ability_priors`` ...) and is skipped by the category list."""
    largest: Dict[str, int] = {}
    for category, (rows, key) in VOCABS.items():
        table = mappings.get(category)
        if not table:
            continue
        hi = -1
        for name, entry in table.items():
            if not isinstance(entry, dict) or "num" not in entry:
                continue
            num = entry["num"]
            if isinstance(num, bool) or not isinstance(num, int) or not 0 <= num < rows:
                raise VocabBoundError(
                    f"{category} {name!r} has dex num {num!r}, outside [0, {rows}) — the {key} embedding table "
                    f"({rows} rows) cannot hold it. Widen the vocabulary (agents/observation/constants.py AND "
                    f"src/rust_sim/src/encoder/layout.rs; a retrain-class weight-shape change), never clamp it.")
            hi = max(hi, num)
        largest[category] = hi
    return largest
