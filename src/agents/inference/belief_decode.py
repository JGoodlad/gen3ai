"""Decode the hidden-opponent belief head's species prediction into human-readable top-k guesses.

The inverse of ``agents.observation.belief_labels``: that builds the privileged species-NUM targets
the ``BeliefHead`` is trained against; this turns the head's per-slot species logits — stashed at
``features_extractor.last_belief_logits["species"]`` (shape ``[6, max_species]``, **indexed by
national-dex num**, the SAME index space as the labels) — back into the model's most-likely species
per *un-revealed* opponent slot. It answers, for the eval forensic trace, "what does the policy think
the still-hidden opponent mons are?".

Pure + dependency-light (the num→name map is injected; the default is the ``gen3_data`` facade) so it
unit-tests with synthetic logits, independent of any model/battle. Read ONLY at eval / trace time —
never on the obs build or training hot path.
"""
from __future__ import annotations

from typing import Dict, List, Optional
import numpy as np

from agents.gen3_data import species as _species

# How many candidate species to surface per hidden slot (the "2 or 3 most likely mons").
BELIEF_TOPK = 3

_num_to_id_cache: Optional[Dict[int, str]] = None


def _num_to_species_id() -> Dict[int, str]:
    """Lazy ``{national-dex num -> species id}`` inverse of the species vocab the model trained on
    (the same ``gen3_species.json`` source that produced the belief labels' ``species_to_num``)."""
    global _num_to_id_cache
    if _num_to_id_cache is None:
        # BASE FORMS ONLY (`gen3_species_formes_v1`): an alternate forme (Deoxys-Speed,
        # Unown-B, Castform-Sunny) SHARES its base's national-dex num, so a plain
        # comprehension over `raw()` would be last-write-wins and decode num 386 as
        # "deoxysspeed". The obs species channel is the num, so the base id is the only
        # honest decode.
        raw = _species.raw()
        _num_to_id_cache = {
            int(raw[sid]["num"]): sid
            for sid in _species.base_form_ids()
            if raw[sid].get("num")
        }
    return _num_to_id_cache


def top_species_per_slot(
    species_logits: np.ndarray,
    num_to_name: Optional[Dict[int, str]] = None,
) -> List[dict]:
    """Per opponent slot, the model's single most-likely species — the CONTENT of that slot.

    The unmasked, top-1 sibling of :func:`decode_species_belief`, and it exists for `β` (the
    opponent-intent switch head, `gen3_opp_intent_v1`): `β` points at a SLOT, and a believed slot is
    an anonymous DETR query, so "β says slot 4" means nothing on its own. The model's own species
    posterior is what names it — the SAME content-addressing `β`'s training target uses
    (`opp_intent.resolve_believed_slot_by_content`), which is what makes the rendered sentence
    *"they will switch to Blissey"* refer to the object the head actually pointed at.

    Every slot is decoded, not just the believed ones: `β`'s candidate mask is alive-and-not-active,
    which includes REVEALED bench mons. On a revealed slot the posterior is un-supervised (the
    species aux only scores believed slots), so it is the model's read rather than a certainty —
    surfaces should say so rather than presenting it as the board.

    Returns ``[{"slot": int, "species": name, "prob": float}, ...]`` in slot order.
    """
    if num_to_name is None:
        num_to_name = _num_to_species_id()
    logits = np.asarray(species_logits, dtype=np.float64)
    out: List[dict] = []
    for slot in range(logits.shape[0]):
        row = logits[slot]
        probs = np.exp(row - row.max())
        probs /= probs.sum()
        best = int(np.argmax(probs))
        out.append({"slot": int(slot),
                    "species": num_to_name.get(best, f"num_{best}"),
                    "prob": float(probs[best])})
    return out


def decode_species_belief(
    species_logits: np.ndarray,
    believed_mask: np.ndarray,
    top_k: int = BELIEF_TOPK,
    num_to_name: Optional[Dict[int, str]] = None,
) -> List[dict]:
    """Per un-revealed opponent slot, the model's top-``k`` most-likely species.

    species_logits : ``[n_slots, n_species]`` float — the belief head's per-slot species logits
                     (logit index == national-dex num, the label index space).
    believed_mask  : ``[n_slots]`` bool — True where the slot is un-revealed (hidden). Only these are
                     decoded: a revealed slot's token is the real mon, not a guess.
    top_k          : how many candidate species to surface per hidden slot.
    num_to_name    : ``{num -> display name}``; defaults to the gen3 species-id map.

    Returns a list (believed slots only, in slot order) of
        ``{"slot": int, "top": [{"species": name, "prob": "NN.N%"}, ...]}``
    — empty when no slot is believed (everything revealed)."""
    if num_to_name is None:
        num_to_name = _num_to_species_id()
    logits = np.asarray(species_logits, dtype=np.float64)
    mask = np.asarray(believed_mask, dtype=bool)
    n_slots, n_species = logits.shape
    k = max(0, min(top_k, n_species))
    out: List[dict] = []
    for slot in range(n_slots):
        if slot >= mask.shape[0] or not mask[slot]:
            continue
        row = logits[slot]
        # Numerically-stable softmax over the full species vocab → a calibrated posterior, so the
        # surfaced probabilities reflect how confident the model is (top-k may sum to < 1).
        probs = np.exp(row - row.max())
        probs /= probs.sum()
        top_idx = np.argsort(probs)[::-1][:k]
        out.append({
            "slot": int(slot),
            "top": [
                {"species": num_to_name.get(int(n), f"num_{int(n)}"),
                 "prob": f"{float(probs[n]) * 100:.1f}%"}
                for n in top_idx
            ],
        })
    return out
