"""What a divergence MEANS for the cutover — the taxonomy the readiness gate is read by.

The parity slices compare two different pairs, and only one of them is the cutover's question:

* **CUTOVER** — the core against the Python path TRAINING reads today: every slice-E event field,
  every slice-T tracker / label / reward, every slice-O row byte and mask, slice V's ``[core]``
  column (``present()`` + legality + the 11-bit mask against the reading), every ``[ALIGN]``
  decision alignment, every slice-N env output, and every REFUSED battle. **The gate: zero.**
* **READING** — the reading (poke-env's, which the core mirrors by design, so BOTH paths hold it)
  against the SIMULATOR's truth: slice V's ``[BOARD]`` audit, its ``[TRUTH]`` checks, and a bare
  ``[SIM-FACT]`` field (the view road's projection reads the engine). Not a cutover blocker — the
  two paths agree — but every class is a poke-env reading FINDING, reported with its rate.
* **VIEW-ROAD** — a bare ``[PRESENTATION/…]`` field: the reading against the legacy view road's
  projection (``view.rs``), which training never reads and which is on the deletion manifest
  (program §4, M2 row). Reported, never a blocker.
"""
from __future__ import annotations

from typing import Dict, Iterable

CUTOVER, READING, VIEW_ROAD = "CUTOVER", "READING", "VIEW-ROAD"
#: Slice N differences that follow a PHANTOM step of the same episode (finding F1, 2026-09-24: the
#: live env records a trainee decision on a step the trainee is not asked to move; the core never
#: does). Still a CUTOVER difference — training inputs change at the switch — but NAMED, so an
#: unexplained class can never hide behind it.
CUTOVER_F1 = "CUTOVER-F1 (named: phantom decision)"


def is_f1_field(key: str) -> bool:
    """The obs fields a phantom ``EpisodeTracker.record`` + ``update_progress_clock`` can move: the
    progress clock (``board +2``, turns since progress) and recency. Any OTHER field that differs
    after a phantom is NOT explained by F1."""
    return key == "observation board +2" or (key.startswith("observation ") and " recency+" in key)


def category(slice_: str, key: str) -> str:
    if key.startswith("REFUSED"):
        return CUTOVER
    if slice_ == "N" and key.endswith("[after-phantom]") and is_f1_field(key[: -len(" [after-phantom]")]):
        return CUTOVER_F1
    if slice_ != "V":
        return CUTOVER
    if key.startswith("[core]") or key.startswith("[ALIGN]"):
        return CUTOVER
    if key.startswith("[BOARD]") or key.startswith("[TRUTH]") or key.startswith("[SIM-FACT]"):
        return READING
    if key.startswith("[PRESENTATION"):
        return VIEW_ROAD
    return CUTOVER   # an unrecognised key is treated as the strictest class, never waved through


def categories(classes: Dict[str, Dict[str, int]]) -> Dict[str, int]:
    """``{category: divergent fields}`` for one battle's ``{slice: {key: n}}``."""
    out: Dict[str, int] = {}
    for s, keys in classes.items():
        for k, n in keys.items():
            c = category(s, k)
            out[c] = out.get(c, 0) + n
    return out


def census(records: Iterable[dict]) -> Dict[str, Dict[str, dict]]:
    """``{category: {"slice key": {"battles": n, "fields": m, "example": label}}}`` over divergent
    battle records (``{"label", "classes"}``)."""
    out: Dict[str, Dict[str, dict]] = {}
    for rec in records:
        for s, keys in rec.get("classes", {}).items():
            for k, n in keys.items():
                c = out.setdefault(category(s, k), {}).setdefault(f"{s} {k}", {"battles": 0, "fields": 0,
                                                                                "example": rec.get("label")})
                c["battles"] += 1
                c["fields"] += n
    return out
