"""X5's DEX-ROW TABLE — the per-mon observation row of a HYPOTHESISED opponent mon, one per species
(`gen3_x5_dex_rows_v1`; `designs/endstate/design_x5_belief_tokens.md` §3.4, build unit U1).

A hypothesis seat (X5's `fixed_mass` arm) is a CONCRETE opponent mon of species ``s``, encoded by the
same ``pokemon_encoder`` as a revealed one. Its 122-dim per-mon row is the row THE observation
encoder (the Rust core's ``BattleVersion::encode`` slot writer) writes for "species s present,
unrevealed set, full HP, no status": ``encoder::hypothesis::hypothesis_slot`` in
``src/rust_sim/src/encoder/hypothesis.rs``. There is NO second encoder — this module never computes
a cell; it only renders what the Rust encoder emitted, and loads it back.

The table is a COMMITTED artifact beside the model code (``hypothesis_dex_rows.json``), not under
``data/``, so a pinned run isolates it with its code. Three gates hold it:

* ``hypothesis_dex_rows_sim_test.py`` (``sim``) regenerates it through the encoder and requires the
  file BYTE-equal;
* ``src/rust_sim/tests/hypothesis_dex_rows_test.rs`` (``cargo test``) — the REAL-STATE cross-check:
  at the first appearance of every base-form species in real Rust battles, the encoder's real row
  equals the synthetic row on every cell outside the DECLARED on-field blocks (``cells`` below);
* ``hypothesis_dex_rows_test.py`` (unmarked) loads it and checks its shape against
  ``Gen3ObservationEncoder.get_layout()``.

**Which species.** One row per BASE-FORM species (``gen3_data.species.base_form_ids()``), keyed by its
national-dex ``num`` — the num axis the T0 species prior and every ``table[species.num]`` buffer use
(a forme shares its base's num; formes get no row of their own: `gen3_species_formes_v1`). The
loader returns a DENSE ``[max_species, POKEMON_FULL_DIM]`` array, zero where no species has that num
(the sentinel 0, and nums past the last dex entry), with a ``valid`` mask.

**Format.** JSON, deterministic, one row per line: each float32 cell written as its SHORTEST
round-trip decimal (numpy's ``str(np.float32)``), and a ``sha256`` of the rows' little-endian float32
bytes that the loader re-derives after parsing and REFUSES on mismatch — so the parse is exact by
check, not by assumption.

Regenerate (after any encoder / ``data/pokemon`` change the sim gate flags)::

    PYTHONPATH=src python -m agents.model.hypothesis_dex_rows --write
"""
from __future__ import annotations

import argparse
import base64
import functools
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

#: The committed artifact, beside this module.
ARTIFACT = Path(__file__).with_name("hypothesis_dex_rows.json")
SCHEMA = "gen3_x5_dex_rows_v1"
#: What every row is the encoder's slot OF (recorded in the artifact).
STATE = "species s present, unrevealed set (no item, ability beyond the pokedex's one-ability inference, " \
        "or move revealed), full HP (100/100), no status, never seen (no tracker evidence), not active"


@dataclass(frozen=True)
class CellBlock:
    """One DECLARED block of the 122-dim slot (``encoder::hypothesis::CELLS``).

    ``klass``: ``species`` (depends on the species) · ``default`` (a pristine default a fresh mon also
    carries) · ``revealed_on_field`` (zero/prior here; the field can reveal it — ``flag`` is the cell
    that says so) · ``on_field`` (the mon's on-field state; the cross-check excludes it)."""

    name: str
    lo: int
    hi: int
    klass: str
    flag: Optional[int]


@dataclass(frozen=True)
class HypothesisDexRows:
    """The loaded table. ``rows[num]`` is species ``num``'s hypothesis row (float32, read-only)."""

    rows: np.ndarray                      # [max_species, POKEMON_FULL_DIM] float32
    valid: np.ndarray                     # [max_species] bool — a base-form species holds the num
    species: Tuple[Optional[str], ...]    # the species id per num (None where invalid)
    cells: Tuple[CellBlock, ...]
    sha256: str                           # of the rows' float32 bytes, in file order


# ---------------------------------------------------------------------------------- generation

def _species_in_num_order() -> List[Tuple[int, str]]:
    from agents.gen3_data import species as gs

    out = sorted((gs.species_data(sid).num, sid) for sid in gs.base_form_ids())
    nums = [n for n, _ in out]
    if len(set(nums)) != len(nums) or min(nums) < 1:
        raise ValueError(f"base_form_ids() is not one species per num >= 1: {out[:5]}…")
    return out


def _run_encoder(ids: Sequence[str]) -> Tuple[dict, Dict[str, np.ndarray]]:
    """``core_events --dex-rows`` over ``ids`` → (its header, ``{id: float32 row}``)."""
    from utils.bridge.sim_bridge_bin import resolve_core_events_bin

    p = subprocess.run([resolve_core_events_bin(), "--dex-rows"], input="\n".join(ids) + "\n",
                       capture_output=True, text=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"core_events --dex-rows failed (exit {p.returncode}): {p.stderr.strip()[-2000:]}")
    lines = p.stdout.splitlines()
    head = json.loads(lines[0])
    rows: Dict[str, np.ndarray] = {}
    for ln in lines[1:]:
        d = json.loads(ln)
        raw = base64.b64decode(d["row"])
        row = np.frombuffer(raw, dtype="<f4").astype(np.float32)
        if row.shape != (head["row_dim"],):
            raise RuntimeError(f"{d['species']}: a {row.shape} row for row_dim {head['row_dim']}")
        if np.isnan(row).any():
            raise RuntimeError(f"{d['species']}: a NaN cell — a slot cell the encoder never wrote")
        rows[d["species"]] = row
    if list(rows) != list(ids):
        raise RuntimeError("core_events --dex-rows answered a different species list than it was asked")
    return head, rows


def _cell(x: np.float32) -> str:
    s = str(x)
    if not np.isfinite(x) or np.float32(float(s)).tobytes() != x.tobytes():
        raise ValueError(f"cell {x!r} does not round-trip through {s!r}")
    return s


def render(nums_ids: Sequence[Tuple[int, str]], rows: Sequence[np.ndarray], head: dict) -> str:
    """The artifact's exact text (the byte gate compares it)."""
    stacked = np.stack([np.asarray(r, dtype="<f4") for r in rows]) if rows else np.zeros((0, 0), "<f4")
    sha = hashlib.sha256(stacked.tobytes()).hexdigest()
    cells = ",\n".join("  " + json.dumps(c, separators=(", ", ": ")) for c in head["cells"])
    body = ",\n".join(
        f"  [{num}, {json.dumps(sid)}, [{', '.join(_cell(x) for x in row)}]]"
        for (num, sid), row in zip(nums_ids, stacked))
    return (
        "{\n"
        f'"schema": {json.dumps(SCHEMA)},\n'
        f'"generator": "python -m agents.model.hypothesis_dex_rows --write (core_events --dex-rows; '
        f'src/rust_sim/src/encoder/hypothesis.rs)",\n'
        f'"state": {json.dumps(STATE)},\n'
        f'"row_dim": {int(head["row_dim"])},\n'
        f'"n_rows": {len(rows)},\n'
        f'"sha256": {json.dumps(sha)},\n'
        '"cells": [\n' + cells + "\n],\n"
        '"rows": [\n' + body + "\n]\n"
        "}\n")


def generate() -> str:
    """Regenerate the artifact's text through THE encoder (needs the Rust ``core_events``)."""
    nums_ids = _species_in_num_order()
    head, rows = _run_encoder([sid for _, sid in nums_ids])
    return render(nums_ids, [rows[sid] for _, sid in nums_ids], head)


# ---------------------------------------------------------------------------------- loading

def _parse(text: str, max_species: int) -> HypothesisDexRows:
    doc = json.loads(text)
    if doc.get("schema") != SCHEMA:
        raise ValueError(f"{ARTIFACT.name}: schema {doc.get('schema')!r}, expected {SCHEMA!r}")
    dim = int(doc["row_dim"])
    entries = doc["rows"]
    if len(entries) != int(doc["n_rows"]):
        raise ValueError(f"{ARTIFACT.name}: {len(entries)} rows, the header says {doc['n_rows']}")
    stacked = np.asarray([e[2] for e in entries], dtype="<f4").reshape(len(entries), dim)
    if hashlib.sha256(stacked.tobytes()).hexdigest() != doc["sha256"]:
        raise ValueError(f"{ARTIFACT.name}: the rows do not hash to the recorded sha256 — edited by hand, "
                         "or a cell did not parse back exactly; regenerate it")
    rows = np.zeros((max_species, dim), dtype=np.float32)
    valid = np.zeros(max_species, dtype=bool)
    species: List[Optional[str]] = [None] * max_species
    for (num, sid, _), row in zip(entries, stacked):
        if not 1 <= num < max_species:
            raise ValueError(f"{ARTIFACT.name}: {sid} has num {num}, outside 1..{max_species - 1}")
        if valid[num]:
            raise ValueError(f"{ARTIFACT.name}: num {num} appears twice ({species[num]}, {sid})")
        rows[num], valid[num], species[num] = row, True, sid
    rows.setflags(write=False)
    valid.setflags(write=False)
    cells = tuple(CellBlock(*c) for c in doc["cells"])
    return HypothesisDexRows(rows=rows, valid=valid, species=tuple(species), cells=cells, sha256=doc["sha256"])


@functools.lru_cache(maxsize=None)
def load_hypothesis_dex_rows(max_species: int) -> HypothesisDexRows:
    """The committed table as a dense, num-indexed ``[max_species, POKEMON_FULL_DIM]`` float32 array
    (``max_species`` = ``Gen3ObservationEncoder.get_layout()["max_species"]``). Read-only arrays; a
    consumer that needs a tensor takes ``torch.from_numpy(rows.copy())`` (or registers it as a
    NON-persistent buffer — data-derived, never a saved weight)."""
    return _parse(ARTIFACT.read_text(encoding="utf-8"), max_species)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true", help="regenerate the committed artifact")
    g.add_argument("--check", action="store_true", help="exit 1 if the artifact is stale")
    args = ap.parse_args(argv)
    text = generate()
    if args.write:
        ARTIFACT.write_text(text, encoding="utf-8")
        print(f"wrote {ARTIFACT} ({len(text.encode())} bytes)")
        return 0
    if ARTIFACT.read_text(encoding="utf-8") != text:
        print(f"{ARTIFACT} is STALE — run: python -m agents.model.hypothesis_dex_rows --write", file=sys.stderr)
        return 1
    print(f"{ARTIFACT} is current")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
