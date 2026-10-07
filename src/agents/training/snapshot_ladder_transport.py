"""THE LADDER'S TRANSPORT BOUNDARY — which stack played an edge, and the refusals that keep two stacks apart (P2).

Until 2026-10-06 every ``snapshot_ladder/games.jsonl`` row was played by two poke-env ``RLPlayer`` s with the PYTHON
encoder (``python_bridge``: unmirrored, the newer node always on seat p1, a draw folded into ``b`` 's column). From
then on every new row is played on the Rust eval engine (``rust_eval``, ``snapshot_ladder_play``: seat-balanced
mirrored pairs, Rust rows, draws recorded and excluded from the edge). The two are DIFFERENT MEASUREMENTS of the same
frozen pair, so (decision of record: ``designs/training/eval_and_rating.md`` "The TRANSPORT boundary"):

* every row carries its ``transport`` (an absent key = ``python_bridge``: every row written before the key existed);
* a fit over rows of BOTH transports is REFUSED (:func:`fit_transport`) unless the caller names one transport (the
  fit then reads only its rows) or accepts the mix (stamped ``mixed`` in the recipe block);
* ``ladder.json``'s recipe block names its transport (:func:`ladder_transport`; absent = ``python_bridge``), and a
  reader that compares two ladders refuses a mismatch unless told (:func:`check_same_transport`).

Pure reads (no torch, no poke-env): ``snapshot_ladder`` re-exports every name here.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Dict, List, Optional, Set, Tuple

TRANSPORT_PYTHON = "python_bridge"   #: poke-env RLPlayer + the Python encoder; every row without a `transport` key
TRANSPORT_RUST = "rust_eval"         #: the Rust eval engine (`snapshot_ladder_play`); what this tree PLAYS
TRANSPORTS = (TRANSPORT_PYTHON, TRANSPORT_RUST)
#: A fit whose edges came from both transports, allowed only on request.
TRANSPORT_MIXED = "mixed"
#: ``snapshot_ladder.EVAL_CYCLE_SOURCE`` (that module asserts the two are equal): a v2 row that is never an edge.
_EVAL_CYCLE = "eval_cycle"


class LadderTransportError(RuntimeError):
    """A fit or a comparison would mix ladder edges played on two transports (module docstring)."""


def row_transport(row: dict) -> str:
    """The transport a ``games.jsonl`` row was played on: its ``transport`` key, else ``python_bridge``. An unknown
    value is refused."""
    t = row.get("transport", TRANSPORT_PYTHON)
    if t not in TRANSPORTS:
        raise LadderTransportError(f"games.jsonl row {row!r}: unknown transport {t!r} (known: {TRANSPORTS})")
    return t


def check_transport_name(transport: Optional[str]) -> None:
    if transport is not None and transport not in TRANSPORTS:
        raise LadderTransportError(f"transport {transport!r} not in {TRANSPORTS}")


def _pair_key(a: int, b: int) -> Tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def _rows(path: str):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                key = _pair_key(int(r["a"]), int(r["b"]))
            except (json.JSONDecodeError, KeyError, ValueError):
                continue
            yield key, r


def pair_transports(path: str) -> Dict[Tuple[int, int], Set[str]]:
    """{(lo, hi): {transport, …}} over a ``games.jsonl``'s rating-edge rows (``eval_cycle`` rows skipped)."""
    out: Dict[Tuple[int, int], Set[str]] = {}
    for key, r in _rows(path):
        if r.get("source") == _EVAL_CYCLE:
            continue
        out.setdefault(key, set()).add(row_transport(r))
    return out


def rows_on_transport(path: str, a: int, b: int, transport: str) -> int:
    """How many rows ``(a, b)`` (either order) already has on ``transport``."""
    key = _pair_key(a, b)
    return sum(1 for k, r in _rows(path) if k == key and row_transport(r) == transport)


def fit_transport(path: str, kept: Optional[Callable[[int, int], bool]] = None, *,
                  transport: Optional[str] = None, allow_mixed: bool = False, run_dir: str = "<run>") -> str:
    """The ONE transport a fit over the kept pairs reads, or a :class:`LadderTransportError`.

    ``transport`` names it (the fit then reads only those rows); else the rows' own transports decide: one → that
    one; none (no measured pair) → :data:`TRANSPORT_RUST`, what this tree plays; two → REFUSED unless
    ``allow_mixed`` (then :data:`TRANSPORT_MIXED`). ``kept(lo, hi)`` restricts the census to the fit's pairs."""
    if transport is not None:
        check_transport_name(transport)
        return transport
    seen: Dict[str, int] = {}
    for (lo, hi), ts in pair_transports(path).items():
        if kept is not None and not kept(lo, hi):
            continue
        for t in ts:
            seen[t] = seen.get(t, 0) + 1
    if not seen:
        return TRANSPORT_RUST
    if len(seen) == 1:
        return next(iter(seen))
    if allow_mixed:
        return TRANSPORT_MIXED
    raise LadderTransportError(
        f"{path}: the ladder's pairs were played on TWO transports — "
        + ", ".join(f"{n} pair(s) on {t}" for t, n in sorted(seen.items()))
        + ". The Python ladder (poke-env RLPlayer + the Python encoder, unmirrored, the newer node always on p1, a draw "
          "folded into b's column) and the Rust ladder (the eval engine, seat-balanced mirrored pairs, draws excluded) "
          "are different measurements (designs/training/eval_and_rating.md \"The TRANSPORT boundary\"). Refusing to fit "
          "them as one.\nFIX: fit ONE transport —\n"
          f"    python -m agents.training.snapshot_ladder {run_dir} --fit-only --transport rust_eval\n"
          f"        (re-measure the old pairs on Rust first: `--backfill` plays every pair the Rust transport lacks)\n"
          f"or accept the mix, stamped `mixed` in the recipe block: --allow-mixed-transport")


def ladder_transport(ladder: dict) -> str:
    """The transport a fitted ladder's edges came from: its recipe block's ``transport``, else ``python_bridge`` —
    every ``ladder.json`` fitted before the key existed was fitted from Python-played rows. NOT part of the recipe
    status (the FIT did not change, so a refit reproduces the same file): a reader comparing ladders asks this."""
    r = ladder.get("recipe")
    t = r.get("transport") if isinstance(r, dict) else None
    return str(t) if t else TRANSPORT_PYTHON


def check_same_transport(ladders: Dict[str, dict], *, allow: bool = False) -> List[str]:
    """REFUSE (:class:`LadderTransportError`) ladders read side by side whose edges came from different transports —
    or any ``mixed`` one — unless ``allow``; then return the warning lines a caller prints beside its numbers.
    ``ladders`` maps a label to a ladder dict. An empty list = one transport, nothing to say."""
    ts = {label: ladder_transport(d) for label, d in ladders.items()}
    if len(set(ts.values())) <= 1 and TRANSPORT_MIXED not in ts.values():
        return []
    msg = ("ladders played on different transports are different measurements (designs/training/eval_and_rating.md "
           "\"The TRANSPORT boundary\"): " + ", ".join(f"{k}: {v}" for k, v in sorted(ts.items())))
    if not allow:
        raise LadderTransportError(msg + " — refusing to compare them; re-measure one side on the other's transport, "
                                   "or pass the reader's allow-transport-mix flag to read them with this caveat")
    return [f"⚠️  TRANSPORT MIX ACCEPTED: {msg}"]
