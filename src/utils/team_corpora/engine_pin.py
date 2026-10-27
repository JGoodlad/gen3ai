"""THE ENGINE PIN (``gen3_corpus_engine_pin_v1``): a frozen corpus's engine-coverage verdict is a PINNED INPUT.

Every corpus builder drops (``w_forum_v1``) or never draws (``d3_setmut_v1``, ``n1_proc_v1``) an id the Rust engine
cannot run, as judged by its coverage oracle ``scan_move_probe``. That oracle GROWS: the gen-3 full-parity batches taught
the engine Blast Burn first (``2016f5c2e``), then Shell Bell, the two-turn family, Spite, Psywave and more. A builder
that asked the LIVE oracle stopped reproducing its committed corpus that day (all three ``--check`` failed on main,
2026-10-27) although no shipped file changed, so ``--check`` could no longer tell a moved input from a moved corpus.

So each frozen corpus records the oracle's answer AT ITS BUILD as data, ``engine_pins.json`` (copied from the field of
its own manifest that recorded it), and its builder decides by the PIN. The live oracle is still asked, and:

* an id the pin calls PLAYABLE that the live engine cannot run RAISES :class:`EnginePinError`: the engine LOST coverage,
  and the shipped corpus may carry an id that now crashes;
* an id the pin calls UNPLAYABLE that the live engine now runs is printed on stderr: coverage grew, the frozen corpus
  keeps its build-time rule, and only a NEW corpus version (a new name, a new pin) may take it in.

Never edit a pin to make ``--check`` pass: a pin that differs from the manifest it was copied from fails
``engine_pin_test.py``.
"""
from __future__ import annotations

import functools
import json
import sys
from typing import Dict, List, Mapping

from utils.paths import repo_path

SCHEMA = "gen3_corpus_engine_pin_v1"
PINS_REL = ("src", "utils", "team_corpora", "engine_pins.json")
KINDS = ("species", "move", "item", "ability")


class EnginePinError(RuntimeError):
    """The live engine cannot run an id a frozen corpus's pin calls playable, or the corpus has no pin."""


@functools.lru_cache(maxsize=1)
def load() -> Dict[str, Dict]:
    """``{corpus: {"built_at", "recorded_in", "unplayable": {kind: [id, ...]}}}``."""
    doc = json.loads(repo_path(*PINS_REL).read_text())
    if doc.get("schema") != SCHEMA:
        raise EnginePinError(f"{'/'.join(PINS_REL)}: schema {doc.get('schema')!r} != {SCHEMA!r}")
    return doc["pins"]


def pinned(name: str) -> Dict[str, frozenset]:
    """The pinned unplayable ids of corpus ``name``, by kind (every kind present)."""
    pins = load()
    if name not in pins:
        raise EnginePinError(f"no engine pin for corpus {name!r} in {'/'.join(PINS_REL)} ({', '.join(sorted(pins))})")
    rec = pins[name]["unplayable"]
    return {k: frozenset(rec.get(k, ())) for k in KINDS}


def unplayable(name: str, live: Mapping[str, Mapping[str, str]]) -> Dict[str, List[str]]:
    """``{kind: sorted unplayable ids}`` for a build of corpus ``name``, decided by its PIN.

    ``live`` is the oracle's answer over the ids the build asked about, ``{kind: {id: verdict}}`` (``"ran"`` =
    playable; the shape of :func:`utils.ladder_corpus.build.engine_verdicts`). Each returned list is the pin restricted
    to the asked ids. Raises :class:`EnginePinError` when the live engine cannot run an id the pin calls playable."""
    pin = pinned(name)
    out: Dict[str, List[str]] = {}
    lost: Dict[str, List[str]] = {}
    for kind, got in live.items():
        if kind not in pin:
            raise EnginePinError(f"unknown oracle kind {kind!r} (known: {', '.join(KINDS)})")
        live_bad = {i for i, v in got.items() if v != "ran"}
        if live_bad - pin[kind]:
            lost[kind] = sorted(live_bad - pin[kind])
        grown = sorted((pin[kind] & set(got)) - live_bad)
        if grown:
            print(f"[engine-pin] {name}: the live engine now runs {kind} {grown}; the frozen corpus keeps its "
                  f"build-time verdict", file=sys.stderr)
        out[kind] = sorted(pin[kind] & set(got))
    if lost:
        raise EnginePinError(f"{name}: the live engine cannot run {lost}, which the pin calls playable; the engine lost "
                             "coverage and the shipped corpus may carry an id that now crashes")
    return out
