"""Shared loader for the ``gen3_data`` concept modules.

One place that resolves a ``data/pokemon/`` file relative to the repo root (so the data loads
regardless of the process CWD), reads + validates the JSON, and memoizes it as a process-global
singleton. Every ``gen3_data`` submodule builds its typed dex on top of :func:`load_json`, so
each file is parsed **once** and the three upstreams (poke-env / Showdown / Smogon) never leak
past the ``tools/`` extractor — the runtime only ever sees ``data/``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Dict

from utils.paths import repo_path

_DATA_DIR = repo_path("data", "pokemon")


def data_path(filename: str) -> Path:
    """Absolute path to a ``data/pokemon/`` file, CWD-independent."""
    return _DATA_DIR / filename


def load_json(filename: str) -> Any:
    """Read + validate a ``data/pokemon/`` JSON file. Crash-don't-drop: a missing or empty file
    raises immediately (a silent gap here would corrupt every observation)."""
    path = data_path(filename)
    if not path.exists():
        raise FileNotFoundError(
            f"CRITICAL: data file missing: {path}. Run tools/pokemon_data_extractor/sync.py first."
        )
    data = json.loads(path.read_text())
    if not data:
        raise ValueError(f"CRITICAL: data file empty: {path}")
    return data


def load_dex_json(filename: str) -> Any:
    """:func:`load_json` for a DEX file (species / moves / items / abilities): additionally every row
    must be an object carrying a numeric ``num``. Crash-don't-drop, at LOAD: a row without one used
    to read as ``num = 0`` through ``get("num", 0)`` — in the typed dex AND in every obs encoder that
    reads the raw rows — i.e. it silently encoded as the dex's index 0 (the Rust encoder tables
    mirror this check, `encoder/data.rs`; F-X5-5, 2026-10-03). No current row trips it."""
    data = load_json(filename)
    for key, row in data.items():
        num = row.get("num") if isinstance(row, dict) else None
        if isinstance(num, bool) or not isinstance(num, (int, float)):
            raise ValueError(
                f"CRITICAL: {filename}: row {key!r} has no numeric `num` (got {num!r}) — it would "
                f"silently encode as 0. Fix the row in tools/pokemon_data_extractor and re-sync.")
    return data


def singleton(builder: Callable[[], Any]) -> Callable[[], Any]:
    """Wrap a zero-arg builder so it runs at most once and caches its result — the lazy-load
    idiom every dex uses (parse on first access, reuse forever after)."""
    cache: Dict[str, Any] = {}

    def get() -> Any:
        if "v" not in cache:
            cache["v"] = builder()
        return cache["v"]

    return get
