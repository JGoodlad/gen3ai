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

# src/agents/gen3_data/_base.py -> repo root is parents[3]; data lives under data/pokemon/.
_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "pokemon"


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


def singleton(builder: Callable[[], Any]) -> Callable[[], Any]:
    """Wrap a zero-arg builder so it runs at most once and caches its result — the lazy-load
    idiom every dex uses (parse on first access, reuse forever after)."""
    cache: Dict[str, Any] = {}

    def get() -> Any:
        if "v" not in cache:
            cache["v"] = builder()
        return cache["v"]

    return get
