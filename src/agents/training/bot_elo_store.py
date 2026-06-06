"""Resumable bot-vs-bot game-count store — the accumulator behind the ELO anchor.

Pure data layer (no bridge / poke-env), split out of ``bot_elo_calibration.py`` so the
calibration script stays focused on *playing battles + driving*, and so the store is
independently unit-testable.

Schema::

    {"git_hash", "names", "updated_at",
     "pairs": {"a|b": {"a", "b", "wins_a", "wins_b", "games"}}}

Pair keys are sorted and wins are tracked for the **lexicographically-lower** name, so counts
sum order-consistently across resumes AND across a fleet of independent processes (see
:func:`merge`).
"""
from __future__ import annotations

import os
import json
from datetime import datetime, timezone


def pair_key(a: str, b: str) -> str:
    return "|".join(sorted((a, b)))


def atomic_write_json(path: str, obj: dict) -> None:
    """Write JSON via tmp+rename so a reader never sees a half-written file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def new_store(git_hash, names) -> dict:
    return {"git_hash": git_hash, "names": list(names), "pairs": {}}


def load(path: str, names: list[str], git_hash, reset: bool = False) -> dict:
    """Load the accumulated counts, or a fresh empty store. Warns (and keeps the counts) if the
    recorded git_hash differs — pass ``reset=True`` to discard stale games."""
    if not reset and os.path.exists(path):
        try:
            with open(path) as f:
                store = json.load(f)
        except (OSError, ValueError):
            store = None
        if store and isinstance(store.get("pairs"), dict):
            if store.get("git_hash") and git_hash and store["git_hash"] != git_hash:
                print(f"⚠️  store git_hash {store['git_hash'][:8]} != current {git_hash[:8]} — "
                      f"bot logic may have changed; counts kept (use --reset to discard).")
            store["names"] = list(names)
            return store
    return new_store(git_hash, names)


def save(path: str, store: dict) -> None:
    store["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(path, store)


def games(store: dict, a: str, b: str) -> int:
    return store["pairs"].get(pair_key(a, b), {}).get("games", 0)


def accumulate(store: dict, a: str, b: str, wins_a: int, wins_b: int, n_games: int) -> None:
    """Add a chunk's results, normalizing wins to the lexicographically-lower name so the
    entry is order-stable however the caller passed (a, b)."""
    lo = min(a, b)
    e = store["pairs"].setdefault(
        pair_key(a, b), {"a": lo, "b": max(a, b), "wins_a": 0, "wins_b": 0, "games": 0})
    if a == lo:
        e["wins_a"] += wins_a
        e["wins_b"] += wins_b
    else:
        e["wins_a"] += wins_b
        e["wins_b"] += wins_a
    e["games"] += n_games


def merge(paths: list[str], names: list[str], git_hash) -> dict:
    """Sum per-pair counts across multiple stores (a fleet of independent processes each wrote
    its own). Order-consistent because every store uses the same sorted-key + lower-name
    convention; disjoint or overlapping pair sets both merge correctly."""
    merged = new_store(git_hash, names)
    for p in paths:
        try:
            with open(p) as f:
                s = json.load(f)
        except (OSError, ValueError):
            print(f"⚠️  skipping unreadable store {p}")
            continue
        for k, e in (s.get("pairs") or {}).items():
            m = merged["pairs"].setdefault(
                k, {"a": e["a"], "b": e["b"], "wins_a": 0, "wins_b": 0, "games": 0})
            m["wins_a"] += e.get("wins_a", 0)
            m["wins_b"] += e.get("wins_b", 0)
            m["games"] += e.get("games", 0)
    return merged


def results_and_matrix(store: dict, names: list[str]):
    """``(results, win_matrix)`` — ``results`` is ``[(a, b, wins_a, games)]`` for
    ``elo.fit_pairwise``; ``win_matrix[a][b]`` = a's observed win rate vs b."""
    results = []
    win_matrix = {a: {} for a in names}
    for e in store["pairs"].values():
        a, b, wa, wb, g = e["a"], e["b"], e["wins_a"], e["wins_b"], e["games"]
        if g <= 0:
            continue
        results.append((a, b, wa, g))
        win_matrix[a][b] = wa / g
        win_matrix[b][a] = wb / g
    return results, win_matrix
