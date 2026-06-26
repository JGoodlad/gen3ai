"""Resolve the EXACT opponent that produced an eval trace → a reloadable checkpoint.

The search-teacher MUST reload the exact opponent for both the better-line interior plies and the
confirm rollout (§5) — distilling "A* beats a proxy" instead of "A* beats THIS opponent" is a
soundness failure, not a degradation. So an opponent we can't resolve exactly is **skipped**
(``'unresolved'``), never approximated by the trainee.

v1 scope (the minimal-correct cut): **sentinels + bots**, the dominant self-play case.
- A heuristic **bot** has no obs→action model → ``(None, 'bot')``: the interior plies fall back to the
  RECORDED move at the divergence ply (faithful at depth-1), and the confirm rebuilds the bot from its
  name (bots are reproducible — ``replay_counterfactual`` does this).
- A **sentinel** (a frozen self snapshot) → its ``snapshot_<step>.zip`` under ``models/<run>/snapshots/``.
  Sentinel labels are POSITIONAL (``sentinel_<i>``) and the index→step map slides with the pool, so it
  is recoverable ONLY for the trace's own cycle — and the search-teacher runs on the LATEST eval cycle,
  whose mapping is in ``metadata.json:latest_eval.pool.sentinels[i].snapshot``. A trace from an older
  cycle (or a snapshot aged out of the window) → ``'unresolved'`` (the honest skip).
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from typing import Optional, Tuple

_SENTINEL_PREFIX = "sentinel_"


def _bot_names() -> set:
    try:
        from agents.training.eval_callback import eval_opponent_names
        return set(eval_opponent_names())
    except Exception:  # noqa: BLE001 — fall back to a static roster if the import path moves
        return {"random", "heuristic", "heuristic2", "staller", "staller_v2",
                "aggressive", "aggressive_v2", "setup_sweep", "setup_sweep_v2"}


@lru_cache(maxsize=8)
def _latest_eval_block(run_dir: str) -> "tuple":
    """(eval_step, {sentinel_index: snapshot_filename}) from the run's metadata.json latest_eval pool.
    Cached per run_dir (the metadata is stable within a search-teacher cycle)."""
    meta_path = os.path.join(run_dir, "metadata.json")
    if not os.path.exists(meta_path):
        return (None, {})
    try:
        with open(meta_path) as f:
            le = (json.load(f).get("latest_eval") or {})
    except (OSError, ValueError):
        return (None, {})
    step = le.get("step")
    sents = (le.get("pool") or {}).get("sentinels") or []
    idx_to_snap = {i: s.get("snapshot") for i, s in enumerate(sents) if s.get("snapshot")}
    return (step, idx_to_snap)


def resolve_opponent(run_dir: str, opponent_label: str, step: int) -> Tuple[Optional[str], str]:
    """Map a trace's ``opponent_label`` (+ its eval ``step``) → ``(opponent_ckpt | None, source)`` where
    ``source`` ∈ {``'ckpt'``, ``'bot'``, ``'unresolved'``}. See the module header for the contract."""
    if opponent_label in _bot_names():
        return None, "bot"

    if opponent_label.startswith(_SENTINEL_PREFIX):
        try:
            idx = int(opponent_label[len(_SENTINEL_PREFIX):])
        except ValueError:
            return None, "unresolved"
        eval_step, idx_to_snap = _latest_eval_block(run_dir)
        # Positional sentinel mapping is only valid for the trace's OWN cycle = the latest eval cycle.
        if eval_step is None or (step is not None and int(step) != int(eval_step)):
            return None, "unresolved"
        snap = idx_to_snap.get(idx)
        if not snap:
            return None, "unresolved"
        path = os.path.join(run_dir, "snapshots", snap)
        return (path, "ckpt") if os.path.exists(path) else (None, "unresolved")

    # ext_<run> stable opponents + anything else: deferred to a later phase (v1 skips them honestly).
    return None, "unresolved"
