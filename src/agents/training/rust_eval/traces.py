"""THE CORE TRACE (``gen3_core_trace_v1``, M5 Lane H) — one captured eval game, persisted from the core.

The Python eval path writes a trace from a live poke-env battle (``battle_recorder.write_battle_record``).
The Rust path has no poke-env battle: its AUTHORITY is the game's input log, replayed by the core's
record writer (``ffi.trace`` → ``crate::trace``) into each side's ``gen3_core_event_v1`` record and
the reconstruction record. Files, per captured game (``prefix`` = ``trace_filename_stem`` in the
opponent's dir, the name the prober's ``_FNAME_RE`` inverts):

* ``<prefix>.p1.jsonl.gz`` / ``<prefix>.p2.jsonl.gz`` — the two ``gen3_core_event_v1`` records (the
  TEXT is the authority; ``core_events --check-records`` reads them);
* ``<prefix>_reconstruction.json`` — the reconstruction record (``sim_bridge``'s ``__RECON__``
  shape) with ``battle_tag`` and ``trainee_username``: the prober's replay views drive it;
* ``<prefix>_states.npz`` — the trainee's served model I/O per decision, in the Python recorder's
  keys: ``obs`` [T, D] f32 (the core's row — byte-equal to the Python encoder's, slice N), ``logits``
  [T, 11] f32 (T2's legal LOG-PROBS; an illegal entry is ``ILLEGAL_LOGIT`` — the ordering among legal
  actions, which is what a reader ranks by, is the logits'), ``values`` [T] f32 (V(s)), ``win_probs``
  [T] NaN (T2 serves no auxiliary head — the prober's ``analyze`` re-runs the model on ``obs``),
  ``has_state`` [T] int8 1, ``actions`` [T] int16, ``action_mask`` [T, 11] bool;
* ``<prefix>_summary.json`` — ``meta`` only (the Python recorder's ``meta`` keys + ``trace_source``
  naming the files above). The per-decision ``invocations`` / ``teams`` are EXPANDED by the prober
  from the records (``main.prober.core_trace``) — the summary does not duplicate them.
"""
from __future__ import annotations

import gzip
import json
import os
from typing import Any, Dict, Optional

import numpy as np

SCHEMA = "gen3_core_trace_v1"
EVENT_SCHEMA = "gen3_core_event_v1"
#: The stored logit of an illegal action (finite, so an unmasked softmax reads it as 0).
ILLEGAL_LOGIT = -1.0e9


def write_core_trace(prefix: str, *, trace: Dict[str, Any], step: int, battle_id: str, result: str,
                     draw_kind: Optional[str], turns: int, trainee_username: str, obs: np.ndarray,
                     logp: np.ndarray, values: np.ndarray, actions: np.ndarray, masks: np.ndarray,
                     cycle: Optional[Dict[str, Any]] = None, turn_limit: Optional[int] = None) -> Dict[str, str]:
    """Write one captured game (module docs). ``trace`` is ``ffi.trace``'s dict. Returns the file map."""
    from agents.training.trace_result import DRAW, DRAW_KIND_KEY, RESULT_VOCABULARY, VOCABULARY_KEY

    os.makedirs(os.path.dirname(prefix), exist_ok=True)
    base = os.path.basename(prefix)
    files = {"p1": f"{base}.p1.jsonl.gz", "p2": f"{base}.p2.jsonl.gz",
             "reconstruction": f"{base}_reconstruction.json", "states": f"{base}_states.npz"}
    d = os.path.dirname(prefix)
    for side in ("p1", "p2"):
        with gzip.open(os.path.join(d, files[side]), "wt") as f:
            f.write(trace["records"][side])
    recon = dict(trace["recon"], battle_tag=battle_id, trainee_username=trainee_username)
    with open(os.path.join(d, files["reconstruction"]), "w") as f:
        json.dump(recon, f, indent=1)
    lp = np.asarray(logp, dtype=np.float32)
    logits = np.where(np.isfinite(lp), lp, np.float32(ILLEGAL_LOGIT)).astype(np.float32)
    t = int(lp.shape[0])
    np.savez_compressed(os.path.join(d, files["states"]),
                        obs=np.asarray(obs, dtype=np.float32), logits=logits,
                        values=np.asarray(values, dtype=np.float32),
                        win_probs=np.full(t, np.nan, dtype=np.float32),
                        has_state=np.ones(t, dtype=np.int8),
                        actions=np.asarray(actions, dtype=np.int16),
                        action_mask=np.asarray(masks, dtype=bool))
    meta: Dict[str, Any] = {"step": int(step), "battle_id": battle_id, "result": result, "turns": int(turns),
                            "invocations": t, VOCABULARY_KEY: RESULT_VOCABULARY}
    if result == DRAW:
        meta[DRAW_KIND_KEY] = draw_kind
    meta["trace_source"] = {"schema": SCHEMA, "kind": EVENT_SCHEMA, "env_core": "rust",
                            "trainee_side": "p1", "records": {"p1": files["p1"], "p2": files["p2"]},
                            "reconstruction": files["reconstruction"], "states": files["states"],
                            **({"turn_limit": int(turn_limit)} if turn_limit is not None else {}),
                            **({"cycle": cycle} if cycle else {})}
    with open(f"{prefix}_summary.json", "w") as f:
        json.dump({"meta": meta}, f, indent=2)
    return {**files, "summary": f"{base}_summary.json"}


def is_core_trace(summary: Dict[str, Any]) -> bool:
    """A summary written by :func:`write_core_trace` (its invocations are expanded on read)."""
    src = (summary or {}).get("meta", {}).get("trace_source") or {}
    return src.get("schema") == SCHEMA
