"""Search-teacher SELECTION worker subprocess — the candidate scan, OFF the training step.

**Why this process exists.** ``SearchTeacherCallback`` is advertised — in its own docstring, in
``src/agents/training/CLAUDE.md`` and in the ``--use-bridge=rust`` startup banner — as
*non-blocking (subprocess workers)*. That was true of the SEARCH half and false of the SELECTION
half: ``_launch`` called ``select_for_mode`` inline in ``_on_step``, which falsifies every loss
trace of the newest eval cycle through the re-roll driver. Measured 2026-09-07 by the composition
gate: **~30 s over 9 traces, ~100 s over the 60-trace default ``scan_limit``**, paid on the TRAINER,
every cycle, growing with the trace count a long run has most of.

**Why a subprocess and not a thread.** Selection is PURE OVER FILES — that is the property that
makes this move safe, and it was verified rather than assumed:

* ``selection.select_candidates`` builds its own ``ProbeSession(run_dir)`` (which loads a model
  LAZILY and is never asked for one here), reads ``eval_traces/`` off disk, and calls
  ``falsifier.falsify_battle`` on the ``*_reconstruction.json`` siblings. No model, no env, no
  ``self.model``, no logger, nothing from the live process but the run directory.
* ``winprob_oneply.select_winprob_candidates`` is the same shape and is explicitly model-free (it
  reads the ``win_probs`` / ``action_mask`` arrays the trace already carries).

So the only thing crossing the boundary is a ``run_dir`` string in and a list of ``Candidate``
records out — and a subprocess buys what a thread cannot: the re-roll driver's own children, its
own memory, and a crash that lands as a status rather than as a dead trainer. It also keeps the
teacher's shape uniform: every expensive thing the teacher does now happens in a child, and every
child reports through the same per-cycle status histogram.

**The protocol.** ``config`` in (``select_config.json``), ``candidates.json`` out — written
atomically via ``os.replace`` so the parent, which polls the PROCESS, can never read a torn file.
Success is ``{"candidates": [...], "n_candidates": N}``; failure is ``{"error": "<Type>: <msg>",
"traceback": ...}`` plus a non-zero exit. A hard death (OOM, SIGKILL) writes NOTHING, and the
parent treats a missing file as exactly that — see ``SearchTeacherCallback._finish_selection``.

🚨 **THE ENGINE IS DELIBERATELY NOT THREADED HERE.** ``select_candidates`` calls
``falsify_battle`` without an ``impl``, so its re-rolls run on **node** even on a
``--use-bridge rust`` run. That is the behaviour this move preserves BIT FOR BIT (the selected
candidate list is asserted identical before and after), and it is recorded as a separate finding in
``designs/ops/TECH_DEBT_BACKLOG.md`` rather than fixed in the same pass: changing the engine changes
which craters survive the falsify gate, which is a different claim needing a different measurement.
That is also why this worker's config is named ``select_config.json`` and NOT ``config_*.json`` —
the composition gate globs the latter and asserts every one of them says ``"impl": "rust"``, which
is a true statement about the SEARCH workers and would become a false one about this.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from dataclasses import asdict


def run(cfg_path: str) -> int:
    """Select this cycle's candidates and publish them. Returns the process exit code."""
    # Imported through the MODULE, not `from … import select_for_mode`: a module-level name would
    # be bound before any test could reach it, and `callback_test` traps this exact symbol to prove
    # the callback never runs selection in-process (`test_stub_vacuity_gate_test.py`'s shape 1).
    from agents.training.teacher import modes

    with open(cfg_path) as f:
        cfg = json.load(f)
    out_path = cfg["out_path"]
    try:
        cands = modes.select_for_mode(
            cfg.get("mode", "crater"), cfg["run_dir"],
            budget=int(cfg["budget"]), scan_limit=int(cfg["scan_limit"]),
            falsify_gate=bool(cfg["falsify_gate"]), window=int(cfg.get("window", 2)),
            wp_band=float(cfg.get("wp_band", 0.15)))
        payload = {"candidates": [asdict(c) for c in cands], "n_candidates": len(cands)}
    except Exception as e:  # noqa: BLE001 — every failure must reach the parent as a STATUS
        payload = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()}
        print(payload["traceback"], file=sys.stderr, flush=True)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, out_path)
    return 3 if "error" in payload else 0


if __name__ == "__main__":
    raise SystemExit(run(sys.argv[1]))
