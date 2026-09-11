"""The PARENT-side driver for `--win-prob-rollout-target`: snapshot, fan out, collect, apply.

Called once per rollout from `WinProbLabelCallback._on_rollout_end`, BETWEEN the terminal back-fill
and ``train()``. It blocks — see :mod:`agents.training.win_prob_rollout` for why a late label has
no row left to land on — so everything here is about making the block SHORT, BOUNDED and VISIBLE:

* **One weights snapshot per rollout.** ``model.save`` into a scratch dir under ``TMPDIR`` (never
  the run directory: this is transient and a run's directory is an artifact). Every worker loads
  the same file, so all R x K continuations are measured under ONE policy — the one that collected
  the buffer.
* **A fixed worker fan-out**, :data:`DEFAULT_WORKERS` derived from the core count with a hard cap.
  The box normally carries a live trainer, and the workers here run while the trainer's own env
  pool is IDLE (collection is over, ``train()`` has not started), so they are filling a real hole
  rather than competing — but only up to the point where they start competing with the OTHER run on
  the box, which is what the cap is for.
* **A wall-clock BOUND, contention-scaled.** A worker that overruns is KILLED and its states keep
  their terminal bit. A timeout here is not a semantic outcome and is never scored as one: the
  states come back unlabelled and ``win_prob/rollout_failed`` counts them.
* **Nothing raises into the training loop.** Every failure path returns "no label for that state",
  which the caller turns into "this row keeps the target it already had". The one thing this
  subsystem must never do is put a fabricated number into the value objective.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from utils.contention import scale_timeout

#: Fan-out. A quarter of the cores, at most 8: the trainer's own env workers are idle at this point
#: in the iteration, but a production box also carries a second run and the arm must not turn a
#: label budget into a starvation event. 1 on a small box, which is correct and slow.
DEFAULT_WORKERS = max(1, min(8, (os.cpu_count() or 4) // 4))

#: Per-worker wall bound, before contention scaling: a fixed model-load allowance plus a per-arm
#: allowance. 45 s covers a cold load + compile; 30 s per continuation is ~10x the measured
#: 1.8-3.4 s and exists to catch a WEDGE, not to bound a slow box (``scale_timeout`` does that).
LOAD_BUDGET_S = 45.0
PER_ARM_BUDGET_S = 30.0


def _worker_argv(job_path: str, out_path: str) -> List[str]:
    return [sys.executable, "-m", "agents.training.win_prob_rollout_worker",
            "--job", job_path, "--out", out_path]


def _nice_child() -> None:                                    # pragma: no cover - trivial
    try:
        os.nice(10)
    except OSError:
        pass


def label_states(*, model, states: Sequence[Dict[str, Any]], rollouts: int, impl: str,
                 workers: Optional[int] = None, log=print
                 ) -> Tuple[List[Optional[float]], Dict[str, float]]:
    """Play ``rollouts`` continuations for every entry of ``states``; return ``(labels, stats)``.

    ``states[i]`` is ``{"id": i, "record": <path>, "turn": int, "salt": str}``. ``labels[i]`` is
    ``wins / n`` in [0, 1], or **None** when that state produced no finished continuation — a state
    with no label keeps whatever target it already had, which is the only safe failure.

    ``stats`` carries the cost meter: ``seconds`` (the stall this added to the training loop),
    ``arms``, ``arms_capped``, ``arm_decisions`` (the MEASURED mean live decisions per continuation,
    which is what prices the fraction), ``load_seconds`` and ``workers``.
    """
    n = len(states)
    labels: List[Optional[float]] = [None] * n
    stats: Dict[str, float] = {"seconds": 0.0, "arms": 0.0, "arms_capped": 0.0,
                               "arm_decisions": 0.0, "load_seconds": 0.0, "workers": 0.0,
                               "worker_failures": 0.0}
    if n == 0:
        return labels, stats

    t0 = time.perf_counter()
    w = int(workers or DEFAULT_WORKERS)
    w = max(1, min(w, n))
    scratch = tempfile.mkdtemp(prefix="wp_rollout_")
    procs: List[Tuple[subprocess.Popen, str, List[int]]] = []
    decisions_total = 0.0
    try:
        weights = os.path.join(scratch, "policy.zip")
        model.save(weights)
        # Round-robin rather than contiguous: the assignment must not correlate with the buffer
        # order, so one worker cannot end up holding every long battle.
        chunks: List[List[int]] = [[] for _ in range(w)]
        for i in range(n):
            chunks[i % w].append(i)
        for k, idxs in enumerate(chunks):
            if not idxs:
                continue
            job = {"weights": weights, "rollouts": int(rollouts), "impl": str(impl),
                   "states": [dict(states[i], id=int(i)) for i in idxs]}
            jp = os.path.join(scratch, f"job_{k}.json")
            op = os.path.join(scratch, f"out_{k}.json")
            with open(jp, "w") as f:
                json.dump(job, f)
            procs.append((subprocess.Popen(_worker_argv(jp, op), stdout=subprocess.DEVNULL,
                                           stderr=subprocess.PIPE, preexec_fn=_nice_child), op,
                          idxs))
        stats["workers"] = float(len(procs))
        deadline = time.perf_counter() + scale_timeout(
            LOAD_BUDGET_S + PER_ARM_BUDGET_S * max(1, (n * int(rollouts)) // max(1, len(procs))))
        for proc, op, idxs in procs:
            remaining = max(1.0, deadline - time.perf_counter())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                # A TIMEOUT IS NEVER A SEMANTIC OUTCOME. Kill by this child's OWN pid, take no
                # labels from it, and count it.
                stats["worker_failures"] += 1.0
                log(f"⚠️  [win_prob_rollout] worker pid {proc.pid} exceeded its wall bound "
                    f"({remaining:.0f}s left of the batch budget) — killed; its "
                    f"{len(idxs)} states keep their terminal bit")
                try:
                    proc.kill()
                    proc.wait(timeout=10)
                except Exception:                                   # noqa: BLE001 pragma: no cover
                    pass
                continue
            if not os.path.exists(op):
                stats["worker_failures"] += 1.0
                err = (proc.stderr.read() or b"").decode("utf-8", "replace")[-400:] \
                    if proc.stderr else ""
                log(f"⚠️  [win_prob_rollout] worker pid {proc.pid} wrote no result "
                    f"(rc={proc.returncode}): {err.strip()[:400]}")
                continue
            try:
                with open(op) as f:
                    payload = json.load(f)
            except Exception as exc:                                # noqa: BLE001
                stats["worker_failures"] += 1.0
                log(f"⚠️  [win_prob_rollout] unreadable worker result: {exc}")
                continue
            if payload.get("fatal"):
                stats["worker_failures"] += 1.0
                log(f"⚠️  [win_prob_rollout] worker FATAL: {payload['fatal']}")
            stats["load_seconds"] = max(stats["load_seconds"],
                                        float(payload.get("load_seconds") or 0.0))
            for row in payload.get("results") or ():
                i = int(row.get("id", -1))
                cnt = int(row.get("n") or 0)
                if not (0 <= i < n):
                    continue
                stats["arms"] += float(cnt)
                stats["arms_capped"] += float(row.get("capped") or 0)
                decisions_total += float(row.get("decisions") or 0.0)
                if cnt > 0:
                    labels[i] = float(row.get("wins") or 0.0) / float(cnt)
                elif row.get("error"):
                    log(f"⚠️  [win_prob_rollout] state {i} produced no continuation: "
                        f"{row['error']}")
    except Exception as exc:                                        # noqa: BLE001
        # The training loop must survive anything this path can do. Report, label nothing.
        log(f"⚠️  [win_prob_rollout] labelling batch failed ({type(exc).__name__}: "
            f"{str(exc)[:200]}) — every sampled state keeps its terminal bit")
    finally:
        for proc, _op, _idxs in procs:
            if proc.poll() is None:                                 # pragma: no cover - defensive
                try:
                    proc.kill()
                except Exception:                                   # noqa: BLE001
                    pass
        shutil.rmtree(scratch, ignore_errors=True)
    if stats["arms"] > 0:
        stats["arm_decisions"] = decisions_total / stats["arms"]
    stats["seconds"] = time.perf_counter() - t0
    return labels, stats
