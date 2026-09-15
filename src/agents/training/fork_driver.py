"""The PARENT-side driver for `--fork-fraction`: snapshot, fan out, collect, score, assemble.

Called once per rollout from `ForkArmCallback._on_rollout_end`, BETWEEN `compute_returns_and_-
advantage` and ``train()``. It BLOCKS — a branch row must land in the buffer it belongs to, and the
rollout buffer is a RING refilled from scratch every iteration, so a row that arrives one rollout
late has no buffer left to enter — and everything here is about making the block SHORT, BOUNDED and
VISIBLE.

The shape is `win_prob_rollout_labeller`'s, and deliberately so; the two differences are the two
places the arms differ:

* **the children return TRANSITIONS, not a scalar.** Rows ride an ``.npz`` beside the result JSON
  rather than through the JSON, because 2,501 float32 per row through a text encoding is minutes of
  wall clock and gigabytes of temp file.
* **values and log-probs are computed HERE**, by the live policy, in one batched forward. Not an
  optimisation: the parent's weights ARE the weights that collected the buffer, so ``old_log_prob``
  is exactly the behaviour policy's and the PPO ratio is 1.0 at the first epoch, which is what the
  clipped objective assumes. A log-prob from a child's re-loaded snapshot would be that number only
  up to serialisation, and an importance ratio does not survive "only up to".

Nothing here raises into the training loop. Every failure path returns "this fork produced no
rows", which the caller turns into "the buffer is exactly what collection made it". A run must be
able to survive its forks failing; the one thing this subsystem must never do is put a fabricated
transition into the objective.
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

import numpy as np

from utils.contention import scale_timeout

#: Fan-out. A quarter of the cores, at most 8 — `win_prob_rollout_labeller.DEFAULT_WORKERS`'
#: reasoning verbatim: the trainer's own env workers are idle at this point in the iteration, but a
#: production box also carries a second run and the arm must not turn a fork budget into a
#: starvation event.
DEFAULT_WORKERS = max(1, min(8, (os.cpu_count() or 4) // 4))

#: Per-worker wall bound before contention scaling: a fixed model-load allowance plus a per-branch
#: allowance. 45 s covers a cold load + compile; 30 s per branch is ~10x the measured 1.8-3.4 s of
#: a continuation and exists to catch a WEDGE, not to bound a slow box (`scale_timeout` does that).
LOAD_BUDGET_S = 45.0
PER_BRANCH_BUDGET_S = 30.0


def _worker_argv(job_path: str, out_path: str) -> List[str]:
    return [sys.executable, "-m", "agents.training.fork_worker",
            "--job", job_path, "--out", out_path]


def _nice_child() -> None:                                    # pragma: no cover - trivial
    try:
        os.nice(10)
    except OSError:
        pass


def _score(model, obs: np.ndarray, masks: np.ndarray, actions: np.ndarray
           ) -> Tuple[np.ndarray, np.ndarray]:
    """``(values, log_probs)`` from the LIVE policy over the branch rows, in chunks.

    ``policy.evaluate_actions`` is the one entry point that returns both against the SAME forward,
    and under ``--critic winprob`` the value it returns is the promoted head — ``sigmoid(logit)`` in
    [0, 1] — which is the quantity the branch's GAE must be built from. Chunked because the row
    count is unbounded in the fork fraction while the trainer's own minibatch is not.
    """
    import torch as th

    n = int(obs.shape[0])
    vals = np.zeros(n, dtype=np.float32)
    lps = np.zeros(n, dtype=np.float32)
    chunk = 512
    dev = model.device
    was_training = model.policy.training
    model.policy.set_training_mode(False)
    try:
        with th.no_grad():
            for lo in range(0, n, chunk):
                hi = min(n, lo + chunk)
                d = {"observation": th.as_tensor(obs[lo:hi]).to(dev),
                     "action_mask": th.as_tensor(masks[lo:hi].astype(np.float32)).to(dev)}
                v, lp, _ent = model.policy.evaluate_actions(
                    d, th.as_tensor(actions[lo:hi]).to(dev),
                    action_masks=masks[lo:hi].astype(np.float32))
                vals[lo:hi] = v.reshape(-1).float().cpu().numpy()
                lps[lo:hi] = lp.reshape(-1).float().cpu().numpy()
    finally:
        model.policy.set_training_mode(was_training)
    return vals, lps


def play_forks(*, model, forks: Sequence[Dict[str, Any]], impl: str, crn: str,
               workers: Optional[int] = None, log=print
               ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """Play every fork's branches; return ``(results, stats)``.

    ``forks[i]`` is ``{"id": i, "record": <path>, "turn": int, "actions": {name: idx},
    "salt": str}``. ``results[i]`` mirrors it with each branch's ``outcome``/``capped``/``turns``/
    ``decisions`` plus the captured rows attached as ``obs``/``mask``/``action`` arrays. A fork that
    produced nothing comes back with an empty ``branches`` dict and is COUNTED, never guessed at.
    """
    n = len(forks)
    stats: Dict[str, float] = {"seconds": 0.0, "branches": 0.0, "branches_capped": 0.0,
                               "rows": 0.0, "load_seconds": 0.0, "workers": 0.0,
                               "worker_failures": 0.0}
    out: List[Dict[str, Any]] = [{"id": int(f["id"]), "branches": {}} for f in forks]
    if n == 0:
        return out, stats

    t0 = time.perf_counter()
    w = max(1, min(int(workers or DEFAULT_WORKERS), n))
    scratch = tempfile.mkdtemp(prefix="fork_arm_")
    procs: List[Tuple[subprocess.Popen, str, List[int]]] = []
    n_branches = sum(len(f.get("actions") or {}) for f in forks)
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
            job = {"weights": weights, "impl": str(impl), "crn": str(crn),
                   "forks": [dict(forks[i], id=int(i)) for i in idxs]}
            jp = os.path.join(scratch, f"job_{k}.json")
            op = os.path.join(scratch, f"out_{k}.json")
            with open(jp, "w") as f:
                json.dump(job, f)
            procs.append((subprocess.Popen(_worker_argv(jp, op), stdout=subprocess.DEVNULL,
                                           stderr=subprocess.PIPE, preexec_fn=_nice_child), op,
                          idxs))
        stats["workers"] = float(len(procs))
        deadline = time.perf_counter() + scale_timeout(
            LOAD_BUDGET_S + PER_BRANCH_BUDGET_S * max(1, n_branches // max(1, len(procs))))
        for proc, op, idxs in procs:
            remaining = max(1.0, deadline - time.perf_counter())
            try:
                proc.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                # A TIMEOUT IS NEVER A SEMANTIC OUTCOME. Kill by this child's OWN pid, take no rows
                # from it, and count it.
                stats["worker_failures"] += 1.0
                log(f"⚠️  [fork_arm] worker pid {proc.pid} exceeded its wall bound "
                    f"({remaining:.0f}s left of the batch budget) — killed; its {len(idxs)} "
                    f"fork(s) contribute no rows")
                try:
                    proc.kill()
                    proc.wait(timeout=10)
                except Exception:                                     # noqa: BLE001 pragma: no cover
                    pass
                continue
            if not os.path.exists(op):
                stats["worker_failures"] += 1.0
                err = (proc.stderr.read() or b"").decode("utf-8", "replace")[-400:] \
                    if proc.stderr else ""
                log(f"⚠️  [fork_arm] worker pid {proc.pid} wrote no result "
                    f"(rc={proc.returncode}): {err.strip()[:400]}")
                continue
            try:
                with open(op) as f:
                    payload = json.load(f)
            except Exception as exc:                                  # noqa: BLE001
                stats["worker_failures"] += 1.0
                log(f"⚠️  [fork_arm] unreadable worker result: {exc}")
                continue
            if payload.get("fatal"):
                stats["worker_failures"] += 1.0
                log(f"⚠️  [fork_arm] worker FATAL: {payload['fatal']}")
            stats["load_seconds"] = max(stats["load_seconds"],
                                        float(payload.get("load_seconds") or 0.0))
            npz = payload.get("npz")
            arrays = None
            if npz and os.path.exists(npz):
                with np.load(npz) as z:
                    arrays = {k: z[k].copy() for k in ("obs", "mask", "action")}
            for row in payload.get("results") or ():
                i = int(row.get("id", -1))
                if not (0 <= i < n):
                    continue
                for name, br in (row.get("branches") or {}).items():
                    stats["branches"] += 1.0
                    stats["branches_capped"] += float(bool(br.get("capped")))
                    lo, cnt = int(br.get("row_start", -1)), int(br.get("n_rows") or 0)
                    if arrays is not None and lo >= 0 and cnt > 0:
                        br["obs"] = arrays["obs"][lo:lo + cnt]
                        br["mask"] = arrays["mask"][lo:lo + cnt]
                        br["action"] = arrays["action"][lo:lo + cnt]
                        stats["rows"] += float(cnt)
                    out[i]["branches"][str(name)] = br
                if row.get("error"):
                    log(f"⚠️  [fork_arm] fork {i} failed: {row['error']}")
    except Exception as exc:                                          # noqa: BLE001
        log(f"⚠️  [fork_arm] fork batch failed ({type(exc).__name__}: {str(exc)[:200]}) — this "
            f"rollout injects nothing")
    finally:
        for proc, _op, _idxs in procs:
            if proc.poll() is None:                                   # pragma: no cover - defensive
                try:
                    proc.kill()
                except Exception:                                     # noqa: BLE001
                    pass
        shutil.rmtree(scratch, ignore_errors=True)
    stats["seconds"] = time.perf_counter() - t0
    return out, stats


def assemble(*, model, results: Sequence[Dict[str, Any]], obs_keys: Sequence[str],
             mask_dims: int, row_budget: int, log=print
             ) -> Tuple[Optional[Dict[str, Any]], int]:
    """Score every branch's rows with the LIVE policy and build the injection block.

    Returns ``(block, dropped_forks)``. Forks are taken in order until ``row_budget`` is reached and
    then DROPPED WHOLE — never partially. A half-injected fork would put one branch of a sibling
    pair in the objective and not the other, which is the one asymmetry the arm exists to avoid.

    The branches' ``succ_value`` (the value of the SUCCESSOR state, row 1) is written back onto
    ``results`` here, because it falls out of the same forward the GAE needs and it is what
    ``fork/pairwise_acc`` is computed from.
    """
    from agents.training.fork_buffer import build_branch_rows, concat_blocks

    usable: List[Tuple[Dict[str, Any], str, Dict[str, Any]]] = []
    for fk in results:
        for name, br in (fk.get("branches") or {}).items():
            if br.get("obs") is None or br.get("outcome") is None or br.get("capped"):
                continue
            usable.append((fk, str(name), br))
    if not usable:
        return None, 0

    obs = np.concatenate([b["obs"] for _f, _n, b in usable], axis=0)
    masks = np.concatenate([b["mask"] for _f, _n, b in usable], axis=0)
    actions = np.concatenate([b["action"] for _f, _n, b in usable], axis=0)
    try:
        values, log_probs = _score(model, obs, masks, actions)
    except Exception as exc:                                          # noqa: BLE001
        log(f"⚠️  [fork_arm] scoring the branch rows failed ({type(exc).__name__}: "
            f"{str(exc)[:200]}) — this rollout injects nothing")
        return None, len(results)

    lo = 0
    for _f, _n, br in usable:
        k = int(br["obs"].shape[0])
        br["values"] = values[lo:lo + k]
        br["log_probs"] = log_probs[lo:lo + k]
        # The SUCCESSOR's value — row 1, the first state after the fork action resolved. A branch
        # that ended AT the fork turn has no successor and reports None rather than reusing row 0,
        # which is the FORK state and identical across the branches by construction.
        br["succ_value"] = float(values[lo + 1]) if k >= 2 else None
        lo += k

    blocks, used, dropped = [], 0, 0
    for fk in results:
        mine = [(n, b) for n, b in (fk.get("branches") or {}).items() if b.get("values") is not None]
        if not mine:
            continue
        need = sum(int(b["obs"].shape[0]) for _n, b in mine)
        if used + need > int(row_budget):
            dropped += 1
            for _n, b in mine:                     # so the meters never score a fork we dropped
                b["values"] = None
                b["succ_value"] = None
            continue
        for _n, b in mine:
            blocks.append(build_branch_rows(
                obs=b["obs"], masks=b["mask"], actions=b["action"], values=b["values"],
                log_probs=b["log_probs"], outcome=float(b["outcome"]),
                gamma=float(model.gamma), gae_lambda=float(model.gae_lambda),
                obs_keys=obs_keys, mask_dims=int(mask_dims)))
        used += need
    if dropped:
        log(f"⚠️  [fork_arm] {dropped} fork(s) dropped WHOLE at the row budget "
            f"({row_budget} rows). Lower --fork-fraction, or raise the budget only if the "
            f"buffer's memory footprint can carry it — a branch row is a full observation.")
    return concat_blocks(blocks), dropped
