"""SPRT PROMOTION on the self-play pool (T6, ``gen3_sprt_promotion_v1``, ``--promotion-sprt``, DEFAULT OFF).

Without it a snapshot is promoted the first eval cycle its ``win_rate_vs_pool`` crosses
``--promote-threshold`` — repeated peeking at a noisy rate, i.e. optional stopping, i.e. a winner's curse.
With it every eval-cycle snapshot is a CANDIDATE once the pool holds at least one snapshot, and its
promotion is decided by its own sequential test (:mod:`agents.training.sprt`): H0 score ≤ 0.50 vs
H1 ≥ 0.55, α = β = 0.05, a pentanomial GSPRT over MIRRORED PAIRS, truncated at a declared cap (the cap
is a REJECT).

THE THREE DISCIPLINE RULES, and how each is made true here rather than intended:

1. **The pool is FROZEN at the test's start.** The test plays the sentinels the CANDIDATE's own cycle
   launched against (captured at launch), every batch. The pool cannot move during it: promotion and
   seeding happen only at an eval collect, and the test runs to its verdict INSIDE the candidate's own
   collect (blocking, in process), so no other cycle can launch, promote or seed in between.
2. **Selection games never pool into the decision.** The cycle's own pool games (``win_rate_vs_pool``)
   are telemetry; the test plays its OWN fresh mirrored pairs on a seed namespace disjoint from every
   cycle's (``sprt_seed``: salted with the batch index), and only those enter the LLR.
3. **A failed test is never re-run.** Every test writes ``start`` and ``verdict`` lines to the run's
   append-only ``sprt_promotion.jsonl``; a candidate step with ANY prior line is never tested again —
   an interrupted test (a restart mid-test) is recorded ``abandoned`` and counts as not promoted.

The test's own score is a SELECTED number (the sample that decided it) and is recorded as provenance
only; a promoted snapshot's strength is read from later, unselected games (``snapshot_ladder``).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import time
from typing import Any, Dict, List, Optional

from agents.training import sprt as S
from agents.training.mirrored_pairs import pooled

LOG_NAME = "sprt_promotion.jsonl"


def sprt_seed(run_seed: int, step: int, batch: int) -> int:
    """The per-GAME seed base of one SPRT batch — disjoint from every eval cycle's (``rust_eval.launch.
    cycle_seed`` hashes a different string), so no selection game is ever replayed as a decision game."""
    d = hashlib.blake2b(f"gen3_sprt_promotion_seed_v1:{int(run_seed)}:{int(step)}:{int(batch)}".encode(),
                        digest_size=8).digest()
    return int.from_bytes(d, "big") & ((1 << 62) - 1)


def split_pairs(n_pairs: int, labels: List[str]) -> Dict[str, int]:
    """``n_pairs`` spread over the sentinels as evenly as possible (round-robin; the first get the rest)."""
    k = len(labels)
    base, rem = divmod(int(n_pairs), k)
    return {lab: base + (1 if i < rem else 0) for i, lab in enumerate(labels)}


def read_log(model_dir: Optional[str]) -> List[dict]:
    if not model_dir:
        return []
    path = os.path.join(model_dir, LOG_NAME)
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    continue
    return out


def append_log(model_dir: Optional[str], row: dict) -> None:
    if not model_dir:
        return
    with open(os.path.join(model_dir, LOG_NAME), "a") as f:
        f.write(json.dumps(row) + "\n")


def already_tested(model_dir: Optional[str], step: int) -> Optional[dict]:
    """The last log line for candidate ``step`` (start, verdict or abandoned), else None."""
    rows = [r for r in read_log(model_dir) if int(r.get("step", -1)) == int(step)]
    return rows[-1] if rows else None


def abandon_unfinished(model_dir: Optional[str], emit=print) -> List[int]:
    """At startup: a test that STARTED and has no verdict was interrupted (a restart mid-test). Record it
    ABANDONED — not promoted, and never re-run (rule 3)."""
    started, decided = {}, set()
    for r in read_log(model_dir):
        st = int(r.get("step", -1))
        if r.get("event") == "start":
            started[st] = r
        elif r.get("event") in ("verdict", "abandoned"):
            decided.add(st)
    out = []
    for st in sorted(set(started) - decided):
        append_log(model_dir, {"event": "abandoned", "step": st, "schema": S.SCHEMA,
                               "reason": "the process ended mid-test", "promoted": False})
        emit(f"⚖️  [SPRT] candidate @{st:,}: its test was interrupted — recorded ABANDONED (not promoted, "
             "never re-run)")
        out.append(st)
    return out


class SprtJob:
    """One candidate's test in flight."""

    def __init__(self, *, step: int, snapshot: str, sentinels: List[dict], cfg: S.SprtConfig, run_seed: int):
        self.step = int(step)
        self.snapshot = snapshot
        self.sentinels = list(sentinels)          # [{"label", "path", "step"}], frozen at test start
        self.state = S.SprtState(cfg)
        self.run_seed = int(run_seed)
        self.batch = 0
        self.t0 = time.monotonic()

    def plan_items(self, n_pairs: int):
        from agents.training.eval_sharding import SENTINEL, EvalItem

        per = split_pairs(n_pairs, [s["label"] for s in self.sentinels])
        return [EvalItem(s["label"], SENTINEL, 2 * per[s["label"]], path=s["path"], step=s["step"])
                for s in self.sentinels if per[s["label"]] > 0]

    def fold(self, merged: dict) -> str:
        """Fold one batch's pentanomials in; returns the verdict (``continue`` / ``accept`` / ``reject``).
        A batch whose games all failed counts nothing and the test is ABANDONED (it cannot be decided on
        games that did not happen, and it must not be retried)."""
        counts = pooled((merged.get("pairs") or {}).get(s["label"]) for s in self.sentinels)
        self.batch += 1
        if counts is None or sum(counts) == 0:
            self.state.verdict, self.state.reason = S.REJECT, "abandoned"
            return self.state.verdict
        return self.state.add_batch(counts)


def record_verdict(cb: Any, job: SprtJob) -> None:
    """The verdict line + the TB scalars, at the CANDIDATE's step."""
    st = job.state
    row = {"event": "verdict", "step": job.step, "promoted": st.verdict == S.ACCEPT,
           "wall_s": round(time.monotonic() - job.t0, 3), "batches": job.batch,
           "pool": [s["step"] for s in job.sentinels], **st.to_json()}
    append_log(cb._model_dir, row)
    logger = getattr(cb, "logger", None)
    if logger is not None:
        logger.record("eval/sprt_llr", float(st.llr))
        logger.record("eval/sprt_pairs", float(st.n_pairs))
        logger.record("eval/sprt_promoted", 1.0 if st.verdict == S.ACCEPT else 0.0)
        logger.dump(job.step)


def cleanup_snapshot(path: Optional[str], run_dir: Optional[str] = None) -> None:
    try:
        if path and os.path.exists(path):
            os.remove(path)
    except OSError:
        pass
    if run_dir:
        shutil.rmtree(run_dir, ignore_errors=True)


class SprtPromotionMixin:
    """``SelfPlayCallback``'s T6 promotion path. Reads the callback's eval plumbing (``_pool``,
    ``_model_dir``, ``_eval_root``, ``model``, ``logger``) and owns ``_sprt_job`` — the one test in
    flight. A test runs to its verdict inside the candidate's own collect (blocking, in process, on the
    Rust eval core; the live weights ARE the candidate's until training resumes)."""

    def _init_sprt(self, enabled: bool) -> None:
        self._sprt_on = bool(enabled)
        self._sprt_cfg = S.SprtConfig()
        self._sprt_job: Optional[SprtJob] = None

    def _sprt_announce(self) -> None:
        if not getattr(self, "_sprt_on", False):
            return
        abandon_unfinished(self._model_dir, emit=print)
        c = self._sprt_cfg.to_json()
        print(f"⚖️  [SPRT] promotion by SPRT (T6): H0 {c['p0']} vs H1 {c['p1']}, alpha=beta={c['alpha']}, "
              f"{c['bounds_mode']} bounds {c['llr_bounds']}, batches of {c['batch_pairs']} mirrored pairs, "
              f"first decision at {c['min_pairs']}, cap {c['max_pairs']} pairs (= reject); --promote-threshold "
              "is not read", flush=True)

    def _sprt_run_seed(self) -> int:
        rc = getattr(self.model, "_rust_collector", None)
        s = getattr(getattr(rc, "cfg", None), "run_seed", None)
        if s is None:
            s = getattr(self.model, "seed", None)
        return int(s) if isinstance(s, int) else 0

    # ------------------------------------------------------------------ the candidate's test
    def _sprt_begin(self, step: int, pending: dict) -> None:
        """Run candidate ``step``'s test to its verdict (promoting the candidate's frozen snapshot on ACCEPT)."""
        sentinels = list(pending.get("sentinels") or [])
        if not sentinels:
            return                       # the pool was empty at launch: nothing to test against
        prior = already_tested(self._model_dir, step)
        if prior is not None:
            print(f"⚖️  [SPRT] candidate @{step:,} already has a test on record ({prior.get('event')}) — "
                  "never re-run (T6 rule 3)", flush=True)
            return
        snap_dir = os.path.join(self._eval_root, f"sprt_{step}")
        os.makedirs(snap_dir, exist_ok=True)
        snap = os.path.join(snap_dir, "candidate.zip")
        shutil.copy2(pending["snapshot"], snap)
        job = SprtJob(step=step, snapshot=snap, sentinels=sentinels, cfg=self._sprt_cfg,
                      run_seed=self._sprt_run_seed())
        append_log(self._model_dir, {"event": "start", "step": step, "schema": S.SCHEMA,
                                     "pool": [s["step"] for s in sentinels], "config": self._sprt_cfg.to_json(),
                                     "selection_games": "the cycle's own pool games are NOT in this test"})
        self._sprt_job = job
        while job.state.verdict == S.CONTINUE:
            merged = self._sprt_play_rust(job)
            if merged is None:
                job.state.verdict, job.state.reason = S.REJECT, "abandoned"
                break
            job.fold(merged)
        self._sprt_finish(job)

    def _sprt_batch_pool(self, job: SprtJob):
        from agents.training.eval_sharding import ShardedEvalPool

        items = job.plan_items(job.state.next_batch_pairs())
        run_dir = os.path.join(self._eval_root, f"sprt_{job.step}", f"b{job.batch}")
        shutil.rmtree(run_dir, ignore_errors=True)
        os.makedirs(os.path.join(run_dir, "claims"), exist_ok=True)
        pool = ShardedEvalPool(items, self._eval_shard_games, step=job.step, mirrored=True)
        pool.write_plan(run_dir)
        return pool, items, run_dir

    def _sprt_play_rust(self, job: SprtJob) -> Optional[dict]:
        from agents.training.eval_collect import merge_eval_results
        from agents.training.rust_eval.executor import EvalCoreError
        from agents.training.rust_eval.launch import run_rust_eval_cycle

        pool, items, run_dir = self._sprt_batch_pool(job)
        try:
            run_rust_eval_cycle(self, pool=pool, run_dir=run_dir, step=job.step,
                                seed=sprt_seed(job.run_seed, job.step, job.batch), forensic=False, record=False)
        except EvalCoreError as e:
            print(f"⚠️ [SPRT] candidate @{job.step:,}: batch {job.batch} failed on the eval core — {e}; the test "
                  "is ABANDONED (not promoted, never re-run)", flush=True)
            return None
        merged, _missing = merge_eval_results(run_dir, [it.key for it in items])
        shutil.rmtree(run_dir, ignore_errors=True)
        return merged

    def _sprt_finish(self, job: SprtJob) -> None:
        """Record the verdict; on ACCEPT promote the candidate's FROZEN snapshot. This runs inside the
        candidate's own collect, whose own pushes + summary write follow."""
        st = job.state
        record_verdict(self, job)
        promoted = st.verdict == S.ACCEPT
        print(f"⚖️  [SPRT] candidate @{job.step:,}: {st.verdict.upper()} ({st.reason}) after {st.n_pairs} pairs "
              f"in {job.batch} batch(es), LLR {st.llr:+.3f}" + (" — PROMOTED" if promoted else ""), flush=True)
        if promoted:
            self._pool.add_from_path(job.snapshot, job.step)
            self._pool_generation += 1
            self.logger.record("train/selfplay_promoted_steps", float(job.step))
            self._spawn_snapshot_ladder_update(job.step)
        self._sprt_job = None
        cleanup_snapshot(None, os.path.join(self._eval_root, f"sprt_{job.step}"))
