"""DistilledOpponentManager — one idempotent reconcile loop.

Backfill and steady-state are the SAME operation: every tick, ``reconcile(active_steps)`` tries to
make the on-disk gate-passed distilled set equal the pool's currently-sampleable snapshots. There is
**no "are we behind?" mode** — if everything already matches, reconcile is a no-op; if some snapshots
are missing a distilled variant (just enabled = backfill; a new promotion = one missing), it spawns
the jobs; if a snapshot was evicted, it cleans up. All state that matters is the distilled ``.pt``
files on disk + a small in-memory job/ladder table, so it is restart-safe (a fresh manager rebuilds
from disk and just keeps reconciling).

The all-or-nothing rule (the per-step barrier — `distill_integration.md` §8): the env uses distilled
opponents iff EVERY sampled opponent is distilled. So ``reconcile`` returns ``all_distilled`` — true
only when every *deployable* snapshot has a gate-passed variant — and the env atomically uses
distilled (when true) or full (when false) for the whole pool, never mixed.

Pure logic via injected hooks (so it is unit-testable without subprocesses/bridge):
  ready_steps_fn()      -> set[int]   steps whose distilled.pt is on disk AND gate-passed
  list_artifacts_fn()   -> set[int]   steps with ANY distilled artifact (for eviction cleanup)
  run_distill_fn(step, size_cfg) -> handle   spawn the (async, GPU) distill+gate job for a snapshot
  poll_fn(handle)       -> None | dict   None = running; dict = {passed: bool, speedup: float, ...}
  remove_fn(step)       -> None        delete a snapshot's distilled artifact
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

# default capacity ladder (smallest first); the manager escalates a snapshot up this ladder
# when its gate fails, trading speedup for fidelity (`distill_integration.md` §8).
DEFAULT_LADDER: List[Dict[str, int]] = [
    {"enc_hidden": 192, "head_hidden": 384, "tf_heads": 8, "tf_ffn": 512},   # ~the validated 4.67x
    {"enc_hidden": 288, "head_hidden": 512, "tf_heads": 8, "tf_ffn": 768},
    {"enc_hidden": 384, "head_hidden": 640, "tf_heads": 8, "tf_ffn": 1024},
]


@dataclass
class ReconcileResult:
    all_distilled: bool        # the atomic flag the env reads: every sampled opponent is distilled
    use_full: bool             # pool-wide fallback (too few deployable opponents) -> full models
    frac_distilled: float      # |ready ∩ deployable| / |deployable|  (distill/frac_active_opponents_distilled)
    n_active: int
    n_ready: int
    n_running: int
    n_missing: int
    n_exhausted: int           # active snapshots that can't be distilled within the ladder/floor
    spawned: List[int]         # steps a distill job was spawned for this tick
    sampleable: Set[int]       # the opponent steps the env should sample from this generation
    # Jobs that FINISHED this tick (for the launcher Events panel). One dict per harvested job:
    #   {step, passed, action: "deployed"|"escalated"|"exhausted", rung, n_rungs,
    #    next_rung?, speedup, h2h, top1}. Empty on a no-op tick — pure data, the callback formats it.
    harvested: List[Dict[str, Any]] = field(default_factory=list)


class DistilledOpponentManager:
    def __init__(
        self,
        *,
        ready_steps_fn: Callable[[], Set[int]],
        list_artifacts_fn: Callable[[], Set[int]],
        run_distill_fn: Callable[[int, Dict[str, int]], Any],
        poll_fn: Callable[[Any], Optional[Dict[str, Any]]],
        remove_fn: Callable[[int], None],
        recover_fn: Optional[Callable[[], Dict[int, Dict[str, Any]]]] = None,
        ladder: Optional[List[Dict[str, int]]] = None,
        min_speedup: float = 2.0,
        max_concurrent: int = 4,
        min_pool: int = 2,
    ):
        self._ready_steps = ready_steps_fn
        self._list_artifacts = list_artifacts_fn
        self._run = run_distill_fn
        self._poll = poll_fn
        self._remove = remove_fn
        self.ladder = ladder or DEFAULT_LADDER
        self.min_speedup = min_speedup
        self.max_concurrent = max_concurrent
        self.min_pool = min_pool
        self._jobs: Dict[int, Any] = {}          # step -> handle (in-flight)
        self._job_rung: Dict[int, int] = {}       # step -> ladder index of the in-flight job
        self._next_rung: Dict[int, int] = {}      # step -> ladder index to try next (after a fail)
        self._exhausted: Set[int] = set()         # ladder fully tried OR speedup below floor
        if recover_fn is not None:                # restart-safe: rebuild escalation state from disk
            self._recover(recover_fn())

    def _rung_of(self, config: Dict[str, Any]):
        for i, c in enumerate(self.ladder):
            if all(config.get(k) == v for k, v in c.items()):
                return i
        return None

    def _recover(self, failed: Dict[int, Dict[str, Any]]) -> None:
        """Reconstruct ladder progress from failed manifests so a restarted manager doesn't
        re-distill a known-unfit snapshot from rung 0: a fail at rung k retries at k+1, or is
        exhausted if k was the last rung."""
        for step, m in failed.items():
            rung = self._rung_of(m.get("config", {}))
            if rung is None:
                continue
            if rung + 1 < len(self.ladder):
                self._next_rung[step] = rung + 1
            else:
                self._exhausted.add(step)

    # ── the whole feature: one idempotent step ──
    def reconcile(self, active_steps) -> ReconcileResult:
        active: Set[int] = set(active_steps)

        # 1) harvest finished jobs -> ready (on disk) or escalate/exhaust
        harvested: List[Dict[str, Any]] = []
        for step, handle in list(self._jobs.items()):
            res = self._poll(handle)
            if res is None:
                continue                          # still running
            rung = self._job_rung.pop(step, 0)
            del self._jobs[step]
            passed = bool(res.get("passed")) and res.get("speedup", 0.0) >= self.min_speedup
            rec = {"step": step, "passed": passed, "rung": rung, "n_rungs": len(self.ladder),
                   "speedup": res.get("speedup"), "h2h": res.get("h2h"), "top1": res.get("top1")}
            if passed:
                self._next_rung.pop(step, None)   # success: its gate-passed .pt is now on disk
                rec["action"] = "deployed"
            else:
                nxt = rung + 1
                if nxt < len(self.ladder):
                    self._next_rung[step] = nxt   # retry bigger next reconcile
                    rec["action"], rec["next_rung"] = "escalated", nxt
                else:
                    self._exhausted.add(step)     # ladder exhausted -> can't distill faithfully
                    rec["action"] = "exhausted"
            harvested.append(rec)

        ready = self._ready_steps() & active
        running = set(self._jobs)

        # 2) spawn for whatever is still missing (idempotent: skip ready/running/exhausted),
        #    bounded by max_concurrent. "missing" = active that isn't ready/running/exhausted.
        missing = active - ready - running - self._exhausted
        spawned: List[int] = []
        for step in sorted(missing):
            if len(self._jobs) >= self.max_concurrent:
                break
            idx = self._next_rung.get(step, 0)
            self._jobs[step] = self._run(step, self.ladder[idx])
            self._job_rung[step] = idx
            spawned.append(step)

        # 3) clean up artifacts/state for evicted snapshots (window slid past them)
        for step in (self._list_artifacts() | self._exhausted) - active:
            if step in self._list_artifacts():
                self._remove(step)
            self._exhausted.discard(step)
            self._next_rung.pop(step, None)

        # 4) flags. Exhausted-but-active snapshots are NOT sampled (they'd be a slow straggler),
        #    so the deployable opponent set drops them; all_distilled requires the deployable set
        #    to be fully ready. If that leaves too few opponents for a valid pool, fall back to
        #    FULL models pool-wide (the exhausted ones are still needed for diversity).
        deployable = active - self._exhausted
        all_distilled = bool(deployable) and deployable.issubset(ready) and len(deployable) >= self.min_pool
        use_full = not all_distilled
        sampleable = deployable if all_distilled else set(active)  # full pool samples everything
        frac = (len(ready & deployable) / len(deployable)) if deployable else 0.0
        return ReconcileResult(
            all_distilled=all_distilled, use_full=use_full, frac_distilled=frac,
            n_active=len(active), n_ready=len(ready), n_running=len(self._jobs),
            n_missing=len(missing), n_exhausted=len(self._exhausted & active),
            spawned=spawned, sampleable=sampleable, harvested=harvested,
        )
