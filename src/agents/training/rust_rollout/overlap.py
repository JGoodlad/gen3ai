"""X14 — OVERLAP the next rollout's COLLECTION with the PPO UPDATE (``--overlap-collect``, default OFF;
`gen3_overlap_collect_v1`; ``designs/training/rust_collector.md`` "Overlap (X14)").

WHAT IT DOES. With the flag ON, from the process's SECOND update on, the collector plays the next rollout on a
WORKER THREAD while the main thread runs ``train()`` on the current one. T2 still holds the weights it was loaded
with at the last ``after_update`` (version k), so every row the worker plays is one policy version stale when it is
trained on (the learner is then at k + 1). The worker plays until the trigger is READY and stops — it never plays
beyond the target and never reloads T2 mid-rollout — so the run stays DETERMINISTIC: which version played a row
never depends on wall-clock timing.

WHAT THE WORKER TOUCHES. Only the collector's side: the core, T2's trainee and opponent slots, the opponent
server, the team stager, the row arena, the collector's counters. NEVER the learner, the logger, the callbacks or
``model.num_timesteps``: the per-host-step decision counts and the infos are LOGGED (`Played`), and the main thread
REPLAYS them after the join (`RustCollector.collect_after_overlap`) — the same ``on_step`` events, at the same
``num_timesteps`` values, with the same infos, as a serial collection would have fired. A callback therefore runs
with the learner at k + 1 exactly as in a serial collection; what moves is only that its side effects (a pool
push, an opponent refresh) reach the env one rollout later.

THE THREE CORRECTNESS PIECES.

1. **K9(b) on stale rows — the BEHAVIOUR SNAPSHOT** (`BehaviourSnapshots`). The learner's K9(b) judges the rows
   played at the CURRENT version; under overlap there are (almost) none. So every ``after_update`` also copies the
   weights it loads into T2 into one of two version-keyed snapshot slots, and ``train()`` judges the rows played at
   version k against the version-k snapshot, swapped into the policy for that one probe forward and swapped back
   bit-exactly (`judge_snapshot`; tags ``behaviour_snapshot/*``). Same gate, same bar, same tie rule.
2. **The objective — DECOUPLED PPO** (`proximal_transform`; Hilton, Cobbe & Schulman 2021, "Batch size-invariance
   for policy optimization"). PPO's ratio π/μ against the stored behaviour log-prob μ keeps the gradient unbiased
   for any behaviour policy, but it also CENTRES THE CLIP AND THE KL ON μ. X26 measured one-version-old rows at the
   START of an update at approx-KL 0.016 vs μ and 22.5 % outside the 0.15 clip band (``staleness/probe_age_1_*``,
   ~380M steps): every row stale would make the KL→LR controller (target 0.01) cut the rate for a lag it cannot
   remove, and anchor the clip to the stale policy. So the PROXIMAL log-prob log π_prox(a|s) — the learner's own
   weights at the update's start — is computed for every row at the FILL, by T2's trainee slot, which holds exactly
   those weights then (``after_update`` loaded them; the worker is idle) and is the very function μ was served by
   (`OverlapRunner.proximal_logp`; the learner's eager fp32 forward costs ≈ 0.34 s per 2,048 rows on X26 —
   ``behaviour/probe_forward_ms`` — i.e. ≈ 16 s a 98,304-row update; T2's captured graphs serve it for a fraction).
   Before the first optimizer step the buffer is rewritten: ``old_log_prob := log π_prox`` (the clip and
   ``train/approx_kl`` are now centred on π_prox, as on a fresh row), and the truncated importance weight
   ``w = min(π_prox / μ, IS_WEIGHT_CAP)`` is stored BESIDE the raw advantage (``RolloutBuffer.is_weights``), never
   folded into it. Each micro-batch's advantage is then NORMALISED FIRST AND WEIGHTED AFTER
   (`decoupled_advantages`: ``Â = norm(A)`` exactly as the micro-step's ``normalize_advantage`` would, then
   ``w · Â``), and the micro-step runs with its own normalisation off (`TrainSetup._micro_static`); the micro-step's
   code is unchanged. Because w > 0, ``-min(w·Â·r, w·Â·clip(r)) = -w·min(Â·r, Â·clip(r))``: the decoupled surrogate,
   exactly (Hilton et al.'s order). The order matters (`gen3_overlap_norm_then_weight_v1`, core review 2026-10-20
   A2): normalising w·A instead leaves, for an advantage that carries no information about the action, an expected
   gradient (E[A]/s)·Σₐ(π_prox(a) − μ(a))·∇log π(a) ≠ 0 — a steady push along μ → π_prox. Everything that reads
   "the advantage" as a quantity — the ``signal/*`` density tags, the ride-along ``adv`` head's target — reads the
   RAW buffer advantage. On a fresh row π_prox = μ to fp32 rounding (both are T2's forward of the same weights; only
   the bucket padding can differ), w = 1, and the loss is plain PPO (bit for bit against the eager micro-step).
3. **CUDA streams.** T2's ``execute`` orders its copies on the CALLING thread's current stream. On the main
   thread that is the legacy default stream, which the learner fills during ``train()``; a worker on it would
   queue every flush behind the update. The worker therefore runs under its OWN stream, acquired at startup, with
   explicit joins at ``begin`` (the worker waits for the main stream) and at ``join`` (the main stream waits for
   the worker's). UNVERIFIED on the card (no lease): the GPU A/B is the check (FINDING).

WHAT IS REFUSED (``validate_decl``): the fork arm (its pass runs between the trigger and the fill on the main
thread's model), and on CUDA any T2 backend but ``graph`` (an eager / AOT forward from a second thread opens a
second cuBLAS handle and workspace — an allocation T2's frozen guard would count after the freeze).

MEMORY. No second row buffer: the rollout buffer being trained and the row arena being filled both exist in a
serial run already (the arena is sized target + 2 games per env either way). The overlap adds the two snapshot
slots and one swap scratch, each one copy of the policy's state dict (``BehaviourSnapshots.nbytes``), plus the
fill's per-row proximal log-probs on the host.
"""
from __future__ import annotations

import contextlib
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from agents.training.lifecycle_decl import startup_builder

if TYPE_CHECKING:
    import torch

#: The decoupled objective's importance-weight TRUNCATION (w = min(π_prox / μ, cap)). At one version of lag X26's
#: age-1 rows read mean |π/μ − 1| 0.106 at an update's start, so the cap binds on a small tail only; it bounds the
#: variance a near-zero μ would put on one row. A declared constant, UNMEASURED as a lever (``overlap/is_weight_*``
#: report how often it binds).
IS_WEIGHT_CAP = 2.0

#: The tag families this module writes (one producer each).
SNAPSHOT_PREFIX = "behaviour_snapshot/"


class OverlapError(RuntimeError):
    """A misuse of the overlap runner (a begin while a rollout is in flight, a join with none)."""


def validate_decl(*, overlap: bool, fork: Any, device: str, backend: str) -> None:
    """Refuse a declaration the overlap cannot serve (module docs, "WHAT IS REFUSED")."""
    if not overlap:
        return
    if fork is not None:
        raise ValueError("RustEnvDecl: --overlap-collect cannot combine with the fork arm (--fork-fraction > 0): the "
                         "fork pass runs between the trigger and the fill on the learner's thread")
    if str(device).startswith("cuda") and str(backend) != "graph":
        raise ValueError(f"RustEnvDecl: --overlap-collect on CUDA needs the T2 'graph' backend, not {backend!r}: an "
                         "eager / AOT forward from the worker thread opens a second cuBLAS handle and workspace after "
                         "the freeze")


# ------------------------------------------------------------------------------------------- the snapshots
class BehaviourSnapshots:
    """Two VERSION-KEYED copies of the policy's state dict (the weights T2 was loaded with) plus one swap scratch,
    all allocated at startup on the policy's device. ``take(policy, v)`` overwrites the slot NOT holding ``v - 1``,
    so after the load of version k the slots hold k − 1 and k — the version the stale rows were played at, and the
    current one."""

    def __init__(self, policy: Any) -> None:
        import torch as th

        self._names = list(policy.state_dict(keep_vars=True).keys())
        live = self._live(policy)
        with th.no_grad():
            self._slots = [[t.detach().clone() for t in live] for _ in range(2)]
            self._scratch = [t.detach().clone() for t in live]
        self.versions: List[Optional[int]] = [None, None]

    def _live(self, policy: Any) -> List[Any]:
        sd = policy.state_dict(keep_vars=True)
        if list(sd.keys()) != self._names:
            raise OverlapError("the policy's state dict changed its keys after startup (the declared lifecycle)")
        return [t for t in sd.values()]

    @property
    def nbytes(self) -> int:
        """Bytes held: two slots + the scratch."""
        one = sum(int(t.numel()) * int(t.element_size()) for t in self._scratch)
        return 3 * one

    def take(self, policy: Any, version: int) -> None:
        import torch as th

        v = int(version)
        i = 0 if self.versions[0] is None else (1 if self.versions[1] is None else
                                                (1 if self.versions[0] == v - 1 else 0))
        with th.no_grad():
            for dst, src in zip(self._slots[i], self._live(policy)):
                dst.copy_(src.detach())
        self.versions[i] = v

    def has(self, version: int) -> bool:
        return int(version) in [x for x in self.versions if x is not None]

    @contextlib.contextmanager
    def swapped(self, policy: Any, version: int) -> Iterator[None]:
        """The policy holds the snapshot of ``version`` inside the block; its own tensors are restored
        BIT-EXACTLY on the way out (also on an exception). Parameters are updated in place, so the optimizer's
        state (keyed by parameter object) is untouched."""
        import torch as th

        if not self.has(version):
            raise OverlapError(f"no behaviour snapshot of version {version} (held: {self.versions})")
        slot = self._slots[self.versions.index(int(version))]
        live = self._live(policy)
        with th.no_grad():
            for s, t in zip(self._scratch, live):
                s.copy_(t.detach())
            for t, src in zip(live, slot):
                t.copy_(src)
        try:
            yield
        finally:
            with th.no_grad():
                for t, s in zip(live, self._scratch):
                    t.copy_(s)


# ------------------------------------------------------------------------------------------- the worker
@dataclass
class Played:
    """What one overlapped rollout played: per host step ``(trainee decisions, infos accumulated so far)``; the
    collector's counter snapshot at ``begin``; the timings."""

    steps: List[Tuple[int, int]] = field(default_factory=list)
    snap: Dict[str, float] = field(default_factory=dict)
    t_begin: float = 0.0
    t_done: float = 0.0
    join_wait_s: float = 0.0
    ready: bool = False


class OverlapRunner:
    """The worker (module docs). Built at startup by ``build_collector`` when the declaration says ``overlap``; its
    thread is started at construction and parks between rollouts (nothing is acquired after the freeze).

    Protocol (main thread): ``begin()`` right before ``train()`` → ``join()`` before the next ``after_update`` →
    the collector replays ``Played``. ``abort()`` on any exception out of ``train()``. ``warm`` is False until the
    process's first ``after_update``: the FIRST update of every process runs alone, so the learner's CUDA
    reservation reaches its peak before any flush runs beside it (T2's frozen guard books only T2's own streams,
    ``gen3_t2_stream_attribution_v1``; this keeps its foreign snapshots and the memory peak as in a serial run)."""

    # @startup_builder: built ONCE by `build_collector` at startup (before the freeze); its CUDA stream is part of
    # the declared lifecycle (`src/learner_lifecycle_gate_test.py`)
    @startup_builder
    def __init__(self, collector: Any, policy: Any) -> None:
        import torch as th

        self.col = collector
        self.snapshots = BehaviourSnapshots(policy)
        self.snapshots.take(policy, int(collector.version))     # what T2 serves at startup
        dev = next(iter(policy.parameters())).device
        self.cuda = dev.type == "cuda"
        self.stream = th.cuda.Stream(dev) if self.cuda else None
        self.device = dev
        self.warm = False
        self.rollouts = 0
        self._cv = threading.Condition()
        self._state = "idle"          # idle | playing | done | closed
        self._stop = False
        self._exc: Optional[BaseException] = None
        self._played: Optional[Played] = None
        self._thread = threading.Thread(target=self._loop, name="gen3-overlap-collect", daemon=True)
        self._thread.start()

    # ---- main thread
    @property
    def active(self) -> bool:
        return self._state in ("playing", "done")

    def begin(self) -> None:
        with self._cv:
            if self._state != "idle":
                raise OverlapError(f"begin() with a rollout already in flight (state {self._state})")
            if self.cuda:
                import torch as th
                assert self.stream is not None
                self.stream.wait_stream(th.cuda.current_stream(self.device))
            self._stop = False
            self._exc = None
            self._played = Played(snap=self.col._snapshot(), t_begin=time.perf_counter())
            self._state = "playing"
            self._cv.notify_all()

    def join(self) -> Played:
        if not self.active:
            raise OverlapError("join() with no rollout in flight")
        t0 = time.perf_counter()
        with self._cv:
            while self._state == "playing":
                self._cv.wait()
            played, exc = self._played, self._exc
            self._state = "idle"
            self._played, self._exc = None, None
        if self.cuda:
            import torch as th
            assert self.stream is not None
            th.cuda.current_stream(self.device).wait_stream(self.stream)
        if exc is not None:
            raise exc
        assert played is not None
        played.join_wait_s = time.perf_counter() - t0
        self.rollouts += 1
        return played

    def abort(self, timeout: float = 120.0) -> None:
        """Stop the worker at its next host-step boundary and wait for it (best effort; never raises)."""
        if not self.active:
            return
        self._stop = True
        with self._cv:
            end = time.monotonic() + float(timeout)
            while self._state == "playing" and time.monotonic() < end:
                self._cv.wait(timeout=max(0.0, end - time.monotonic()))
            if self._state == "done":
                self._state = "idle"
                self._played, self._exc = None, None

    def close(self) -> None:
        self.abort()
        with self._cv:
            self._state = "closed"
            self._cv.notify_all()
        self._thread.join(timeout=10.0)

    def proximal_logp(self, buf: Any) -> np.ndarray:
        """log π_prox(a|s) of every row of the learner's buffer ``[n_steps, n_envs]``, served by T2's trainee slot
        — called at the FILL, on the main thread, with the worker idle, when the slot holds the learner's current
        weights (``after_update`` loaded them at this collection's start). Flushes of at most the declared arena."""
        from agents.inference.service.spec import Priority
        from agents.training.rust_rollout.store import KEY_OBSERVATION

        if self.active:
            raise OverlapError("proximal_logp() while the worker is playing (T2 is the worker's)")
        col = self.col
        svc = col.svc
        n_steps, n_envs = int(buf.buffer_size), int(buf.n_envs)
        n = n_steps * n_envs
        obs = buf.observations[KEY_OBSERVATION].reshape(n, -1)
        mask = buf.action_masks.reshape(n, -1)
        acts = buf.actions.reshape(n).astype(np.int64)
        step = max(1, int(svc.spec.max_rows_per_flush))
        out = np.empty(n, dtype=np.float64)
        for s in range(0, n, step):
            e = min(n, s + step)
            t = svc.submit(int(col.current_slot), obs[s:e], mask[s:e], Priority.ROLLOUT)
            svc.flush()
            lp, _v, _g = t.host()
            out[s:e] = lp[np.arange(e - s), acts[s:e]]
        return out.reshape(n_steps, n_envs)

    # ---- the worker thread
    def _loop(self) -> None:
        while True:
            with self._cv:
                while self._state not in ("playing", "closed"):
                    self._cv.wait()
                if self._state == "closed":
                    return
            try:
                if self.stream is not None:
                    import torch as th
                    with th.cuda.stream(self.stream):
                        self._play()
                else:
                    self._play()
            except BaseException as exc:      # handed to the main thread at join()
                self._exc = exc
            with self._cv:
                if self._played is not None:
                    self._played.t_done = time.perf_counter()
                if self._state == "playing":
                    self._state = "done"
                self._cv.notify_all()

    def _play(self) -> None:
        col, played = self.col, self._played
        assert played is not None
        while not self._stop and not col.ready():
            try:
                k = col.host_step()
            except Exception as exc:
                if not col._is_respawned_death(exc):
                    raise
                col.recover_respawn()
                k = 0
            played.steps.append((int(k), len(col._infos)))
        played.ready = bool(col.ready())


# ------------------------------------------------------------------------------------------- the learner side
def runner_of(model: Any) -> Optional[OverlapRunner]:
    """The model's overlap runner, or None (OFF)."""
    return getattr(getattr(model, "_rust_collector", None), "overlap", None)


def judge_snapshot(model: Any, runner: OverlapRunner) -> Optional[Dict[str, float]]:
    """K9(b) on the rows played at version (current − 1), under that version's BEHAVIOUR SNAPSHOT (module docs,
    piece 1): the same probe, gate and tie rule as `consistency.behaviour_probe`, with the snapshot's weights in the
    policy for the forward and its own ages (rows of the snapshot's version are its "current" rows). Tags
    ``behaviour_snapshot/*``; no ``staleness/*`` (the learner's probe writes those). None when the check is off or
    no row was played at that version."""
    from agents.training.rust_rollout.consistency import behaviour_probe

    if str(getattr(model, "behaviour_check", "off") or "off") == "off":
        return None
    versions = getattr(model, "_rust_row_versions", None)
    cur = getattr(model, "_rust_version", None)
    buf = model.rollout_buffer
    if versions is None or cur is None:
        return None
    versions = np.asarray(versions, dtype=np.int64)
    if versions.shape != (int(buf.buffer_size), int(buf.n_envs)):
        return None
    v = int(cur) - 1
    logger = getattr(model, "logger", None)
    if not (versions == v).any() or not runner.snapshots.has(v):
        if logger is not None:
            logger.record(SNAPSHOT_PREFIX + "rows_current", 0.0)
        return None
    ages = v - versions
    ages = np.where(ages < 0, np.iinfo(np.int32).max, ages)     # rows NEWER than the snapshot: never its current
    with runner.snapshots.swapped(model.policy, v):
        return behaviour_probe(model, ages=ages, prefix=SNAPSHOT_PREFIX, staleness=False,
                               where=f"policy version {v} (the BEHAVIOUR SNAPSHOT the overlapped rollout played)",
                               metrics_attr="_behaviour_snapshot_metrics")


def decoupled(model: Any) -> bool:
    """Is ``model``'s update DECOUPLED PPO (`proximal_transform` runs before it, and the micro-step's own
    advantage normalisation is OFF in favour of `decoupled_advantages`)? Exactly when an overlap runner is
    attached — the predicate `TrainSetup._overlap_prepare` reads, so the static the R1 compile declares at startup
    and the update's own agree."""
    return runner_of(model) is not None


def decoupled_advantages(advantages: "torch.Tensor", is_weights: "torch.Tensor", normalize: bool) -> "torch.Tensor":
    """One micro-batch's decoupled-PPO advantage (`gen3_overlap_norm_then_weight_v1`): the RAW advantage
    normalised FIRST — the micro-step's own expression, operation for operation, so ``is_weights`` of ones gives
    the serial update's normalised advantage bit for bit — and weighted AFTER (``w · Â``)."""
    adv = advantages
    if normalize and adv.numel() > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    return adv * is_weights.to(adv.dtype)


def proximal_transform(model: Any, cap: float = IS_WEIGHT_CAP) -> Dict[str, float]:
    """DECOUPLED PPO as a buffer rewrite (module docs, piece 2), from the fill's proximal log-probs
    (``model._overlap_prox``, `OverlapRunner.proximal_logp`): ``log_probs := log π_prox`` and
    ``is_weights := min(π_prox / μ, cap)`` BESIDE the raw advantages, which stay untouched (the micro-batch weights
    them after normalising, `decoupled_advantages`). Before any optimizer step and AFTER the K9(b) probes (they
    read μ). Records and returns the ``overlap/*`` weight tags."""
    buf = model.rollout_buffer
    n_steps, n_envs = int(buf.buffer_size), int(buf.n_envs)
    prox = getattr(model, "_overlap_prox", None)
    if prox is None or np.asarray(prox).shape != (n_steps, n_envs):
        raise OverlapError("the decoupled update has no proximal log-probs for this buffer (the fill computes them: "
                           f"got {None if prox is None else np.asarray(prox).shape}, want {(n_steps, n_envs)})")
    if bool(getattr(buf, "generator_ready", False)):
        raise OverlapError("the decoupled rewrite needs the buffer in its [n_steps, n_envs] layout, but get() has "
                           "already flattened it (a second update on one buffer?)")
    prox = np.asarray(prox, dtype=np.float64).reshape(-1)
    n = prox.size
    mu = buf.log_probs.reshape(n).astype(np.float64)
    log_w = prox - mu
    w_raw = np.exp(log_w)
    w = np.minimum(w_raw, float(cap))
    if not np.isfinite(w).all() or not np.isfinite(prox).all():
        from main.exit_codes import NonFiniteLearnerError
        raise NonFiniteLearnerError("[X14] a NON-FINITE proximal log-prob or importance weight before the update")
    buf.log_probs[...] = prox.reshape(n_steps, n_envs).astype(buf.log_probs.dtype)
    buf.is_weights = w.reshape(n_steps, n_envs).astype(buf.advantages.dtype)
    model._overlap_prox = None                  # consumed: a second update on this buffer must not re-weight it
    out = {
        "overlap/is_weight_mean": float(w.mean()),
        "overlap/is_weight_max_raw": float(w_raw.max()),
        "overlap/is_weight_truncated_frac": float((w_raw > cap).mean()),
        "overlap/is_ess": float(w.sum() ** 2 / (n * float((w * w).sum()))) if n else 1.0,
        "overlap/behaviour_kl": float(((w_raw - 1.0) - log_w).mean()),
    }
    logger = getattr(model, "logger", None)
    if logger is not None:
        for k, v in out.items():
            logger.record(k, v)
    return out
