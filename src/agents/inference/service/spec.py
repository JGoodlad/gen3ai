"""The inference service's DECLARATION: what a service will ever hold, fixed before ``startup()``.

The M5 design principle (``designs/endstate/program_rust_core.md`` §2 M5, "a DECLARED LIFECYCLE"):
startup declares and acquires everything the steady state will use, then freezes; anything that
would appear lazily afterwards is a counted, typed failure. For T2 that means the slot groups (one
architecture each, a fixed slot count), the bucket set, the backend, the device and the output
arena's capacity are all named HERE, and nothing reachable after the freeze can add to them.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any, Optional, Tuple

from agents.model.parity_probe import PERTURB_LADDER, PERTURB_MAX_SCALE

#: The backends. ``eager`` is the reference (CPU or CUDA); ``graph`` is the Inductor-compiled
#: decision forward captured as one CUDA graph per slot x bucket (CUDA only; the default); ``aot``
#: is one AOTInductor package per slot group x bucket with the weights as inputs (CUDA, torch
#: >= 2.8 only — ``aot.py`` says why, and why it is not the default). Measured choice: the program
#: doc's T2 DESIGN paragraph.
BACKENDS = ("eager", "graph", "aot")

#: Smallest legal bucket. torch 2.8 cannot lower a batch-1 CUDA graph of this extractor (Lane K1
#: finding, `designs/training/compile_flags.md`), and a batch dim of 1 is a degenerate broadcast
#: that can hide a shape bug; a single row is padded to the smallest bucket instead.
MIN_BUCKET = 2


class ServiceError(RuntimeError):
    """Base of every typed inference-service failure."""


class LifecycleViolation(ServiceError):
    """An operation outside the declared lifecycle (before the freeze, after a poison, or a
    resource acquired after the freeze — the ``*_after_freeze`` counters)."""


class SlotArchMismatch(ServiceError):
    """A weight set whose architecture differs from the slot group's (state-dict signature or
    forward-config fingerprint)."""


class ParityFailure(ServiceError):
    """The served decision disagrees with the eager reference beyond the compile gate's bars."""


class VacuousParity(ParityFailure):
    """The parity comparison cannot bite: the eager reference's legal log-probs are constant per row
    (a FRESH policy — zero-init pointer head) or its V is constant across rows (a COLLAPSED win-prob
    critic, saturated). Fail-closed: the service re-runs the gate on seeded perturbations of the
    weights up the declared ladder (``ServiceSpec.perturb_ladder``) and raises this only when NO
    rung is informative (gen3_fresh_parity_probe_v1, gen3_parity_perturb_ladder_v1). Deterministic
    in (code, weights, fixture), so a restart replays it: the trainer exits ``FATAL_CONFIG``."""


class NonFiniteWeights(ServiceError):
    """A weight set (a group template at startup, or a ``load``) carries a NaN / Inf parameter or
    buffer. Refused BEFORE anything is copied into a slot — never judged, never perturbed (noise on
    a NaN is a NaN). The trainer exits ``FATAL_NONFINITE``: a restart resumes the same weights."""


class CallerError(ServiceError):
    """A malformed request (shapes, dtypes, an all-illegal row, a stale result, over capacity)."""


class Priority(IntEnum):
    """Scheduling classes, highest first. ``flush()`` serves every ROLLOUT row and at most
    ``filler_batches_per_flush`` bucket batches of the rest."""

    ROLLOUT = 0
    EVAL = 1
    FILLER = 2


@dataclass(frozen=True)
class SlotGroupSpec:
    """One architecture served from ``n_slots`` fixed slots.

    ``template`` is a Gen3 dual-head policy: it FIXES the group's architecture (its state-dict
    signature and forward-config fingerprint) and its weights are what every slot holds at startup
    (the parity gate runs on them — give it REAL weights, e.g. the trainee's checkpoint, where one
    exists: a miscompile can be weight-dependent). The template is deep-copied; the caller's object
    is never mutated or served.
    """

    name: str
    n_slots: int
    template: Any


@dataclass(frozen=True)
class ServiceSpec:
    groups: Tuple[SlotGroupSpec, ...]
    device: str = "cuda"
    buckets: Tuple[int, ...] = (8, 48, 128)
    backend: str = "graph"
    #: Rows one flush may serve (the declared output arena). ROLLOUT rows beyond it are a
    #: `CallerError` at flush; filler work is only scheduled while it fits.
    max_rows_per_flush: int = 1024
    #: Bucket batches of EVAL / FILLER work one ``flush()`` may add after the ROLLOUT rows.
    filler_batches_per_flush: int = 1
    #: The bucket every ``load`` and ``canary`` verifies at (default: the smallest).
    verify_bucket: Optional[int] = None
    #: Concurrency LANES (CUDA streams, each with its own graph pool and static inputs); slot s
    #: runs on lane s % lanes. 1 on CPU.
    lanes: int = 1
    #: PER-SLOT bucket caps (gen3_slot_bucket_caps_v1), one per global slot in declaration order, or
    #: EMPTY = every slot serves every bucket. Slot s captures / serves only the buckets <= its cap
    #: (always at least the smallest); a request larger than its largest bucket is CHUNKED, never
    #: refused. Each lane's private graph pool is sized by the largest capture on that lane, so a
    #: small cap on the opponent slots keeps all but the trainee's lane small (measured 2026-10-01:
    #: 232 MiB per lane at bucket 256, 60 at 64, 34 at 32).
    slot_bucket_caps: Tuple[int, ...] = ()
    #: Where backend 'aot' writes its packages (default: a fresh temp dir — never shared).
    artifact_dir: Optional[str] = None
    #: The seeded-perturbation LADDER a vacuous parity comparison climbs (`parity_probe`, module
    #: docs): the first informative rung judges the slot, none ⇒ `VacuousParity`. EMPTY disables
    #: the perturbed path, so a flat slot is REFUSED — never passed.
    perturb_ladder: Tuple[Tuple[float, int], ...] = PERTURB_LADDER

    def validate(self) -> None:
        if not self.groups:
            raise ValueError("ServiceSpec: declare at least one slot group")
        names = [g.name for g in self.groups]
        if len(set(names)) != len(names):
            raise ValueError(f"ServiceSpec: duplicate slot-group names {names}")
        for g in self.groups:
            if int(g.n_slots) < 1:
                raise ValueError(f"slot group {g.name!r}: n_slots must be >= 1")
        b = tuple(int(x) for x in self.buckets)
        if not b or list(b) != sorted(set(b)):
            raise ValueError(f"ServiceSpec: buckets must be strictly ascending, got {self.buckets}")
        if b[0] < MIN_BUCKET:
            raise ValueError(f"ServiceSpec: the smallest bucket is {MIN_BUCKET} (got {b[0]}): a "
                             "batch-1 graph does not lower on torch 2.8 and is a degenerate "
                             "broadcast; a single row is padded instead")
        if self.backend not in BACKENDS:
            raise ValueError(f"ServiceSpec: backend {self.backend!r} not in {BACKENDS}")
        if self.backend in ("graph", "aot") and not str(self.device).startswith("cuda"):
            raise ValueError(f"ServiceSpec: backend {self.backend!r} is CUDA only")
        if int(self.lanes) < 1 or (int(self.lanes) > 1 and not str(self.device).startswith("cuda")):
            raise ValueError("ServiceSpec: lanes must be >= 1, and > 1 only on CUDA")
        if int(self.max_rows_per_flush) < b[-1]:
            raise ValueError("ServiceSpec: max_rows_per_flush must hold at least the largest bucket")
        if int(self.filler_batches_per_flush) < 0:
            raise ValueError("ServiceSpec: filler_batches_per_flush must be >= 0")
        try:
            lad = [(float(sc), int(k)) for sc, k in self.perturb_ladder]
        except (TypeError, ValueError):
            raise ValueError(f"ServiceSpec: perturb_ladder rungs are (scale, seed offset) pairs, got "
                             f"{self.perturb_ladder}") from None
        if (any(not (0.0 < sc <= PERTURB_MAX_SCALE) or k < 0 for sc, k in lad)
                or len(set(lad)) != len(lad) or lad != sorted(lad)):
            raise ValueError(f"ServiceSpec: perturb_ladder must be distinct (scale, seed offset) rungs "
                             f"in ascending order, 0 < scale <= {PERTURB_MAX_SCALE} (the bars' "
                             f"calibrated range), offset >= 0 — or empty; got {self.perturb_ladder}")
        n_slots = sum(int(g.n_slots) for g in self.groups)
        caps = tuple(int(c) for c in self.slot_bucket_caps)
        if caps and (len(caps) != n_slots or min(caps) < b[0]):
            raise ValueError(f"ServiceSpec: slot_bucket_caps must name one cap >= the smallest bucket "
                             f"{b[0]} per declared slot ({n_slots}), got {self.slot_bucket_caps}")
        if self.verify_bucket is not None and int(self.verify_bucket) not in b:
            raise ValueError(f"ServiceSpec: verify_bucket {self.verify_bucket} is not a declared bucket")

    @property
    def verify_at(self) -> int:
        return int(self.verify_bucket) if self.verify_bucket is not None else int(self.buckets[0])
