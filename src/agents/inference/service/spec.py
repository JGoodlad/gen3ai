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

#: The two backends. ``eager`` is the reference (CPU or CUDA); ``graph`` is the Inductor-compiled
#: decision forward captured as one CUDA graph per slot x bucket (CUDA only). Why not AOTInductor
#: on torch 2.5.1: the program doc's T2 DESIGN paragraph (measured).
BACKENDS = ("eager", "graph")

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
    (a FRESH policy — zero-init pointer head) or its V is constant across rows. Fail-closed: a
    caller that does not re-run the gate on a seeded perturbation of the weights gets a
    `ParityFailure` (gen3_fresh_parity_probe_v1)."""


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
        if self.backend == "graph" and not str(self.device).startswith("cuda"):
            raise ValueError("ServiceSpec: backend 'graph' is CUDA graphs — CUDA only")
        if int(self.max_rows_per_flush) < b[-1]:
            raise ValueError("ServiceSpec: max_rows_per_flush must hold at least the largest bucket")
        if int(self.filler_batches_per_flush) < 0:
            raise ValueError("ServiceSpec: filler_batches_per_flush must be >= 0")
        if self.verify_bucket is not None and int(self.verify_bucket) not in b:
            raise ValueError(f"ServiceSpec: verify_bucket {self.verify_bucket} is not a declared bucket")

    @property
    def verify_at(self) -> int:
        return int(self.verify_bucket) if self.verify_bucket is not None else int(self.buckets[0])
