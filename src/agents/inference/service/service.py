"""THE INFERENCE SERVICE (M5 Lane T2): fixed weight slots x fixed buckets, a declared lifecycle.

Design, measurements and the API contract: ``designs/endstate/program_rust_core.md`` §2, the "T2
DESIGN" paragraph. In one breath: ``InferenceService(spec).startup()`` builds every slot replica in
stacked storage, builds the ``Engine`` (arenas, lanes, one compiled forward per bucket captured as
one CUDA graph per slot x bucket — backend ``graph`` — or one AOT package per group x bucket),
parity-gates EVERY slot x bucket (and, with lanes, every slot at once) on the committed real-obs
fixture, and FREEZES. After the freeze a ``flush()`` only packs, copies, replays and copies out —
it compiles, captures and allocates nothing, and three ``*_after_freeze`` counters prove it on
every call. This module is the API, the scheduling and the lifecycle; ``engine.py`` executes.

Not thread-safe: one caller loop drives it (the M5 env core is in-process). Results are views into
the arenas, valid until the next ``flush()`` (a later read is a `CallerError`, never stale data).
"""
from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch

from agents.inference.service.engine import Engine, PlanItem
from agents.inference.service.parity import ParityReport, fixture_rows, gate_slot, judge
from agents.inference.service.slots import SlotGroup
from agents.model.parity_probe import PERTURB_SEED, perturbed_parameters
from agents.inference.service.spec import (
    CallerError, LifecycleViolation, ParityFailure, Priority, ServiceError, ServiceSpec,
    VacuousParity,
)



@dataclass
class Decision:
    """``logp [n, A]`` (illegal = -inf), ``value [n]``, ``greedy [n]`` — device tensors."""

    logp: torch.Tensor
    value: torch.Tensor
    greedy: torch.Tensor

    def numpy(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.logp.cpu().numpy(), self.value.cpu().numpy(), self.greedy.cpu().numpy()


class Ticket:
    """One submitted request. ``result()`` after the flush that served it (and before the next)."""

    def __init__(self, svc: "InferenceService", slot: int, obs: np.ndarray, mask: np.ndarray,
                 priority: Priority, seq: int):
        self._svc, self.slot, self.obs, self.mask = svc, slot, obs, mask
        self.priority, self.seq, self.n = priority, seq, int(obs.shape[0])
        self._epoch: Optional[int] = None
        self._off = -1

    @property
    def done(self) -> bool:
        return self._epoch is not None

    def result(self) -> Decision:
        svc = self._svc
        if svc._poison is not None:
            raise LifecycleViolation(f"the service is POISONED ({svc._poison}); no result is valid")
        if self._epoch is None:
            raise CallerError(f"ticket {self.seq} (slot {self.slot}) has not been served — flush()")
        if self._epoch != svc._epoch:
            raise CallerError(f"ticket {self.seq}: its result was overwritten by a later flush "
                              "(results are valid until the next flush; copy what you keep)")
        sl = slice(self._off, self._off + self.n)
        e = svc.engine
        return Decision(e.out_logp[sl], e.out_value[sl], e.out_greedy[sl])

    def host(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(logp, value, greedy)`` as NumPy VIEWS into the pinned host output arena (the flush
        copied every output there in one transfer). Waits for that transfer; same validity rule
        as ``result()``: copy what you keep before the next flush."""
        self.result()                                   # the validity checks
        e = self._svc.engine
        e.wait_host()
        sl = slice(self._off, self._off + self.n)
        return e.host_out[0][sl].numpy(), e.host_out[1][sl].numpy(), e.host_out[2][sl].numpy()


def _segments(device: torch.device) -> int:
    if device.type != "cuda":
        return 0
    return int(torch.cuda.memory_stats(device).get("segment.all.allocated", 0))


def _dynamo_graphs() -> int:
    from agents.model.compile_control import dynamo_graphs_total

    return dynamo_graphs_total()


class InferenceService:
    def __init__(self, spec: ServiceSpec):
        spec.validate()
        self.spec = spec
        self.device = torch.device(spec.device)
        self.buckets: Tuple[int, ...] = tuple(int(b) for b in spec.buckets)
        self.state = "DECLARED"
        self.counters: Dict[str, int] = {
            "compiles_after_freeze": 0, "captures_after_freeze": 0,
            "cuda_segments_after_freeze": 0, "rows_served": 0, "rows_padded": 0, "batches": 0,
            "loads": 0, "flushes": 0}
        self.batches_by_bucket: Dict[int, int] = {b: 0 for b in self.buckets}
        self.startup_reports: List[ParityReport] = []
        self.startup_seconds: Dict[str, float] = {}
        self._pending: List[Ticket] = []
        self._seq = 0
        self._epoch = 0
        self._poison: Optional[str] = None

    # ------------------------------------------------------------------ lifecycle
    def _require(self, *states: str) -> None:
        if self._poison is not None:
            raise LifecycleViolation(f"the service is POISONED ({self._poison}); every call refuses")
        if self.state not in states:
            raise LifecycleViolation(f"{self.state}: this call needs state {states}")

    def _die(self, why: str, exc: Optional[BaseException] = None) -> None:
        self._poison = why
        self.state = "POISONED"
        if exc is not None and not isinstance(exc, ServiceError):
            raise LifecycleViolation(why) from exc

    @contextlib.contextmanager
    def _frozen_guard(self, where: str) -> Iterator[None]:
        """Count what a steady-state call acquired; any acquisition poisons the service."""
        g0, s0, c0 = _dynamo_graphs(), _segments(self.device), self.engine.captures
        yield
        dg, ds = _dynamo_graphs() - g0, _segments(self.device) - s0
        dc = self.engine.captures - c0
        self.counters["compiles_after_freeze"] += max(dg, 0)
        self.counters["cuda_segments_after_freeze"] += max(ds, 0)
        self.counters["captures_after_freeze"] += max(dc, 0)
        if dg > 0 or ds > 0 or dc > 0:
            why = (f"{where} acquired after the freeze: {dg} dynamo graphs, {ds} CUDA segments, "
                   f"{dc} graph captures (the declared lifecycle allows none)")
            self._die(why)
            raise LifecycleViolation(why)
        if torch.get_float32_matmul_precision() != self._precision:
            why = (f"{where}: fp32 matmul precision changed after the freeze "
                   f"({self._precision!r} -> {torch.get_float32_matmul_precision()!r})")
            self._die(why)
            raise LifecycleViolation(why)

    def startup(self) -> "InferenceService":
        self._require("DECLARED")
        self.state = "STARTING"
        try:
            self._startup()
        except BaseException as exc:
            self._die(f"startup failed: {type(exc).__name__}: {exc}", exc)
            raise
        self.state = "FROZEN"
        return self

    def _startup(self) -> None:
        t0 = time.perf_counter()
        self._precision = torch.get_float32_matmul_precision()
        self.groups = [SlotGroup(g, self.device) for g in self.spec.groups]
        dims = {(int(g.policies[0].observation_space["observation"].shape[0]),
                 int(g.policies[0].action_space.n)) for g in self.groups}
        if len(dims) != 1:
            raise ServiceError(f"slot groups disagree on (obs_dim, n_actions): {sorted(dims)}")
        self.obs_dim, self.n_actions = dims.pop()
        self._slots: List[Tuple[int, int]] = [(gi, i) for gi, g in enumerate(self.groups)
                                              for i in range(g.n_slots)]
        self.engine = Engine(groups=self.groups, slots=self._slots, device=self.device,
                             buckets=self.buckets, backend=self.spec.backend,
                             max_rows=int(self.spec.max_rows_per_flush), lanes=int(self.spec.lanes),
                             obs_dim=self.obs_dim, n_actions=self.n_actions,
                             artifact_dir=self.spec.artifact_dir)
        if self.spec.backend == "aot":
            from agents.inference.service import aot

            aot.check_available()
        self.startup_seconds["allocate"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        self.engine.build()
        self.startup_seconds["build"] = time.perf_counter() - t1
        t2 = time.perf_counter()
        for gi, g in enumerate(self.groups):
            for i in range(g.n_slots):
                for b in self.buckets:
                    self.startup_reports.extend(self._gate(gi, i, b))
        for b in self.buckets:                  # every slot AT ONCE: lanes replay concurrently
            self.startup_reports.extend(self._gate_concurrent(b))
        self.startup_seconds["parity"] = time.perf_counter() - t2
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    # ------------------------------------------------------------------ slots
    def slot(self, group: str, index: int) -> int:
        """The global slot id of ``group``'s ``index``-th slot."""
        self._require("FROZEN")
        names = [g.name for g in self.groups]
        if group not in names:
            raise CallerError(f"no slot group {group!r} (declared: {names})")
        gi = names.index(group)
        if not 0 <= int(index) < self.groups[gi].n_slots:
            raise CallerError(f"slot group {group!r} has {self.groups[gi].n_slots} slots, not {index}")
        return self._slots.index((gi, int(index)))

    def model_id(self, slot: int) -> str:
        gi, i = self._slot(slot)
        return self.groups[gi].model_ids[i]

    def _slot(self, slot: int) -> Tuple[int, int]:
        if not 0 <= int(slot) < len(self._slots):
            raise CallerError(f"slot {slot} is not declared (0..{len(self._slots) - 1})")
        return self._slots[int(slot)]

    def load(self, slot: int, policy: Any, model_id: str) -> ParityReport:
        """Copy ``policy``'s weights into ``slot`` (in place) and parity-verify the slot at the
        verify bucket. A mismatch is `SlotArchMismatch` (the slot is untouched); a parity failure
        POISONS the service (the weights are in the slot and disagree with eager)."""
        self._require("FROZEN")
        gi, i = self._slot(slot)
        group = self.groups[gi]
        sd = group.check_loadable(policy)
        with self._frozen_guard(f"load(slot {slot})"):
            group.copy_in(i, sd)
        group.model_ids[i] = str(model_id)
        self.counters["loads"] += 1
        try:
            reports = self._gate(gi, i, self.spec.verify_at)
        except ParityFailure as exc:
            self._die(f"load(slot {slot}, {model_id!r}) failed parity: {exc}")
            raise
        return reports[0]

    def canary(self, slot: Optional[int] = None) -> List[ParityReport]:
        """The K6 IN-RUN CANARY: re-run the parity gate at the verify bucket on one slot (or all).
        A failure poisons the service."""
        self._require("FROZEN")
        out: List[ParityReport] = []
        for s in ([int(slot)] if slot is not None else range(len(self._slots))):
            gi, i = self._slot(s)
            try:
                out.extend(self._gate(gi, i, self.spec.verify_at))
            except ParityFailure as exc:
                self._die(f"canary(slot {s}) failed: {exc}")
                raise
        if slot is None:
            try:
                out.extend(self._gate_concurrent(self.spec.verify_at))
            except ParityFailure as exc:
                self._die(f"canary(concurrent) failed: {exc}")
                raise
        return out

    def _gate_concurrent(self, b: int) -> List[ParityReport]:
        """ONE flush carrying every slot's fixture rows (a full chunk of bucket ``b``), so every
        lane replays at the same time, each slot judged against its own eager reference. The
        per-slot gate runs one slot per flush and can never see two lanes interfere (the shared
        cuBLAS-workspace defect read 0.048 on log-probs only under concurrency).

        FRESH weights (gen3_fresh_parity_probe_v1): if any slot's comparison is vacuous, the flush
        is re-run with EVERY slot perturbed in place (a different seed per slot, so concurrent
        slots compute different functions), then once more on the real weights with the vacuity
        guard waived; the real reports come first."""
        if self.engine.n_lanes < 2 or len(self._slots) < 2:
            return []
        try:
            return self._concurrent_once(b, tag="", allow_vacuous=False)
        except VacuousParity:
            with contextlib.ExitStack() as stack:
                for s, (gi, i) in enumerate(self._slots):
                    stack.enter_context(perturbed_parameters(self.groups[gi].policies[i],
                                                             seed=PERTURB_SEED + s))
                probe = self._concurrent_once(b, tag=" [fresh weights, seeded perturbation]",
                                              allow_vacuous=False)
            return self._concurrent_once(b, tag="", allow_vacuous=True) + probe

    def _concurrent_once(self, b: int, *, tag: str, allow_vacuous: bool) -> List[ParityReport]:
        guard = self._frozen_guard if self.state == "FROZEN" else self._startup_guard
        obs, mask = fixture_rows(self.obs_dim, b)
        plan: List[PlanItem] = [(s, gi, i, obs, mask) for s, (gi, i) in enumerate(self._slots)]
        self._epoch += 1
        with guard(f"concurrent parity bucket={b}"):
            offsets, _ = self.engine.execute(plan)
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        e = self.engine
        o = torch.as_tensor(obs, device=self.device)
        m = torch.as_tensor(mask, device=self.device)
        reports = []
        for (s, gi, i, _, _), off in zip(plan, offsets):
            sl = slice(off, off + b)
            reports.append(judge(
                where=f"{self.spec.backend} CONCURRENT lanes={e.n_lanes} slot={s} bucket={b}{tag}",
                policy=self.groups[gi].policies[i], obs=o, mask=m,
                served=(e.out_logp[sl].clone(), e.out_value[sl].clone(), e.out_greedy[sl].clone()),
                allow_vacuous=allow_vacuous))
        return reports

    def _gate(self, gi: int, i: int, b: int) -> Tuple[ParityReport, ...]:
        g = self.groups[gi]
        guard = self._frozen_guard if self.state == "FROZEN" else self._startup_guard
        where = f"{self.spec.backend} group={g.name} slot={i} bucket={b}"

        slot = self._slots.index((gi, i))

        def serve(obs: np.ndarray, mask: np.ndarray) -> Tuple[torch.Tensor, ...]:
            n = obs.shape[0]
            if n not in (b, b - 1):
                raise ServiceError(f"parity rows {n} do not exercise bucket {b}")
            self._epoch += 1                     # a gate run overwrites the output arena
            with guard(f"parity {where}"):       # the service's own execution only, through the
                self.engine.execute([(slot, gi, i, obs, mask)])      # REAL staging path
            e = self.engine
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()
            return (e.out_logp[:n].clone(), e.out_value[:n].clone(), e.out_greedy[:n].clone())

        try:
            return gate_slot(where=where, policy=g.policies[i], obs_dim=self.obs_dim, bucket=b,
                             device=self.device, serve=serve)
        except VacuousParity:
            # gen3_fresh_parity_probe_v1: FRESH weights (a zero-init pointer head) cannot judge the
            # slot. Judge the SAME slot and graph on a seeded perturbation of its weights — in place
            # (the slot's parameters are views into the group's stacked storage, which the graphs
            # read), restored bit-exactly, private RNG — then the real weights with the vacuity
            # guard waived. The real report stays first (`load` returns reports[0]).
            with perturbed_parameters(g.policies[i]):
                probe = gate_slot(where=f"{where} [fresh weights, seeded perturbation]",
                                  policy=g.policies[i], obs_dim=self.obs_dim, bucket=b,
                                  device=self.device, serve=serve)
            real = gate_slot(where=where, policy=g.policies[i], obs_dim=self.obs_dim, bucket=b,
                             device=self.device, serve=serve, allow_vacuous=True)
            return real + probe

    @contextlib.contextmanager
    def _startup_guard(self, where: str) -> Iterator[None]:
        yield

    # ------------------------------------------------------------------ requests
    def submit(self, slot: int, obs: Any, mask: Any, priority: Priority = Priority.ROLLOUT) -> Ticket:
        self._require("FROZEN")
        self._slot(slot)
        # COPIED: a caller's rows are often a view into a buffer its env core rewrites on the next
        # step (Lane A/B's column mappings); a ticket must hold what was submitted, not what that
        # buffer holds at flush time.
        o = np.array(obs, dtype=np.float32, order="C", copy=True)
        m = np.array(mask, dtype=bool, order="C", copy=True)
        if o.ndim != 2 or o.shape[1] != self.obs_dim:
            raise CallerError(f"obs must be [n, {self.obs_dim}] float32, got {o.shape}")
        if m.shape != (o.shape[0], self.n_actions):
            raise CallerError(f"mask must be [n, {self.n_actions}] bool, got {m.shape}")
        if o.shape[0] < 1:
            raise CallerError("a request needs at least one row")
        if o.shape[0] > self.spec.max_rows_per_flush:
            raise CallerError(f"{o.shape[0]} rows exceed max_rows_per_flush "
                              f"{self.spec.max_rows_per_flush} (the declared arena)")
        if not m.any(axis=1).all():
            raise CallerError("a row with no legal action has no decision to serve")
        t = Ticket(self, int(slot), o, m, Priority(priority), self._seq)
        self._seq += 1
        self._pending.append(t)
        return t

    def score(self, slot: int, obs: Any, mask: Any, priority: Priority = Priority.ROLLOUT) -> Decision:
        """Submit + flush for a synchronous caller (serves anything else pending too)."""
        t = self.submit(slot, obs, mask, priority)
        self.flush()
        if not t.done:                                  # a lower class beyond the filler budget
            self.drain()
        return t.result()

    @property
    def pending_rows(self) -> Dict[str, int]:
        out = {p.name: 0 for p in Priority}
        for t in self._pending:
            out[t.priority.name] += t.n
        return out

    def _chunks(self, n: int) -> List[Tuple[int, int]]:
        return self.engine.chunks(n)

    def flush(self) -> int:
        """Serve every ROLLOUT row, then up to ``filler_batches_per_flush`` batches of the rest."""
        return self._flush(filler_budget=int(self.spec.filler_batches_per_flush))

    def drain(self) -> int:
        """Serve everything pending, one arena-full per flush."""
        total = 0
        while self._pending:
            total += self._flush(filler_budget=None)
        return total

    def _flush(self, filler_budget: Optional[int]) -> int:
        self._require("FROZEN")
        R = int(self.spec.max_rows_per_flush)
        rollout = [t for t in self._pending if t.priority == Priority.ROLLOUT]
        rows = sum(t.n for t in rollout)
        if rows > R:
            raise CallerError(f"{rows} ROLLOUT rows pending exceed max_rows_per_flush {R} "
                              "(the declared arena); flush more often or declare a larger arena")
        admitted = list(rollout)
        used = 0
        for t in sorted((t for t in self._pending if t.priority != Priority.ROLLOUT),
                        key=lambda t: (t.priority, t.seq)):         # FIFO within a class: stop,
            if filler_budget is not None:                            # never skip ahead
                need = len(self._chunks(t.n))
                if filler_budget == 0 or (used > 0 and used + need > filler_budget):
                    break
                used += need
            if rows + t.n > R:
                break
            admitted.append(t)
            rows += t.n
        if not admitted:
            return 0
        self._epoch += 1
        plan: List[PlanItem] = []
        for t in admitted:
            gi, i = self._slot(t.slot)
            plan.append((t.slot, gi, i, t.obs, t.mask))
        try:
            with self._frozen_guard("flush"):
                offsets, per_bucket = self.engine.execute(plan)
        except BaseException as exc:         # a half-served flush must never be read as served
            if self._poison is None:
                self._die(f"flush failed mid-way: {type(exc).__name__}: {exc}")
            raise
        for t, off in zip(admitted, offsets):
            t._off, t._epoch = off, self._epoch
        for b, k in per_bucket.items():
            self.batches_by_bucket[b] += k
            self.counters["batches"] += k
        self.counters["rows_padded"] += sum(b * k for b, k in per_bucket.items()) - rows
        done = {id(t) for t in admitted}
        self._pending = [t for t in self._pending if id(t) not in done]
        self.counters["flushes"] += 1
        self.counters["rows_served"] += rows
        return rows

    # ------------------------------------------------------------------ reporting
    def stats(self) -> Dict[str, Any]:
        out: Dict[str, Any] = dict(self.counters)
        out["state"] = self.state
        out["batches_by_bucket"] = dict(self.batches_by_bucket)
        out["slots"] = len(getattr(self, "_slots", ()))
        eng = getattr(self, "engine", None)
        out["graphs"] = len(eng.graphs) if eng is not None else 0
        out["packages"] = len(eng.packages) if eng is not None else 0
        out["lanes"] = eng.n_lanes if eng is not None else 0
        out["startup_seconds"] = dict(self.startup_seconds)
        return out
