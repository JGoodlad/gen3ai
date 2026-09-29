"""THE INFERENCE SERVICE (M5 Lane T2): fixed weight slots x fixed buckets, a declared lifecycle.

Design, measurements and the API contract: ``designs/endstate/program_rust_core.md`` §2, the "T2
DESIGN" paragraph. In one breath: ``InferenceService(spec).startup()`` builds every slot replica in
stacked storage, compiles one decision forward per bucket and captures one CUDA graph per slot x
bucket (backend ``graph``), parity-gates EVERY slot x bucket on the committed real-obs fixture, and
FREEZES. After the freeze a ``flush()`` only stages rows into pinned host buffers, copies them into
the static device inputs, replays, and copies the outputs into the declared arena — it compiles,
captures and allocates nothing, and three ``*_after_freeze`` counters prove it on every call.

Not thread-safe: one caller loop drives it (the M5 env core is in-process). Results are views into
the arena, valid until the next ``flush()`` (a later read is a `CallerError`, never stale data).
"""
from __future__ import annotations

import contextlib
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch

from agents.inference.service.parity import ParityReport, fixture_rows, gate_slot
from agents.inference.service.slots import SlotGroup
from agents.model.parity_probe import perturbed_parameters
from agents.inference.service.spec import (
    CallerError, LifecycleViolation, ParityFailure, Priority, ServiceError, ServiceSpec,
    VacuousParity,
)


def _decide(module: torch.nn.Module, obs: torch.Tensor, mask: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The one callable every backend runs (and ``graph`` compiles): logp, V and greedy."""
    logp, value = module(obs, mask)
    return logp, value, logp.argmax(-1)


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
        return Decision(svc._arena_logp[sl], svc._arena_value[sl], svc._arena_greedy[sl])


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
        self._captures = 0
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
        g0, s0, c0 = _dynamo_graphs(), _segments(self.device), self._captures
        yield
        dg, ds, dc = _dynamo_graphs() - g0, _segments(self.device) - s0, self._captures - c0
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
        R, A, D, dev = int(self.spec.max_rows_per_flush), self.n_actions, self.obs_dim, self.device
        pin = dev.type == "cuda"
        self._arena_logp = torch.empty(R, A, dtype=torch.float32, device=dev)
        self._arena_value = torch.empty(R, dtype=torch.float32, device=dev)
        self._arena_greedy = torch.empty(R, dtype=torch.int64, device=dev)
        self._stage: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._static: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        for b in self.buckets:
            o, m = fixture_rows(D, b)
            ho = torch.empty(b, D, dtype=torch.float32, pin_memory=pin)
            hm = torch.empty(b, A, dtype=torch.bool, pin_memory=pin)
            ho.numpy()[:] = o
            hm.numpy()[:] = m
            self._stage[b] = (ho, hm)
            self._static[b] = (ho.to(dev).clone(), hm.to(dev).clone())
        self.startup_seconds["allocate"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        self._graphs: Dict[Tuple[int, int, int], Tuple[Any, Tuple[torch.Tensor, ...]]] = {}
        if self.spec.backend == "graph":
            self._build_graphs()
        else:
            with torch.no_grad():                     # warm every slot x bucket (allocator)
                for gi, g in enumerate(self.groups):
                    for i in range(g.n_slots):
                        for b in self.buckets:
                            _decide(g.modules[i], *self._static[b])
        self.startup_seconds["build"] = time.perf_counter() - t1
        t2 = time.perf_counter()
        for gi, g in enumerate(self.groups):
            for i in range(g.n_slots):
                for b in self.buckets:
                    self.startup_reports.extend(self._gate(gi, i, b))
        self.startup_seconds["parity"] = time.perf_counter() - t2
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _build_graphs(self) -> None:
        compiled = torch.compile(_decide, dynamic=False)
        pool = torch.cuda.graph_pool_handle()
        limit = max(int(torch._dynamo.config.cache_size_limit),
                    len(self.buckets) * len(self.groups) + 4)
        # suppress_errors=False: a compile failure RAISES rather than silently running eager (the
        # eager forward would then fail capture anyway — it makes 24 host syncs — but the typed
        # error names the real cause).
        with torch._dynamo.config.patch(  # type: ignore[attr-defined]
                suppress_errors=False, cache_size_limit=limit), \
                torch.no_grad():
            for b in self.buckets:
                so, sm = self._static[b]
                for gi, g in enumerate(self.groups):
                    for i in range(g.n_slots):
                        mod = g.modules[i]
                        side = torch.cuda.Stream(self.device)
                        side.wait_stream(torch.cuda.current_stream(self.device))
                        with torch.cuda.stream(side):
                            for _ in range(2):
                                compiled(mod, so, sm)
                        torch.cuda.current_stream(self.device).wait_stream(side)
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, pool=pool):
                            outs = compiled(mod, so, sm)
                        self._captures += 1
                        self._graphs[(gi, i, b)] = (graph, tuple(outs))

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
        return out

    def _gate(self, gi: int, i: int, b: int) -> Tuple[ParityReport, ...]:
        g = self.groups[gi]
        guard = self._frozen_guard if self.state == "FROZEN" else self._startup_guard
        where = f"{self.spec.backend} group={g.name} slot={i} bucket={b}"

        def serve(obs: np.ndarray, mask: np.ndarray) -> Tuple[torch.Tensor, ...]:
            n = obs.shape[0]
            with guard(f"parity {where}"):       # the service's own execution only
                self._stage_rows(b, [(obs, mask, 0, n)], n)
                outs = self._run(gi, i, b)
            return tuple(t[:n].clone() for t in outs)   # read before anything else runs

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
        o = np.ascontiguousarray(np.asarray(obs, dtype=np.float32))
        m = np.ascontiguousarray(np.asarray(mask).astype(bool, copy=False))
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
        """``(bucket, rows)`` chunks for ``n`` rows: largest buckets, then the smallest that fits."""
        big, out = self.buckets[-1], []
        while n > big:
            out.append((big, big))
            n -= big
        out.append((next(b for b in self.buckets if b >= n), n))
        return out

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
        off = 0
        for t in admitted:
            t._off, t._epoch = off, self._epoch
            off += t.n
        by_slot: Dict[int, List[Ticket]] = {}
        for t in sorted(admitted, key=lambda t: t.seq):
            by_slot.setdefault(t.slot, []).append(t)
        try:
            with self._frozen_guard("flush"):
                for slot, tickets in by_slot.items():
                    self._serve_slot(slot, tickets)
        except BaseException as exc:         # a half-served flush must never be read as served
            if self._poison is None:
                self._die(f"flush failed mid-way: {type(exc).__name__}: {exc}")
            raise
        done = {id(t) for t in admitted}
        self._pending = [t for t in self._pending if id(t) not in done]
        self.counters["flushes"] += 1
        self.counters["rows_served"] += rows
        return rows

    def _serve_slot(self, slot: int, tickets: Sequence[Ticket]) -> None:
        gi, i = self._slot(slot)
        # Flatten the tickets' rows into segments (ticket, row0, n), then cut into bucket chunks.
        segs = [(t, 0, t.n) for t in tickets]
        total = sum(t.n for t in tickets)
        for b, n in self._chunks(total):
            parts: List[Tuple[Ticket, int, int, int]] = []    # (ticket, t_row0, n, batch_row0)
            k = 0
            while k < n:
                t, r0, left = segs[0]
                take = min(left, n - k)
                parts.append((t, r0, take, k))
                k += take
                segs[0] = (t, r0 + take, left - take)
                if segs[0][2] == 0:
                    segs.pop(0)
            self._stage_rows(b, [(t.obs[r0:r0 + c], t.mask[r0:r0 + c], k0, c)
                                 for t, r0, c, k0 in parts], n)
            logp, value, greedy = self._run(gi, i, b)
            for t, r0, c, k0 in parts:
                dst = slice(t._off + r0, t._off + r0 + c)
                self._arena_logp[dst].copy_(logp[k0:k0 + c])
                self._arena_value[dst].copy_(value[k0:k0 + c])
                self._arena_greedy[dst].copy_(greedy[k0:k0 + c])
            self.counters["batches"] += 1
            self.counters["rows_padded"] += b - n
            self.batches_by_bucket[b] += 1

    def _stage_rows(self, b: int, parts: Sequence[Tuple[np.ndarray, np.ndarray, int, int]],
                    n: int) -> None:
        """Rows into bucket ``b``'s pinned host buffers, pad rows = copies of row 0, then into the
        static device inputs. Synchronous H2D: the host buffer is reused by the next batch."""
        ho, hm = self._stage[b]
        o_np, m_np = ho.numpy(), hm.numpy()
        for obs, mask, k0, c in parts:
            o_np[k0:k0 + c] = obs
            m_np[k0:k0 + c] = mask
        if n < b:
            o_np[n:b] = o_np[0]
            m_np[n:b] = m_np[0]
        so, sm = self._static[b]
        so.copy_(ho)
        sm.copy_(hm)

    def _run(self, gi: int, i: int, b: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.spec.backend == "graph":
            graph, outs = self._graphs[(gi, i, b)]
            graph.replay()
            return outs[0], outs[1], outs[2]
        with torch.no_grad():
            return _decide(self.groups[gi].modules[i], *self._static[b])

    # ------------------------------------------------------------------ reporting
    def stats(self) -> Dict[str, Any]:
        out: Dict[str, Any] = dict(self.counters)
        out["state"] = self.state
        out["batches_by_bucket"] = dict(self.batches_by_bucket)
        out["slots"] = len(getattr(self, "_slots", ()))
        out["graphs"] = len(getattr(self, "_graphs", {}))
        out["startup_seconds"] = dict(self.startup_seconds)
        return out
