"""The service's EXECUTION ENGINE: arenas, staging, lanes and the per-slot x bucket forwards.

Everything here is allocated in ``__init__`` / ``build()`` (startup) and only reused afterwards.
``execute(plan)`` serves one flush:

1. **Pack** every admitted request's rows into the flush's PINNED host input arena, grouped by slot
   (so each slot's rows are contiguous and so are its outputs). The host arena is DOUBLE-BUFFERED
   (M5 T2 unit 4): flush k packs arena k % 2 while flush k-1's host-to-device copy may still be in
   flight from the other one; before re-packing an arena the engine waits on the event recorded
   after that arena's last copy, so a still-in-flight copy can never read half-written rows.
2. **One** asynchronous host-to-device copy of the whole flush into the device input arena.
3. Per slot, per bucket chunk, on the slot's LANE stream: device-to-device copies into that lane's
   static bucket inputs (pad rows repeat the chunk's first row), a graph replay (or an eager
   forward), and a device-to-device copy of the outputs into the output arena.
4. The caller's stream waits for every lane used, then **one** asynchronous device-to-host copy of
   the outputs into the pinned host output arena, and an event the host readers wait on.

LANES (M5 T2 unit 3). A lane is a CUDA stream with its OWN graph memory pool and its OWN static
inputs; slot ``s`` is bound to lane ``s % lanes``. Graphs sharing a pool must never run
concurrently (they share intermediate buffers) — one pool per lane, graphs within a lane run in
order on its stream, so the rule holds by construction. Different lanes run concurrently, which
is the only way several SMALL per-slot batches can share the GPU without changing the model (a
single multi-slot forward over the stacked weights — ``torch.func.vmap`` — is blocked by the
extractor's in-place writes into freshly created tensors; measured 2026-09-29, T2 PROGRESS).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from agents.inference.service.parity import fixture_rows
from agents.inference.service.slots import SlotGroup

# One request's rows, addressed to a slot: (global slot id, group index, slot-in-group, obs, mask).
PlanItem = Tuple[int, int, int, np.ndarray, np.ndarray]


def decide(module: torch.nn.Module, obs: torch.Tensor, mask: torch.Tensor
           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The one callable every backend runs (and ``graph`` compiles): logp, V and greedy."""
    logp, value = module(obs, mask)
    return logp, value, logp.argmax(-1)


class Engine:
    def __init__(self, *, groups: List[SlotGroup], slots: List[Tuple[int, int]],
                 device: torch.device, buckets: Tuple[int, ...], backend: str, max_rows: int,
                 lanes: int, obs_dim: int, n_actions: int, artifact_dir: Optional[str] = None):
        self.groups, self.slots, self.device = groups, slots, device
        self.buckets, self.backend, self.R = buckets, backend, int(max_rows)
        self.cuda = device.type == "cuda"
        self.n_lanes = int(lanes) if self.cuda else 1
        R, D, A = self.R, int(obs_dim), int(n_actions)
        self.D, self.A = D, A
        pin = self.cuda
        # --- host staging: double-buffered pinned input arenas + a pinned output arena
        self.host_in = [(torch.empty(R, D, dtype=torch.float32, pin_memory=pin),
                         torch.empty(R, A, dtype=torch.bool, pin_memory=pin)) for _ in range(2)]
        self.host_out = (torch.empty(R, A, dtype=torch.float32, pin_memory=pin),
                         torch.empty(R, dtype=torch.float32, pin_memory=pin),
                         torch.empty(R, dtype=torch.int64, pin_memory=pin))
        self.in_events: List[Optional[Any]] = [None, None]
        self.out_event: Optional[Any] = torch.cuda.Event() if self.cuda else None
        self._flip = 0
        # --- device arenas
        self.dev_obs = torch.zeros(R, D, dtype=torch.float32, device=device)
        self.dev_mask = torch.ones(R, A, dtype=torch.bool, device=device)
        self.out_logp = torch.empty(R, A, dtype=torch.float32, device=device)
        self.out_value = torch.empty(R, dtype=torch.float32, device=device)
        self.out_greedy = torch.empty(R, dtype=torch.int64, device=device)
        # --- lanes: a stream + static bucket inputs each (+ a graph pool, at build)
        self.streams = [torch.cuda.Stream(device) for _ in range(self.n_lanes)] if self.cuda else []
        self.static: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        for lane in range(self.n_lanes):
            for b in buckets:
                o, m = fixture_rows(D, b)
                self.static[(lane, b)] = (torch.as_tensor(o).to(device).clone(),
                                          torch.as_tensor(m).to(device).clone())
        self.graphs: Dict[Tuple[int, int, int], Tuple[Any, Tuple[torch.Tensor, ...]]] = {}
        self.captures = 0
        # backend 'aot': one package per (group, bucket); per slot, its weight-input tensors
        self.artifact_dir = artifact_dir
        self.packages: Dict[Tuple[int, int], Any] = {}
        self.slot_weights: Dict[Tuple[int, int], List[torch.Tensor]] = {}

    # ------------------------------------------------------------------ build
    def lane_of(self, slot: int) -> int:
        return int(slot) % self.n_lanes

    def _on(self, lane: int) -> Any:
        import contextlib

        return torch.cuda.stream(self.streams[lane]) if self.cuda else contextlib.nullcontext()

    def build(self) -> None:
        if self.backend == "graph":
            self._build_graphs()
            return
        if self.backend == "aot":
            self._build_packages()
            return
        with torch.no_grad():               # warm every slot x bucket ON ITS LANE'S STREAM (the
            for s, (gi, i) in enumerate(self.slots):          # allocator caches per stream)
                lane = self.lane_of(s)
                with self._on(lane):
                    for b in self.buckets:
                        decide(self.groups[gi].modules[i], *self.static[(lane, b)])
        if self.cuda:
            torch.cuda.synchronize(self.device)

    def _build_graphs(self) -> None:
        from agents.model.compile_cache import ensure_hermetic_cache
        ensure_hermetic_cache("T2 graph")         # K3: the run's cache, or a private one — never shared
        compiled = torch.compile(decide, dynamic=False)
        pools = [torch.cuda.graph_pool_handle() for _ in range(self.n_lanes)]
        limit = max(int(torch._dynamo.config.cache_size_limit),
                    len(self.buckets) * len(self.groups) + 4)
        main = torch.cuda.current_stream(self.device)
        # suppress_errors=False: a compile failure RAISES rather than silently running eager (the
        # eager forward would then fail capture anyway — it makes 24 host syncs — but the typed
        # error names the real cause).
        with torch._dynamo.config.patch(  # type: ignore[attr-defined]
                suppress_errors=False, cache_size_limit=limit), torch.no_grad():
            for b in self.buckets:
                for s, (gi, i) in enumerate(self.slots):
                    lane = self.lane_of(s)
                    so, sm = self.static[(lane, b)]
                    mod = self.groups[gi].modules[i]
                    # Warm up AND capture on the LANE's own stream. torch.cuda.graph's default
                    # capture stream is ONE class-level stream, and cuBLAS keys its workspace by
                    # (handle, stream) — so graphs captured there all bake in the SAME workspace,
                    # and two of them replaying concurrently on different lanes overwrite each
                    # other's matmul scratch (measured 2026-09-29: max|dlogp| 0.048 with 2 lanes).
                    ls = self.streams[lane]
                    ls.wait_stream(main)
                    with torch.cuda.stream(ls):
                        for _ in range(2):
                            compiled(mod, so, sm)
                    main.wait_stream(ls)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=pools[lane], stream=ls):
                        outs = compiled(mod, so, sm)
                    self.captures += 1
                    self.graphs[(gi, i, b)] = (graph, tuple(outs))
        torch.cuda.synchronize(self.device)

    def _build_packages(self) -> None:
        from agents.inference.service import aot

        root = aot.artifact_dir(self.artifact_dir)
        for gi, g in enumerate(self.groups):
            names: List[str] = []
            for b in self.buckets:
                so, sm = self.static[(0, b)]
                runner, names = aot.build_package(g.modules[0], so, sm,
                                                  root / f"{g.name}_b{b}.pt2")
                self.packages[(gi, b)] = runner
            for i, mod in enumerate(g.modules):
                # the replica's own tensors ARE views into the group's stacked storage, so a load
                # (an in-place copy) is what the next call reads — nothing to refresh.
                self.slot_weights[(gi, i)] = aot.weight_tensors(mod, names)
        with torch.no_grad():                # warm every slot x bucket on its lane (allocator)
            for s, (gi, i) in enumerate(self.slots):
                with self._on(self.lane_of(s)):
                    for b in self.buckets:
                        self.packages[(gi, b)](*self.static[(self.lane_of(s), b)],
                                               *self.slot_weights[(gi, i)])
        torch.cuda.synchronize(self.device)

    # ------------------------------------------------------------------ serve
    def chunks(self, n: int) -> List[Tuple[int, int]]:
        """``(bucket, rows)`` chunks for ``n`` rows: largest buckets, then the smallest that fits."""
        big, out = self.buckets[-1], []
        while n > big:
            out.append((big, big))
            n -= big
        out.append((next(b for b in self.buckets if b >= n), n))
        return out

    def execute(self, plan: Sequence[PlanItem]) -> Tuple[List[int], Dict[int, int]]:
        """Serve ``plan`` (one flush). Returns each item's row offset in the output arenas and the
        batches run per bucket."""
        # 1. pack, grouped by slot (stable within a slot)
        order = sorted(range(len(plan)), key=lambda k: (plan[k][0], k))
        offsets = [0] * len(plan)
        k = self._flip
        self._flip ^= 1
        ev = self.in_events[k]
        if ev is not None:
            ev.synchronize()               # arena k's previous H2D has finished reading it
        h_obs, h_mask = self.host_in[k]
        o_np, m_np = h_obs.numpy(), h_mask.numpy()
        off = 0
        runs: List[Tuple[int, int, int, int]] = []           # (slot, gi, i, start) per slot run
        for idx in order:
            s, gi, i, obs, mask = plan[idx]
            n = obs.shape[0]
            o_np[off:off + n] = obs
            m_np[off:off + n] = mask
            offsets[idx] = off
            if not runs or runs[-1][0] != s:
                runs.append((s, gi, i, off))
            off += n
        rows = off
        # 2. one H2D of the whole flush
        main: Any = torch.cuda.current_stream(self.device) if self.cuda else None
        self.dev_obs[:rows].copy_(h_obs[:rows], non_blocking=self.cuda)
        self.dev_mask[:rows].copy_(h_mask[:rows], non_blocking=self.cuda)
        if self.cuda:
            ev = torch.cuda.Event()
            ev.record(main)
            self.in_events[k] = ev
        # 3. per slot, per chunk, on the slot's lane
        ends = [r[3] for r in runs[1:]] + [rows]
        per_bucket: Dict[int, int] = {}
        used = set()
        for (s, gi, i, start), end in zip(runs, ends):
            lane = self.lane_of(s)
            if self.cuda and lane not in used:
                self.streams[lane].wait_stream(main)
                used.add(lane)
            with self._on(lane):
                a = start
                for b, n in self.chunks(end - start):
                    self._batch(gi, i, lane, b, a, n)
                    per_bucket[b] = per_bucket.get(b, 0) + 1
                    a += n
        # 4. join the lanes, one D2H of the outputs
        if self.cuda:
            for lane in used:
                main.wait_stream(self.streams[lane])
        for dst, src in zip(self.host_out, (self.out_logp, self.out_value, self.out_greedy)):
            dst[:rows].copy_(src[:rows], non_blocking=self.cuda)
        if self.cuda:
            assert self.out_event is not None
            self.out_event.record(main)
        return offsets, per_bucket

    def _batch(self, gi: int, i: int, lane: int, b: int, a: int, n: int) -> None:
        so, sm = self.static[(lane, b)]
        so[:n].copy_(self.dev_obs[a:a + n])
        sm[:n].copy_(self.dev_mask[a:a + n])
        if n < b:                                     # pad rows repeat the chunk's first row
            so[n:b].copy_(self.dev_obs[a:a + 1].expand(b - n, self.D))
            sm[n:b].copy_(self.dev_mask[a:a + 1].expand(b - n, self.A))
        if self.backend == "graph":
            graph, outs = self.graphs[(gi, i, b)]
            graph.replay()
        elif self.backend == "aot":
            with torch.no_grad():
                outs = self.packages[(gi, b)](so, sm, *self.slot_weights[(gi, i)])
        else:
            with torch.no_grad():
                outs = decide(self.groups[gi].modules[i], so, sm)
        self.out_logp[a:a + n].copy_(outs[0][:n])
        self.out_value[a:a + n].copy_(outs[1][:n])
        self.out_greedy[a:a + n].copy_(outs[2][:n])

    def wait_host(self) -> None:
        """Block until the last flush's host outputs are readable."""
        if self.out_event is not None:
            self.out_event.synchronize()
