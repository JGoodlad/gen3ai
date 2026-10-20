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
order on its stream, so the rule holds by construction. Different lanes run concurrently, so several
SMALL per-slot batches share the GPU without changing the model. (A single multi-slot forward over the
stacked weights — ``torch.func.vmap`` — was blocked by the extractor's in-place writes into freshly
created tensors, measured 2026-09-29, T2 PROGRESS; X45's GROUPED SLOTS below get past it without touching
the model, ``grouped.py``.)

LEAD SLOTS (gen3_t2_lead_lanes_v1, ``ServiceSpec.lead_slots``; default EMPTY = the plan above). The
declared lead slots (the trainee's) each get a lane of their own (lanes 0..k-1), every other slot
round-robins over the rest in slot order, and step 3 LAUNCHES the lead slots' runs first. Packing (step
1) is unchanged, so every output offset is too; each graph computes the same forward on the same rows.

GROUPED SLOTS (X45, gen3_t2_grouped_forward_v1, ``ServiceSpec.grouped_slots``; default EMPTY = the plan
above). Each slot group's declared grouped slots (one contiguous range) are served by ONE grouped forward
(``grouped.GroupedForward``: vmap over the stacked weights) per ROUND instead of one forward per slot.
Packing (step 1) is unchanged, so every output offset is too. In step 2 every round's row indices are
computed on the host and copied with ONE further host-to-device transfer: a round gathers each slot's
next chunk (<= the range's largest bucket) into the range's static ``[S x b]`` inputs — a slot with fewer
rows repeats its chunk's first row, a slot with none repeats the range's first real row — runs the grouped
forward (eager, or one captured graph per range x bucket on the range's lane), and scatters only the REAL
rows' outputs into the output arenas. Rounds repeat while any slot has rows left.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from agents.inference.service.parity import fixture_rows
from agents.inference.service.slots import SlotGroup
from agents.training.lifecycle_decl import startup_builder

# One request's rows, addressed to a slot: (global slot id, group index, slot-in-group, obs, mask).
PlanItem = Tuple[int, int, int, np.ndarray, np.ndarray]


def lane_plan(n_slots: int, n_lanes: int, lead: Sequence[int] = ()) -> List[int]:
    """The lane of every global slot. No lead slots (or one lane): slot ``s`` on lane ``s % n_lanes`` — the
    default plan. Lead slots (gen3_t2_lead_lanes_v1): the ``j``-th lead slot alone on lane ``j``, every
    other slot, in slot order, round-robin over lanes ``k .. n_lanes - 1`` (``ServiceSpec.validate`` holds
    ``n_lanes > k`` whenever ``n_lanes > 1``)."""
    if not lead or n_lanes <= 1:
        return [s % n_lanes for s in range(n_slots)]
    k = len(lead)
    if n_lanes <= k:
        raise ValueError(f"lane_plan: {k} lead slot(s) need more than {n_lanes} lanes")
    out = [-1] * n_slots
    for j, s in enumerate(lead):
        out[int(s)] = j
    rest = [s for s in range(n_slots) if out[s] < 0]
    for r, s in enumerate(rest):
        out[s] = k + r % (n_lanes - k)
    return out


def launch_order(runs: Sequence[Tuple[int, int, int, int]], ends: Sequence[int],
                 lead: Sequence[int] = ()) -> List[Tuple[Tuple[int, int, int, int], int]]:
    """``(run, end)`` pairs in LAUNCH order: slot order (the packing order), the lead slots' runs moved to
    the front in their declared order (gen3_t2_lead_lanes_v1). No lead slots: exactly ``zip(runs, ends)``."""
    pairs = list(zip(runs, ends))
    if not lead:
        return pairs
    rank = {int(s): j for j, s in enumerate(lead)}
    return sorted(pairs, key=lambda p: (rank.get(int(p[0][0]), len(rank)),))


def decide(module: torch.nn.Module, obs: torch.Tensor, mask: torch.Tensor
           ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The one callable every backend runs (and ``graph`` compiles): logp, V and greedy."""
    logp, value = module(obs, mask)
    return logp, value, logp.argmax(-1)


class Engine:
    # @startup_builder: the engine is built ONCE at startup (T2's build) and its lanes' streams are a
    # declared acquisition — a stream owns allocator cache (`src/learner_lifecycle_gate_test.py`, the
    # cuda_resource kind, gen3_staged_compute_stream_v1).
    @startup_builder
    def __init__(self, *, groups: List[SlotGroup], slots: List[Tuple[int, int]],
                 device: torch.device, buckets: Tuple[int, ...], backend: str, max_rows: int,
                 lanes: int, obs_dim: int, n_actions: int, artifact_dir: Optional[str] = None,
                 slot_caps: Sequence[int] = (), lead_slots: Sequence[int] = (),
                 grouped: Sequence[int] = ()):
        self.groups, self.slots, self.device = groups, slots, device
        self.buckets, self.backend, self.R = buckets, backend, int(max_rows)
        # gen3_slot_bucket_caps_v1: the buckets each slot captures and serves (<= its cap, at least the
        # smallest); no caps = every bucket for every slot.
        caps = [int(c) for c in slot_caps] or [int(buckets[-1])] * len(slots)
        self.slot_buckets: List[Tuple[int, ...]] = [
            tuple(b for b in buckets if b <= c) or (buckets[0],) for c in caps]
        self.cuda = device.type == "cuda"
        self.n_lanes = int(lanes) if self.cuda else 1
        # gen3_t2_lead_lanes_v1: the lead slots launch first, each on its own lane (module docs).
        self.lead: Tuple[int, ...] = tuple(int(s) for s in lead_slots)
        self._lane: List[int] = lane_plan(len(slots), self.n_lanes, self.lead)
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
        # --- grouped slots (X45, gen3_t2_grouped_forward_v1; module docs). EMPTY = OFF: nothing below is used.
        from agents.inference.service.grouped import grouped_ranges
        self.ranges = grouped_ranges(slots, grouped) if grouped else []
        self.grouped_of: Dict[int, int] = {s: r for r, rng in enumerate(self.ranges) for s in rng[3]}
        self.gfwd: List[Any] = []                                   # GroupedForward per range, at build
        self.gstatic: Dict[Tuple[int, int], Tuple[torch.Tensor, torch.Tensor]] = {}
        self.ggraphs: Dict[Tuple[int, int], Tuple[Any, Tuple[torch.Tensor, ...]]] = {}
        self.grouped_rounds = 0
        for r, (gi, lo, hi, ss) in enumerate(self.ranges):
            if len({self.slot_buckets[x] for x in ss}) != 1:
                raise ValueError(f"grouped range {r}: its slots serve different bucket sets")
            n = hi - lo
            for b in self.slot_buckets[ss[0]]:
                o, m = fixture_rows(D, b)
                self.gstatic[(r, b)] = (torch.as_tensor(np.tile(o, (n, 1))).to(device).clone(),
                                        torch.as_tensor(np.tile(m, (n, 1))).to(device).clone())
        # one flush's grouped indices (gather [S x b] per round, then dst / src of the real rows): bounded by
        # (R + cap) x S per range (every round but the last is full) plus 2 R; DOUBLE-BUFFERED like host_in
        n_idx = sum((R + self.slot_buckets[rng[3][0]][-1]) * (rng[2] - rng[1]) for rng in self.ranges) + 2 * R
        self.g_host_idx = [torch.empty(n_idx, dtype=torch.int64, pin_memory=pin) for _ in range(2)
                           ] if self.ranges else []
        self.g_idx_events: List[Optional[Any]] = [None, None]
        self.g_dev_idx = torch.empty(n_idx, dtype=torch.int64, device=device) if self.ranges else None
        self._gflip = 0

    # ------------------------------------------------------------------ build
    def lane_of(self, slot: int) -> int:
        return self._lane[int(slot)]

    def _on(self, lane: int) -> Any:
        import contextlib

        return torch.cuda.stream(self.streams[lane]) if self.cuda else contextlib.nullcontext()

    def range_lane(self, r: int) -> int:
        """A grouped range runs on its FIRST slot's lane."""
        return self.lane_of(self.ranges[r][3][0])

    def range_buckets(self, r: int) -> Tuple[int, ...]:
        return self.slot_buckets[self.ranges[r][3][0]]

    def _build_grouped(self) -> None:
        from agents.inference.service.grouped import GroupedForward

        for r, (gi, lo, hi, _ss) in enumerate(self.ranges):
            go, gm = self.gstatic[(r, self.range_buckets(r)[0])]
            self.gfwd.append(GroupedForward(self.groups[gi], lo, hi, obs=go, mask=gm))

    def _grouped_inputs(self, r: int, b: int) -> Tuple[torch.Tensor, torch.Tensor]:
        n = self.ranges[r][2] - self.ranges[r][1]
        go, gm = self.gstatic[(r, b)]
        return go.view(n, b, self.D), gm.view(n, b, self.A)

    def build(self) -> None:
        self._build_grouped()
        if self.backend == "graph":
            self._build_graphs()
            return
        if self.backend == "aot":
            self._build_packages()
            return
        with torch.no_grad():               # warm every slot x bucket ON ITS LANE'S STREAM (the
            for s, (gi, i) in enumerate(self.slots):          # allocator caches per stream)
                if s in self.grouped_of:
                    continue
                lane = self.lane_of(s)
                with self._on(lane):
                    for b in self.slot_buckets[s]:
                        decide(self.groups[gi].modules[i], *self.static[(lane, b)])
            for r, gf in enumerate(self.gfwd):
                with self._on(self.range_lane(r)):
                    for b in self.range_buckets(r):
                        gf(*self._grouped_inputs(r, b), gf.zeros)
        if self.cuda:
            torch.cuda.synchronize(self.device)

    @startup_builder
    def _build_graphs(self) -> None:
        from agents.model.compile_cache import ensure_hermetic_cache
        ensure_hermetic_cache("T2 graph")         # K3: the run's cache, or a private one — never shared
        from agents.model.inductor_compile import compile_on    # THE compile seam (gen3_cpu_compile_sites_v1)
        compiled = compile_on(decide, device=self.device, dynamic=False)
        pools = [torch.cuda.graph_pool_handle() for _ in range(self.n_lanes)]
        limit = max(int(torch._dynamo.config.cache_size_limit),
                    len(self.buckets) * (len(self.groups) + len(self.ranges)) + 4)
        main = torch.cuda.current_stream(self.device)
        # suppress_errors=False: a compile failure RAISES rather than silently running eager (the
        # eager forward would then fail capture anyway — it makes 24 host syncs — but the typed
        # error names the real cause).
        with torch._dynamo.config.patch(  # type: ignore[attr-defined]
                suppress_errors=False, cache_size_limit=limit), torch.no_grad():
            for b in self.buckets:
                for s, (gi, i) in enumerate(self.slots):
                    if b not in self.slot_buckets[s] or s in self.grouped_of:
                        continue
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
            # X45: one graph per grouped range x bucket — the compiled grouped forward, the weight VIEWS as
            # graph inputs (a load copies into the same storage), captured on the range's lane like the rest
            for r, gf in enumerate(self.gfwd):
                gcomp = compile_on(gf.call_with, device=self.device, dynamic=False, fullgraph=True)
                lane = self.range_lane(r)
                ls = self.streams[lane]
                for b in self.range_buckets(r):
                    go, gm = self._grouped_inputs(r, b)
                    args = (gf.batched, gf.consts, go, gm, gf.zeros)
                    ls.wait_stream(main)
                    with torch.cuda.stream(ls):
                        for _ in range(2):
                            gcomp(*args)
                    main.wait_stream(ls)
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, pool=pools[lane], stream=ls):
                        outs = gcomp(*args)
                    self.captures += 1
                    self.ggraphs[(r, b)] = (graph, tuple(outs))
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
                    for b in self.slot_buckets[s]:
                        self.packages[(gi, b)](*self.static[(self.lane_of(s), b)],
                                               *self.slot_weights[(gi, i)])
        torch.cuda.synchronize(self.device)

    # ------------------------------------------------------------------ serve
    def chunks(self, n: int, slot: Optional[int] = None) -> List[Tuple[int, int]]:
        """``(bucket, rows)`` chunks for ``n`` rows on ``slot``'s buckets (every bucket when None):
        largest buckets, then the smallest that fits."""
        bs = self.buckets if slot is None else self.slot_buckets[int(slot)]
        big, out = bs[-1], []
        while n > big:
            out.append((big, big))
            n -= big
        out.append((next(b for b in bs if b >= n), n))
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
        ends = [r[3] for r in runs[1:]] + [rows]
        # 2b. X45: every grouped round's indices, one more H2D (before any lane waits on the main stream)
        gwork = self._stage_grouped(runs, ends, main) if self.ranges else []
        # 3. per slot, per chunk, on the slot's lane (the lead slots' runs launched first)
        per_bucket: Dict[int, int] = {}
        used = set()
        for (s, gi, i, start), end in launch_order(runs, ends, self.lead):
            if s in self.grouped_of:
                continue                              # served by its range's grouped rounds (3b)
            lane = self.lane_of(s)
            if self.cuda and lane not in used:
                self.streams[lane].wait_stream(main)
                used.add(lane)
            with self._on(lane):
                a = start
                for b, n in self.chunks(end - start, s):
                    self._batch(gi, i, lane, b, a, n)
                    per_bucket[b] = per_bucket.get(b, 0) + 1
                    a += n
        # 3b. X45: the grouped rounds, each on its range's lane
        for r, b, g_off, d_off, s_off, n_real in gwork:
            lane = self.range_lane(r)
            if self.cuda and lane not in used:
                self.streams[lane].wait_stream(main)
                used.add(lane)
            with self._on(lane):
                self._grouped_round(r, b, g_off, d_off, s_off, n_real)
            n_s = self.ranges[r][2] - self.ranges[r][1]
            per_bucket[b] = per_bucket.get(b, 0) + n_s     # S slots x b rows computed (rows_padded stays honest)
            self.grouped_rounds += 1
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

    def _stage_grouped(self, runs: Sequence[Tuple[int, int, int, int]], ends: Sequence[int], main: Any
                       ) -> List[Tuple[int, int, int, int, int, int]]:
        """Plan every grouped round of this flush and copy its indices to the device in ONE transfer.
        Returns ``(range, bucket, gather offset, dst offset, src offset, real rows)`` per round, offsets into
        ``g_dev_idx``: the gather is ``[S x b]`` arena rows (pads repeat a real row), dst the arena rows of the
        round's REAL rows, src their positions in the round's flat ``[S x b]`` output."""
        span: Dict[int, Tuple[int, int]] = {}
        for (s, _gi, _i, start), end in zip(runs, ends):
            if s in self.grouped_of:
                span[s] = (start, end)
        if not span:
            return []
        k = self._gflip
        self._gflip ^= 1
        ev = self.g_idx_events[k]
        if ev is not None:
            ev.synchronize()                   # buffer k's previous H2D has finished reading it
        h = self.g_host_idx[k].numpy()
        plan: List[Tuple[int, int, int, np.ndarray, np.ndarray]] = []
        pos = 0
        for r, (_gi, lo, hi, ss) in enumerate(self.ranges):
            have = [(j, span[s]) for j, s in enumerate(ss) if s in span]
            if not have:
                continue
            bs = self.range_buckets(r)
            cap, n = bs[-1], hi - lo
            pad = have[0][1][0]                # the range's first real row: an idle slot's input
            longest = max(e - a for _, (a, e) in have)
            for rd in range((longest + cap - 1) // cap):
                cnt = {j: min(max(e - a - rd * cap, 0), cap) for j, (a, e) in have}
                b = next(x for x in bs if x >= max(cnt.values()))
                g = h[pos:pos + n * b].reshape(n, b)
                g[:] = pad
                dst, src = [], []
                for j, (a, _e) in have:
                    c = cnt[j]
                    if c <= 0:
                        continue
                    first = a + rd * cap
                    g[j, :c] = np.arange(first, first + c)
                    g[j, c:] = first             # pad rows repeat the chunk's first row
                    dst.append(np.arange(first, first + c))
                    src.append(j * b + np.arange(c))
                plan.append((r, b, pos, np.concatenate(dst), np.concatenate(src)))
                pos += n * b
        out = []
        for r, b, g_off, dst, src in plan:
            m = len(dst)
            h[pos:pos + m], h[pos + m:pos + 2 * m] = dst, src
            out.append((r, b, g_off, pos, pos + m, m))
            pos += 2 * m
        assert self.g_dev_idx is not None
        self.g_dev_idx[:pos].copy_(self.g_host_idx[k][:pos], non_blocking=self.cuda)
        if self.cuda:
            e = torch.cuda.Event()
            e.record(main)
            self.g_idx_events[k] = e
        return out

    def _grouped_round(self, r: int, b: int, g_off: int, d_off: int, s_off: int, n_real: int) -> None:
        """Gather the round's ``[S x b]`` rows into the range's static inputs, run the grouped forward, scatter
        the REAL rows' outputs into the arenas."""
        assert self.g_dev_idx is not None
        n = self.ranges[r][2] - self.ranges[r][1]
        go, gm = self.gstatic[(r, b)]
        gidx = self.g_dev_idx[g_off:g_off + n * b]
        torch.index_select(self.dev_obs, 0, gidx, out=go)
        torch.index_select(self.dev_mask, 0, gidx, out=gm)
        if self.backend == "graph":
            graph, outs = self.ggraphs[(r, b)]
            graph.replay()
        else:
            gf = self.gfwd[r]
            with torch.no_grad():
                outs = gf(go.view(n, b, self.D), gm.view(n, b, self.A), gf.zeros)
        dst = self.g_dev_idx[d_off:d_off + n_real]
        src = self.g_dev_idx[s_off:s_off + n_real]
        self.out_logp.index_copy_(0, dst, outs[0].reshape(n * b, self.A).index_select(0, src))
        self.out_value.index_copy_(0, dst, outs[1].reshape(n * b).index_select(0, src))
        self.out_greedy.index_copy_(0, dst, outs[2].reshape(n * b).index_select(0, src))

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
