"""K8 — the DEVICE-RESIDENT MICRO-BATCH (`gen3_device_batches_v1`, M5 Lane K8 fix #4).

The learner's `RolloutBuffer.get(batch_size)` (`agents/training/rollout_buffer.py`, ours since deletion pass
U4; sb3-contrib's before, with the same `get`) draws ONE `np.random.permutation` per epoch and builds every
micro-batch with `_get_samples(indices)`: numpy fancy indexing over the flattened host arrays, then
`th.tensor(...)` — a host gather and an H2D copy of every field, every micro-batch, every epoch
(10 epochs x 48 micro-batches of a 2,761-float observation at production shape: ~11 GB of H2D per
update, and the K8 inventory's `batch` phase, 2.95 s of a 53 s update).

`install(buffer)` (at the top of `train()`; `uninstall` after its epoch loop) puts an INSTANCE-level
`_get_samples` on the buffer that gathers from a copy of the flattened arrays made ONCE on the
buffer's device (where the buffer's `to_torch` puts every micro-batch) at the first micro-batch (after the buffer's own `get()` flattened them). Everything else is the buffer's own `get()`: the SAME
permutation draw from the SAME global numpy RNG, the same slicing into micro-batches, the same
`RolloutSamples` type — and a gather is exact, so every micro-batch is
BIT-IDENTICAL to the host path's (`device_batches_test`). `uninstall` removes the instance
attribute and drops the device copy, and `install` always replaces a previous gather (so an update
that raised mid-loop cannot leave a stale copy serving the next), so the learner's quiescent CUDA floor between updates (what
K6's memory trend reads) is unchanged; the update's PEAK grows by the copy (~1.1 GB at production
shape).

Only the learner's `RolloutBuffer` on a CUDA device is served (``force`` is the CPU test seam); anything
else — a CPU buffer — keeps the host path untouched.
"""
from __future__ import annotations

import contextlib
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch

from agents.training.rollout_buffer import RolloutBuffer, RolloutSamples

#: The flat per-row arrays `_get_samples` reads beside the observation dict, and how it shapes each.
_FLAT = RolloutBuffer.FLAT


def _servable(buffer: Any) -> bool:
    return isinstance(buffer, RolloutBuffer) and all(t in buffer.__dict__ for t in _FLAT)


class _DeviceGather:
    """The instance `_get_samples`: the device copy is made at the first call (the arrays are
    flattened by then) and reused for every later micro-batch of the update."""

    def __init__(self, buffer: Any, device: torch.device) -> None:
        self.buffer = buffer
        self.device = device
        self.obs: Optional[Dict[str, torch.Tensor]] = None
        self.flat: Dict[str, torch.Tensor] = {}
        self.copies = 0
        self.nbytes = 0                                   # the device copy's size (gen3_cuda_ledger_v1)

    def _materialise(self) -> None:
        b = self.buffer
        # `th.tensor(array)` — what the buffer's `to_torch` does — keeps the numpy dtype; so does this.
        self.obs = {k: torch.as_tensor(np.ascontiguousarray(v)).to(self.device)
                    for k, v in b.observations.items()}
        self.flat = {t: torch.as_tensor(np.ascontiguousarray(b.__dict__[t])).to(self.device)
                     for t in _FLAT}
        self.copies += 1
        self.nbytes = (sum(int(v.numel()) * v.element_size() for v in self.obs.values())
                       + sum(int(v.numel()) * v.element_size() for v in self.flat.values()))

    def __call__(self, batch_inds: np.ndarray, env: Any = None) -> RolloutSamples:
        if self.obs is None:
            self._materialise()
        assert self.obs is not None
        idx = torch.as_tensor(np.asarray(batch_inds, dtype=np.int64)).to(self.device)
        f = self.flat
        return RolloutSamples(
            observations={k: v.index_select(0, idx) for k, v in self.obs.items()},
            actions=f["actions"].index_select(0, idx),
            old_values=f["values"].index_select(0, idx).flatten(),
            old_log_prob=f["log_probs"].index_select(0, idx).flatten(),
            advantages=f["advantages"].index_select(0, idx).flatten(),
            returns=f["returns"].index_select(0, idx).flatten(),
            action_masks=f["action_masks"].index_select(0, idx).reshape(-1, self.buffer.mask_dims),
        )


class _StagedGather:
    """Mode ``staged`` (gen3_device_batch_mode_v1): each micro-batch is gathered on the HOST by one
    prefetch thread (``np.take`` into a pinned block from torch's caching host allocator, ``lookahead``
    micro-batches ahead) and copied to the device, non-blocking, on the COMPUTE stream when the update
    asks for it. On the card at any moment: the micro-batch in use plus the one being copied (~2 x 22 MB
    at production shape), instead of the whole flattened buffer (~1.1 GB at 98k rows). The rows are
    the SAME rows from the SAME permutation (the buffer's own ``get`` is run unchanged — its
    `_get_samples` hands back the index slice, this class gathers it), and a gather is exact, so every
    micro-batch is BIT-IDENTICAL to the host path's and the resident path's
    (`instrumented_ppo_device_batches_test`).

    🚨 NO SIDE STREAM (`gen3_staged_compute_stream_v1`, 2026-10-01). The copy used to run on a side
    stream built HERE, i.e. a NEW ``torch.cuda.Stream`` every update — drawn round-robin from torch's
    pool of 32 per priority. Under ``expandable_segments:True`` (the launcher's allocator mode) the
    caching allocator keeps one segment PER STREAM and a cached block serves only its own stream, so
    each new stream stranded its in-flight micro-batch blocks (+46-66 MiB) until the pool wrapped:
    sizing arm B (N = 256) climbed 7,814 -> 9,674 MiB reserved over exactly 31 updates + the startup
    dry update = 32, then went flat, with D-6 failing at steady state. The side stream bought no
    overlap anyway: it waited on the compute stream before every copy and the compute stream waited
    on it after, so the order of work is the same on one stream. On the compute stream the blocks come
    from the compute stream's own cache, reused every micro-batch, and the startup update fit check
    (`update_fit`) sees the steady state. A stream is a declared STARTUP acquisition
    (`src/learner_lifecycle_gate_test.py`, the ``cuda_resource`` kind)."""

    def __init__(self, buffer: Any, device: torch.device, lookahead: int = 2) -> None:
        from concurrent.futures import ThreadPoolExecutor

        self.buffer = buffer
        self.device = device
        self.cuda = device.type == "cuda"
        self.lookahead = max(1, int(lookahead))
        self.pool: Any = ThreadPoolExecutor(1, thread_name_prefix="devb-stage")
        self.copies = 0                                   # micro-batches staged this update
        self.nbytes = 0                                   # the most bytes staged on the card at once

    def indices(self, batch_inds: np.ndarray, env: Any = None) -> np.ndarray:
        """Installed as the buffer's `_get_samples`: its own ``get`` yields the index slices."""
        return np.array(batch_inds, dtype=np.int64, copy=True)

    def _host(self, inds: np.ndarray) -> Any:
        b = self.buffer

        def take(a: np.ndarray) -> torch.Tensor:
            a = np.asarray(a)
            out = torch.empty((len(inds),) + tuple(a.shape[1:]), dtype=torch.from_numpy(a[:0]).dtype,
                              pin_memory=self.cuda)
            np.take(a, inds, axis=0, out=out.numpy())
            return out
        return ({k: take(v) for k, v in b.observations.items()}, {t: take(b.__dict__[t]) for t in _FLAT})

    def _serve(self, fut: Any) -> RolloutSamples:
        obs, f = fut.result()
        if self.cuda:
            # on the CURRENT (compute) stream: its cache serves every micro-batch of every update
            # (the class docs: a per-update side stream stranded its blocks, gen3_staged_compute_stream_v1)
            obs = {k: v.to(self.device, non_blocking=True) for k, v in obs.items()}
            f = {k: v.to(self.device, non_blocking=True) for k, v in f.items()}
        n = (sum(int(v.numel()) * v.element_size() for v in obs.values())
             + sum(int(v.numel()) * v.element_size() for v in f.values()))
        self.nbytes = max(self.nbytes, 2 * n)
        self.copies += 1
        return RolloutSamples(
            observations=obs, actions=f["actions"], old_values=f["values"].flatten(),
            old_log_prob=f["log_probs"].flatten(), advantages=f["advantages"].flatten(),
            returns=f["returns"].flatten(),
            action_masks=f["action_masks"].reshape(-1, self.buffer.mask_dims))

    def get(self, batch_size: Optional[int] = None) -> Iterator[Any]:
        """Installed as the buffer's `get`: the class's own generator (the permutation draw, the
        flatten, the slicing), its index slices gathered ``lookahead`` ahead."""
        import collections

        pend: Any = collections.deque()
        for inds in type(self.buffer).get(self.buffer, batch_size):
            pend.append(self.pool.submit(self._host, inds))
            if len(pend) > self.lookahead:
                yield self._serve(pend.popleft())
        while pend:
            yield self._serve(pend.popleft())

    def close(self) -> None:
        self.pool.shutdown(wait=True)


#: How the update's micro-batches reach the learner's device (gen3_device_batch_mode_v1). Every mode
#: serves BIT-IDENTICAL micro-batches from the same permutation draw; they differ only in memory and
#: time. ``resident`` = one device copy of the whole flattened buffer per update (K8.6, +~1.1 GB of
#: update peak at 98k rows); ``staged`` = a prefetch thread gathers each micro-batch on the host and it
#: is copied non-blocking on the compute stream (~2 micro-batches on the card); ``host`` = the buffer's own
#: per-micro-batch host gather + blocking copy.
MODES = ("resident", "staged", "host")
DEFAULT_MODE = "staged"


def install(buffer: Any, *, force: bool = False, mode: str = DEFAULT_MODE) -> Any:
    """Serve ``buffer``'s micro-batches by ``mode`` (`MODES`) until `uninstall`. Always replaces any
    previous gather first, so a stale copy can never serve a later update. Returns the gather (None
    when the host path is kept: mode ``host``, not the learner's `RolloutBuffer`, or a CPU buffer without
    ``force``)."""
    uninstall(buffer)
    if mode not in MODES:
        raise ValueError(f"device batch mode {mode!r} is not one of {MODES}")
    dev = torch.device(getattr(buffer, "device", "cpu"))
    if mode == "host" or not (_servable(buffer) and (dev.type == "cuda" or force)):
        return None
    if mode == "staged":
        staged = _StagedGather(buffer, dev)
        buffer._get_samples = staged.indices
        buffer.get = staged.get
        buffer.__dict__["_devb_staged"] = staged
        return staged
    gather = _DeviceGather(buffer, dev)
    buffer._get_samples = gather
    return gather


def uninstall(buffer: Any) -> None:
    """Back to the buffer's own `get` / `_get_samples`; a device copy is dropped, a stager closed."""
    g = buffer.__dict__.pop("_get_samples", None)
    if isinstance(g, _DeviceGather):
        g.obs, g.flat = None, {}
    staged = buffer.__dict__.pop("_devb_staged", None)
    buffer.__dict__.pop("get", None)
    if staged is not None:
        staged.close()


@contextlib.contextmanager
def device_samples(buffer: Any, *, force: bool = False, mode: str = "resident") -> Iterator[Any]:
    """`install` for the body of the context, `uninstall` on the way out (the tests' form)."""
    gather = install(buffer, force=force, mode=mode)
    try:
        yield gather
    finally:
        uninstall(buffer)
