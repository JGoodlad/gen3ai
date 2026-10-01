"""K8 — the DEVICE-RESIDENT MICRO-BATCH (`gen3_device_batches_v1`, M5 Lane K8 fix #4).

sb3's `RolloutBuffer.get(batch_size)` draws ONE `np.random.permutation` per epoch and builds every
micro-batch with `_get_samples(indices)`: numpy fancy indexing over the flattened host arrays, then
`th.tensor(...)` — a host gather and an H2D copy of every field, every micro-batch, every epoch
(10 epochs x 48 micro-batches of a 2,761-float observation at production shape: ~11 GB of H2D per
update, and the K8 inventory's `batch` phase, 2.95 s of a 53 s update).

`install(buffer)` (at the top of `train()`; `uninstall` after its epoch loop) puts an INSTANCE-level
`_get_samples` on the buffer that gathers from a copy of the flattened arrays made ONCE on the
buffer's device (where sb3's `to_torch` puts every micro-batch) at the first micro-batch (after the buffer's own `get()` flattened them, and after
a fork buffer concatenated its extra rows). Everything else is the buffer's own `get()`: the SAME
permutation draw from the SAME global numpy RNG, the same slicing into micro-batches, the same
`MaskableDictRolloutBufferSamples` type — and a gather is exact, so every micro-batch is
BIT-IDENTICAL to the host path's (`device_batches_test`). `uninstall` removes the instance
attribute and drops the device copy, and `install` always replaces a previous gather (so an update
that raised mid-loop cannot leave a stale copy serving the next), so the learner's quiescent CUDA floor between updates (what
K6's memory trend reads) is unchanged; the update's PEAK grows by the copy (~1.1 GB at production
shape).

Only a buffer of sb3-contrib's maskable dict layout on a CUDA device is served (``force`` is the
CPU test seam); anything else — a CPU buffer, the rust core's host-side buffer — keeps the host path
untouched.
"""
from __future__ import annotations

import contextlib
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch

#: The flat per-row arrays `_get_samples` reads beside the observation dict, and how it shapes each.
_FLAT = ("actions", "values", "log_probs", "advantages", "returns", "action_masks")


def _servable(buffer: Any) -> bool:
    try:
        from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBuffer
    except Exception:                                    # pragma: no cover — sb3-contrib is a dependency
        return False
    return isinstance(buffer, MaskableDictRolloutBuffer) and all(t in buffer.__dict__ for t in _FLAT)


class _DeviceGather:
    """The instance `_get_samples`: the device copy is made at the first call (the arrays are
    flattened by then) and reused for every later micro-batch of the update."""

    def __init__(self, buffer: Any, device: torch.device) -> None:
        self.buffer = buffer
        self.device = device
        self.obs: Optional[Dict[str, torch.Tensor]] = None
        self.flat: Dict[str, torch.Tensor] = {}
        self.copies = 0

    def _materialise(self) -> None:
        b = self.buffer
        # `th.tensor(array)` — what sb3's `to_torch` does — keeps the numpy dtype; so does this.
        self.obs = {k: torch.as_tensor(np.ascontiguousarray(v)).to(self.device)
                    for k, v in b.observations.items()}
        self.flat = {t: torch.as_tensor(np.ascontiguousarray(b.__dict__[t])).to(self.device)
                     for t in _FLAT}
        self.copies += 1

    def __call__(self, batch_inds: np.ndarray, env: Any = None) -> Any:
        from sb3_contrib.common.maskable.buffers import MaskableDictRolloutBufferSamples
        if self.obs is None:
            self._materialise()
        assert self.obs is not None
        idx = torch.as_tensor(np.asarray(batch_inds, dtype=np.int64)).to(self.device)
        f = self.flat
        return MaskableDictRolloutBufferSamples(
            observations={k: v.index_select(0, idx) for k, v in self.obs.items()},
            actions=f["actions"].index_select(0, idx),
            old_values=f["values"].index_select(0, idx).flatten(),
            old_log_prob=f["log_probs"].index_select(0, idx).flatten(),
            advantages=f["advantages"].index_select(0, idx).flatten(),
            returns=f["returns"].index_select(0, idx).flatten(),
            action_masks=f["action_masks"].index_select(0, idx).reshape(-1, self.buffer.mask_dims),
        )


def install(buffer: Any, *, force: bool = False) -> Optional[_DeviceGather]:
    """Serve ``buffer``'s micro-batches from a copy on the BUFFER's own device (where sb3's
    `to_torch` puts them) until `uninstall`. Always replaces any previous gather first, so a stale
    copy can never serve a later update. Returns the gather (None when the host path is kept: not a
    maskable dict buffer, or a CPU buffer without ``force``)."""
    uninstall(buffer)
    dev = torch.device(getattr(buffer, "device", "cpu"))
    if not (_servable(buffer) and (dev.type == "cuda" or force)):
        return None
    gather = _DeviceGather(buffer, dev)
    buffer._get_samples = gather
    return gather


def uninstall(buffer: Any) -> None:
    """Back to the buffer's own `_get_samples`; the device copy is dropped."""
    g = buffer.__dict__.pop("_get_samples", None)
    if isinstance(g, _DeviceGather):
        g.obs, g.flat = None, {}


@contextlib.contextmanager
def device_samples(buffer: Any, *, force: bool = False) -> Iterator[Optional[_DeviceGather]]:
    """`install` for the body of the context, `uninstall` on the way out (the tests' form)."""
    gather = install(buffer, force=force)
    try:
        yield gather
    finally:
        uninstall(buffer)
