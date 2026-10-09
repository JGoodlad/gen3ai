"""`read_floats` — many 0-d device scalars read to the host in ONE transfer (`gen3_batched_host_reads_v1`).

The diagnostics probes (`grad_balance`, the per-term noise sampler) used to read every norm with its own
``float(t)`` — on CUDA one host sync (a full GPU-queue drain) each. This stacks them and reads once. A
float32 value read out of the stacked copy is the value ``float(t)`` gave, so every logged number is
bit-identical; tensors that cannot share one stack (mixed dtypes or devices) are read one by one, as before.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import torch as th


def read_floats(tensors: Sequence[Optional[th.Tensor]], none: float = 0.0) -> List[float]:
    """``[float(t) for t in tensors]`` (``none`` for a None entry) with ONE device->host transfer."""
    live = [t for t in tensors if t is not None]
    if not live:
        return [none] * len(tensors)
    same = all(t.dtype == live[0].dtype and t.device == live[0].device and t.numel() == 1 for t in live)
    if same:
        host = th.stack([t.detach().reshape(()) for t in live]).cpu().tolist()
    else:
        host = [float(t) for t in live]
    it = iter(host)
    return [none if t is None else float(next(it)) for t in tensors]
