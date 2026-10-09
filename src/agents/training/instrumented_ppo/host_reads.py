"""THE UPDATE'S DEFERRED HOST READS (`gen3_batched_host_reads_v1`, T25 item 1).

On CUDA every device->host read (`.cpu()`, `.item()`, `float(t)`, ...) blocks the host main thread until
the GPU queue drains to it. The 2026-10-03 bottleneck profile counted ~764 such reads per production
update (`designs/research_state/measurements/bottleneck_profile_2026-10-03/`), each waiting on a ~22 ms
deep queue: ~11 s of a 40 s update spent spin-blocked, and the queue kept shallow behind every one. Most
of them read values nobody needs until LATER: a micro-batch's metrics (read at the end of the update), its
loss's finiteness (needed before the OPTIMIZER STEP it joins, not before its own backward), epoch 0's
calibration rows (folded into bin counts at the end).

`HostReadQueue` holds those reads as DEVICE tensors and makes them in ONE transfer at the next point the
host has to read anyway — the optimizer step's pre-clip gradient norm (`learner_gates`), or a micro-batch
whose own values ARE needed at once (`ppo.train`'s immediate path). Each queued read carries a callback
that does exactly what the inline read-and-use did, on the same float32 values in the same order, so every
logged number, every list the metrics export averages and every K9(c) verdict is bit-identical; only WHEN
the host blocks moves.

Rules a caller must keep:

* every queued tensor is float32 and on ONE device (one ``cat``, one copy; a float32 slice of the host
  copy is the value ``.cpu()`` of that tensor alone would have given);
* a queued tensor must be a fresh tensor the caller owns (a ``cat`` / ``stack`` / ``reshape`` of a value
  that is not mutated later) — the read happens after more of the update has run;
* callbacks run in PUSH order, so a list that a callback appends to grows in the order the inline code
  appended to it; a callback that raises (a K9(c) FATAL) stops the drain, and the update with it.
"""
from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import torch as th

Apply = Callable[[np.ndarray], None]


class HostReadQueue:
    """Pending device->host reads of one `train()` call (module docstring)."""

    __slots__ = ("_items", "drains", "reads")

    def __init__(self) -> None:
        self._items: List[Tuple[th.Tensor, Apply]] = []
        #: how many transfers this queue made, and how many queued reads they carried (telemetry / tests)
        self.drains = 0
        self.reads = 0

    def __len__(self) -> int:
        return len(self._items)

    def push(self, flat: th.Tensor, apply: Apply) -> None:
        """Queue ``flat`` (float32; flattened here) and the host-side ``apply(values)`` that consumes it."""
        if flat.dtype != th.float32:
            raise TypeError(f"HostReadQueue holds float32 reads only, got {flat.dtype}: a cast in the "
                            "shared transfer would change the value the inline read gave")
        self._items.append((flat.detach().reshape(-1), apply))

    def drain(self, extra: Sequence[th.Tensor] = ()) -> Optional[np.ndarray]:
        """ONE device->host transfer of every queued read plus ``extra`` (float32 0-d / 1-d tensors read
        for the caller); applies each queued callback in push order; returns ``extra``'s values (a
        float32 host array), or None when ``extra`` is empty. No queued read and no ``extra``: no
        transfer at all."""
        items, self._items = self._items, []
        ex = [e.detach().reshape(-1) for e in extra]
        for e in ex:
            if e.dtype != th.float32:
                raise TypeError(f"HostReadQueue.drain extra must be float32, got {e.dtype}")
        parts = ex + [f for f, _ in items]
        if not parts:
            return None
        host = th.cat(parts).cpu().numpy() if len(parts) > 1 else parts[0].cpu().numpy()
        self.drains += 1
        self.reads += len(items)
        n_ex = sum(int(e.numel()) for e in ex)
        off = n_ex
        for flat, apply in items:
            n = int(flat.numel())
            apply(host[off:off + n])
            off += n
        return host[:n_ex] if ex else None
