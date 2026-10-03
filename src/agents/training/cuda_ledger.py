"""The CUDA MEMORY LEDGER (`gen3_cuda_ledger_v1`, 2026-10-01) — where the learner process's card goes.

The sizing study measured production (N = 48, torch 2.8, fp32, the Rust env core) reserving 9.49 GiB
of the 12 GiB card with 1.2 GiB free — under K6's own ceiling margin — and ~3.9 GiB of it unattributed.
This ledger attributes it, in EVERY run's log, at the startup steps that acquire device memory:

    weights on the card -> rust env core (T2 slots, staging, arena) -> the compiled regions' gate +
    prewarm -> optimizer state declared

Each row is the allocator's ALLOCATED and RESERVED bytes after the step, the step's delta, and the
PEAK allocated during it (the peak counter is reset at each mark). `<run_dir>/cuda_ledger.json`
keeps the rows. Per update, the memory watch (`learner_lifecycle.CudaMemoryWatch`) records the
update's and the rollout's peaks (`lifecycle/cuda_update_peak_*_mib`, `lifecycle/cuda_rollout_peak_*`)
and `train()` the device-resident batch (`lifecycle/device_batch_mib`).

Inert off CUDA. A diagnostic: it never raises.
"""
from __future__ import annotations

import contextlib
import json
import os
from typing import Any, Dict, List, Optional

MiB = 1 << 20


class CudaLedger:
    def __init__(self, device: Any = None) -> None:
        import torch
        dev = torch.device(device) if device is not None else None
        self.device = dev if dev is not None and dev.type == "cuda" and torch.cuda.is_available() else None
        self.rows: List[Dict[str, Any]] = []
        self._prev_alloc = 0
        self._prev_reserved = 0

    @property
    def enabled(self) -> bool:
        return self.device is not None

    def mark(self, label: str) -> Optional[Dict[str, Any]]:
        """Record the card after ``label``'s step (allocated, reserved, the step's deltas, the peak
        allocated during it); resets the peak counter for the next step."""
        if self.device is None:
            return None
        try:
            import torch
            torch.cuda.synchronize(self.device)
            alloc = int(torch.cuda.memory_allocated(self.device))
            res = int(torch.cuda.memory_reserved(self.device))
            peak = int(torch.cuda.max_memory_allocated(self.device))
            free, total = (int(x) for x in torch.cuda.mem_get_info(self.device))
            row = {"step": label, "allocated_mib": alloc / MiB, "reserved_mib": res / MiB,
                   "d_allocated_mib": (alloc - self._prev_alloc) / MiB,
                   "d_reserved_mib": (res - self._prev_reserved) / MiB,
                   "peak_allocated_mib": peak / MiB, "device_free_mib": free / MiB,
                   "device_total_mib": total / MiB}
            self._prev_alloc, self._prev_reserved = alloc, res
            torch.cuda.reset_peak_memory_stats(self.device)
            self.rows.append(row)
            return row
        except Exception:                                  # a diagnostic never breaks the run
            return None

    def lines(self) -> List[str]:
        out = ["🧮 [CudaLedger] the learner process's card, by startup step (MiB; allocated / reserved "
               "after the step, its delta, the peak allocated during it):"]
        for r in self.rows:
            out.append(f"    {r['step']:<58} alloc {r['allocated_mib']:8.0f} ({r['d_allocated_mib']:+8.0f})"
                       f"  reserved {r['reserved_mib']:8.0f} ({r['d_reserved_mib']:+8.0f})"
                       f"  peak {r['peak_allocated_mib']:8.0f}  free {r['device_free_mib']:8.0f}")
        return out

    def report(self, run_dir: Optional[str]) -> None:
        if not self.rows:
            return
        for line in self.lines():
            print(line, flush=True)
        if run_dir:
            with contextlib.suppress(OSError):
                with open(os.path.join(str(run_dir), "cuda_ledger.json"), "w") as f:
                    json.dump({"schema": "gen3_cuda_ledger_v1", "rows": self.rows}, f, indent=1)


def start(device: Any, label: str = "weights on the card (policy + ride-along heads)") -> CudaLedger:
    """A ledger for this startup, its first row taken now."""
    led = CudaLedger(device)
    led.mark(label)
    return led
