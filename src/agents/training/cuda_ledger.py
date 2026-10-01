"""The CUDA MEMORY LEDGER (`gen3_cuda_ledger_v1`, 2026-10-01) — where the learner process's card goes.

The sizing study measured production (N = 48, torch 2.8, fp32, `--env-core rust`) reserving 9.49 GiB
of the 12 GiB card with 1.2 GiB free — under K6's own ceiling margin — and ~3.9 GiB of it unattributed.
This ledger attributes it, in EVERY run's log, at the startup steps that acquire device memory:

    weights on the card -> rust env core (T2 slots, staging, arena) -> the extractor gate (2.5.1 only)
    -> the compiled regions' gate + prewarm -> optimizer state declared

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


class DeviceBatchWontFit(RuntimeError):
    """The declared device-resident batch cannot fit next to the learner's measured update step:
    refused at STARTUP (FATAL_CONFIG), never an OOM at the first update."""


#: Kept free on the card beyond the device batch and one learner step (K6's ceiling margin).
FIT_MARGIN_MIB = 512.0


def check_device_batch_fits(led: "CudaLedger", batch_bytes: int,
                            step_label: str = "compiled regions: gate + prewarm + lock") -> Optional[str]:
    """At the end of startup: the declared device batch (`device_batches.planned_bytes`) must fit in
    what the card can still give — its free bytes plus the allocator's cached-but-unused ones — after
    room for ONE learner step (the peak the regions' gate measured above the bytes allocated before
    it) and `FIT_MARGIN_MIB`. Raises `DeviceBatchWontFit` naming the levers; returns the log line
    (None when the ledger is inert or there is no device batch). NECESSARY, NOT SUFFICIENT: the real
    update peaks above the gate's step (8.75 GiB vs a 4.3 GiB gate step at N = 48), and N = 256 with
    the X26 heads PASSED this check and then ran out of memory in its first update (10.19 GiB allocated,
    2026-10-01). It catches a gross misconfiguration (N = 256 x 2,048 steps); a pass is no promise."""
    if not led.enabled or not led.rows or batch_bytes <= 0:
        return None
    last = led.rows[-1]
    rows = {r["step"]: (i, r) for i, r in enumerate(led.rows)}
    step = 0.0
    if step_label in rows:
        i, r = rows[step_label]
        before = led.rows[i - 1]["allocated_mib"] if i > 0 else 0.0
        step = max(0.0, r["peak_allocated_mib"] - before)
    available = last["device_free_mib"] + (last["reserved_mib"] - last["allocated_mib"])
    need = batch_bytes / MiB
    room = available - step - FIT_MARGIN_MIB
    line = (f"🧮 [CudaLedger] device batch {need:.0f} MiB vs room {room:.0f} MiB (free "
            f"{last['device_free_mib']:.0f} + cached {last['reserved_mib'] - last['allocated_mib']:.0f} "
            f"- one learner step {step:.0f} - margin {FIT_MARGIN_MIB:.0f})")
    if need > room:
        raise DeviceBatchWontFit(
            f"{line}: the DEVICE-RESIDENT BATCH cannot fit (gen3_cuda_ledger_v1). Its size is "
            f"n_steps x n_envs rows x the obs width — shrink the ROLLOUT (n_steps = D / n_envs keeps "
            f"D ~98k rows), or keep the batch on the host (chunked staging / no device batch — K8.6 "
            f"`instrumented_ppo/device_batches.py`). Refused at startup rather than OOM at the first update.")
    return line


def start(device: Any, label: str = "weights on the card (policy + ride-along heads)") -> CudaLedger:
    """A ledger for this startup, its first row taken now."""
    led = CudaLedger(device)
    led.mark(label)
    return led
