"""T62 LEARNER cell: the frozen GPU suite's `update_bf16` phase (the learner's real compiled `train()` at the
production shape, X26's training surface) with `hypothesis_set.TAU_SOLVE_DTYPE` set BEFORE anything is built or
compiled -- ``--tau float32`` is the pre-T62 solve (bd3126e02^: the scores reach `fixed_size_tau` in float32, the bf16
autocast is confined to the trunk rounds), ``--tau float64`` the shipped one. One arm per process, so each compiles
its own program in its own hermetic cache.

    python3 t62_update_cell.py --tau float64 <cfg.json>      # a cfg written by battery.py (the suite's phase_config)
    python3 t62_update_cell.py --check                       # CPU: the module exposes the knob; no torch device touched

The compile sentinel's startup parity gate reads the K9 golden recorded under float64: the float32 arm may be
REFUSED there, and the suite cell then CONTINUES with install + prewarm only (recorded as ``gate.verdict``) -- the
timing is still the compiled update's.
"""
from __future__ import annotations

import argparse
import sys

#: Exit code of a T62 cell at a commit carrying FP32_ALT (no TAU_SOLVE_DTYPE): battery.py records a SKIP.
SUPERSEDED = 9


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tau", choices=("float32", "float64"))
    ap.add_argument("--check", action="store_true")
    ap.add_argument("cfg", nargs="?")
    a = ap.parse_args(argv)
    import torch

    from agents.model import hypothesis_set as HS
    if not hasattr(HS, "TAU_SOLVE_DTYPE") and "fp32_alt" in getattr(HS, "TAU_SOLVES", ()):
        print("SUPERSEDED by FP32_ALT (gen3_tau_fp32_alt_v1): hypothesis_set.TAU_SOLVE_DTYPE is gone; the FP32_ALT "
              "cells are ../gpu_window_battery_2026-10-20/")
        return SUPERSEDED
    if not hasattr(HS, "TAU_SOLVE_DTYPE"):
        print("PRECONDITION FAILED: hypothesis_set.TAU_SOLVE_DTYPE missing (a pre-T62 checkout)", file=sys.stderr)
        return 3
    if a.check:
        from agents.training import (
            gpu_bench_suite,  # noqa: F401 -- the phase worker imports
            gpu_bench_update,  # noqa: F401
        )
        print(f"ok: TAU_SOLVE_DTYPE default {HS.TAU_SOLVE_DTYPE}")
        return 0
    if not (a.tau and a.cfg):
        ap.error("--tau and cfg are required (or --check)")
    HS.TAU_SOLVE_DTYPE = getattr(torch, a.tau)
    from agents.training.gpu_bench_suite import phase_main
    rc = phase_main("update_bf16", a.cfg)
    if HS.TAU_SOLVE_DTYPE is not getattr(torch, a.tau):      # nothing may have reset it mid-cell
        print("TAU_SOLVE_DTYPE changed under the cell", file=sys.stderr)
        return 4
    return rc


if __name__ == "__main__":
    sys.exit(main())
