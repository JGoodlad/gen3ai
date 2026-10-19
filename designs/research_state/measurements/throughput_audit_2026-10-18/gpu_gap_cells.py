"""The throughput audit's GPU cells for the 2026-10-19 GPU gap: the FROZEN suite's own update / collect cells
(`agents.training.gpu_bench_suite`, suite_version 1) at the shapes the audit's levers need. Writes one config JSON
per cell; `run_gpu_gap.sh` runs them. Pure (no GPU, no torch import): safe to run anywhere.

    python3 designs/research_state/measurements/throughput_audit_2026-10-18/gpu_gap_cells.py --out DIR [--list]

Every cell is `phase_config(...)` of the suite with ONE field changed, so the frozen inputs (the X26 argv, the K9
learner, the fixture rows, the seeded fixture pool) are the suite's. The changed field is the cell's lever:

* ``u_e10_beff8k``   — the bf16 update at micro-batch 2,048 x accumulation 4 (effective batch 8,192; 12 optimizer
  steps an epoch instead of 2), epochs 1 and 3: the per-optimizer-step overhead the "fewer epochs, smaller batch"
  lever pays (compare its marginal epoch with the suite's `update_bf16`);
* ``u_e10_beff2k``   — micro-batch 2,048 x accumulation 1 (48 steps an epoch): the same, at the extreme;
* ``c_n512`` / ``c_n1024`` — collection at 512 / 1,024 envs (n_steps 192 / 96, the same 98,304 rows), the 20-snapshot
  pool: does a bigger inference batch amortise the opponent fan-out on THIS network (SIZING measured it only on the
  pre-end-state network, 2026-10-01);
* ``c_pool5`` / ``c_pool1`` — collection at 256 envs against a 5- / 1-snapshot pool: what capping the snapshots
  live in one rollout would buy (T24 measured +1.0 ms of GPU wait per host step per snapshot on X26).

The suite's own `run` (all phases, the PENDING local 3080 Ti row of `designs/ops/cloud_gpu_benchmark.md` §0) comes
FIRST in `run_gpu_gap.sh` and is not generated here.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def cells(out_dir: str) -> dict:
    from agents.training.gpu_bench_suite import phase_config

    def upd(name, accum):
        c = phase_config("update_bf16", False, "cuda", out_dir)
        c.update(name=name, grad_accum=accum, out=str(Path(out_dir) / "phases" / f"{name}.json"))
        return c

    def col(name, extra, pool=20):
        c = phase_config("collect", False, "cuda", out_dir)
        c.update(name=name, extra=list(extra), pool_snapshots=pool,
                 out=str(Path(out_dir) / "phases" / f"{name}.json"),
                 scratch=str(Path(out_dir) / "scratch" / f"{name}_run"))
        return c

    return {
        "u_e10_beff8k": upd("u_e10_beff8k", 4),
        "u_e10_beff2k": upd("u_e10_beff2k", 1),
        "c_n512": col("c_n512", ["--n-envs", "512", "--n-steps", "192"]),
        "c_n1024": col("c_n1024", ["--n-envs", "1024", "--n-steps", "96"]),
        "c_pool5": col("c_pool5", [], pool=5),
        "c_pool1": col("c_pool1", [], pool=1),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--list", action="store_true", help="print the cell names in run order and exit")
    a = ap.parse_args(argv)
    cs = cells(str(Path(a.out).resolve()))
    if a.list:   # "<the suite's phase KIND> <cell name>": `_phase` dispatches on the kind, the cfg names the cell
        for name, c in cs.items():
            print(("update_bf16" if name.startswith("u_") else "collect"), name)
        return 0
    d = Path(a.out) / "phases"
    d.mkdir(parents=True, exist_ok=True)
    for name, c in cs.items():
        (d / f"{name}.cfg.json").write_text(json.dumps(c, indent=1))
        print(d / f"{name}.cfg.json", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
