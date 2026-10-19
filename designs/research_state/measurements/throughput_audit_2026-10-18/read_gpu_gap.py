"""Read `run_gpu_gap.sh`'s output into the audit's lever table (pure reads of the suite's JSONs; no GPU).

    python3 designs/research_state/measurements/throughput_audit_2026-10-18/read_gpu_gap.py <out-dir>

Prints, per cell: the update estimate (10 epochs), the marginal epoch and the fixed part (E1 - marginal), and the
E-epoch cycle each implies on X26's measured cycle; for collection, env steps/s, s per 98,304 rows, GPU wait and the
core per host step, and the cycle each implies with the update unchanged. A missing or failed cell prints its status
— ABSENCE IS NOT A ZERO.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

#: X26's cycle at ~312M (gap_decomposition.py's inputs): the update, the collect, everything else (s).
X26_TRAIN, X26_COLLECT, X26_OTHER, ROWS = 41.962, 26.065, 5.01, 98304


def load(p: Path):
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError) as e:
        return {"status": f"unreadable ({e.__class__.__name__})"}


def update_row(name: str, body: dict) -> None:
    if body.get("status") != "ok":
        print(f"| {name} | {body.get('status')} | | | |")
        return
    t = body["train"]
    m = float(t["marginal_epoch_s"])
    fixed = float(t["median_s"]["1"]) - m
    sh = body.get("shape", {})
    for e in (10, 5, 2, 1):
        u = fixed + e * m
        cyc = u + X26_COLLECT + X26_OTHER
        print(f"| {name} (B_eff {sh.get('effective_batch')}) | E{e} | {u:.1f} s | {cyc:.1f} s | "
              f"{ROWS / cyc:,.0f} steps/s |")


def collect_row(name: str, body: dict) -> None:
    s = body.get("summary") or {}
    if body.get("status") != "ok" or not s:
        print(f"| {name} | {body.get('status')} | | | | |")
        return
    eps = float(s["env_steps_per_s_median"])
    col = ROWS / eps
    cyc = X26_TRAIN + col + X26_OTHER
    print(f"| {name} | {eps:,.0f} | {col:.1f} s | {s['gpu_wait_ms_per_host_step_median']:.1f} ms | "
          f"{s['core_ms_per_host_step_median']:.1f} ms | {cyc:.1f} s ({ROWS / cyc:,.0f} steps/s) |")


def main(argv=None) -> int:
    out = Path((argv or sys.argv[1:])[0])
    suite = load(out / "suite_3080ti" / "gpu_bench_suite.json")
    ph = suite.get("phases", {})
    print("UPDATE (bf16, production shape unless named)\n")
    print("| cell | epochs | update | cycle (X26 collect + other) | rate |\n|---|---|---|---|---|")
    update_row("suite update_bf16", ph.get("update_bf16", {"status": "missing"}))
    for n in ("u_e10_beff8k", "u_e10_beff2k"):
        update_row(n, load(out / "cells" / "phases" / f"{n}.json"))
    print("\nCOLLECT (fixture pool, untrained policies: compare cells with EACH OTHER, not with X26's absolute)\n")
    print("| cell | env steps/s | s per 98,304 | GPU wait / host step | core / host step | cycle (X26 update) |")
    print("|---|---|---|---|---|---|")
    collect_row("suite collect (N 256, pool 20)", ph.get("collect", {"status": "missing"}))
    for n in ("c_n512", "c_n1024", "c_pool5", "c_pool1"):
        collect_row(n, load(out / "cells" / "phases" / f"{n}.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
