"""K9(b) — the TAIL of TF32's healthy |Δ log π|, for the persistence rule's false-FATAL estimate.

The first pass (``measure.py``) gave 96 TF32 micro-batch maxima, correlated (the same states re-scored
across updates). Extreme-value fits on them disagree by four orders of magnitude on P(max > 0.071), so
this pass adds INDEPENDENT healthy micro-batches: fresh seeds (new rollout rows, a new perturbation per
seed), TF32 eager, rollout-shaped vs learner-shaped forwards exactly as ``measure.py`` — and keeps every
row's |Δ| (float32), so the per-row tail can be fitted (peaks over threshold) as well as the block maxima.

Units of ``SEEDS_PER_UNIT`` seeds, each its own ``scripts/ops/gpu_lock.sh timeout 1200 ...`` acquisition,
durable ``parts/tail_u<k>.npz`` (a re-run resumes).

    python designs/research_state/measurements/k9_behaviour_bar_2026-09-30/tail.py --units 8
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch as th

sys.path.insert(0, str(Path(__file__).parent))
import measure as M  # noqa: E402

SEEDS_PER_UNIT = 6
FIRST_SEED = 100


def run_unit(k: int, out: Path) -> None:
    th.set_float32_matmul_precision("high")
    rows_d, micro_max, seeds = [], [], []
    for s in range(FIRST_SEED + k * SEEDS_PER_UNIT, FIRST_SEED + (k + 1) * SEEDS_PER_UNIT):
        rows = M.collect_rows(s)
        m = M.learner(s, rows, False)
        mu = M.logp(m, rows, batch=M.N_ENVS, train_mode=False)
        pi = M.logp(m, rows, batch=M.MICRO, train_mode=True)
        d = np.abs(pi - mu).astype(np.float32)
        rows_d.append(d)
        micro_max += [float(x[0]) for x in M.micro_stats(pi - mu)]
        seeds.append(s)
        del m
        th.cuda.empty_cache()
    np.savez_compressed(out, rows=np.concatenate(rows_d), micro_max=np.array(micro_max), seeds=np.array(seeds))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units", type=int, default=8)
    ap.add_argument("--unit", type=int, default=None, help="internal: run one unit in this process")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.unit is not None:
        run_unit(a.unit, Path(a.out))
        return 0
    parts = Path(__file__).parent / "parts"
    parts.mkdir(exist_ok=True)
    lock = str(Path(__file__).resolve().parents[4] / "scripts" / "ops" / "gpu_lock.sh")
    for k in range(a.units):
        part = parts / f"tail_u{k}.npz"
        if part.exists():
            continue
        tmp = parts / f"tail_u{k}.partial.npz"
        r = subprocess.run([lock, "timeout", "1200", sys.executable, __file__, "--unit", str(k), "--out", str(tmp)])
        if r.returncode != 0:
            raise SystemExit(f"tail unit {k} failed ({r.returncode}); written units are kept")
        tmp.rename(part)
        print(f"unit {k} written", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
