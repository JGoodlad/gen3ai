"""F-LE-8's measurement (M5 Lane G): the policy opponents' SAMPLING cost per host step, today's
per-row torch generators (``rust_env_opponents.sample_actions``, bit-identical to ``RLPlayer``) vs the
counter-based KEYED DRAW (``keyed_draw.keyed_sample``), on the production shape (Lane E's read: ~41
opponent rows per host step at 48 envs, 11 actions, a stochastic route at T = 1).

Interleaved A B / B A blocks of ``--calls`` calls each, fresh log-probs per call (real-shaped: a random
legal mask, a softmax); the per-call median and a 95 % bootstrap CI of the ratio over blocks. A
benchmark's output IS the measurement: it WARNS on a busy box and never stretches.

    export PYTHONPATH=$PYTHONPATH:src
    python -m agents.training.keyed_draw_benchmark --rows 41 --blocks 10 --calls 400
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np


def _logp(rng: np.random.Generator, rows: int) -> np.ndarray:
    raw = rng.normal(size=(rows, 11)) * 2.0
    mask = rng.random((rows, 11)) < 0.7
    mask[np.arange(rows), rng.integers(0, 11, rows)] = True
    x = np.where(mask, raw, -np.inf)
    x = x - np.log(np.exp(np.where(mask, raw, -np.inf)).sum(1, keepdims=True))
    return x.astype(np.float32)


def run(rows: int, blocks: int, calls: int, seed: int = 0) -> Dict[str, Any]:
    import torch

    from agents.training import keyed_draw as KD
    from agents.training.rust_env_opponents import sample_actions
    from utils.contention import warn_if_contended

    busy = bool(warn_if_contended("keyed draw benchmark"))
    rng = np.random.default_rng(seed)
    gens = [torch.Generator().manual_seed(i) for i in range(rows)]
    temps = np.ones(rows)
    envs = np.arange(rows)
    per: Dict[str, List[float]] = {"generator": [], "keyed": []}
    k = 0
    for b in range(blocks):
        order = ("generator", "keyed") if b % 2 == 0 else ("keyed", "generator")
        for arm in order:
            lps = [_logp(rng, rows) for _ in range(calls)]
            t0 = time.perf_counter()
            for lp in lps:
                if arm == "generator":
                    sample_actions(lp, temps, gens)
                else:
                    KD.keyed_sample(lp, seed=7, stream=KD.STREAM_OPPONENT, env=envs, episode=k, decision=envs)
                k += 1
            per[arm].append(1e3 * (time.perf_counter() - t0) / calls)
    g, kd = np.array(per["generator"]), np.array(per["keyed"])
    r = np.log(kd / g)
    boot = np.random.default_rng(1).integers(0, r.size, (10_000, r.size))
    lo, hi = np.exp(np.percentile(r[boot].mean(1), [2.5, 97.5]))
    return {"rows": rows, "blocks": blocks, "calls_per_block": calls, "busy_box_warning": busy,
            "load1": os.getloadavg()[0], "torch_threads": torch.get_num_threads(),
            "ms_per_call": {"generator_median": float(np.median(g)), "keyed_median": float(np.median(kd))},
            "keyed_over_generator": {"point": float(np.exp(r.mean())), "lo": float(lo), "hi": float(hi)},
            "per_block_ms": per}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--rows", type=int, default=41)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--calls", type=int, default=400)
    ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)
    out = run(a.rows, a.blocks, a.calls)
    text = json.dumps(out, indent=1)
    print(text)
    if a.json:
        with open(a.json, "w") as f:
            f.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
