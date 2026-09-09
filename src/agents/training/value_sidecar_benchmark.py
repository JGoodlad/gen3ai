"""THE VALUE SIDECAR'S COST, at production buffer shape (`gen3_value_sidecar_v1`).

`python3 src/agents/training/value_sidecar_benchmark.py [--rollout-seconds S]`

**WHAT IT MEASURES AND WHY THAT SHAPE.** The sidecar's whole cost is one `_on_rollout_end`: a
numpy slice over the rollout buffer plus a `json.dumps` per sampled row and ONE append. So the
quantity to measure is that call at the shape production actually runs — `--n-steps 2048
--n-envs 64`, i.e. 131,072 buffer cells of which ~2,048 are written at the default 1/64.

🚨 **A SMOKE A/B CANNOT MEASURE THIS AND MUST NOT BE QUOTED AS IF IT COULD.** Measured 2026-09-08:
two `--debug --steps 10000` runs differing only in `--value-sidecar` came out at 2:39.10 (ON) and
2:45.60 (OFF) — the arm WITH the sidecar was 6.5 s faster. That is not a negative cost; it is a
run-to-run spread of a few percent on a battle-simulation-bound smoke, which is one to two orders
of magnitude wider than the thing being measured. The honest decomposition is to measure the
NUMERATOR directly, here, and to divide it by a rollout time quoted with its own provenance.

🚨 **THE DENOMINATOR IS AN INPUT, NOT A MEASUREMENT.** `--rollout-seconds` is whatever a real
rollout takes on the box in question; this script cannot observe it without running training. The
default is deliberately the most HOSTILE plausible value rather than a typical one — a fast rollout
makes the overhead look worse, so a pass at the default passes everywhere slower.

**BENCHMARKS WARN, THEY DO NOT STRETCH.** Per the root `CLAUDE.md`: a benchmark's output IS the
measurement, so contention is REPORTED and the number is left alone. A timing taken on a busy box
is a fact about a busy box, and silently rescaling it would destroy exactly the information a
reader needs to discount it.
"""

from __future__ import annotations

import argparse
import os
import tempfile
import time

import numpy as np

from agents.observation.constants import MAX_TURNS
from agents.training.value_sidecar import (
    CLOCK_LINEAR_INDEX, DEFAULT_SIDECAR_FRACTION, ValueSidecarCallback, sidecar_path,
)
from utils.contention import cpu_contention_factor, describe_contention

#: Production rollout geometry — `--n-steps 2048 --n-envs 64`, the shape every arm on the win-prob
#: ladder runs at.
N_STEPS, N_ENVS = 2048, 64

#: The hostile default denominator: a rollout of 131,072 steps in 120 s is ~1,090 steps/s, at the
#: fast end of anything this project has measured on cuda. A slower box only improves the ratio.
DEFAULT_ROLLOUT_SECONDS = 120.0

#: The budget the design committed to.
BUDGET_PCT = 1.0


def _production_model(seed: int = 0):
    """A rollout buffer with production's shape and dtypes, and NOTHING borrowed from the trainer.

    The point of the fake is that the sidecar touches only arrays, so a synthetic buffer of the
    right shape costs the callback exactly what a real one does — while keeping the benchmark
    runnable in two seconds with no battles, no model and no GPU.
    """
    import types
    rng = np.random.default_rng(seed)
    obs_w = CLOCK_LINEAR_INDEX + 1
    obs = {
        "observation": np.zeros((N_STEPS, N_ENVS, obs_w), dtype=np.float32),
        "win_target": rng.integers(0, 2, (N_STEPS, N_ENVS, 1)).astype(np.float32),
        "win_mask": np.ones((N_STEPS, N_ENVS, 1), dtype=np.float32),
        "win_margin": rng.uniform(-1, 1, (N_STEPS, N_ENVS, 1)).astype(np.float32),
        "opp_class": rng.integers(0, 4, (N_STEPS, N_ENVS, 1)).astype(np.int64),
    }
    turns = np.tile(np.arange(N_STEPS, dtype=np.float32)[:, None] % 60.0, (1, N_ENVS))
    obs["observation"][:, :, CLOCK_LINEAR_INDEX] = 1.0 - (turns / MAX_TURNS)
    buf = types.SimpleNamespace(
        observations=obs,
        values=rng.uniform(0, 1, (N_STEPS, N_ENVS)).astype(np.float32),
        # An episode boundary every ~40 steps, which is the measured order for gen3 OU.
        episode_starts=(np.tile((np.arange(N_STEPS) % 40 == 0)[:, None], (1, N_ENVS))
                        ).astype(np.float32),
    )
    return types.SimpleNamespace(rollout_buffer=buf, num_timesteps=10_000_000)


def measure(repeats: int, fraction: float):
    """Per-rollout seconds and bytes. Returns (times, bytes_per_rollout, rows_per_rollout)."""
    model = _production_model()
    times = []
    with tempfile.TemporaryDirectory() as d:
        cb = ValueSidecarCallback(d, fraction=fraction, critic_mode="winprob")
        cb.model = model
        for _ in range(repeats):
            t0 = time.perf_counter()
            cb._on_rollout_end()
            times.append(time.perf_counter() - t0)
        size = os.path.getsize(sidecar_path(d))
    return times, size / repeats, cb.rows_written / repeats


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--fraction", type=float, default=DEFAULT_SIDECAR_FRACTION)
    ap.add_argument("--rollout-seconds", type=float, default=DEFAULT_ROLLOUT_SECONDS,
                    help=f"wall-clock of ONE production rollout, the DENOMINATOR "
                         f"(default {DEFAULT_ROLLOUT_SECONDS}s — deliberately hostile)")
    a = ap.parse_args(argv)

    times, nbytes, rows = measure(a.repeats, a.fraction)
    med = sorted(times)[len(times) // 2]
    worst = max(times)
    pct, pct_worst = 100.0 * med / a.rollout_seconds, 100.0 * worst / a.rollout_seconds

    factor = cpu_contention_factor()
    print("=" * 78)
    print("VALUE SIDECAR — per-rollout cost at production shape "
          f"(n_steps={N_STEPS}, n_envs={N_ENVS}, fraction={a.fraction:g})")
    print("=" * 78)
    print(f"  contention        : {describe_contention()}")
    if factor >= 1.05:
        print("  ⚠️  THE BOX IS BUSY. The number below is a fact about a BUSY box and is NOT")
        print("      rescaled — a benchmark's output IS the measurement. Discount it, or re-run")
        print("      when the box is quiet; do not adjust it.")
    print(f"  rows / rollout    : {rows:,.0f} of {N_STEPS * N_ENVS:,} states")
    print(f"  bytes / rollout   : {nbytes / 1e6:.2f} MB")
    print(f"  median            : {med * 1e3:.1f} ms")
    print(f"  worst of {a.repeats:<3d}      : {worst * 1e3:.1f} ms")
    print(f"  denominator       : {a.rollout_seconds:.1f} s per rollout (AN INPUT, not measured "
          "here)")
    print(f"  OVERHEAD          : {pct:.4f}% median, {pct_worst:.4f}% worst")
    verdict = "WITHIN" if pct_worst < BUDGET_PCT else "OVER"
    print(f"  verdict           : {verdict} the {BUDGET_PCT:g}% budget "
          f"(on the WORST sample, not the median)")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
