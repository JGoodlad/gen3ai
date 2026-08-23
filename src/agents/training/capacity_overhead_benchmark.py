"""Measure `--capacity-telemetry`'s TRAIN-STEP overhead on the real Gen3 architecture.

Builds the real feature extractor + dual-head policy over the live observation space, fills a
rollout buffer, and times `model.train()` with the flag OFF and ON. **The arms are INTERLEAVED**
(off/on/on/off per round) so a box that warms or cools mid-run cannot masquerade as a result —
the failure mode that voided three separate investigations in this tree.

The shape is chosen to match PRODUCTION's minibatch RATIO, not its absolute sizes: 80 minibatches
per `train()` (production is a 2048x64 rollout / batch 16384 / 10 epochs = 80), so the half-batch
cosine fires exactly once per `train()` as it would live. Get that wrong and the cosine — the most
expensive of the three probes — is measured at the wrong duty cycle.

⚠️ **The number this prints is a CONSERVATIVE BOUND for production**, and knowing why is the point:
it runs the BASELINE extractor chain (no damage op, no belief heads — whatever
`get_features_extractor_kwargs` defaults to), whose forward is far cheaper than the production one.
A cheaper denominator makes the probes' share LARGER. On CUDA under `--compile-trainer` it is
smaller again, since the canary's tiny eager MLP is noise against a compiled forward+backward.

⚠️ A benchmark's output IS the measurement, so this warns about contention rather than scaling
anything (`warn_if_contended`), and pinning BLAS threads matters: an unpinned run on a busy box
measures the scheduler.

Run: OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 \
     /home/goodlad/miniconda3/envs/gen3ai_stable/bin/python3 \
       src/agents/training/capacity_overhead_benchmark.py
(in a linked worktree, first: export PYTHONPATH=$PYTHONPATH:src)

Measured 2026-08-23, CPU, 2,047,958-param policy, obs_dim 2501, 10 reps per arm, QUIET box — two
independent runs, arms disjoint in both:
    OFF median 5797.2 ms · ON median 5943.4 ms · OVERHEAD +2.52% (mins +2.42%)
    OFF median 5800.6 ms · ON median 5938.6 ms · OVERHEAD +2.38% (mins +2.32%)

A third run taken while a 4-worker pytest suite shared the box read +4.28%, with the OFF arm alone
spreading 13% against a clean arm's 1.7% — and `warn_if_contended` did not fire, because the
one-minute load average lags a job that just started. **Read the printed per-arm spread before
believing the delta.** Interleaving preserves the SIGN under contention, not the size.
"""
import statistics
import time

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from agents.model.features_extractor import Gen3FeaturesExtractor, NET_ARCH
from agents.model.policy import Gen3DualHeadMaskablePolicy
from agents.observation.state_encoder import Gen3ObservationEncoder, load_mappings
from agents.training.instrumented_ppo import InstrumentedMaskablePPO
from utils.contention import warn_if_contended

N_STEPS, N_ENVS, BATCH, EPOCHS = 256, 8, 256, 10
COSINE_EVERY = 50
ROUNDS = 5              # each round times 2 OFF + 2 ON


class _RandEnv(gym.Env):
    """A Dict-obs maskable env emitting random observations of the REAL width.

    Values are held in [0, 1) deliberately: several obs slots are read as INTEGER embedding
    indices, so an unbounded draw indexes out of range. Every lookup lands on row 0, which costs
    the benchmark nothing — timing depends on the SHAPES, and those are the production ones.
    """

    def __init__(self, obs_dim: int):
        self.observation_space = spaces.Dict({
            "observation": spaces.Box(0.0, 1.0, shape=(obs_dim,), dtype=np.float32),
            "action_mask": spaces.Box(0, 1, shape=(11,), dtype=np.int8)})
        self.action_space = spaces.Discrete(11)
        self._dim = obs_dim
        self._rng = np.random.default_rng(0)

    def action_masks(self):
        return np.ones(11, dtype=np.int8)

    def _obs(self):
        return {"observation": self._rng.random(self._dim).astype(np.float32),
                "action_mask": np.ones(11, dtype=np.int8)}

    def reset(self, *, seed=None, options=None):
        return self._obs(), {}

    def step(self, action):
        return self._obs(), 0.1, False, False, {}


def main() -> None:
    warn_if_contended()
    enc = Gen3ObservationEncoder(load_mappings())
    obs_dim = enc.dimension
    print(f"obs_dim={obs_dim}  n_steps={N_STEPS} n_envs={N_ENVS} batch={BATCH} epochs={EPOCHS}")

    venv = DummyVecEnv([(lambda: _RandEnv(obs_dim)) for _ in range(N_ENVS)])
    model = InstrumentedMaskablePPO(
        Gen3DualHeadMaskablePolicy, venv,
        policy_kwargs={"features_extractor_class": Gen3FeaturesExtractor,
                       "features_extractor_kwargs": enc.get_features_extractor_kwargs(),
                       "net_arch": NET_ARCH},
        n_steps=N_STEPS, batch_size=BATCH, n_epochs=EPOCHS, device="cpu", seed=0)
    print(f"policy params: {sum(p.numel() for p in model.policy.parameters()):,}")
    model.learn(total_timesteps=N_STEPS * N_ENVS)

    def timed(on: bool, reps: int = 1):
        model.capacity_telemetry = bool(on)
        model._capacity_state = None                  # a fresh probe each arm — no warm canary
        model.canary_reset_steps = 1_000_000
        model.capacity_cosine_every = COSINE_EVERY
        model.capacity_velocity_every = 50
        out = []
        for _ in range(reps):
            np.random.seed(7)                         # same minibatch permutation in both arms
            th.manual_seed(7)
            t0 = time.perf_counter()
            model.train()
            out.append(time.perf_counter() - t0)
        return out

    timed(False, 2)                                   # warm the allocator / the caches
    off, on = [], []
    for _ in range(ROUNDS):
        off += timed(False)
        on += timed(True)
        on += timed(True)
        off += timed(False)

    mo, mn = statistics.median(off), statistics.median(on)
    print(f"OFF median {mo * 1000:8.1f} ms   (min {min(off) * 1000:.1f})")
    print(f"ON  median {mn * 1000:8.1f} ms   (min {min(on) * 1000:.1f})")
    print(f"OVERHEAD: {100.0 * (mn - mo) / mo:+.2f}%   "
          f"(mins: {100.0 * (min(on) - min(off)) / min(off):+.2f}%)")
    print(f"  OFF {sorted(round(x * 1000) for x in off)}")
    print(f"  ON  {sorted(round(x * 1000) for x in on)}")
    n_mb = (N_STEPS * N_ENVS // BATCH) * EPOCHS
    print(f"minibatches/train(): {n_mb} -> cosine fires {n_mb // COSINE_EVERY}x per train()")


if __name__ == "__main__":
    main()
