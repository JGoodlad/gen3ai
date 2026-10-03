"""REGRESSION PIN (F-X5-4, `gen3_single_thread_init_v1`): a FRESH production model builds the SAME
starting weights at any torch thread count, and the build leaves the caller's thread count as it found it.

Mechanism (measured 2026-10-03, torch 2.8.0+cu126): SB3's `_build` re-initialises every Linear with
`torch.nn.init.orthogonal_`, a LAPACK QR whose blocked reduction order follows the BLAS thread count.
The RNG draws are identical; the rounding is not (max |delta| ~1.1e-6, ~95% of a 512x512 matrix's bytes).
Nothing in `main/train` pinned threads, so a fresh run's init depended on the core count and
`OMP_NUM_THREADS`. `model_build.construct_fresh_learner` (the trainer's fresh build) and
`fresh_checkpoint.build_fresh_model` (the production-surface fixture) now build inside the shared
`utils.torch_state_guard.single_thread_build`.

FAILS on revert: without the wrapper the 8-thread state_dict differs from the 1-thread one (the probe
below printed `eb409f05…` at 1 thread and `ff9276ac…` at 8 unwrapped).

The probe runs in a CHILD interpreter per thread count: the root conftest pins OMP & co. to 1 for the test
process, and BLAS reads them once at init, so only a fresh process can measure a multi-thread build.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

_PROBE = r"""
import hashlib, json, sys
import gymnasium as gym, numpy as np, torch
n = int(sys.argv[1]); site = sys.argv[2]
torch.set_num_threads(n)
if site == "trainer":
    from agents.training.rust_rollout.testkit import ToyVecEnv
    from main.fresh_checkpoint import _production_policy_kwargs
    from main.train.model_build import construct_fresh_learner
    args, layout, pk = _production_policy_kwargs()
    args.device = "cpu"; args.seed = 0
    args.batch_size = min(args.batch_size, args.n_steps)
    d = layout["total_dim"]
    space = gym.spaces.Dict({"observation": gym.spaces.Box(-np.inf, np.inf, (d,), np.float32),
                             "action_mask": gym.spaces.MultiBinary(11)})
    class E(gym.Env):
        observation_space = space; action_space = gym.spaces.Discrete(11)
        def reset(self, **k): return {"observation": np.zeros(d, np.float32), "action_mask": np.ones(11, np.int8)}, {}
        def step(self, a): return self.reset()[0], 0.0, False, False, {}
    model = construct_fresh_learner(args, ToyVecEnv([E]), pk)
else:
    from main.fresh_checkpoint import build_fresh_model
    model, _a, _pk = build_fresh_model(0)
h = hashlib.sha256()
for k, v in sorted(model.policy.state_dict().items()):
    h.update(k.encode()); h.update(v.detach().cpu().contiguous().numpy().tobytes())
print(json.dumps({"asked": n, "after": torch.get_num_threads(), "sha": h.hexdigest()}))
"""


def _build_at(site: str, threads: int) -> dict:
    env = dict(os.environ)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[v] = str(threads)
    r = subprocess.run([sys.executable, "-c", _PROBE, str(threads), site], env=env, capture_output=True,
                       text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize("site", ["trainer", "fixture"])
def test_a_fresh_production_build_is_byte_identical_at_1_and_8_threads(site: str) -> None:
    one, eight = _build_at(site, 1), _build_at(site, 8)
    assert one["after"] == 1 and eight["after"] == 8, (
        "the build must restore the caller's thread count (training runs at it from the first rollout)")
    assert one["sha"] == eight["sha"], (
        f"the {site} fresh build's init depends on the torch thread count (F-X5-4): "
        f"1 thread {one['sha'][:16]} vs 8 threads {eight['sha'][:16]}")


def test_single_thread_build_restores_the_callers_count_even_on_an_exception() -> None:
    import torch

    from utils.torch_state_guard import single_thread_build, torch_globals

    with torch_globals(num_threads=3):
        with single_thread_build():
            assert torch.get_num_threads() == 1
        assert torch.get_num_threads() == 3
        with pytest.raises(RuntimeError, match="boom"):
            with single_thread_build():
                raise RuntimeError("boom")
        assert torch.get_num_threads() == 3
