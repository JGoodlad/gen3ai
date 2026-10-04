"""Cold compile wall of the extractor forward on CUDA, one arm per PROCESS (X5 U2 GPU checks, F-X5-22).

    GEN3AI_TEST_ALLOW_GPU=1 python compile_ab.py --arm {blob,fixed_mass,fixed_mass_tau_op,fixed_mass_iters28}
        --batch 256 [--grad] --out result.json

Isolates the compile cost of the unrolled 64-step bisection from everything else a launch does:
the SAME production-surface extractor forward, a PRIVATE hermetic Inductor cache (cold), a fresh process,
`torch.compile(fn, dynamic=False, fullgraph=True)`, the first call timed (compile + first run), then the
steady per-call time of the compiled forward and of the eager forward.

ARMS
  blob                 the production arm (no hypothesis builder)
  fixed_mass           `--belief-tokens fixed_mass` as built (64 bisection steps unrolled in the graph)
  fixed_mass_tau_op    MEASUREMENT ONLY (F-X5-22's "tau computed outside the compiled graph"): `fixed_size_tau`
                       is replaced by an opaque `torch.library.custom_op` whose eager implementation is the same
                       bisection and whose fake implementation returns an empty [B] tensor, so the compiled
                       graph sees ONE op (no unrolled steps, no graph break, fullgraph still holds). NOT a design.
  fixed_mass_iters0    MEASUREMENT ONLY: ZERO bisection steps (tau = the bracket's midpoint, meaningless values, the SAME
                       graph otherwise): what the construction's unrolled steps cost, by subtraction
  fixed_mass_iters28   MEASUREMENT ONLY: the bisection at 28 steps (F-X5-22's "fewer fp32 steps"; fp32 stops moving
                       after ~25). Changes sec 3.2's "same count in every dtype": NOT a design.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import deque

import numpy as np
import torch


def build(arm: str):
    from agents.model.policy import Gen3DualHeadMaskablePolicy
    from agents.training.instrumented_ppo import InstrumentedMaskablePPO
    from agents.training.rust_rollout.build import trainee_spaces
    from agents.training.rust_vec_env import RustVecEnv
    from main.fresh_checkpoint import _production_policy_kwargs
    from main.train.production_args import production_args
    from utils.torch_state_guard import single_thread_build

    args = production_args()
    if arm != "blob":
        args.belief_tokens = "fixed_mass"
    obs, act = trainee_spaces(args)
    env = RustVecEnv(n_envs=4, observation_space=obs, action_space=act, build=lambda m: None)
    _a, _l, pk = _production_policy_kwargs(args)
    torch.manual_seed(0)
    with single_thread_build():
        model = InstrumentedMaskablePPO(Gen3DualHeadMaskablePolicy, env, n_steps=16, batch_size=16,
                                        n_epochs=1, device="cpu", seed=0, policy_kwargs=pk, verbose=0)
    model.ep_info_buffer = deque(maxlen=100)
    return model


def patch_tau_op() -> None:
    import agents.model.hypothesis_set as HS

    orig = HS.fixed_size_tau

    @torch.library.custom_op("x5meas::fixed_size_tau", mutates_args=())
    def tau_op(scores: torch.Tensor, cand: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        return orig(scores, cand, k, HS.BISECTION_ITERS)

    @tau_op.register_fake
    def _(scores, cand, k):
        return scores.new_empty(scores.shape[0])

    def patched(scores, cand, k, n_iter=HS.BISECTION_ITERS):
        return tau_op(scores.detach(), cand, k)

    HS.fixed_size_tau = patched


def patch_iters(n: int) -> None:
    import agents.model.hypothesis_set as HS

    orig = HS.fixed_size_tau
    HS.fixed_size_tau = lambda scores, cand, k, n_iter=n: orig(scores, cand, k, n)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["blob", "fixed_mass", "fixed_mass_tau_op", "fixed_mass_iters28",
                                                   "fixed_mass_iters0"])
    ap.add_argument("--obs", required=True)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--grad", action="store_true")
    ap.add_argument("--profile", action="store_true",
                    help="after timing, torch.profiler 3 compiled steps; write kernel-class totals + top kernels")
    ap.add_argument("--bwd", action="store_true",
                    help="grad mode AND time forward + backward of (pi.sum() + vf.sum()) — the learner's shape")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from agents.model.compile_cache import ensure_hermetic_cache
    from agents.model.parity_probe import PERTURB_SCALE, PERTURB_SEED, perturb_

    root = ensure_hermetic_cache(f"x5 compile ab {a.arm}")
    if a.arm == "fixed_mass_tau_op":
        patch_tau_op()
    if a.arm == "fixed_mass_iters28":
        patch_iters(28)
    if a.arm == "fixed_mass_iters0":
        patch_iters(0)
    model = build(a.arm)
    pol = model.policy
    pol.eval()
    perturb_(pol, seed=PERTURB_SEED, scale=PERTURB_SCALE)
    pol.to("cuda")
    fe = pol.features_extractor
    with np.load(a.obs) as z:
        obs = {k[4:]: torch.as_tensor(z[k][:a.batch]).to("cuda") for k in z.files if k.startswith("obs:")}

    def fn(b):
        pi, vf = fe(b)
        return pi, vf

    grad = a.grad or a.bwd
    ctx = torch.enable_grad() if grad else torch.no_grad()
    res = {"arm": a.arm, "batch": a.batch, "grad": grad, "bwd": a.bwd, "torch": torch.__version__}
    comp = torch.compile(fn, dynamic=False, fullgraph=True)

    def step(f):
        out = f(obs)
        if a.bwd:
            (out[0].float().sum() + out[1].float().sum()).backward()
            pol.zero_grad(set_to_none=True)
        return out

    with ctx:
        torch.cuda.synchronize(); t0 = time.time()
        step(comp)
        torch.cuda.synchronize()
        res["first_call_compile_s"] = round(time.time() - t0, 1)

        def timed(f, n=30):
            for _ in range(5):
                step(f)
            torch.cuda.synchronize(); t = time.time()
            for _ in range(n):
                step(f)
            torch.cuda.synchronize()
            return round((time.time() - t) / n * 1000, 3)

        res["compiled_ms_per_call"] = timed(comp)
        res["eager_ms_per_call"] = timed(fn, n=10)
        if a.profile:
            from torch.profiler import ProfilerActivity, profile
            n_it = 3
            torch.cuda.synchronize()
            with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
                for _ in range(n_it):
                    step(comp)
                torch.cuda.synchronize()
            rows = {}
            for e in prof.key_averages():
                us = getattr(e, "self_device_time_total", getattr(e, "self_cuda_time_total", 0))
                if us > 0:
                    rows[e.key] = (us / n_it / 1000.0, e.count / n_it)

            def cls(name: str) -> str:
                n = name.lower()
                if "fmha" in n or "flash" in n or "attention" in n or "sdpa" in n:
                    return "attention"
                if "gemm" in n or "cutlass" in n or "cublas" in n or "sgemm" in n:
                    return "gemm"
                if n.startswith("triton_"):
                    return "triton_reduction" if ("red" in n.split("_")[1] or "per" in n.split("_")[1]) else "triton_pointwise"
                if "memcpy" in n or "memset" in n:
                    return "memcpy"
                return "other"
            by = {}
            for k, (ms, c) in rows.items():
                by.setdefault(cls(k), [0.0, 0.0])
                by[cls(k)][0] += ms; by[cls(k)][1] += c
            res["kernel_class_ms_per_step"] = {k: [round(v[0], 3), round(v[1], 1)] for k, v in sorted(by.items())}
            res["kernel_total_ms_per_step"] = round(sum(v[0] for v in by.values()), 3)
            res["kernel_launches_per_step"] = round(sum(v[1] for v in by.values()), 1)
            top = sorted(rows.items(), key=lambda kv: -kv[1][0])[:25]
            res["top_kernels_ms_per_step"] = [[k[:140], round(v[0], 3), round(v[1], 1)] for k, v in top]
    from torch._dynamo.utils import counters
    res["graph_breaks"] = dict(counters.get("graph_break", {}))
    res["unique_graphs"] = int(counters["stats"].get("unique_graphs", 0))
    res["cache_root"] = root
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
