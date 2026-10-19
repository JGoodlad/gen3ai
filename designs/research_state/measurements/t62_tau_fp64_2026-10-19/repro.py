"""T62 REPRODUCTION: do two fp32 PROGRAMS agree on the K9(b) violators' species tau once the constraint is solved
in float64? Before (the pre-T62 in-dtype fp32 solve) they disagreed by about the band ulp(k) / S'(tau) ~ 1e-2.

For each X26 tau-class event dump (u32180, u32320, u35950) and each bracketing checkpoint (read-only copies):
1. CAPTURE: a float64 forward (the diagnosis's `tau_programs.forward64`) records call 0 of `fixed_size_tau`'s
   inputs -- the species scores [n, 400], the candidate mask and k -- for the violator and the scan's 32 rows.
2. Cast the scores to fp32 (what both training programs hold) and solve tau with
   OLD = `hypothesis_set.TAU_SOLVE_DTYPE` patched to float32 (bit-identical to the pre-T62 arithmetic), and
   NEW = the shipped float64 solve, each under two kinds of "second program":
   (a) ORDER: P random permutations of the candidate axis, eager (the diagnosis's emulation);
   (b) COMPILED: eager CPU vs an inductor-compiled CPU `fixed_size_tau` (a real second program: a fused C++
       reduction against ATen's).
   Reported per row: max |tau_i - tau_j| over the program pair, and |tau - tau_fp64|.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<worktree>/src CUDA_VISIBLE_DEVICES= \
        python3 repro.py --out repro.json
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch as th

_DIAG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "x26_a3_stop_316M_2026-10-18")
_spec = importlib.util.spec_from_file_location("x26_a3_tau_programs", os.path.join(_DIAG, "tau_programs.py"))
TP = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(TP)  # type: ignore[union-attr]
RN = TP.RN

CACHE = os.path.expanduser("~/.cache/gen3ai")
#: event -> (dump dir, rows file, [neighbour checkpoints])
EVENTS: Dict[int, Tuple[str, List[str]]] = {
    32180: ("x26_a3_stop", ["316000231", "317000041"]),
    32320: ("x26_a3_stop", ["317000041", "320000071"]),
    35950: ("x26_a6_u35950", ["353000151", "354000424"]),
}
#: the policy template (any X26 checkpoint; its weights are replaced) + the run's model_config.json beside it
TEMPLATE = os.path.join(CACHE, "x26_a6_rediag", "u35950", "in", "checkpoint_316000231_steps.zip")


def solve(s: th.Tensor, cand: th.Tensor, k: th.Tensor, dt: th.dtype, compiled: bool = False) -> np.ndarray:
    from agents.model import hypothesis_set as HS
    import torch._dynamo as dynamo
    saved = HS.TAU_SOLVE_DTYPE
    HS.TAU_SOLVE_DTYPE = dt
    try:
        if compiled:
            dynamo.reset()
            fn = th.compile(HS.fixed_size_tau, fullgraph=True, backend="inductor", dynamic=False)
            out = fn(s, cand, k)
            dynamo.reset()
        else:
            out = HS.fixed_size_tau(s, cand, k)
    finally:
        HS.TAU_SOLVE_DTYPE = saved
    return out.double().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--perms", type=int, default=64)
    a = ap.parse_args()
    th.set_float32_matmul_precision("highest")
    th.set_num_threads(int(os.environ.get("T62_THREADS", "3")))
    from main.capacity import load_policy
    model, _cfg, _ = load_policy(TEMPLATE, os.path.dirname(TEMPLATE), "cpu")
    template = copy.deepcopy(model.policy)
    del model
    rng = np.random.default_rng(20261019)
    res: Dict[str, Any] = {"perms": a.perms, "torch": th.__version__}
    for u, (d, ckpts) in EVENTS.items():
        z = np.load(os.path.join(CACHE, d, f"behaviour_violation_u{u}_rows.npz"))
        rows = RN.RD.rows_of(z)
        o, m = rows["obs"]["observation"], rows["obs"]["action_mask"].astype(bool)
        for c in ckpts:
            t0 = time.time()
            sd = RN.policy_sd(os.path.join(CACHE, d, f"checkpoint_{c}_steps.zip"))
            keep: Dict[str, Any] = {}
            TP.forward64(template, sd, o, m, 0.0, keep)
            s64 = th.as_tensor(keep["scores"])
            n = o.shape[0]
            s32 = s64.float()
            cand = th.as_tensor(keep["cand"]).bool()
            k = th.as_tensor(keep["k"])
            ref = solve(s32.double(), cand, k, th.float64).reshape(n, -1)[:, 0]      # fp64 root of the fp32 scores
            pi = th.sigmoid(s32.double() + th.as_tensor(ref).unsqueeze(-1))
            slope = th.where(cand, pi * (1 - pi), th.zeros(())).sum(-1).numpy()
            kk = k.numpy().reshape(n, -1)[:, 0]
            per: Dict[str, Any] = {"k_violator": int(kk[0]), "slope_violator": float(slope[0]),
                                   "fp32_band_violator": float(np.finfo(np.float32).eps * max(kk[0], 1) / slope[0]),
                                   "tau_fp64_violator": float(ref[0])}
            for name, dt in (("old_fp32_solve", th.float32), ("new_fp64_solve", th.float64)):
                taus = []
                for _ in range(a.perms):
                    p = th.as_tensor(rng.permutation(s32.shape[-1]))
                    taus.append(solve(s32[..., p], cand[..., p], k, dt).reshape(n, -1)[:, 0])
                taus = np.array(taus)
                spread = taus.max(0) - taus.min(0)
                eager = solve(s32, cand, k, dt).reshape(n, -1)[:, 0]
                comp = solve(s32, cand, k, dt, compiled=True).reshape(n, -1)[:, 0]
                ce = np.abs(eager - comp)
                per[name] = {
                    "order_spread_violator": float(spread[0]), "order_spread_others_max": float(spread[1:].max()),
                    "order_spread_others_median": float(np.median(spread[1:])),
                    "compiled_vs_eager_violator": float(ce[0]), "compiled_vs_eager_others_max": float(ce[1:].max()),
                    "err_vs_fp64_violator": float(np.abs(eager[0] - ref[0])),
                    "err_vs_fp64_max_over_orders_violator": float(np.abs(taus[:, 0] - ref[0]).max()),
                    "err_vs_fp64_others_max": float(np.abs(eager[1:] - ref[1:]).max())}
                r = per[name]
                print(f"u{u} ckpt {c} {name}: k={kk[0]} slope {slope[0]:.2e} band {per['fp32_band_violator']:.2e} | order spread "
                      f"v {r['order_spread_violator']:.2e} (others max {r['order_spread_others_max']:.1e}) | compiled-eager v "
                      f"{r['compiled_vs_eager_violator']:.2e} (others max {r['compiled_vs_eager_others_max']:.1e}) | |tau-tau64| v "
                      f"{r['err_vs_fp64_violator']:.2e} (max over orders {r['err_vs_fp64_max_over_orders_violator']:.2e})", flush=True)
            per["seconds"] = round(time.time() - t0, 1)
            res[f"u{u}@{c}"] = per
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
