"""WHY a bf16 forward under grad differs from the same forward under no_grad (CPU): one trunk round at its real
inputs, under the production bf16 class, grad-on (the round's INPUT requiring grad, as in the probe forward) vs no-grad, with the attention kernel left to torch or forced
to MATH, and each sub-op (the Linears, the attention) compared alone.

    python diag_grad_mode.py --out diag_grad_mode.json [--device cpu]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict

import torch as th

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    if a.device == "cpu":
        th.set_num_threads(a.threads)
    from torch.nn.attention import SDPBackend, sdpa_kernel

    from agents.model.trunk_precision import Bf16BiasedEncoderLayer, Bf16IdentityInitRound

    model, env, _ = C.build(state="perturb05", weights_from=None, seed=1001, n_envs=8, n_steps=64, run_seed=1001,
                            device=a.device, t2_backend="eager", trunk="bf16")
    res: Dict[str, Any] = {"device": a.device, "torch": th.__version__}
    try:
        C.collect(model, env)
        R = C.rows(model)
        pol = model.policy
        rounds = [m for m in pol.modules() if type(m) in (Bf16BiasedEncoderLayer, Bf16IdentityInitRound)]
        caps: Dict[int, Any] = {}
        hs = [m.register_forward_pre_hook(lambda mod, args, i=i: caps.setdefault(i, tuple(
            t.detach().clone() if isinstance(t, th.Tensor) else t for t in args))) for i, m in enumerate(rounds)]
        C.forward(model, R, chunk=R["n"], train=False, grad=False)
        for h in hs:
            h.remove()
        for i, rnd in enumerate(rounds):
            x, bias = (caps[i] + (None,))[:2]
            rnd.eval()
            out: Dict[str, Any] = {"class": type(rnd).__name__}

            def run(grad: bool, kernel: Any = None) -> th.Tensor:
                ctx = th.enable_grad() if grad else th.no_grad()
                with ctx:
                    xx = x.clone().requires_grad_(True) if grad else x    # the probe's input carries grad
                    if kernel is None:
                        return rnd(xx, bias).detach()
                    with sdpa_kernel([kernel]):
                        return rnd(xx, bias).detach()
            g, ng = run(True), run(False)
            out["grad_vs_nograd_max"] = float((g - ng).abs().max())
            gm, ngm = run(True, SDPBackend.MATH), run(False, SDPBackend.MATH)
            out["math_kernel_grad_vs_nograd_max"] = float((gm - ngm).abs().max())
            out["default_vs_math_nograd_max"] = float((ng - ngm).abs().max())
            out["default_vs_math_grad_max"] = float((g - gm).abs().max())
            # a Linear alone, under autocast, grad vs no grad
            lin = next((m for m in rnd.modules() if isinstance(m, th.nn.Linear) and m.in_features == x.shape[-1]), None)
            out["linears"] = [type(m).__name__ for m in rnd.modules() if isinstance(m, th.nn.Linear)]
            xin = x.reshape(-1, x.shape[-1])
            if lin is not None:
                with th.autocast(device_type=x.device.type, dtype=th.bfloat16):
                    with th.enable_grad():
                        lg = lin(xin).detach().float()
                    with th.no_grad():
                        lng = lin(xin).float()
                out["linear_grad_vs_nograd_max"] = float((lg - lng).abs().max())
            res[f"round_{i}"] = out
        with open(a.out, "w") as f:
            json.dump(res, f, indent=1)
        print(json.dumps(res, indent=1))
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
