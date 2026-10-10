"""The noisy-OR's `torch.prod` BACKWARD, eager vs Inductor on CUDA, against float64 (gpu_checks_endstate, 2026-10-09).

    python prod_backward_probe.py

`op_reduction.noisy_or` = 1 − Π_c (1 − p_c). Eager `prod`'s backward takes an exact zero-aware path when an
input is 0 (a certain KO: p_c = 1); a decomposition may not. Cases: no zero, exactly one zero, two zeros, values
near 1 (1 − p tiny), mixed — each [B, 6, C] like the op's, a random upstream gradient; per case the relative
error of the gradient wrt p, eager fp32 and compiled fp32 against eager float64 (CPU), and any non-finite.
"""
import json

import torch as th

from agents.model.op_reduction import noisy_or


def cases(B=2048, C=12, seed=0):
    g = th.Generator().manual_seed(seed)
    base = th.rand(B, 6, C, generator=g) * 0.5
    out = {"no_zero": base.clone()}
    one = base.clone()
    one[: B // 2, :, 3] = 1.0
    out["one_zero"] = one
    two = base.clone()
    two[: B // 2, :, 3] = 1.0
    two[: B // 4, :, 7] = 1.0
    out["two_zeros"] = two
    near = base.clone()
    near[:, :, 5] = 1.0 - th.rand(B, 6, generator=g) * 1e-6
    out["near_one"] = near
    sparse = th.zeros(B, 6, C)
    m = th.rand(B, 6, C, generator=g) < 0.1
    sparse[m] = th.rand(int(m.sum()), generator=g)
    sparse[: B // 8, :, 0] = 1.0
    out["sparse_with_certain"] = sparse
    return out


def grad(fn, p, G, dev, dtype):
    x = p.to(dev, dtype).clone().requires_grad_(True)
    y = fn(x)
    (y * G.to(dev, dtype)).sum().backward()
    return y.detach().double().cpu(), x.grad.detach().double().cpu()


def main():
    th.set_float32_matmul_precision("highest")
    from agents.model.compile_cache import ensure_hermetic_cache
    ensure_hermetic_cache("gpu checks endstate prod probe")
    dev = th.device("cuda")
    th._dynamo.reset()
    comp = th.compile(noisy_or, fullgraph=True, dynamic=False)
    res = {}
    for name, p in cases().items():
        G = th.randn(p.shape[:-1], generator=th.Generator().manual_seed(1))
        y64, g64 = grad(noisy_or, p, G, "cpu", th.float64)
        ye, ge = grad(noisy_or, p, G, dev, th.float32)
        yc, gc = grad(comp, p, G, dev, th.float32)

        def rel(a, b):
            return float((a - b).norm() / b.norm()) if float(b.norm()) > 0 else None
        res[name] = {"fwd_eager_vs_64": rel(ye, y64), "fwd_comp_vs_64": rel(yc, y64),
                     "grad_eager_vs_64": rel(ge, g64), "grad_comp_vs_64": rel(gc, g64),
                     "grad_comp_vs_eager": rel(gc, ge),
                     "grad_comp_nonfinite": int((~th.isfinite(gc)).sum()),
                     "grad_eager_nonfinite": int((~th.isfinite(ge)).sum())}
        print(name, json.dumps({k: (f"{v:.3e}" if isinstance(v, float) else v) for k, v in res[name].items()}),
              flush=True)
    print(json.dumps(res))


if __name__ == "__main__":
    main()
