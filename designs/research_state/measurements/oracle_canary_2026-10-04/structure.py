"""Where the zero species head's fp32 gradient error lives (oracle fatal weights, live, CPU eager).

Captures dL/d(species logits) `d` [B*6, S] and the head's input `x` [B*6, D] in an fp32 and an fp64
eager R1 micro-step (repro.py's patches), then reports: the relative error of d, of x, and of
grad W = d^T x; how the error of d splits over labelled vs unlabelled (row, slot) terms and over
species columns. Usage: structure.py <B>"""
import sys

import torch as th

from agents.model import compile_regions as cr
from agents.training.instrumented_ppo.micro_step import micro_step

sys.path.insert(0, __file__.rsplit("/", 1)[0])
import repro as R  # noqa: E402


def run(B, dtype64):
    import stable_baselines3.common.policies as _sbp
    m = R.build(R.STATES["oracle_fatal"])
    if dtype64:
        m.policy.double()
    args = cr._r1_args(m, cr.r1_batch(m, B))
    if dtype64:
        args = tuple(R._to64(a) for a in args)
    real_pre, real_float, prev = _sbp.preprocess_obs, th.Tensor.float, th.get_default_dtype()
    if dtype64:
        th.set_default_dtype(th.float64)
        _sbp.preprocess_obs = lambda o, s, normalize_images=True: R._to64(real_pre(o, s, normalize_images))
        th.Tensor.float = lambda self, *a, **k: self.to(th.get_default_dtype())
    try:
        m.policy.set_training_mode(True)
        _, _, cap = R.grads(m, micro_step, args, hook=True)
    finally:
        th.Tensor.float, _sbp.preprocess_obs = real_float, real_pre
        th.set_default_dtype(prev)
    return cap["x"].double().reshape(-1, cap["x"].shape[-1]), cap["d"].double().reshape(-1, cap["d"].shape[-1])


def rel(a, b):
    return float((a - b).norm() / b.norm())


def main():
    B = int(sys.argv[1])
    th.set_float32_matmul_precision("highest")
    x32, d32 = run(B, False)
    x64, d64 = run(B, True)
    print(f"B={B}: rel err x {rel(x32, x64):.2e}  d {rel(d32, d64):.2e}  "
          f"gradW {rel(d32.t() @ x32, d64.t() @ x64):.2e}  (d32 with x64: {rel(d32.t() @ x64, d64.t() @ x64):.2e})")
    live = d64.abs().sum(-1) > 0
    print(f"terms with nonzero d: {int(live.sum())} of {d64.shape[0]}")
    err_row = (d32 - d64).norm(dim=-1)
    print(f"|d| per term: median {float(d64.norm(dim=-1)[live].median()):.3e}; err per term: median "
          f"{float(err_row[live].median()):.3e} max {float(err_row.max()):.3e}")
    col_err = (d32 - d64).norm(dim=0)
    col_n = d64.norm(dim=0)
    top = th.argsort(col_err, descending=True)[:8]
    print("worst species columns (col, err, |d64| col norm, d64 col sum):")
    for c in top.tolist():
        print(f"  {c:4d}  {float(col_err[c]):.3e}  {float(col_n[c]):.3e}  {float(d64[:, c].sum()):+.3e}")
    share = float((col_err[top] ** 2).sum() / (col_err ** 2).sum())
    print(f"top-8 columns carry {share:.1%} of d's squared error; the gradient's top-8 rows of W carry "
          f"{float(((d32 - d64).t() @ x64).norm(dim=-1)[top].pow(2).sum() / ((d32 - d64).t() @ x64).pow(2).sum()):.1%}")
    rel_entry = ((d32 - d64).abs() / d64.abs().clamp_min(1e-300))[live]
    print(f"per-entry relative error of d on live terms: median {float(rel_entry.median()):.2e} "
          f"p99 {float(rel_entry.quantile(0.99) if rel_entry.numel() < 2**24 else float('nan')):.2e}")


if __name__ == "__main__":
    main()
