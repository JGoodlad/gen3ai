"""GATE — the era's KL copy must equal the gen era's imported `masked_kl_rows`, bit for bit.

Runs in the GEN tree (both functions importable there). A cross-era comparison of a statistic that
is not provably the SAME statistic is worthless, so this is a gate, not a nicety.

Run: python kl_unit_test.py     (exit 0 = identical on every case)
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import torch as th

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from era_kl import masked_kl_rows_era                                      # noqa: E402
from agents.training.instrumented_ppo.distill_anchor import masked_kl_rows  # noqa: E402


def main():
    g = th.Generator().manual_seed(20260905)
    cases = []
    # 1. random logits, random masks with >=2 legal actions
    for B, A in ((64, 11), (7, 11), (1, 11), (33, 5)):
        p = th.randn(B, A, generator=g) * 3.0
        q = th.randn(B, A, generator=g) * 3.0
        m = (th.rand(B, A, generator=g) > 0.4).float()
        m[:, 0] = 1.0; m[:, 1] = 1.0          # guarantee >=2 legal
        cases.append(("random", p, q, m))
    # 2. identical policies -> exactly 0
    p = th.randn(16, 11, generator=g)
    m = th.ones(16, 11)
    cases.append(("identical", p, p.clone(), m))
    # 3. an illegal action carrying a huge logit -- the case a naive KL gets wrong
    p = th.zeros(4, 11); q = th.zeros(4, 11)
    p[:, 5] = 50.0; q[:, 5] = -50.0
    m = th.zeros(4, 11); m[:, 0] = 1; m[:, 1] = 1
    cases.append(("illegal_dominant", p, q, m))
    # 4. one legal action only -> KL must be 0
    p = th.randn(5, 11, generator=g); q = th.randn(5, 11, generator=g)
    m = th.zeros(5, 11); m[:, 3] = 1
    cases.append(("single_legal", p, q, m))

    worst = 0.0
    for name, p, q, m in cases:
        a = masked_kl_rows(p, q, m)
        b = masked_kl_rows_era(p, q, m)
        d = float((a - b).abs().max())
        worst = max(worst, d)
        print(f"  {name:18s} B={tuple(p.shape)}  max|gen-era| = {d:.3e}  "
              f"mean KL {float(a.mean()):.6f}")
        if not th.equal(a, b):
            print(f"  !! {name}: NOT BIT-IDENTICAL")
            return 1
    print(f"\n  PASS — era copy is bit-identical to the imported masked_kl_rows on "
          f"{len(cases)} cases (worst |Δ| {worst:.3e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
