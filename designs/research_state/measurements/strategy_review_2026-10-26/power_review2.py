"""REVIEW_2's checks of PR-1 (strategy_if_levers_null_2026-10-26.md section 3.1), 2026-10-27.

Reuses the revision's simulator unchanged (strategy_revision_2026-10-26/power_precision_read.py: binomial games per
seed final, seed effects N(0, sigma_s), AM-5's phi widening, the futility interim, rule 8). Three questions:
  1. How often does the REGISTERED estimand (D1, D3, N1-P vs C0 on forum72) read BREADTH-GAIN when only D1 moves F,
     i.e. when the whole "effect" is D1's own forum population (D1's corpus is forum-sourced; F is forum72)?
  2. The leave-D1-out estimand (D3 and N1-P vs C0, both unhazarded on F) at the same and at fewer games.
  3. The C0 allocation: the doc's "sqrt(3)x a breadth arm" against the pooled-mean optimum (C0 = k x a breadth arm).
Deterministic (default_rng(27102026)). CPU only, about a minute:
    python power_review2.py > power_review2.log
"""
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
SRC = HERE.parent / "strategy_revision_2026-10-26" / "power_precision_read.py"
spec = importlib.util.spec_from_file_location("ppr", SRC)
ppr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ppr)


def var_ratio(k, n_b, n_c):
    """Binomial-only variance of the pooled mean contrast, in units of p(1-p)."""
    return 1.0 / (k * n_b) + 1.0 / n_c


def main():
    rng = np.random.default_rng(27102026)
    print(f"{ppr.DRAWS} draws a cell; base F {ppr.BASE}; KILL bar +1.5 pp; effects in pp; simulator {SRC.name}")

    print("\n## 1. The registered design (3 x 8,000 + C0 14,000), seeds D1 5 / D3 5 / N1-P 3 / C0 5: only D1 moves F")
    for sigma in (0.0, 0.005, 0.01):
        for d1 in (0.02, 0.03, 0.04):
            r = ppr.simulate(rng, (d1, 0.0, 0.0), (5, 5, 3, 5), 8000, 14000, sigma)
            print(f"  sigma {100 * sigma:.1f}  D1 {100 * d1:+.1f} / D3 0 / N1-P 0 (family mean {100 * d1 / 3:+.2f}): "
                  + ppr.fmt(r))

    print("\n## 2. Leave-D1-out primary: D3 (5 seeds) and N1-P (3 seeds) vs C0 (5 seeds)")
    designs = {
        "2 x 8,000 + C0 14,000 = 30,000 (the registered arms minus D1)": (8000, 14000),
        "2 x 7,500 + C0 15,000 = 30,000 (C0 = 2 x an arm, the pooled optimum)": (7500, 15000),
        "2 x 9,500 + C0 19,000 = 38,000 (the registered budget, optimum)": (9500, 19000),
    }
    for name, (nb, nc) in designs.items():
        for sigma in (0.0, 0.005, 0.01):
            print(f"  ### {name}; sigma {100 * sigma:.1f} pp")
            for dl in (0.0, 0.01, 0.015, 0.02):
                r = ppr.simulate(rng, (dl, dl), (5, 3, 5), nb, nc, sigma)
                print(f"    each {100 * dl:+.1f}: " + ppr.fmt(r))

    print("\n## 3. C0 allocation for the 3-arm pooled mean at 38,000 games (binomial variance only, x p(1-p))")
    for label, nb, nc in (("registered 3 x 8,000 + 14,000", 8000, 14000),
                          ("sqrt(3) rule 3 x 7,610 + 13,170", 7610, 13170),
                          ("optimum C0 = 3 x arm: 3 x 6,333 + 19,000", 6333, 19000)):
        v = var_ratio(3, nb, nc)
        print(f"  {label:42s} var {v:.4e}  SE ratio to optimum {np.sqrt(v / var_ratio(3, 6333, 19000)):.4f}")
    n_eq = 4.0 / var_ratio(3, 8000, 14000)
    print(f"  games the optimum needs for the registered design's variance: {n_eq:,.0f} (vs 38,000)")
    r = ppr.simulate(rng, (0.015,) * 3, (5, 5, 3, 5), 6333, 19000, 0.0)
    print("  optimum allocation, each +1.5, sigma 0: " + ppr.fmt(r))


if __name__ == "__main__":
    main()
