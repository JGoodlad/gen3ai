"""X61P PREREG §5 power: Monte Carlo of the REGISTERED rule (main.x35.d1.read_d1's primary for arm X61P: forum72
alone, 1,200 games per seed final, AM-5's widening, the sign rule, look 1 -> CONTINUE -> look 2), deterministic
(numpy default_rng(6161)). Base F score 0.49 (C0's own forum72 at D1's read: 342 / 700). Seed SD sigma is the
between-seed SD of a final's TRUE F score, per arm. CPU only; ~1 min.

    python designs/research_state/measurements/x61p_n1_production_2026-10-26/power/power.py > power.log
"""
import numpy as np

from main.x35 import d1 as D

BASE, N, DRAWS = 0.49, 1200, 4000


def finals(rng, mean, sigma, k):
    p = np.clip(mean + sigma * rng.standard_normal(k), 0.01, 0.99)
    return [int(x) for x in rng.binomial(N, p)]


def one(rng, delta, sigma):
    on = finals(rng, BASE + delta, sigma, 5)
    off = finals(rng, BASE, sigma, 5)
    cells = lambda w, k: {str(3501 + i): {"forum72": [w[i], N]} for i in range(k)}  # noqa: E731
    h = D.d1_primary(1, cells(on, 3), cells(off, 3), panels=("forum72",), units=60)
    if h.outcome != D.CONTINUE:
        return h.cls or D.INCONCLUSIVE, h.cls or D.INCONCLUSIVE
    h2 = D.d1_primary(2, cells(on, 5), cells(off, 5), panels=("forum72",), units=60)
    return "CONTINUE", h2.cls or D.INCONCLUSIVE


def main():
    rng = np.random.default_rng(6161)
    print(f"base {BASE}, {N} games per final, {DRAWS} draws per cell; classes = look-1 outcome / FINAL class")
    for sigma in (0.0, 0.01, 0.02, 0.03):
        for delta in (0.0, 0.02, 0.03, 0.04, 0.05):
            l1, fin = {}, {}
            for _ in range(DRAWS):
                a, b = one(rng, delta, sigma)
                l1[a] = l1.get(a, 0) + 1
                fin[b] = fin.get(b, 0) + 1
            f = {k: round(v / DRAWS, 3) for k, v in sorted(fin.items())}
            g = {k: round(v / DRAWS, 3) for k, v in sorted(l1.items())}
            print(f"sigma {100*sigma:.0f} pp, delta {100*delta:+.0f} pp: look1 {g} | final {f}")


if __name__ == "__main__":
    main()
