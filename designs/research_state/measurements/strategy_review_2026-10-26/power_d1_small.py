"""The power of one more D1-frame fork (N1-P's registered rule, F alone, 1,200 games per seed final, look 1 -> look 2)
at the +1.0 / +1.5 pp effects the strategy draft guesses for its T2 forks. Reuses N1-P's registered Monte Carlo
(x61p_n1_production_2026-10-26/power/power.py: base F 0.49, numpy default_rng) with this review's seed 5354. CPU only.
    PYTHONPATH=src python power_d1_small.py > power_d1_small.log
"""
import importlib.util
from pathlib import Path

import numpy as np

_P = Path(__file__).resolve().parents[1] / "x61p_n1_production_2026-10-26" / "power" / "power.py"
_spec = importlib.util.spec_from_file_location("n1p_power", _P)
n1p = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(n1p)

DRAWS = 4000


def main():
    rng = np.random.default_rng(5354)
    print(f"N1-P's registered rule on F; {DRAWS} draws a cell; classes = FINAL class shares")
    for sigma in (0.0, 0.01):
        for delta in (0.01, 0.015):
            fin = {}
            for _ in range(DRAWS):
                _a, b = n1p.one(rng, delta, sigma)
                fin[b] = fin.get(b, 0) + 1
            f = {k: round(v / DRAWS, 3) for k, v in sorted(fin.items())}
            print(f"sigma {100 * sigma:.0f} pp, delta {100 * delta:+.1f} pp: final {f}")


if __name__ == "__main__":
    main()
