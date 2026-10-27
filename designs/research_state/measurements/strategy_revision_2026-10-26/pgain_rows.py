"""The strategy's P(GAIN) column and its honest pool (strategy_if_levers_null_2026-10-26.md §2, §3), 2026-10-26.

1. D1's registered rule on F (N1-P's power.py, reused: 1,200 F games a seed final, look 1 -> look 2, AM-5, the
   sign rule) at a grid of true effects; then each ranked row's P(GAIN) = E[power(delta)] over its GUESS, the GUESS
   read as a normal with the row's point as mean and its 80 % interval's width / 2.563 as SD (a stated
   approximation). sigma (fork seed SD) 0 and 1 pp.
   Beside it, the same row read by the PR rule (§3.1's interval, one arm vs a same-engine C0 at 17,400 F games each,
   5 seeds; AM-5 not modelled): P(GAIN | d) = Phi(d / SE - 1.96), SE = sqrt(2 * 0.25 / 17,400), integrated in closed
   form over the same normal GUESS.
2. M6's pools: inverse-variance means of the published estimates (SE = 95 % width / 3.92), the selected four and
   the six in scope. Neither is an estimate of one effect (the levers differ); printed to show the selection.
Deterministic (default_rng(2627)). CPU only, a few minutes:
    PYTHONPATH=src python pgain_rows.py > pgain_rows.log
"""
import importlib.util
import math
from pathlib import Path

import numpy as np

_P = Path(__file__).resolve().parents[1] / "x61p_n1_production_2026-10-26" / "power" / "power.py"
_spec = importlib.util.spec_from_file_location("n1p_power", _P)
n1p = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(n1p)

DRAWS = 1500
GRID = (-0.02, -0.01, 0.0, 0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08)
ROWS = {  # name: (point, lo80, hi80) in pp, the revised doc's GUESSES
    "N2 set-channel noise": (1.5, 0.0, 3.0),
    "breadth bet, plateau stack (after PR-1 GAIN)": (1.5, 0.0, 3.5),
    "breadth bet, plateau stack (no PR-1)": (1.0, -1.5, 3.5),
    "H-TEAMS (X60)": (1.0, -1.0, 4.0),
    "N3 regret curriculum": (0.5, 0.0, 3.0),
    "N5 offline breadth student, from scratch": (1.0, -2.0, 5.0),
    "fork extension N1-P +40M": (1.0, -1.0, 3.0),
    "batch x2 with breadth": (0.5, -1.0, 3.0),
    "specialist distillation K~100": (1.5, -1.0, 4.0),
}


def power_curve(rng, sigma):
    out = {}
    for d in GRID:
        g = 0
        for _ in range(DRAWS):
            _a, b = n1p.one(rng, d, sigma)
            g += b == "GAIN"
        out[d] = g / DRAWS
    return out


def expected(curve, mu, sd):
    xs = np.array(sorted(curve))
    ys = np.array([curve[x] for x in xs])
    z = np.linspace(-4, 4, 801)
    w = np.exp(-z * z / 2)
    w /= w.sum()
    d = (mu + sd * z) / 100
    return float((np.interp(d, xs, ys) * w).sum())


def main():
    rng = np.random.default_rng(2627)
    print(f"## 1. D1's rule on F, final GAIN, {DRAWS} draws a cell")
    curves = {}
    for sigma in (0.0, 0.01):
        curves[sigma] = power_curve(rng, sigma)
        print(f"sigma {100 * sigma:.0f} pp: " + ", ".join(f"{100 * d:+.1f}: {p:.3f}" for d, p in curves[sigma].items()))
    print("\n## P(GAIN) per row (sigma 0 / 1 pp)")
    for name, (pt, lo, hi) in ROWS.items():
        sd = (hi - lo) / 2.563
        se = 100 * math.sqrt(2 * 0.25 / 17400)
        pr = 0.5 * (1 + math.erf((pt - 1.959963984540054 * se) / math.sqrt(se * se + sd * sd) / math.sqrt(2)))
        print(f"  {name:48s} GUESS {pt:+.1f} ({lo:+.1f}, {hi:+.1f}): D1 rule "
              f"{expected(curves[0.0], pt, sd):.2f} / {expected(curves[0.01], pt, sd):.2f} | PR rule {pr:.2f}")
    print("\n## 2. M6's pools (inverse-variance; not an estimate of one effect)")
    est = {"D1 H": (2.10, -0.92, 5.11), "D3 look 1 H": (1.27, -2.63, 5.16), "X52 unpractised": (1.3, -3.5, 6.1),
           "belief lambda 0.25": (1.50, -2.74, 5.74), "belief lambda 1": (-1.50, -5.47, 2.47),
           "X26c": (0.08, -3.90, 4.06)}
    for label, keys in (("the selected four", list(est)[:4]), ("all six in scope", list(est))):
        w = [1 / ((est[k][2] - est[k][1]) / 3.92) ** 2 for k in keys]
        m = sum(wi * est[k][0] for wi, k in zip(w, keys)) / sum(w)
        se = math.sqrt(1 / sum(w))
        print(f"  {label}: {m:+.2f} pp, SE {se:.2f}, [{m - 1.96 * se:+.2f}, {m + 1.96 * se:+.2f}]")


if __name__ == "__main__":
    main()
