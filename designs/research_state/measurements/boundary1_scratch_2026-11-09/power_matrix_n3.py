"""BOUNDARY-1's 2 x 2, meter 1, at THREE runs an arm: Amendment 1's operating characteristics (PREREG §11, §18).

``power_matrix.py`` is the registration's script: it simulates a look cell by cell and applies the rule twice (a
numpy copy of the formula, and ``main.b1.cross`` itself on the first draws of every row), and it sized the design at
two runs an arm, with three runs only in its parts (1) and (3). Amendment 1 registers three runs an arm up front, so
this script repeats EVERY part that decides or stops something at ``n = 3``, with the registration's own helpers
(imported from ``power_matrix.py``, not copied): the same score model, the same cell SDs, the same floor, bar and
level, the same agreement check (0 mismatches or exit 1). What changes with ``n``: 54 cells a look (24), 8 degrees of
freedom (4), and after a 50M stop a survivor look of 3 against 3 on 4 degrees of freedom (2 against 2 on 2).

Part (10) of the registration (the two-stage extension from two runs to three) has no twin here: Amendment 1
withdraws the extension, so the three-run class is read ONCE and part (1)'s false-WIN column is the whole rate.

Deterministic (numpy default_rng(20261110)); CPU only, a few minutes:
    PYTHONPATH=<checkout>/src python power_matrix_n3.py > power_matrix_n3.log
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import power_matrix as P  # noqa: E402  (the registration's helpers, from this folder)

from main.b1 import cross as X  # noqa: E402

N = 3
SEED = 20261110
SIGMAS = (1.76, 2.5, 3.1)


def main() -> int:
    rng = np.random.default_rng(SEED)
    cs, frac = P.cell_sd, P.frac
    print("cell SD by view (pp): " + ", ".join(f"{v} {cs(v):.2f}" for v in X.VIEWS))
    print(f"n {N} runs an arm: {len(X.design_of({a: tuple(f'{a}{k}' for k in range(N)) for a in X.ARMS}).cells())} "
          f"cells a look, {4 * (N - 1)} df; floor {X.FLOOR_PP} pp, bar +{X.BAR_PP} pp, harm stop {X.HARM_PP} pp, "
          f"level {X.LEVEL}, band {X.EPS_PP} pp; {P.REPS} draws a row, the first {P.CHECK} also through main.b1.cross")
    print("\n(1) ONE FACTOR'S MAIN EFFECT at 100M, view N (the other factor at +3 pp, no interaction)")
    for sigma in SIGMAS:
        row = []
        for delta in (0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
            f = P.simulate(rng, N, sigma, cs("N"), {"C": 0.0, "W": delta, "Br": 3.0, "WB": delta + 3.0})
            P.check(f, N)
            d, half = P.factor_np(f, N, "width")
            c = P.classes_100(d, half)
            row.append(f"{delta:+.0f}: WIN {frac(c, X.WIN):.2f} NUG {frac(c, X.NO_USEFUL_GAIN):.2f} "
                       f"UND {frac(c, X.UNDECIDED):.2f} LOSS {frac(c, X.LOSS):.3f}")
        print(f"sigma {sigma:.2f} (median half-width {np.median(half):.2f} pp) | " + " | ".join(row))
    print("\n(2) THE 50M HARM STOP, view N: P(STOP-HARM) at a true effect of")
    for sigma in SIGMAS:
        row = []
        for delta in (0.0, -2.0, -4.0, -5.0, -6.0, -8.0):
            f = P.simulate(rng, N, sigma, cs("N"), {"C": 0.0, "W": delta, "Br": 0.0, "WB": delta})
            P.check(f, N)
            d, half = P.factor_np(f, N, "width")
            row.append(f"{delta:+.0f}: {frac(P.classes_50(d, half), X.STOP_HARM):.3f}")
        print(f"sigma {sigma:.2f} | " + " | ".join(row))
    print("\n(3) THE INTERACTION (REPORTED), view N: P(SYNERGY) / P(ANTAGONISM) at a true interaction of")
    for sigma in (1.76, 2.5):
        row = []
        for inter in (0.0, 4.0, 6.0, 8.0, 10.0):
            f = P.simulate(rng, N, sigma, cs("N"), {"C": 0.0, "W": 2.0, "Br": 1.0, "WB": 3.0 + inter})
            P.check(f, N)
            _e, half, c = P.interaction_np(f, N)
            row.append(f"{inter:+.0f}: {frac(c, X.SYNERGY):.2f} / {frac(c, X.ANTAGONISM):.3f}")
        print(f"sigma {sigma:.2f} (median half-width {np.median(half):.2f} pp) | " + " | ".join(row))
    print("\n(4) ONE FACTOR ON A SINGLE VIEW at 100M (forum72 DECIDES breadth; U and home72 are REPORTED)")
    for view in ("forum72", "U", "home72"):
        for sigma in SIGMAS:
            row = []
            for delta in (0.0, 2.0, 4.0, 6.0, 8.0):
                f = P.simulate(rng, N, sigma, cs(view), {"C": 0.0, "W": 0.0, "Br": delta, "WB": delta})
                P.check(f, N)
                d, half = P.factor_np(f, N, "breadth")
                c = P.classes_100(d, half)
                row.append(f"{delta:+.0f}: WIN {frac(c, X.WIN):.2f} NUG {frac(c, X.NO_USEFUL_GAIN):.2f} "
                           f"UND {frac(c, X.UNDECIDED):.2f}")
            print(f"{view} sigma {sigma:.2f} (median half-width {np.median(half):.2f} pp) | " + " | ".join(row))
    print("\n(6) BREADTH AS REGISTERED at 100M: its class on forum72, then the net rule (home72 + forum72 > 0). "
          "P(WIN = adoptable) / P(WIN-NET-NEGATIVE) at a true (home72, forum72) effect of")
    for sigma in (1.76, 2.5):
        for rho in (1.0, 0.0):
            row = []
            for d_h, d_f in ((0.0, 0.0), (0.0, 4.0), (-2.0, 4.0), (-4.0, 4.0), (-6.0, 4.0), (-3.28, 2.0), (-3.0, 6.0),
                             (-6.0, 6.0), (-9.0, 6.0), (-6.0, 0.0)):
                fh, ff = P.simulate2(rng, N, sigma, rho, d_h, d_f)
                P.check_look(fh, ff, N)
                c, _c50 = P.breadth_np(fh, ff, N)
                row.append(f"({d_h:+.2f}, {d_f:+.0f}): {frac(c, X.WIN):.2f} / {frac(c, X.WIN_NET_NEGATIVE):.2f}")
            print(f"sigma {sigma:.2f} rho {rho:.0f} | " + " | ".join(row))
    print("\n(7) BREADTH'S 50M HARM STOP on the net view HF = 1/2 home72 + 1/2 forum72 (cell SD "
          f"{cs('HF'):.2f} pp): P(STOP-HARM) at a true (home72, forum72) effect of")
    for sigma in (1.76, 2.5):
        for rho in (1.0, 0.0):
            row = []
            for d_h, d_f in ((0.0, 0.0), (-4.0, 4.0), (-4.0, 0.0), (-8.0, 0.0), (-12.0, 0.0), (-16.0, 0.0), (-6.0, -6.0),
                             (-12.0, 4.0)):
                fh, ff = P.simulate2(rng, N, sigma, rho, d_h, d_f)
                P.check_look(fh, ff, N)
                _c, c50 = P.breadth_np(fh, ff, N)
                row.append(f"({d_h:+.0f}, {d_f:+.0f}): {frac(c50, X.STOP_HARM):.3f}")
            print(f"sigma {sigma:.2f} rho {rho:.0f} | " + " | ".join(row))
    print("\n(8) AFTER A 50M STOP-HARM OF THE OTHER FACTOR: the surviving factor is one arm against one arm (3 v 3, "
          "4 df). REPORTED: the registration reads NO class there (PREREG §8); what the +4 pp bar WOULD give, "
          "P(WIN) / P(NO-USEFUL-GAIN), at a true effect of")
    for view in ("N", "forum72"):
        for sigma in (1.76, 2.5):
            row = []
            for delta in (0.0, 4.0, 6.0, 8.0, 10.0):
                f = P.simulate(rng, N, sigma, cs(view), {"C": 0.0, "W": delta, "Br": 0.0, "WB": 0.0})
                d, half = P.simple_np(f, N, "W", "C")
                c = P.classes_100(d, half)
                P.check_simple(f, N, "W", "C", c)
                row.append(f"{delta:+.0f}: {frac(c, X.WIN):.2f} / {frac(c, X.NO_USEFUL_GAIN):.2f}")
            print(f"{view} sigma {sigma:.2f} (median half-width {np.median(half):.2f} pp) | " + " | ".join(row))
    print("\n(9) UNEQUAL SEED SPREADS (the interval assumes one spread for every arm), view N, NO true effect: "
          "P(false WIN) / P(false LOSS) / P(NO-USEFUL-GAIN) at a seed SD (pp) of (C, W, Br, WB)")
    for sds in ((1.76, 1.76, 1.76, 1.76), (1.0, 6.0, 1.0, 1.0), (0.5, 8.0, 0.5, 0.5), (1.0, 6.0, 1.0, 6.0),
                (6.0, 1.0, 6.0, 1.0), (1.0, 1.0, 6.0, 6.0)):
        f = P.simulate(rng, N, np.repeat(np.array(sds), N), cs("N"), {a: 0.0 for a in X.ARMS})
        P.check(f, N)
        d, half = P.factor_np(f, N, "width")
        c = P.classes_100(d, half)
        print(f"{sds} | width: {frac(c, X.WIN):.3f} / {frac(c, X.LOSS):.3f} / {frac(c, X.NO_USEFUL_GAIN):.2f} "
              f"(median half-width {np.median(half):.2f} pp)")
    print(f"\nRULE CODE vs THIS FILE'S FORMULA: {P.MISMATCH[0]} mismatches in {P.MISMATCH[1]} classes")
    return 1 if P.MISMATCH[0] else 0


if __name__ == "__main__":
    sys.exit(main())
