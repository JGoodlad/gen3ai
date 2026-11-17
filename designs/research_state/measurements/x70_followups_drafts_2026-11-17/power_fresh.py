"""DRAFTS (ii) and (iii): the power of ONE fresh cell against ONE banked BOUNDARY-1 cell, by simulation.

A frozen record of two DRAFTS, not a tool. The statistic is BOUNDARY-1's (PREREG §5.2) with one arm a side: the
classes and the 50M harm stop are ``main.b1.cross.classify`` and the floored within-arm variances
``main.b1.cross._arm_var``, imported, so the bar (+4 pp), the harm stop (-2 pp), the floor (1.76 pp), the level
(90 %) and the 0.005 pp boundary are B1's own. With one arm a side the general form
V = s2_rows / (2n) + s2_cols / (2n) on 4(n - 1) df becomes V = s2_on / n_on + s2_off / n_off on
(n_on - 1) + (n_off - 1) df. Whether that is the frame P5 registers is P5's (the queue proposal's §4).

The model: h[i][j] = 50 + D + a_i - b_j + e_ij, a, b ~ N(0, sigma^2) at the fresh-run seed SD, e ~ N(0, 0.68^2)
(B1 §11: the N view's cell SD at 1,000 home72 + 3 x 500 unfamiliar pairs). Rows = the new arm's runs, columns =
the banked cell's three.

    python power_fresh.py            -> power_fresh.log (20,000 draws a row, rng 20261117)
"""
from __future__ import annotations

import sys

import numpy as np

from main.b1.cross import FLOOR_PP, _arm_var, _stat, classify

CELL_SD = 0.68
REPS = 20000
SEED = 20261117
SIGMAS = (1.76, 2.5, 3.1)
DESIGNS = ((3, 3), (6, 3))
D_UP = (0.0, 2.0, 3.0, 4.0, 5.0, 6.0)
D_HARM = (0.0, -2.0, -4.0, -6.0)


def one(h: np.ndarray, look: str) -> tuple:
    n_on, n_off = h.shape
    rows = [f"on{i}" for i in range(n_on)]
    cols = [f"off{j}" for j in range(n_off)]
    g_on = _arm_var("ON", rows, [float(x) for x in h.mean(axis=1)], FLOOR_PP)
    g_off = _arm_var("OFF", cols, [float(x) for x in h.mean(axis=0)], FLOOR_PP)
    v = g_on.s2 / n_on + g_off.s2 / n_off
    stat = _stat("lever", "N", float(h.mean()) - 50.0, (g_on, g_off), v, (n_on - 1) + (n_off - 1), 0.90, FLOOR_PP)
    return classify(look, stat).outcome, stat.hi - stat.lo


def main() -> int:
    rng = np.random.default_rng(SEED)
    print(f"ONE fresh cell vs ONE banked cell, B1's rule one arm a side; reps {REPS}, rng {SEED}, cell SD {CELL_SD}")
    print("n_on n_off sigma | median half-width | P(WIN) at D = " + " / ".join(f"{d:+g}" for d in D_UP)
          + " | at D = 0: P(NO-USEFUL-GAIN) P(UNDECIDED) P(LOSS) | P(STOP-HARM at 50M) at D = "
          + " / ".join(f"{d:+g}" for d in D_HARM))
    for n_on, n_off in DESIGNS:
        for sig in SIGMAS:
            win, hw, zero = [], [], {}
            for d in D_UP:
                cnt = 0
                for _ in range(REPS):
                    a = rng.normal(0.0, sig, n_on)
                    b = rng.normal(0.0, sig, n_off)
                    h = 50.0 + d + a[:, None] - b[None, :] + rng.normal(0.0, CELL_SD, (n_on, n_off))
                    out, width = one(h, "100M")
                    cnt += out == "WIN"
                    if d == 0.0:
                        hw.append(width / 2.0)
                        zero[out] = zero.get(out, 0) + 1
                win.append(cnt / REPS)
            harm = []
            for d in D_HARM:
                cnt = 0
                for _ in range(REPS):
                    a = rng.normal(0.0, sig, n_on)
                    b = rng.normal(0.0, sig, n_off)
                    h = 50.0 + d + a[:, None] - b[None, :] + rng.normal(0.0, CELL_SD, (n_on, n_off))
                    cnt += one(h, "50M")[0] == "STOP-HARM"
                harm.append(cnt / REPS)
            z = {k: zero.get(k, 0) / REPS for k in ("NO-USEFUL-GAIN", "UNDECIDED", "LOSS")}
            print(f"{n_on:4d} {n_off:5d} {sig:5.2f} | {float(np.median(hw)):17.2f} | "
                  + " / ".join(f"{w:.2f}" for w in win)
                  + f" | {z['NO-USEFUL-GAIN']:.2f} {z['UNDECIDED']:.2f} {z['LOSS']:.3f} | "
                  + " / ".join(f"{x:.3f}" for x in harm))
    return 0


if __name__ == "__main__":
    sys.exit(main())
