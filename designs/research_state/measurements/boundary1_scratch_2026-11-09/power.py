"""Power of the boundary1_scratch DRAFT rules (DRAFT_PREREG §5-§6). Not a registered record: the draft's sizing.

(1) The head-to-head cross. k seeds an arm, a k x k matrix H (rows = the new arm's seeds, columns = the control's),
    H_ij = delta + a_i - b_j + e_ij in pp, a and b ~ N(0, sigma) (the seed-to-seed SD, seed_spread.log), e ~ N(0,
    CELL) (the cell noise at 1,000 mirrored pairs, seed_spread.log: 0.80-1.10). The statistic is
    `main.h2h.cross.cross_stat`'s: d = mean(H), V = s2_rows / k + s2_cols / k, 2 (k - 1) df, with each s2 FLOORED at
    FLOOR ** 2 (the 15M pooled seed SD), so a lucky small variance at small k cannot pass. Classes:
      WIN        the 90 % interval's lower end > 0;
      LOSS       its upper end < 0;
      NO-USEFUL  not WIN and the upper end < BAR (the smallest gain worth the arm's cost);
      UNDECIDED  otherwise.
(1b) The same eight runs crossed 2 x 2 (width x breadth, 2 seeds a cell; DRAFT_DESIGN §7). A factor's main effect is
    the mean score of its 4 ON runs against its 4 OFF runs (16 cells); its variance is the pooled WITHIN-cell
    variance of the runs' mean scores (4 df, each floored the same way) times 2 / 4. The other factor's true effect
    cancels in the contrast and does not enter the within-cell variance.
(2) Kakuna on U, analytic: the SE of (new - control) with k finals an arm, g games a final a panel on 3 panels, a
    per-final seed SD s_k on this meter, and a team-by-arm interaction SD TAU over the 216 shared teams.
Deterministic (numpy default_rng(20261109)). CPU only, seconds:
    python power.py > power.log
"""
import math

import numpy as np
from scipy import stats

CELL = 1.05
FLOOR = 1.76
BAR = 4.0
REPS = 50_000
TAU = 5.0


def cross(rng, k, sigma, delta):
    a = sigma * rng.standard_normal((REPS, k, 1))
    b = sigma * rng.standard_normal((REPS, 1, k))
    h = delta + a - b + CELL * rng.standard_normal((REPS, k, k))
    d = h.mean((1, 2))
    s2r = np.maximum(h.mean(2).var(1, ddof=1), FLOOR ** 2)
    s2c = np.maximum(h.mean(1).var(1, ddof=1), FLOOR ** 2)
    half = stats.t.ppf(0.95, 2 * (k - 1)) * np.sqrt((s2r + s2c) / k)
    win = d - half > 0
    loss = d + half < 0
    nouse = ~win & (d + half < BAR)
    return win.mean(), loss.mean(), (nouse & ~loss).mean(), np.median(half)


def factorial(rng, sigma, delta):
    run = sigma * rng.standard_normal((REPS, 2, 4)) + CELL / 2 * rng.standard_normal((REPS, 2, 4))
    d = delta + run[:, 0].mean(1) - run[:, 1].mean(1)
    pairs = run.reshape(REPS, 4, 2)
    s2 = np.maximum(pairs.var(2, ddof=1), FLOOR ** 2).mean(1)
    half = stats.t.ppf(0.95, 4) * np.sqrt(2 * s2 / 4)
    win = d - half > 0
    return win.mean(), (~win & (d + half < BAR) & ~(d + half < 0)).mean(), (d + half < 0).mean(), np.median(half)


def main():
    rng = np.random.default_rng(20261109)
    print(f"(1) head-to-head cross; cell SD {CELL}, variance floor {FLOOR} pp, NO-USEFUL bar +{BAR} pp, {REPS} reps")
    for k in (2, 3, 4, 5, 8):
        for sigma in (1.76, 2.5, 3.1):
            row = []
            for delta in (0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
                w, l, n, half = cross(rng, k, sigma, delta)
                row.append(f"{delta:+.0f}: WIN {w:.2f} NO-USEFUL {n:.2f} LOSS {l:.3f}")
            print(f"k {k} sigma {sigma:.2f} (median half-width {half:.2f} pp) | " + " | ".join(row))
    print("(1b) the 2 x 2 at 2 seeds a cell: one factor's main effect (4 runs v 4), pooled within-cell variance, 4 df")
    for sigma in (1.76, 2.5, 3.1):
        row = []
        for delta in (0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
            w, n, l, half = factorial(rng, sigma, delta)
            row.append(f"{delta:+.0f}: WIN {w:.2f} NO-USEFUL {n:.2f} LOSS {l:.3f}")
        print(f"sigma {sigma:.2f} (median half-width {half:.2f} pp) | " + " | ".join(row))
    print("(2) Kakuna on U: SE of (new - control), pp; MDE = (1.645 + 0.842) SE (one-sided 5 %, power 0.80)")
    for k in (2, 3, 4):
        for g in (360, 720):
            for s_k in (0.0, 1.0, 1.5, 2.5):
                binom = 2 * 0.25 * 1e4 / (k * 3 * g)
                seed = 2 * s_k ** 2 / k
                team = 2 * TAU ** 2 / 216 / k
                se = math.sqrt(binom + seed + team)
                print(f"k {k} games/final/panel {g} seed SD {s_k:.1f}: SE {se:.2f}, MDE {2.487 * se:.2f} pp, "
                      f"Kakuna games both arms {2 * k * 3 * g}")


if __name__ == "__main__":
    main()
