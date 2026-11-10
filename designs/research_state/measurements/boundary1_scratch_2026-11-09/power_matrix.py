"""BOUNDARY-1's 2 x 2, meter 1: the operating characteristics of the REGISTERED rule code (PREREG §5, §8, §11).

``power.py`` sized the draft at RUN level (a run = one number). This script simulates the look as it is played, CELL
by cell: ``n`` runs an arm, every pair of runs from different arms once, a cell's score

    h(p, o) = 50 + strength(p) - strength(o) + e,    strength = the arm's effect + a run's own N(0, sigma),

``e ~ N(0, cell SD of the view)``, and applies the rule two ways: a vectorised numpy copy of the formula on every
draw, and ``main.b1.cross`` itself (``factor_stat`` / ``classify`` / ``interaction_stat`` / ``interaction_class``) on
the first ``CHECK`` draws of every row. The two must agree on EVERY class (0 mismatches) or the script exits 1: that
shows the simulated numbers are the shipped code's, and it is the SAME formula transcribed, never an independent
validation of the formula.
Parts (6) and (7) are BREADTH AS REGISTERED: its class on forum72 with the net rule on home72 + forum72, and its 50M
harm stop on the net view HF, with the first ``CHECK_LOOK`` draws of every row through ``read_look`` end to end. A
run's own strength there is one draw shared by the two panels (``rho`` 1) or one draw a panel (``rho`` 0).
Part (9) breaks the interval's one assumption (every arm has the same seed spread); part (10) is the two-stage
false-WIN rate of the registered extension.

The cell SD of a view follows from ``seed_spread.log``'s 1.05 pp at 1,000 mirrored pairs: 1.05 on home72, 1.05 *
sqrt(2) on a 500-pair panel, and for a weighted view the root of the weighted sum of squares (N: 0.68 pp).
``sigma`` is the seed-to-seed SD of a run's strength ON THAT VIEW: 1.76 pp is measured (the pool, 15M); 2.5 and 3.1
bracket it; nothing is measured on forum72 alone.

Deterministic (numpy default_rng(20261109)); CPU only, about a minute:
    PYTHONPATH=<checkout>/src python power_matrix.py > power_matrix.log
"""
from __future__ import annotations

import math
import sys

import numpy as np
from scipy import stats

from main.b1 import cross as X

REPS = 50_000
CHECK = 1_000
CHECK_LOOK = 200
CELL_1000 = 1.05
SEED = 20261109


def cell_sd(view: str) -> float:
    return math.sqrt(sum((w * CELL_1000 * math.sqrt(1000.0 / X.PAIRS[p])) ** 2 for p, w in X.VIEWS[view].items()))


def simulate(rng: np.random.Generator, n: int, sigma: float, cell: float, eff: dict) -> np.ndarray:
    """F[rep, i, j] = run i's score against run j (runs in ARMS order, n an arm); 50 on same-arm pairs (unused)."""
    r = 4 * n
    arm = np.repeat([eff["C"], eff["W"], eff["Br"], eff["WB"]], n)
    s = arm + sigma * rng.standard_normal((REPS, r))
    e = np.triu(cell * rng.standard_normal((REPS, r, r)), 1)
    e = e - np.transpose(e, (0, 2, 1))
    return 50.0 + s[:, :, None] - s[:, None, :] + e


def arm_idx(n: int, arm: str) -> list:
    k = X.ARMS.index(arm)
    return list(range(k * n, k * n + n))


def floored_group_var(means: np.ndarray, n: int) -> np.ndarray:
    """means[rep, 2n] (two arms of n) -> the mean of the two floored within-arm variances."""
    g = means.reshape(means.shape[0], 2, n)
    return np.maximum(g.var(2, ddof=1), X.FLOOR_PP ** 2).mean(1)


def factor_np(f: np.ndarray, n: int, factor: str) -> tuple:
    on, off = X.FACTORS[factor]
    ri = [i for a in on for i in arm_idx(n, a)]
    ci = [i for a in off for i in arm_idx(n, a)]
    h = f[:, ri][:, :, ci]
    d = h.mean((1, 2)) - 50.0
    v = floored_group_var(h.mean(2), n) / (2 * n) + floored_group_var(h.mean(1), n) / (2 * n)
    half = stats.t.ppf(0.5 + X.LEVEL / 2.0, 4 * (n - 1)) * np.sqrt(v)
    return d, half


def classes_100(d: np.ndarray, half: np.ndarray) -> np.ndarray:
    lo, hi = d - half, d + half
    out = np.full(d.shape, X.UNDECIDED, dtype=object)
    win = lo > X.EPS_PP
    loss = ~win & (hi < -X.EPS_PP)
    nug = ~win & ~loss & (hi < X.BAR_PP - X.EPS_PP)
    out[nug], out[loss], out[win] = X.NO_USEFUL_GAIN, X.LOSS, X.WIN
    return out


def classes_50(d: np.ndarray, half: np.ndarray) -> np.ndarray:
    return np.where(d + half < X.HARM_PP - X.EPS_PP, X.STOP_HARM, X.CONTINUE).astype(object)


def interaction_np(f: np.ndarray, n: int) -> tuple:
    i = {a: arm_idx(n, a) for a in X.ARMS}

    def m(a: str, b: str) -> np.ndarray:
        return f[:, i[a]][:, :, i[b]]

    est = 0.5 * ((m("WB", "Br").mean((1, 2)) - m("W", "C").mean((1, 2)))
                 + (m("WB", "W").mean((1, 2)) - m("Br", "C").mean((1, 2))))
    u = {"WB": 0.5 * (m("WB", "Br").mean(2) + m("WB", "W").mean(2)), "C": 0.5 * (m("C", "W").mean(2) + m("C", "Br").mean(2)),
         "W": 0.5 * (m("W", "C").mean(2) + m("W", "WB").mean(2)), "Br": 0.5 * (m("Br", "C").mean(2) + m("Br", "WB").mean(2))}
    v = sum(np.maximum(u[a].var(1, ddof=1), X.FLOOR_PP ** 2) for a in X.ARMS) / n
    half = stats.t.ppf(0.5 + X.LEVEL / 2.0, 4 * (n - 1)) * np.sqrt(v)
    cls = np.full(est.shape, X.NOT_SHOWN, dtype=object)
    cls[est - half > X.EPS_PP] = X.SYNERGY
    cls[est + half < -X.EPS_PP] = X.ANTAGONISM
    return est, half, cls


def rule_code(f: np.ndarray, n: int) -> tuple:
    """The same draws through ``main.b1.cross``: (width 100M, width 50M, breadth 100M, interaction) classes."""
    design = X.design_of({a: tuple(f"{a}{k}" for k in range(n)) for a in X.ARMS})
    idx = {r: j for j, r in enumerate(r for a in X.ARMS for r in design.runs[a])}
    cells = design.cells()
    out: tuple = ([], [], [], [])
    for rep in range(f.shape[0]):
        scores = {(p, o): float(f[rep, idx[p], idx[o]]) for p, o in cells}
        w = X.factor_stat(design, scores, "width")
        out[0].append(X.classify("100M", w).outcome)
        out[1].append(X.classify("50M", w).outcome)
        out[2].append(X.classify("100M", X.factor_stat(design, scores, "breadth")).outcome)
        out[3].append(X.interaction_class(X.interaction_stat(design, scores))[0])
    return out


MISMATCH = [0, 0]


def simulate2(rng: np.random.Generator, n: int, sigma: float, rho: float, d_home: float, d_forum: float) -> tuple:
    """(home72, forum72) score arrays for a breadth effect of ``d_home`` / ``d_forum`` pp and no width effect."""
    r = 4 * n
    shared = rng.standard_normal((REPS, r))
    out = []
    for view, d in (("home72", d_home), ("forum72", d_forum)):
        own = math.sqrt(rho) * shared + math.sqrt(1.0 - rho) * rng.standard_normal((REPS, r))
        s = np.repeat([0.0, 0.0, d, d], n) + sigma * own
        e = np.triu(cell_sd(view) * rng.standard_normal((REPS, r, r)), 1)
        e = e - np.transpose(e, (0, 2, 1))
        out.append(50.0 + s[:, :, None] - s[:, None, :] + e)
    return out[0], out[1]


def breadth_np(fh: np.ndarray, ff: np.ndarray, n: int) -> tuple:
    """(100M outcome, 50M outcome) of breadth by this file's formula."""
    d_f, half_f = factor_np(ff, n, "breadth")
    d_h, _half_h = factor_np(fh, n, "breadth")
    out = classes_100(d_f, half_f)
    out[(out == X.WIN) & ~(d_h + d_f > X.EPS_PP)] = X.WIN_NET_NEGATIVE
    d_hf, half_hf = factor_np(0.5 * fh + 0.5 * ff, n, "breadth")
    return out, classes_50(d_hf, half_hf)


def check_look(fh: np.ndarray, ff: np.ndarray, n: int) -> None:
    design = X.design_of({a: tuple(f"{a}{k}" for k in range(n)) for a in X.ARMS})
    idx = {r: j for j, r in enumerate(r for a in X.ARMS for r in design.runs[a])}
    cells = design.cells()
    want100, want50 = breadth_np(fh[:CHECK_LOOK], ff[:CHECK_LOOK], n)
    for rep in range(CHECK_LOOK):
        panel = {"home72": {(p, o): float(fh[rep, idx[p], idx[o]]) for p, o in cells},
                 "forum72": {(p, o): float(ff[rep, idx[p], idx[o]]) for p, o in cells}}
        by = {**panel, "heldout72_hpfix": panel["forum72"], "p72": panel["forum72"]}
        got100 = X.read_look(design, by, "100M")["factors"]["breadth"]["outcome"]
        got50 = X.read_look(design, by, "50M")["factors"]["breadth"]["outcome"]
        MISMATCH[0] += int(got100 != want100[rep]) + int(got50 != want50[rep])
        MISMATCH[1] += 2


def check(f: np.ndarray, n: int) -> None:
    got = rule_code(f[:CHECK], n)
    dw, hw = factor_np(f[:CHECK], n, "width")
    db, hb = factor_np(f[:CHECK], n, "breadth")
    want = (classes_100(dw, hw), classes_50(dw, hw), classes_100(db, hb), interaction_np(f[:CHECK], n)[2])
    for g, w in zip(got, want):
        MISMATCH[0] += int(sum(a != b for a, b in zip(g, w)))
        MISMATCH[1] += len(g)


def simple_np(f: np.ndarray, n: int, a: str, b: str) -> tuple:
    h = f[:, arm_idx(n, a)][:, :, arm_idx(n, b)]
    floor = X.FLOOR_PP ** 2
    v = np.maximum(h.mean(2).var(1, ddof=1), floor) / n + np.maximum(h.mean(1).var(1, ddof=1), floor) / n
    return h.mean((1, 2)) - 50.0, stats.t.ppf(0.5 + X.LEVEL / 2.0, 2 * (n - 1)) * np.sqrt(v)


def check_simple(f: np.ndarray, n: int, a: str, b: str, want: np.ndarray) -> None:
    design = X.design_of({arm: tuple(f"{arm}{k}" for k in range(n)) for arm in X.ARMS})
    idx = {r: j for j, r in enumerate(r for arm in X.ARMS for r in design.runs[arm])}
    cells = design.cells()
    for rep in range(CHECK):
        scores = {(p, o): float(f[rep, idx[p], idx[o]]) for p, o in cells}
        MISMATCH[0] += int(X.classify("100M", X.simple_effect(design, scores, a, b)).outcome != want[rep])
        MISMATCH[1] += 1


def frac(cls: np.ndarray, name: str) -> float:
    return float((cls == name).mean())


def main() -> int:
    rng = np.random.default_rng(SEED)
    print(f"cell SD by view (pp): " + ", ".join(f"{v} {cell_sd(v):.2f}" for v in X.VIEWS))
    print(f"floor {X.FLOOR_PP} pp, bar +{X.BAR_PP} pp, harm stop {X.HARM_PP} pp, level {X.LEVEL}, band {X.EPS_PP} pp; "
          f"{REPS} draws a row, the first {CHECK} also through main.b1.cross")
    print("\n(1) ONE FACTOR'S MAIN EFFECT at 100M, view N (the other factor at +3 pp, no interaction)")
    for n in (2, 3):
        for sigma in (1.76, 2.5, 3.1):
            row = []
            for delta in (0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0):
                f = simulate(rng, n, sigma, cell_sd("N"), {"C": 0.0, "W": delta, "Br": 3.0, "WB": delta + 3.0})
                check(f, n)
                d, half = factor_np(f, n, "width")
                c = classes_100(d, half)
                row.append(f"{delta:+.0f}: WIN {frac(c, X.WIN):.2f} NUG {frac(c, X.NO_USEFUL_GAIN):.2f} "
                           f"UND {frac(c, X.UNDECIDED):.2f} LOSS {frac(c, X.LOSS):.3f}")
            print(f"n {n} sigma {sigma:.2f} (median half-width {np.median(half):.2f} pp) | " + " | ".join(row))
    print("\n(2) THE 50M HARM STOP, view N, n 2: P(STOP-HARM) at a true effect of")
    for sigma in (1.76, 2.5, 3.1):
        row = []
        for delta in (0.0, -2.0, -4.0, -5.0, -6.0, -8.0):
            f = simulate(rng, 2, sigma, cell_sd("N"), {"C": 0.0, "W": delta, "Br": 0.0, "WB": delta})
            check(f, 2)
            d, half = factor_np(f, 2, "width")
            row.append(f"{delta:+.0f}: {frac(classes_50(d, half), X.STOP_HARM):.3f}")
        print(f"sigma {sigma:.2f} | " + " | ".join(row))
    print("\n(3) THE INTERACTION (REPORTED), view N: P(SYNERGY) / P(ANTAGONISM) at a true interaction of")
    for n in (2, 3):
        for sigma in (1.76, 2.5):
            row = []
            for inter in (0.0, 4.0, 6.0, 8.0, 10.0):
                f = simulate(rng, n, sigma, cell_sd("N"), {"C": 0.0, "W": 2.0, "Br": 1.0, "WB": 3.0 + inter})
                check(f, n)
                _e, half, c = interaction_np(f, n)
                row.append(f"{inter:+.0f}: {frac(c, X.SYNERGY):.2f} / {frac(c, X.ANTAGONISM):.3f}")
            print(f"n {n} sigma {sigma:.2f} (median half-width {np.median(half):.2f} pp) | " + " | ".join(row))
    print("\n(4) ONE FACTOR ON A SINGLE VIEW at 100M, n 2 (forum72 DECIDES breadth; U and home72 are REPORTED)")
    for view in ("forum72", "U", "home72"):
        for sigma in (1.76, 2.5, 3.1):
            row = []
            for delta in (0.0, 2.0, 4.0, 6.0, 8.0):
                f = simulate(rng, 2, sigma, cell_sd(view), {"C": 0.0, "W": 0.0, "Br": delta, "WB": delta})
                check(f, 2)
                d, half = factor_np(f, 2, "breadth")
                c = classes_100(d, half)
                row.append(f"{delta:+.0f}: WIN {frac(c, X.WIN):.2f} NUG {frac(c, X.NO_USEFUL_GAIN):.2f}")
            print(f"{view} sigma {sigma:.2f} (median half-width {np.median(half):.2f} pp) | " + " | ".join(row))
    print("\n(5) AN INTERACTION DOES NOT BIAS A MAIN EFFECT'S INTERVAL: width +4 with interaction +6 reads as +7 "
          "(4 + 6/2); P(WIN), n 2, sigma 1.76")
    f = simulate(rng, 2, 1.76, cell_sd("N"), {"C": 0.0, "W": 4.0, "Br": 1.0, "WB": 11.0})
    check(f, 2)
    d, half = factor_np(f, 2, "width")
    print(f"mean estimate {d.mean():+.2f} pp, median half-width {np.median(half):.2f} pp, "
          f"P(WIN) {frac(classes_100(d, half), X.WIN):.2f}")
    print("\n(6) BREADTH AS REGISTERED at 100M, n 2: its class on forum72, then the net rule (home72 + forum72 > 0). "
          "P(WIN = adoptable) / P(WIN-NET-NEGATIVE) at a true (home72, forum72) effect of")
    for sigma in (1.76, 2.5):
        for rho in (1.0, 0.0):
            row = []
            for d_h, d_f in ((0.0, 0.0), (0.0, 4.0), (-2.0, 4.0), (-4.0, 4.0), (-6.0, 4.0), (-3.28, 2.0), (-3.0, 6.0),
                             (-6.0, 6.0), (-9.0, 6.0), (-6.0, 0.0)):
                fh, ff = simulate2(rng, 2, sigma, rho, d_h, d_f)
                check_look(fh, ff, 2)
                c, _c50 = breadth_np(fh, ff, 2)
                row.append(f"({d_h:+.2f}, {d_f:+.0f}): {frac(c, X.WIN):.2f} / {frac(c, X.WIN_NET_NEGATIVE):.2f}")
            print(f"sigma {sigma:.2f} rho {rho:.0f} | " + " | ".join(row))
    print("\n(7) BREADTH'S 50M HARM STOP on the net view HF = 1/2 home72 + 1/2 forum72 (cell SD "
          f"{cell_sd('HF'):.2f} pp), n 2: P(STOP-HARM) at a true (home72, forum72) effect of")
    for sigma in (1.76, 2.5):
        for rho in (1.0, 0.0):
            row = []
            for d_h, d_f in ((0.0, 0.0), (-4.0, 4.0), (-4.0, 0.0), (-8.0, 0.0), (-12.0, 0.0), (-16.0, 0.0), (-6.0, -6.0),
                             (-12.0, 4.0)):
                fh, ff = simulate2(rng, 2, sigma, rho, d_h, d_f)
                check_look(fh, ff, 2)
                _c, c50 = breadth_np(fh, ff, 2)
                row.append(f"({d_h:+.0f}, {d_f:+.0f}): {frac(c50, X.STOP_HARM):.3f}")
            print(f"sigma {sigma:.2f} rho {rho:.0f} | " + " | ".join(row))
    print("\n(8) AFTER A 50M STOP-HARM OF THE OTHER FACTOR: the surviving factor is one arm against one arm (2 v 2, "
          "2 df), n 2. REPORTED: the registration reads NO class there (PREREG §8); what the +4 pp bar WOULD give, "
          "P(WIN) / P(NO-USEFUL-GAIN), at a true effect of")
    for view in ("N", "forum72"):
        for sigma in (1.76, 2.5):
            row = []
            for delta in (0.0, 4.0, 6.0, 8.0, 10.0):
                f = simulate(rng, 2, sigma, cell_sd(view), {"C": 0.0, "W": delta, "Br": 0.0, "WB": 0.0})
                d, half = simple_np(f, 2, "W", "C")
                c = classes_100(d, half)
                check_simple(f, 2, "W", "C", c)
                row.append(f"{delta:+.0f}: {frac(c, X.WIN):.2f} / {frac(c, X.NO_USEFUL_GAIN):.2f}")
            print(f"{view} sigma {sigma:.2f} (median half-width {np.median(half):.2f} pp) | " + " | ".join(row))
    print("\n(9) UNEQUAL SEED SPREADS (the interval assumes one spread for every arm), view N, n 2, NO true effect: "
          "P(false WIN) / P(false LOSS) / P(NO-USEFUL-GAIN) at a seed SD (pp) of (C, W, Br, WB)")
    for sds in ((1.76, 1.76, 1.76, 1.76), (1.0, 6.0, 1.0, 1.0), (0.5, 8.0, 0.5, 0.5), (1.0, 6.0, 1.0, 6.0),
                (6.0, 1.0, 6.0, 1.0), (1.0, 1.0, 6.0, 6.0)):
        f = simulate(rng, 2, np.repeat(np.array(sds), 2), cell_sd("N"), {a: 0.0 for a in X.ARMS})
        check(f, 2)
        d, half = factor_np(f, 2, "width")
        c = classes_100(d, half)
        print(f"{sds} | width: {frac(c, X.WIN):.3f} / {frac(c, X.LOSS):.3f} / {frac(c, X.NO_USEFUL_GAIN):.2f} "
              f"(median half-width {np.median(half):.2f} pp)")
    print("\n(10) THE EXTENSION (PREREG §9: an UNDECIDED factor may get a third run an arm, and the three-run class then "
          "replaces the two-run one), view N: P(WIN at two runs) + P(UNDECIDED at two, WIN at three) at a true effect of")
    first2 = [i for a in X.ARMS for i in arm_idx(3, a)[:2]]
    for sigma in (1.76, 2.5):
        row = []
        for delta in (0.0, 2.0, 4.0):
            f3 = simulate(rng, 3, sigma, cell_sd("N"), {"C": 0.0, "W": delta, "Br": 0.0, "WB": delta})
            f2 = f3[:, first2][:, :, first2]
            check(f3, 3)
            check(f2, 2)
            c2 = classes_100(*factor_np(f2, 2, "width"))
            c3 = classes_100(*factor_np(f3, 3, "width"))
            und = c2 == X.UNDECIDED
            row.append(f"{delta:+.0f}: {frac(c2, X.WIN):.3f} + {float((und & (c3 == X.WIN)).mean()):.3f} = "
                       f"{float(((c2 == X.WIN) | (und & (c3 == X.WIN))).mean()):.3f} "
                       f"(false LOSS {float(((c2 == X.LOSS) | (und & (c3 == X.LOSS))).mean()):.3f})")
        print(f"sigma {sigma:.2f} | " + " | ".join(row))
    print(f"\nRULE CODE vs THIS FILE'S FORMULA: {MISMATCH[0]} mismatches in {MISMATCH[1]} classes")
    return 1 if MISMATCH[0] else 0


if __name__ == "__main__":
    sys.exit(main())
