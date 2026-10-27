"""X62 design (``designs/endstate/design_x62_team_recombination.md`` §5): power and P(GAIN) of the two candidate
reads, closed form and CPU-trivial (< 1 s). Deterministic.

    python designs/research_state/measurements/x62_design_2026-10-26/power_pgain.py > power_pgain.log

* D1's RULE on F (the N1-P / D3 read: 1,200 F games per seed final, looks of 3 then 5 seeds, AM-5's widening, the
  sign rule): its FINAL-GAIN curve is copied from ``strategy_revision_2026-10-26/pgain_rows.log`` §1 (a Monte Carlo of
  ``main.x35.d1.d1_primary``, 1,500 draws a cell, base F 0.49), at sigma_s = 0 and 1 pp, linearly interpolated.
* The PR RULE (strategy §3.2: one arm vs a same-engine C0, 17,400 F games each, 5 seeds): GAIN iff the 95 % lower
  bound > 0. SE^2 = 2 * 0.25 / 17,400 + 2 * sigma_s^2 / 5. "widened" = the interval uses that SE (a correct AM-5);
  "unwidened" = it uses the binomial SE alone, so its false-GAIN rate is the honest one at a seed SD the interval
  ignores.
* P(GAIN) integrates a GUESS (point, 80 % interval) read as a normal with SD = width / 2.563 (pgain_rows' convention).
"""
import math
from statistics import NormalDist

N = NormalDist()
SE0 = math.sqrt(2 * 0.25 / 17400)
D1_FINAL_GAIN = {  # pgain_rows.log §1, pp -> P(final GAIN)
    0.0: {-2: 0.0, -1: 0.001, 0: 0.026, 0.5: 0.081, 1: 0.188, 1.5: 0.365, 2: 0.57, 3: 0.891, 4: 0.98, 5: 0.999, 6: 1.0},
    1.0: {-2: 0.001, -1: 0.009, 0: 0.047, 0.5: 0.102, 1: 0.205, 1.5: 0.314, 2: 0.491, 3: 0.774, 4: 0.939, 5: 0.993, 6: 1.0},
}
GUESSES = {  # (point, 80 % lo, 80 % hi) in pp of F, X62 - C0
    "X62 | H-CTX SUPPORTED, N1-P not GAIN": (2.0, -1.0, 5.0),
    "X62 | H-CTX SUPPORTED, N1-P NULL": (1.5, -1.5, 4.5),
    "X62 | H-CTX UNDECIDED": (1.0, -1.5, 3.5),
}


def interp(curve, x):
    ks = sorted(curve)
    if x <= ks[0]:
        return curve[ks[0]]
    if x >= ks[-1]:
        return curve[ks[-1]]
    for a, b in zip(ks, ks[1:]):
        if a <= x <= b:
            return curve[a] + (curve[b] - curve[a]) * (x - a) / (b - a)
    raise AssertionError(x)


def se_pr(sigma_pp):
    return math.sqrt(SE0 ** 2 + 2 * (sigma_pp / 100) ** 2 / 5)


def pr_gain(d_pp, sigma_pp):
    return 1 - N.cdf(1.96 - (d_pp / 100) / se_pr(sigma_pp))


def integrate(f, point, lo, hi, n=4000):
    sd = (hi - lo) / 2.563
    return sum(f(point + sd * N.inv_cdf((i + 0.5) / n)) for i in range(n)) / n


def main():
    print(f"PR rule: binomial SE {100 * SE0:.3f} pp per arm contrast")
    for s in (0.0, 0.5, 1.0):
        row = ", ".join(f"{d:+.1f}: {pr_gain(d, s):.3f}" for d in (0.0, 1.5, 2.0, 3.0))
        fg = 1 - N.cdf(1.96 * SE0 / se_pr(s))
        print(f"  sigma_s {s} pp: P(GAIN) widened {row}; unwidened false GAIN {fg:.3f}")
    print("D1 rule (final GAIN, from pgain_rows.log): " + "; ".join(
        f"sigma_s {s} pp: +1.5 {interp(c, 1.5):.3f}, +3 {interp(c, 3):.3f}, 0 {interp(c, 0):.3f}"
        for s, c in D1_FINAL_GAIN.items()))
    for name, (pt, lo, hi) in GUESSES.items():
        d1 = [integrate(lambda d, c=c: interp(c, d), pt, lo, hi) for c in D1_FINAL_GAIN.values()]
        pr = [integrate(lambda d, s=s: pr_gain(d, s), pt, lo, hi) for s in (0.0, 1.0)]
        print(f"{name}: GUESS {pt:+.1f} ({lo:+.1f}, {hi:+.1f}) -> P(GAIN) D1 rule {d1[0]:.2f} / {d1[1]:.2f}; "
              f"PR rule {pr[0]:.2f} / {pr[1]:.2f} (sigma_s 0 / 1 pp)")


if __name__ == "__main__":
    main()
