"""X62 adversarial review (``designs/endstate/reviews/x62_review_2026-10-27.md``): Monte Carlo checks of the design's
read (§5, §7) that its closed form leaves out. CPU, numpy, deterministic (seeded), < 1 min.

    python .../review_2026-10-27/power_checks.py > power_checks.log

A. The PR rule WITH AM-5 as ``main.x35.d1.overdispersion`` computes it (phi-hat = the mean of the two arms' seed
   dispersion ratios, half-width x sqrt(max(1, phi-hat)), normal 1.96): 5 seeds x 3,480 F games an arm, base F 0.49.
B. The shared control: N1-P (D1's rule, approximated as LB > 0 on 5 x 1,200 F games a side with AM-5) and X62 (A)
   are read against the SAME five C0 seeds, whose true seed offsets they share. P(X62 GAIN | N1-P not GAIN), the only
   branch on which X62 runs (design §6 G1), against P(X62 GAIN).
C. The co-primary P (in-house S on home72; Welch 97.5 %, NON-INFERIOR iff LB > -2.0 pp), 5 v 5 seeds, at the per-seed
   SDs of D1's final (D1 0.86 pp, C0 0.35 pp; ledger 2026-10-24 "X35 D1 LOOK 2"); P(NI) at fixed effects and over the
   design's own GUESS (-1.0, 80 % (-2.5, +0.5)).
D. The headline odds recomputed: the GUESS on the branch where X62 actually runs (+1.5, 80 % (-1.5, +4.5), §4 / §6),
   A's power, joint with C's P(NI) (independence assumed), per resource-day = (GPU-h + lane-h) / 24 (strategy doc).
E. X62's gate G0, the H-CTX read (PREREG Amendment 1/2, rule ``premium_recovery``): rc72b 1,200 / home72 800 / F 800
   games, home72 0.598 and F 0.530 (X51) as the truth, rc72b = F + rho * P; the verdict's classes by rho, binomial
   (unit dispersion 1; the 95 % bound on P as a normal approximation of the unit bootstrap).
"""
import numpy as np
from scipy import stats

RNG = np.random.default_rng(20261027)
P0 = 0.49


def am5_read(delta, sig, n_seed, k=5, draws=20000, c0_off=None, rng=RNG):
    """Return (gain bool array, c0 seed offsets used)."""
    if c0_off is None:
        c0_off = rng.normal(0, sig, (draws, k))
    on_true = np.clip(P0 + delta + rng.normal(0, sig, (draws, k)), 0.01, 0.99)
    off_true = np.clip(P0 + c0_off, 0.01, 0.99)
    on = rng.binomial(n_seed, on_true) / n_seed
    off = rng.binomial(n_seed, off_true) / n_seed

    def ratio(x):
        e = (x * (1 - x) / n_seed).mean(axis=1)
        return x.var(axis=1, ddof=1) / e
    phi = (ratio(on) + ratio(off)) / 2
    d = on.mean(axis=1) - off.mean(axis=1)
    se = np.sqrt((on * (1 - on)).mean(axis=1) / (k * n_seed) + (off * (1 - off)).mean(axis=1) / (k * n_seed))
    lb = d - 1.96 * se * np.sqrt(np.maximum(1.0, phi))
    return lb > 0, c0_off


def part_a():
    print("A. PR rule with AM-5 (5 x 3,480 F a side): P(GAIN)")
    for sig in (0.0, 0.5, 1.0):
        row = []
        for d in (0.0, 1.5, 2.0, 3.0):
            g, _ = am5_read(d / 100, sig / 100, 3480, draws=40000)
            row.append(f"{d:+.1f}: {g.mean():.3f}")
        print(f"  sigma_s {sig} pp: " + ", ".join(row))


def part_b():
    print("B. shared C0: P(X62 GAIN) vs P(X62 GAIN | N1-P not GAIN), X62 true +1.5 pp, PR rule")
    for sig in (0.5, 1.0):
        for th_n in (0.0, 1.5):
            draws = 60000
            c0 = RNG.normal(0, sig / 100, (draws, 5))
            gn, _ = am5_read(th_n / 100, sig / 100, 1200, draws=draws, c0_off=c0)
            gx, _ = am5_read(1.5 / 100, sig / 100, 3480, draws=draws, c0_off=c0)
            gx0, _ = am5_read(0.0, sig / 100, 3480, draws=draws, c0_off=c0)
            print(f"  sigma_s {sig} pp, N1-P true {th_n:+.1f}: P(GAIN) {gx.mean():.3f} -> | N1-P not GAIN "
                  f"{gx[~gn].mean():.3f} (P(N1-P not GAIN) {1 - gn.mean():.3f}); X62 true 0: false GAIN "
                  f"{gx0.mean():.3f} -> | N1-P GAIN {gx0[gn].mean() if gn.any() else float('nan'):.3f}")


def ni_prob(delta, sd_on=0.86, sd_off=0.35, k=5, draws=100000):
    on = RNG.normal(delta, sd_on, (draws, k))
    off = RNG.normal(0.0, sd_off, (draws, k))
    v1, v2 = on.var(axis=1, ddof=1) / k, off.var(axis=1, ddof=1) / k
    se = np.sqrt(v1 + v2)
    df = (v1 + v2) ** 2 / (v1 ** 2 / (k - 1) + v2 ** 2 / (k - 1))
    lb = on.mean(axis=1) - off.mean(axis=1) - stats.t.ppf(0.9875, df) * se
    return (lb > -2.0).mean()


def part_c():
    print("C. co-primary P(NON-INFERIOR), Welch 97.5 %, margin -2.0 pp, 5 v 5 seeds (seed SD 0.86 / 0.35 pp)")
    print("  fixed true effect: " + ", ".join(f"{d:+.1f}: {ni_prob(d):.3f}" for d in (-1.61, -1.0, -0.5, 0.0, 0.5)))
    sd = 3.0 / 2.563
    grid = stats.norm.ppf((np.arange(200) + 0.5) / 200) * sd - 1.0
    p = np.mean([ni_prob(g, draws=4000) for g in grid])
    print(f"  over the design's GUESS -1.0 (80 % -2.5, +0.5): P(NI) {p:.3f}")
    return p


def part_d(p_ni):
    print("D. headline odds on the branch where X62 runs (GUESS +1.5, 80 % (-1.5, +4.5)), PR rule with AM-5")
    sd = 6.0 / 2.563
    grid = stats.norm.ppf((np.arange(100) + 0.5) / 100) * sd + 1.5
    for sig in (0.0, 0.5, 1.0):
        pg = np.mean([am5_read(g / 100, sig / 100, 3480, draws=3000)[0].mean() for g in grid])
        cells = []
        for lab, lane in (("PR-1's C0 banked, 126-181 lane-h", (126, 181)), ("C0 fresh, 187-269 lane-h", (187, 269))):
            rd = [(20.5 + x) / 24 for x in lane]
            cells.append(f"{lab}: {pg / rd[1]:.3f}-{pg / rd[0]:.3f} per resource-day "
                         f"(joint with P NI: {pg * p_ni / rd[1]:.3f}-{pg * p_ni / rd[0]:.3f})")
        print(f"  sigma_s {sig} pp: P(GAIN) {pg:.3f}, P(GAIN and P NI) {pg * p_ni:.3f}; " + "; ".join(cells))


def part_e(draws=200000):
    print("E. G0 (H-CTX) verdict probabilities by the true recovery fraction rho = R / P (home72 0.598, F 0.530)")
    h, f = 0.598, 0.530
    for rho in (0.0, 0.25, 0.4, 0.6, 0.75, 1.0):
        r = f + rho * (h - f)
        xr = RNG.binomial(1200, r, draws) / 1200
        xh = RNG.binomial(800, h, draws) / 800
        xf = RNG.binomial(800, f, draws) / 800
        P, R = xh - xf, xr - xf
        se_p = np.sqrt(xh * (1 - xh) / 800 + xf * (1 - xf) / 800)
        inc = P - 1.96 * se_p <= 0
        sup = ~inc & (R <= 0.25 * P)
        kil = ~inc & (R >= 0.75 * P)
        und = ~inc & ~sup & ~kil
        print(f"  rho {rho:.2f}: INCONCLUSIVE {inc.mean():.3f}, SUPPORTED {sup.mean():.3f}, UNDECIDED {und.mean():.3f}, "
              f"KILLED {kil.mean():.3f}")


if __name__ == "__main__":
    part_e()
    part_a()
    part_b()
    p_ni = part_c()
    part_d(p_ni)
