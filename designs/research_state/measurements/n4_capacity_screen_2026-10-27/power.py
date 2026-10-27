"""N4 PREREG power (CPU, deterministic; `python -I power.py > power.log`). Writes nothing else.

THE REGISTERED SCREEN (PREREG §5), one look, 3 seed finals a side, matched steps:
  S  (in-house, `readout.population_score` per final): Welch on the 3 + 3 finals; a ONE-SIDED (1 - alpha) bound pair.
     GO-S iff LB > 0; else NO-GO-S iff UB < +2.0 pp (the X35 bar); else UNDECIDED-S.
  K  (Kakuna on forum72): delta-hat of the 3 + 3 finals' pooled scores, SE binomial widened by sqrt(max(1, phi-hat)),
     phi-hat the seeds' dispersion ratio on 2(k - 1) df (`main.x35.d1.overdispersion` / `widen` shape), z one-sided.
     GO-K iff LB > 0; else NO-GO-K iff UB < +1.5 pp (PR-1's KILL bar); else UNDECIDED-K. (The registered estimator
     is `d1`'s Newcombe hybrid of Wilson intervals, PREREG §5.2; this normal SE approximates it.)
  alpha = 0.05 per meter (Bonferroni over the two meters: family false GO <= 0.10).
Also printed for comparison: the X35 R1 sequential rule as registered (`main.x35.readout`: Welch two-sided 97.5 %,
bar +-2.0 pp, look 1 = 3 seeds, look 2 = 5) -- REPORTED beside the screen, never deciding.
Common random numbers across delta (draws seeded 20261027), so each row is monotone in delta.
"""
import numpy as np
from scipy.stats import norm, t as student_t

DRAWS, SEED = 200_000, 20261027
S_BAR, K_BAR, ALPHA = 2.0, 0.015, 0.05
DELTAS_S = (0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
DELTAS_K = (0.0, 0.015, 0.02, 0.03, 0.04)


def welch(a, b, conf_two_sided):
    n = a.shape[1]
    va, vb = a.var(axis=1, ddof=1) / n, b.var(axis=1, ddof=1) / n
    v = va + vb
    df = v * v / (va * va / (n - 1) + vb * vb / (n - 1))
    q = student_t.ppf(1 - (1 - conf_two_sided) / 2, df)
    d = a.mean(axis=1) - b.mean(axis=1)
    return d, d - q * np.sqrt(v), d + q * np.sqrt(v)


def s_screen(sigma):
    rng = np.random.default_rng(SEED)
    on, off = rng.normal(0, sigma, (DRAWS, 5)), rng.normal(0, sigma, (DRAWS, 5))
    rows = []
    for delta in DELTAS_S:
        _, lo, hi = welch(on[:, :3] + delta, off[:, :3], 1 - 2 * ALPHA)
        go = lo > 0
        nogo = ~go & (hi < S_BAR)
        # X35 R1 sequential, as registered (reported only)
        d1, lo1, hi1 = welch(on[:, :3] + delta, off[:, :3], 0.975)
        _, lo2, hi2 = welch(on + delta, off, 0.975)
        g1, l1, n1 = lo1 > 2.0, hi1 < -2.0, (lo1 > -2.0) & (hi1 < 2.0)
        cont = ~g1 & ~l1 & ~n1 & (d1 > 0)
        r1_gain = g1 | (cont & (lo2 > 2.0))
        r1_null = n1 | (cont & (lo2 > -2.0) & (hi2 < 2.0))
        rows.append((delta, go.mean(), nogo.mean(), 1 - go.mean() - nogo.mean(), r1_gain.mean(), r1_null.mean(),
                     cont.mean()))
    return rows


def k_screen(n, sigma_s, p0=0.42, k=3):
    rng = np.random.default_rng(SEED + 1)
    draws = 100_000
    e_on, e_off = rng.normal(0, sigma_s, (draws, k)), rng.normal(0, sigma_s, (draws, k))
    z = norm.ppf(1 - ALPHA)
    rows = []
    for delta in DELTAS_K:
        r = np.random.default_rng(SEED + 2)  # common binomial stream across delta
        x_on = r.binomial(n, np.clip(p0 + delta + e_on, 0.01, 0.99)) / n
        x_off = r.binomial(n, np.clip(p0 + e_off, 0.01, 0.99)) / n
        d = x_on.mean(1) - x_off.mean(1)
        pbar = (x_on.mean(1) + x_off.mean(1)) / 2
        v1 = pbar * (1 - pbar) / n
        se = np.sqrt(2 * v1 / k)
        phi = np.maximum(1.0, (x_on.var(1, ddof=1) + x_off.var(1, ddof=1)) / 2 / v1)
        hw = z * se * np.sqrt(phi)
        go = d - hw > 0
        nogo = ~go & (d + hw < K_BAR)
        rows.append((delta * 100, go.mean(), nogo.mean(), 1 - go.mean() - nogo.mean(), float(np.median(hw)) * 100))
    return rows


if __name__ == "__main__":
    print(f"one-sided alpha {ALPHA} per meter; S bar +{S_BAR} pp; K bar +{K_BAR * 100} pp")
    for sigma in (0.75, 1.07):
        print(f"\n(1) S screen, 3 + 3 seed finals, per-seed SD {sigma} pp "
              f"(0.75 = toy C0 / X47 finals' SD; 1.07 = the X35 registered sigma)")
        print("    delta pp | GO-S  | NO-GO-S | UNDEC-S || X35 R1 (reported): GAIN by look 2 | NULL | look 2 runs")
        for d, g, ng, u, rg, rn, c in s_screen(sigma):
            print(f"    {d:+5.1f}    | {g:.3f} | {ng:.3f}   | {u:.3f}   ||                    {rg:.3f}          "
                  f"| {rn:.3f} | {c:.3f}")
    for n in (600, 1000):
        for ss in (0.0, 0.01):
            print(f"\n(2) K screen, forum72 vs Kakuna, {n} games per final x 3 finals a side, base 0.42, "
                  f"sigma_s {ss * 100:.0f} pp")
            print("    delta pp | GO-K  | NO-GO-K | UNDEC-K | median one-sided half-width pp")
            for d, g, ng, u, hw in k_screen(n, ss):
                print(f"    {d:+5.1f}    | {g:.3f} | {ng:.3f}   | {u:.3f}   | {hw:.2f}")
