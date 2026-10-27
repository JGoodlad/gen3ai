"""X48 PREREG §6 power: the N4 screen rule (N4 PREREG §5, with the N4 REVIEW's M2 MIXED veto) at PRODUCTION's numbers.

`python -I power.py > power.log` (CPU, deterministic, seconds). Writes nothing else. It re-implements the rule exactly
as the N4 review's `review_power.py` does (S by Welch on 3 + 3 finals at two-sided 90 %; K by a binomial SE widened by
sqrt(max(1, phi-hat)) at z one-sided 1.6449), adds M2's veto (a GO on one meter while the OTHER meter's UB < 0 reads
MIXED, classed UNDECIDED), and uses production's inputs:
  * S: sigma per final at 2,000 mirrored pairs a cell (M3). Production C0's three G1 finals at 500 pairs read SD 0.48 pp
    with a per-final measurement SE of 0.55 pp (`~/gen3ai_archive/x35_g1/results.json`), so the true seed SD is small;
    at 2,000 pairs the measurement SE is ~0.27 pp. The grid 0.40-1.07 pp brackets it (1.07 = the toy's registered
    planning sigma, the pessimistic end).
  * K: forum72 against Kakuna, 1,000 games a final (unit seeds 0-49), base 0.49 (C0's own forum72 at D1's read,
    342 / 700), the seed SD of a final's true F score sigma_k 0-2 pp (production's banked phi-hat 0.33-0.77 < 1).
Then P(GO) integrated over the PREREG's GUESS (a normal: the point as mean, the 80 % interval's width / 2.563 as SD),
the strategy doc's P(GAIN) convention (`strategy_revision_2026-10-26/pgain_rows.py`).
"""
import numpy as np
from scipy.stats import norm, t as student_t

SEED = 4827
S_BAR, K_BAR, ALPHA, EPS = 2.0, 0.015, 0.05, 1e-9
Z = norm.ppf(1 - ALPHA)
N_K, P0, K_SEEDS = 1000, 0.49, 3


def welch(a, b, conf_two_sided):
    na, nb = a.shape[1], b.shape[1]
    va, vb = a.var(axis=1, ddof=1) / na, b.var(axis=1, ddof=1) / nb
    v = va + vb
    df = v * v / (va * va / (na - 1) + vb * vb / (nb - 1))
    q = student_t.ppf(1 - (1 - conf_two_sided) / 2, df)
    d = a.mean(axis=1) - b.mean(axis=1)
    return d, d - q * np.sqrt(v), d + q * np.sqrt(v)


def s_classes(on, off):
    _, lo, hi = welch(on, off, 1 - 2 * ALPHA)
    go = lo > EPS
    return go, ~go & (hi < S_BAR - EPS), hi < -EPS


def k_classes(x_on, x_off, n=N_K, k=K_SEEDS):
    d = x_on.mean(1) - x_off.mean(1)
    pbar = (x_on.mean(1) + x_off.mean(1)) / 2
    v1 = pbar * (1 - pbar) / n
    se = np.sqrt(2 * v1 / k)
    phi = np.maximum(1.0, (x_on.var(1, ddof=1) + x_off.var(1, ddof=1)) / 2 / v1)
    hw = Z * se * np.sqrt(phi)
    go = d - hw > EPS
    return go, ~go & (d + hw < K_BAR - EPS), d + hw < -EPS


def screen(ds, dk, sig_s, sig_k, rho, draws, seed):
    """The class shares of the registered screen: GO / MIXED / NO-GO / other UNDECIDED (MIXED is UNDECIDED)."""
    rng = np.random.default_rng(seed)
    z_on, z_off = rng.standard_normal((draws, 3)), rng.standard_normal((draws, 3))
    w_on, w_off = rng.standard_normal((draws, 3)), rng.standard_normal((draws, 3))
    s_on, s_off = ds + sig_s * z_on, sig_s * z_off
    k_on = P0 + dk / 100 + sig_k * (rho * z_on + np.sqrt(1 - rho * rho) * w_on)
    k_off = P0 + sig_k * (rho * z_off + np.sqrt(1 - rho * rho) * w_off)
    r = np.random.default_rng(seed + 1)
    x_on = r.binomial(N_K, np.clip(k_on, 0.01, 0.99)) / N_K
    x_off = r.binomial(N_K, np.clip(k_off, 0.01, 0.99)) / N_K
    gs, ngs, harm_s = s_classes(s_on, s_off)
    gk, _, harm_k = k_classes(x_on, x_off)
    mixed = (gs & harm_k) | (gk & harm_s)
    go = (gs | gk) & ~mixed
    nogo = ngs & ~gk
    return {"GO": go.mean(), "MIXED": mixed.mean(), "NO-GO": nogo.mean(), "UNDEC": (~go & ~nogo).mean()}


def part1(draws=200_000):
    print("(1) S alone, 3 + 3 finals: P(GO-S) / P(NO-GO-S)")
    print("    sigma_S pp | " + " | ".join(f"true {d:+.0f} pp" for d in (0, 1, 2, 3)))
    for sig in (0.40, 0.60, 0.80, 1.07):
        rng = np.random.default_rng(SEED)
        e_on, e_off = rng.standard_normal((draws, 3)), rng.standard_normal((draws, 3))
        cells = []
        for d in (0.0, 1.0, 2.0, 3.0):
            go, nogo, _ = s_classes(d + sig * e_on, sig * e_off)
            cells.append(f"{go.mean():.3f} / {nogo.mean():.3f}")
        print(f"    {sig:4.2f}       | " + " | ".join(cells))


def part2(draws=100_000):
    print(f"\n(2) K alone, {N_K} F games a final, base {P0}: P(GO-K) / P(NO-GO-K)")
    print("    sigma_k pp | " + " | ".join(f"true {d:+.0f} pp" for d in (0, 2, 3, 4)))
    for sk in (0.0, 0.01, 0.02):
        rng = np.random.default_rng(SEED + 1)
        e_on, e_off = rng.standard_normal((draws, 3)), rng.standard_normal((draws, 3))
        cells = []
        for d in (0.0, 0.02, 0.03, 0.04):
            r = np.random.default_rng(SEED + 2)
            x_on = r.binomial(N_K, np.clip(P0 + d + sk * e_on, 0.01, 0.99)) / N_K
            x_off = r.binomial(N_K, np.clip(P0 + sk * e_off, 0.01, 0.99)) / N_K
            go, nogo, _ = k_classes(x_on, x_off)
            cells.append(f"{go.mean():.3f} / {nogo.mean():.3f}")
        print(f"    {sk * 100:4.1f}       | " + " | ".join(cells))


def part3(draws=100_000):
    print("\n(3) The JOINT screen with M2's MIXED veto (GO iff a meter is GO and the other's UB >= 0; MIXED = UNDECIDED),"
          " sigma_k 1 pp")
    print("    sigma_S | rho | true (dS, dK) pp | P(GO) | P(MIXED) | P(NO-GO) | P(UNDECIDED incl. MIXED)")
    for sig_s in (0.60, 1.07):
        for rho in (0.0, 0.7):
            for ds, dk in ((0.0, 0.0), (0.0, 3.0), (1.0, 0.0), (2.0, 0.0), (2.0, 2.0), (2.0, -3.0), (0.0, -3.0)):
                c = screen(ds, dk, sig_s, 0.01, rho, draws, SEED + 3)
                print(f"    {sig_s:4.2f}    | {rho:.1f} | ({ds:+.0f}, {dk:+.0f})         | {c['GO']:.3f} | "
                      f"{c['MIXED']:.3f}    | {c['NO-GO']:.3f}    | {c['UNDEC']:.3f}")


def part4(draws=40_000):
    """P(GO) over the PREREG's GUESS (independent normals on dS and dK), on a grid, sigma_S 0.6 / 1.07, sigma_k 1 pp."""
    guess = {"dS": (0.3, -1.5, 2.0), "dK": (0.3, -2.0, 2.5)}
    print(f"\n(4) P(GO) and P(NO-GO) over the GUESS {guess} (pp; 80 % intervals read as normals), rho 0")
    grid = np.linspace(-4, 4, 17)
    w = np.exp(-grid * grid / 2)
    w /= w.sum()
    (ms, los, his), (mk, lok, hik) = guess["dS"], guess["dK"]
    sds, sdk = (his - los) / 2.563, (hik - lok) / 2.563
    for sig_s in (0.60, 1.07):
        pg = pn = 0.0
        for i, zs in enumerate(grid):
            for j, zk in enumerate(grid):
                c = screen(ms + sds * zs, mk + sdk * zk, sig_s, 0.01, 0.0, draws, SEED + 10 + i * 17 + j)
                pg += w[i] * w[j] * c["GO"]
                pn += w[i] * w[j] * c["NO-GO"]
        print(f"    sigma_S {sig_s:4.2f}: P(GO) {pg:.3f}, P(NO-GO) {pn:.3f}, P(UNDECIDED) {1 - pg - pn:.3f}")


if __name__ == "__main__":
    print(f"one-sided alpha {ALPHA} per meter; S bar +{S_BAR} pp; K bar +{K_BAR * 100} pp; rule-8 eps {EPS}")
    part1()
    part2()
    part3()
    part4()
