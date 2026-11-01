"""N4's screen error rates UNDER AMENDMENT AM-N4-FIXES (CPU, deterministic).

`python -I power_am_fixes.py > power_am_fixes.log`. Writes nothing else. The simulation model is the REVIEW's
(`review_power.py`: S by Welch on 3 + 3 finals at two-sided 90 %; K by a binomial SE widened by sqrt(max(1, phi-hat))
at z one-sided 1.6449, 1,000 forum72 games a final, base 0.42), with the AMENDED rule in the registered class order
(`main.reads.rules_n4_screen`: MIXED > GO > NO-GO > UNDECIDED):
  - m3: GO-S needs LB_S > 0 AND the point estimate Delta-hat_S > +1.0 pp (past it by more than rule 8's eps);
  - M2: GO-S with UB_K < 0, or GO-K with UB_S < 0, reads MIXED (classed UNDECIDED, never GO);
  - GO iff GO-S or GO-K (and not MIXED); NO-GO iff NO-GO-S (UB_S < +2.0 pp) and not GO-K (and not MIXED).
M3 (2,000 pairs per S cell) enters as the per-final SD of S: the REVIEW's ESTIMATE at 2,000 pairs is 0.55-0.65 pp
(about half of S's variance at 500 pairs is evaluation noise, an UPPER bound on that share); 0.75 and 1.07 (the
registered planning values at 500 pairs) are kept as the pessimistic rows, in case the noise share is smaller.
  (1) the S meter alone; (2) the joint screen, seed effects correlated rho in {0, 0.7} between the meters.
Seeds are fixed, so every row reproduces.
"""
import numpy as np
from scipy.stats import norm, t as student_t

SEED = 20261031
S_BAR, K_BAR, ALPHA, GO_S_FLOOR, EPS = 2.0, 0.015, 0.05, 1.0, 1e-9
Z = norm.ppf(1 - ALPHA)
SIGMA_S = (0.56, 0.65, 0.75, 1.07)


def welch(a, b, conf_two_sided):
    na, nb = a.shape[1], b.shape[1]
    va, vb = a.var(axis=1, ddof=1) / na, b.var(axis=1, ddof=1) / nb
    v = va + vb
    df = v * v / (va * va / (na - 1) + vb * vb / (nb - 1))
    q = student_t.ppf(1 - (1 - conf_two_sided) / 2, df)
    d = a.mean(axis=1) - b.mean(axis=1)
    return d, d - q * np.sqrt(v), d + q * np.sqrt(v)


def s_meter(on, off):
    d, lo, hi = welch(on, off, 1 - 2 * ALPHA)
    go = (lo > EPS) & (d > GO_S_FLOOR + EPS)
    nogo = ~go & (hi < S_BAR - EPS)
    return go, nogo, hi < -EPS


def k_meter(x_on, x_off, n, k=3):
    d = x_on.mean(1) - x_off.mean(1)
    pbar = (x_on.mean(1) + x_off.mean(1)) / 2
    v1 = pbar * (1 - pbar) / n
    se = np.sqrt(2 * v1 / k)
    phi = np.maximum(1.0, (x_on.var(1, ddof=1) + x_off.var(1, ddof=1)) / 2 / v1)
    hw = Z * se * np.sqrt(phi)
    go = d - hw > EPS
    nogo = ~go & (d + hw < K_BAR - EPS)
    return go, nogo, d + hw < -EPS


def screen(gs, ngs, harm_s, gk, harm_k):
    mixed = (gs & harm_k) | (gk & harm_s)
    go = (gs | gk) & ~mixed
    nogo = ngs & ~gk & ~mixed
    return go, nogo, mixed


def part1(draws=200_000):
    print("(1) S meter alone, 3 + 3 finals, the m3 floor on: P(GO-S) / P(NO-GO-S)")
    print("    sigma_S pp | " + " | ".join(f"true {d:+.0f} pp" for d in (0, 1, 2, 3)))
    for sig in SIGMA_S:
        rng = np.random.default_rng(SEED)
        e_on, e_off = rng.standard_normal((draws, 3)), rng.standard_normal((draws, 3))
        cells = []
        for delta in (0.0, 1.0, 2.0, 3.0):
            go, nogo, _ = s_meter(delta + sig * e_on, sig * e_off)
            cells.append(f"{go.mean():.3f} / {nogo.mean():.3f}")
        print(f"    {sig:4.2f}       | " + " | ".join(cells))


def part2(n=1000, draws=100_000, p0=0.42):
    print(f"\n(2) The JOINT screen as amended, {n} F games a final, base {p0}")
    print("    sigma_S | sigma_s(K) | rho | true (dS, dK) pp | P(GO) | P(GO via K only) | P(MIXED) | P(NO-GO) | "
          "P(UNDECIDED, MIXED excluded)")
    for sig_s in SIGMA_S:
        for sig_k in (0.0, 0.01):
            for rho in (0.0, 0.7):
                for ds, dk in ((0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0), (0.0, 3.0), (0.0, 4.0), (2.0, -3.0),
                               (0.0, -3.0)):
                    rng = np.random.default_rng(SEED + 3)
                    z_on, z_off = rng.standard_normal((draws, 3)), rng.standard_normal((draws, 3))
                    w_on, w_off = rng.standard_normal((draws, 3)), rng.standard_normal((draws, 3))
                    s_on, s_off = ds + sig_s * z_on, sig_s * z_off
                    k_on = p0 + dk / 100 + sig_k * (rho * z_on + np.sqrt(1 - rho * rho) * w_on)
                    k_off = p0 + sig_k * (rho * z_off + np.sqrt(1 - rho * rho) * w_off)
                    r = np.random.default_rng(SEED + 4)
                    x_on = r.binomial(n, np.clip(k_on, 0.01, 0.99)) / n
                    x_off = r.binomial(n, np.clip(k_off, 0.01, 0.99)) / n
                    gs, ngs, harm_s = s_meter(s_on, s_off)
                    gk, _, harm_k = k_meter(x_on, x_off, n)
                    go, nogo, mixed = screen(gs, ngs, harm_s, gk, harm_k)
                    print(f"    {sig_s:4.2f}    | {sig_k * 100:3.0f}        | {rho:.1f} | ({ds:+.0f}, {dk:+.0f})"
                          f"          | {go.mean():.3f} | {(go & gk & ~gs).mean():.3f}            | "
                          f"{mixed.mean():.4f}   | {nogo.mean():.3f}    | {(~go & ~nogo & ~mixed).mean():.3f}")


if __name__ == "__main__":
    print(f"AM-N4-FIXES: one-sided alpha {ALPHA} per meter; S bar +{S_BAR} pp; GO-S floor +{GO_S_FLOOR} pp; "
          f"K bar +{K_BAR * 100} pp; MIXED veto on; rule-8 eps {EPS}")
    part1()
    part2()
