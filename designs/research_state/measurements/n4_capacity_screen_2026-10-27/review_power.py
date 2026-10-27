"""REVIEW of the N4 screen's error rates under the reviewer's own sigma assumptions (CPU, deterministic).

`python -I review_power.py > review_power.log`. Writes nothing else. It re-implements the REGISTERED screen exactly as
`power.py` does (PREREG §5: S by Welch on 3 + 3 finals at two-sided 90 %; K by a binomial SE widened by
sqrt(max(1, phi-hat)) at z one-sided 1.6449; GO iff GO-S or GO-K; NO-GO iff NO-GO-S and not GO-K) and varies:
  (1) the per-seed SD of S: equal on both arms over 0.50-2.00 pp, and UNEQUAL (the grown arm noisier);
  (2) the seed SD of the Kakuna F score sigma_s over 0-3 pp;
  (3) the JOINT screen (both meters, seed effects correlated rho in {0, 0.7}) under nulls and alternatives,
      including a harm case (S up, K down) the rule cannot see;
  (4) the planning-sigma interval: the chi-square 90 % interval on the S SD from the banked finals.
Seeds are fixed, so every row reproduces.
"""
import numpy as np
from scipy.stats import chi2, norm, t as student_t

SEED = 20261027
S_BAR, K_BAR, ALPHA = 2.0, 0.015, 0.05
Z = norm.ppf(1 - ALPHA)


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
    go = lo > 1e-9
    nogo = ~go & (hi < S_BAR - 1e-9)
    return go, nogo, hi < 0


def k_classes(x_on, x_off, n, k=3):
    d = x_on.mean(1) - x_off.mean(1)
    pbar = (x_on.mean(1) + x_off.mean(1)) / 2
    v1 = pbar * (1 - pbar) / n
    se = np.sqrt(2 * v1 / k)
    phi = np.maximum(1.0, (x_on.var(1, ddof=1) + x_off.var(1, ddof=1)) / 2 / v1)
    hw = Z * se * np.sqrt(phi)
    go = d - hw > 1e-9
    nogo = ~go & (d + hw < K_BAR - 1e-9)
    return go, nogo, d + hw < 0


def part1(draws=200_000):
    print("(1) S screen, 3 + 3 finals: P(GO-S) / P(NO-GO-S)")
    cases = [(0.50, 0.50), (0.56, 0.56), (0.75, 0.75), (1.07, 1.07), (1.50, 1.50), (2.00, 2.00),
             (1.50, 0.75), (2.00, 0.75)]
    print("    sigma N4 / C0' pp | " + " | ".join(f"true {d:+.0f} pp" for d in (0, 1, 2, 3)))
    for s_on, s_off in cases:
        rng = np.random.default_rng(SEED)
        e_on, e_off = rng.standard_normal((draws, 3)), rng.standard_normal((draws, 3))
        cells = []
        for delta in (0.0, 1.0, 2.0, 3.0):
            go, nogo, _ = s_classes(delta + s_on * e_on, s_off * e_off)
            cells.append(f"{go.mean():.3f} / {nogo.mean():.3f}")
        print(f"    {s_on:4.2f} / {s_off:4.2f}       | " + " | ".join(cells))


def part2(n=1000, draws=100_000, p0=0.42):
    print(f"\n(2) K screen, {n} F games per final, base {p0}: P(GO-K) / P(NO-GO-K)")
    print("    sigma_s pp | " + " | ".join(f"true {d:+.0f} pp" for d in (0, 2, 3, 4)))
    for ss in (0.0, 0.01, 0.02, 0.03):
        rng = np.random.default_rng(SEED + 1)
        e_on, e_off = rng.standard_normal((draws, 3)), rng.standard_normal((draws, 3))
        cells = []
        for delta in (0.0, 0.02, 0.03, 0.04):
            r = np.random.default_rng(SEED + 2)
            x_on = r.binomial(n, np.clip(p0 + delta + ss * e_on, 0.01, 0.99)) / n
            x_off = r.binomial(n, np.clip(p0 + ss * e_off, 0.01, 0.99)) / n
            go, nogo, _ = k_classes(x_on, x_off, n)
            cells.append(f"{go.mean():.3f} / {nogo.mean():.3f}")
        print(f"    {ss * 100:4.1f}       | " + " | ".join(cells))


def part3(n=1000, draws=100_000, p0=0.42, sig_k=0.01):
    print(f"\n(3) The JOINT screen (GO iff GO-S or GO-K; NO-GO iff NO-GO-S and not GO-K), sigma_s(K) {sig_k * 100:.0f} pp,"
          " seed effects correlated rho between the meters")
    print("    sigma_S | rho | true (dS pp, dK pp) | P(GO) | P(GO via K only) | P(NO-GO) | P(UNDEC) | P(GO and the other meter's UB < 0)")
    for sig_s in (0.75, 1.07):
        for rho in (0.0, 0.7):
            for ds, dk in ((0.0, 0.0), (0.0, 3.0), (1.0, 0.0), (2.0, 0.0), (2.0, 2.0), (2.0, -3.0), (0.0, -3.0)):
                rng = np.random.default_rng(SEED + 3)
                z_on, z_off = rng.standard_normal((draws, 3)), rng.standard_normal((draws, 3))
                w_on, w_off = rng.standard_normal((draws, 3)), rng.standard_normal((draws, 3))
                s_on, s_off = ds + sig_s * z_on, sig_s * z_off
                k_on = p0 + dk / 100 + sig_k * (rho * z_on + np.sqrt(1 - rho * rho) * w_on)
                k_off = p0 + sig_k * (rho * z_off + np.sqrt(1 - rho * rho) * w_off)
                r = np.random.default_rng(SEED + 4)
                x_on = r.binomial(n, np.clip(k_on, 0.01, 0.99)) / n
                x_off = r.binomial(n, np.clip(k_off, 0.01, 0.99)) / n
                gs, ngs, harm_s = s_classes(s_on, s_off)
                gk, _, harm_k = k_classes(x_on, x_off, n)
                go = gs | gk
                nogo = ngs & ~gk
                harm = (gs & harm_k) | (gk & harm_s)
                print(f"    {sig_s:4.2f}    | {rho:.1f} | ({ds:+.0f}, {dk:+.0f})            | {go.mean():.3f} | "
                      f"{(gk & ~gs).mean():.3f}            | {nogo.mean():.3f}    | {(~go & ~nogo).mean():.3f}    | "
                      f"{harm.mean():.4f}")


def part4():
    print("\n(4) The planning sigma's own uncertainty (chi-square 90 % two-sided interval on an SD)")
    for label, sds in (("toy C0 finals alone (n 3)", [0.73]), ("toy C0 + X47 finals pooled (n 3 + 3)", [0.73, 0.53])):
        df = 2 * len(sds)
        s = float(np.sqrt(np.mean(np.square(sds))))
        lo, hi = s * np.sqrt(df / chi2.ppf(0.95, df)), s * np.sqrt(df / chi2.ppf(0.05, df))
        print(f"    {label}: s {s:.2f} pp on {df} df -> [{lo:.2f}, {hi:.2f}] pp")


if __name__ == "__main__":
    print(f"one-sided alpha {ALPHA} per meter; S bar +{S_BAR} pp; K bar +{K_BAR * 100} pp; rule-8 eps 1e-9")
    part1()
    part2()
    part3()
    part4()
