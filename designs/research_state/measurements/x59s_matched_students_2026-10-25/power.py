"""X59-S's power WITH the students' training-seed variance (REVIEW D M3), CPU, seeded, deterministic.

Run: python power.py  (numpy + scipy, seconds). Prints one JSON document; the PREREG quotes `results/power.json`.

The model. A student is one (arm, training seed). Its pooled-U win rate vs Kakuna is
    p_arm + s_(arm,seed) + binomial noise,   s ~ N(0, sigma_s),
read on three panels (H, F, P72) at m / 3 games each, pooled equal-weight. k seeds per arm. sigma_s is UNKNOWN (no
from-scratch student exists); it is swept over 0..3 pp. The base rate p is UNKNOWN too (REVIEW D M3: the draft's 0.30
had no source); it is swept over 0.15 / 0.30 / 0.50, and the registered game counts are sized at p = 0.50 (the
binomial worst case), so the power quoted for a decision never rests on a guessed base.

Two interval rules are simulated for a contrast A - B (k seeds each):
- WELCH: a Welch t interval on the k + k per-student pooled-U rates (Satterthwaite df). It needs no assumption about
  sigma_s; with k = 3 it pays t(df ~ 4) = 2.8.
- BOUND (the registered rule): a z interval whose variance is the binomial part plus 2 * max(SIGMA_MAX, sigma_hat)^2
  / k, sigma_hat from the pooled within-arm spread of ALL THREE arms (df = 3 (k - 1)) minus its binomial part. A seed
  SD above the registered bound widens it (never narrows it). Its guard: if sigma_hat's one-sided 95 % LOWER bound
  exceeds SIGMA_MAX, the bound is refuted and the contrast is classified by WELCH instead.
Each prints P(LB > 0) (the "better" class) at the true delta, the false-positive rate at delta = 0, and the median
half-width; plus the guard's firing rate.
"""
from __future__ import annotations

import json

import numpy as np
from scipy.stats import chi2, t as tdist

SEED = 20261028
Z = 1.959963985
PANELS = 3
SIGMA_MAX = 0.015          # the registered seed-SD bound (1.5 pp of pooled-U win rate)
REPS = 100_000


def students(rng, p, delta, sigma_s, k, m):
    """(REPS, k) observed pooled-U rates of one arm at true rate p + delta."""
    s = rng.normal(0.0, sigma_s, (REPS, k))
    q = np.clip(p + delta + s, 0.005, 0.995)
    g = m // PANELS
    return (rng.binomial(g, q[..., None].repeat(PANELS, -1)) / g).mean(-1)


def classify(rng, p, delta, sigma_s, k, m):
    a = students(rng, p, delta, sigma_s, k, m)        # the arm under test (S-H)
    b = students(rng, p, 0.0, sigma_s, k, m)          # its comparator (S-S or S-P)
    c = students(rng, p, 0.0, sigma_s, k, m)          # the third arm (pooled into sigma_hat only)
    d = a.mean(1) - b.mean(1)
    # WELCH
    va, vb = a.var(1, ddof=1) / k, b.var(1, ddof=1) / k
    df = (va + vb) ** 2 / (va ** 2 / (k - 1) + vb ** 2 / (k - 1))
    hw_w = tdist.ppf(0.975, df) * np.sqrt(va + vb)
    # BOUND
    allv = np.concatenate([a, b, c], 1)
    within = (a.var(1, ddof=1) + b.var(1, ddof=1) + c.var(1, ddof=1)) / 3          # df = 3 (k - 1)
    binom = (allv * (1 - allv)).mean(1) / (m / PANELS) / PANELS                    # per-student binomial variance
    dfp = 3 * (k - 1)
    s2_hat = np.maximum(0.0, within - binom)
    lb95 = np.maximum(0.0, within * dfp / chi2.ppf(0.95, dfp) - binom)              # one-sided 95 % lower bound
    guard = lb95 > SIGMA_MAX ** 2
    sig2 = np.maximum(SIGMA_MAX ** 2, s2_hat)
    pa, pb = a.mean(1), b.mean(1)
    vbin = (pa * (1 - pa) + pb * (1 - pb)) / (m / PANELS) / PANELS / k
    hw_b = Z * np.sqrt(vbin + 2 * sig2 / k)
    lo_b, hi_b = d - hw_b, d + hw_b
    lo_w, hi_w = d - hw_w, d + hw_w
    lo = np.where(guard, lo_w, lo_b)
    hi = np.where(guard, hi_w, hi_b)
    return {"BOUND P(LB>0)": round(float((lo > 0).mean()), 3), "BOUND P(UB<0)": round(float((hi < 0).mean()), 4),
            "BOUND halfwidth_pp": round(float(np.median(np.where(guard, hw_w, hw_b))) * 100, 2),
            "guard fires": round(float(guard.mean()), 3),
            "WELCH P(LB>0)": round(float((lo_w > 0).mean()), 3), "WELCH P(UB<0)": round(float((hi_w < 0).mean()), 4),
            "WELCH halfwidth_pp": round(float(np.median(hw_w)) * 100, 2)}


def main():
    rng = np.random.default_rng(SEED)
    out = {"seed": SEED, "sigma_max_pp": SIGMA_MAX * 100, "panels": PANELS, "reps": REPS, "grid": {}}
    for k, m in ((3, 1200), (3, 2400), (5, 1200)):
        for p in (0.50, 0.30, 0.15):
            for sig in (0.0, 0.01, 0.015, 0.02, 0.03):
                for dl in (0.0, 0.03, 0.04, 0.05):
                    if p != 0.50 and sig not in (0.0, 0.015):
                        continue
                    key = f"k={k} m={m} p={p:.2f} sigma_s={sig*100:.1f}pp delta={dl*100:.0f}pp"
                    out["grid"][key] = classify(rng, p, dl, sig, k, m)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
