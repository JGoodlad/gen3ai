"""X55: what class each X35 arm is PREDICTED to read under R1's actual power, given a predicted effect.

    PYTHONPATH=<checkout>/src python predictions_sim.py > predictions_sim.txt

R1 (design_compute_efficiency.md §5.6.4; main.x35.readout.decide_r1): Welch two-sample interval over seed finals of S,
97.5 % two-sided per look, ±2.0 pp bar; look 1 = 3 v 3, an UNDECIDED with Δ̂ > 0 is CONTINUE, look 2 = 5 v 5. Per
draw: a TRUE effect Δ ~ N(mu, sd) (the prediction's own uncertainty) and a per-final SD σ from the σ prior (lognormal,
median 1.0 pp, 10-90 % 0.70-1.43 pp; bracket from X35_BATCH1_REVIEW.md §1), then seed finals S ~ N(Δ or 0, σ²).
Also G1 (AM-2): σ̂ = pooled within-arm SD over A1 / C0 × 3 seeds (4 df), PASS iff σ̂ ≤ 1.2.

The vectorised class is cross-checked against the kit's decide_r1 on 2,000 draws (0 mismatches required).
Deterministic: one fixed seed.
"""
from __future__ import annotations

import sys

import numpy as np
from scipy.stats import t as T

from main.x35 import readout as R

N = 200_000
BAR = 2.0
rng = np.random.default_rng(20261020)


def welch(on, off, conf=0.975):
    n1, n2 = on.shape[1], off.shape[1]
    a = on.var(axis=1, ddof=1) / n1
    b = off.var(axis=1, ddof=1) / n2
    v = a + b
    df = v * v / (a * a / (n1 - 1) + b * b / (n2 - 1))
    q = T.ppf(1 - (1 - conf) / 2, df)
    d = on.mean(1) - off.mean(1)
    se = np.sqrt(v)
    return d, d - q * se, d + q * se


def cls(lo, hi):
    c = np.full(lo.shape, "UNDECIDED", dtype=object)
    c[lo > BAR] = "GAIN"
    c[hi < -BAR] = "LOSS"
    c[(lo > -BAR) & (hi < BAR)] = "NULL"
    return c


def sigma_prior(n):
    return np.exp(rng.normal(np.log(1.0), 0.28, n))       # median 1.0; 10-90 % ≈ 0.70-1.43


def run(mu, sd, label):
    sig = sigma_prior(N)
    delta = rng.normal(mu, sd, N)
    on = rng.normal(delta[:, None], sig[:, None], (N, 5))
    off = rng.normal(0.0, sig[:, None], (N, 5))
    d1, lo1, hi1 = welch(on[:, :3], off[:, :3])
    c1 = cls(lo1, hi1)
    cont = (c1 == "UNDECIDED") & (d1 > 0)
    look1 = np.where(cont, "CONTINUE", c1)
    d2, lo2, hi2 = welch(on, off)
    c2 = cls(lo2, hi2)
    final = np.where(cont, c2, c1)
    pooled = np.sqrt((on[:, :3].var(axis=1, ddof=1) + off[:, :3].var(axis=1, ddof=1)) / 2)
    out = {"label": label, "mu": mu, "sd": sd,
           "look1": {k: round(float(np.mean(look1 == k)), 3) for k in ("GAIN", "NULL", "LOSS", "CONTINUE",
                                                                      "UNDECIDED")},
           "final_with_look2": {k: round(float(np.mean(final == k)), 3) for k in ("GAIN", "NULL", "LOSS",
                                                                                 "UNDECIDED")},
           "final_UNDECIDED_dhat_pos": round(float(np.mean((final == "UNDECIDED") & (np.where(cont, d2, d1) > 0))), 3),
           "look1_dhat_pos": round(float(np.mean(d1 > 0)), 3),
           "dhat_look1_q10_q50_q90": [round(float(np.quantile(d1, q)), 2) for q in (0.1, 0.5, 0.9)],
           "g1_pass": round(float(np.mean(pooled <= 1.2)), 3),
           "sigma_hat_q10_q50_q90": [round(float(np.quantile(pooled, q)), 2) for q in (0.1, 0.5, 0.9)]}
    return out, (on, off, c1)


def main() -> int:
    _, (on, off, c1) = run(2.5, 0.0, "crosscheck")
    bad = 0
    for i in range(2000):
        r = R.decide_r1(1, {str(s): float(on[i, s]) for s in range(3)}, {str(s): float(off[i, s]) for s in range(3)})
        want = r.cls
        bad += want != c1[i]
    print(f"cross-check vs main.x35.readout.decide_r1 (look 1 class): {bad} mismatches / 2000")
    arms = [(0.5, 1.2, "A1 archive (predicted Δ +0.5, sd 1.2)"),
            (0.0, 0.6, "A2 PFSP over the pool (predicted Δ 0.0, sd 0.6)"),
            (-0.3, 1.0, "A3 main exploiters folded in, R1 only (predicted Δ -0.3, sd 1.0)"),
            (-0.3, 1.2, "A6 league exploiters, R1 only (predicted Δ -0.3, sd 1.2)"),
            (0.0, 0.0, "reference: a truly inert arm"),
            (3.0, 0.0, "reference: a true +3.0 pp effect (the measured cyclic size)")]
    import json
    for mu, sd, lab in arms:
        o, _ = run(mu, sd, lab)
        print(json.dumps(o))
    return 0


if __name__ == "__main__":
    sys.exit(main())
