"""Independent Monte Carlo of X53-N1's staged chain (design_x53_n1_fresh_breadth.md §8), for the 2026-10-26 review.

The chain as the DRAFT words it, one seed pair at Stage A, a second at Stage B:
  L25  600 F games per arm:  HARM-STOP iff Newcombe(method 10) UB of d < -2.0 pp
  L50  1,200 per arm:        FUTILITY-STOP iff d_hat <= -2.0 pp
  L100 2,400 per arm:        SCREEN-GO iff d_hat >= +2.0 pp, else SCREEN-STOP
  B    a fresh pair to 100M (2,400 per arm, no early stops); pooled d over both pairs, normal interval with
       per-arm variance p(1-p)/4800 + s^2/2, s = max(measured seed SD, 1.5 pp) ("max" variant) or s = 1.5 pp
       ("floor" variant); GAIN iff LB > 0 and both pairs' d_hat >= 0.
The measured seed SD is sqrt(max(0, pooled within-arm run variance (2 df) - binomial variance per run)).

Truth: p = base(look) + frac(look) * delta + e_run. e_run ~ N(0, sigma) per run, either shared by the run's looks
("shared", the DRAFT's assumption) or redrawn per look ("indep"). Deterministic: numpy default_rng(5353).
CPU only, a few seconds:
    python power_x53.py > power_x53.log
"""
import numpy as np

Z = 1.959963984540054
N25, N50, N100 = 600, 1200, 2400
DRAWS = 300_000
COURSES = {
    "draft (1/4, 1/2, 1)": (0.25, 0.5, 1.0),
    "late (0, 0, 1)": (0.0, 0.0, 1.0),
    "dip (-1/2, 0, 1)": (-0.5, 0.0, 1.0),
}
BASES = {"flat 0.49": (0.49, 0.49, 0.49), "rising 0.30/0.40/0.45": (0.30, 0.40, 0.45)}


def wilson(w, n):
    p = w / n
    den = 1 + Z * Z / n
    c = (p + Z * Z / (2 * n)) / den
    h = Z * np.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / den
    return p, c - h, c + h


def newcombe_ub(w1, n1, w2, n2):
    p1, _l1, u1 = wilson(w1, n1)
    p2, l2, _u2 = wilson(w2, n2)
    return (p1 - p2) + np.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def chain(rng, delta, sigma, course, base, seed_mode, floor_mode, n100=N100, floor=0.015):
    f = COURSES[course]
    b = BASES[base]
    k = DRAWS

    def eff(shape):
        return sigma * rng.standard_normal(shape)

    eN, eC = eff(k), eff(k)

    def p_at(i, arm_e):
        e = arm_e if seed_mode == "shared" else eff(k)
        return np.clip(b[i] + e, 0.01, 0.99)

    pN = [np.clip(p_at(i, eN) + f[i] * delta, 0.01, 0.99) for i in range(3)]
    pC = [p_at(i, eC) for i in range(3)]
    w = [(rng.binomial(n, pN[i]), rng.binomial(n, pC[i])) for i, n in enumerate((N25, N50, n100))]
    harm = newcombe_ub(w[0][0], N25, w[0][1], N25) < -0.02 - 1e-9
    d50 = (w[1][0] - w[1][1]) / N50
    fut = ~harm & (d50 <= -0.02 + 1e-9)
    d1 = (w[2][0] - w[2][1]) / n100
    alive = ~harm & ~fut
    go = alive & (d1 >= 0.02 + 1e-9)
    # Stage B: a fresh pair at 100M
    eN2, eC2 = eff(k), eff(k)
    pN2 = np.clip(b[2] + eN2 + delta, 0.01, 0.99)
    pC2 = np.clip(b[2] + eC2, 0.01, 0.99)
    wN2, wC2 = rng.binomial(n100, pN2), rng.binomial(n100, pC2)
    d2 = (wN2 - wC2) / n100
    xN1, xC1, xN2, xC2 = w[2][0] / n100, w[2][1] / n100, wN2 / n100, wC2 / n100
    pN_pool, pC_pool = (xN1 + xN2) / 2, (xC1 + xC2) / 2
    dpool = pN_pool - pC_pool
    s2_within = ((xN1 - xN2) ** 2 / 2 + (xC1 - xC2) ** 2 / 2) / 2
    binom = (pN_pool * (1 - pN_pool) + pC_pool * (1 - pC_pool)) / 2 / n100
    s_meas = np.sqrt(np.maximum(0.0, s2_within - binom))
    s = np.maximum(s_meas, floor) if floor_mode == "max" else np.full(k, floor)
    var = pN_pool * (1 - pN_pool) / (2 * n100) + pC_pool * (1 - pC_pool) / (2 * n100) + s * s
    lb = dpool - Z * np.sqrt(var)
    ub = dpool + Z * np.sqrt(var)
    gain = go & (lb > 1e-9) & (d1 >= 0) & (d2 >= 0)
    gph = np.where(harm, 9.85, np.where(fut, 19.7, 39.4)) + np.where(go, 39.4, 0.0)
    null_reach = go & (lb > -0.03) & (ub < 0.03)
    return {
        "harm": harm.mean(),
        "fut": fut.mean(),
        "go": go.mean(),
        "gain": gain.mean(),
        "gpuh": gph.mean(),
        "bias_screen": (d1[go].mean() - delta) if go.any() else float("nan"),
        "bias_gain": (dpool[gain].mean() - delta) if gain.any() else float("nan"),
        "min_halfwidth": (Z * np.sqrt(var[go])).min() if go.any() else float("nan"),
        "null_reach": null_reach.mean(),
    }


def fmt(r):
    return (
        f"harm {r['harm']:.3f} fut {r['fut']:.3f} GO {r['go']:.3f} GAIN {r['gain']:.4f} "
        f"E[GPU-h A+B] {r['gpuh']:.1f} bias|GO {100 * r['bias_screen']:+.2f}pp bias|GAIN {100 * r['bias_gain']:+.2f}pp "
        f"min-halfwidth {100 * r['min_halfwidth']:.2f}pp NULL-reach {r['null_reach']:.4f}"
    )


def main():
    rng = np.random.default_rng(5353)
    print(f"{DRAWS} draws per cell; delta and sigma in pp; GAIN is the final Stage-B GAIN of the whole chain")
    print("\n## A. The DRAFT's own assumptions (flat base 0.49, course 1/4-1/2-1, shared seed effect)")
    for floor_mode in ("floor", "max"):
        for sigma in (0, 1, 2, 3, 4):
            for delta in (0, 2, 4, 6):
                r = chain(rng, delta / 100, sigma / 100, "draft (1/4, 1/2, 1)", "flat 0.49", "shared", floor_mode)
                print(f"[{floor_mode}] sigma {sigma} delta {delta:+d}: {fmt(r)}")
    print("\n## B. Sensitivity: the effect's time course and per-look seed effects ('max' interval)")
    for course in COURSES:
        for seed_mode in ("shared", "indep"):
            for sigma in (1, 3):
                for delta in (0, 4):
                    r = chain(rng, delta / 100, sigma / 100, course, "rising 0.30/0.40/0.45", seed_mode, "max")
                    print(f"{course} | {seed_mode} | sigma {sigma} delta {delta:+d}: {fmt(r)}")
    print("\n## C. More games do not buy power once sigma >= 2 pp (L100 at 4,800 / 9,600 per arm, draft course)")
    for n100 in (2400, 4800, 9600):
        for sigma in (0, 2, 3):
            r = chain(rng, 0.04, sigma / 100, "draft (1/4, 1/2, 1)", "flat 0.49", "shared", "max", n100=n100)
            print(f"n100 {n100} sigma {sigma} delta +4: {fmt(r)}")


if __name__ == "__main__":
    main()
