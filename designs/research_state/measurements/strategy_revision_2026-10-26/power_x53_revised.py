"""Power of X53-N1's chain, DRAFT and REVISED (design_x53_n1_fresh_breadth.md §8), for the 2026-10-26 revision.

Extends the review's rebuild (../strategy_review_2026-10-26/power_x53.py, whose DRAFT numbers this reproduces) with
the revision's knobs:
  - stops: "draft" (L25 HARM iff Newcombe UB < -2 pp; L50 FUTILITY iff d_hat <= -2 pp) or "revised" (L25 HARM iff
    UB < -5 pp, a catastrophic bar; L50 report-only, never stops);
  - the screen bar at L100 (d_hat >= bar);
  - Stage B: m_B further seed pairs to 100M; the GAIN test pools every pair (normal interval, per-run seed SD
    s = max(measured, floor), measured from the within-arm run spread); the sign rule: every pair's d_hat >= 0;
  - the ESTIMATE reported: the pooled d (screen-selected) and the Stage-B pairs alone (unselected, M4);
  - RETIRE (M5): a SCREEN-STOP, or Stage B not GAIN with its UB < +3 pp ("KILL").
Truth: p = base(look) + frac(look) * delta + e_run, e_run ~ N(0, sigma) shared by a run's looks. Base rising
0.30 / 0.40 / 0.45 (the review's §B), or flat 0.49. Deterministic: numpy default_rng(5354). CPU only, ~1 min:
    python power_x53_revised.py > power_x53_revised.log
"""
import numpy as np

Z = 1.959963984540054
EPS = 1e-9
N25, N50, N100 = 600, 1200, 2400
DRAWS = 100_000
COURSES = {"draft": (0.25, 0.5, 1.0), "late": (0.0, 0.0, 1.0), "dip": (-0.5, 0.0, 1.0)}
GPUH_RUN_100M = 19.7


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


def chain(rng, delta, sigma, *, course="draft", base=(0.30, 0.40, 0.45), stops="revised", bar=0.02,
          m_b=1, floor=0.015):
    f = COURSES[course]
    k = DRAWS
    eN, eC = sigma * rng.standard_normal(k), sigma * rng.standard_normal(k)
    pN = [np.clip(base[i] + eN + f[i] * delta, 0.01, 0.99) for i in range(3)]
    pC = [np.clip(base[i] + eC, 0.01, 0.99) for i in range(3)]
    w = [(rng.binomial(n, pN[i]), rng.binomial(n, pC[i])) for i, n in enumerate((N25, N50, N100))]
    harm_bar = -0.02 if stops == "draft" else -0.05
    harm = newcombe_ub(w[0][0], N25, w[0][1], N25) < harm_bar - EPS
    if stops == "draft":
        fut = ~harm & ((w[1][0] - w[1][1]) / N50 <= -0.02 + EPS)
    else:
        fut = np.zeros(k, bool)
    alive = ~harm & ~fut
    d1 = (w[2][0] - w[2][1]) / N100
    go = alive & (d1 >= bar + EPS)
    # Stage B: m_b fresh pairs at 100M
    xN = [w[2][0] / N100]
    xC = [w[2][1] / N100]
    for _ in range(m_b):
        e2N, e2C = sigma * rng.standard_normal(k), sigma * rng.standard_normal(k)
        xN.append(rng.binomial(N100, np.clip(base[2] + e2N + delta, 0.01, 0.99)) / N100)
        xC.append(rng.binomial(N100, np.clip(base[2] + e2C, 0.01, 0.99)) / N100)
    XN, XC = np.stack(xN, 1), np.stack(xC, 1)
    m = m_b + 1
    pn, pc = XN.mean(1), XC.mean(1)
    dpool = pn - pc
    dB = XN[:, 1:].mean(1) - XC[:, 1:].mean(1)
    s2w = (XN.var(1, ddof=1) + XC.var(1, ddof=1)) / 2
    binom = (pn * (1 - pn) + pc * (1 - pc)) / 2 / N100
    s = np.maximum(np.sqrt(np.maximum(0.0, s2w - binom)), floor)
    var = (pn * (1 - pn) + pc * (1 - pc)) / (m * N100) + 2 * s * s / m
    lb, ub = dpool - Z * np.sqrt(var), dpool + Z * np.sqrt(var)
    sign = ((XN - XC) >= 0).all(1)
    gain = go & (lb > EPS) & sign
    kill = go & ~gain & (ub < 0.03 - EPS)
    retire = alive & ~go | kill
    gph = np.where(harm, 2 * GPUH_RUN_100M / 4, np.where(fut, GPUH_RUN_100M, 2 * GPUH_RUN_100M))
    gph = gph + np.where(go, 2 * m_b * GPUH_RUN_100M, 0.0)
    return dict(stop=(harm | fut).mean(), go=go.mean(), gain=gain.mean(), retire=retire.mean(), kill=kill.mean(),
                gpuh=gph.mean(),
                bias_pool=100 * (dpool[gain].mean() - delta) if gain.sum() > 20 else float("nan"),
                bias_b=100 * (dB[gain].mean() - delta) if gain.sum() > 20 else float("nan"),
                bias_b_go=100 * (dB[go].mean() - delta) if go.sum() > 20 else float("nan"))


def fmt(r):
    return (f"early-stop {r['stop']:.3f} GO {r['go']:.3f} GAIN {r['gain']:.4f} RETIRE {r['retire']:.3f} "
            f"(KILL {r['kill']:.3f}) E[GPU-h] {r['gpuh']:.1f} | bias|GAIN pooled {r['bias_pool']:+.2f} pp, "
            f"Stage-B-only {r['bias_b']:+.2f} pp; Stage-B-only bias|GO {r['bias_b_go']:+.2f} pp")


def main():
    rng = np.random.default_rng(5354)
    print(f"{DRAWS} draws a cell; delta, sigma, floor in pp; base rising 0.30/0.40/0.45 unless stated")
    print("\n## A. M1: false GAIN (delta 0) of the DRAFT chain (draft stops, bar +2, one Stage-B pair) by floor")
    for floor in (0.015, 0.03):
        for sigma in (0.0, 0.01, 0.02, 0.03):
            r = chain(rng, 0.0, sigma, stops="draft", floor=floor, base=(0.49,) * 3)
            print(f"  floor {100*floor:.1f} sigma {100*sigma:.0f} (flat base 0.49): {fmt(r)}")
    print("\n## B. The DRAFT chain at realistic effects (M2): floor 1.5 vs 3.0, draft course")
    for floor in (0.015, 0.03):
        for sigma in (0.01, 0.02, 0.03):
            for d in (0.02, 0.03, 0.04):
                r = chain(rng, d, sigma, stops="draft", floor=floor)
                print(f"  floor {100*floor:.1f} sigma {100*sigma:.0f} delta +{100*d:.0f}: {fmt(r)}")
    print("\n## C. M3: the early stops under late / dip courses, draft vs revised stops (sigma 1, floor 1.5, delta +4)")
    for stops in ("draft", "revised"):
        for course in COURSES:
            r = chain(rng, 0.04, 0.01, course=course, stops=stops)
            print(f"  stops {stops:7s} course {course:5s}: {fmt(r)}")
    print("\n## D. The REVISED chain (revised stops, bar +2): Stage-B pairs m_B and floor (= the proxy's sigma bound)")
    for m_b in (1, 3):
        for floor in (0.01, 0.015, 0.02, 0.03):
            for sigma in (floor,):
                for d in (0.0, 0.02, 0.03, 0.04):
                    for course in (("draft", "late") if d else ("draft",)):
                        r = chain(rng, d, sigma, m_b=m_b, floor=floor, course=course)
                        print(f"  m_B {m_b} floor=sigma {100*floor:.1f} delta +{100*d:.0f} course {course:5s}: {fmt(r)}")


if __name__ == "__main__":
    main()
