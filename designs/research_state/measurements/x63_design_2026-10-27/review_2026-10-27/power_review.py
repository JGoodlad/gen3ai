"""X63 adversarial review (2026-10-27): the design's decision rule re-run under the reviewer's own assumptions.

Re-implements the registered shape of design_x63_long_breadth_continuation.md §7 exactly as power_x63.py does
(Stage A 1 v 1 screen at +1.0 pp; Stage B 2 v 2 with sigma_eff = max(1 pp, sigma_hat), GAIN iff LB > 0, RETIRE iff
UB < +1.5 pp, INCONCLUSIVE iff sigma_hat > 2 pp; Stage C 3 v 3 iff B UNDECIDED with Delta_B >= +1.5 pp), and adds:

  A. a wider sigma grid (to 3 pp: a +100M continuation's seed spread is unmeasured, and 100M is 5x a 20M fork's
     divergence time), and a HETEROSCEDASTIC case (arm seeds 2x the control's spread, D1's S pattern 0.86 vs 0.35);
  B. the same rule with a Student-t quantile on sigma_hat's df instead of 1.96 (what a df-2 / df-4 estimate earns);
  C. the S "price ceiling" (TRADE iff Welch 97.5 % LB of Delta_S <= -6.0 pp) at k = 2 and 3, with D1's per-seed S SDs;
  D. the goal-(a) floor (pooled home72 LB > 0.50 at 1,200 games a seed): the largest drop from P*'s 0.598 it passes;
  E. the +20M "dose curve" point (1,200 F games a model) and X26c's plateau-wobble trigger (three checkpoints at
     1,200 / 1,200 / 4,000 games, spread net of binomial, trigger > 1.5 pp);
  F. P(GAIN) over the design's GUESS integrated over a sigma prior (not a single sigma).

Deterministic (default_rng(20261027)). CPU, numpy + math only, a few seconds:
    python power_review.py > power_review.log
"""
import math

import numpy as np

P0 = 0.53
G = 4000
SIGMA_A = 0.01
C_SCREEN = 0.01
C_STAGE_C = 0.015
KILL_BAR = 0.015
DRAWS = 40000
Z = 1.959963984540054


def t_q(df, q=0.975):
    """Student-t quantile by bisection on the regularised incomplete beta (numpy-free of scipy)."""
    def cdf(x):
        # P(T <= x) for x >= 0 via the incomplete beta; numeric integration of the density is enough here
        xs = np.linspace(0, x, 4001)
        dens = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2)) * (1 + xs ** 2 / df) ** (-(df + 1) / 2)
        return 0.5 + np.trapezoid(dens, xs)
    lo, hi = 0.0, 200.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if cdf(mid) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def models(rng, n, mean, sigma, games):
    true = np.clip(mean + sigma * rng.standard_normal((DRAWS, n)), 0.01, 0.99)
    return rng.binomial(games, true) / games


def bv(p, g):
    return p * (1 - p) / g


def read_kvk(a, c, g, quantile="z"):
    k = a.shape[1]
    delta = a.mean(1) - c.mean(1)
    binv = (bv(a, g).sum(1) + bv(c, g).sum(1)) / k ** 2
    s2 = (a.var(1, ddof=1) + c.var(1, ddof=1)) / 2 - (bv(a, g).mean(1) + bv(c, g).mean(1)) / 2
    sig_hat = np.sqrt(np.clip(s2, 0, None))
    se = np.sqrt(binv + np.maximum(SIGMA_A, sig_hat) ** 2 * 2 / k)
    q = Z if quantile == "z" else t_q(2 * (k - 1))
    lb, ub = delta - q * se, delta + q * se
    inconc = sig_hat > 2 * SIGMA_A
    return delta, (lb > 0) & ~inconc, (ub < KILL_BAR) & ~inconc, inconc


def staged(rng, d, sig_arm, sig_ctl, quantile="z"):
    a = models(rng, 3, P0 + d, sig_arm, G)
    c = models(rng, 3, P0, sig_ctl, G)
    cont = (a[:, 0] - c[:, 0]) >= C_SCREEN
    dB, gB, kB, iB = read_kvk(a[:, :2], c[:, :2], G, quantile)
    undB = cont & ~gB & ~kB & ~iB
    go_c = undB & (dB >= C_STAGE_C)
    _d, gC, kC, iC = read_kvk(a, c, G, quantile)
    gain = (cont & gB) | (go_c & gC)
    kill = (cont & kB) | (go_c & kC)
    inconc = (cont & iB) | (go_c & iC)
    return dict(cont=cont.mean(), gain=gain.mean(), retire=(~cont).mean() + kill.mean(), inconc=inconc.mean())


def part_a_b(rng):
    print("## A/B. The registered rule (z) and the same rule with a t quantile on sigma_hat's df (2 at B, 4 at C)")
    print("sigma_arm | sigma_ctl | true Delta | CONTINUE | GAIN z | GAIN t | RETIRE z | INCONCLUSIVE z")
    cases = [(s, s) for s in (0.0, 0.01, 0.015, 0.02, 0.025, 0.03)] + [(0.02, 0.01), (0.03, 0.015)]
    for sa, sc in cases:
        for d in (0.0, 0.01, 0.02, 0.03):
            rz = staged(rng, d, sa, sc, "z")
            rt = staged(rng, d, sa, sc, "t")
            print(f"{100 * sa:.1f} | {100 * sc:.1f} | {100 * d:+.1f} | {rz['cont']:.3f} | {rz['gain']:.3f} | "
                  f"{rt['gain']:.3f} | {rz['retire']:.3f} | {rz['inconc']:.3f}")


def part_c(rng):
    print("\n## C. The S price ceiling: P(TRADE) = P(Welch 97.5 % LB of Delta_S <= -6.0 pp)")
    print("   per-model S SD (seed + measurement): arm 0.86 pp, control 0.35 pp (D1's finals, the X62 review's C)")
    for k in (2, 3):
        for true in (0.0, -1.6, -2.5, -4.0, -6.0):
            a = true + 0.86 * rng.standard_normal((DRAWS, k))
            c = 0.35 * rng.standard_normal((DRAWS, k))
            va, vc = a.var(1, ddof=1) / k, c.var(1, ddof=1) / k
            se = np.sqrt(va + vc)
            df = (va + vc) ** 2 / (va ** 2 / (k - 1) + vc ** 2 / (k - 1))
            # vectorised t quantile: tabulate on a df grid
            grid = np.linspace(1.0, 2 * (k - 1), 41)
            tq = np.array([t_q(g) for g in grid])
            q = np.interp(np.clip(df, 1.0, 2 * (k - 1)), grid, tq)
            lb = (a.mean(1) - c.mean(1)) - q * se
            print(f"   k = {k}, true Delta_S {true:+.1f} pp: P(TRADE) {np.mean(lb <= -6.0):.3f}; median LB "
                  f"{np.median(lb):+.1f} pp; median t quantile {np.median(q):.1f}")


def part_d():
    print("\n## D. The goal-(a) floor: pooled home72 LB > 0.50 (Wilson), 1,200 games a seed")
    for k in (1, 2, 3):
        n = 1200 * k
        for p in np.arange(0.50, 0.60, 0.001):
            centre = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
            half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
            if centre - half > 0.50:
                print(f"   {k} seed(s), {n} games: passes at an observed home72 >= {p:.3f}, i.e. a drop of up to "
                      f"{100 * (0.598 - p):.1f} pp from P*'s 0.598 still PASSES")
                break


def part_e(rng):
    print("\n## E1. The +20M dose-curve point: Delta at 1,200 F games a model, 1 v 1")
    se = math.sqrt(2 * P0 * (1 - P0) / 1200)
    print(f"   binomial SE of a 1 v 1 Delta at 1,200 games: {100 * se:.2f} pp (95 % half-width {100 * Z * se:.1f} pp)")
    se100 = math.sqrt(2 * P0 * (1 - P0) / 4000)
    print(f"   SE of Delta(+100M) - Delta(+20M) within one pair (independent games): "
          f"{100 * math.sqrt(se ** 2 + se100 ** 2):.2f} pp; the extension trigger (rise >= 1 pp) fires by noise alone "
          f"with P = {0.5 * math.erfc(1.0 / (100 * math.sqrt(se ** 2 + se100 ** 2)) / math.sqrt(2)):.2f}")
    print("\n## E2. X26c's plateau wobble as a sigma proxy: 3 checkpoints at 1,200 / 1,200 / 4,000 F games")
    n = np.array([1200, 1200, 4000])
    for sw in (0.0, 0.01, 0.015, 0.02):
        true = np.clip(P0 + sw * rng.standard_normal((DRAWS, 3)), 0.01, 0.99)
        p = rng.binomial(n, true) / n
        s2 = p.var(1, ddof=1) - (p * (1 - p) / n).mean(1)
        net = np.sqrt(np.clip(s2, 0, None))
        print(f"   true wobble SD {100 * sw:.1f} pp: P(net spread > 1.5 pp) {np.mean(net > 0.015):.3f}; "
              f"median net {100 * np.median(net):.2f} pp")


def part_f(rng):
    print("\n## F. P(GAIN) over the design's GUESS +2.0 (-1.0, +5.0), integrated over a sigma prior")
    sd = 6.0 / 2.563
    deltas = np.array([-0.02, -0.01, 0.0, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06])
    priors = {"design (sigma 1 pp)": {0.01: 1.0},
              "reviewer (0.5/1/1.5/2/3 pp at 0.15/0.25/0.25/0.2/0.15)": {0.005: 0.15, 0.01: 0.25, 0.015: 0.25,
                                                                          0.02: 0.2, 0.03: 0.15}}
    curves = {}
    for s in (0.005, 0.01, 0.015, 0.02, 0.03):
        curves[s] = np.array([staged(rng, d, s, s)["gain"] for d in deltas])
    z = np.linspace(-4, 4, 801)
    w = np.exp(-z * z / 2)
    w /= w.sum()
    for name, pr in priors.items():
        tot = 0.0
        for s, ws in pr.items():
            tot += ws * float((np.interp((2.0 + sd * z) / 100, deltas, curves[s]) * w).sum())
        print(f"   {name}: P(GAIN) {tot:.2f}")
    # the both-NULL GUESS +1.0 (-1.5, +4.0)
    sd2 = 5.5 / 2.563
    pr = priors["reviewer (0.5/1/1.5/2/3 pp at 0.15/0.25/0.25/0.2/0.15)"]
    tot = sum(ws * float((np.interp((1.0 + sd2 * z) / 100, deltas, curves[s]) * w).sum()) for s, ws in pr.items())
    print(f"   both-NULL GUESS +1.0 (-1.5, +4.0), reviewer sigma prior: P(GAIN) {tot:.2f}")
    # a PR-1-KILL-conditioned GUESS: the family mean at 20M has UB < +1.5; the reviewer's MY JUDGEMENT +0.5 (-2, +3)
    sd3 = 5.0 / 2.563
    tot = sum(ws * float((np.interp((0.5 + sd3 * z) / 100, deltas, curves[s]) * w).sum()) for s, ws in pr.items())
    print(f"   after a PR-1 KILL, GUESS +0.5 (-2.0, +3.0) (reviewer's judgement), reviewer sigma prior: P(GAIN) {tot:.2f}")


def main():
    rng = np.random.default_rng(20261027)
    print(f"# X63 review power. P0 {P0}, {G} F games a model, the registered rule of design §7; {DRAWS} draws a cell")
    part_a_b(rng)
    part_c(rng)
    part_d()
    part_e(rng)
    part_f(rng)


if __name__ == "__main__":
    main()
