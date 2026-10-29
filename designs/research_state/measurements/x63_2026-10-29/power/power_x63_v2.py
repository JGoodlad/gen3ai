"""X63 registration (2026-10-29): the REVISED decision rule's power, after the review (x63_review_2026-10-27.md).

What changed against power_x63.py (the draft's) and the review's power_review.py:
  * X63-C s6302 (the control replicate) is TRAINED, so Stage A reads 1 arm v 2 controls (X26c s2601 + X63-C s6302) and
    measures sigma_CC (the control-to-control seed spread at +100M, 1 df) at once. Stage B adds ONE arm seed (2 v 2);
    Stage C adds an arm and a control seed (3 v 3).
  * A Stage-A stop is NOT PURSUED (no era claim; review B1 fix 1). RETIRE needs k >= 2 and UB < +1.5 pp.
  * The seed term is sigma_eff = max(FLOOR, sigma_hat); FLOOR and the quantile (z, or Student-t on sigma_hat's df
    whenever sigma_hat > FLOOR) are the review's B1 fix 2 options, simulated side by side here.
  * INCONCLUSIVE iff sigma_hat > INC (a spread too large for any affordable k).
  * The price: net = Delta_home72 + r * Delta_F (X62's form with F, X63's clean panel, in place of U), PRICE WITHIN iff
    net's point estimate >= 0; the goal-(a) floor on the arm's pooled home72 at HOME games a model.

Deterministic (default_rng(20261029)). CPU, numpy + math only, seconds:
    python power_x63_v2.py > power_x63_v2.log
"""
import math

import numpy as np

P0 = 0.53          # F base (X26c / P*-like on F against Kakuna; H-CTX's engine read P* F 0.540)
H0 = 0.60          # home72 base (P* 0.598)
G = 4000           # F games a model at +100M
HOME = 2000        # home72 games a model at +100M
C_SCREEN = 0.01    # Stage A CONTINUE iff Delta_A >= +1.0 pp
C_STAGE_C = 0.015  # Stage C iff B UNDECIDED with Delta_B >= +1.5 pp
KILL_BAR = 0.015   # RETIRE iff UB < +1.5 pp (k >= 2)
DRAWS = 40000
Z = 1.959963984540054


def t_q(df, q=0.975):
    def cdf(x):
        xs = np.linspace(0, x, 4001)
        dens = math.gamma((df + 1) / 2) / (math.sqrt(df * math.pi) * math.gamma(df / 2)) * (1 + xs ** 2 / df) ** (-(df + 1) / 2)
        return 0.5 + np.trapezoid(dens, xs)
    lo, hi = 0.0, 200.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(mid) < q else (lo, mid)
    return (lo + hi) / 2


TQ = {2: t_q(2), 4: t_q(4)}


def models(rng, n, mean, sigma, games):
    true = np.clip(mean + sigma * rng.standard_normal((DRAWS, n)), 0.01, 0.99)
    return rng.binomial(games, true) / games


def bv(p, g):
    return p * (1 - p) / g


def read_kvk(a, c, g, floor, inc, quant):
    """k v k: Delta, GAIN, RETIRE, INCONCLUSIVE masks. sigma_hat pooled over both sides (df = 2(k-1)), net of binomial."""
    k = a.shape[1]
    delta = a.mean(1) - c.mean(1)
    binv = (bv(a, g).sum(1) + bv(c, g).sum(1)) / k ** 2
    s2 = (a.var(1, ddof=1) + c.var(1, ddof=1)) / 2 - (bv(a, g).mean(1) + bv(c, g).mean(1)) / 2
    sig = np.sqrt(np.clip(s2, 0, None))
    se = np.sqrt(binv + np.maximum(floor, sig) ** 2 * 2 / k)
    q = np.full(DRAWS, Z) if quant == "z" else np.where(sig > floor, TQ[2 * (k - 1)], Z)
    lb, ub = delta - q * se, delta + q * se
    inconc = sig > inc
    return delta, (lb > 0) & ~inconc, (ub < KILL_BAR) & ~inconc, inconc, sig


def staged(rng, d, sa, sc, floor, inc, quant):
    a = models(rng, 3, P0 + d, sa, G)
    c = models(rng, 3, P0, sc, G)
    # Stage A: arm a0 v controls c0 (X26c) and c1 (X63-C)
    dA = a[:, 0] - (c[:, 0] + c[:, 1]) / 2
    cont = dA >= C_SCREEN
    dB, gB, kB, iB, _ = read_kvk(a[:, :2], c[:, :2], G, floor, inc, quant)
    undB = cont & ~gB & ~kB & ~iB
    go_c = undB & (dB >= C_STAGE_C)
    _d, gC, kC, iC, _ = read_kvk(a, c, G, floor, inc, quant)
    gain = (cont & gB) | (go_c & gC)
    kill = (cont & kB) | (go_c & kC)
    inconc = (cont & iB) | (go_c & iC)
    gpu = 19.9 + 19.9 * cont.mean() + 39.8 * go_c.mean()
    return dict(cont=cont.mean(), gain=gain.mean(), np_=(~cont).mean(), retire=kill.mean(), inconc=inconc.mean(),
                gpu=gpu, stage_c=go_c.mean())


SIG_GRID = [(0.0, 0.0), (0.005, 0.005), (0.01, 0.01), (0.015, 0.015), (0.02, 0.02), (0.03, 0.03), (0.02, 0.01),
            (0.03, 0.015)]
RULES = {"floor 1.0 z, INC 2.0 (the draft)": (0.01, 0.02, "z"),
         "floor 1.5 z, INC 3.0": (0.015, 0.03, "z"),
         "floor 1.5 t-when-above, INC 3.0": (0.015, 0.03, "t"),
         "floor 1.0 t-when-above, INC 3.0 (REGISTERED)": (0.01, 0.03, "t")}


def part_a(rng):
    print("## A. The staged rule under each candidate seed term (Stage A 1 v 2 screen at +1.0 pp)")
    print("rule | sigma_arm | sigma_ctl | true Delta | CONTINUE | GAIN | NOT PURSUED | RETIRE | INCONCLUSIVE | P(Stage C) | E[GPU-h]")
    for name, (fl, inc, qu) in RULES.items():
        for sa, sc in SIG_GRID:
            for d in (0.0, 0.01, 0.02, 0.03, 0.04):
                r = staged(rng, d, sa, sc, fl, inc, qu)
                print(f"{name} | {100 * sa:.1f} | {100 * sc:.1f} | {100 * d:+.1f} | {r['cont']:.3f} | {r['gain']:.3f} | "
                      f"{r['np_']:.3f} | {r['retire']:.3f} | {r['inconc']:.3f} | {r['stage_c']:.3f} | {r['gpu']:.1f}")
        print()


def part_b(rng):
    print("## B. sigma_CC from X26c v X63-C (1 df, 4,000 F games each): what one pair says about sigma")
    for s in (0.0, 0.005, 0.01, 0.015, 0.02, 0.03):
        c = models(rng, 2, P0, s, G)
        s2 = ((c[:, 0] - c[:, 1]) ** 2 - bv(c[:, 0], G) - bv(c[:, 1], G)) / 2
        sh = np.sqrt(np.clip(s2, 0, None))
        print(f"   true sigma {100 * s:.1f} pp: median sigma_CC {100 * np.median(sh):.2f} pp; P(sigma_CC = 0) "
              f"{np.mean(s2 <= 0):.2f}; P(sigma_CC > 1.5) {np.mean(sh > 0.015):.2f}; P(> 3.0) {np.mean(sh > 0.03):.2f}")


def part_c(rng):
    print("\n## C. The price: net = Delta_home72 + r * Delta_F, PRICE WITHIN iff net >= 0 (point), given a GAIN-sized Delta_F")
    print("   home72 at 2,000 games a model, seed SD on home72 = the F sigma; k arms v k controls")
    for k in (2, 3):
        for s in (0.01, 0.02):
            for dF, dH in ((0.02, 0.0), (0.02, -0.02), (0.03, -0.02), (0.03, -0.04), (0.02, -0.04)):
                for r in (0.5, 1.0, 2.0):
                    aF = models(rng, k, P0 + dF, s, G); cF = models(rng, k, P0, s, G)
                    aH = models(rng, k, H0 + dH, s, HOME); cH = models(rng, k, H0, s, HOME)
                    net = (aH.mean(1) - cH.mean(1)) + r * (aF.mean(1) - cF.mean(1))
                    print(f"   k {k}, sigma {100 * s:.0f} pp, true dF {100 * dF:+.0f} dH {100 * dH:+.0f}, r {r}: true net "
                          f"{100 * (dH + r * dF):+.1f} pp; P(WITHIN) {np.mean(net >= 0):.2f}")


def part_d():
    print("\n## D. The goal-(a) floor: the arm's pooled home72 Wilson LB > 0.50 at 2,000 games a seed")
    for k in (1, 2, 3):
        n = HOME * k
        for p in np.arange(0.50, 0.60, 0.001):
            centre = (p + Z * Z / (2 * n)) / (1 + Z * Z / n)
            half = Z * math.sqrt(p * (1 - p) / n + Z * Z / (4 * n * n)) / (1 + Z * Z / n)
            if centre - half > 0.50:
                print(f"   {k} seed(s), {n} games: passes at observed >= {p:.3f} (a drop of up to {100 * (0.598 - p):.1f} pp from P*)")
                break


def part_e(rng):
    print("\n## E. P(GAIN) over each registered GUESS, integrated over the review's sigma prior (0.5/1/1.5/2/3 pp at "
          "0.15/0.25/0.25/0.2/0.15), REGISTERED rule")
    fl, inc, qu = RULES["floor 1.0 t-when-above, INC 3.0 (REGISTERED)"]
    deltas = np.array([-0.02, -0.01, 0.0, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05, 0.06])
    prior = {0.005: 0.15, 0.01: 0.25, 0.015: 0.25, 0.02: 0.2, 0.03: 0.15}
    curves, gpus = {}, {}
    for s in prior:
        rows = [staged(rng, d, s, s, fl, inc, qu) for d in deltas]
        curves[s] = np.array([x["gain"] for x in rows])
        gpus[s] = np.array([x["gpu"] for x in rows])
    z = np.linspace(-4, 4, 801)
    w = np.exp(-z * z / 2)
    w /= w.sum()
    guesses = {"U3-K3 as registered, no PR-1' read yet: +1.5 (-1.5, +4.5)": (1.5, 6.0),
               "after a PR-1' KILL: +0.5 (-2.0, +3.0)": (0.5, 5.0),
               "after a PR-1' BREADTH-GAIN: +2.5 (-0.5, +5.5)": (2.5, 6.0),
               "the draft's favourable GUESS: +2.0 (-1.0, +5.0)": (2.0, 6.0)}
    for name, (mu, width) in guesses.items():
        sd = width / 2.563
        pg = sum(ws * float((np.interp((mu + sd * z) / 100, deltas, curves[s]) * w).sum()) for s, ws in prior.items())
        eg = sum(ws * float((np.interp((mu + sd * z) / 100, deltas, gpus[s]) * w).sum()) for s, ws in prior.items())
        print(f"   {name}: P(GAIN) {pg:.2f}; E[GPU-h] {eg:.1f}")


def main():
    rng = np.random.default_rng(20261029)
    print(f"# X63 revised rule power. F base {P0}, {G} F games a model; home72 base {H0}, {HOME} games; {DRAWS} draws a cell")
    print(f"# t quantiles: df 2 {TQ[2]:.3f}, df 4 {TQ[4]:.3f}")
    part_a(rng)
    part_b(rng)
    part_c(rng)
    part_d()
    part_e(rng)


if __name__ == "__main__":
    main()
