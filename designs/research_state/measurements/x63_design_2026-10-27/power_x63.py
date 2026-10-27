"""X63's power and P(GAIN) (design_x63_long_breadth_continuation.md §6), 2026-10-27.

The registered shape (the design's §6):
- Stage A: ONE arm seed (X63-U s2601, X26 @380M + 100M on the union mix) against X26c @480M (the banked pool-only
  continuation from the same checkpoint at the same pin), G_A F games each against Kakuna. A SCREEN, not a verdict:
  CONTINUE iff the point estimate Delta_A >= C_SCREEN, else STOP (RETIRE, with this power printed).
- Stage B (on CONTINUE): one more arm seed (s6302) and ONE pool-only control replicate (X63-C s6302, X26c's argv at
  seed 6302), G_B F games each; the two Stage-A models stay at G_A = G_B games. The final contrast is 2 v 2:
  Delta = mean(arms) - mean(controls); SE^2 = sum_i p_i(1 - p_i) / G / 4 + sigma_eff^2, sigma_eff = max(SIGMA_A,
  sigma_hat), sigma_hat from the two within-arm seed pairs (df 2) net of the binomial part. INCONCLUSIVE iff
  sigma_hat > 2 * SIGMA_A (more seeds needed). Classes: GAIN iff LB > 0; KILL iff UB < +1.5 pp; else UNDECIDED.
- Stage C (only if Stage B is UNDECIDED with Delta_B >= C_STAGE_C): one more pair (s6303), read 3 v 3 by the same
  rule. The two looks are not alpha-corrected; the false-GAIN column prints what that costs.
- Each model's TRUE F = P0 + (Delta if arm) + N(0, sigma^2): sigma is the true between-seed SD of a +100M plateau
  continuation's F score, the design's one unknown.
Beside it: the same without Stage C, a 3 v 3 Stage B at once, the registered shape with a +0.5 pp screen, and a
1 v 1 final (Stage A read as a verdict with SIGMA_A in the interval, and without it), for the cost-power trade. Then
P(GAIN) over the design's GUESSES, each read as a normal with its 80 % interval's width / 2.563 as SD (pgain_rows.py's
approximation).
Deterministic (default_rng(6327)). CPU only, numpy, well under a minute:
    python power_x63.py > power_x63.log
"""
import math

import numpy as np

P0 = 0.53          # X26c @480M's F (217 / 400 = 0.543) and P*'s (0.530), old regime; the new regime's is unread
G_A = 4000         # F games per model at Stage A's final (200 units)
G_B = 4000         # F games per Stage-B model (the Stage-A models are not topped up)
SIGMA_A = 0.01     # the registered seed allowance (1 pp)
C_SCREEN = 0.01    # Stage A CONTINUE iff Delta_A >= +1.0 pp
C_STAGE_C = 0.015  # Stage C (a third pair) iff Stage B is UNDECIDED with Delta_B >= +1.5 pp
KILL_BAR = 0.015
DRAWS = 40000
DELTAS = (-0.01, 0.0, 0.01, 0.015, 0.02, 0.03, 0.04)
SIGMAS = (0.0, 0.005, 0.01, 0.015, 0.02)
GPU_H_PER_100M = 19.92   # X26c's own dose record: 380M -> 480M in 19.92 GPU-h at 1,402 steps/s (x35 dose/rb_x26c_s2601.json)
GUESSES = {"X63, gates favourable": (2.0, -1.0, 5.0), "X63, N1-P and D3 both NULL": (1.0, -1.5, 4.0)}


def models(rng, n, mean, sigma, games):
    true = np.clip(mean + sigma * rng.standard_normal((DRAWS, n)), 0.01, 0.99)
    return rng.binomial(games, true) / games


def classify(delta, se):
    lb, ub = delta - 1.959963984540054 * se, delta + 1.959963984540054 * se
    return lb > 0, ub < KILL_BAR


def binom_var(p, g):
    return p * (1 - p) / g


def read_kvk(a, c, g):
    """The k v k final: (delta, gain, kill, inconclusive) per draw, a / c the arm / control seed scores."""
    k = a.shape[1]
    delta = a.mean(1) - c.mean(1)
    bv = (binom_var(a, g).sum(1) + binom_var(c, g).sum(1)) / k ** 2
    # sigma_hat^2: the within-arm sample variance of the seed scores, net of each score's binomial variance
    s2 = (a.var(1, ddof=1) + c.var(1, ddof=1)) / 2 - (binom_var(a, g).mean(1) + binom_var(c, g).mean(1)) / 2
    sig_hat = np.sqrt(np.clip(s2, 0, None))
    se = np.sqrt(bv + np.maximum(SIGMA_A, sig_hat) ** 2 * 2 / k)
    gain, kill = classify(delta, se)
    inconc = sig_hat > 2 * SIGMA_A
    return delta, gain & ~inconc, kill & ~inconc, inconc


def staged(rng, d, sigma, pairs_b, stage_c=False, c_screen=C_SCREEN):
    """Stage A (1 v 1, screen), Stage B (+`pairs_b` pairs), and, with `stage_c`, Stage C (+1 pair) iff Stage B is
    UNDECIDED with its estimate >= C_STAGE_C. Returns the outcome shares and E[GPU-h]."""
    n = 1 + pairs_b + (1 if stage_c else 0)
    a = models(rng, n, P0 + d, sigma, G_A)
    c = models(rng, n, P0, sigma, G_A)
    cont = (a[:, 0] - c[:, 0]) >= c_screen
    kb = 1 + pairs_b
    dB, gB, kB, iB = read_kvk(a[:, :kb], c[:, :kb], G_A)
    undB = cont & ~gB & ~kB & ~iB
    gain, kill, inconc = cont & gB, cont & kB, cont & iB
    gpu_pairs = 2 * pairs_b * cont.astype(float)
    if stage_c:
        go_c = undB & (dB >= C_STAGE_C)
        _dC, gC, kC, iC = read_kvk(a, c, G_A)
        gain = gain | (go_c & gC)
        kill = kill | (go_c & kC)
        inconc = inconc | (go_c & iC)
        gpu_pairs = gpu_pairs + 2 * go_c
    retire = ~cont
    und = ~retire & ~gain & ~kill & ~inconc
    gpu = GPU_H_PER_100M * (1 + gpu_pairs.mean())
    return dict(cont=cont.mean(), gain=gain.mean(), kill=kill.mean(), retire=retire.mean(),
                und=und.mean(), inconc=inconc.mean(), gpu=gpu)


def one_v_one(rng, d, sigma, allowance):
    a = models(rng, 1, P0 + d, sigma, G_A)[:, 0]
    c = models(rng, 1, P0, sigma, G_A)[:, 0]
    se = np.sqrt(binom_var(a, G_A) + binom_var(c, G_A) + 2 * allowance ** 2)
    gain, kill = classify(a - c, se)
    return gain.mean(), kill.mean()


def expected(curve, mu, sd):
    xs = np.array(sorted(curve))
    ys = np.array([curve[x] for x in xs])
    z = np.linspace(-4, 4, 801)
    w = np.exp(-z * z / 2)
    w /= w.sum()
    return float((np.interp((mu + sd * z) / 100, xs, ys) * w).sum())


def main():
    rng = np.random.default_rng(6327)
    print(f"# X63 power. P0 {P0}, G_A {G_A}, G_B {G_B} F games a model, SIGMA_A {100 * SIGMA_A:.1f} pp, "
          f"screen Delta_A >= {100 * C_SCREEN:+.1f} pp, KILL iff UB < {100 * KILL_BAR:+.1f} pp, {DRAWS} draws a cell; "
          f"{GPU_H_PER_100M:.2f} GPU-h per 100M (X26c's measured)")
    curves = {}
    shapes = (("reg", 1, True, C_SCREEN, "THE REGISTERED SHAPE: A 1 v 1 screen (+1.0 pp), B 2 v 2, C 3 v 3 iff B UNDECIDED "
                                         "with Delta_B >= +1.5 pp"),
              ("2v2", 1, False, C_SCREEN, "A 1 v 1 screen (+1.0 pp), B 2 v 2, no Stage C"),
              ("3v3", 2, False, C_SCREEN, "A 1 v 1 screen (+1.0 pp), B 3 v 3 at once"),
              ("reg05", 1, True, 0.005, "the registered shape with a +0.5 pp screen"))
    for key, pairs_b, stage_c, cs, label in shapes:
        print(f"\n## {label}")
        print("true Delta | sigma | CONTINUE | GAIN | KILL (final) | RETIRE (screen stop) | UNDECIDED | INCONCLUSIVE "
              "| E[GPU-h]")
        for s in SIGMAS:
            for d in DELTAS:
                r = staged(rng, d, s, pairs_b, stage_c, cs)
                curves.setdefault((key, s), {})[d] = r["gain"]
                print(f"{100 * d:+.1f} | {100 * s:.1f} | {r['cont']:.3f} | {r['gain']:.3f} | {r['kill']:.3f} | "
                      f"{r['retire']:.3f} | {r['und']:.3f} | {r['inconc']:.3f} | {r['gpu']:.1f}")
    print("\n## 1 v 1 read as a FINAL (Stage A alone): GAIN / KILL, with the 1 pp allowance | binomial SE only")
    for s in SIGMAS:
        cells = []
        for d in DELTAS:
            g1, k1 = one_v_one(rng, d, s, SIGMA_A)
            g0, k0 = one_v_one(rng, d, s, 0.0)
            cells.append(f"{100 * d:+.1f}: {g1:.3f}/{k1:.3f} | {g0:.3f}/{k0:.3f}")
        print(f"sigma {100 * s:.1f} pp: " + "; ".join(cells))
    print("\n## P(GAIN) over the GUESSES (normal; SD = 80 % width / 2.563)")
    for name, (pt, lo, hi) in GUESSES.items():
        sd = (hi - lo) / 2.563
        row = ", ".join(f"{k} sigma {100 * s:.1f}: {expected(curves[(k, s)], pt, sd):.2f}"
                        for k in ("reg", "2v2", "3v3", "reg05") for s in (0.005, 0.01, 0.015))
        print(f"  {name:28s} GUESS {pt:+.1f} ({lo:+.1f}, {hi:+.1f}): {row}")
    print("\n## Lane-h at the measured 160-230 F games an hour (Stage A / Stage B / both)")
    a_games = 2 * G_A + 4 * 1200 + 2 * 1200          # finals, the +20M / +50M trajectory, home72 at the final
    b_games = 2 * G_B + 2 * 1200                      # the new pair's finals and home72
    for n, lab in ((a_games, "Stage A"), (b_games, "Stage B"), (a_games + b_games, "both")):
        print(f"  {lab}: {n:,} games = {n / 230:.0f}-{n / 160:.0f} lane-h")


if __name__ == "__main__":
    main()
