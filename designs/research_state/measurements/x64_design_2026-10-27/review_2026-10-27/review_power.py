"""X64 adversarial review (2026-10-27): the registered slope rule re-simulated with a LEVEL seed effect.

The design's own Monte Carlo (§5, scratch, not committed) modelled seed noise on G = home72 - U only, and put half of
every change on each side. This script adds what that model leaves out:
  * a per-final LEVEL effect (overall strength, shared by every panel of one final; it cancels in G but not in U),
    at the X61 review's fresh-run floor (3 pp) and at 2 pp;
  * scenarios where the change is carried by home72 alone (the in-pool price at 1/18 of the pool dose), which the
    design's own predictions (§8: Delta_home -2.5, Delta_U +0.5) say is the expected case.
It then reports, for each scenario: the registered classes, the SHRINKS-U / SHRINKS-HOME routing, and two
alternative primaries the review proposes:
  * U-PRIMARY: Delta_U's 90 % two-sided interval (one-sided 95 %) LB > 0, level floor 3 pp (the design's own
    floor for a level, §4.3);
  * the same rule at endpoints only (4 / 0 / 4, the same 8 runs).
Plus a check that, with the level effect switched off and the change split half / half, the registered rule
reproduces the design's §5 table (validation of the model).

CPU only, numpy only, seeded; no game, no GPU, no file read. Run:
    python review_power.py > review_power.log
"""
from __future__ import annotations

import math

import numpy as np

Z90 = 1.6449
X = {"719": math.log10(719), "3.7k": math.log10(3719), "12.7k": math.log10(12719)}
R = X["12.7k"] - X["719"]
N_HOME, N_U = 1200, 800  # games per final: home72; each of H, F, P72
BASE_HOME = 0.47
BASE_U = (0.42, 0.45, 0.42)  # H, F, P72 at A-719: G = 0.47 - 0.43 = +4 pp (the design's K0 point, §8)
DRAWS = 20000


def _fit(x: np.ndarray, y: np.ndarray, v: np.ndarray, floor: float):
    """OLS slope over finals (axis 1), the design's §4.2 variance with a declared seed floor. Returns (delta, se)."""
    xm = x.mean()
    sxx = ((x - xm) ** 2).sum()
    b = ((x - xm) * (y - y.mean(axis=1, keepdims=True))).sum(axis=1) / sxx
    a = y.mean(axis=1) - b * xm
    resid = y - (a[:, None] + b[:, None] * x)
    n = x.size
    s2 = (resid ** 2).sum(axis=1) / (n - 2)
    vbar = v.mean(axis=1)
    sig2 = np.maximum(np.maximum(s2 - vbar, 0.0), floor ** 2)
    se = np.sqrt((vbar + sig2) / sxx)
    return b * R, se * R, vbar, sig2


def simulate(seeds: tuple[int, int, int], d_home: float, d_u: float, sd_g: float, sd_lvl: float,
             rng: np.random.Generator, base_home: float = BASE_HOME) -> dict:
    """One scenario. d_home / d_u are the true changes over the whole range (719 -> 12,719), in proportion."""
    xs = np.concatenate([np.full(m, X[k]) for k, m in zip(("719", "3.7k", "12.7k"), seeds)])
    t = (xs - X["719"]) / R  # 0 .. 1 along the range
    nf = xs.size
    lvl = rng.normal(0.0, sd_lvl, (DRAWS, nf))
    prem = rng.normal(0.0, sd_g, (DRAWS, nf))
    p_home_true = np.clip(base_home + d_home * t + lvl + prem, 0.01, 0.99)
    ph = rng.binomial(N_HOME, p_home_true) / N_HOME
    pu = []
    for base in BASE_U:
        p_true = np.clip(base + d_u * t + lvl, 0.01, 0.99)
        pu.append(rng.binomial(N_U, p_true) / N_U)
    u = (pu[0] + pu[1] + pu[2]) / 3.0
    g = ph - u
    v_h = ph * (1 - ph) / N_HOME
    v_u = sum(p * (1 - p) / N_U for p in pu) / 9.0
    v_g = v_h + v_u
    dg, seg, vbar_g, sig2_g = _fit(xs, g, v_g, 0.02)
    du, seu, _, _ = _fit(xs, u, v_u, 0.03)
    lb, ub = dg - Z90 * seg, dg + Z90 * seg
    m719 = seeds[0]
    if m719 > 0:
        g719 = g[:, :m719].mean(axis=1)
        g719_lb = g719 - Z90 * np.sqrt((vbar_g + sig2_g) / m719)
    else:
        g719_lb = np.full(DRAWS, np.nan)
    shrinks = ub < 0
    s_u = shrinks & (du > 0)
    s_home = shrinks & ~(du > 0)
    grows = ~shrinks & (lb > 0)
    rest = ~shrinks & ~grows
    flat = rest & (lb > -0.02) & (g719_lb > 0)
    uninf = rest & (lb > -0.02) & ~(g719_lb > 0)
    undecided = rest & ~flat & ~uninf
    u_gain = (du - Z90 * seu) > 0
    return {
        "SHRINKS": shrinks.mean(), "S-U": s_u.mean(), "S-HOME": s_home.mean(),
        "S-U|S": (s_u.sum() / shrinks.sum()) if shrinks.sum() else float("nan"),
        "GROWS": grows.mean(), "FLAT": flat.mean(), "UNINF": uninf.mean(), "UNDEC": undecided.mean(),
        "U-GAIN": u_gain.mean(), "hw_G": float(np.median(Z90 * seg)), "hw_U": float(np.median(Z90 * seu)),
    }


def main() -> None:
    rng = np.random.default_rng(20261027)
    print("X64 review power check. Draws per cell:", DRAWS, " R = log10(12719/719) =", round(R, 4))
    print("Games per final: home72", N_HOME, ", H / F / P72", N_U, "each. A-719 truth: home 0.47, U 0.43 (G +4 pp).")
    print()

    print("(1) VALIDATION: the design's own model (no level effect, half of the change on each side), 3/2/3.")
    print("    Design §5 reads P(SHRINKS) 0.34 at -3 pp and 0.52 at -4 pp (sigma_g 2); 0.040 / 0.20 FLAT at 0.")
    for g719, bh in (("+4 (the design's §8 point at 100M)", BASE_HOME), ("+5 (the design's §5 model)", 0.48)):
        print(f"    A-719 G {g719}:")
        for dg in (0.0, -0.03, -0.04, -0.06):
            r = simulate((3, 2, 3), dg / 2, -dg / 2, 0.02, 0.0, rng, base_home=bh)
            print(f"      true Delta_G {dg*100:+.0f} pp: SHRINKS {r['SHRINKS']:.3f}  FLAT {r['FLAT']:.3f}  "
                  f"UNDEC {r['UNDEC']:.3f}  half-width(G) {r['hw_G']*100:.1f} pp")
    print()

    scen = [
        ("null", 0.0, 0.0),
        ("pure starvation: home -3, U 0", -0.03, 0.0),
        ("starvation + slower broad arm: home -4, U -1", -0.04, -0.01),
        ("the design's §8 GUESS: home -2.5, U +0.5", -0.025, 0.005),
        ("the design's §5 model at -3: home -1.5, U +1.5", -0.015, 0.015),
        ("breadth buys U, with price: home -3, U +3", -0.03, 0.03),
        ("breadth buys U, no price: home 0, U +3", 0.0, 0.03),
        ("a large U gain: home -2, U +5", -0.02, 0.05),
    ]
    for sd_lvl in (0.03, 0.02):
        for sd_g in (0.02,):
            print(f"(2) The registered rule, 3/2/3, sigma_g {sd_g*100:.0f} pp, LEVEL sigma {sd_lvl*100:.0f} pp. "
                  "U-GAIN = the alternative U-primary (Delta_U 90 % LB > 0, 3 pp floor).")
            hdr = (f"    {'scenario':52s} {'SHRINKS':>7s} {'S-U':>6s} {'S-HOME':>6s} {'S-U|S':>6s} {'FLAT':>6s} "
                   f"{'UNDEC':>6s} {'GROWS':>6s} | {'U-GAIN':>6s} hw_U")
            print(hdr)
            for name, dh, du in scen:
                r = simulate((3, 2, 3), dh, du, sd_g, sd_lvl, rng)
                print(f"    {name:52s} {r['SHRINKS']:7.3f} {r['S-U']:6.3f} {r['S-HOME']:6.3f} {r['S-U|S']:6.3f} "
                      f"{r['FLAT']:6.3f} {r['UNDEC']:6.3f} {r['GROWS']:6.3f} | {r['U-GAIN']:6.3f} "
                      f"{r['hw_U']*100:.1f}")
            print()

    print("(3) The U-primary at other shapes (LEVEL sigma 3 pp, sigma_g 2 pp): P(U-GAIN), and its median half-width.")
    for seeds in ((3, 2, 3), (4, 0, 4), (6, 0, 6)):
        cells = []
        for du in (0.0, 0.01, 0.02, 0.03, 0.05):
            r = simulate(seeds, -0.03, du, 0.02, 0.03, rng)
            cells.append(f"U {du*100:+.0f}: {r['U-GAIN']:.3f}")
        r0 = simulate(seeds, 0.0, 0.0, 0.02, 0.03, rng)
        print(f"    {seeds}: " + "  ".join(cells) + f"   (hw_U {r0['hw_U']*100:.1f} pp)")
    print()

    print("(4) K0 (design §4.6): P(PASS) = P(G_hat >= +2.0) x P(Gctx_hat >= +1.0), pooled SEs 1.4 / 1.7 pp")
    print("    (independence assumed; the design's own SEs).")
    phi = lambda z: 0.5 * (1 + math.erf(z / math.sqrt(2)))  # noqa: E731
    for g, gc in ((0.0, 0.0), (1.0, 0.5), (2.0, 1.0), (3.0, 1.5), (4.0, 2.0), (6.0, 3.0)):
        p = phi((g - 2.0) / 1.4) * phi((gc - 1.0) / 1.7)
        print(f"    true G {g:+.1f}, G_ctx {gc:+.1f}: P(K0 PASS) {p:.3f}")


if __name__ == "__main__":
    main()
