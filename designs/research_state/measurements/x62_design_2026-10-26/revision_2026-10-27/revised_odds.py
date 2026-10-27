"""X62 revision (``designs/endstate/design_x62_team_recombination.md`` §0, §5, §7; RESPONSE B1, M2, M4): the revised
odds. CPU, numpy, deterministic (seeded), < 1 min. Plays nothing.

    CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 5-7,13-15 python .../revision_2026-10-27/revised_odds.py > revised_odds.log

A. ADOPT odds under the revised read (§5), integrated over the GUESS of each branch, 5 v 5 seeds, every panel with the
   same between-seed SD sigma_s:
   - GAIN: F by the PR rule with AM-5 (``../review_2026-10-27/power_checks.py`` A): 3,480 F games a final, the
     normal half-width widened by sqrt(max(1, phi-hat)), LB > 0.
   - PRICE (B1; the exchange rate r = 1, the orchestrator's default pending the owner): net = d_home72 + r * d_U, with
     d_U = mean(d_F, d_H, d_P72), all against Kakuna, 400 games a final on home72, H and P72. WITHIN iff the point
     estimate of net >= 0; EXCEEDED iff the Welch 95 % UB of net over the per-seed values < 0.
   - FLOOR (goal (a)): the arm's pooled home72 against Kakuna (2,000 games), LB > 0.50, widened like F.
   - ADOPT-CANDIDATE = GAIN and PRICE WITHIN and FLOOR; also GAIN and not EXCEEDED and FLOOR.
   The truths (GUESS; MY JUDGEMENT): d_F ~ the branch's GUESS; d_H = 0.5 d_F + N(0, 1 pp); d_P72 = 0.25 d_F + N(0,
   1 pp); d_home72 ~ N(-1.5, 80 % (-4.0, +1.0)) independent of d_F (D1's in-pool cost, re-derived under H-CTX). Bases:
   X51's P* vs Kakuna (home72 0.598, F 0.530 -> 0.49 as the power scripts use, H 0.497, P72 0.433).
B. The same at a fixed truth (d_F +1.5, d_home72 -1.6, d_H +0.75, d_P72 +0.4) and at the null (all 0).
C. G0 routing (M4): the registered classes by the true recovery fraction rho, and the review option this revision did NOT
   take (SHELVE only when R / P's lower bound >= 0.5): it almost never shelves. R / P's bound is a delta-method
   normal approximation of the read's percentile bootstrap (rc72b 1,200 / home72 800 / F 800 games).
D. Per resource-day = (GPU-h + lane-h) / 24 (the strategy doc's unit) at 160-230 games a lane-h, 20.5 GPU-h: X62's 5
   finals x (3,480 F + 400 each of home72, H, P72, rc72b) + the dose line (400 F at each arm's and C0's 10M checkpoint)
   + C0's 5 finals, either FRESH (all re-played at the read's pin) or with C0's F BANKED at that pin by PR-1' (30,000
   games, C0 = 2 x an arm: 3,000 F a seed, topped up to 3,480).
"""
import numpy as np
from scipy import stats

RNG = np.random.default_rng(20261027)
K = 5
N_F, N_X = 3480, 400
BASE = {"F": 0.49, "H": 0.497, "P72": 0.433, "home": 0.598}


def seed_rates(base, delta, sig, n, draws):
    true = np.clip(base + delta[:, None] + RNG.normal(0, sig, (draws, K)), 0.01, 0.99)
    return RNG.binomial(n, true) / n


def disp(x, n):
    e = (x * (1 - x) / n).mean(axis=1)
    return x.var(axis=1, ddof=1) / np.maximum(e, 1e-12)


def read(dF, dH, dP, dHome, sig, draws, r=1.0):
    z = np.zeros(draws)
    on = {k: seed_rates(BASE[k], d, sig, N_F if k == "F" else N_X, draws)
          for k, d in (("F", dF), ("H", dH), ("P72", dP), ("home", dHome))}
    off = {k: seed_rates(BASE[k], z, sig, N_F if k == "F" else N_X, draws) for k in BASE}
    phi = (disp(on["F"], N_F) + disp(off["F"], N_F)) / 2
    d = on["F"].mean(1) - off["F"].mean(1)
    se = np.sqrt((on["F"] * (1 - on["F"])).mean(1) / (K * N_F) + (off["F"] * (1 - off["F"])).mean(1) / (K * N_F))
    gain = d - 1.96 * se * np.sqrt(np.maximum(1.0, phi)) > 0
    net_on = on["home"] + r * (on["F"] + on["H"] + on["P72"]) / 3
    net_off = off["home"] + r * (off["F"] + off["H"] + off["P72"]) / 3
    v1, v2 = net_on.var(1, ddof=1) / K, net_off.var(1, ddof=1) / K
    df = (v1 + v2) ** 2 / (v1 ** 2 / (K - 1) + v2 ** 2 / (K - 1) + 1e-30)
    net = net_on.mean(1) - net_off.mean(1)
    within = net >= 0
    exceeded = net + stats.t.ppf(0.975, df) * np.sqrt(v1 + v2) < 0
    h = on["home"].mean(1)
    phih = disp(on["home"], N_X)
    floor = h - 1.96 * np.sqrt(h * (1 - h) / (K * N_X)) * np.sqrt(np.maximum(1.0, phih)) > 0.50
    return gain, within, exceeded, floor


def guess(mean, lo, hi, draws):
    return RNG.normal(mean, (hi - lo) / 2.563, draws) / 100


def part_a(draws=200000):
    print("A. ADOPT odds over each branch's GUESS (PR rule + AM-5 on F; PRICE r = 1; FLOOR home72 LB > 0.50)")
    for lab, (m, lo, hi) in (("SUPPORTED, run branch: d_F +1.5 (-1.5, +4.5)", (1.5, -1.5, 4.5)),
                             ("UNDECIDED / soft KILLED: d_F +1.0 (-1.5, +3.5)", (1.0, -1.5, 3.5))):
        for sig in (0.0, 0.5, 1.0):
            dF = guess(m, lo, hi, draws)
            dH = 0.5 * dF + RNG.normal(0, 0.01, draws)
            dP = 0.25 * dF + RNG.normal(0, 0.01, draws)
            dHome = guess(-1.5, -4.0, 1.0, draws)
            g, w, x, f = read(dF, dH, dP, dHome, sig / 100, draws)
            print(f"  {lab}, sigma_s {sig} pp: P(GAIN) {g.mean():.3f}; P(PRICE WITHIN) {w.mean():.3f}; "
                  f"P(EXCEEDED) {x.mean():.3f}; P(FLOOR) {f.mean():.3f}; P(GAIN & WITHIN & FLOOR) {(g & w & f).mean():.3f}; "
                  f"P(GAIN & not EXCEEDED & FLOOR) {(g & ~x & f).mean():.3f}")


def part_a_rate(draws=200000):
    print("A2. the exchange rate r (the owner's to change), SUPPORTED run branch, sigma_s 0.5 pp")
    for r in (0.5, 1.0, 2.0, 3.0):
        dF = guess(1.5, -1.5, 4.5, draws)
        dH = 0.5 * dF + RNG.normal(0, 0.01, draws)
        dP = 0.25 * dF + RNG.normal(0, 0.01, draws)
        dHome = guess(-1.5, -4.0, 1.0, draws)
        g, w, x, f = read(dF, dH, dP, dHome, 0.005, draws, r=r)
        print(f"  r {r}: P(GAIN & WITHIN & FLOOR) {(g & w & f).mean():.3f}; P(GAIN & not EXCEEDED & FLOOR) "
              f"{(g & ~x & f).mean():.3f}")


def part_b(draws=200000):
    print("B. fixed truths, sigma_s 0.5 pp")
    for lab, t in (("d_F +1.5, d_H +0.75, d_P72 +0.4, d_home72 -1.6", (1.5, 0.75, 0.4, -1.6)),
                   ("d_F +1.5, d_H +0.75, d_P72 +0.4, d_home72 -3.0", (1.5, 0.75, 0.4, -3.0)),
                   ("null: all 0", (0, 0, 0, 0))):
        a = [np.full(draws, v / 100) for v in t]
        g, w, x, f = read(*a, 0.005, draws)
        print(f"  {lab}: P(GAIN) {g.mean():.3f}; WITHIN {w.mean():.3f}; EXCEEDED {x.mean():.3f}; FLOOR {f.mean():.3f}; "
              f"ADOPT-CANDIDATE {(g & w & f).mean():.3f}")


def part_c(draws=200000):
    print("C. G0 routing (M4): the registered classes; BUILD = SUPPORTED or UNDECIDED (the revised tree); the R/P-bound option, delta-method")
    h, fF = 0.598, 0.530
    for rho in (0.0, 0.25, 0.4, 0.6, 0.75, 1.0):
        xr = RNG.binomial(1200, fF + rho * (h - fF), draws) / 1200
        xh = RNG.binomial(800, h, draws) / 800
        xf = RNG.binomial(800, fF, draws) / 800
        P, R = xh - xf, xr - xf
        vr, vh, vf = xr * (1 - xr) / 1200, xh * (1 - xh) / 800, xf * (1 - xf) / 800
        se_p = np.sqrt(vh + vf)
        inc = P - 1.96 * se_p <= 0
        with np.errstate(divide="ignore", invalid="ignore"):  # P <= 0 is INCONCLUSIVE, never SHELVE
            q = R / np.where(P > 0, P, np.nan)
            # var of R/P by the delta method: R = r - f and P = h - f share f
            var_q = (vr + vf) / P ** 2 + R ** 2 * (vh + vf) / P ** 4 - 2 * R * vf / P ** 3
            lb = np.nan_to_num(q - 1.96 * np.sqrt(np.maximum(var_q, 0)), nan=-np.inf)
        kil = ~inc & (R >= 0.75 * P)
        shelve = kil & (lb >= 0.5)
        sup = ~inc & (R <= 0.25 * P)
        print(f"  rho {rho:.2f}: SUPPORTED {sup.mean():.3f}, KILLED {kil.mean():.3f}, SHELVE if routed by R/P LB >= 0.5 (the review option NOT taken) {shelve.mean():.3f}, "
              f"BUILD by the revised tree (SUPPORTED or UNDECIDED) {(~inc & ~kil).mean():.3f}, INCONCLUSIVE {inc.mean():.3f}")


def part_d(odds):
    print("D. per resource-day (GPU 20.5 h; lane 160-230 games / h)")
    per_final = 3480 + 4 * 400
    x62, dose = 5 * per_final, 10 * 400
    for lab, c0 in (("C0 FRESH", 5 * per_final), ("C0 F BANKED by PR-1'", 5 * (3480 - 3000) + 5 * 4 * 400)):
        games = x62 + dose + c0
        lo_h, hi_h = games / 230, games / 160
        rd = ((20.5 + lo_h) / 24, (20.5 + hi_h) / 24)
        cells = "; ".join(f"{k} {v:.3f}: {v / rd[1]:.3f}-{v / rd[0]:.3f}" for k, v in odds)
        print(f"  {lab}: {games:,} games, {lo_h:.0f}-{hi_h:.0f} lane-h, {rd[0]:.2f}-{rd[1]:.2f} resource-days; {cells}")


if __name__ == "__main__":
    part_a()
    part_a_rate()
    part_b()
    part_c()
    part_d((("P(GAIN) sigma_s 0", 0.558), ("sigma_s 1", 0.487), ("P(ADOPT-CANDIDATE) sigma_s 0", 0.305),
            ("sigma_s 1", 0.276)))
