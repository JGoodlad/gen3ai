"""Power of PR-1, the zero-GPU PRECISION READ (strategy_if_levers_null_2026-10-26.md §3.1), for the 2026-10-26 revision.

The design: the banked 20M-fork finals of the breadth family (D1, D3, N1-P) and the shared control C0, replayed on
forum72 (F) against Kakuna ckpt 34 at ONE playing commit, with new unit seeds. The estimand is the breadth-family
mean contrast  D = mean_b(F_b) - F_C0,  each arm's F the equal-weight mean of its seed finals. The interval is
normal, widened by AM-5's sqrt(max(1, phi_hat)) with phi_hat the mean of the four arms' seed-dispersion ratios
(main.x35.d1._arm_ratio's estimator). Classes (rule 8: a bound within 1e-9 of a threshold is UNDECIDED):
  BREADTH-GAIN  iff LB > 0
  KILL          iff UB < +1.5 pp           (LOSS iff UB < 0 is a sub-case, reported)
  UNDECIDED     otherwise
A binding futility-only interim at half the games: KILL-STOP iff the half-sample UB < +1.5 pp.

Truth: p_seed = BASE + delta_arm + sigma_s * z (z ~ N(0,1) per seed final). Deterministic (default_rng(2626)).
CPU only, a few seconds:
    python power_precision_read.py > power_precision_read.log
"""
import numpy as np

Z = 1.959963984540054
EPS = 1e-9
BASE = 0.49  # C0's own forum72 at D1's read: 342 / 700
DRAWS = 200_000
KILL_PP = 0.015


def simulate(rng, deltas, seeds, n_b, n_c, sigma_s, interim=True):
    """deltas: per breadth arm true effect; seeds: per arm seed count, C0 last."""
    arms = list(deltas) + [0.0]
    ns = [n_b] * len(deltas) + [n_c]
    k_draw = DRAWS
    half_x, full_x, half_n, full_n = [], [], [], []
    for d, k, n in zip(arms, seeds, ns):
        n_seed = n // k
        n_half = n_seed // 2
        p = np.clip(BASE + d + sigma_s * rng.standard_normal((k_draw, k)), 0.01, 0.99)
        w_half = rng.binomial(n_half, p)
        w_rest = rng.binomial(n_seed - n_half, p)
        half_x.append(w_half)
        half_n.append(n_half)
        full_x.append(w_half + w_rest)
        full_n.append(n_seed)

    def read(xs, nseeds):
        rates, var_arm, ratios = [], [], []
        for w, n in zip(xs, nseeds):
            ps = w / n
            k = ps.shape[1]
            m = ps.mean(axis=1)
            e = (ps * (1 - ps) / n).mean(axis=1)
            rates.append(m)
            var_arm.append(e / k)
            ratios.append(((ps - m[:, None]) ** 2).sum(axis=1) / (k - 1) / np.maximum(e, 1e-12))
        nb = len(rates) - 1
        d = sum(rates[:nb]) / nb - rates[nb]
        var = sum(var_arm[:nb]) / nb**2 + var_arm[nb]
        phi = sum(ratios) / len(ratios)
        h = Z * np.sqrt(var) * np.sqrt(np.maximum(1.0, phi))
        return d, d - h, d + h, phi

    dh, lbh, ubh, _ = read(half_x, half_n)
    d, lb, ub, phi = read(full_x, full_n)
    stop = (ubh < KILL_PP - EPS) if interim else np.zeros(k_draw, bool)
    near = (np.abs(lb) <= EPS) | (np.abs(ub - KILL_PP) <= EPS)
    gain = ~stop & ~near & (lb > 0)
    kill = stop | (~near & (ub < KILL_PP))
    loss = ~stop & ~near & (ub < 0)
    games_full = sum(n * k for n, k in zip(full_n, seeds))
    games_half = sum(n * k for n, k in zip(half_n, seeds))
    eg = np.where(stop, games_half, games_full).mean()
    hw = (ub - lb)[~stop].mean() / 2 if (~stop).any() else float("nan")
    return dict(gain=gain.mean(), kill=kill.mean(), loss=loss.mean(), und=1 - gain.mean() - kill.mean(),
                stop=stop.mean(), eg=eg, full=games_full, hw=100 * hw, phi=np.median(phi),
                bias_gain=100 * (d[gain].mean() - np.mean(deltas)) if gain.any() else float("nan"))


def fmt(r):
    return (f"GAIN {r['gain']:.3f} KILL {r['kill']:.3f} (LOSS {r['loss']:.3f}, interim stop {r['stop']:.3f}) "
            f"UNDECIDED {r['und']:.3f} | E[games] {r['eg']:,.0f} of {r['full']:,} | mean half-width {r['hw']:.2f} pp "
            f"| median phi {r['phi']:.2f} | bias|GAIN {r['bias_gain']:+.2f} pp")


def main():
    rng = np.random.default_rng(2626)
    print(f"{DRAWS} draws a cell; base F {BASE}; KILL bar +{100 * KILL_PP:.1f} pp; effects in pp")
    designs = {
        "POOLED 3 breadth arms x 8,000 + C0 14,000 (5/5/5/5 seeds)": ((5, 5, 5, 5), 8000, 14000),
        "POOLED, D3 and N1-P at 3 seeds (5/3/3/5)": ((5, 3, 3, 5), 8040, 14000),
        "POOLED, smaller: 3 x 6,000 + C0 10,000 (5/5/5/5)": ((5, 5, 5, 5), 6000, 10000),
    }
    scen = {
        "null 0/0/0": (0.0, 0.0, 0.0),
        "+0.5 each": (0.005,) * 3,
        "+1.0 each": (0.01,) * 3,
        "+1.5 each": (0.015,) * 3,
        "+2.0 each": (0.02,) * 3,
        "hetero D1 +2 / D3 +1 / N1-P 0 (mean +1.0)": (0.02, 0.01, 0.0),
        "hetero D1 +3 / D3 +1.5 / N1-P 0 (mean +1.5)": (0.03, 0.015, 0.0),
    }
    for name, (seeds, nb, nc) in designs.items():
        for sigma in (0.0, 0.005, 0.01):
            print(f"\n## {name}; between-seed sigma {100 * sigma:.1f} pp")
            for sname, deltas in scen.items():
                r = simulate(rng, deltas, seeds, nb, nc, sigma)
                print(f"  {sname:45s} {fmt(r)}")
    print("\n## The per-arm alternative: ONE breadth arm vs C0 at 17,400 F games each (5/5 seeds), no interim")
    for sigma in (0.0, 0.005, 0.01):
        for dl in (0.0, 0.01, 0.015):
            r = simulate(rng, (dl,), (5, 5), 17400, 17400, sigma, interim=False)
            print(f"  sigma {100 * sigma:.1f} delta {100 * dl:+.1f}: {fmt(r)}")


if __name__ == "__main__":
    main()
