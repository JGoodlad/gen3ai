"""M5-M8: group-sequential NON-INFERIORITY on the mirrored head-to-head, estimated variance, looks at n = 3, 5, 8 seeds/arm.
Model (pp of H2H win rate, X5 vs blob): h_ij = Delta + a_i - b_j + g_ij + e_ij; a,b ~ N(0, sig^2) run strength,
g ~ N(0, sig_int^2) seed x seed interaction, e ~ N(0, sig_e^2) game noise of one cell (1,000 mirrored pairs ~ 1.1 pp).
CROSS design: every X5 seed plays every blob seed. Delta_hat = grand mean; V_hat = (s_R^2 + s_C^2)/n (row / column means),
df = 2(n-1). PAIRED design (comparison): d_i = h_ii, one-sample t, df n-1.
Boundaries: O'Brien-Fleming (classic C/sqrt(t_k)) on z for one-sided alpha, information fractions n_k/8, mapped to the t scale
at the same nominal p (Pocock 1977 / Jennison & Turnbull 1991 'significance level' approach); then VERIFIED by simulation and,
if needed, the constant C re-calibrated by simulation (reported)."""
import numpy as np
from scipy import stats, optimize
LOOKS = (3, 5, 8); NMAX = 8
ALPHA = 0.05
def obf_z(alpha, looks=LOOKS, nmax=NMAX):
    t = np.array(looks) / nmax
    cov = np.sqrt(np.minimum.outer(t, t) / np.maximum.outer(t, t))
    mvn = stats.multivariate_normal(mean=np.zeros(len(t)), cov=cov, seed=0, abseps=1e-7, releps=1e-7)  # fixed QMC seed: the cdf is randomised otherwise
    f = lambda C: (1 - mvn.cdf(C / np.sqrt(t))) - alpha
    C = optimize.brentq(f, 1.0, 5.0, xtol=1e-10)
    return C, C / np.sqrt(t)
def t_bounds(zb, dfs): return np.array([stats.t.ppf(stats.norm.cdf(z), df) for z, df in zip(zb, dfs)])
rng = np.random.default_rng(20261003)
def sim(design, sig, delta, Delta, tb, fut=None, sig_int=0.0, sig_e=1.1, n=100_000):
    a = rng.normal(0, sig, (n, NMAX)); b = rng.normal(0, sig, (n, NMAX))
    h = Delta + a[:, :, None] - b[:, None, :] + rng.normal(0, np.sqrt(sig_int**2 + sig_e**2), (n, NMAX, NMAX))
    res = np.zeros(n, int); stop_look = np.full(n, len(LOOKS) - 1); done = np.zeros(n, bool)
    est_at_stop = np.zeros(n)
    for li, m in enumerate(LOOKS):
        if design == "cross":
            hm = h[:, :m, :m]; R = hm.mean(2); Cc = hm.mean(1); d = hm.mean((1, 2))
            se = np.sqrt((R.var(1, ddof=1) + Cc.var(1, ddof=1)) / m)
        else:
            dd = np.diagonal(h, axis1=1, axis2=2)[:, :m]; d = dd.mean(1); se = dd.std(1, ddof=1) / np.sqrt(m)
        tstat = (d + delta) / se
        ni = (tstat >= tb[li] + 1e-9) & ~done                     # rule 8: within 1e-9 of the boundary = not crossed
        fu = np.zeros(n, bool)
        if fut is not None and li < len(LOOKS) - 1:
            fu = (d <= -fut * delta) & ~done & ~ni                  # non-binding futility: point estimate at/beyond -delta*fut
        res[ni] = 1; res[fu] = 2
        newly = ni | fu
        stop_look[newly] = li; est_at_stop[newly] = d[newly]
        done |= newly
    return res, stop_look
def runs_cost(stop_look, x5_h=2.6, blob_h=2.5):
    pairs = np.array(LOOKS)[stop_look]
    return (pairs * (x5_h + blob_h)).mean()
if __name__ == "__main__":
    import json
    out = {}
    C, zb = obf_z(ALPHA)
    print(f"OBF z boundaries (one-sided {ALPHA}, t = 3/8, 5/8, 1): C={C:.4f}", np.round(zb, 4))
    for design in ("cross", "paired"):
        dfs = [2 * (m - 1) for m in LOOKS] if design == "cross" else [m - 1 for m in LOOKS]
        tb = t_bounds(zb, dfs)
        print(f"\n=== {design}: t boundaries on df {dfs}: {np.round(tb, 3)}")
        # verify type I at Delta=-delta, no futility (conservative) and with futility
        for delta in (3.5,):
            for sig in (2.5, 3.43, 4.8):
                for si in (0.0, 1.5):
                    r0, _ = sim(design, sig, delta, -delta, tb, sig_int=si)
                    print(f"  delta {delta} sig {sig} sig_int {si}: typeI(no fut) {np.mean(r0==1):.4f}")
        out[design] = {"t_bounds": tb.tolist(), "dfs": dfs}
    # calibrate C by simulation for the cross so typeI = 0.05 at sig 3.43 (pivotal check across sigma afterwards)
    print("\n=== cross: simulation-calibrated boundaries")
    dfs = [2 * (m - 1) for m in LOOKS]
    def ti(scale):
        tb = t_bounds(zb * scale, dfs); r, _ = sim("cross", 3.43, 3.5, -3.5, tb, n=200_000); return np.mean(r == 1)
    print("  scale 1.0 typeI", ti(1.0))
    json.dump(out, open("../out/gs_bounds.json", "w"), indent=1)
