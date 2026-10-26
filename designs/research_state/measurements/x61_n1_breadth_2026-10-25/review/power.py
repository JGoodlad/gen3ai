"""X61 review check: Monte Carlo of the PREREG's registered two-look rule (section 5), including K0's sigma estimate.
CPU, deterministic (seed 6101). Normal-approximation intervals on the difference (the Newcombe hybrid is close at
these n). Model: per-seed true U_HF = base + N(0, sigma); each panel 600 games per seed (look 1 m = 1, look 2 m = 2);
K0 = 3 seeds x 200 games per panel at the same sigma; sigma_s = max(K0 estimate, 2 pp); look 2 also takes the look's
own-seed overdispersion. Classes as registered (GO: LB > 0 and d >= 4; KILL: UB < 2; BELOW-GO: UB < 4; CONTINUE on
UNDECIDED with d > 0). Run: python designs/research_state/measurements/x61_n1_breadth_2026-10-25/review/power.py"""
import numpy as np, sys
rng = np.random.default_rng(6101)
Z = 1.959964
def sim(delta, sig, p0=0.40, g=600, nsim=200000, k0=True, floor=0.02, m1=1, m2=2, kill_null=False):
    d = delta/100; s = sig/100
    # per-seed true U_HF per arm
    def arm_obs(mu, m):
        seeds = mu + rng.normal(0, s, size=(nsim, m))
        seeds = np.clip(seeds, 0.01, 0.99)
        # two panels, each g games; U_HF = mean of the two panel rates (panel effects common, ignore)
        wins = rng.binomial(g, np.repeat(seeds[:, :, None], 2, axis=2))
        rates = wins / g
        return rates  # nsim x m x 2
    # K0: 3 old-toy seeds, 200 games per panel
    if k0:
        seeds = p0 + rng.normal(0, s, size=(nsim, 3))
        r = rng.binomial(200, np.repeat(seeds[:, :, None], 2, axis=2)) / 200
        u = r.mean(axis=2)
        binv = (r * (1 - r) / 200).sum(axis=2) / 4
        v = u.var(axis=1, ddof=1) - binv.mean(axis=1)
        sk = np.sqrt(np.clip(v, 0, None))
    else:
        sk = np.zeros(nsim)
    sig1 = np.maximum(sk, floor)
    out = {}
    A = arm_obs(p0 + d, m2); B = arm_obs(p0, m2)
    def read(m, sig_s):
        a = A[:, :m, :].mean(axis=1); b = B[:, :m, :].mean(axis=1)  # pooled per panel
        n = g * m
        ua, ub = a.mean(axis=1), b.mean(axis=1)
        va = (a * (1 - a) / n).sum(axis=1) / 4; vb = (b * (1 - b) / n).sum(axis=1) / 4
        dh = ua - ub
        phi = 1 + 2 * sig_s**2 / (va + vb) / m
        hw = Z * np.sqrt(va + vb) * np.sqrt(phi)
        return dh * 100, (dh - hw) * 100, (dh + hw) * 100, A[:, :m, :], B[:, :m, :]
    def classify(dh, lb, ub):
        go = (lb > 0) & (dh >= 4); kill = ub < 2; below = (~go) & (~kill) & (ub < 4)
        und = ~(go | kill | below)
        return go, kill, below, und
    dh, lb, ub, _, _ = read(m1, sig1)
    go1, k1, b1, u1 = classify(dh, lb, ub)
    cont = u1 & (dh > 0)
    # look 2 overdispersion from look's own seeds (2 per arm): between-seed var pooled over arms
    ua_s = A.mean(axis=2); ub_s = B.mean(axis=2)
    bin_s = lambda X: ((X * (1 - X) / g).sum(axis=2) / 4).mean(axis=1)
    vb_ = (ua_s.var(axis=1, ddof=1) - bin_s(A) + ub_s.var(axis=1, ddof=1) - bin_s(B)) / 2
    s_od = np.sqrt(np.clip(vb_, 0, None))
    sig2 = np.maximum(sig1, s_od)
    dh2, lb2, ub2, _, _ = read(m2, sig2)
    go2, k2, b2, u2 = classify(dh2, lb2, ub2)
    final = {
        "GO": go1 | (cont & go2), "KILL": k1 | (cont & k2), "BELOW": b1 | (cont & b2),
        "UND": (u1 & ~cont) | (cont & u2)}
    return {k: round(v.mean(), 3) for k, v in final.items()}, {"look1 GO": round(go1.mean(), 3), "look1 KILL": round(k1.mean(), 3), "look1 BELOW": round(b1.mean(),3), "cont": round(cont.mean(), 3)}
for sig in (2, 3, 4):
    for delta in (0, 2, 4, 5, 8):
        f, l1 = sim(delta, sig, nsim=100000)
        print(f"sigma={sig} delta={delta:+} final {f} | {l1}")
print("--- validation: K0 off (sigma_s = floor exactly)")
for delta in (5, 8):
    print(delta, sim(delta, 2, nsim=100000, k0=False))
