"""ExIt-0 two-look power: D per deal, discordance d, true effect delta. Normal approx of the per-deal mean, looks nested."""
import numpy as np
rng = np.random.default_rng(20261105)
N1, N2 = 2700, 7200
def run(delta, d, z1go, z2go, nullbar, R=200000):
    var = d - delta**2
    m1 = delta + rng.standard_normal(R) * np.sqrt(var / N1)
    m_add = delta + rng.standard_normal(R) * np.sqrt(var / (N2 - N1))
    m2 = (m1 * N1 + m_add * (N2 - N1)) / N2
    se1, se2 = np.sqrt(var / N1), np.sqrt(var / N2)
    go1 = m1 - z1go * se1 > 0
    null1 = (m1 + 1.96 * se1 < nullbar) & ~go1
    cont = ~go1 & ~null1
    go2 = cont & (m2 - z2go * se2 > 0)
    null2 = cont & ~go2 & (m2 + 1.96 * se2 < nullbar)
    return dict(GO=(go1 | go2).mean(), GO_look1=go1.mean(), NULL=(null1 | null2).mean(), cont=cont.mean(),
                UNDEC=(cont & ~go2 & ~null2).mean())
for d in (0.45, 0.50):
    for z1, z2 in ((2.576, 2.0), (2.807, 1.98)):
        print(f"d={d} z1={z1} z2={z2}")
        for delta in (0.0, 0.01, 0.015, 0.02, 0.023, 0.03, 0.046):
            r = run(delta, d, z1, z2, 0.02)
            print(f"  delta {delta*100:4.1f} pp: " + " ".join(f"{k} {v:.3f}" for k, v in r.items()))
