"""X66 ExIt-0H two-look power (PREREG §8): D per deal, discordance d, true effect delta; a normal approximation of the
per-deal mean, the looks nested. Look 1 = S'-HONEST's banked deals at unit seeds 700-744 (2,700), look 2 = all of
700-784 (5,100). d: S'-HONEST's measured 0.350 (a 10-turn shared prefix: a floor for a student that departs from
turn 1) and X65's planning range 0.45-0.50. X65's power.py with its N and d changed."""
import numpy as np

rng = np.random.default_rng(20261109)
N1, N2 = 2700, 5100


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


for d in (0.35, 0.45, 0.50):
    print(f"d={d} z1=2.576 z2=2.0 null bar +2.0 pp")
    for delta in (0.0, 0.005, 0.01, 0.015, 0.02, 0.03):
        r = run(delta, d, 2.576, 2.0, 0.02)
        print(f"  delta {delta*100:4.1f} pp: " + " ".join(f"{k} {v:.3f}" for k, v in r.items()))
