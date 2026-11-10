"""Where did X68's fine-tune lose the move leg? Per panel: the log-likelihood change from the NON-true candidates
a unit (X68-FT and P* against the prior), the moves that carry it, the split by revealed count, and the
candidates above 0.9 a unit. numpy only.   peek.py <panel>"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).parent))
from analyze import load, lsig, sig, units
panel = sys.argv[1]
zs = {}
for label in ("pstar", "x68ft"):
    z, t = load(label, panel); u = units(z, t); ok = u["ok"]
    zs[label] = (z, u, ok)
z, u, ok = zs["x68ft"]
truth, cand = u["truth"][ok], u["cand"][ok]
post, prior = z["post"][ok].astype(np.float64), z["prior"][ok].astype(np.float64)
pp = zs["pstar"][0]["post"][ok].astype(np.float64)
sp = z["species"][ok]; r = u["r"][ok]
cost = np.where(cand & ~truth, lsig(-post) - lsig(-prior), 0)          # ll change on non-truths (negative = harm)
print("units", ok.sum(), "ll change from non-truths, mean a unit:", cost.sum(-1).mean(), "pstar:", np.where(cand & ~truth, lsig(-pp) - lsig(-prior), 0).sum(-1).mean())
bym = cost.sum(0); order = np.argsort(bym)[:15]
tot = cost.sum()
print("top moves by harm (move id, share of the harm, mean prior p on its candidates, mean post p x68ft, mean post p pstar, n cand, truth rate):")
for m in order:
    c = cand[:, m] & ~truth[:, m]
    call = cand[:, m]
    print(f"  move {m:3d} share {bym[m]/tot:.3f} prior p {sig(prior[c, m]).mean():.3f} x68ft p {sig(post[c, m]).mean():.3f} pstar p {sig(pp[c, m]).mean():.3f} n {c.sum()} truth rate {truth[call, m].mean():.3f}")
# by revealed count
for k in range(4):
    m = r == k
    print(f"r={k} units {m.sum()} mean non-truth ll change x68ft {cost[m].sum(-1).mean():+.2f}  mass x68ft {np.where(cand, sig(post), 0)[m].sum(-1).mean():.2f} pstar {np.where(cand, sig(pp), 0)[m].sum(-1).mean():.2f} prior {np.where(cand, sig(prior), 0)[m].sum(-1).mean():.2f}")
# how many non-truth candidates per unit above 0.9
print("non-truth candidates with p>0.9 a unit: x68ft", ((sig(post) > 0.9) & cand & ~truth).sum(-1).mean(), "pstar", ((sig(pp) > 0.9) & cand & ~truth).sum(-1).mean(), "prior", ((sig(prior) > 0.9) & cand & ~truth).sum(-1).mean())
print("truth candidates with p>0.9 a unit: x68ft", ((sig(post) > 0.9) & truth).sum(-1).mean(), "pstar", ((sig(pp) > 0.9) & truth).sum(-1).mean(), "prior", ((sig(prior) > 0.9) & truth).sum(-1).mean())
