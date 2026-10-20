import numpy as np, collections

from agents.training import cycle_monitor as CM
from main import cycle_monitor as M
from utils.paths import main_models_dir
edges, meta = M.ladder_edges(main_models_dir() / "rb_x26_s2601")
newest = 268000171
sub = [e for e in edges if e[1] <= newest and e[3] <= newest]
mx = CM.build_matrix(sub)
n, m, v, y, w = CM.edge_stats(mx.counts)
r = CM.wls_fit(mx.n, mx.ei, mx.ej, y, w)
pred = 1/(1+np.exp(-(r[mx.ei]-r[mx.ej])))
# the observed per-pair shape: share of WL/DD (cat 2) etc pooled
pooled = mx.counts.sum(0)/mx.counts.sum(); print("pooled pent", np.round(pooled,3))
rule = CM.MonitorRule(n_boot=400, posterior_draws=5)
rng = np.random.default_rng(123)
cnt = collections.Counter(); ps=[]
for t in range(30):
    cs = []
    for e in range(len(n)):
        p = pred[e]
        # mirrored pair of two games with a shared team-pair effect: per-pair logit shift ~ N(0, 0.5)
        sh = rng.normal(0, 0.5, int(n[e]))
        pp = 1/(1+np.exp(-(np.log(p/(1-p)) + sh)))
        # in a mirrored pair the teams swap: game 2 sees the opposite shift
        pq = 1/(1+np.exp(-(np.log(p/(1-p)) - sh)))
        wins = (rng.random(int(n[e])) < pp).astype(int) + (rng.random(int(n[e])) < pq).astype(int)
        cs.append(np.bincount(2*wins, minlength=5).astype(float))
    syn = CM.Matrix(keys=mx.keys, steps=mx.steps, ei=mx.ei, ej=mx.ej, counts=np.array(cs))
    res = CM.signals(syn, f"snap:{newest}", rule, seed=t)
    cnt[res.signals['hodge']] += 1; ps.append(res.detail['hodge']['p_boot'])
    cnt['tri_'+res.signals['triangles']] += 1
print(cnt); print(np.round(sorted(ps),3))
