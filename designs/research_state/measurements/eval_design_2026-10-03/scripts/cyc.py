"""Measure non-transitivity on banked dense ladders: BT (logistic) fit on pair counts, residual deviance vs df,
Hodge-style cyclic share (least-squares on logit edge flows, weighted), and pool win-rate spread."""
import json, sys, math, collections, glob, os
import numpy as np
from scipy.optimize import minimize
from scipy.stats import chi2

def load(run, drop_eval=True):
    p = f"/home/goodlad/dev/gen3ai/models/{run}/snapshot_ladder/games.jsonl"
    lad = json.load(open(os.path.dirname(p)+"/ladder.json"))
    nodes = [k for k in lad["ratings"] if k.isdigit()]
    agg = collections.defaultdict(lambda: [0,0])
    for l in open(p):
        r = json.loads(l)
        if drop_eval and r.get("source") == "eval_cycle": continue
        a,b = str(r["a"]), str(r["b"])
        if a not in nodes or b not in nodes: continue
        if a > b: a,b = b,a; w = r["games"]-r["wins_a"]
        else: w = r["wins_a"]
        agg[(a,b)][0]+=w; agg[(a,b)][1]+=r["games"]
    return sorted(nodes, key=int), agg

def bt(nodes, agg):
    idx = {n:i for i,n in enumerate(nodes)}; K=len(nodes)
    E = [(idx[a],idx[b],w,g) for (a,b),(w,g) in agg.items()]
    def nll(th):
        th = np.concatenate([[0],th]); s=0; gr=np.zeros(K)
        for i,j,w,g in E:
            p = 1/(1+math.exp(-(th[i]-th[j])))
            s -= w*math.log(p)+(g-w)*math.log(1-p)
            gr[i] -= w-g*p; gr[j] += w-g*p
        return s, gr[1:]
    res = minimize(nll, np.zeros(K-1), jac=True, method="L-BFGS-B")
    th = np.concatenate([[0],res.x])
    dev=0; pear=0
    for i,j,w,g in E:
        p = 1/(1+math.exp(-(th[i]-th[j]))); 
        for o,e in ((w,g*p),(g-w,g*(1-p))):
            if o>0: dev += 2*o*math.log(o/e)
        pear += (w-g*p)**2/(g*p*(1-p))
    df = len(E)-(K-1)
    return th, dev, pear, df, E

def main(run):
    nodes, agg = load(run)
    if len(nodes)<5 or len(agg)<10: return
    th, dev, pear, df, E = bt(nodes, agg)
    games = sum(g for *_,g in E)
    # overdispersion phi = pearson/df; excess cyclic variance on the pp scale
    phi = pear/df
    # mean binomial variance of a pp estimate at 100 games: ~25 pp^2 -> excess sd = sqrt((phi-1)*meanvar)
    mv = np.mean([ (1/(1+math.exp(-(th[i]-th[j]))))*(1-1/(1+math.exp(-(th[i]-th[j]))))/g for i,j,w,g in E])*1e4
    exc = math.sqrt(max(phi-1,0)*mv)
    spread = (th.max()-th.min())*400/math.log(10)
    pval = 1-chi2.cdf(pear, df)
    print(f"{run:42s} K={len(nodes):2d} edges={len(E):3d} games={games:6d} BTspread={spread:6.0f}Elo  dev/df={dev/df:5.2f} pearson/df={phi:5.2f} p={pval:.3g} excess-sd≈{exc:4.1f}pp")

if __name__=="__main__":
    runs = sys.argv[1:] or sorted({p.split('/')[-3] for p in glob.glob('/home/goodlad/dev/gen3ai/models/*/snapshot_ladder/games.jsonl')})
    for r in runs:
        try: main(r)
        except Exception as e: print(r, "ERR", e)
