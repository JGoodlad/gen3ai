"""M2: operating characteristics of the owner-registered plateau check — newest vs W-back, mirrored pairs,
pentanomial GSPRT H0 0.50 / H1 0.52, alpha = beta = 0.05, Wald bounds, 40-pair batches (sprt.py's own code).
Pentanomial shape from P0's cross edges (pooled), exponentially tilted to each true mean."""
import json, glob, numpy as np
from scipy.optimize import brentq
from agents.training.sprt import SprtConfig, SprtState
S=np.array([0,.25,.5,.75,1.])
c=np.zeros(5)
for p in glob.glob('/home/goodlad/dev/gen3ai/designs/research_state/measurements/x5_p0_h2h_2026-10-03/rows/*.jsonl'):
    for l in open(p):
        r=json.loads(l)
        if r["player"]["id"]!=r["opponent"]["id"]: c+=np.array(r["pairs"]["counts"])
base=c/c.sum(); m=base@S; sd=np.sqrt(base@(S-m)**2)
print("P0 cross pentanomial",np.round(base,4),"mean %.4f pairSD %.4f pairs %d"%(m,sd,c.sum()))
base=(base+base[::-1])/2   # symmetrise -> mean exactly 1/2
def tilt(mu):
    f=lambda t: (base*np.exp(t*S))@S/((base*np.exp(t*S)).sum())-mu
    t=brentq(f,-20,20); q=base*np.exp(t*S); return q/q.sum()
rng=np.random.default_rng(11)
for cap in (6000, 9000):
    cfg=SprtConfig(p0=0.50,p1=0.52,alpha=0.05,beta=0.05,max_pairs=cap,batch_pairs=40,min_pairs=0,bounds_mode="wald")
    for mu in (0.49,0.50,0.505,0.51,0.515,0.52,0.53):
        q=tilt(mu); R=2000; acc0=acc1=capd=0; N=[]
        for r in range(R):
            s=SprtState(cfg); v="continue"
            while v=="continue":
                v=s.add_batch(rng.multinomial(s.next_batch_pairs(),q).tolist())
            N.append(s.n_pairs)
            if s.reason=="cap": capd+=1
            elif v=="reject": acc0+=1
            else: acc1+=1
        N=np.array(N)
        print(f"cap {cap:5d} mu {mu:.3f}: P(accept H0 = plateau) {acc0/R:.3f}  P(H1) {acc1/R:.3f}  P(cap) {capd/R:.3f}  E[pairs] {N.mean():6.0f}  E[games] {2*N.mean():6.0f}  q90 pairs {np.quantile(N,.9):.0f}")
