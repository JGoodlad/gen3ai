"""Cycle monitor sizing (owner input 2026-10-03). Synthetic run: thinned snapshots every 10M; Elo r(t) = 300(1-exp(-t/30M))
(N0-like concave growth); mirrored-pair edge scores ~ N(p, sigma_pair^2/n_pairs), sigma_pair = 0.333 (P0).
Thinning schemes: UNIFORM (newest plays every earlier thinned node), GEOMETRIC (lags 1,2,4,8,16 x 10M),
HYBRID (lags 1..8 dense + 16, 32 geometric). Each new row is appended at its check (frozen edges never change).
Signals at the check, each at alpha = 0.05 (one-sided where directional):
 (a) BT residual chi-square on the ACCUMULATED matrix (logit WLS, delta-method weights), df = E - (K-1)
 (b) newest row non-monotone in lag: max over adjacent-lag pairs of z(score at shorter lag - score at longer lag), Bonferroni
 (c) intransitive triangles through the newest node: |cyclic logit flow|/SE, Holm over triangles
Alternatives: NULL (transitive); DIFFUSE (every edge's logit gets an antisymmetric N(0, tau) perturbation, tau quoted in pp at 1/2);
HOLE (the newest node's score vs ONE older node, at a uniformly random older lag, is lowered by delta pp: a treadmill hole)."""
import numpy as np, itertools, math
from scipy.stats import norm, chi2
rng=np.random.default_rng(3)
SIG=0.333; D=10
def elo(t): return 300*(1-math.exp(-t/30))
def lags(scheme,m):
    if scheme=="uniform": return list(range(1,m))
    if scheme=="geometric": return [l for l in (1,2,4,8,16,32) if l<m]
    if scheme=="hybrid": return [l for l in list(range(1,9))+[16,32] if l<m]
def edges(scheme,M):
    E=set()
    for m in range(2,M+1):           # node index m-1 is the newest at check m (nodes 0..M-1 at 10M,20M,...)
        for l in lags(scheme,m): E.add((m-1-l,m-1))
    return sorted(E)
def sim(scheme,M,npairs,alt,size,R=1500):
    E=edges(scheme,M); K=M; new=K-1
    th=np.array([elo(D*(i+1)) for i in range(K)])*math.log(10)/400
    hits={"a":0,"b":0,"c":0,"any":0}
    for r in range(R):
        P={}
        hole=None
        if alt=="hole":
            older=[i for (i,j) in E if j==new and i<new-1]   # exclude the lag-1 node: the hole is in the OLDER past
            hole=older[rng.integers(len(older))] if older else None
        for (i,j) in E:
            x=th[j]-th[i]
            if alt=="diffuse": x+=rng.normal(0,size/100*4)   # tau pp at 1/2 -> logit: dp = dx/4
            p=1/(1+math.exp(-x))
            if alt=="hole" and j==new and i==hole: p-=size/100
            s=np.clip(p+rng.normal(0,SIG/math.sqrt(npairs)),1e-3,1-1e-3)
            P[(i,j)]=s
        # logit flows + weights
        f=np.array([math.log(P[e]/(1-P[e])) for e in E]); v=np.array([SIG**2/npairs/(P[e]*(1-P[e]))**2 for e in E]); w=1/v
        B=np.zeros((len(E),K))
        for k,(i,j) in enumerate(E): B[k,j]=1;B[k,i]=-1
        B=B[:,1:]; W=np.diag(w)
        beta=np.linalg.lstsq(B.T@W@B,B.T@W@f,rcond=None)[0]; res=f-B@beta
        stat=(res**2*w).sum(); df=len(E)-(K-1)
        a = df>0 and stat>chi2.ppf(.95,df)
        row={i:(P[(i,new)],SIG/math.sqrt(npairs)) for (i,j) in E if j==new}
        ls=sorted(row,reverse=True)  # shorter lag first (higher index = more recent)
        zs=[(row[ls[q]][0]-row[ls[q+1]][0])/math.hypot(row[ls[q]][1],row[ls[q+1]][1]) for q in range(len(ls)-1)]
        b = len(zs)>0 and max(zs)>norm.ppf(1-.05/len(zs))
        tri=[(i,k) for i in row for k in row if i<k and (i,k) in P]
        pv=[]
        for (i,k) in tri:
            c=f[E.index((i,new))]-f[E.index((k,new))]-f[E.index((i,k))]   # loop i->k->new->i
            se=math.sqrt(v[E.index((i,new))]+v[E.index((k,new))]+v[E.index((i,k))])
            pv.append(2*norm.sf(abs(c)/se))
        c_=False
        if pv:
            pv=np.sort(pv); c_=any(pv[q]<=.05/(len(pv)-q) for q in range(len(pv)) if all(pv[:q+1]<=.05/(len(pv)-np.arange(q+1))))
        for key,val in (("a",a),("b",b),("c",c_)): hits[key]+=val
        hits["any"]+= (a or b or c_)
    return {k:round(v/R,3) for k,v in hits.items()}, sum(1 for e in E if e[1]==new)
for M in (8,15):
    for scheme in ("uniform","geometric","hybrid"):
        for npairs in (500,1000):
            out=[]
            for alt,size in (("null",0),("diffuse",1.0),("diffuse",1.5),("diffuse",2.0),("hole",4),("hole",6),("hole",8)):
                h,k=sim(scheme,M,npairs,alt,size,R=600)
                out.append(f"{alt}{size}:{h['a']}/{h['b']}/{h['c']}|{h['any']}")
            print(f"check@{M*10}M {scheme:9s} k={k:2d} pairs/edge={npairs:4d} games/row={2*npairs*k:6d} :: "+"  ".join(out), flush=True)
