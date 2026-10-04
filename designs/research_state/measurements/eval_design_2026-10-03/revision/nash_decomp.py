import numpy as np, math, time
from cyc import load, bt
from nashlib import maxent_nash
rng=np.random.default_rng(1)
nodes,agg=load("ai_v14_01_base"); th,dev,pear,df,E=bt(nodes,agg); K=len(nodes)
arch=list(range(K-1)); new=K-1
def pt(i,j):
    k=(nodes[i],nodes[j])
    if k in agg: w,g=agg[k]; return w/g
    return 1/(1+math.exp(-(th[i]-th[j])))
Pnew_pt=np.array([1-pt(j,new) for j in arch])  # P(new beats j); nodes sorted so j<new
W=[];Pn=[];both=[]
for d in range(200):
    P=np.full((K,K),.5)
    for i in range(K):
        for j in range(i+1,K):
            k=(nodes[i],nodes[j])
            if k in agg: w,g=agg[k]; p=rng.beta(w+1,g-w+1)
            else: p=1/(1+math.exp(-(th[i]-th[j]+rng.normal(0,0.15))))
            P[i,j]=p; P[j,i]=1-p
    w=maxent_nash((P-.5)[np.ix_(arch,arch)])
    W.append(w); Pn.append(P[new,arch].copy()); both.append(P[new,arch]@w)
W=np.array(W); Pn=np.array(Pn); wbar=W.mean(0)
print("original (both vary) sd %.4f"%np.std(both))
print("mixture only (newest row fixed at point est) sd %.4f"%np.std(W@Pnew_pt))
print("newest row only (mixture fixed = posterior mean) sd %.4f"%np.std(Pn@wbar))
print("posterior-mean weights:",np.round(wbar,3))
print("newest score vs posterior-mean ref (point est): %.4f"%(Pnew_pt@wbar))
