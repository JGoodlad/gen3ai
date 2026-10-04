import numpy as np, math, sys
from cyc import load, bt
from scipy.optimize import minimize, linprog
def maxent_nash(A):
    # symmetric zero-sum game, A antisymmetric payoff to the row player (P - 1/2). Nash set: {p: A^T p >= 0}, i.e. no pure strategy j beats the mixture: sum_i p_i A[i,j] >= 0... use (p^T A)_j >= -tol
    K=len(A); tol=1e-9
    cons=[{'type':'ineq','fun':lambda p,A=A: p@A + tol, 'jac':lambda p,A=A: A.T},
          {'type':'eq','fun':lambda p: p.sum()-1,'jac':lambda p: np.ones_like(p)}]
    # feasible start from LP
    res=linprog(np.zeros(K),A_ub=-A.T,b_ub=np.full(K,tol),A_eq=np.ones((1,K)),b_eq=[1],bounds=[(0,1)]*K)
    p0=res.x if res.success else np.ones(K)/K
    f=lambda p: np.sum(np.where(p>1e-12,p*np.log(np.maximum(p,1e-12)),0))
    g=lambda p: np.log(np.maximum(p,1e-12))+1
    r=minimize(f,np.clip(p0,1e-6,1),jac=g,constraints=cons,bounds=[(0,1)]*K,method='SLSQP',options={'maxiter':500,'ftol':1e-12})
    return np.clip(r.x,0,1)/np.clip(r.x,0,1).sum()
rng=np.random.default_rng(1)
run=sys.argv[1] if len(sys.argv)>1 else "ai_v14_01_base"
nodes,agg=load(run); th,dev,pear,df,E=bt(nodes,agg); K=len(nodes)
arch=list(range(K-1)); new=K-1   # archive = all but the newest node
sup=np.zeros(K-1); vals=[]
for d in range(200):
    P=np.full((K,K),.5)
    for i in range(K):
        for j in range(i+1,K):
            k=(nodes[i],nodes[j])
            if k in agg: w,g=agg[k]; p=rng.beta(w+1,g-w+1)
            else:
                mu=th[i]-th[j]; p=1/(1+math.exp(-(mu+rng.normal(0,0.15))))
            P[i,j]=p; P[j,i]=1-p
    A=(P-.5)[np.ix_(arch,arch)]
    w=maxent_nash(A); sup+= (w>0.01)
    vals.append(float(P[new,arch]@w))
print(run,"nodes",[int(n)//1_000_000 for n in nodes])
print("P(weight>0.01) per archive node:",np.round(sup/200,2))
print("newest(%dM) score vs archive Nash: mean %.3f sd %.3f"%(int(nodes[new])//1e6,np.mean(vals),np.std(vals)))
P=np.full((K,K),.5)
for i in range(K):
    for j in range(K):
        if i!=j: P[i,j]=1/(1+math.exp(-(th[i]-th[j])))
w=maxent_nash((P-.5)[np.ix_(arch,arch)]); print("BT-smoothed point Nash:",np.round(w,2))
