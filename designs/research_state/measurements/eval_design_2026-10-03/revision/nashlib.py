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
