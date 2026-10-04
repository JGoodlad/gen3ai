import numpy as np, math
from cyc import load, bt  # noqa: E402
rng=np.random.default_rng(0)
for run in ["ai_v14_01_base","ai_v12_02_winprob_critic","ai_v13_02_flywheel_winprob"]:
    nodes,agg=load(run); th,dev,pear,df,E=bt(nodes,agg)
    vals=[]
    for rep in range(30):
        agg2={}
        for (a,b),(w,g) in agg.items():
            i,j=nodes.index(a),nodes.index(b); p=1/(1+math.exp(-(th[i]-th[j])))
            agg2[(a,b)]=[int(rng.binomial(g,p)),g]
        _,_,pe2,df2,_=bt(nodes,agg2); vals.append(pe2/df2)
    print(run, "observed", round(pear/df,3), "simulated mean", round(np.mean(vals),3), "sd", round(np.std(vals),3))
