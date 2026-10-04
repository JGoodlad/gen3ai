"""M6 v2: per-phase paired adjacent-update log differences, excluding eval-straddling updates and the periodic
heavy update (train_ms > 1.12 x the run median: a periodic side task every ~10 updates), robust SD (1.4826 x MAD),
and a Monte Carlo power check of a Hodges-Lehmann / Wilcoxon one-sided bound under the measured, heavy-tailed noise
(bootstrap of the observed differences, recentred)."""
import glob, numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scipy.stats import norm, wilcoxon
rng=np.random.default_rng(7)
def hl_upper(d, alpha=0.05):
    # one-sided upper bound of the Hodges-Lehmann shift via Walsh averages (normal approx to the signed-rank null)
    n=len(d); w=np.sort([(d[i]+d[j])/2 for i in range(n) for j in range(i,n)]); N=len(w)
    k=int(np.floor(n*(n+1)/4 - norm.ppf(1-alpha)*np.sqrt(n*(n+1)*(2*n+1)/24)))
    return w[N-1-max(k,0)]
for run in ["sizing_B_n256_e10_s1001","sizing_C_n256_e5_s1001"]:
    f=sorted(glob.glob(f'/home/goodlad/dev/gen3ai/models/{run}/tb/events.out*'))[0]
    ea=EventAccumulator(f,size_guidance={'scalars':0}); ea.Reload()
    def S(tag): return {e.step:e.value for e in ea.Scalars(tag)}
    tr=S('train/train_ms'); co=S('rollout/collect_ms'); de=S('rollout/collect_decisions'); el=S('rollout/ep_len_mean')
    st=np.array(sorted(set(tr)&set(co)&set(de)&set(el)))
    keep=np.ones(len(st),bool)
    for i in range(1,len(st)):
        if st[i]//2_000_000!=st[i-1]//2_000_000: keep[i]=False; keep[i-1]=False
    T=np.array([tr[s] for s in st]); keep &= T < 1.12*np.median(T); keep[0]=False
    x_t=np.log(T); x_c=np.log([co[s]/de[s] for s in st])
    for name,x in [("train_ms",x_t),("collect_ms/decision",x_c)]:
        d=np.array([x[i+1]-x[i] for i in range(0,len(x)-1,2) if keep[i] and keep[i+1]])
        rsd=1.4826*np.median(np.abs(d-np.median(d)))
        za,zb=norm.ppf(.95),norm.ppf(.90); m=np.log(1.01)
        n_norm=int(np.ceil(((za+zb)*rsd/m)**2))
        # MC power of HL upper bound <= log(1.01) at true shift 0, resampling the recentred observed d
        dc=d-np.median(d); res={}
        for n in (50,100,150,200,300):
            ok=0
            for r in range(400):
                s=rng.choice(dc,n,replace=True)*rng.choice([-1,1],n)  # symmetrised
                ok+= hl_upper(s) <= m
            res[n]=ok/400
        print(f"{run:24s} {name:20s} pairs={len(d):2d} sd={d.std(ddof=1):.4f} robust_sd={rsd:.4f} n(normal,rob)={n_norm} MC power(HL bound<=1%) {res}")
