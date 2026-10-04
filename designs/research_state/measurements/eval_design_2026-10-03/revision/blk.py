import numpy as np, glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
f=glob.glob('/home/goodlad/dev/gen3ai/models/sizing_B_n256_e10_s1001/tb/events.out*')[0]
ea=EventAccumulator(f,size_guidance={'scalars':0}); ea.Reload()
ev=ea.Scalars('train/train_ms'); wt=np.array([e.wall_time for e in ev]); st=np.array([e.step for e in ev])
cyc=np.diff(wt); stp=st[1:]
# eval steps: multiples of 2M crossed
ev_steps=set()
for i in range(1,len(st)):
    if st[i]//2_000_000 != st[i-1]//2_000_000: ev_steps.add(i-1)
keep=np.array([i not in ev_steps for i in range(len(cyc))])
c=cyc[keep]; print("n cycles",len(c),"median",np.median(c),"sd",c.std())
# blocks of 5 consecutive (kept) cycles -> block medians; paired adjacent blocks log-ratio
bm=[np.median(c[i:i+5]) for i in range(0,len(c)-4,5)]
lr=np.log(np.array(bm[1::2][:len(bm[::2])])/np.array(bm[::2][:len(bm[1::2])]))
print("blocks",len(bm),"paired log-ratio sd %.4f"%lr.std(ddof=1))
s=lr.std(ddof=1); print("bound margin under null 1.895*s/sqrt8 = %.4f"%(1.895*s/8**.5))
from scipy.stats import t,nct
# P(adopt | true ratio r): P(mean + t*s/sqrt8 <= 0.01)
for r in [0.0,0.005,0.01]:
    # mean ~ N(r, s^2/8); s estimated -> noncentral t approx
    d=(0.01-r)/(s/8**.5); print(r, "P(adopt)~", round(nct.sf(1.895,7,d),3))
print(np.round(c,1))
print(np.round(bm,1))
