import json, collections, math
import numpy as np
recs = json.load(open('tmp/probeO_recs_final.json'))
rows = json.load(open('tmp/probeO_census.json'))
import re
def era(run):
    m=re.match(r"ai_v9_(\d+)_",run)
    if m: return "clock" if int(m.group(1))>=13 else "gen1_10_nohead"
    return "preclock" if re.match(r"ai_v[678]_",run) else "other"

# --- base rates: cap endings as a fraction of TRACED losses, per era
print("===== base rate of cap endings among TRACED losses (quota-sampled: 10 losses/opp/step) =====")
for e in ("clock","preclock"):
    sub=[r for r in rows if era(r["run"])==e]
    L=[r for r in sub if r["result"]=="LOSS"]
    C=[r for r in L if (r["turns"] or 0)>=250]
    print(f"  {e:9s} traced losses {len(L):6d}  cap-length {len(C):4d}  = {len(C)/max(1,len(L)):.4f}")

# --- THE REGISTERED META-ANALYSIS (gen14_endofrun_runbook): per-run WITHIN difference, pooled
print("\n===== META-ANALYSIS: per-run WITHIN difference (cap-length losses vs ordinary losses), pooled =====")
print("    (the runbook's sanctioned cross-generation operation: pool the DIFFERENCES, never the levels)")
for e in ("clock","preclock"):
    for m,label in (("vpos","V>0 at final decision"),("detect_le05","phi_T<=0.5 (detection)"),
                    ("overconf","max phi over last 5 >= 0.7"),("c3band","mean5 in C3 band")):
        byrun=collections.defaultdict(lambda: {"cap":[], "ord":[]})
        for r in recs:
            if r["era"]!=e: continue
            if r["base"]=="CAP": byrun[r["run"]]["cap"].append(r[m])
            elif r["cls"] in ("REG_LOSS","LONG_LOSS_SLOW","LONG_LOSS_FAST"): byrun[r["run"]]["ord"].append(r[m])
        diffs=[]; wts=[]
        for run,d in byrun.items():
            if len(d["cap"])>=1 and len(d["ord"])>=10:
                diffs.append(np.mean(d["cap"])-np.mean(d["ord"])); wts.append(len(d["cap"]))
        if len(diffs)<3: continue
        diffs=np.array(diffs); wts=np.array(wts,float)
        w=(wts*diffs).sum()/wts.sum()
        rng=np.random.default_rng(5)
        bs=[np.average(diffs[i],weights=wts[i]) for i in (rng.integers(0,len(diffs),(4000,len(diffs))))]
        lo,hi=np.percentile(bs,2.5),np.percentile(bs,97.5)
        sig="SIG" if (lo>0 or hi<0) else "n.s."
        print(f"  [{e:8s}] {label:34s} pooled within-run diff {w:+.3f} CI95 [{lo:+.3f},{hi:+.3f}] {sig}   ({len(diffs)} runs, {int(wts.sum())} cap games, {sum(1 for x in diffs if x>0)} runs positive)")

# --- the 13/14 analogue, spelled out
print("\n===== THE 13/14 ANALOGUE (sign of V at the FINAL decision of a cap-length loss) =====")
for e,lbl in (("preclock","PRE-CLOCK era pool"),("clock","CLOCK era pool")):
    g=[r for r in recs if r["era"]==e and r["base"]=="CAP"]
    k=sum(r["vpos"] for r in g); n=len(g)
    v=[r["v_T"] for r in g]
    print(f"  {lbl:20s} {k}/{n} = {k/n:.3f} positive V   mean V {np.mean(v):+.2f}  median V {np.median(v):+.2f}")
print(f"  HISTORICAL (ai_v9_09 @16M, pre-clock)  13/14 = 0.929 positive V   mean V +9.33")
print(f"  gen-13 spot check (runbook)             2/9  = 0.222")

# --- phi_T >= 0.5 = an outright MISS on a game that is a LOSS by construction
print("\n===== OUTRIGHT MISS RATE on cap endings (phi_T >= 0.5 where the true label is LOSS) =====")
for e in ("clock","preclock"):
    g=[r for r in recs if r["era"]==e and r["base"]=="CAP"]
    o=[r for r in recs if r["era"]==e and r["cls"] in ("REG_LOSS","LONG_LOSS_SLOW","LONG_LOSS_FAST")]
    print(f"  {e:9s} cap {np.mean([r['phi_T']>=0.5 for r in g]):.3f} (n={len(g)})  vs ordinary losses {np.mean([r['phi_T']>=0.5 for r in o]):.3f} (n={len(o)})   ratio {np.mean([r['phi_T']>=0.5 for r in g])/max(1e-9,np.mean([r['phi_T']>=0.5 for r in o])):.1f}x")
    print(f"            cap phi_T>=0.9: {np.mean([r['phi_T']>=0.9 for r in g]):.3f}   >=0.98: {np.mean([r['phi_T']>=0.98 for r in g]):.3f}")
