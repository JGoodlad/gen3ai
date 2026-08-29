"""V-sign only (no win-prob head needed) on cap-length losses, across ALL eras --
the direct test of the runbook's causal claim for gen3_deadline_clock_v1."""
import json, os, re, collections
import numpy as np
rows = json.load(open('tmp/probeO_census.json'))
def fam(run):
    m=re.match(r"ai_v9_(\d+)_",run)
    if m:
        n=int(m.group(1))
        if n<=9:  return "ai_v9_01-09 (gen1-8, PRE-clock, no wp head)"
        if n<=12: return "ai_v9_10-12 (gen9-10, clock LANDS 08-12)"
        return "ai_v9_13+ (gen11+, clock present)"
    if re.match(r"ai_v[678]_",run): return "ai_v6/v7/v8 (PRE-clock)"
    return "other"
g=collections.defaultdict(list)
for r in rows:
    if (r["turns"] or 0) < 250: continue
    p=r["path"].replace("_summary.json","_states.npz")
    if not os.path.exists(p): continue
    try:
        with np.load(p) as z:
            v=np.asarray(z["values"],float); hs=np.asarray(z["has_state"]).astype(bool)
    except Exception: continue
    i=np.where(hs)[0]
    if not len(i): continue
    g[fam(r["run"])].append((float(v[i][-1]), r["run"]))
print("V at the FINAL decision of a CAP-LENGTH (>=250 turn) loss, by era family")
print(f"{'family':44s} {'n':>5s}  {'frac V>0':>9s}  {'mean V':>9s}  {'median V':>9s}")
order=["ai_v6/v7/v8 (PRE-clock)","ai_v9_01-09 (gen1-8, PRE-clock, no wp head)","ai_v9_10-12 (gen9-10, clock LANDS 08-12)","ai_v9_13+ (gen11+, clock present)"]
for k in order:
    if k not in g: continue
    v=np.array([x[0] for x in g[k]])
    print(f"{k:44s} {len(v):5d}  {np.mean(v>0):9.3f}  {v.mean():+9.2f}  {np.median(v):+9.2f}")
# ai_v9_09 specifically (the run the 13/14 came from)
v9=[x[0] for x in sum(g.values(),[]) if "ai_v9_09" in x[1]]
if v9:
    v9=np.array(v9); print(f"\n  ai_v9_09 ONLY (the 13/14 source run): n={len(v9)} frac V>0={np.mean(v9>0):.3f} mean={v9.mean():+.2f}")
for r in ("ai_v9_01","ai_v9_02","ai_v9_04"):
    vv=np.array([x[0] for x in sum(g.values(),[]) if r in x[1]])
    if len(vv): print(f"  {r} ONLY: n={len(vv)} frac V>0={np.mean(vv>0):.3f} mean={vv.mean():+.2f}")
