"""Probe O final pass. Refined classes + run-clustered bootstrap."""
import json, os, re, random, collections, math
import numpy as np

rows = json.load(open('tmp/probeO_census.json'))
K = 5
def era(run):
    m = re.match(r"ai_v9_(\d+)_", run)
    if m: return "clock" if int(m.group(1)) >= 13 else "gen1_10_nohead"
    return "preclock" if re.match(r"ai_v[678]_", run) else "other"
def is_current(run):
    m = re.match(r"ai_v9_(\d+)_", run); return bool(m) and int(m.group(1)) >= 29

def faints_tail(path, tail=20):
    try: s = json.load(open(path))
    except Exception: return None
    invs = s.get("invocations", [])
    if not invs: return None
    lt = max((i.get("turn") or 0) for i in invs)
    return sum(sum(1 for e in ((i.get("outcome") or {}).get("events") or []) if ":fainted" in e)
               for i in invs if (i.get("turn") or 0) >= lt - tail)

def read(path):
    npz = path.replace("_summary.json", "_states.npz")
    if not os.path.exists(npz): return None
    try:
        with np.load(npz) as z:
            if "win_probs" not in z: return None
            w = np.asarray(z["win_probs"], float); v = np.asarray(z["values"], float)
            hs = np.asarray(z["has_state"]).astype(bool)
    except Exception: return None
    i = np.where(hs & ~np.isnan(w))[0]
    return (w[i], v[i]) if len(i) else None

random.seed(20260829)
cand = collections.defaultdict(list)
for r in rows:
    e = era(r["run"])
    if e not in ("clock", "preclock"): continue
    t = r["turns"] or 0; res = r["result"]
    if t >= 250: cand[("CAP", e)].append(r)
    elif res == "LOSS" and 100 <= t < 250: cand[("LONG_LOSS", e)].append(r)
    elif res == "LOSS" and t < 50: cand[("REG_LOSS", e)].append(r)
    elif res == "WIN" and t >= 100: cand[("LONG_WIN", e)].append(r)

CAP_CTRL = 1200
recs = []
for (c, e), lst in cand.items():
    if c in ("REG_LOSS", "LONG_LOSS", "LONG_WIN") and len(lst) > CAP_CTRL:
        lst = random.sample(lst, CAP_CTRL)
    for r in lst:
        pv = read(r["path"])
        if pv is None: continue
        phi, val = pv
        cls = c
        nf = None
        if c in ("CAP", "LONG_LOSS"):
            nf = faints_tail(r["path"])
            if nf is None: continue
            if c == "CAP": cls = "CAP_STALL" if nf == 0 else "CAP_TRADE"
            else: cls = "LONG_LOSS_SLOW" if nf <= 1 else "LONG_LOSS_FAST"
        last5 = phi[-K:]
        recs.append(dict(run=r["run"], era=e, current=is_current(r["run"]), cls=cls, base=c,
                         turns=r["turns"], result=r["result"], opp=r["opp"], ndec=int(len(phi)),
                         faints_tail20=nf, phi_traj=[float(x) for x in phi[-10:]],
                         phi_T=float(phi[-1]), mean5=float(np.mean(last5)),
                         max5=float(np.max(last5)), v_T=float(val[-1]),
                         detect_le05=bool(phi[-1] <= 0.5),
                         detect_decl=bool(len(phi) >= K and phi[-1] < phi[-K]),
                         detect=bool(phi[-1] <= 0.5 or (len(phi) >= K and phi[-1] < phi[-K])),
                         overconf=bool(np.max(last5) >= 0.70),
                         c3band=bool(0.70 <= float(np.mean(last5)) <= 0.98),
                         vpos=bool(val[-1] > 0)))
json.dump(recs, open("tmp/probeO_recs_final.json", "w"))
print("records:", len(recs))

ORDER = ["CAP_STALL", "CAP_TRADE", "LONG_LOSS_SLOW", "LONG_LOSS_FAST", "REG_LOSS", "LONG_WIN"]
def wil(k, n, z=1.96):
    if not n: return (float('nan'),)*2
    p=k/n; d=1+z*z/n; c=p+z*z/(2*n); h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n)); return ((c-h)/d,(c+h)/d)

def table(sub, title):
    print(f"\n===== {title} (n={len(sub)}) =====")
    print(f"{'class':16s}{'n':>5s} |   T-4    T-3    T-2    T-1      T  | phi_T mean | detect  le0.5   decl | overconf  c3band   V>0")
    out={}
    for c in ORDER:
        g=[r for r in sub if r["cls"]==c]
        if not g: print(f"{c:16s}{0:5d} |  -- none --"); continue
        n=len(g); cols=[np.median([r["phi_traj"][-(5-k)] for r in g if len(r["phi_traj"])>=5-k]) for k in range(5)]
        f=lambda m: sum(r[m] for r in g)/n
        lo,hi=wil(sum(r['detect_le05'] for r in g),n)
        print(f"{c:16s}{n:5d} | " + " ".join(f"{x:6.3f}" for x in cols) +
              f" | {np.mean([r['phi_T'] for r in g]):10.3f} | {f('detect'):.3f}  {f('detect_le05'):.3f}[{lo:.2f},{hi:.2f}] {f('detect_decl'):.3f} |"
              f"  {f('overconf'):.3f}   {f('c3band'):.3f}  {f('vpos'):.3f}")
        out[c]=dict(n=n, phi_med=[float(x) for x in cols], phi_T_mean=float(np.mean([r['phi_T'] for r in g])),
                    detect=f('detect'), detect_le05=f('detect_le05'), detect_decl=f('detect_decl'),
                    overconf=f('overconf'), c3band=f('c3band'), vpos=f('vpos'),
                    frac_phiT_ge07=float(np.mean([r['phi_T']>=0.7 for r in g])),
                    frac_phiT_ge05=float(np.mean([r['phi_T']>=0.5 for r in g])))
    return out

res = {}
res["clock"] = table([r for r in recs if r["era"]=="clock"], "CLOCK ERA (ai_v9_13+, deadline clock in obs)")
res["preclock"] = table([r for r in recs if r["era"]=="preclock"], "PRE-CLOCK ERA (ai_v6_03..ai_v8_20, head on, no clock)")
res["current"] = table([r for r in recs if r["current"]], "CURRENT-ARCH SUB-POOL (ai_v9_29 rev-1 onward)")

# ---- run-clustered bootstrap on contrasts -------------------------------------
def cluster_boot(a, b, metric, B=5000, seed=3):
    """a,b: lists of recs. Resample RUNS with replacement within each arm."""
    rng = np.random.default_rng(seed)
    ga = collections.defaultdict(list); gb = collections.defaultdict(list)
    for r in a: ga[r["run"]].append(r[metric])
    for r in b: gb[r["run"]].append(r[metric])
    ka, kb = list(ga), list(gb)
    d = []
    for _ in range(B):
        va = np.concatenate([ga[ka[i]] for i in rng.integers(0, len(ka), len(ka))])
        vb = np.concatenate([gb[kb[i]] for i in rng.integers(0, len(kb), len(kb))])
        d.append(va.mean() - vb.mean())
    obs = np.mean([r[metric] for r in a]) - np.mean([r[metric] for r in b])
    return obs, float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), len(ka), len(kb)

print("\n===== REGISTERED CONTRAST: stall/cap vs REGULAR LOSSES (run-clustered bootstrap) =====")
res["contrasts"] = {}
for e in ("clock", "preclock"):
    sub = [r for r in recs if r["era"]==e]
    cap = [r for r in sub if r["base"]=="CAP"]; reg = [r for r in sub if r["cls"]=="REG_LOSS"]
    for m in ("detect", "detect_le05", "overconf", "c3band", "vpos"):
        o, lo, hi, na, nb = cluster_boot(cap, reg, m)
        sig = "SIG" if (lo > 0 or hi < 0) else "n.s."
        print(f"  [{e:8s}] {m:12s} CAP {np.mean([r[m] for r in cap]):.3f} (n={len(cap)}, {na} runs) vs REG_LOSS {np.mean([r[m] for r in reg]):.3f} (n={len(reg)}, {nb} runs)  diff {o:+.3f} CI95 [{lo:+.3f},{hi:+.3f}] {sig}")
        res["contrasts"][f"{e}_cap_vs_reg_{m}"] = dict(diff=float(o), lo=lo, hi=hi, n_a=len(cap), n_b=len(reg), runs_a=na, runs_b=nb)

print("\n===== HISTORICAL DELTA: CAP endings, CLOCK era vs PRE-CLOCK era (run-clustered) =====")
for m in ("detect", "detect_le05", "overconf", "c3band", "vpos"):
    a=[r for r in recs if r["era"]=="clock" and r["base"]=="CAP"]; b=[r for r in recs if r["era"]=="preclock" and r["base"]=="CAP"]
    o, lo, hi, na, nb = cluster_boot(a, b, m)
    sig = "SIG" if (lo > 0 or hi < 0) else "n.s."
    print(f"  {m:12s} clock {np.mean([r[m] for r in a]):.3f} (n={len(a)}, {na} runs) vs preclock {np.mean([r[m] for r in b]):.3f} (n={len(b)}, {nb} runs)  diff {o:+.3f} CI95 [{lo:+.3f},{hi:+.3f}] {sig}")
    res["contrasts"][f"cap_clock_vs_preclock_{m}"] = dict(diff=float(o), lo=lo, hi=hi, n_a=len(a), n_b=len(b))

print("\n===== OPPONENT SPLIT: clock-era CAP endings =====")
def oclass(o):
    o=o.lower()
    for k in ("staller","sentinel","heuristic","random","aggressive","setup_sweep","ext_"): 
        if o.startswith(k): return k.rstrip("_")
    return o
g=collections.defaultdict(list)
for r in recs:
    if r["era"]=="clock" and r["base"]=="CAP": g[oclass(r["opp"])].append(r)
res["opp_split_clock_cap"] = {}
for k in sorted(g, key=lambda k: -len(g[k])):
    v=g[k]
    print(f"  {k:12s} n={len(v):3d}  phi_T mean {np.mean([r['phi_T'] for r in v]):.3f}  overconf {np.mean([r['overconf'] for r in v]):.3f}  le0.5 {np.mean([r['detect_le05'] for r in v]):.3f}  V>0 {np.mean([r['vpos'] for r in v]):.3f}")
    res["opp_split_clock_cap"][k]=dict(n=len(v), phi_T_mean=float(np.mean([r['phi_T'] for r in v])),
        overconf=float(np.mean([r['overconf'] for r in v])), detect_le05=float(np.mean([r['detect_le05'] for r in v])),
        vpos=float(np.mean([r['vpos'] for r in v])))

print("\n===== C3 BAND CHECK (clock era) — fraction of stall tails still in phi 0.70-0.98 =====")
for c in ("CAP_STALL","CAP_TRADE"):
    v=[r for r in recs if r["era"]=="clock" and r["cls"]==c]
    if not v: continue
    print(f"  {c:11s} n={len(v):3d}  mean5 in [0.70,0.98]: {np.mean([r['c3band'] for r in v]):.3f}   max5>=0.70: {np.mean([r['overconf'] for r in v]):.3f}   phi_T>=0.5 (a MISS): {np.mean([r['phi_T']>=0.5 for r in v]):.3f}")
res["meta"]=dict(K=K, n_records=len(recs), corpus_summaries=len(rows),
                 classes={c:int(sum(1 for r in recs if r['cls']==c)) for c in ORDER})
json.dump(res, open("tmp/probeO_result.json","w"), indent=1)
