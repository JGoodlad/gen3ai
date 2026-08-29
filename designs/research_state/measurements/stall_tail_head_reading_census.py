"""Probe O step 1: CENSUS ONLY. No phi is read here."""
import json, os, sys, glob
import numpy as np

RUNS = sys.argv[1:]
rows = []
for run in RUNS:
    base = f"/home/goodlad/dev/gen3ai/models/{run}/eval_traces"
    if not os.path.isdir(base):
        print(f"{run}: NO eval_traces"); continue
    for s in glob.glob(base + "/**/*_summary.json", recursive=True):
        try:
            d = json.load(open(s))
        except Exception:
            continue
        m = d.get("meta", {})
        npz = s.replace("_summary.json", "_states.npz")
        rows.append(dict(run=run, path=s, step=m.get("step"), result=m.get("result"),
                         turns=m.get("turns"), inv=m.get("invocations"),
                         opp=os.path.basename(os.path.dirname(s)),
                         has_npz=os.path.exists(npz)))
print("total summaries:", len(rows))
import collections
by_run = collections.Counter(r["run"] for r in rows)
for k, v in sorted(by_run.items()):
    sub = [r for r in rows if r["run"] == k]
    t = [r["turns"] for r in sub if r["turns"] is not None]
    res = collections.Counter(r["result"] for r in sub)
    npzc = sum(r["has_npz"] for r in sub)
    print(f"{k:32s} n={v:5d} npz={npzc:5d} results={dict(res)} turns med={np.median(t):.0f} p90={np.percentile(t,90):.0f} max={max(t)}")
    for lo in (150, 200, 240, 245, 249, 250):
        c = sum(1 for r in sub if (r["turns"] or 0) >= lo)
        cn = sum(1 for r in sub if (r["turns"] or 0) >= lo and r["has_npz"])
        if c: print(f"      turns>={lo}: {c} (npz {cn})  results={dict(collections.Counter(r['result'] for r in sub if (r['turns'] or 0)>=lo))}")
json.dump(rows, open("/home/goodlad/dev/gen3ai/.claude/worktrees/agent-a6db0fc1ce9791e72/tmp/probeO_census.json","w"))
