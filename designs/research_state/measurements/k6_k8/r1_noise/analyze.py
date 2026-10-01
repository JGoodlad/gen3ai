"""Per-state, per-parameter envelope analysis of noise_cuda.jsonl."""
import json, sys
import numpy as np
rows = [json.loads(l) for l in open(sys.argv[1])]
print(f"{'state':10} {'gate max':>9} {'(param)':42} {'c64 max':>8} {'e64 max':>8} {'x64 max':>8} {'ee_cpu':>8} {'ee_det':>8} {'cc_rep':>8} | outside-envelope params (c64 > 2*max(e64,x64) and gate>1e-4)")
for r in rows:
    s, pp = r["summary"], r["per_param"]
    out = []
    for n, v in pp.items():
        env = max(v.get("e64") or 0, v.get("x64") or 0)
        if v.get("c64") is not None and v.get("gate") and v["gate"] > 1e-4 and v["c64"] > 2 * env:
            out.append((n, v["c64"], env, v["gate"], v.get("norm_frac")))
    out.sort(key=lambda t: -t[3])
    g = s["gate"]
    print(f"{r['state']:10} {g['max']:9.2e} {g['argmax'][-42:]:42} {s['c64']['max']:8.2e} {s['e64']['max']:8.2e} {s['x64']['max']:8.2e} "
          f"{s['ee_cpu']['max']:8.2e} {s['ee_det']['max']:8.2e} {s['cc_rep']['max']:8.2e} | {len(out)}: "
          + "; ".join(f"{n.split('.',1)[-1][:40]} c64 {c:.1e} env {e:.1e} gate {g_:.1e} nf {nf:.0e}" for n, c, e, g_, nf in out[:4]))
