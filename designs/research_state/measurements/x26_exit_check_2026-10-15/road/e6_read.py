"""E6 (design_x26_readiness.md §4 row E6): the cost re-measure at P_X26 from rb_xs_s2701's update tables 10-50 — the median
train_ms and fps — vs rb_bf_eb's 42.07 s / 1,631 fps, within ±7 % (rule 8: a value within 1e-9 of a bound is not out).
Prints one JSON line. Usage: e6_read.py <run_dir>"""
import json, re, statistics as st, sys
from pathlib import Path
pat = re.compile(r"^\|\s+(\S+)\s+\|\s+([-+0-9.e]+)\s+\|")
tables, cur = [], {}
for line in (Path(sys.argv[1]) / "launcher_child.log").read_text(errors="replace").splitlines():
    if line.startswith("----") and cur:
        if "total_timesteps" in cur: tables.append(cur)
        cur = {}; continue
    m = pat.match(line)
    if m and m.group(1) in ("train_ms", "fps", "total_timesteps") and m.group(1) not in cur:
        try: cur[m.group(1)] = float(m.group(2))
        except ValueError: pass
use = [t for t in tables[9:50] if "train_ms" in t and "fps" in t]
out = {"tables": len(tables), "used": len(use)}
if len(use) >= 20:
    tm, fp = st.median(t["train_ms"] for t in use) / 1000.0, st.median(t["fps"] for t in use)
    rt, rf = tm / 42.07 - 1.0, fp / 1631.0 - 1.0
    ok = lambda r: abs(r) <= 0.07 + 1e-9
    out.update({"train_s_median": tm, "fps_median": fp, "train_rel": rt, "fps_rel": rf,
                "verdict": "WITHIN" if ok(rt) and ok(rf) else "OUTSIDE"})
else:
    out["verdict"] = "INSUFFICIENT (< 20 update tables in 10-50)"
print(json.dumps(out))
