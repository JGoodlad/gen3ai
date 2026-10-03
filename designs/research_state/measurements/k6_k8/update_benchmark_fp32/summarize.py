"""Summarize the fp32 update-benchmark units (B = the committed tree, P = the planted-regression tree, L = B under
6 CPU-bound bystanders).

    python3 summarize.py [<results dir>]      # default: ./results, rows copied there by `collect.sh`

Per unit: the un-bracketed update, the bracketed one, the profiled compiled share, the update's work tags (the
same program ran: `train/loss` must agree across B units to the digit), and the unit's box state. Then the
B units' spread."""
import glob
import json
import os
import statistics
import sys

d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
rows = {}
for f in sorted(glob.glob(os.path.join(d, "*_time_result.json"))):
    u = os.path.basename(f).split("_")[0]
    r = json.load(open(f))
    a_path = f.replace("_time_result.json", "_time_analysis.json")
    t = json.load(open(a_path))["profiles"][0]["totals"] if os.path.exists(a_path) else {}
    unb = (r.get("unbracketed") or {}).get("train_ms")
    rows[u] = dict(status=r.get("status"), regions=r.get("compiled_regions"), restored=r.get("buffer_restored"),
                   rows=(r.get("geometry") or {}).get("rows"), unbracketed_s=None if unb is None else unb / 1000,
                   bracketed_s=r["bracketed"]["train_ms"] / 1000 if r.get("bracketed") else None,
                   share=t.get("compiled_share_of_train_wall"), kernel_share=t.get("compiled_kernel_share"),
                   idle_ms=t.get("gpu_idle_ms"),
                   loss=((r.get("unbracketed") or {}).get("work") or {}).get("train/loss"))
for u, r in rows.items():
    print(u, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
B = [r["unbracketed_s"] for u, r in rows.items() if u.startswith("B") and r["unbracketed_s"] is not None]
S = [r["share"] for u, r in rows.items() if u.startswith("B") and r["share"] is not None]
if len(B) >= 2:
    mean, lo, hi = statistics.mean(B), min(B), max(B)
    spread = (hi - lo) / mean
    print(f"B: n={len(B)} update mean {mean:.2f} s, median {statistics.median(B):.2f}, range {lo:.2f}..{hi:.2f}, "
          f"relative spread (max-min)/mean {spread:.2%}, stdev {statistics.stdev(B):.2f} s")
    print(f"   compiled share of the update wall: min {min(S):.3f} mean {statistics.mean(S):.3f} max {max(S):.3f}")
for u, r in rows.items():
    if u[0] in "PL" and r["unbracketed_s"] is not None and B:
        what = "planted regression" if u[0] == "P" else "6 CPU-bound bystanders"
        print(f"{u} ({what}): {r['unbracketed_s']:.2f} s = {r['unbracketed_s'] / statistics.mean(B):.3f}x the B mean "
              f"({r['unbracketed_s'] / statistics.mean(B) - 1:+.2%}), share {r['share']}")
