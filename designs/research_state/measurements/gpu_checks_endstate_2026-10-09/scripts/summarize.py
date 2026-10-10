"""Join the per-launch reads (`read_run.py` outputs) into the cost table (gpu_checks_endstate, 2026-10-09).

    python summarize.py --results results --cfgs P R E R2 --base P --out results/numbers.json

Every delta is (cfg − base) / base on the STEADY medians (quiet, non-warm-up, non-canary updates, standing rule 8).
Where a config keeps fewer than 3 quiet updates, the table also carries the median over EVERY non-warm-up,
non-canary update (contended included), marked DESCRIPTIVE, so the read never rests on a single update silently.
"""
from __future__ import annotations

import argparse
import json
import statistics as st
from pathlib import Path

ROWS = [("train/train_ms", "train_ms per update (ms)"),
        ("cycle_fps", "cycle rows/s (rollout + train)"),
        ("cycle_rollout_s", "rollout wall per update (s)"),
        ("rust_env/gpu_wait_ms_per_host_step", "T2 GPU wait per host step (ms)"),
        ("rust_env/flush_ms_per_host_step", "T2 host flush per host step (ms)"),
        ("lifecycle/cuda_update_peak_reserved_mib", "update peak reserved (MiB)"),
        ("lifecycle/cuda_device_free_mib", "card free after the dry update (MiB)")]


def all_but_warm(d, tag):
    rows = d["updates"]
    ok = [r["k"] for r in rows if not (r["excluded"] or "").startswith(("warm-up", "compile-canary", "the stop"))]
    if tag in ("cycle_fps", "cycle_rollout_s"):
        key = "fps" if tag == "cycle_fps" else "rollout_s"
        vals = [c[key] for c in d["cycles"] if c["k"] in ok]
    else:
        s = d["tb"][tag]["all"]
        vals = [s[k] for k in ok if k < len(s)]
    return round(st.median(vals), 4) if vals else None, len(vals)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--cfgs", nargs="+", required=True)
    ap.add_argument("--base", default="P")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    R = {c: json.loads((Path(a.results) / f"{c}.json").read_text()) for c in a.cfgs}
    out = {"configs": {}, "deltas_vs_base": {}, "base": a.base}
    for c, d in R.items():
        row = {"kept_updates": d["kept_updates"], "contended_updates": d["contended_updates"],
               "t2_up_s": d.get("t2_up_s"), "r1_reset_prewarm_s": d.get("prewarm_s"),
               "start_to_first_update_s": d.get("start_to_first_update_s"),
               "startup_windows": d.get("startup_windows"), "startup_windows_ge_1.05": d.get("startup_windows_ge_1.05"),
               "startup_max_factor": d.get("startup_max_factor"), "update_fit": d.get("update_fit"),
               "canary": d.get("canary_verdicts"), "graphs_total": d["tb"]["compile/graphs_total"]["all"][-1:],
               "recompiles_after_lock_max": max(d["tb"]["compile/recompiles_after_lock"]["all"] or [None]),
               "cache_limit_hits_max": max(d["tb"]["compile/cache_limit_hits"]["all"] or [None])}
        for tag, _label in ROWS:
            t = d["tb"][tag]
            row[tag] = {"steady": t["median_steady"], "n_steady": t["n_steady"]}
            row[tag]["all_but_warm_DESCRIPTIVE"], row[tag]["n_all"] = all_but_warm(d, tag)
        out["configs"][c] = row
    b = out["configs"][a.base]
    for c in a.cfgs:
        if c == a.base:
            continue
        dl = {}
        for tag, _ in ROWS:
            x, y = out["configs"][c][tag], b[tag]
            for k in ("steady", "all_but_warm_DESCRIPTIVE"):
                if x[k] is not None and y[k]:
                    dl.setdefault(tag, {})[k] = round(100.0 * (x[k] - y[k]) / y[k], 2)
        out["deltas_vs_base"][c] = dl
    Path(a.out).write_text(json.dumps(out, indent=1))
    hdr = f"{'quantity':42s}" + "".join(f"{c:>26s}" for c in a.cfgs)
    print(hdr)
    for tag, label in ROWS:
        cells = []
        for c in a.cfgs:
            x = out["configs"][c][tag]
            cells.append(f"{x['steady']} (n{x['n_steady']}) / {x['all_but_warm_DESCRIPTIVE']} (n{x['n_all']})")
        print(f"{label:42s}" + "".join(f"{s:>26s}" for s in cells))
    for c, dl in out["deltas_vs_base"].items():
        print(c, "vs", a.base, json.dumps(dl))
    for c in a.cfgs:
        r = out["configs"][c]
        print(c, {k: r[k] for k in ("t2_up_s", "r1_reset_prewarm_s", "start_to_first_update_s", "startup_windows",
                                    "startup_windows_ge_1.05", "startup_max_factor", "graphs_total",
                                    "recompiles_after_lock_max", "cache_limit_hits_max")})


if __name__ == "__main__":
    main()
