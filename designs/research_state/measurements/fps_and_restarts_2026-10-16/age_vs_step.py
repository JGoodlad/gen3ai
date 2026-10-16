"""Does the PROCESS'S AGE slow a rollout, at a matched training step? (T24's natural experiment.)

    python .../age_vs_step.py <cache.json> <out.json>

The bf16 A/B arms ``rb_bf_e_*`` and ``rb_bf_eb_*`` each ran eight seeds of ONE recipe: seeds
2001-2004 in TWO processes (0 -> 5M, then a fresh process 5M -> 15M) and 2005-2008 in ONE. At a
matched step the two groups share the recipe, the self-play regime and the pool size (it grows with
the step), and differ in the process's age by ~1 h (the legged seed's process restarted at 5M). If
a long-lived process slowed itself (allocator fragmentation, a leak, a growing cache) the
single-process seeds would be slower at the same step.

Second, an OLS over every steady self-play cycle of the long production-size runs:
``collect_s_per_100k ~ run + ep_len + pool_snapshots + process_age_h``, the run as a fixed effect.
Pool size and age are collinear within one process, so the coefficient on age is identified only
by the runs whose process restarted (the legged seeds); it is reported with that caveat.
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze import cycles  # noqa: E402

BIN = 1_000_000
ARMS = ("rb_bf_e_", "rb_bf_eb_")


def _last_before(series, step):
    v = None
    for s, _, x in series:
        if s <= step:
            v = x
        else:
            break
    return v


def run_rows(run: str, r: dict) -> list[dict]:
    rows = []
    procs = [p for p in r["procs"] if "inherited" not in p["file"]
             and p["series"].get("rollout/collect_ms")]
    for k, p in enumerate(procs):
        cm = p["series"]["rollout/collect_ms"]
        t0 = cm[0][1] - cm[0][2] / 1000
        s = p["series"]
        for c in cycles(p):
            if c["eval"]:
                continue
            sp = _last_before(s.get("train/selfplay_fraction", []), c["step"])
            rows.append({"run": run, "proc": k, "step": c["step"], "age_h": (c["wall"] - t0) / 3600,
                         "collect_100k": c["collect_s"] / c["ds"] * 1e5, "train_s": c["train_s"],
                         "fps": c["fps"],
                         "ep_len": _last_before(s.get("rollout/ep_len_mean", []), c["step"]),
                         "pool": _last_before(s.get("eval/pool_snapshot_count", []), c["step"]),
                         "sp": sp,
                         "gpu_wait": _last_before(s.get("rust_env/gpu_wait_ms_per_host_step", []), c["step"]),
                         "core": _last_before(s.get("rust_env/core_ms_per_host_step", []), c["step"])})
    # the pool count of a legged seed's second process is logged at its first eval; carry it back
    return rows


def matched(cache: dict) -> dict:
    out = {}
    for arm in ARMS:
        legged = [f"{arm}s{n}" for n in (2001, 2002, 2003, 2004)]
        single = [f"{arm}s{n}" for n in (2005, 2006, 2007, 2008)]
        rows = {g: [x for run in runs if run in cache for x in run_rows(run, cache[run])]
                for g, runs in (("legged", legged), ("single", single))}
        bins = []
        for b in range(5, 15):
            lo, hi = b * BIN, (b + 1) * BIN
            rec = {"step_bin_M": f"{b}-{b + 1}"}
            for g, rs in rows.items():
                sel = [x for x in rs if lo <= x["step"] < hi]
                if not sel:
                    continue
                rec[g] = {"n": len(sel), "age_h": round(st.median(x["age_h"] for x in sel), 2),
                          "collect_100k": round(st.median(x["collect_100k"] for x in sel), 2),
                          "train_s": round(st.median(x["train_s"] for x in sel), 2),
                          "fps": round(st.median(x["fps"] for x in sel)),
                          "gpu_wait": round(st.median(x["gpu_wait"] for x in sel), 2),
                          "core": round(st.median(x["core"] for x in sel), 2),
                          "ep_len": round(st.median(x["ep_len"] for x in sel), 1)}
            if "legged" in rec and "single" in rec:
                rec["single_minus_legged_collect_pct"] = round(
                    100 * (rec["single"]["collect_100k"] / rec["legged"]["collect_100k"] - 1), 1)
                rec["single_minus_legged_fps_pct"] = round(
                    100 * (rec["single"]["fps"] / rec["legged"]["fps"] - 1), 1)
            bins.append(rec)
        ds = [b["single_minus_legged_fps_pct"] for b in bins if "single_minus_legged_fps_pct" in b]
        out[arm] = {"bins": bins, "fps_pct_median_over_bins": round(st.median(ds), 1) if ds else None}
    return out


def ols(cache: dict, prefixes=("rb_bf_e_", "rb_bf_eb_", "rb_es_end2_", "rb_es_prod_", "rb_st_",
                               "rb_x5ab_", "rb_x26_")) -> dict:
    rows = [x for run, r in cache.items() if run.startswith(prefixes)
            for x in run_rows(run, r)
            if x["sp"] is not None and x["sp"] >= 0.85 and x["pool"] is not None and x["ep_len"]]
    runs = sorted({x["run"] for x in rows})
    if not rows:
        return {}
    X = np.array([[1.0 if x["run"] == r else 0.0 for r in runs] + [x["ep_len"], x["pool"], x["age_h"]]
                  for x in rows])
    y = np.array([x["collect_100k"] for x in rows])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = len(y) - X.shape[1]
    sigma2 = float(resid @ resid) / dof
    cov = sigma2 * np.linalg.pinv(X.T @ X)
    names = ["ep_len", "pool", "age_h"]
    k = len(runs)
    # cycles within a run are autocorrelated: the plain SE is optimistic; report it as such
    return {"n_cycles": len(rows), "n_runs": len(runs), "mean_collect_100k": round(float(y.mean()), 2),
            "coef": {n: {"b": round(float(beta[k + i]), 4), "se_iid": round(float(np.sqrt(cov[k + i, k + i])), 4)}
                     for i, n in enumerate(names)},
            "caveat": "iid SEs over autocorrelated cycles: optimistic by a large factor"}


def main(argv):
    cache = json.loads(Path(argv[0]).read_text())
    res = {"matched_step": matched(cache), "ols_selfplay_cycles": ols(cache)}
    Path(argv[1]).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["ols_selfplay_cycles"], indent=1))
    for arm, m in res["matched_step"].items():
        print(arm, "median single-vs-legged fps %:", m["fps_pct_median_over_bins"])
        for b in m["bins"]:
            lg, sg = b.get("legged", {}), b.get("single", {})
            print(f"  {b['step_bin_M']:>6}M  legged age {lg.get('age_h')} c {lg.get('collect_100k')} fps {lg.get('fps')} gw {lg.get('gpu_wait')} core {lg.get('core')} | "
                  f"single age {sg.get('age_h')} c {sg.get('collect_100k')} fps {sg.get('fps')} gw {sg.get('gpu_wait')} core {sg.get('core')} | "
                  f"dfps {b.get('single_minus_legged_fps_pct')}%")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
