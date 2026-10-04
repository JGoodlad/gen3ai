"""Join the four launch reads (blob a/b, fixed_mass a/b) into the numbers the README quotes.

    python summarize.py [--dir results/raw] [--results results] [--out results/numbers.json] [--check]

Inputs are all COMMITTED beside the script: `results/x5{blob,fm}_{a,b}.json` (`read_run.py`'s per-launch read, taken
from the launch's child log, TensorBoard and nvidia-smi in a job-scratch directory that is not kept) and the
phase / CPU-sampler rows `results/raw/<launch>.{phases,cpu}.jsonl`. `--check` resolves them and exits non-zero on a
missing one (the measurements readout gate, `src/measurements_readout_gate_test.py`).

Statistics: the steady updates are the QUIET ones (windowed contention factor < 1.05, warm-up and the
compile-canary update excluded — `read_run.py`). Median per arm over the pooled quiet updates, a
bootstrap 95 % CI of each median and of the DIFFERENCE (fixed_mass - blob) over the pooled rows
(`bottleneck_profile_2026-10-03/scripts/update_ab.py`'s resampler), and each launch's own median so the
launch-to-launch spread is visible (the pooled-row CI treats updates as independent; they are
autocorrelated within a launch, so the per-launch medians are the honest spread). The rollout (play)
wall per cycle comes from the `collect` phase markers, the same contention rule.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics as st
import sys
from pathlib import Path


def boot_median(x, n=10000, seed=0):
    rng = random.Random(seed)
    m = sorted(st.median(rng.choices(x, k=len(x))) for _ in range(n))
    return m[int(0.025 * n)], m[int(0.975 * n)]


def boot_diff(a, b, n=10000, seed=1):
    rng = random.Random(seed)
    d = sorted(st.median(rng.choices(a, k=len(a))) - st.median(rng.choices(b, k=len(b))) for _ in range(n))
    return d[int(0.025 * n)], d[int(0.975 * n)]


def play_walls(phases_path, cpu_path):
    ph = [json.loads(line) for line in open(phases_path)]
    col = sorted([r for r in ph if r["phase"] == "collect"], key=lambda r: r["t0"])
    samp = [json.loads(line) for line in open(cpu_path) if line.strip()]
    samp = [s for s in samp if "t" in s]
    out = []
    for i, c in enumerate(col):
        if i < 2:
            continue
        f = [s["factor"] for s in samp if c["t0"] <= s["t"] - s["dt"] and s["t"] <= c["t1"]]
        if not f or max(f) >= 1.05:
            continue
        out.append(round(c["t1"] - c["t0"], 3))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent.parent
    ap.add_argument("--dir", default=str(here / "results" / "raw"))
    ap.add_argument("--results", default=str(here / "results"))
    ap.add_argument("--out", default=str(here / "results" / "numbers.json"))
    ap.add_argument("--check", action="store_true", help="resolve every input, print the missing ones, compute nothing")
    a = ap.parse_args()
    d, rdir = Path(a.dir), Path(a.results)
    arms = {"blob": ["x5blob_a", "x5blob_b"], "fixed_mass": ["x5fm_a", "x5fm_b"]}
    need = [rdir / f"{t}.json" for ts in arms.values() for t in ts] + \
           [d / f"{t}.{k}.jsonl" for ts in arms.values() for t in ts for k in ("phases", "cpu")]
    missing = [str(p) for p in need if not p.is_file()]
    if a.check:
        print("\n".join(f"MISSING {m}" for m in missing) or f"all {len(need)} inputs resolve")
        sys.exit(1 if missing else 0)
    if missing:
        sys.exit("missing inputs:\n" + "\n".join(missing))
    runs = {t: json.load(open(rdir / f"{t}.json")) for ts in arms.values() for t in ts}
    out = {"launches": {}, "arms": {}}
    for t, r in runs.items():
        out["launches"][t] = {
            "startup_s_start_to_first_collect": r["startup_wall_s_start_to_first_collect"],
            "T2_up_s": float(r["T2_up_s"]), "R1_reset_prewarm_s": float(r["R1_prewarm_s"]),
            "startup_contended_windows": f'{r.get("startup_windows_ge_1.05")}/{r.get("startup_windows")}',
            "startup_max_contention_factor": round(r.get("startup_max_contention_factor") or 0, 2),
            "dry_update_s": r["updates"][0]["dur_s"],
            "update_steady_n": r["update_wall_s_steady"]["n"], "update_steady_median_s": r["update_wall_s_steady"]["median"],
            "update_steady_values": r["update_wall_s_steady"]["values"],
            "T2_flush_host_ms_per_step": r["tb"]["rust_env/flush_ms_per_host_step"]["median_steady"],
            "T2_gpu_wait_ms_per_step": r["tb"]["rust_env/gpu_wait_ms_per_host_step"]["median_steady"],
            "update_headroom_line": r["dry_update_line"],
            "cuda_device_free_mib": r["tb"]["lifecycle/cuda_device_free_mib"]["median_steady"],
            "cuda_update_peak_reserved_mib": r["tb"]["lifecycle/cuda_update_peak_reserved_mib"]["median_steady"],
            "cuda_update_peak_alloc_mib": r["tb"]["lifecycle/cuda_update_peak_alloc_mib"]["median_steady"],
            "smi_memory_used_mib_max": r["smi_memory_used_mib_max"],
            "graphs_total": r["tb"]["compile/graphs_total"]["median_steady"],
            "recompiles_after_lock_max": max(r["tb"]["compile/recompiles_after_lock"]["all"]),
            "cache_limit_hits_max": max(r["tb"]["compile/cache_limit_hits"]["all"]),
            "play_wall_s_quiet": play_walls(d / f"{t}.phases.jsonl", d / f"{t}.cpu.jsonl"),
        }
    for arm, ts in arms.items():
        ups = [v for t in ts for v in out["launches"][t]["update_steady_values"]]
        pw = [v for t in ts for v in out["launches"][t]["play_wall_s_quiet"]]
        out["arms"][arm] = {
            "update_n": len(ups), "update_median_s": st.median(ups), "update_median_ci95": boot_median(ups),
            "play_n": len(pw), "play_median_s": st.median(pw) if pw else None,
            "T2_gpu_wait_ms": [out["launches"][t]["T2_gpu_wait_ms_per_step"] for t in ts],
            "T2_flush_host_ms": [out["launches"][t]["T2_flush_host_ms_per_step"] for t in ts],
            "headroom_mib": [out["launches"][t]["cuda_device_free_mib"] for t in ts],
            "peak_reserved_mib": [out["launches"][t]["cuda_update_peak_reserved_mib"] for t in ts],
            "T2_up_s": [out["launches"][t]["T2_up_s"] for t in ts],
            "R1_reset_prewarm_s": [out["launches"][t]["R1_reset_prewarm_s"] for t in ts],
            "startup_s": [out["launches"][t]["startup_s_start_to_first_collect"] for t in ts],
        }
    b, f = out["arms"]["blob"], out["arms"]["fixed_mass"]
    ub = [v for t in arms["blob"] for v in out["launches"][t]["update_steady_values"]]
    uf = [v for t in arms["fixed_mass"] for v in out["launches"][t]["update_steady_values"]]
    out["delta"] = {
        "update_median_s": f["update_median_s"] - b["update_median_s"],
        "update_median_pct": 100 * (f["update_median_s"] / b["update_median_s"] - 1),
        "update_median_diff_ci95_s": boot_diff(uf, ub),
        "update_median_diff_ci95_pct": tuple(100 * x / b["update_median_s"] for x in boot_diff(uf, ub)),
        "per_launch_update_median_pct": [100 * (out["launches"][ft]["update_steady_median_s"] /
                                                out["launches"][bt]["update_steady_median_s"] - 1)
                                         for ft, bt in zip(arms["fixed_mass"], arms["blob"])],
        "T2_gpu_wait_pct": 100 * (st.mean(f["T2_gpu_wait_ms"]) / st.mean(b["T2_gpu_wait_ms"]) - 1),
        "T2_gpu_wait_ms": st.mean(f["T2_gpu_wait_ms"]) - st.mean(b["T2_gpu_wait_ms"]),
        "T2_flush_plus_wait_pct": 100 * ((st.mean(f["T2_gpu_wait_ms"]) + st.mean(f["T2_flush_host_ms"])) /
                                         (st.mean(b["T2_gpu_wait_ms"]) + st.mean(b["T2_flush_host_ms"])) - 1),
        "headroom_mib": st.mean(f["headroom_mib"]) - st.mean(b["headroom_mib"]),
        "peak_reserved_mib": st.mean(f["peak_reserved_mib"]) - st.mean(b["peak_reserved_mib"]),
        "T2_up_s": st.mean(f["T2_up_s"]) - st.mean(b["T2_up_s"]),
        "R1_reset_prewarm_s": st.mean(f["R1_reset_prewarm_s"]) - st.mean(b["R1_reset_prewarm_s"]),
        "startup_s": st.mean(f["startup_s"]) - st.mean(b["startup_s"]),
    }
    if b["play_median_s"] and f["play_median_s"]:
        out["delta"]["play_median_pct"] = 100 * (f["play_median_s"] / b["play_median_s"] - 1)
    Path(a.out).write_text(json.dumps(out, indent=1, default=list))
    print(json.dumps(out["arms"], indent=1, default=list))
    print(json.dumps(out["delta"], indent=1, default=list))


if __name__ == "__main__":
    main()
