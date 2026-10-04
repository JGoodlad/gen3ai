"""TOP KERNELS + ROOFLINE placement for one phase of an nsys capture (bottleneck profile, 2026-10-03).

    python roofline.py --db r2win.sqlite --phase update --compile-cache <run>/compile_cache \
        --torch-rows r2_torch_rows.json --ceilings gpu_ceilings.json --out-json top.json --out-png roofline.png

Per kernel NAME inside the phase's NVTX windows: count, total, mean, share of the phase's kernel time.
Arithmetic: bytes for an Inductor Triton kernel = its own ``kernel_num_gb`` (Inductor's count of the bytes the
kernel must move, emitted with ``TORCHINDUCTOR_BENCHMARK_KERNEL=1``; NOT a counter reading — `ncu` is not
available: the driver restricts GPU performance counters to admin, RmProfilingAdminOnly=1). FLOPs for a
GEMM = the torch profiler's own count for the aten matmul ops (record_shapes + with_flops), matched by the
cuBLAS kernel's per-call time; attention FLOPs from the SDPA op's shapes (4·B·H·Lq·Lk·D forward, 2.5x that
backward). Everything else is placed by time only. Peaks: the measured ceilings (gpu_ceilings.py) beside the
spec sheet's (NVIDIA GA102 / RTX 3080 Ti: 34.1 TFLOP/s fp32 at the 1665 MHz boost, 912 GB/s GDDR6X).
"""
from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sqlite3
from collections import defaultdict

import numpy as np

SPEC_FP32_TFLOPS = 34.1
SPEC_DRAM_GBS = 912.0


def num_gb_table(cache_dir):
    pat = re.compile(r"'kernel_name': '([^']+)'.*?'kernel_num_gb': ([0-9.eE+-]+)")
    tab = defaultdict(set)
    for root, _, files in os.walk(cache_dir):
        for f in files:
            if not f.endswith(".py"):
                continue
            try:
                txt = open(os.path.join(root, f), errors="replace").read()
            except OSError:
                continue
            for m in pat.finditer(txt):
                tab[m.group(1)].add(float(m.group(2)))
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("--phase", default="update")
    ap.add_argument("--compile-cache", required=True)
    ap.add_argument("--torch-rows", default=None)
    ap.add_argument("--ceilings", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-png", default=None)
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()
    ceil = json.load(open(a.ceilings))
    c = sqlite3.connect(a.db)
    S = dict(c.execute("select id, value from StringIds"))
    wins = [(s, e) for s, e, t, tid in c.execute("select start, end, text, textId from NVTX_EVENTS where end is not null")
            if (t or S.get(tid)) == a.phase]
    K = np.array(c.execute("select start, end, demangledName from CUPTI_ACTIVITY_KIND_KERNEL").fetchall(), dtype=np.int64)
    K = K[np.argsort(K[:, 0])]
    agg = defaultdict(lambda: [0, 0])
    tot = 0
    wall = 0
    for s, e in wins:
        i0, i1 = np.searchsorted(K[:, 0], [s, e])
        k = K[i0:i1]
        wall += e - s
        for nm, d in zip(k[:, 2], k[:, 1] - k[:, 0]):
            agg[int(nm)][0] += 1
            agg[int(nm)][1] += int(d)
            tot += int(d)
    nwin = max(1, len(wins))
    gb = num_gb_table(a.compile_cache)
    # GEMM flops from torch rows: per aten mm-like op, flops per call and self cuda us per call
    gemm_rows = []
    attn_rows = []
    if a.torch_rows:
        for r in json.load(open(a.torch_rows)):
            n = r["name"]
            if n in ("aten::mm", "aten::addmm", "aten::bmm", "aten::baddbmm") and r["flops"] and r["self_cuda_us"]:
                gemm_rows.append({"op": n, "shapes": r["shapes"], "count": r["count"], "flops_per_call": r["flops"] / r["count"],
                                  "cuda_us_per_call": r["self_cuda_us"] / r["count"],
                                  "tflops": r["flops"] / (r["self_cuda_us"] * 1e-6) / 1e12})
            if "efficient_attention" in n or "flash_attention" in n:
                attn_rows.append({"op": n, "shapes": r["shapes"], "count": r["count"], "self_cuda_us": r["self_cuda_us"]})
    rows = []
    for nm, (cnt, d) in sorted(agg.items(), key=lambda kv: -kv[1][1]):
        name = S.get(nm, str(nm))
        short = name.split("(")[0][:120]
        mean_s = d / cnt / 1e9
        row = {"name": short, "calls_per_window": cnt / nwin, "total_s_per_window": d / 1e9 / nwin,
               "mean_us": mean_s * 1e6, "share_of_kernel_time": d / max(1, tot)}
        g = gb.get(short)
        if g:
            if len(g) == 1:
                v = next(iter(g))
                row["num_gb"] = v
                row["achieved_GBs"] = v / mean_s
                row["frac_of_measured_dram"] = row["achieved_GBs"] / ceil["dram_copy_gbs"]
            else:
                row["num_gb_ambiguous"] = sorted(g)
        rows.append(row)
    top = rows[:a.top]
    out = {"phase": a.phase, "windows": len(wins), "wall_s_per_window": wall / 1e9 / nwin,
           "kernel_s_per_window": tot / 1e9 / nwin, "distinct_kernels": len(rows), "top": top,
           "triton_bytes_coverage": {
               "kernel_time_with_num_gb": sum(r["total_s_per_window"] for r in rows if "num_gb" in r),
               "weighted_frac_of_measured_dram": (sum(r["total_s_per_window"] * r["frac_of_measured_dram"] for r in rows if "num_gb" in r)
                                                  / max(1e-12, sum(r["total_s_per_window"] for r in rows if "num_gb" in r)))},
           "gemm_ops": sorted(gemm_rows, key=lambda r: -r["cuda_us_per_call"] * r["count"])[:20],
           "attention_ops": sorted(attn_rows, key=lambda r: -r["self_cuda_us"])[:8],
           "peaks": {"spec_fp32_tflops": SPEC_FP32_TFLOPS, "spec_dram_GBs": SPEC_DRAM_GBS,
                     "measured_fp32_gemm_tflops": max(v["tflops"] for v in ceil["gemm_fp32"].values()),
                     "measured_dram_GBs": ceil["dram_copy_gbs"]}}
    if gemm_rows:
        tf = sum(r["flops_per_call"] * r["count"] for r in gemm_rows)
        tt = sum(r["cuda_us_per_call"] * r["count"] for r in gemm_rows) * 1e-6
        out["gemm_aggregate"] = {"tflops_achieved": tf / tt / 1e12, "gemm_s": tt, "tflop": tf / 1e12}
    json.dump(out, open(a.out_json, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("gemm_ops", "attention_ops")}, indent=1)[:5000])
    if a.out_png:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        P = out["peaks"]["measured_fp32_gemm_tflops"] * 1e12
        B = out["peaks"]["measured_dram_GBs"] * 1e9
        fig, ax = plt.subplots(figsize=(7.5, 5))
        ai = np.logspace(-2, 3, 200)
        ax.loglog(ai, np.minimum(P, B * ai) / 1e12, color="#444", lw=1.5, label="measured roof (fp32 GEMM, D2D copy)")
        ax.loglog(ai, np.minimum(SPEC_FP32_TFLOPS * 1e12, SPEC_DRAM_GBS * 1e9 * ai) / 1e12, color="#aaa", lw=1, ls="--",
                  label="spec roof (34.1 TFLOP/s, 912 GB/s)")
        # GEMMs: FLOPs known, bytes approximated as (MK + KN + MN) * 4 from shapes
        for r in out["gemm_ops"][:12]:
            try:
                sh = ast.literal_eval(r["shapes"])
                mats = [s for s in sh if isinstance(s, list) and len(s) >= 2]
                byts = sum(int(np.prod(s)) for s in mats) * 4
                f = r["flops_per_call"]
                outb = f / 2 / max(1, mats[0][-1]) * 4 if mats else 0
                x = f / max(1, byts + outb)
                ax.scatter([x], [r["tflops"]], s=12 + 300 * r["cuda_us_per_call"] * r["count"] * 1e-6 / max(1e-9, out["kernel_s_per_window"] * nwin),
                           color="#d1495b", alpha=0.7)
            except Exception:
                continue
        # Triton kernels: bytes known, FLOPs unknown -> placed on the bandwidth axis at AI = 0.25 (pointwise fp32)
        for r in rows[:60]:
            if "achieved_GBs" in r:
                x = 0.25
                ax.scatter([x * (1 + 0.6 * np.random.rand())], [r["achieved_GBs"] * 1e9 * x / 1e12],
                           s=12 + 300 * r["share_of_kernel_time"], color="#00798c", alpha=0.6)
        ax.scatter([], [], color="#d1495b", label="GEMM ops (FLOPs from torch profiler)")
        ax.scatter([], [], color="#00798c", label="Triton kernels (bytes from Inductor; AI ~0.25 assumed)")
        ax.set_xlabel("arithmetic intensity (FLOP / byte)")
        ax.set_ylabel("achieved TFLOP/s")
        ax.set_title(f"RTX 3080 Ti, fp32 — top kernels of the {a.phase} (marker area = time share)")
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(True, which="both", alpha=0.2)
        fig.tight_layout()
        fig.savefig(a.out_png, dpi=130)


if __name__ == "__main__":
    main()
