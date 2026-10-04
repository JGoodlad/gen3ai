"""The README's two figures from the committed result JSONs (bottleneck profile, 2026-10-03).

    python make_figures.py   # writes ../roofline.png and ../cycle_share.png

(a) ROOFLINE (fp32, TF32 off): the measured roof (best fp32 GEMM 24.9 TFLOP/s, D2D copy 825 GB/s;
gpu_ceilings.json) and the spec roof (34.1 TFLOP/s, 912 GB/s). Points: the update's GEMMs in aggregate
(150.3 TFLOP per update from the torch profiler's shapes, 13.2 s of GEMM kernel time from nsys, minimum
traffic 4.53 TB from the shapes), the two SDPA (memory-efficient attention) kernels (FLOPs and minimum bytes
from the op shapes, time from nsys), and the top Inductor Triton kernels (bytes = Inductor's own
`kernel_num_gb`; FLOPs are not counted for them, so they are drawn as achieved BANDWIDTH against the DRAM
roof in panel b). (c) the training cycle's wall shares.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent.parent
RES = HERE / "results"


def main():
    ceil = json.load(open(RES / "gpu_ceilings.json"))
    top = json.load(open(RES / "top_kernels_update.json"))
    P = max(v["tflops"] for v in ceil["gemm_fp32"].values())
    B = ceil["dram_copy_gbs"]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={"width_ratios": [1.15, 1]})
    ai = np.logspace(-1, 3, 200)
    ax.loglog(ai, np.minimum(P, B * ai / 1e3), color="#333", lw=1.6, label=f"measured roof ({P:.1f} TFLOP/s, {B:.0f} GB/s)")
    ax.loglog(ai, np.minimum(34.1, 912 * ai / 1e3), color="#999", lw=1, ls="--", label="spec roof (34.1 TFLOP/s, 912 GB/s)")
    pts = [  # (label, AI flop/byte, achieved TFLOP/s, share of update kernel time)
        ("GEMMs, all (fp32 SIMT sgemm)", 150.34e12 / 4.528e12, 150.34 / 13.20, 0.347),
        ("SDPA forward (L=61, d=32, bias)", 3.90e9 / 0.378e9, 3.90e9 / 840.5e-6 / 1e12, 0.043),
        ("SDPA backward", 9.75e9 / 0.756e9, 9.75e9 / 2013.1e-6 / 1e12, 0.102),
    ]
    colors = ["#c0392b", "#8e44ad", "#8e44ad"]
    for (lab, x, y, sh), c in zip(pts, colors):
        ax.scatter([x], [y], s=60 + 1500 * sh, color=c, alpha=0.75, edgecolor="k", lw=0.5)
        ax.annotate(f"{lab}\n{y:.1f} TFLOP/s, {100 * sh:.0f}% of kernel time", (x, y), textcoords="offset points",
                    xytext=(8, -18 if "back" in lab else 6), fontsize=7)
    ax.set_xlabel("arithmetic intensity (FLOP / byte, minimum traffic)")
    ax.set_ylabel("achieved TFLOP/s")
    ax.set_title("(a) update, fp32: GEMMs + attention on the roofline", fontsize=9)
    ax.set_ylim(0.3, 60)
    ax.grid(True, which="both", alpha=0.2)
    ax.legend(fontsize=7, loc="lower right")
    tri = [r for r in top["top"] if "achieved_GBs" in r]
    names = [r["name"][:44] + ("…" if len(r["name"]) > 44 else "") for r in tri]
    vals = [r["achieved_GBs"] for r in tri]
    shares = [r["share_of_kernel_time"] for r in tri]
    y = np.arange(len(tri))
    bx.barh(y, vals, color="#16a085")
    bx.axvline(B, color="#333", lw=1.2, label=f"measured DRAM copy {B:.0f} GB/s")
    bx.axvline(912, color="#999", ls="--", lw=1, label="spec 912 GB/s")
    for i, (v, s) in enumerate(zip(vals, shares)):
        bx.text(v + 10, i, f"{v:.0f} GB/s · {100 * s:.1f}%", va="center", fontsize=7)
    bx.set_yticks(y, names, fontsize=6.5)
    bx.invert_yaxis()
    bx.set_xlim(0, 1100)
    bx.set_xlabel("achieved bandwidth (Inductor kernel_num_gb / mean duration)")
    cov = top["triton_bytes_coverage"]
    bx.set_title(f"(b) top Triton kernels vs the DRAM roof\n(all byte-counted Triton kernels: "
                 f"{cov['kernel_time_with_num_gb']:.1f} s/update at {100 * cov['weighted_frac_of_measured_dram']:.0f}% of the roof, time-weighted)",
                 fontsize=8)
    bx.legend(fontsize=7, loc="lower right")
    fig.tight_layout()
    fig.savefig(HERE / "roofline.png", dpi=130)

    # (c) cycle shares (numbers from the README's phase table)
    parts = [("update (PPO, 10 epochs x 48 micro-batches)", 40.0, "#c0392b"),
             ("T2 inference (flush + GPU wait)", 0.0215 * 400, "#2980b9"),
             ("Rust core step", 0.0075 * 400, "#27ae60"),
             ("host glue + fill", 0.0021 * 400 + 0.24, "#f39c12"),
             ("eval (amortised, 1 per 2M steps)", 0.96, "#7f8c8d")]
    tot = sum(p[1] for p in parts)
    fig, cx = plt.subplots(figsize=(10, 1.9))
    left = 0.0
    for lab, v, c in parts:
        cx.barh([0], [v], left=left, color=c, edgecolor="white")
        if v / tot > 0.04:
            cx.text(left + v / 2, 0, f"{lab}\n{v:.1f} s · {100 * v / tot:.0f}%", ha="center", va="center", fontsize=7, color="white")
        left += v
    cx.set_xlim(0, tot)
    cx.set_yticks([])
    cx.set_xlabel(f"seconds per training cycle (98,304 rows; total {tot:.1f} s)")
    cx.set_title("training cycle at N = 256, production arch, 20-snapshot self-play pool (steady state)", fontsize=9)
    fig.tight_layout()
    fig.savefig(HERE / "cycle_share.png", dpi=130)


if __name__ == "__main__":
    main()
