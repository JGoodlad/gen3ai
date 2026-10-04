"""Measured GPU CEILINGS for the roofline (bottleneck profile, 2026-10-03) — run on an IDLE GPU.

    python gpu_ceilings.py --out ceilings.json

* fp32 GEMM (TF32 OFF: ``torch.backends.cuda.matmul.allow_tf32 = False``), square 8192 and the
  update's own skinny shapes, best of 20 after warm-up -> achieved TFLOP/s.
* DRAM bandwidth: a device-to-device copy of 1 GiB (read + write bytes counted) and an elementwise
  add (2 reads + 1 write) -> achieved GB/s.
* kernel-launch floor: an empty-ish kernel (``x.add_(0)`` on 1 element) launched 20,000 times
  back to back, eager, -> µs per launch (host-side, the launch-latency floor of eager PyTorch);
  and the same 20,000 ops captured in one CUDA graph -> µs per node replayed.
"""
from __future__ import annotations

import argparse
import json
import time

import torch


def _time_cuda(fn, reps):
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(reps):
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        best = min(best, s.elapsed_time(e) / 1e3)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    prev = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        dev = torch.device("cuda")
        out = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__,
               "sm_count": torch.cuda.get_device_properties(0).multi_processor_count}
        gemm = {}
        for (m, k, n) in [(8192, 8192, 8192), (4096, 4096, 4096), (2048 * 6, 256, 256), (2048 * 6, 256, 1024),
                          (2048 * 6, 1024, 256), (2048, 256, 256), (256, 256, 256)]:
            A = torch.randn(m, k, device=dev)
            B = torch.randn(k, n, device=dev)
            for _ in range(3):
                A @ B
            t = _time_cuda(lambda: A @ B, 20)
            gemm[f"{m}x{k}x{n}"] = {"s": t, "tflops": 2 * m * k * n / t / 1e12}
        out["gemm_fp32"] = gemm
        x = torch.empty(256 * 2 ** 20, device=dev)          # 1 GiB fp32
        y = torch.empty_like(x)
        z = torch.empty_like(x)
        t = _time_cuda(lambda: y.copy_(x), 20)
        out["dram_copy_gbs"] = 2 * x.numel() * 4 / t / 1e9
        t = _time_cuda(lambda: torch.add(x, y, out=z), 20)
        out["dram_add_gbs"] = 3 * x.numel() * 4 / t / 1e9
        one = torch.zeros(1, device=dev)
        n = 20000
        for _ in range(1000):
            one.add_(0)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            one.add_(0)
        t_host = time.perf_counter() - t0
        torch.cuda.synchronize()
        t_all = time.perf_counter() - t0
        out["eager_launch_us_host"] = 1e6 * t_host / n
        out["eager_launch_us_wall"] = 1e6 * t_all / n
        g = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        with torch.cuda.stream(s):
            with torch.cuda.graph(g):
                for _ in range(2000):
                    one.add_(0)
        torch.cuda.synchronize()
        t = _time_cuda(g.replay, 10)
        out["graph_node_us"] = 1e6 * t / 2000
        del g
        print(json.dumps(out, indent=1))
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = prev


if __name__ == "__main__":
    main()
