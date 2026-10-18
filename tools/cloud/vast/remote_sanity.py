"""The torch / CUDA SANITY on a rented GPU (runs ON the box). SANITY, NOT A BENCHMARK: it proves the stack works.

Checks (each recorded with its wall time in the --out JSON; the process exits 1 if any FAILS):
- the build: torch version, CUDA build, cuDNN, ``torch.cuda.get_arch_list()`` and that it holds a kernel the device
  can RUN (``arch_runs``): the device's own ``sm_<major><minor>``, or a cubin of the SAME major and a lower-or-equal
  minor (CUDA's binary compatibility: an RTX 4090 is sm_89 and runs the cu126 wheel's sm_86 cubins). A 5090 is
  sm_120 and the cu126 wheels stop at sm_90 (major 9), so for Blackwell this is still THE check;
- an fp32 4096^2 matmul on the GPU against the CPU result (max abs diff, relative to the CPU norm);
- a tiny ``torch.compile`` (Inductor -> Triton) of an MLP, its output against eager, first-call (compile) time;
- a bf16 matmul (the future speed lever) runs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time


def arch_runs(cap: tuple, arches: list) -> bool:
    """True when ``arches`` (``torch.cuda.get_arch_list()``) holds a kernel a device of compute capability ``cap`` runs:
    its own ``sm_XY`` or an ``sm_`` cubin of the same major with minor <= the device's (pure)."""
    major, minor = int(cap[0]), int(cap[1])
    for a in arches:
        if not a.startswith("sm_"):
            continue
        digits = a[3:]
        am, an = int(digits[:-1]), int(digits[-1])
        if am == major and an <= minor:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch

    res = {"torch": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
           "checks": {}}
    ok = True

    def check(name, fn):
        nonlocal ok
        t0 = time.perf_counter()
        try:
            out = fn()
            passed = bool(out.pop("pass", True))
        except Exception as e:  # noqa: BLE001 — record, never hide
            out, passed = {"error": repr(e)[:2000]}, False
        out["pass"], out["seconds"] = passed, round(time.perf_counter() - t0, 3)
        res["checks"][name] = out
        ok = ok and passed
        print(f"[sanity] {name}: {'PASS' if passed else 'FAIL'} {out}", flush=True)

    def build():
        cap = torch.cuda.get_device_capability(0)
        sm = f"sm_{cap[0]}{cap[1]}"
        arches = torch.cuda.get_arch_list()
        return {"device": torch.cuda.get_device_name(0), "capability": sm, "arch_list": arches,
                "cudnn": torch.backends.cudnn.version(), "compiled_cuda": torch._C._cuda_getCompiledVersion(),
                "vram_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
                "pass": arch_runs(cap, arches)}

    def matmul():
        torch.manual_seed(0)
        x = torch.randn(4096, 4096)
        y = torch.randn(4096, 4096)
        prev = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        try:
            g = (x.cuda() @ y.cuda()).cpu()
        finally:
            torch.backends.cuda.matmul.allow_tf32 = prev
        c = x @ y
        rel = float((g - c).abs().max() / c.abs().max())
        return {"max_rel_diff": rel, "pass": rel < 1e-4}

    def compiled():
        torch.manual_seed(0)
        m = torch.nn.Sequential(torch.nn.Linear(256, 512), torch.nn.GELU(), torch.nn.Linear(512, 64)).cuda()
        x = torch.randn(1024, 256, device="cuda")
        cm = torch.compile(m)
        t0 = time.perf_counter()
        y = cm(x)
        torch.cuda.synchronize()
        first = time.perf_counter() - t0
        diff = float((y - m(x)).abs().max())
        return {"first_call_s": round(first, 2), "max_abs_diff_vs_eager": diff, "pass": diff < 1e-3}

    def bf16():
        x = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
        y = (x @ x).float()
        torch.cuda.synchronize()
        return {"finite": bool(torch.isfinite(y).all()), "pass": bool(torch.isfinite(y).all())}

    if not res["cuda_available"]:
        ok = False
    else:
        check("build_arch", build)
        check("matmul_fp32", matmul)
        check("compile_inductor", compiled)
        check("matmul_bf16", bf16)
    res["pass"] = ok
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    print(f"[sanity] {'ALL PASS' if ok else 'FAILED'}", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
