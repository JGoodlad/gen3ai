"""M5 T2 — the inference service's THROUGHPUT benchmark (a measurement, run as a script).

Builds a service from a checkpoint (or a perturbed fresh production-arch policy), reports the
startup phases (allocate / build = compile + capture / parity), then per bucket the time of one
flush that fills it exactly, and a mixed flush across slots. A benchmark's output IS the
measurement: it WARNS on a busy box (another GPU process, a high load average) and never stretches.

    export PYTHONPATH=$PYTHONPATH:src
    python3 src/agents/inference/service/service_benchmark.py --backend graph --buckets 8,48,128 \\
        --slots 2 [--ckpt models/<run>/final_model.zip]

Compile cache (K3): a FRESH private Inductor/Triton cache per run, deleted at exit, unless a caller
declared one (`agents.model.compile_cache.ensure_hermetic_cache`) — pass a shared dir by exporting
`GEN3AI_COMPILE_CACHE_DIR` + `TORCHINDUCTOR_CACHE_DIR` to measure a WARM start.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from typing import Any, Dict, List


def _load(ckpt: str) -> Any:
    from agents.model.snapshot import current_model_version, load_foreign_opponent
    from agents.observation.state_encoder import load_mappings

    model, _ = load_foreign_opponent(ckpt, current_version=current_model_version(load_mappings()),
                                     device="cpu")
    return model.policy.eval()


def _busy_box_warnings() -> List[str]:
    out = []
    load1 = os.getloadavg()[0]
    if load1 > (os.cpu_count() or 1) * 0.5:
        out.append(f"load average {load1:.1f} on {os.cpu_count()} cores")
    try:
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
        others = [p for p in q.stdout.split() if p.strip() and int(p) != os.getpid()]
        if others:
            out.append(f"other GPU processes: {others}")
    except Exception as exc:                       # no nvidia-smi: say so, never assume idle
        out.append(f"GPU occupancy unknown ({type(exc).__name__})")
    return out


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default=None, help="a checkpoint zip (default: perturbed fresh)")
    ap.add_argument("--backend", default="graph", choices=("eager", "graph", "aot"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--buckets", default="8,48,128")
    ap.add_argument("--slots", type=int, default=2)
    ap.add_argument("--lanes", type=int, default=1, help="concurrency lanes (CUDA streams)")
    ap.add_argument("--opponent-rows", type=int, default=48,
                    help="Lane E's shape: this many rows spread evenly over every slot per flush")
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--json", default=None, help="also write the result here")
    args = ap.parse_args(argv)

    from agents.model.compile_cache import ensure_hermetic_cache
    ensure_hermetic_cache("service_benchmark")              # K3: fresh, private, deleted at exit
    import numpy as np
    import torch

    from agents.inference.service import InferenceService, ServiceSpec, SlotGroupSpec
    from agents.inference.service.parity import fixture_rows
    from agents.inference.service.fixtures import perturbed_fresh_policy as _perturbed_policy

    warnings = _busy_box_warnings()
    policy = _load(args.ckpt) if args.ckpt else _perturbed_policy(0)
    buckets = tuple(int(b) for b in args.buckets.split(","))
    spec = ServiceSpec(groups=(SlotGroupSpec("bench", args.slots, policy),), device=args.device,
                       backend=args.backend, buckets=buckets,
                       max_rows_per_flush=max(1024, buckets[-1] * args.slots), lanes=args.lanes)
    t0 = time.perf_counter()
    svc = InferenceService(spec).startup()
    result: Dict[str, Any] = {
        "torch": torch.__version__, "backend": args.backend, "device": args.device,
        "ckpt": args.ckpt or "fresh-perturbed(seed 0)", "slots": args.slots, "buckets": buckets,
        "lanes": args.lanes,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "startup_s": round(time.perf_counter() - t0, 1), "startup_phases_s":
        {k: round(v, 1) for k, v in svc.startup_seconds.items()},
        "parity_worst": {"legal_logprob": max(r.legal_logprob_max for r in svc.startup_reports),
                         "value": max(r.value_max for r in svc.startup_reports)},
        "busy_box_warnings": warnings, "per_bucket": {}}
    obs, mask = fixture_rows(svc.obs_dim, max(buckets) * args.slots)

    def timed(fn: Any) -> float:
        for _ in range(3):
            fn()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        t = time.perf_counter()
        for _ in range(args.reps):
            fn()
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        return (time.perf_counter() - t) / args.reps * 1e3

    for b in buckets:
        def one() -> None:
            svc.submit(0, obs[:b], mask[:b])
            svc.flush()
        ms = timed(one)
        result["per_bucket"][b] = {"ms_per_flush": round(ms, 3), "us_per_row": round(ms / b * 1e3, 1)}

    def mixed() -> None:                       # every slot, a ragged row count each
        for s in range(args.slots):
            n = int(np.clip(buckets[-1] // (s + 2), 1, None))
            svc.submit(s, obs[:n], mask[:n])
        svc.flush()
    result["mixed_flush_ms"] = round(timed(mixed), 3)

    per = [args.opponent_rows // args.slots + (1 if s < args.opponent_rows % args.slots else 0)
           for s in range(args.slots)]

    def lane_e() -> None:                      # Lane E: N opponent rows over every slot
        for s, n in enumerate(per):
            if n:
                svc.submit(s, obs[:n], mask[:n])
        svc.flush()
    result["lane_e_flush_ms"] = round(timed(lane_e), 3)
    result["lane_e_rows_per_slot"] = per
    st = svc.stats()
    result["after_freeze"] = {k: st[k] for k in ("compiles_after_freeze", "captures_after_freeze",
                                                  "cuda_segments_after_freeze")}
    if args.device.startswith("cuda"):
        result["max_memory_allocated_mib"] = round(torch.cuda.max_memory_allocated() / 2**20)
    text = json.dumps(result, indent=2)
    print(text)
    if warnings:
        print("⚠️  BUSY BOX — the numbers above are NOT a quiet-box measurement: " + "; ".join(warnings))
    if args.json:
        with open(args.json, "w") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
