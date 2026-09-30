"""K3 — COLD vs WARM startup cost of the hermetic per-run compile cache (a measurement, run as a script).

    export PYTHONPATH=$PYTHONPATH:src
    scripts/ops/gpu_lock.sh python3 src/agents/model/compile_cache_benchmark.py run --device cuda \\
        --json <out.json> [--rounds 2] [--parts learner,t2,opponent]

What a run's STARTUP compiles, each part timed in a FRESH child process (a restart is a new process:
dynamo's in-memory caches never carry over, only the on-disk cache does):

  * ``learner``  — `compile_trainer_extractor` (the real-obs parity gate: eager vs compiled train /
                   eval graphs) + `arm_compile_sentinel`'s prewarm of every production signature
                   (rollout n_envs, the update's micro-batch, the half-batch), CUDA;
  * ``t2``       — `InferenceService.startup()` at the training shape (a trainee slot + a 20-slot
                   pool group, buckets (8, 48), 8 lanes, backend graph): compile + CUDA-graph capture
                   per slot x bucket + the startup parity gate;
  * ``opponent`` — one env worker's CPU opponent compile (`maybe_compile_extractor`, CUDA hidden).

Per ROUND a new empty cache root is made; COLD = the first process on it (what a FRESH launch pays),
WARM = a second process on the same root (what the run's own RESTART pays). The model is a seeded,
perturbed fresh production-arch policy (weights are graph inputs: they never change the compile).
A benchmark's output IS the measurement — contention is REPORTED, never rescaled.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Sequence

PARTS = ("learner", "t2", "opponent")


def _worker(part: str, root: str, device: str) -> Dict[str, Any]:
    from agents.model import compile_cache as CC
    CC._export(root)                                  # before torch is imported
    if part == "opponent":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    t_import = time.perf_counter()
    import torch

    from agents.inference.service.fixtures import perturbed_fresh_policy
    out: Dict[str, Any] = {"part": part, "torch": torch.__version__}
    if part == "learner":
        from agents.model.compile_trainer import arm_compile_sentinel, compile_trainer_extractor
        from agents.model.parity_probe import perturb_
        from main.fresh_checkpoint import build_fresh_model
        model, _, _ = build_fresh_model(0)
        perturb_(model.policy, seed=1000, scale=0.05)
        model.policy.to(device)
        model.device = torch.device(device)
        t0 = time.perf_counter()
        compile_trainer_extractor(model, True, emit=None)
        out["gate_s"] = time.perf_counter() - t0
        t1 = time.perf_counter()
        arm_compile_sentinel(model, n_envs=48, batch_size=2048, critic="winprob", emit=None)
        out["prewarm_s"] = time.perf_counter() - t1
        out["startup_s"] = out["gate_s"] + out["prewarm_s"]
    elif part == "t2":
        from agents.inference.service import InferenceService, ServiceSpec, SlotGroupSpec
        policy = perturbed_fresh_policy(0)
        spec = ServiceSpec(groups=(SlotGroupSpec("trainee", 1, policy), SlotGroupSpec("pool", 20, policy)),
                           device=device, backend="graph", buckets=(8, 48), lanes=8,
                           max_rows_per_flush=max(1024, 21 * 48))
        t0 = time.perf_counter()
        svc = InferenceService(spec).startup()
        out["startup_s"] = time.perf_counter() - t0
        out["counters"] = {k: v for k, v in dict(getattr(svc, "counters", {})).items()
                           if isinstance(v, (int, float))}
    elif part == "opponent":
        from agents.model.compile_opponents import maybe_compile_extractor
        from main.fresh_checkpoint import build_fresh_model
        model, _, _ = build_fresh_model(0)
        t0 = time.perf_counter()
        out["kept"] = bool(maybe_compile_extractor(model, True, label="bench", hide_cuda=True))
        out["startup_s"] = time.perf_counter() - t0
    else:
        raise SystemExit(f"unknown part {part!r}")
    out["process_s"] = time.perf_counter() - t_import
    out["cache_bytes"] = CC.dir_bytes(root)
    return out


def _run_part(part: str, root: str, device: str, timeout_s: float) -> Dict[str, Any]:
    cmd = [sys.executable, os.path.abspath(__file__), "worker", part, root, "--device", device]
    t0 = time.perf_counter()
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    wall = time.perf_counter() - t0
    if r.returncode != 0:
        return {"part": part, "error": r.stderr[-4000:], "wall_s": wall}
    res = json.loads(r.stdout.strip().splitlines()[-1])
    res["wall_s"] = wall
    return res


def _contention() -> Dict[str, Any]:
    out: Dict[str, Any] = {"load1": os.getloadavg()[0], "cpus": os.cpu_count()}
    try:
        q = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=20)
        out["gpu_apps"] = [ln.strip() for ln in q.stdout.splitlines() if ln.strip()]
    except Exception as exc:  # noqa: BLE001
        out["gpu_apps"] = f"unreadable: {exc}"
    return out


def run(parts: Sequence[str], rounds: int, device: str, timeout_s: float,
        root_bases: Optional[Sequence[str]] = None, warm_reps: int = 2) -> Dict[str, Any]:
    """Per round and part: one fresh cache root per BASE (default: the on-disk scratch root); COLD on
    each base, then `warm_reps` WARM reads INTERLEAVED across the bases. `root_bases` is the explicit
    override for the storage A/B (e.g. the scratch root vs `/dev/shm`) — it bypasses
    `compile_cache.scratch_root`'s tmpfs refusal ON PURPOSE, for this measurement only."""
    from agents.model.compile_cache import fs_type, scratch_root
    bases = list(root_bases) if root_bases else [scratch_root()]
    rows: List[Dict[str, Any]] = []
    for rnd in range(rounds):
        for part in parts:
            roots = [tempfile.mkdtemp(prefix=f"k3bench_{part}_{os.getpid()}_", dir=b) for b in bases]
            order = list(range(len(bases))) if rnd % 2 == 0 else list(reversed(range(len(bases))))
            try:
                plan = [("cold", i) for i in order]
                for k in range(warm_reps):
                    plan += [("warm" if k == 0 else f"warm{k + 1}", i) for i in order]
                for phase, i in plan:
                    before = _contention()
                    res = _run_part(part, roots[i], device, timeout_s)
                    res.update(round=rnd, phase=phase, base=bases[i], base_fs=fs_type(bases[i]),
                               contention_before=before)
                    rows.append(res)
                    print(json.dumps({k: res.get(k) for k in ("round", "part", "phase", "base_fs", "startup_s",
                                                              "gate_s", "prewarm_s", "process_s", "wall_s",
                                                              "cache_bytes", "error")}), flush=True)
            finally:
                for root in roots:
                    shutil.rmtree(root, ignore_errors=True)
    return {"schema": "k3_compile_cache_benchmark_v1", "device": device, "bases": bases, "rows": rows,
            "when": time.strftime("%Y-%m-%dT%H:%M:%S%z")}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--device", default="cuda")
    r.add_argument("--rounds", type=int, default=2)
    r.add_argument("--parts", default=",".join(PARTS))
    r.add_argument("--timeout-s", type=float, default=3600.0)
    r.add_argument("--root-bases", default=None,
                   help="comma-separated cache-root BASES to A/B (default: the on-disk scratch root). An "
                        "explicit override for the storage measurement only: it may name a tmpfs such as "
                        "/dev/shm, which the production path refuses")
    r.add_argument("--warm-reps", type=int, default=2)
    r.add_argument("--json", default=None)
    w = sub.add_parser("worker")
    w.add_argument("part")
    w.add_argument("root")
    w.add_argument("--device", default="cuda")
    a = ap.parse_args(argv)
    if a.cmd == "worker":
        print(json.dumps(_worker(a.part, a.root, a.device)), flush=True)
        return 0
    res = run([p for p in a.parts.split(",") if p], a.rounds, a.device, a.timeout_s,
              root_bases=[b for b in (a.root_bases or "").split(",") if b] or None, warm_reps=a.warm_reps)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
