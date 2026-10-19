"""T62 ROLLOUT cell: T2's DECISION FORWARD (`engine.decide` over `DecisionModule`, compiled by inductor with
``dynamic=False`` and captured as a CUDA graph -- exactly T2's ``graph`` backend build) at X26's surface, pre-T62
(``TAU_SOLVE_DTYPE`` float32) against T62 (float64), in ONE process on the SAME weights and rows.

Each arm is compiled after ``torch._dynamo.reset()`` with its dtype set, warmed twice and captured per bucket; a
captured graph replays the kernels it captured, so the two arms coexist. Then INTERLEAVED blocks (old, new, new,
old, ...): per block and bucket, ``--reps`` replays per arm timed with CUDA events. Reported per bucket: each block's
median ms per arm, the per-block delta (new - old), and their min / median / max -- the reader's T62 band.

    python3 t62_decision_cell.py --out cell.json               # on the GPU (the battery runs it under gpu_lock)
    python3 t62_decision_cell.py --check                       # CPU: imports, argv, fixture rows; builds nothing

Buckets: 256 (the trainee's ~245-row step) and 64 (an opponent slot's chunk: ~212 rows over 20 slots is ~11 a slot,
served by the smallest bucket that fits, 64 under the production (8, 64, 256) ladder; 8 costs at most 64's).
The policy is a seeded FRESH build at X26's training surface (`gpu_bench_suite.PRODUCTION_ARGV`): `fixed_size_tau`
runs a fixed 64-step bisection on static shapes, so its cost does not depend on the weights.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import statistics
import sys
import time
from pathlib import Path
from typing import Any

BUCKETS = (64, 256)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--blocks", type=int, default=8)
    ap.add_argument("--reps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=2601)
    a = ap.parse_args(argv)

    import torch

    from agents.inference.service.decision import DecisionModule
    from agents.inference.service.engine import decide
    from agents.inference.service.parity import fixture_rows
    from agents.model import hypothesis_set as HS
    from agents.model.compile_cache import ensure_hermetic_cache
    from agents.training.gpu_bench_suite import PRODUCTION_ARGV
    from agents.training.gpu_bench_update import resolved_args
    from main.fresh_checkpoint import build_fresh_model

    if not hasattr(HS, "TAU_SOLVE_DTYPE"):
        print("PRECONDITION FAILED: hypothesis_set.TAU_SOLVE_DTYPE missing (a pre-T62 checkout)", file=sys.stderr)
        return 3
    if a.check:
        args = resolved_args(shlex.split(PRODUCTION_ARGV), "cpu")
        from main.fresh_checkpoint import _production_policy_kwargs
        _, layout, _ = _production_policy_kwargs(args)
        D = int(layout["total_dim"])
        o, m = fixture_rows(D, max(BUCKETS))
        print(f"ok: argv resolves (arch {getattr(args, 'arch', '?')}), fixture rows {o.shape} / {m.shape}, "
              f"TAU_SOLVE_DTYPE default {HS.TAU_SOLVE_DTYPE}")
        return 0
    if not a.out:
        ap.error("--out is required (or --check)")
    if not torch.cuda.is_available():
        print("no visible CUDA device", file=sys.stderr)
        return 2
    out = Path(a.out)
    res: dict[str, Any] = {"cell": "t62_decision", "status": "running", "buckets": list(BUCKETS),
                           "blocks": a.blocks, "reps": a.reps, "torch": torch.__version__,
                           "gpu": torch.cuda.get_device_name(0), "load1_start": os.getloadavg()[0]}

    def save() -> None:
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(res, indent=1, default=str))
        tmp.replace(out)
    save()
    torch.set_float32_matmul_precision("highest")
    dev = torch.device("cuda")
    t0 = time.perf_counter()
    args = resolved_args(shlex.split(PRODUCTION_ARGV), "cpu")
    model, _, _ = build_fresh_model(a.seed, args=args)
    dm = DecisionModule(model.policy.eval()).eval().to(dev)
    D = int(model.observation_space["observation"].shape[0])
    res["build_s"] = time.perf_counter() - t0
    static: dict[int, Any] = {}
    for b in BUCKETS:
        o, m = fixture_rows(D, b)
        static[b] = (torch.as_tensor(o).to(dev).clone(), torch.as_tensor(m).to(dev).clone())
    ensure_hermetic_cache("gpu window battery t62 decision cell")
    graphs: dict[str, dict[int, Any]] = {}
    outs: dict[str, dict[int, Any]] = {}
    res["compile_s"] = {}
    stream = torch.cuda.Stream(dev)
    for arm, dt in (("old", torch.float32), ("new", torch.float64)):
        HS.TAU_SOLVE_DTYPE = dt
        torch._dynamo.reset()
        compiled = torch.compile(decide, dynamic=False)
        graphs[arm], outs[arm] = {}, {}
        t0 = time.perf_counter()
        with torch._dynamo.config.patch(suppress_errors=False), torch.no_grad():  # type: ignore[attr-defined]
            for b in BUCKETS:
                so, sm = static[b]
                stream.wait_stream(torch.cuda.current_stream(dev))
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        compiled(dm, so, sm)
                torch.cuda.current_stream(dev).wait_stream(stream)
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, stream=stream):
                    o3 = compiled(dm, so, sm)
                graphs[arm][b], outs[arm][b] = g, o3
        torch.cuda.synchronize(dev)
        res["compile_s"][arm] = time.perf_counter() - t0
        save()
    HS.TAU_SOLVE_DTYPE = torch.float64
    # the two programs on the same rows: logp / value agreement (fp32 reassociation-scale expected, T62 equivalence)
    agree: dict[str, Any] = {}
    for b in BUCKETS:
        graphs["old"][b].replay()
        graphs["new"][b].replay()
        torch.cuda.synchronize(dev)
        lo, ln = outs["old"][b][0], outs["new"][b][0]
        fin = torch.isfinite(lo) & torch.isfinite(ln)
        agree[str(b)] = {"max_abs_dlogp": float((lo - ln)[fin].abs().max()),
                         "max_abs_dvalue": float((outs["old"][b][1] - outs["new"][b][1]).abs().max()),
                         "greedy_agree": float((outs["old"][b][2] == outs["new"][b][2]).float().mean())}
    res["agreement"] = agree
    # interleaved timing
    per: dict[str, dict[str, list[float]]] = {str(b): {"old": [], "new": []} for b in BUCKETS}
    for blk in range(a.blocks):
        order = ("old", "new") if blk % 2 == 0 else ("new", "old")
        for b in BUCKETS:
            for arm in order:
                g = graphs[arm][b]
                g.replay()                                   # one untimed replay: the arm's switch-in
                ts: list[float] = []
                for _ in range(a.reps):
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    g.replay()
                    e1.record()
                    e1.synchronize()
                    ts.append(float(e0.elapsed_time(e1)))
                per[str(b)][arm].append(statistics.median(ts))
        res["blocks_done"] = blk + 1
        res["block_medians_ms"] = per
        save()
    summ: dict[str, Any] = {}
    for b, d in per.items():
        deltas = [n - o for o, n in zip(d["old"], d["new"])]
        summ[b] = {"old_ms": statistics.median(d["old"]), "new_ms": statistics.median(d["new"]),
                   "delta_ms": {"min": min(deltas), "median": statistics.median(deltas), "max": max(deltas)}}
    res["summary"] = summ
    res["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    res["load1_end"] = os.getloadavg()[0]
    res["status"] = "ok"
    save()
    print(json.dumps(summ, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
