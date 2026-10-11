"""T25 perf lever 4 — the paired OFF / ON read of T2's LEAD SLOTS on a ROLLOUT-shaped flush (a measurement script).

``gen3_t2_lead_lanes_v1`` (``ServiceSpec.lead_slots``, ``--t2-lead-trainee``) launches the trainee's slot first, on a
lane of its own. This script builds TWO services from the same weights at production's slot layout and lane count —
``--opp-slots`` active pool slots, ``--spare-slots`` idle ones, the trainee next (slot 24 at the defaults, which the
default plan puts on lane 0 behind pool slots 0, 8 and 16) — one with the default plan (OFF) and one with the trainee
leading (ON), then times one host step's T2 work in INTERLEAVED blocks (OFF, ON, OFF, ON, …): every opponent slot's
rows (a seeded multinomial of ``--opp-rows`` over the active slots, the SAME draws for both arms) plus
``--trainee-rows`` on the trainee's slot, ONE flush, and the host wait for its outputs (``Ticket.host()``, as the
collector's ``gpu_wait``). It reports per arm the median ``flush`` (the host launch) and ``wait`` ms per step, the
per-block ON / OFF ratio of the step medians, and ON's max |Δ| against OFF on the first step (expected 0: only the
launch order and lane binding differ). A benchmark's output IS the measurement: it WARNS on a busy box, never stretches.

    export PYTHONPATH=$PYTHONPATH:src
    python3 src/agents/inference/service/lead_lanes_benchmark.py --json <out.json>     # CUDA, graph backend

Defaults = production at N = 256 in the steady self-play regime (the 2026-10-03 bottleneck profile: ~85k of 98k
opponent decisions per rollout were pool rows over ~400 host steps, i.e. ~212 a step over 20 active pool slots; the
trainee's batch ~245 of 256). Compile cache: a private one per run (K3), unless ``GEN3AI_COMPILE_CACHE_DIR`` +
``TORCHINDUCTOR_CACHE_DIR`` name a warm one; the ON service compiles warm from the OFF service's in-process cache.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from typing import Any, Dict, List


def main(argv: "list[str] | None" = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt", default=None, help="a checkpoint zip (default: a perturbed fresh production policy)")
    ap.add_argument("--backend", default="graph", choices=("eager", "graph"))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--buckets", default="8,64,256")
    ap.add_argument("--opponent-cap", type=int, default=64, help="the opponent slots' largest bucket (0 = uncapped)")
    ap.add_argument("--opp-slots", type=int, default=20, help="ACTIVE pool slots (each gets rows)")
    ap.add_argument("--spare-slots", type=int, default=4, help="declared pool slots that get no rows (production: 4)")
    ap.add_argument("--opp-rows", type=int, default=212, help="opponent rows per step, spread multinomially")
    ap.add_argument("--trainee-rows", type=int, default=245)
    ap.add_argument("--lanes", type=int, default=8)
    ap.add_argument("--blocks", type=int, default=6, help="interleaved blocks per arm")
    ap.add_argument("--reps", type=int, default=50, help="host steps per block")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="also write the result here")
    args = ap.parse_args(argv)

    from agents.model.compile_cache import ensure_hermetic_cache
    ensure_hermetic_cache("lead_lanes_benchmark")             # K3: fresh, private, deleted at exit
    import numpy as np
    import torch

    from agents.inference.service import InferenceService, ServiceSpec, SlotGroupSpec
    from agents.inference.service.fixtures import perturbed_fresh_policy
    from agents.inference.service.parity import fixture_rows
    from agents.inference.service.service_benchmark import _busy_box_warnings, _load

    warnings = _busy_box_warnings()
    policy = _load(args.ckpt) if args.ckpt else perturbed_fresh_policy(0)
    buckets = tuple(sorted({int(b) for b in args.buckets.split(",")}))
    n_pool = args.opp_slots + args.spare_slots
    trainee = n_pool                                          # production: the trainee's slot follows the pool's
    n_slots = n_pool + 1
    caps = tuple(buckets[-1] if s == trainee else max(buckets[0], args.opponent_cap) for s in range(n_slots)) \
        if args.opponent_cap > 0 else ()
    cuda = args.device.startswith("cuda")
    lanes = args.lanes if cuda else 1

    def build(lead: tuple) -> Any:
        spec = ServiceSpec(groups=(SlotGroupSpec("pool", n_slots, policy),), device=args.device,
                           backend=args.backend, buckets=buckets, lanes=lanes, slot_bucket_caps=caps,
                           max_rows_per_flush=max(1024, n_slots * buckets[-1]), lead_slots=lead)
        t0 = time.perf_counter()
        svc = InferenceService(spec).startup()
        return svc, round(time.perf_counter() - t0, 1)

    svc_off, up_off = build(())
    svc_on, up_on = build((trainee,))
    rng = np.random.default_rng(args.seed)
    total = args.blocks * args.reps
    split = rng.multinomial(args.opp_rows, [1.0 / args.opp_slots] * args.opp_slots, size=total)
    obs, mask = fixture_rows(svc_off.obs_dim, max(args.trainee_rows, int(split.max())))

    def step(svc: Any, k: int) -> "tuple[float, float, Any]":
        tickets = [svc.submit(s, obs[:n], mask[:n]) for s, n in enumerate(split[k % total].tolist()) if n]
        tickets.append(svc.submit(trainee, obs[:args.trainee_rows], mask[:args.trainee_rows]))
        t0 = time.perf_counter()
        svc.flush()
        t1 = time.perf_counter()
        tickets[-1].host()                                    # one host wait covers the flush's D2H (the collector's)
        return t1 - t0, time.perf_counter() - t1, tickets

    # bit-identity on one step before any timing (only the launch order and lane binding differ)
    _, _, t_off = step(svc_off, 0)
    out_off = [tuple(np.array(x, copy=True) for x in t.host()) for t in t_off]
    _, _, t_on = step(svc_on, 0)
    out_on = [t.host() for t in t_on]
    max_d = 0.0
    for a, b in zip(out_off, out_on):
        fa, fb = np.isfinite(a[0]), np.isfinite(b[0])
        if not np.array_equal(fa, fb):                        # the illegal (-inf) pattern differs
            max_d = float("inf")
            continue
        max_d = max(max_d, float(np.abs(a[0][fa] - b[0][fb]).max(initial=0.0)),
                    float(np.abs(a[1] - b[1]).max(initial=0.0)), float((a[2] != b[2]).sum()))
    for svc in (svc_off, svc_on):                             # warm both
        for k in range(5):
            step(svc, k)
    if cuda:
        torch.cuda.synchronize()
    per: Dict[str, Dict[str, List[float]]] = {a: {"flush": [], "wait": [], "block_step_median": []}
                                              for a in ("off", "on")}
    for blk in range(args.blocks):
        for arm, svc in (("off", svc_off), ("on", svc_on)) if blk % 2 == 0 else (("on", svc_on), ("off", svc_off)):
            steps = []
            for r in range(args.reps):
                f, w, _ = step(svc, blk * args.reps + r)
                per[arm]["flush"].append(f * 1e3)
                per[arm]["wait"].append(w * 1e3)
                steps.append((f + w) * 1e3)
            per[arm]["block_step_median"].append(statistics.median(steps))
    ratios = [on / off for on, off in zip(per["on"]["block_step_median"], per["off"]["block_step_median"])]
    st_on = svc_on.stats()
    result: Dict[str, Any] = {
        "lever": "gen3_t2_lead_lanes_v1", "torch": torch.__version__, "backend": args.backend, "device": args.device,
        "ckpt": args.ckpt or "fresh-perturbed(seed 0)", "buckets": buckets, "opponent_cap": args.opponent_cap,
        "slots": n_slots, "trainee_slot": trainee, "lanes": lanes, "opp_rows": args.opp_rows,
        "trainee_rows": args.trainee_rows, "blocks": args.blocks, "reps": args.reps,
        "startup_s": {"off": up_off, "on": up_on},
        "trainee_lane": {"off": svc_off.engine.lane_of(trainee), "on": svc_on.engine.lane_of(trainee)},
        "slots_sharing_the_trainee_lane": {
            arm: [s for s in range(n_slots) if s != trainee and svc.engine.lane_of(s) == svc.engine.lane_of(trainee)]
            for arm, svc in (("off", svc_off), ("on", svc_on))},
        "on_vs_off_first_step_max_abs_diff": max_d,
        "per_arm_ms": {arm: {"flush_median": round(statistics.median(v["flush"]), 3),
                             "wait_median": round(statistics.median(v["wait"]), 3),
                             "step_median": round(statistics.median([f + w for f, w in zip(v["flush"], v["wait"])]), 3),
                             "block_step_medians": [round(x, 3) for x in v["block_step_median"]]}
                       for arm, v in per.items()},
        "on_over_off_block_ratios": [round(x, 4) for x in ratios],
        "on_over_off_ratio_median": round(statistics.median(ratios), 4),
        "after_freeze": {k: st_on[k] for k in ("compiles_after_freeze", "captures_after_freeze",
                                               "cuda_segments_after_freeze")},
        "busy_box_warnings": warnings}
    if cuda:
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
