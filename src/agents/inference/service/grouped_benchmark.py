"""X45 — the paired OFF / ON read of T2's GROUPED FORWARD on a ROLLOUT-shaped flush (a measurement script).

``gen3_t2_grouped_forward_v1`` (``ServiceSpec.grouped_slots``, ``--t2-grouped-opponents``) serves every pool slot of
a host step in ONE vmapped forward over the stacked weights instead of one forward per slot. This script builds
TWO services from the same weights at production's slot layout — ``--opp-slots`` ACTIVE pool slots,
``--spare-slots`` declared ones that get no rows, the trainee next — one per-slot (OFF) and one with the pool
slots grouped (ON), then times one host step's T2 work in INTERLEAVED blocks (OFF, ON, ON, OFF, …): every
opponent slot's rows (a seeded multinomial of ``--opp-rows`` over the active slots, the SAME draws for both arms)
plus ``--trainee-rows`` on the trainee's slot (0 = none), ONE flush, and the host wait for its outputs
(``Ticket.host()``, the collector's ``gpu_wait``). It reports per arm the median ``flush`` / ``wait`` / step ms,
the per-block ON / OFF ratio of the step medians, and ON's max |Δ| against OFF on the first step (NOT 0: the
grouped forward sums in another order — fp32 reassociation, ``grouped.py``). A benchmark's output IS the
measurement: it WARNS on a busy box, never stretches.

    export PYTHONPATH=$PYTHONPATH:src
    # the GPU read (production shape at N = 256; needs a GPU lease — root CLAUDE.md rule 5):
    python3 src/agents/inference/service/grouped_benchmark.py --opp-slots 20 --json <out_pool20.json>
    python3 src/agents/inference/service/grouped_benchmark.py --opp-slots 5 --json <out_pool5.json>
    # the CPU read (tiny buckets, opponents only):
    python3 src/agents/inference/service/grouped_benchmark.py --device cpu --backend eager --buckets 4,16 \\
        --opponent-cap 16 --trainee-rows 0 --spare-slots 0 --load copy --blocks 4 --reps 2 --opp-slots 20

Defaults = production at N = 256 in the steady self-play regime (the lead-lanes benchmark's: ~212 opponent rows a
step over 20 active pool slots + 4 spare, the trainee's batch ~245 of 256, buckets (8, 64, 256), cap 64, 8 lanes).
Compile cache: a private one per run (K3).
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
    ap.add_argument("--opponent-cap", type=int, default=64, help="the opponent slots' largest bucket")
    ap.add_argument("--opp-slots", type=int, default=20, help="ACTIVE pool slots (each gets rows)")
    ap.add_argument("--spare-slots", type=int, default=4, help="declared pool slots that get no rows (production: 4)")
    ap.add_argument("--opp-rows", type=int, default=212, help="opponent rows per step, spread multinomially")
    ap.add_argument("--trainee-rows", type=int, default=245, help="0 = no trainee rows (an opponents-only read)")
    ap.add_argument("--lanes", type=int, default=8)
    ap.add_argument("--threads", type=int, default=1, help="torch CPU threads (the CPU read)")
    ap.add_argument("--blocks", type=int, default=6, help="interleaved blocks per arm")
    ap.add_argument("--reps", type=int, default=50, help="host steps per block")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load", default="gated", choices=("gated", "copy"),
                    help="how the distinct pool weights get in: T2's gated `load` (default), or a verified in-place "
                         "copy without the per-load parity gate (`copy`: the CPU read, where every gate run of a "
                         "grouped slot is a whole-range eager forward)")
    ap.add_argument("--json", default=None, help="also write the result here")
    args = ap.parse_args(argv)

    from agents.model.compile_cache import ensure_hermetic_cache
    ensure_hermetic_cache("grouped_benchmark")                 # K3: fresh, private, deleted at exit
    import numpy as np
    import torch

    from agents.inference.service import InferenceService, ServiceSpec, SlotGroupSpec
    from agents.inference.service.fixtures import perturbed_fresh_policy
    from agents.inference.service.parity import fixture_rows
    from agents.inference.service.service_benchmark import _busy_box_warnings, _load

    torch.set_num_threads(int(args.threads))
    warnings = _busy_box_warnings()
    policy = _load(args.ckpt) if args.ckpt else perturbed_fresh_policy(0)
    others = [perturbed_fresh_policy(k) for k in range(1, 8)]
    buckets = tuple(sorted({int(b) for b in args.buckets.split(",")}))
    n_pool = args.opp_slots + args.spare_slots
    trainee = n_pool                                          # production: the trainee's slot follows the pool's
    n_slots = n_pool + 1
    cap = max(buckets[0], args.opponent_cap)
    caps = tuple(buckets[-1] if s == trainee else cap for s in range(n_slots))
    cuda = args.device.startswith("cuda")
    lanes = args.lanes if cuda else 1

    def build(grouped: tuple) -> Any:
        spec = ServiceSpec(groups=(SlotGroupSpec("pool", n_slots, policy),), device=args.device,
                           backend=args.backend, buckets=buckets, lanes=lanes, slot_bucket_caps=caps,
                           max_rows_per_flush=max(1024, n_slots * buckets[-1]), grouped_slots=grouped)
        t0 = time.perf_counter()
        svc = InferenceService(spec).startup()
        # DISTINCT weights per pool slot (a grouped forward that read one slot's weights for all would differ)
        for s in range(1, n_pool):
            src = others[s % len(others)]
            if args.load == "gated":
                svc.load(s, src, f"perturbed:{s % len(others) + 1}")
            else:
                g = svc.groups[0]
                sd = g.check_loadable(src)
                g.copy_in(s, sd)
                g.verify_copy(s, sd, f"benchmark copy into slot {s}")
        return svc, round(time.perf_counter() - t0, 1)

    svc_off, up_off = build(())
    svc_on, up_on = build(tuple(range(n_pool)))
    rng = np.random.default_rng(args.seed)
    total = args.blocks * args.reps
    split = rng.multinomial(args.opp_rows, [1.0 / args.opp_slots] * args.opp_slots, size=total)
    obs, mask = fixture_rows(svc_off.obs_dim, max(args.trainee_rows, int(split.max()), 1))

    def step(svc: Any, k: int) -> "tuple[float, float, Any]":
        tickets = [svc.submit(s, obs[:n], mask[:n]) for s, n in enumerate(split[k % total].tolist()) if n]
        if args.trainee_rows:
            tickets.append(svc.submit(trainee, obs[:args.trainee_rows], mask[:args.trainee_rows]))
        t0 = time.perf_counter()
        svc.flush()
        t1 = time.perf_counter()
        tickets[-1].host()                                    # one host wait covers the flush's D2H (the collector's)
        return t1 - t0, time.perf_counter() - t1, tickets

    _, _, t_off = step(svc_off, 0)
    out_off = [tuple(np.array(x, copy=True) for x in t.host()) for t in t_off]
    _, _, t_on = step(svc_on, 0)
    out_on = [t.host() for t in t_on]
    d_lp = d_v = 0.0
    greedy_diff = 0
    pattern_ok = True
    for a, b in zip(out_off, out_on):
        fa, fb = np.isfinite(a[0]), np.isfinite(b[0])
        if not np.array_equal(fa, fb):                        # the illegal (-inf) pattern differs
            pattern_ok = False
            continue
        d_lp = max(d_lp, float(np.abs(a[0][fa] - b[0][fb]).max(initial=0.0)))
        d_v = max(d_v, float(np.abs(a[1] - b[1]).max(initial=0.0)))
        greedy_diff += int((a[2] != b[2]).sum())
    for svc in (svc_off, svc_on):                             # warm both
        for k in range(2):
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
        "lever": "gen3_t2_grouped_forward_v1 (X45)", "torch": torch.__version__, "backend": args.backend,
        "device": args.device, "threads": int(args.threads), "ckpt": args.ckpt or "fresh-perturbed (distinct per slot)",
        "buckets": buckets, "opponent_cap": cap, "active_pool_slots": args.opp_slots, "spare_slots": args.spare_slots,
        "slots": n_slots, "trainee_slot": trainee, "lanes": lanes, "opp_rows": args.opp_rows,
        "trainee_rows": args.trainee_rows, "blocks": args.blocks, "reps": args.reps, "load": args.load,
        "startup_s": {"off": up_off, "on": up_on},
        "on_vs_off_first_step": {"max_abs_dlogp_legal": d_lp, "max_abs_dv": d_v, "greedy_rows_differing": greedy_diff,
                                 "illegal_pattern_identical": pattern_ok},
        "per_arm_ms": {arm: {"flush_median": round(statistics.median(v["flush"]), 3),
                             "wait_median": round(statistics.median(v["wait"]), 3),
                             "step_median": round(statistics.median([f + w for f, w in zip(v["flush"], v["wait"])]), 3),
                             "block_step_medians": [round(x, 3) for x in v["block_step_median"]]}
                       for arm, v in per.items()},
        "on_over_off_block_ratios": [round(x, 4) for x in ratios],
        "on_over_off_ratio_median": round(statistics.median(ratios), 4),
        "grouped_rounds": st_on.get("grouped_rounds"),
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
