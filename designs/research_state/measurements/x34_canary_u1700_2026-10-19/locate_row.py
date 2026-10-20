"""X34 (rb_toy_ent_s4002_deep) canary u1700 UNCONFIRMED: locate the fixture row and the op behind the bf16 arm's
0.856 log-prob shift. CPU only. Run with PYTHONPATH pointed at the X34 PIN's src/ (ae2d0901), so the learner is the
run's own code:

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=<pin>/src python locate_row.py --zip <ckpt.zip> --out result_<label>.json \
        [--dither 16] [--compiled]

Reads, on the canary's own rows (`compile_regions.r1_batch`, B = 2048, the K9 golden's 64 rows tiled; slice 2 = rows
0-31, slice 1 = rows 32-63), through the run's own `micro_step`:
  * E32 (eager fp32, the canary's reference) and Ebf (eager bf16): per fixture row |d log pi(a)| (the taken action's
    log-prob, the canary's clip-margin statistic) and the full masked log-prob vector of the worst row;
  * THE CONDITIONING GAUGE: N seeded multiplicative weight dithers at relative 2^-9 (half a bf16 ulp; the
    `diag_bf16_canary.dithered` null), each read at fp32 against the undithered fp32 — how far a row's log-prob moves
    when only the ROUNDING of the weights changes, with no bf16 arithmetic and no compiler at all;
  * THE MODULE BISECT: every module's output on the worst row, E32 vs Ebf (relative L2 error), so the first module
    whose output error jumps is named — and the pointer head's logits for that row (gap between the top two legal
    logits, the logit range).
  * optionally (--compiled) Cbf: the production `compile_regions.install` on CPU Inductor (C++ codegen, not the
    run's Triton kernels: it reads the graph-level path only).
Nothing here trains or writes under models/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch


def per_row(a: Any, b: Any, uniq: List[int]) -> List[float]:
    return [float((a.log_prob[i].double() - b.log_prob[i].double()).abs()) for i in uniq]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dither", type=int, default=16)
    ap.add_argument("--compiled", action="store_true")
    a = ap.parse_args()
    torch.set_num_threads(6)
    from main.ops import diag_bf16_canary as D
    from agents.model.trunk_precision import fp32_scope
    from agents.training.instrumented_ppo.micro_step import micro_step
    t0 = time.perf_counter()
    model = D.build_learner(Path(a.zip), "cpu", batch=2048)
    print(f"[locate] learner built {time.perf_counter() - t0:.0f}s", flush=True)
    res: Dict[str, Any] = {"zip": a.zip, "batch": 2048, "slices": {}}
    from agents.model import compile_regions as cr
    rows_of = {2: list(range(32)), 1: list(range(32))}          # batch positions 0..31 hold each slice's 32 rows once
    first_row = {2: 0, 1: 32}
    arms_by_slice: Dict[int, Dict[str, Any]] = {}
    for sl in (2, 1):
        b = cr.r1_batch(model, 2048, sl)
        args = cr._r1_args(model, b)
        e32 = D.run_arm(model, micro_step, args, fp32_trunk=True)
        ebf = D.run_arm(model, micro_step, args, fp32_trunk=False)
        d = per_row(ebf, e32, rows_of[sl])
        worst = max(range(32), key=lambda i: d[i])
        masks = b.action_masks.bool().cpu()
        arms_by_slice[sl] = {"E32": e32, "Ebf": ebf, "args": args, "b": b}
        res["slices"][str(sl)] = {
            "fixture_rows": f"{first_row[sl]}-{first_row[sl] + 31}",
            "Ebf_vs_E32_max_abs_dlogp": max(d), "worst_fixture_row": first_row[sl] + worst, "worst_batch_pos": worst,
            "per_row_dlogp": {str(first_row[sl] + i): round(v, 5) for i, v in enumerate(d)},
            "worst_row": {
                "action": int(b.actions[worst].item()),
                "legal": [int(x) for x in masks[worst].nonzero().flatten()],
                "E32_logp": [round(float(x), 4) for x in e32.logp[worst]],
                "Ebf_logp": [round(float(x), 4) for x in ebf.logp[worst]],
                "E32_log_prob_taken": float(e32.log_prob[worst]), "Ebf_log_prob_taken": float(ebf.log_prob[worst]),
                "E32_entropy": float(e32.entropy[worst]),
            },
        }
        print(f"[locate] slice {sl}: Ebf max |dlogp| {max(d):.4f} at fixture row {first_row[sl] + worst}", flush=True)

    # THE CONDITIONING GAUGE: fp32 only, the weights dithered by half a bf16 ulp
    for sl in (2, 1):
        e32, args = arms_by_slice[sl]["E32"], arms_by_slice[sl]["args"]
        per_seed: List[List[float]] = []
        for s in range(a.dither):
            with D.dithered([model], s + 1):
                dd = D.run_arm(model, micro_step, args, fp32_trunk=True)
            per_seed.append(per_row(dd, e32, rows_of[sl]))
        mx = [max(ps[i] for ps in per_seed) for i in range(32)]
        med = [sorted(ps[i] for ps in per_seed)[len(per_seed) // 2] for i in range(32)]
        rec = res["slices"][str(sl)]
        rec["dither_fp32"] = {"n": a.dither, "rel": D.DITHER_REL,
                              "per_row_max": {str(first_row[sl] + i): round(v, 5) for i, v in enumerate(mx)},
                              "per_row_median": {str(first_row[sl] + i): round(v, 5) for i, v in enumerate(med)},
                              "slice_max": max(mx),
                              "worst_row_seeds": [round(ps[rec["worst_batch_pos"]], 5) for ps in per_seed]}
        print(f"[locate] slice {sl}: fp32 dither null max |dlogp| {max(mx):.4f}; worst row's seeds "
              f"{rec['dither_fp32']['worst_row_seeds'][:8]}", flush=True)

    # THE MODULE BISECT on slice 2's worst row
    sl = 2
    w = res["slices"]["2"]["worst_batch_pos"]
    args = arms_by_slice[sl]["args"]
    caught: Dict[str, Dict[str, torch.Tensor]] = {"E32": {}, "Ebf": {}}
    cur = {"arm": "E32"}

    def hook(name: str):
        def f(_m: Any, _i: Any, out: Any) -> None:
            t = out[0] if isinstance(out, (tuple, list)) and out and isinstance(out[0], torch.Tensor) else out
            if isinstance(t, torch.Tensor) and t.dim() >= 1 and t.shape[0] == 2048 and t.is_floating_point():
                caught[cur["arm"]].setdefault(name, t[w].detach().double().clone())
        return f
    hs = [m.register_forward_hook(hook(n)) for n, m in model.policy.named_modules() if n]
    try:
        for arm in ("E32", "Ebf"):
            cur["arm"] = arm
            D.run_arm(model, micro_step, args, fp32_trunk=(arm == "E32"))
    finally:
        for h in hs:
            h.remove()
    bis = []
    for n in caught["E32"]:
        if n in caught["Ebf"] and caught["E32"][n].shape == caught["Ebf"][n].shape:
            r, q = caught["E32"][n], caught["Ebf"][n]
            bis.append({"module": n, "rel_l2": float((q - r).norm() / r.norm().clamp_min(1e-30)),
                        "abs_max": float((q - r).abs().max()), "ref_abs_max": float(r.abs().max())})
    res["bisect_slice2_worst_row"] = bis
    # the module ORDER of first appearance (forward order) is the dict's insertion order
    if a.compiled:
        t1 = time.perf_counter()
        cr.install(model)
        for sl in (2, 1):
            args = arms_by_slice[sl]["args"]
            cbf = D.run_arm(model, model._compiled_micro_step, args, fp32_trunk=False)
            d = per_row(cbf, arms_by_slice[sl]["E32"], rows_of[sl])
            d2 = per_row(cbf, arms_by_slice[sl]["Ebf"], rows_of[sl])
            res["slices"][str(sl)]["Cbf_cpu_inductor"] = {
                "vs_E32_max_abs_dlogp": max(d), "vs_E32_worst_row": first_row[sl] + max(range(32), key=lambda i: d[i]),
                "vs_Ebf_max_abs_dlogp": max(d2),
                "per_row_vs_E32": {str(first_row[sl] + i): round(v, 5) for i, v in enumerate(d)}}
            print(f"[locate] slice {sl}: Cbf(CPU Inductor) vs E32 {max(d):.4f}, vs Ebf {max(d2):.4f} "
                  f"({time.perf_counter() - t1:.0f}s)", flush=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(f"[locate] wrote {a.out} in {time.perf_counter() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
