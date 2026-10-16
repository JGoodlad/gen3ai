"""PER-COMPONENT CPU COST of the end-state (production) network — params, FLOPs and CPU time per row, forward and
backward, attributed to the component that issued each op (design_compute_efficiency.md §2).

    CUDA_VISIBLE_DEVICES="" nice -n 19 taskset -c 5-7,13-15 python component_cost.py --out result/cost_production.json
    ... component_cost.py --toy <models/rb_toy_ent_s4002/final_model.zip> --out result/cost_toy.json

METHOD (CPU, eager, fp32, ONE torch thread, real labelled rows: the K9 learner golden's buffer via `r1_batch`):
- Every module of the policy down to depth 3, the `DamageOperator`'s public methods (the edge-cell kernels are
  called as methods, not modules) and the extractor's private helpers are wrapped in a `record_function` LABEL.
- FORWARD: each profiled op's SELF time and counted FLOPs go to its innermost label.
- BACKWARD: each `autograd::engine::evaluate_function: XBackward` node goes, with its whole subtree (time and
  FLOPs), to the label of the FORWARD op carrying the same `sequence_nr` (autograd's forward/backward link).
  `AccumulateGrad` nodes (sequence_nr -1) go to `(accumulate_grad)`.
- The profiler inflates many-small-op code (the damage op's elementwise kernels) more than GEMM code, so the
  PROFILED shares are reported beside UNPROFILED totals (median of reps); FLOP counts are exact for
  mm / addmm / bmm / baddbmm (the profiler's `with_flops`), and elementwise FLOPs are partly counted.
- PRODUCTION (default): the learner exactly as `learner_golden.build_learner()` builds production (trunk fp32 on CPU;
  the production recipe is bf16 on the GPU), stepped through the REAL micro-step (`micro_step` + backward: PPO +
  every belief / intent / win-prob loss; the ride-along heads are the separate eager tail and NOT included here —
  their cost is MEASURED by `ridealong_step_benchmark.py`). INFERENCE: `DecisionModule` (the T2 forward) in eval mode
  under no_grad.
- `--toy <zip>`: a checkpoint loaded by `load_checkpoint_strict`; INFERENCE as above and a SURROGATE training step
  (the T2 forward with grad, loss = −mean log π(a) − mean V), also run on production for a like-for-like ratio.
Read-only: writes only --out.
"""
from __future__ import annotations

import argparse
import functools
import json
import os
import statistics
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List

import torch as th
from torch.profiler import ProfilerActivity, profile, record_function

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "src")))

FE_HELPERS = ("_typed_hp_posterior", "_build_hypothesis_species", "_attach_move_group", "_apply_move_belief",
              "_spread_hp_damage", "_other_edge_cells", "_flat_consumer_ops", "_op_content_rows")


def _wrap_call(mod: th.nn.Module, label: str) -> None:
    orig = mod.forward

    @functools.wraps(orig)
    def fwd(*a: Any, **k: Any) -> Any:
        with record_function(label):
            return orig(*a, **k)
    mod.forward = fwd                                       # instance attribute shadows the class method


def _wrap_method(obj: Any, name: str, label: str) -> None:
    orig = getattr(obj, name)

    @functools.wraps(orig)
    def m(*a: Any, **k: Any) -> Any:
        with record_function(label):
            return orig(*a, **k)
    setattr(obj, name, m)


def instrument(policy: Any) -> None:
    fe = policy.features_extractor
    seen = set()
    roots = [("fe", fe)] + [(n, c) for n, c in policy.named_children()
                            if c is not fe and n not in ("pi_features_extractor", "vf_features_extractor")]
    for root_name, root in roots:
        for qn, m in root.named_modules():
            depth = 0 if not qn else qn.count(".") + 1
            if depth > 3 or id(m) in seen:
                continue
            seen.add(id(m))
            _wrap_call(m, f"mod:{root_name}{'.' + qn if qn else ''}")
    op = getattr(fe, "damage_op", None)
    if op is not None:
        for name in dir(type(op)):
            if name.startswith("_") or name in ("forward",):
                continue
            attr = getattr(type(op), name, None)
            if callable(attr) and not isinstance(attr, type) and name not in dir(th.nn.Module):
                _wrap_method(op, name, f"meth:damage_op.{name}")
    for h in FE_HELPERS:
        if hasattr(fe, h):
            _wrap_method(fe, h, f"meth:fe.{h}")


def _label_of(e: Any) -> str:
    p = e
    while p is not None:
        if p.name.startswith(("mod:", "meth:")):
            return p.name
        p = p.cpu_parent
    return "(outside any module: losses, masking, glue)"


def attribute(prof: Any) -> Dict[str, Dict[str, float]]:
    evs = list(prof.events())
    out: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))
    seq_label: Dict[int, str] = {}
    bwd_roots = []

    def in_bwd(e: Any) -> bool:
        p = e
        while p is not None:
            if p.name.startswith("autograd::engine::evaluate_function"):
                return True
            p = p.cpu_parent
        return False
    for e in evs:
        if e.name.startswith("autograd::engine::evaluate_function"):
            bwd_roots.append(e)
            continue
        if e.name.startswith(("mod:", "meth:")) or in_bwd(e):
            continue
        lab = _label_of(e)
        if e.sequence_nr is not None and e.sequence_nr >= 0 and e.sequence_nr not in seq_label:
            seq_label[e.sequence_nr] = lab
        d = out[lab]
        d["fwd_self_us"] += float(e.self_cpu_time_total)
        d["fwd_flops"] += float(e.flops or 0)
        if e.name.startswith("aten::"):
            d["fwd_aten_calls"] += 1

    def subtree_flops(e: Any) -> float:
        return float(e.flops or 0) + sum(subtree_flops(c) for c in e.cpu_children)
    for r in bwd_roots:
        lab = ("(accumulate_grad)" if "AccumulateGrad" in r.name
               else seq_label.get(r.sequence_nr, "(backward: unmatched sequence_nr)"))
        d = out[lab]
        d["bwd_us"] += float(r.cpu_time_total)
        d["bwd_flops"] += subtree_flops(r)
        d["bwd_nodes"] += 1
    return {k: dict(v) for k, v in out.items()}


def params_by_label(policy: Any) -> Dict[str, int]:
    fe = policy.features_extractor
    res: Dict[str, int] = {}
    seen = set()
    for n, c in fe.named_children():
        k = 0
        for p in c.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                k += p.numel()
        res[f"mod:fe.{n}"] = k
    for n, c in policy.named_children():
        if c is fe or n in ("pi_features_extractor", "vf_features_extractor"):
            continue
        k = 0
        for p in c.parameters():
            if id(p) not in seen:
                seen.add(id(p))
                k += p.numel()
        res[f"mod:{n}"] = k
    res["(total, unique)"] = sum(p.numel() for p in {id(p): p for p in policy.parameters()}.values())
    return res


def timed(fn: Any, reps: int) -> Dict[str, float]:
    fn()                                                   # warm-up
    xs: List[float] = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        xs.append(time.perf_counter() - t0)
    return {"median_s": statistics.median(xs), "min_s": min(xs), "max_s": max(xs), "reps": reps}


def build_production() -> Any:
    from agents.training import learner_golden as LG
    model = LG.build_learner(trunk_precision="fp32")
    LG.load_buffer_into(model)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--infer-rows", type=int, nargs="+", default=[64, 256])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--toy", default=None, help="a checkpoint .zip to measure instead of production")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    th.set_num_threads(a.threads)
    from agents.inference.service.decision import DecisionModule
    from agents.model import compile_regions as cr

    prod = build_production()
    b = cr.r1_batch(prod, a.rows)
    if a.toy:
        from agents.model.snapshot import historical_load_kwargs, load_checkpoint_strict
        model = load_checkpoint_strict(a.toy, device="cpu", **historical_load_kwargs(a.toy))
        name = os.path.basename(os.path.dirname(os.path.abspath(a.toy))) + "/" + os.path.basename(a.toy)
    else:
        model, name = prod, "production (learner_golden.build_learner, trunk fp32)"
    pol = model.policy
    res: Dict[str, Any] = {"model": name, "rows": a.rows, "threads": a.threads, "torch": th.__version__,
                           "params": params_by_label(pol)}
    obs_t = b.obs["observation"]
    mask_t = b.obs["action_mask"].bool() if "action_mask" in b.obs else b.action_masks.bool()
    acts = b.actions.reshape(-1).long()

    # --- training: the real micro-step (production) and the surrogate step (both) -------------------------
    dm = DecisionModule(pol)

    def surrogate() -> None:
        pol.set_training_mode(True)
        logp, v = dm(obs_t, mask_t)
        loss = -logp.gather(1, acts[:, None]).clamp_min(-50).mean() - v.mean()
        loss.backward()
        pol.optimizer.zero_grad(set_to_none=True)

    steps = {"surrogate": surrogate}
    if not a.toy:
        import agents.training.instrumented_ppo.micro_step as MS
        st = model._micro_static(model._resolve_fold_flags())

        def micro() -> None:
            pol.set_training_mode(True)
            out = MS.micro_step(pol, b.obs, b.actions, b.action_masks, b.old_log_prob, b.old_values,
                                b.advantages, b.returns, st)
            out.loss.backward()
            pol.optimizer.zero_grad(set_to_none=True)

        def micro_fwd() -> None:
            pol.set_training_mode(True)
            out = MS.micro_step(pol, b.obs, b.actions, b.action_masks, b.old_log_prob, b.old_values,
                                b.advantages, b.returns, st)
            del out
        steps = {"micro_step": micro, "surrogate": surrogate}
        res["micro_step_fwd_only_unprofiled"] = timed(micro_fwd, a.reps)

    for nm, fn in steps.items():
        res[f"{nm}_unprofiled"] = timed(fn, a.reps)
    # inference (the T2 forward)
    for r in a.infer_rows:
        ob, mk = obs_t[:r].contiguous(), mask_t[:r].contiguous()

        def infer(ob: th.Tensor = ob, mk: th.Tensor = mk) -> None:
            pol.set_training_mode(False)
            with th.no_grad():
                dm(ob, mk)
        res[f"infer_{r}_unprofiled"] = timed(infer, a.reps)

    instrument(pol)
    for nm, fn in steps.items():
        fn()
        with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
            t0 = time.perf_counter()
            fn()
            res[f"{nm}_profiled_wall_s"] = time.perf_counter() - t0
        res[f"{nm}_by_label"] = attribute(prof)
    for r in a.infer_rows:
        ob, mk = obs_t[:r].contiguous(), mask_t[:r].contiguous()
        pol.set_training_mode(False)
        with th.no_grad():
            dm(ob, mk)
            with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
                t0 = time.perf_counter()
                dm(ob, mk)
                res[f"infer_{r}_profiled_wall_s"] = time.perf_counter() - t0
        res[f"infer_{r}_by_label"] = attribute(prof)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, sort_keys=True)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
