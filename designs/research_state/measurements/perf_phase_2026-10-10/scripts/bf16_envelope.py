"""THE bf16 PARITY ENVELOPE (T25 perf phase 2, 2026-10-10): the numbers a bf16 R1 parity rule is derived from.

    # under the GPU lease (gpu_lock + mem_cap), from any cwd; this checkout's src/ is put first (DIAG_SRC overrides)
    python bf16_envelope.py --argv "$(python configs.py --argv E)" --out ../results/window_D/E_bf16_envelope.json
    python bf16_envelope.py --argv "..." --out /tmp/x.json --cpu-smoke     # the code path on CPU (eager stands in)

WHAT IT MEASURES. One learner built for the argv at the production shape (`update_bench`'s build), its trunk rounds set
to bf16 (`trunk_precision`), R1 compiled ONCE with them (`compile_regions.install`, the launch's own compile, no gate).
Then, at several WEIGHT STATES, on the R1 startup gate's own rows (`compile_regions.r1_batch`, the K9 golden's real
labelled rows tiled to the batch; slices 0 and 1 = two independent batches), these R1 ARMS (`compile_regions._r1_arm`:
the loss + the flat gradient over every judged parameter):

    Cbf, Cbf2   compiled R1, bf16 trunk (the graph a bf16 run trains with), twice (run-to-run)
    Ebf, Ebf2   eager R1, bf16 trunk, twice
    E32, E32b   eager R1, fp32 trunk (the production function: the REFERENCE), twice

and, per pair, the gate's own statistics: loss rel |a - b| / max(1, |b|), the global gradient cosine (float64), and the
per-parameter relative error ||a_p - b_p|| / ||b_p|| for EVERY parameter (no floor: the floor is applied offline, with
each parameter's gradient-norm ratio to the largest recorded). Plus the decision readout (`_decision_readout`: legal
log-probs, V) bf16-eager vs fp32-eager, no grad, eval mode: the epoch-0 PPO-ratio noise of a bf16 learner against an
fp32 behaviour.

WEIGHT STATES: `fresh` (the build), `pert05_k<k>` / `pert10_k<k>` (the parity ladder's rungs (0.05, k) / (0.1, k), in
place, restored bit-exactly), then `trained_<n>` after n real 1-epoch `train()` calls on the update-fit fixture through the
compiled bf16 R1 (the weights a bf16 run actually reaches).

Nothing is judged here: the derivation reads the JSON (PLAN.md §2a). CPU smoke: eager stands in for the compiled arm
(its pairs read 0 by construction), batch 256.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SRC = os.environ.get("DIAG_SRC") or os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", "..", "src"))
sys.path.insert(0, SRC)


def pair_stats(a, b, sizes):
    """The gate's statistics of arm ``a`` against arm ``b`` (``b`` is the denominator)."""
    import torch
    la, lb = float(a["loss"]), float(b["loss"])
    ga, gb = a["grad"].double(), b["grad"].double()
    na, nb = float(ga.norm()), float(gb.norm())
    cos = float(torch.dot(ga, gb) / (na * nb)) if na > 0 and nb > 0 else float("nan")
    per = []
    for x, y in zip(torch.split(ga, sizes), torch.split(gb, sizes)):
        ny = float(y.norm())
        per.append(float((x - y).norm()) / ny if ny > 0 else (0.0 if float(x.norm()) == 0 else float("inf")))
    rel_l2 = float((ga - gb).norm()) / nb if nb > 0 else float("nan")
    return {"loss_a": la, "loss_b": lb, "loss_rel": abs(la - lb) / max(1.0, abs(lb)), "cos": cos, "rel_l2": rel_l2,
            "per_param_rel": per}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--argv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-calls", default="1,3,6", help="cumulative 1-epoch train() calls at which to read")
    ap.add_argument("--cpu-smoke", action="store_true")
    a = ap.parse_args()

    import numpy as np
    import torch

    cpu = a.cpu_smoke
    res = {"argv": a.argv, "src": SRC, "torch": torch.__version__, "load1_start": os.getloadavg()[0],
           "device": "cpu" if cpu else "cuda", "states": {}, "phases_s": {}}

    def save():
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        with open(a.out + ".tmp", "w") as f:
            json.dump(res, f, indent=1, default=str)
        os.replace(a.out + ".tmp", a.out)

    import update_bench as UB
    if cpu:
        UB.N_STEPS, UB.N_ENVS = 32, 64
        _sync = lambda: None  # noqa: E731
    else:
        assert torch.cuda.is_available(), "needs the GPU (lease + gpu_lock)"
        torch.cuda.set_device(0)
        _sync = torch.cuda.synchronize
    from agents.model.compile_cache import ensure_hermetic_cache
    ensure_hermetic_cache("perf phase bf16 envelope")
    torch.set_float32_matmul_precision("highest")
    from agents.model import compile_regions as cr
    from agents.model import compile_trainer as ct
    from agents.model.trunk_precision import set_trunk_precision, trunk_precision_of
    from agents.model.parity_probe import perturbed_parameters, rung_seed
    from agents.training import learner_golden as LG
    from agents.training import update_fit as UF
    from agents.training.instrumented_ppo.micro_step import micro_step

    t0 = time.perf_counter()
    args = UB.resolved_args(shlex.split(a.argv), "cpu" if cpu else "cuda")
    LG.N_STEPS, LG.N_ENVS = UB.N_STEPS, UB.N_ENVS
    model = LG.build_learner(args=args)
    dev = torch.device("cpu" if cpu else "cuda")
    model.policy.to(dev)
    model.device = dev
    seen = set()
    for holder in [model, model.policy] + list(model.policy.modules()):
        for v in list(vars(holder).values()):
            for o in (v if isinstance(v, (list, tuple)) else list(v.values()) if isinstance(v, dict) else [v]):
                if isinstance(o, torch.optim.Optimizer) and id(o) not in seen:
                    seen.add(id(o))
                    for st in o.state.values():
                        for k, x in list(st.items()):
                            if torch.is_tensor(x) and (k != "step" or any(
                                    g.get("fused") or g.get("capturable") for g in o.param_groups)):
                                st[k] = x.to(dev)
    model.rollout_buffer.device = dev
    model.batch_size = 256 if cpu else int(args.batch_size)
    model.grad_accum_steps = 4 if cpu else int(getattr(args, "grad_accum_steps", 1) or 1)
    res["batch_size"] = model.batch_size
    res["trunk_rounds_set"] = set_trunk_precision(model.policy, "bf16")
    from main.train.lifecycle import _apply_grad_checkpointing
    _apply_grad_checkpointing(model, bool(getattr(args, "grad_checkpointing", False)))
    res["phases_s"]["build"] = time.perf_counter() - t0
    names = [n for n, _ in ct.grad_parameters(model, model.policy.features_extractor)]
    res["param_names"] = names
    save()

    # ---- ONE compile of R1 with the bf16 rounds (the launch's install + prewarm; no gate: this tool judges nothing)
    t0 = time.perf_counter()
    lines = []
    if cpu:
        compiled = micro_step
        res["compiled_arm"] = "EAGER STAND-IN (cpu smoke)"
    else:
        from agents.model.compile_control import control
        ctl = control(lines.append).install()
        print(ctl.reset(), flush=True)
        cr.install(model)
        ctl.prewarm(cr.prewarm_calls(model, batch_size=model.batch_size))
        compiled = model._compiled_micro_step
        res["compiled_arm"] = "compile_regions.install (fullgraph, static, default preset), bf16 rounds"
    _sync()
    res["phases_s"]["compile_prewarm"] = time.perf_counter() - t0
    save()

    policy = model.policy

    def arms(slice_):
        b = cr.r1_batch(model, int(model.batch_size), slice_)
        out = {}
        was = policy.training
        policy.set_training_mode(True)
        try:
            with torch.random.fork_rng(devices=[] if cpu else [torch.cuda.current_device()]):
                assert trunk_precision_of(policy) == "bf16"
                rargs = cr._r1_args(model, b)
                out["Cbf"] = cr._r1_arm(model, compiled, rargs)
                out["Ebf"] = cr._r1_arm(model, micro_step, rargs)
                out["Cbf2"] = cr._r1_arm(model, compiled, rargs)
                out["Ebf2"] = cr._r1_arm(model, micro_step, rargs)
                set_trunk_precision(policy, "fp32")
                try:
                    out["E32"] = cr._r1_arm(model, micro_step, rargs)
                    out["E32b"] = cr._r1_arm(model, micro_step, rargs)
                finally:
                    set_trunk_precision(policy, "bf16")
            # the decision readout, eval mode, no grad: bf16 eager vs fp32 eager (the PPO epoch-0 ratio noise)
            policy.set_training_mode(False)
            obs = {k: v for k, v in b.obs.items()}
            mask = b.action_masks.bool().cpu().numpy()
            dbf = cr._decision_readout(model, obs, mask)
            set_trunk_precision(policy, "fp32")
            try:
                d32 = cr._decision_readout(model, obs, mask)
            finally:
                set_trunk_precision(policy, "bf16")
        finally:
            policy.set_training_mode(was)
            for p in policy.parameters():
                p.grad = None
        sizes = [int(x) for x in out["E32"]["grad_sizes"].tolist()]
        e32n = [float(g.norm()) for g in torch.split(out["E32"]["grad"].double(), sizes)]
        top = max(e32n) if e32n else 0.0
        pairs = {f"{x}_vs_{y}": pair_stats(out[x], out[y], sizes) for x, y in (
            ("Cbf", "E32"), ("Ebf", "E32"), ("Cbf2", "E32"), ("Ebf2", "E32"), ("E32b", "E32"),
            ("Cbf", "Ebf"), ("Ebf2", "Ebf"), ("Cbf2", "Cbf"))}
        legal = torch.as_tensor(mask, device=dbf["legal_logprob"].device)
        dl = (dbf["legal_logprob"] - d32["legal_logprob"]).abs()[legal].double().cpu().numpy()
        dv = (dbf["value"] - d32["value"]).abs().double().cpu().numpy()
        q = lambda x, p: float(np.quantile(x, p)) if x.size else float("nan")  # noqa: E731
        return {"source": b.source, "grad_norm_ratio_E32": [n / top if top else 0.0 for n in e32n],
                "losses": {k: float(v["loss"]) for k, v in out.items()}, "pairs": pairs,
                "decision_bf16_vs_fp32": {
                    "legal_logprob_abs": {"p50": q(dl, .5), "p99": q(dl, .99), "p999": q(dl, .999),
                                          "max": float(dl.max()) if dl.size else float("nan"), "n": int(dl.size)},
                    "value_abs": {"p50": q(dv, .5), "p99": q(dv, .99), "max": float(dv.max()), "n": int(dv.size)}}}

    def read(tag, ctx=contextlib.nullcontext):
        t = time.perf_counter()
        try:
            with ctx():
                res["states"][tag] = {f"slice{s}": arms(s) for s in (0, 1)}
        except Exception as exc:  # noqa: BLE001
            res["states"][tag] = {"error": f"{type(exc).__name__}: {exc}", "tb": traceback.format_exc()[-3000:]}
        _sync()
        res["states"][tag]["seconds"] = time.perf_counter() - t
        if not cpu:
            # the class swap (fp32 reference arm) must not cost R1 a recompile: the compiled route's cache entries
            from agents.model.compile_control import cache_entries_by_code
            res["states"][tag]["r1_cache_entries"] = sum(
                v for k, v in cache_entries_by_code().items() if k.split(" ")[0] == "micro_step")
        s0 = res["states"][tag].get("slice0", {}).get("pairs", {})
        print(f"[envelope] {tag}: " + ", ".join(
            f"{k} loss {v['loss_rel']:.2e} cos {v['cos']:.6f} pmax {max(v['per_param_rel']):.2e}"
            for k, v in s0.items() if k in ("Cbf_vs_E32", "Ebf_vs_E32", "Cbf_vs_Ebf", "E32b_vs_E32")), flush=True)
        save()

    read("fresh")
    for scale, k in ((0.05, 0), (0.05, 1), (0.05, 2), (0.05, 3), (0.1, 0), (0.1, 1)):
        read(f"pert{int(round(scale * 100)):02d}_k{k}",
             lambda s=scale, k=k: perturbed_parameters(policy, seed=rung_seed(k), scale=s))

    # ---- partly trained weights: real 1-epoch train() calls through the compiled bf16 R1 on the fixture rollout
    live_buf = model.rollout_buffer
    model.behaviour_check = "off"
    done = 0
    rows_per = UB.N_STEPS * UB.N_ENVS
    for target in [int(x) for x in a.train_calls.split(",") if x]:
        t = time.perf_counter()
        while done < target:
            model.rollout_buffer = live_buf
            b, _ = UF.fixture_buffer(model)
            model.rollout_buffer = b
            model.num_timesteps = rows_per * (1 + done)
            model.n_epochs = 1
            model.train()
            done += 1
        _sync()
        res["phases_s"][f"train_to_{target}"] = time.perf_counter() - t
        model.rollout_buffer = live_buf
        read(f"trained_{target}")
    res["load1_end"] = os.getloadavg()[0]
    save()
    print(f"[envelope] DONE -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
