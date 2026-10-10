"""ISOLATE the E-config R1 gate failure to a module (gpu_checks_endstate, 2026-10-09). GPU, under the lease.

    python isolate_obs_facts.py --out results/isolate_obs_facts.json [--k 0] [--model-seed 42]

The launch's own weight state (the golden build at the trainer's seed 42, unperturbed — `init_match.py` proved it
bit-equal to a launch's init — then the R1 gate's perturbed rung k), the gate's batch. One EAGER fp32 R1
micro-step on CUDA with hooks captures `ObsFactsInject`'s inputs and the REAL upstream gradient at its output.
Then the module ALONE, `loss = <out, G>` with that G, in three arms — eager fp32 CUDA, Inductor-compiled fp32
CUDA (fullgraph, dynamic=False), eager float64 CPU — and the per-parameter / per-input relative error of each
fp32 arm against float64. If the compiled arm alone is off here, the defect is INSIDE the module's compiled
backward; if not, it comes from the rest of the graph.
"""
import argparse
import json
import os
import sys

import torch as th

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import noise_cfg as NC                                   # noqa: E402
from agents.model import compile_regions as cr          # noqa: E402
from agents.training.instrumented_ppo.micro_step import micro_step   # noqa: E402


def rel(a, b):
    nb = float(b.double().norm())
    return float((a.double() - b.double()).norm()) / nb if nb > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--model-seed", type=int, default=42)
    a = ap.parse_args()
    NC.MODEL_SEED = a.model_seed
    th.set_float32_matmul_precision("highest")
    from agents.model.compile_cache import ensure_hermetic_cache
    ensure_hermetic_cache("gpu checks endstate isolate")
    dev = th.device("cuda")
    model, cm, note = NC.build("E", a.k)
    cm.__enter__()
    fe = model.policy.features_extractor
    mod = fe.obs_facts_inject
    b = cr.r1_batch(model, 2048)
    model.policy.set_training_mode(True)
    model.policy.to(dev)
    model.device = dev
    model.rollout_buffer.device = dev
    cap = {}

    def fwd_hook(m, args, out):
        cap["args"] = [x.detach().clone() if th.is_tensor(x) else x for x in args]
        out.register_hook(lambda g: cap.__setitem__("G", g.detach().clone()))
    h = mod.register_forward_hook(fwd_hook)
    out = micro_step(*NC.args_for(model, model.policy, b, dev))
    out.loss.backward()
    h.remove()
    for p in model.policy.parameters():
        p.grad = None
    facts, role, i_our, i_opp, emb = cap["args"]
    G = cap["G"]
    print("captured:", facts.shape, role.shape, i_our.shape, i_opp.shape, "G", G.shape, float(G.norm()), flush=True)

    def arm(fn, module, embeddings, facts, role, i_our, i_opp, G):
        role = role.clone().requires_grad_(True)
        facts = facts.clone().requires_grad_(False)
        for p in list(module.parameters()) + list(embeddings.parameters()):
            p.grad = None
        o = fn(facts, role, i_our, i_opp, embeddings)
        (o * G).sum().backward()
        res = {f"obs_facts_inject.{n}": p.grad.detach().clone() for n, p in module.named_parameters() if p.grad is not None}
        res.update({f"embeddings.{n}": p.grad.detach().clone() for n, p in embeddings.named_parameters()
                    if p.grad is not None})
        res["d_role_tokens"] = role.grad.detach().clone()
        res["out"] = o.detach().clone()
        return res

    e32 = arm(lambda *x: mod(*x), mod, emb, facts, role, i_our, i_opp, G)
    th._dynamo.reset()
    cfn = th.compile(lambda f, r, i1, i2, e: mod(f, r, i1, i2, e), fullgraph=True, dynamic=False)
    c32 = arm(cfn, mod, emb, facts, role, i_our, i_opp, G)
    th._dynamo.reset()
    import copy
    mod64 = copy.deepcopy(mod).double().cpu()
    emb64 = copy.deepcopy(emb).double().cpu()
    prev = th.get_default_dtype()
    th.set_default_dtype(th.float64)
    try:
        r64 = arm(lambda *x: mod64(*x), mod64, emb64, facts.double().cpu(), role.double().cpu(), i_our.cpu(),
                  i_opp.cpu(), G.double().cpu())
    finally:
        th.set_default_dtype(prev)
    rows = {}
    for k in sorted(r64):
        if k not in e32 or k not in c32:
            continue
        rows[k] = {"c64": rel(c32[k].cpu(), r64[k]), "e64": rel(e32[k].cpu(), r64[k]),
                   "gate": rel(c32[k].cpu(), e32[k].cpu()), "norm64": float(r64[k].norm())}
    for k, v in rows.items():
        print(f"  {k:60s} " + " ".join(f"{kk} {vv:.3e}" if vv is not None else f"{kk} None" for kk, vv in v.items()))
    choice = facts[:, mod._choice[0]:mod._choice[0] + mod._choice[1]]
    info = {"choice_col2_unique_head": th.unique(choice[:, 2]).cpu().tolist()[:40],
            "choice_col2_is_integral": bool((choice[:, 2] == choice[:, 2].round()).all()),
            "choice_cols_absmax": choice.abs().amax(0).cpu().tolist(),
            "rows_with_nonzero_choice_flags": int((choice[:, 0:2].abs().sum(1) > 0).sum())}
    print(json.dumps(info), flush=True)
    json.dump({"note": note, "k": a.k, "model_seed": a.model_seed, "rows": rows, "choice_info": info},
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
