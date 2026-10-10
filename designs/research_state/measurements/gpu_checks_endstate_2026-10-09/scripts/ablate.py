"""LEVER ABLATION of the E-config R1 gate failure (gpu_checks_endstate, 2026-10-09). GPU, under the lease.

    python ablate.py --out results/ablate.jsonl --variants E,E-obs_facts,E-move_resolution,... [--k 0]

For each variant (E with ONE lever reverted to R's / production's value), the launch-equivalent weight state
(the golden build at the trainer's seed 42, unperturbed — bit-equal to a launch's init, `init_match.py` — then
the R1 gate's perturbed rung k) and the gate's batch: R1's gradient COMPILED (Inductor, fullgraph, dynamic=False)
vs EAGER, both fp32 CUDA, per parameter with the gate's own function and floor. The variant whose reversion
removes the disagreement names the lever whose compiled backward is off; the float64 attribution of the
disagreement to the COMPILED arm is `noise_cfg.py`'s.
"""
import argparse
import json
import os
import sys
import time

import torch as th

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import noise_cfg as NC                                     # noqa: E402
from cuda_parity_targets import CFG                        # noqa: E402
from agents.model import compile_regions as cr            # noqa: E402
from agents.training.instrumented_ppo.micro_step import micro_step   # noqa: E402

E = CFG["E"]


def without(flag, value):
    """E with ``flag``'s value replaced (``value`` None drops the flag and its value)."""
    out, i = [], 0
    while i < len(E):
        if E[i] == flag:
            if value is not None:
                out += [flag, value]
            i += 2
            continue
        out.append(E[i])
        i += 1
    return out


VARIANTS = {
    "E": E,
    "E-obs_facts": without("--obs-facts", "off"),
    "E-move_resolution": without("--move-resolution", "off"),
    "E-speed_physics": without("--speed-physics", "off"),
    "E-op_reduction": without("--op-reduction", "max"),
    "E+value_threat_inject": [x for x in E if x != "--no-value-threat-inject"] + ["--value-threat-inject"],
    "R+obs_facts": CFG["R"] + ["--obs-facts", "v1", "--allow-nonproduction-arch"],
    "P+obs_facts": CFG["P"] + ["--obs-facts", "v1", "--allow-nonproduction-arch"],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--model-seed", type=int, default=42)
    a = ap.parse_args()
    NC.MODEL_SEED = a.model_seed
    th.set_float32_matmul_precision("highest")
    from agents.model.compile_cache import ensure_hermetic_cache
    ensure_hermetic_cache("gpu checks endstate ablate")
    dev = th.device("cuda")
    for v in a.variants.split(","):
        t0 = time.time()
        CFG[v] = VARIANTS[v]
        model, cm, note = NC.build(v, a.k)
        cm.__enter__()
        names = [n for n, _ in model.policy.named_parameters() if not n.startswith("ridealong.")]
        b = cr.r1_batch(model, 2048)
        model.policy.set_training_mode(True)
        model.policy.to(dev)
        model.device = dev
        model.rollout_buffer.device = dev
        args = NC.args_for(model, model.policy, b, dev)
        le, ge, sizes = NC.N.grad_of(model.policy, micro_step, args)
        th._dynamo.reset()
        comp = th.compile(micro_step, fullgraph=True, dynamic=False)
        lc, gc, _ = NC.N.grad_of(model.policy, comp, args)
        th._dynamo.reset()
        gate = NC.N.rel(gc, ge, sizes)
        top = sorted(gate.items(), key=lambda t: -t[1])[:8]
        row = {"variant": v, "argv": VARIANTS[v], "k": a.k, "model_seed": a.model_seed, "n_judged": len(gate),
               "worst": [(names[i], e) for i, e in top], "loss": {"E32": le, "C32": lc},
               "secs": round(time.time() - t0, 1)}
        with open(a.out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(v, f"judged {len(gate)}", [(names[i].replace('features_extractor.', ''), f"{e:.2e}") for i, e in top[:5]],
              f"{row['secs']}s", flush=True)
        del model
        th.cuda.empty_cache()


if __name__ == "__main__":
    main()
