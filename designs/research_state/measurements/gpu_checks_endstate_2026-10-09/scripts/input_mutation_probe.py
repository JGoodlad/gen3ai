"""Does R1 MUTATE its own inputs? (gpu_checks_endstate, 2026-10-09). CPU.

    python input_mutation_probe.py --cfgs P,R,E

The gate's batch (`compile_regions.r1_batch`) for each configuration's learner (seed 42 + the gate's perturbed
rung 0); every tensor of R1's argument tuple is cloned, ONE eager `micro_step` (+ backward) runs, and every input
tensor that changed is reported (key, how many elements, max |Δ|). An input mutated in place is a hazard for the
compiled graph: Inductor functionalizes the mutation, and a graph that reads the input's PRE-mutation value
anywhere computes something eager never does — and the first consumer of a fresh batch (the training update, the
startup gate's compiled arm) is the one that sees it.
"""
import argparse
import os
import sys

import torch as th

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import noise_cfg as NC                                     # noqa: E402
from agents.model import compile_regions as cr            # noqa: E402
from agents.training.instrumented_ppo.micro_step import micro_step   # noqa: E402


def flat(args):
    out = {}
    for i, x in enumerate(args):
        if th.is_tensor(x):
            out[f"arg{i}"] = x
        elif isinstance(x, dict):
            for k, v in x.items():
                if th.is_tensor(v):
                    out[f"arg{i}.{k}"] = v
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", default="P,R,E")
    a = ap.parse_args()
    NC.MODEL_SEED = 42
    for cfg in a.cfgs.split(","):
        model, cm, _ = NC.build(cfg, 0)
        cm.__enter__()
        b = cr.r1_batch(model, 256)
        model.policy.set_training_mode(True)
        args = NC.args_for(model, model.policy, b, th.device("cpu"))
        before = {k: v.detach().clone() for k, v in flat(args).items()}
        out = micro_step(*args)
        out.loss.backward()
        changed = []
        for k, v in flat(args).items():
            if not th.equal(v.detach(), before[k]):
                d = (v.detach().double() - before[k].double()).abs()
                changed.append((k, int((d > 0).sum()), float(d.max())))
        print(cfg, "MUTATED inputs:", changed if changed else "none", flush=True)


if __name__ == "__main__":
    main()
