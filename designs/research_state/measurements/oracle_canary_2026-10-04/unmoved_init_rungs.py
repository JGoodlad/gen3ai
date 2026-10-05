"""CPU eager fp32 vs float64, per parameter, on the oracle FATAL checkpoint, for the parameters the
INIT rule calls unmoved (`gen3_r1_unmoved_init_v1`): on the live weights and on the init rule's unmoved
rung (ONLY those parameters perturbed, the ladder's first rung). It is the noise floor of each rung,
which is what the rule's bars sit on. It says nothing about CUDA compile parity.

Usage (repo root): unmoved_init_rungs.py <unmoved_init_species_fatal.json> <out.jsonl> [B]"""
import json
import os
import sys
import time

import torch as th

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import repro as R  # noqa: E402

from agents.model import compile_regions as cr  # noqa: E402
from agents.training.instrumented_ppo.micro_step import micro_step  # noqa: E402


def main():
    listing, out = sys.argv[1], sys.argv[2]
    B = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
    ident = json.load(open(listing))["bit_identical"]
    th.set_float32_matmul_precision("highest")
    path = R.STATES["oracle_fatal"]
    judged_names = [n for n, _ in R._params(R.build(path))]
    unmoved = [n for n in ident if n in judged_names]
    cr.unmoved_parameters = lambda model: list(unmoved)       # the init rule's set, measured
    for perturb in (False, "unmoved"):
        t0 = time.time()
        g64, names, _ = R.r64(path, B, perturb)
        m = R.build(path)
        with R._ctx(m, perturb):
            m.policy.set_training_mode(True)
            x32, _, _ = R.grads(m, micro_step, cr._r1_args(m, cr.r1_batch(m, B)))
        sizes = g64["grad_sizes"]
        x64 = R.rel(x32["grad"], g64["grad"], sizes)
        norms = [float(x.norm()) for x in th.split(g64["grad"], sizes)]
        top = max(norms)
        row = {"state": "oracle_fatal", "rung": "live" if not perturb else "init_rule_unmoved", "B": B,
               "unmoved_judged": unmoved,
               "per_param": {n: {"x64": x64.get(names.index(n)),
                                 "grad_norm_frac": norms[names.index(n)] / top} for n in unmoved},
               "x64_max": max(x64.values()), "x64_argmax": names[max(x64, key=x64.get)], "n_judged": len(x64),
               "secs": round(time.time() - t0, 1)}
        with open(out, "a") as fh:
            fh.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
