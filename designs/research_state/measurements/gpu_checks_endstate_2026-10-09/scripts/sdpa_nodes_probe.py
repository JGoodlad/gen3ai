"""WHERE does Inductor's pattern matcher put an SDPA call in the E-config R1 graph? (gpu_checks_endstate, 2026-10-09).
GPU, under the lease.

    python sdpa_nodes_probe.py --cfg E [--cfg2 R]

Compiles R1 (`micro_step`, fullgraph, dynamic=False) on the launch-equivalent weight state with a post-grad
custom pass that only RECORDS (never edits) every attention node of the final graph — the SDPA family
(`_scaled_dot_product_{efficient,flash,cudnn}_attention[_backward]`, `_scaled_dot_product_attention_math`) —
with the source line its `stack_trace` meta names, plus Inductor's own counters (`fuse_attention`,
`pattern_matcher_count`). A node whose stack trace points at a manual `softmax(q kᵀ) v` (not at an explicit
`scaled_dot_product_attention` call) was CREATED by the pattern matcher.
"""
import argparse
import collections
import json
import os
import sys

import torch as th

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import noise_cfg as NC                                     # noqa: E402
from agents.model import compile_regions as cr            # noqa: E402
from agents.training.instrumented_ppo.micro_step import micro_step   # noqa: E402


def last_frames(st, n=3):
    if not st:
        return None
    lines = [ln.strip() for ln in st.splitlines() if ln.strip().startswith("File ")]
    return [ln.replace("/home/goodlad/dev/gen3ai/", "") for ln in lines[-n:]]


def probe(cfg, k, seed):
    import torch._inductor.config as ic
    from torch._dynamo.utils import counters
    NC.MODEL_SEED = seed
    dev = th.device("cuda")
    model, cm, _ = NC.build(cfg, k)
    cm.__enter__()
    b = cr.r1_batch(model, 2048)
    model.policy.set_training_mode(True)
    model.policy.to(dev)
    model.device = dev
    model.rollout_buffer.device = dev
    args = NC.args_for(model, model.policy, b, dev)
    found = []

    def record(graph):
        for nd in graph.nodes:
            t = str(nd.target)
            if "scaled_dot_product" in t or "_efficient_attention" in t or "flash_attention" in t:
                found.append({"target": t, "where": last_frames(nd.meta.get("stack_trace"))})

    th._dynamo.reset()
    counters.clear()
    with ic.patch(post_grad_custom_post_pass=record):
        comp = th.compile(micro_step, fullgraph=True, dynamic=False)
        NC.N.grad_of(model.policy, comp, args)
    th._dynamo.reset()
    agg = collections.Counter((f["target"], json.dumps(f["where"])) for f in found)
    out = {"cfg": cfg, "inductor_counters": {kk: v for kk, v in counters["inductor"].items()
                                             if any(s in kk for s in ("attention", "pattern", "sfdp", "fuse"))},
           "attention_nodes": [{"target": t, "where": json.loads(w), "count": c} for (t, w), c in agg.items()]}
    print(json.dumps(out, indent=1), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfgs", default="E")
    ap.add_argument("--k", type=int, default=0)
    ap.add_argument("--model-seed", type=int, default=42)
    ap.add_argument("--out")
    a = ap.parse_args()
    th.set_float32_matmul_precision("highest")
    from agents.model.compile_cache import ensure_hermetic_cache
    ensure_hermetic_cache("gpu checks endstate sdpa nodes")
    res = [probe(c, a.k, a.model_seed) for c in a.cfgs.split(",")]
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
