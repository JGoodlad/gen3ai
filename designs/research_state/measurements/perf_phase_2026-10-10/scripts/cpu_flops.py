"""CPU FLOP / op-class read of ONE R1 micro-step (eager, forward + backward), torch.profiler with_flops.

    python cpu_flops.py --arch endstate --rows 256 --out ../results/cpu_flops_endstate.json

Reports per-row FLOPs per op so a reader can scale to B = 2,048 x 480 micro-batches per update. CPU only.
"""
import argparse
import contextlib
import io
import json
import os
import sys
import time
from collections import defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "..", "..", "..", "..", "..", "src")))
import torch as th  # noqa: E402


def resolved_args(argv, device="cpu"):
    from main.train.config import resolve_config
    from main.train.parser import build_parser
    p = build_parser()
    a = p.parse_args(list(argv) + ["--device", device])
    with contextlib.redirect_stdout(io.StringIO()):
        resolve_config(a, p)
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="endstate")
    ap.add_argument("--cfg", default=None, help="a configs.py tag (overrides --arch)")
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from agents.model import compile_regions as cr
    from agents.training import learner_golden as LG
    import agents.training.instrumented_ppo.micro_step as MS
    print("imports from", MS.__file__, flush=True)
    import shlex
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from configs import CONFIGS
    argv = shlex.split(CONFIGS[a.cfg]) if a.cfg else ["--arch", a.arch]
    model = LG.build_learner(args=resolved_args(argv))
    LG.load_buffer_into(model)
    b = cr.r1_batch(model, a.rows)
    st = model._micro_static(model._resolve_fold_flags())
    pol = model.policy
    pol.set_training_mode(True)

    def step():
        out = MS.micro_step(pol, b.obs, b.actions, b.action_masks, b.old_log_prob, b.old_values,
                            b.advantages, b.returns, st)
        out.loss.backward()
        pol.optimizer.zero_grad()

    step()
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU], with_flops=True) as prof:
        t0 = time.perf_counter()
        step()
        wall = time.perf_counter() - t0
    by = defaultdict(lambda: {"flops": 0, "self_cpu_us": 0.0, "calls": 0})
    for e in prof.key_averages():
        d = by[e.key]
        d["flops"] += int(e.flops or 0)
        d["self_cpu_us"] += float(e.self_cpu_time_total)
        d["calls"] += int(e.count)
    tot_fl = sum(v["flops"] for v in by.values())
    top = sorted(by.items(), key=lambda kv: -kv[1]["self_cpu_us"])[:40]
    nparams = sum(p.numel() for p in pol.parameters())
    res = {"arch": a.arch, "rows": a.rows, "wall_s": wall, "params": nparams,
           "total_flops_counted": tot_fl, "flops_per_row": tot_fl / a.rows,
           "by_op_flops": {k: v for k, v in sorted(by.items(), key=lambda kv: -kv[1]["flops"]) if v["flops"]},
           "top_self_cpu": dict(top)}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({"wall_s": wall, "params": nparams, "GFLOP": tot_fl / 1e9,
                      "flop_ops_GFLOP": {k: round(v["flops"] / 1e9, 3)
                                         for k, v in list(res["by_op_flops"].items())[:12]}}, indent=1))
    print("top self cpu ms:", [(k, round(v["self_cpu_us"] / 1e3, 1), v["calls"]) for k, v in top[:30]])


if __name__ == "__main__":
    main()
