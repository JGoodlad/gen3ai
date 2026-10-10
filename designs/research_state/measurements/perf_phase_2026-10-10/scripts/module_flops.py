"""Per-MODULE FLOPs of ONE R1 micro-step (eager, CPU, forward + backward) — torch.utils.flop_counter.

    python module_flops.py --arch endstate --rows 256 --out ../results/module_flops_endstate.json

Where the matmul FLOPs live by module: which modules a declared bf16 region must cover to move most of them.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cpu_flops import resolved_args  # noqa: E402  (also puts this checkout's src first)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arch", default="endstate")
    ap.add_argument("--rows", type=int, default=256)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    from torch.utils.flop_counter import FlopCounterMode

    from agents.model import compile_regions as cr
    from agents.training import learner_golden as LG
    import agents.training.instrumented_ppo.micro_step as MS
    model = LG.build_learner(args=resolved_args(["--arch", a.arch]))
    LG.load_buffer_into(model)
    b = cr.r1_batch(model, a.rows)
    st = model._micro_static(model._resolve_fold_flags())
    pol = model.policy
    pol.set_training_mode(True)
    fc = FlopCounterMode(display=False, depth=None)
    with fc:
        out = MS.micro_step(pol, b.obs, b.actions, b.action_masks, b.old_log_prob, b.old_values,
                            b.advantages, b.returns, st)
        out.loss.backward()
    counts = fc.get_flop_counts()
    tot = sum(counts.get("Global", {}).values())
    rows = []
    for mod, d in counts.items():
        s = sum(d.values())
        if s:
            rows.append((mod, s, {str(k): v for k, v in d.items()}))
    rows.sort(key=lambda r: -r[1])
    res = {"arch": a.arch, "rows": a.rows, "total_flops": tot,
           "modules": [{"module": m, "flops": s, "frac": s / tot if tot else 0.0, "ops": ops}
                       for m, s, ops in rows]}
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("total GFLOP", tot / 1e9)
    for m, s, ops in rows[:60]:
        if m.count(".") <= 3:
            print(f"{s / tot:7.3f}  {s / 1e9:8.3f}  {m}")


if __name__ == "__main__":
    main()
