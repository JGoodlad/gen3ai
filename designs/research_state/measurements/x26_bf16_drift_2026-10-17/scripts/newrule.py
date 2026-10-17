"""The NEW reading (gen3_r1_loss_on_golden_rows_v1) through production's own functions on X26 states, CPU, B = 64 and
2048: eager fp32 reference vs eager bf16 on the relocated rows, then `golden_row_losses` -> the judged drift; beside
it the old (relocated) drift."""
from __future__ import annotations

import sys

import torch

from main.ops import diag_bf16_canary as D


def main() -> int:
    torch.set_num_threads(1)
    from agents.model import compile_regions as cr
    run_dir = D.resolve_run("rb_x26_s2601")
    D.select_buffer(run_dir, "golden")
    B = int(sys.argv[2])
    D.RUN_BATCH = B
    res = [D.resolve_checkpoint(run_dir, s) for s in sys.argv[1].split(",")]
    model = D.build_learner(res[0][1], "cpu", batch=B)
    from agents.training.instrumented_ppo.micro_step import micro_step
    from agents.model.trunk_precision import fp32_scope
    for label, z in res:
        D.load_weights([model], z)
        args0 = cr._r1_args(model, cr.r1_batch(model, B, 0))
        args, fix = cr.relocate_clip_boundary(model, args0)
        with fp32_scope(model.policy):
            ref = cr._r1_arm(model, micro_step, args)
        eager = cr._r1_arm(model, micro_step, args)
        gl = cr.golden_row_losses({"ref": ref, "eager": eager}, fix, [args])
        old = cr._rel(float(eager["loss"]), float(ref["loss"]))
        new = cr._rel(gl["eager"], gl["ref"])
        print(f"{label} B={B}: relocated-row drift {old:.2e}  golden-row drift {new:.2e}  "
              f"(bar {cr.BF16_DRIFT_BARS['loss']}; relocated {int(fix.moved.sum())}/{B})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
