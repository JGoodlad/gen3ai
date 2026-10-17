"""The startup gate's LOSS DRIFT null on X26's elevated-regime states at shift_max 0.18 (today) and 0.28 (the T37 raise
the headroom watch stages), through production's own functions: per checkpoint N dithers (rel 2^-9), each read on the
relocated rows (what X26's pinned gate judges) and on the golden's own rows (gen3_r1_loss_on_golden_rows_v1)."""
from __future__ import annotations

import json
import sys

import torch

from main.ops import diag_bf16_canary as D


def main() -> int:
    torch.set_num_threads(1)
    from agents.model import compile_regions as cr
    from agents.model.trunk_precision import fp32_scope
    from agents.training.instrumented_ppo.micro_step import micro_step
    run_dir = D.resolve_run("rb_x26_s2601")
    D.select_buffer(run_dir, "golden")
    D.RUN_BATCH = 64
    res = [D.resolve_checkpoint(run_dir, s) for s in sys.argv[1].split(",")]
    n = int(sys.argv[2])
    model = D.build_learner(res[0][1], "cpu", batch=64)
    out = {}
    for label, z in res:
        D.load_weights([model], z)
        args0 = cr._r1_args(model, cr.r1_batch(model, 64, 0))
        rec = {}
        for sm in (None, 0.28):
            model._canary_shift_max = sm
            vals = []
            for s in range(-1, n):
                with (D.dithered([model], 5000 + s) if s >= 0 else _null()):
                    args, fix = cr.relocate_clip_boundary(model, args0)
                    with fp32_scope(model.policy):
                        ref = cr._r1_arm(model, micro_step, args)
                    eager = cr._r1_arm(model, micro_step, args)
                    gl = cr.golden_row_losses({"ref": ref, "eager": eager}, fix, [args])
                vals.append((cr._rel(float(eager["loss"]), float(ref["loss"])), cr._rel(gl["eager"], gl["ref"])))
            rec[str(sm or 0.18)] = vals
        model._canary_shift_max = None
        out[label] = rec
        print(label, {k: (max(v[0] for v in vs), max(v[1] for v in vs)) for k, vs in rec.items()}, flush=True)
    json.dump(out, open(sys.argv[3], "w"))
    return 0


class _null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


if __name__ == "__main__":
    sys.exit(main())
