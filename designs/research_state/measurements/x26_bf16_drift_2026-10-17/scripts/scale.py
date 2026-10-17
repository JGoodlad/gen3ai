"""Is row 9's ill-conditioning SMOOTH (|d log pi| linear in the dither scale) or a NEAR-DISCONTINUITY (a near-tie in a
discrete op: |d log pi| of the same size at tiny scales)? fp32 eager only, relocated golden rows, 6 seeds per scale."""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drift as DR  # noqa: E402
from main.ops import diag_bf16_canary as D  # noqa: E402


def main() -> int:
    torch.set_num_threads(1)
    run_dir = D.resolve_run("rb_x26_s2601")
    D.select_buffer(run_dir, "golden")
    D.RUN_BATCH = 64
    from agents.model import compile_regions as cr
    rows = [int(x) for x in sys.argv[2].split(",")]
    for spec in sys.argv[1].split(","):
        label, z = D.resolve_checkpoint(run_dir, spec)
        model = D.build_learner(z, "cpu", batch=64)
        names, _ = D.param_table(model)
        args = cr._r1_args(model, cr.r1_batch(model, 64, 0))
        e = DR.run(model, args, fp32=True, names=names)
        print(label)
        for k in (20, 16, 13, 11, 9, 7):
            ds = []
            for s in range(6):
                with D.dithered([model], 3000 + s, rel=2.0 ** -k):
                    x = DR.run(model, args, fp32=True, names=names)
                ds.append((x["log_prob"] - e["log_prob"]).abs())
            t = torch.stack(ds)
            med = t.median(0).values
            print(f"  rel 2^-{k:<2d}: " + "  ".join(f"row{r} {float(med[r]):.2e}" for r in rows)
                  + f"  all-rows median {float(t.median()):.2e}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
