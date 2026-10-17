"""Row-level attribution of the bf16 loss drift + the FUNCTION's own conditioning per row (scratch, agent a57c).

Per checkpoint (relocated golden rows, as the gate): per row A, rho, delta_bf = logpi_bf16 - logpi_fp32, the row's
policy-term change dterm = -(A rho delta-ish exact)/B. Under N weight dithers (rel 2^-9, half a bf16 ulp): per row the
bf16 shift at each seed AND the fp32 function's own move (fp32 at the dithered weights vs fp32 at the weights) --
a row whose fp32 log-prob moves by ~ its bf16 shift under a half-ulp dither sits on an ill-conditioned point of the
FUNCTION (a near-tie inside the forward), and bf16 rounding of the same size lands there.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drift as DR  # noqa: E402
from main.ops import diag_bf16_canary as D  # noqa: E402


def main() -> int:
    specs = sys.argv[1].split(",")
    n = int(sys.argv[2])
    out = Path(sys.argv[3])
    torch.set_num_threads(1)
    run_dir = D.resolve_run("rb_x26_s2601")
    D.select_buffer(run_dir, "golden")
    D.RUN_BATCH = 64
    res = [D.resolve_checkpoint(run_dir, s) for s in specs]
    model = D.build_learner(res[0][1], "cpu", batch=64)
    names, sizes = D.param_table(model)
    from agents.model import compile_regions as cr
    allrec = {}
    for label, z in res:
        D.load_weights([model], z)
        args0 = cr._r1_args(model, cr.r1_batch(model, 64, 0))
        clip = float(args0[-1].clip_range)
        r0 = DR.run(model, args0, fp32=True, names=names)
        fix = cr.clip_relocation(r0["log_prob"].float(), args0[4], r0["adv"].float(), clip, cr.R1_CLIP_MARGIN)
        args = args0[:4] + (fix.old_log_prob,) + args0[5:]
        e32 = DR.run(model, args, fp32=True, names=names)
        ebf = DR.run(model, args, fp32=False, names=names)
        old = args[4].double()
        rho = (e32["log_prob"] - old).exp()
        A = e32["adv"]
        d_bf = ebf["log_prob"] - e32["log_prob"]
        dterm = (DR.pg_rows(ebf, args[4], clip) - DR.pg_rows(e32, args[4], clip)) / 64
        seeds_bf, seeds_f32 = [], []
        for s in range(n):
            with D.dithered([model], 2000 + s):
                x32 = DR.run(model, args, fp32=True, names=names)     # SAME relocated old log-probs: rows comparable
                xbf = DR.run(model, args, fp32=False, names=names)
            seeds_bf.append(xbf["log_prob"] - x32["log_prob"])
            seeds_f32.append(x32["log_prob"] - e32["log_prob"])
        sb, sf = torch.stack(seeds_bf), torch.stack(seeds_f32)
        top = torch.argsort(dterm.abs(), descending=True)[:5]
        rows = []
        for i in top.tolist():
            rows.append({"row": i, "A": float(A[i]), "rho": float(rho[i]), "relocated": bool(fix.moved[i]),
                         "dterm": float(dterm[i]), "d_bf": float(d_bf[i]),
                         "null_bf_absmax": float(sb[:, i].abs().max()), "null_bf_absmed": float(sb[:, i].abs().median()),
                         "fp32_dither_absmax": float(sf[:, i].abs().max()),
                         "fp32_dither_absmed": float(sf[:, i].abs().median())})
        # whole-fixture reference: the median row's sensitivities
        allrec[label] = {"loss_diff": ebf["loss"] - e32["loss"], "dterm_sum": float(dterm.sum()), "top": rows,
                         "share_top1": float(dterm[top[0]] / dterm.sum()) if float(dterm.sum()) != 0 else None,
                         "all_rows_bf_absmed": float(sb.abs().median()),
                         "all_rows_fp32_dither_absmed": float(sf.abs().median())}
        print(label, json.dumps({k: v for k, v in allrec[label].items() if k != "top"}), flush=True)
        for r in rows:
            print("   ", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()}, flush=True)
    out.write_text(json.dumps(allrec, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
