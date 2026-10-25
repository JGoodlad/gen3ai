"""The per-row dither null's COST and its per-row values at this checkout (the design's cost line; README here).

On the K9 golden's committed rows (B = 64, batch row == fixture row), with the learner the canary builds
(`diag_bf16_canary.build_learner`, bf16 trunk, decisions in fp32), train mode, no grad: the fp32 REFERENCE forward
(the trunk under `fp32_scope`: `margin_read.forward`'s E32), then K seeded half-bf16-ulp weight dithers
(`diag_bf16_canary.dithered`, relative 2^-9) of it, per row the max over seeds of |d log pi(taken)|. Times each piece
(the forward at B = 64 and at the canary's B = 2048, the dither's apply + restore). CPU or CUDA (CUDA ONLY under the
GPU lease: gpu_census.sh). Nothing trains; nothing is written under models/.

    python null_cost.py --ckpt <zip> --out cost.json [--device cpu] [--k 16] [--threads 6]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "x26c_canary_margin_2026-10-25"))


def _sync(dev: str) -> None:
    if dev == "cuda":
        torch.cuda.synchronize()


def main() -> int:
    from margin_read import forward  # the x26c read's E32 forward, verbatim
    from main.ops import diag_bf16_canary as D
    from agents.model import compile_regions as cr
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--vf-coef", default=None,
                    help="typed through to the resume resolution (the x26c README's F7: a re-read of rb_x26c_s2601 is "
                         "refused for --vf-coef); the run's model_config.json value. It weights the critic loss only, "
                         "so no log-prob read here depends on it")
    a = ap.parse_args()
    if a.vf_coef is not None:
        import main.ops.cuda_build_validate as V
        _orig = V.resolved_args
        V.resolved_args = lambda argv, device: _orig(list(argv) + ["--vf-coef", str(a.vf_coef)], device)
        out_vf = str(a.vf_coef)
    else:
        out_vf = None
    torch.set_num_threads(a.threads)
    dev = a.device
    out: dict = {"ckpt": a.ckpt, "device": dev, "k": a.k, "threads": a.threads, "torch": torch.__version__,
                 "vf_coef_typed": out_vf}
    t0 = time.perf_counter()
    B = D.n_fixture_rows()
    model = D.build_learner(Path(a.ckpt), dev, precision="bf16", batch=B)
    out["build_s"] = time.perf_counter() - t0
    args = cr._r1_args(model, cr.r1_batch(model, B, 0))
    model.policy.set_training_mode(True)
    forward(model, args, fp32=True)                                   # warm
    ts = []
    for _ in range(a.reps):
        _sync(dev); t = time.perf_counter(); E32 = forward(model, args, fp32=True); _sync(dev)
        ts.append(time.perf_counter() - t)
    out["fwd_fp32_B64_s"] = {"median": float(np.median(ts)), "min": float(min(ts))}
    per_seed, t_dither, t_fwd = [], [], []
    for s in range(1, a.k + 1):
        _sync(dev); t = time.perf_counter()
        cm = D.dithered([model], s)
        cm.__enter__(); _sync(dev); t1 = time.perf_counter()
        x = forward(model, args, fp32=True); _sync(dev); t2 = time.perf_counter()
        cm.__exit__(None, None, None); _sync(dev); t3 = time.perf_counter()
        t_dither.append((t1 - t) + (t3 - t2)); t_fwd.append(t2 - t1)
        per_seed.append((x["log_prob"] - E32["log_prob"]).abs().numpy())
    d = np.stack(per_seed)
    out["null_K_total_s"] = float(sum(t_dither) + sum(t_fwd))
    out["dither_apply_restore_s_median"] = float(np.median(t_dither))
    out["dithered_fwd_s_median"] = float(np.median(t_fwd))
    out["null_per_row_max"] = d.max(0).tolist()
    out["null_max"] = {"value": float(d.max()), "row": int(d.max(0).argmax())}
    # the canary's own batch: the same forward at B = 2048 (the 64 rows tiled) — the cost a full-batch null would pay
    Bc = 2048
    model.batch_size = Bc
    args2 = cr._r1_args(model, cr.r1_batch(model, Bc, 0))
    forward(model, args2, fp32=True)
    ts = []
    for _ in range(max(2, a.reps // 2)):
        _sync(dev); t = time.perf_counter(); forward(model, args2, fp32=True); _sync(dev)
        ts.append(time.perf_counter() - t)
    out["fwd_fp32_B2048_s"] = {"median": float(np.median(ts)), "min": float(min(ts))}
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "null_per_row_max"}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
