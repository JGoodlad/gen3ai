"""Is a row's large fp32 dither move SMOOTH (|d log pi| shrinks in proportion to the dither amplitude) or a JUMP (a
discrete op inside the trunk crosses a tie: |d log pi| stays O(1) as the amplitude shrinks)? CPU, eager fp32 only, at
the run's pin. Per checkpoint: the rows whose 16-seed half-ulp (2^-9) fp32 dither move is largest; for each, its worst
seed re-run at relative amplitudes 2^-9 ... 2^-21. Run like margin_read.py:

    python eps_scaling.py --ckpt 380M=<zip> [--ckpt ...] --out eps.json [--top 4] [--seeds 16]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from margin_read import RowTrace, forward  # noqa: E402

RELS = [2.0 ** -k for k in (9, 11, 13, 15, 17, 19, 21)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--top", type=int, default=4)
    ap.add_argument("--seeds", type=int, default=16)
    ap.add_argument("--rows", default="", help="extra rows always read (e.g. 50,31)")
    a = ap.parse_args()
    torch.set_num_threads(6)
    from main.ops import diag_bf16_canary as D
    from agents.model import compile_regions as cr
    res = {}
    for spec in a.ckpt:
        label, zp = spec.split("=", 1)
        B = D.n_fixture_rows()
        model = D.build_learner(Path(zp), "cpu", precision="bf16", batch=B)
        args = cr._r1_args(model, cr.r1_batch(model, B, 0))
        t0 = RowTrace(B)
        ref = forward(model, args, fp32=True, trace=t0)["log_prob"]
        per_seed = []
        for s in range(1, a.seeds + 1):
            with D.dithered([model], s):
                per_seed.append((forward(model, args, fp32=True)["log_prob"] - ref).abs().numpy())
        M = np.stack(per_seed)                      # [seeds, B]
        rows = list(np.argsort(-M.max(0))[: a.top]) + [int(r) for r in a.rows.split(",") if r]
        out = {}
        for r in dict.fromkeys(int(x) for x in rows):
            s = int(M[:, r].argmax()) + 1
            curve = []
            for rel in RELS:
                with D.dithered([model], s, rel=rel):
                    t = RowTrace(B)
                    x = forward(model, args, fp32=True, trace=t)["log_prob"]
                ch = sorted({f"{p[0]}@{p[1]}" for p, q in zip(t0.log, t.log)
                             if p[2] is not None and q[2] is not None and not np.array_equal(p[2][r], q[2][r])})
                curve.append({"rel": rel, "abs_dlogp": float((x - ref).abs()[r]), "n_changed_sites": len(ch),
                              "changed_sites": ch[:12]})
            g = [c["abs_dlogp"] / c["rel"] for c in curve]
            out[str(r)] = {"seed": s, "curve": curve, "gain_dlogp_per_rel": g}
            print(f"[eps] {label} row {r} seed {s}: " + ", ".join(
                f"2^{int(round(np.log2(c['rel'])))}: {c['abs_dlogp']:.3g} ({c['n_changed_sites']} sites)"
                for c in curve), flush=True)
        res[label] = out
        del model
    Path(a.out).write_text(json.dumps(res, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
