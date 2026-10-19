"""WHERE in the edge-bias OUTPUT the violators' fp32 error sits (`locate.py` found the family maps clean).

Hooks `features_extractor.edge_bias` (its incoming ``bias`` argument and its output) in an fp32 and a float64
forward of the T2 decision module on the same weights, and reports for the violator the largest-error entries
[head, query, key] with their fp64 values, the incoming bias's own error at those entries, and the same
statistics' median over the other 32 rows. CPU only, pinned code:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<pin_9c033489>/src CUDA_VISIBLE_DEVICES= \
        python3 locate_bias.py --in <copies> --out locate_bias.json
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from typing import Any, Dict

import numpy as np
import torch as th

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("x26_a3_replay_neighbours", os.path.join(_HERE, "replay_neighbours.py"))
RN = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(RN)  # type: ignore[union-attr]
R = RN.R


def capture(template: Any, sd: Dict[str, th.Tensor], obs: np.ndarray, masks: np.ndarray, dtype: th.dtype) -> Dict[str, np.ndarray]:
    from agents.inference.service.decision import DecisionModule
    p = copy.deepcopy(template)
    p.load_state_dict(sd)
    if dtype == th.float64:
        p = p.double()
    got: Dict[str, np.ndarray] = {}

    def pre(_m: Any, args: Any) -> None:
        if "in" not in got:
            got["in"] = args[0].detach().double().numpy().copy()

    def post(_m: Any, _args: Any, out: Any) -> None:
        if "out" not in got:
            t = out[0] if isinstance(out, tuple) else out
            got["out"] = t.detach().double().numpy().copy()
    eb = p.features_extractor.edge_bias
    h1, h2 = eb.register_forward_pre_hook(pre), eb.register_forward_hook(post)
    try:
        with th.no_grad(), R._ctx(dtype):
            DecisionModule(p).eval()(th.as_tensor(obs).to(dtype), th.as_tensor(masks).bool())
    finally:
        h1.remove()
        h2.remove()
    return got


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    th.set_float32_matmul_precision("highest")
    th.set_num_threads(int(os.environ.get("K9B_THREADS", "3")))
    from main.capacity import load_policy
    model, _cfg, _ = load_policy(os.path.join(a.inp, "checkpoint_316000231_steps.zip"), a.inp, "cpu")
    template = copy.deepcopy(model.policy)
    del model
    recs = {json.loads(line)["n_updates"]: json.loads(line) for line in open(os.path.join(a.inp, "behaviour_violations.jsonl"))}
    sds = {k: RN.policy_sd(os.path.join(a.inp, f"checkpoint_{k}_steps.zip")) for k in RN.CKPTS}
    res: Dict[str, Any] = {}
    for u, (lo, hi) in RN.BRACKET.items():
        rec = recs[u]
        alpha = (rec["num_timesteps"] - RN.CKPTS[lo]) / (RN.CKPTS[hi] - RN.CKPTS[lo])
        d = RN.RD.rows_of(np.load(os.path.join(a.inp, f"behaviour_violation_u{u}_rows.npz")))
        o, m = d["obs"]["observation"], d["obs"]["action_mask"].astype(bool)
        per: Dict[str, Any] = {}
        for name, sd in {f"ckpt_{lo}": sds[lo], f"ckpt_{hi}": sds[hi], f"interp_{alpha:.3f}": RN.interp(sds[lo], sds[hi], alpha)}.items():
            c32, c64 = capture(template, sd, o, m, th.float32), capture(template, sd, o, m, th.float64)
            row: Dict[str, Any] = {"shape": list(c64["out"].shape)}
            for key in ("in", "out"):
                v32, v64 = c32[key], c64[key]
                fin = np.isfinite(v32) & np.isfinite(v64) & (np.abs(v64) < 1e6)
                err = np.where(fin, np.abs(v32 - v64), 0.0).reshape(v64.shape[0], -1)
                row[f"{key}_abs_err_max_violator"] = float(err[0].max())
                row[f"{key}_abs_err_max_others_median"] = float(np.median(err[1:].max(1)))
                row[f"{key}_nonfinite_or_huge_violator"] = int((~fin.reshape(v64.shape[0], -1)[0]).sum())
                if key == "out":
                    top = np.argsort(-err[0])[:6]
                    shp = v64.shape[1:]
                    row["out_top"] = [{"idx": [int(x) for x in np.unravel_index(j, shp)], "v64": float(v64.reshape(v64.shape[0], -1)[0, j]),
                                       "v32": float(v32.reshape(v32.shape[0], -1)[0, j]),
                                       "in_err_here": float(np.abs(c32["in"] - c64["in"]).reshape(v64.shape[0], -1)[0, j])} for j in top]
            per[name] = row
            print(f"u{u} {name}: " + json.dumps(row), flush=True)
        res[str(u)] = per
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
