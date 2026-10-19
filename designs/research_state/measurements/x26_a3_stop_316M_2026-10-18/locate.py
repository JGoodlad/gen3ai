"""WHERE the violators' fp32 error is born: the edge-bias family maps, fp32 against float64.

`replay_neighbours.py` finds both violators the most fp32-ill-conditioned rows of their dumps under every
neighbour weight set, with the error first appearing in `features_extractor.edge_bias`. This hooks each
edge-bias family's `Linear(cell -> 2 * n_heads)` map and records, per row, its INPUT cell and OUTPUT bias in
an fp32 and a float64 forward of the T2 decision module (same weights), on the bracketing weights and the
interpolation:

* per family: the violator's absolute and relative fp32 error of the input cell and of the output, against
  the other 32 rows' median, and the cell's largest magnitude;
* the cell entries that carry the violator's input error (the largest |cell32 - cell64| positions).

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<pin_9c033489>/src CUDA_VISIBLE_DEVICES= \
        python3 locate.py --in <copies> --out locate.json
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


def capture(template: Any, sd: Dict[str, th.Tensor], obs: np.ndarray, masks: np.ndarray, dtype: th.dtype) -> Dict[str, Any]:
    from agents.inference.service.decision import DecisionModule
    p = copy.deepcopy(template)
    p.load_state_dict(sd)
    if dtype == th.float64:
        p = p.double()
    eb = p.features_extractor.edge_bias
    got: Dict[str, Any] = {}
    hooks = []
    for fam in eb.families:
        lin = getattr(eb, f"{fam}_map")

        def h(_m: Any, inp: Any, out: Any, fam: str = fam) -> None:
            if fam not in got:      # the FIRST call (one per family per forward)
                got[fam] = (inp[0].detach().double().reshape(inp[0].shape[0], -1).numpy().copy(),
                            out.detach().double().reshape(out.shape[0], -1).numpy().copy())
        hooks.append(lin.register_forward_hook(h))
    try:
        with th.no_grad(), R._ctx(dtype):
            lp, _v = DecisionModule(p).eval()(th.as_tensor(obs).to(dtype), th.as_tensor(masks).bool())
    finally:
        for hk in hooks:
            hk.remove()
    return {"fams": got, "logp": lp.double().numpy()}


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
    out: Dict[str, Any] = {}
    for u, (lo, hi) in RN.BRACKET.items():
        rec = recs[u]
        alpha = (rec["num_timesteps"] - RN.CKPTS[lo]) / (RN.CKPTS[hi] - RN.CKPTS[lo])
        d = RN.RD.rows_of(np.load(os.path.join(a.inp, f"behaviour_violation_u{u}_rows.npz")))
        o, m = d["obs"]["observation"], d["obs"]["action_mask"].astype(bool)
        per: Dict[str, Any] = {}
        for name, sd in {f"ckpt_{lo}": sds[lo], f"ckpt_{hi}": sds[hi], f"interp_{alpha:.3f}": RN.interp(sds[lo], sds[hi], alpha)}.items():
            c32, c64 = capture(template, sd, o, m, th.float32), capture(template, sd, o, m, th.float64)
            fams = {}
            for fam in sorted(c64["fams"]):
                i32, o32 = c32["fams"][fam]
                i64, o64 = c64["fams"][fam]
                fin = np.isfinite(i64) & np.isfinite(i32)
                ie = np.where(fin, np.abs(i32 - i64), 0.0)
                oe = np.abs(o32 - o64).max(1)
                in_abs = ie.max(1)
                in_rel = in_abs / (np.where(fin, np.abs(i64), 0.0).max(1) + 1e-30)
                top = np.argsort(-ie[0])[:5]
                fams[fam] = {"in_abs_err_violator": float(in_abs[0]), "in_abs_err_others_median": float(np.median(in_abs[1:])),
                             "in_rel_err_violator": float(in_rel[0]), "in_rel_err_others_median": float(np.median(in_rel[1:])),
                             "out_abs_err_violator": float(oe[0]), "out_abs_err_others_median": float(np.median(oe[1:])),
                             "in_absmax_violator": float(np.abs(np.where(fin[0], i64[0], 0)).max()),
                             "in_absmax_others_median": float(np.median(np.abs(np.where(fin[1:], i64[1:], 0)).max(1))),
                             "violator_top_err_entries": [{"flat_index": int(j), "v64": float(i64[0, j]), "v32": float(i32[0, j])} for j in top]}
            ok = m[0]
            per[name] = {"logp_err32_violator_maxabs": float(np.abs(c32["logp"][0] - c64["logp"][0])[ok].max()), "families": fams}
            worst = sorted(fams.items(), key=lambda kv: -kv[1]["out_abs_err_violator"])[:4]
            print(f"u{u} {name}: logp err {per[name]['logp_err32_violator_maxabs']:.2e}; worst families by output err: " +
                  "; ".join(f"{f} out {v['out_abs_err_violator']:.2e} (others {v['out_abs_err_others_median']:.1e}) in {v['in_abs_err_violator']:.2e} "
                            f"(others {v['in_abs_err_others_median']:.1e}) inmax {v['in_absmax_violator']:.3g}" for f, v in worst), flush=True)
        out[str(u)] = per
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
