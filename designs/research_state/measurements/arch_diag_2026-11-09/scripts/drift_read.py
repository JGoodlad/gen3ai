"""Architecture diagnostic, the weight-drift read (2026-11-09): do the tanh tower's weights still MOVE?

    drift_read.py <out_dir>     (reads <out_dir>/model_read.json for the per-unit saturation; writes drift_read.json)

Reads X26's banked checkpoints' `policy.pth` (state dicts only; no model code, no forward). For each interval:
the relative displacement ||theta_b - theta_a|| / ||theta_a|| per module group. A layer whose gradient is dead
stops moving under Adam only if its gradient is ~0 against the epsilon; so the read is whether the saturated
layers' displacement falls BELOW the rest of the network's. Per tanh-1 unit: the displacement of its incoming
weight row against the unit's saturation rate."""
from __future__ import annotations

import io
import json
import os
import sys
import zipfile

import numpy as np
import torch as th

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_read import group_of

CK = "/home/goodlad/dev/gen3ai/models/rb_x26_s2601/checkpoints/checkpoint_{}_steps.zip"
STEPS = {"20M": 20000043, "30M": 30000045, "100M": 100000037, "180M": 180000055, "270M": 270000128, "279M": 279000319,
         "280M (P*)": 280000170, "281M": 281000438, "290M": 290000216, "380M": 380000195}
PAIRS = [("20M", "30M"), ("270M", "280M (P*)"), ("280M (P*)", "290M"), ("279M", "280M (P*)"), ("280M (P*)", "281M"),
         ("180M", "280M (P*)"), ("280M (P*)", "380M")]


def load(step):
    with zipfile.ZipFile(CK.format(step)) as z:
        sd = th.load(io.BytesIO(z.read("policy.pth")), map_location="cpu", weights_only=True)
    return {k: v.double() for k, v in sd.items() if v.is_floating_point() and not k.startswith("ridealong.")}


def main(out_dir):
    with open(os.path.join(out_dir, "model_read.json")) as fh:
        mr = json.load(fh)
    sat = np.array(mr["tanh1_unit_sat_rate"]["on"]); tprime = np.array(mr["tanh1_unit_sat_rate"]["mean_tanh_prime_on"])
    sds = {k: load(s) for k, s in STEPS.items()}
    res = {"pairs": {}, "weight_norm": {}}
    for a, b in PAIRS:
        A, Bd = sds[a], sds[b]
        keys = [k for k in A if k in Bd and A[k].shape == Bd[k].shape]
        by = {}
        for k in keys:
            g = group_of(k)
            d = by.setdefault(g, [0.0, 0.0, 0])
            d[0] += float(((Bd[k] - A[k]) ** 2).sum()); d[1] += float((A[k] ** 2).sum()); d[2] += A[k].numel()
        tot = [sum(v[0] for v in by.values()), sum(v[1] for v in by.values())]
        row = {g: {"rel_disp": (v[0] / max(v[1], 1e-30)) ** 0.5, "rms_step_per_param": (v[0] / v[2]) ** 0.5, "elements": v[2]}
               for g, v in by.items()}
        row["ALL"] = {"rel_disp": (tot[0] / tot[1]) ** 0.5}
        for nm in ("mlp_extractor.policy_net.0.weight", "mlp_extractor.policy_net.2.weight",
                   "features_extractor.projection.weight", "pointer_head.ctx_proj.weight"):
            row[nm] = {"rel_disp": float((Bd[nm] - A[nm]).norm() / A[nm].norm()),
                       "rms_step_per_param": float(((Bd[nm] - A[nm]) ** 2).mean().sqrt())}
        # per tanh-1 unit: its incoming row's displacement against its saturation rate (P*'s, on-pool rows)
        w_a, w_b = A["mlp_extractor.policy_net.0.weight"], Bd["mlp_extractor.policy_net.0.weight"]
        rd = ((w_b - w_a).norm(dim=1) / w_a.norm(dim=1)).numpy()
        hi, lo = sat >= 0.99, sat < 0.5
        row["tanh1_units"] = {"corr_rowdisp_vs_sat_rate": float(np.corrcoef(rd, sat)[0, 1]),
                              "corr_rowdisp_vs_mean_tanh_prime": float(np.corrcoef(rd, tprime)[0, 1]),
                              "rowdisp_units_sat_ge99pct": float(rd[hi].mean()) if hi.any() else None, "n_sat_ge99pct": int(hi.sum()),
                              "rowdisp_units_sat_lt50pct": float(rd[lo].mean()) if lo.any() else None, "n_sat_lt50pct": int(lo.sum()),
                              "rowdisp_all": float(rd.mean())}
        res["pairs"][f"{a} -> {b}"] = row
        print(a, b, row["ALL"], flush=True)
    for k, sd in sds.items():
        res["weight_norm"][k] = {nm: float(sd[nm].norm()) for nm in ("mlp_extractor.policy_net.0.weight", "mlp_extractor.policy_net.2.weight",
                                                                     "features_extractor.projection.weight")}
        res["weight_norm"][k]["tanh1_bias_abs_mean"] = float(sd["mlp_extractor.policy_net.0.bias"].abs().mean())
    with open(os.path.join(out_dir, "drift_read.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("done")


if __name__ == "__main__":
    main(sys.argv[1])
