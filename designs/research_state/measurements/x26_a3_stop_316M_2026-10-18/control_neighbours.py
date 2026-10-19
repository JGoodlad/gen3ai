"""The CALIBRATION CONTROL for `replay_neighbours.py`: does the neighbour-weight replay keep its power?

The four K9(b) dumps that DO carry their weights (u13900, u19300, u20490, u20630; all diagnosed NUMERICAL by
177f61c0 / d29f781c, the violator ranking 1-2 of 33 on dither spread with a gap of 3.2-3.7 dither std) are
replayed exactly as the two weight-less events are: on their TRUE weights, on the bracketing 1M checkpoints,
and on the step-linear interpolation between those. If a known-numerical violator still ranks at the top of
its dump and its gap stays a few dither std under NEIGHBOUR weights, the row's conditioning is a property
that survives a checkpoint's distance, and the weight-less events' ordinary conditioning under their
neighbours is evidence. If it does not survive, the neighbour replay has no power and says nothing.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<pin_9c033489>/src CUDA_VISIBLE_DEVICES= \
        python3 control_neighbours.py --in <copies> --out control_neighbours.json

u13900 sits on the ABANDONED lineage (A4-F2): 136000153 precedes it on the shared trunk, 137000325 is on the
live lineage, so its "after" neighbour is a sibling, not a successor (reported, not excluded).
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import time
from typing import Any, Dict

import numpy as np
import torch as th

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("x26_a3_replay_neighbours", os.path.join(_HERE, "replay_neighbours.py"))
RN = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(RN)  # type: ignore[union-attr]
RD = RN.RD

#: dump -> (before, after) checkpoint step
CONTROL = {13900: (136_000_153, 137_000_325), 19300: (190_000_228, 191_000_457),
           20490: (201_000_180, 202_000_230), 20630: (203_000_130, 204_000_126)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=32)
    a = ap.parse_args()
    th.set_float32_matmul_precision("highest")
    th.set_num_threads(int(os.environ.get("K9B_THREADS", "3")))
    th.manual_seed(0)
    from main.capacity import load_policy
    t0 = time.time()
    model, _cfg, _ = load_policy(os.path.join(a.inp, "checkpoint_190000228_steps.zip"), a.inp, "cpu")
    template = copy.deepcopy(model.policy)
    del model
    recs = {json.loads(line)["n_updates"]: json.loads(line) for line in open(os.path.join(a.inp, "behaviour_violations.jsonl"))}
    out: Dict[str, Any] = {"k_perturb": a.k, "dumps": {}}
    for u, (lo, hi) in CONTROL.items():
        rec = recs[u]
        z = np.load(os.path.join(a.inp, f"behaviour_violation_u{u}_rows.npz"))
        sd_lo = RN.policy_sd(os.path.join(a.inp, f"checkpoint_{lo}_steps.zip"))
        sd_hi = RN.policy_sd(os.path.join(a.inp, f"checkpoint_{hi}_steps.zip"))
        alpha = (rec["num_timesteps"] - lo) / (hi - lo)
        weights = {"true": th.load(os.path.join(a.inp, f"behaviour_violation_u{u}_policy.pt"), map_location="cpu"),
                   f"ckpt_{lo}": sd_lo, f"ckpt_{hi}": sd_hi, f"interp_{alpha:.3f}": RN.interp(sd_lo, sd_hi, alpha)}
        per: Dict[str, Any] = {"alpha_event": alpha, "steps_to_before": rec["num_timesteps"] - lo,
                               "steps_to_after": hi - rec["num_timesteps"], "weights": {}}
        gap = float(rec["rows"][0]["abs_dlogp"])
        for name, sd in weights.items():
            res = RD.one_dump(template, sd, z, rec, a.k, 20261018 + u)
            rows = res["rows"]
            pert = np.array([r["pert_std"] for r in rows])
            cpu_err = np.array([np.sqrt(np.mean([(r["learn32_cpu_batch"] - r["ref64"]) ** 2, (r["learn32_cpu_b1"] - r["ref64"]) ** 2]
                                                + [(v - r["ref64"]) ** 2 for v in r["dec32_cpu"].values()])) for r in rows])
            wd = np.array([abs(r["stored"] - r["ref64"]) for r in rows])
            per["weights"][name] = {
                "violator_pert_std": float(pert[0]), "others_median_pert_std": float(np.median(pert[1:])),
                "violator_pert_std_rank_of_n": [int((pert > pert[0]).sum()) + 1, len(rows)],
                "violator_cpu_fp32_err": float(cpu_err[0]), "others_median_cpu_fp32_err": float(np.median(cpu_err[1:])),
                "violator_cpu_fp32_err_rank_of_n": [int((cpu_err > cpu_err[0]).sum()) + 1, len(rows)],
                "event_gap_over_violator_pert_std": gap / float(pert[0]),
                "weight_distance_violator": float(wd[0]), "weight_distance_others_median": float(np.median(wd[1:])),
                "pert_std_all": pert.tolist(), "cpu_fp32_err_all": cpu_err.tolist()}
            w = per["weights"][name]
            print(f"[ctrl u{u} {name}] {time.time() - t0:.0f}s pert v {w['violator_pert_std']:.2e} rank "
                  f"{w['violator_pert_std_rank_of_n']} (others med {w['others_median_pert_std']:.2e}); cpu err v "
                  f"{w['violator_cpu_fp32_err']:.2e} rank {w['violator_cpu_fp32_err_rank_of_n']}; gap/std "
                  f"{w['event_gap_over_violator_pert_std']:.1f}; weight distance v {w['weight_distance_violator']:.3e} "
                  f"others med {w['weight_distance_others_median']:.3e}", flush=True)
            out["dumps"][str(u)] = per
            with open(a.out, "w") as fh:
                json.dump(out, fh, indent=1)
    out["seconds"] = time.time() - t0
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
