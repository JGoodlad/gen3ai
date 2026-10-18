"""The 177f61c0 mechanism check, re-run on every K9(b) dump of rb_x26_s2601 that carries its weights.

Reuses `../x26_k9b_stop_2026-10-17/replay.py`'s forwards unchanged (the float64 TorchFunctionMode reference,
the fp32 learner path, the fp32 T2 decision forward at buckets 8 / 64 / 256, the one-ulp parameter dither).
Run with the RUN'S PINNED code (9c033489) on PYTHONPATH, CPU only:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<pin_9c033489>/src CUDA_VISIBLE_DEVICES= \
        python3 replay_dumps.py --in <copies of the dumps> --out replay_dumps.json

Per dump, the rows are the PROBE violator (``obs__*``, the row the gate judged) followed by the 32 worst rows
of the scan (``scan_obs__*``; under ``warn`` a seeded 4,096-row sample of the buffer, so the violator is
generally NOT among them). Per row: stored mu, the GPU learner's pi (both from the dump), ref64, the fp32 CPU
paths, and the dither spread (``pert_std``, K draws). The class test of 177f61c0 asks, for the violator:
(a) does every fp32 path miss ref64 by a comparable amount (an fp32-ill-conditioned row), (b) does its
dither spread rank at or near the top of the dump's rows, and (c) is its gate gap ordinary FOR ITS
conditioning (|stored - gpu| / pert_std within the other rows' range)?
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import time
from typing import Any, Dict, List

import numpy as np
import torch as th

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "k9b_replay_177f61c0", os.path.join(_HERE, "..", "x26_k9b_stop_2026-10-17", "replay.py"))
R = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(R)  # type: ignore[union-attr]


def rows_of(z: Any) -> Dict[str, Any]:
    """The violator first, then the scan's 32 worst (one array per obs key)."""
    vio = {k[len("obs__"):]: z[k][:1] for k in z.files if k.startswith("obs__")}
    scan = {k[len("scan_obs__"):]: z[k] for k in z.files if k.startswith("scan_obs__")}
    obs = {k: np.concatenate([vio[k], scan[k]]) for k in vio}
    return {"obs": obs,
            "acts": np.concatenate([z["action"][:1], z["scan_action"]]).astype(np.int64),
            "stored": np.concatenate([z["logp_stored"][:1], z["scan_logp_stored"]]).astype(np.float64),
            "gpu": np.concatenate([z["logp_recomputed"][:1], z["scan_logp_recomputed"]]).astype(np.float64),
            "t": np.concatenate([z["t"][:1], z["scan_t"]]), "e": np.concatenate([z["e"][:1], z["scan_e"]])}


def one_dump(template: Any, sd: Dict[str, Any], z: Any, rec: Dict[str, Any], k: int, seed: int) -> Dict[str, Any]:
    from agents.inference.service.decision import DecisionModule
    policy = copy.deepcopy(template)
    policy.load_state_dict(sd)
    base = copy.deepcopy(policy)          # never runs a forward: the dither copies come from it
    d = rows_of(z)
    obs, acts, stored, gpu = d["obs"], d["acts"], d["stored"], d["gpu"]
    masks = obs["action_mask"].astype(bool)
    n = acts.shape[0]
    r0 = rec["rows"][0]
    checks = {"violator_logp_stored_matches_json": bool(float(stored[0]) == r0["logp_stored"]),
              "violator_action_matches_json": bool(int(acts[0]) == r0["action"]),
              "violator_mask_matches_json": bool("".join("1" if x else "0" for x in masks[0]) == r0["mask"]),
              "violator_t_e_match_json": bool(int(d["t"][0]) == r0["buffer_t"] and int(d["e"][0]) == r0["buffer_e"]),
              "violator_gate_d_reproduces": float(abs(stored[0] - gpu[0])),
              "json_abs_dlogp": r0["abs_dlogp"]}
    learn32 = R.learner_logp(policy, obs, acts, masks)
    p64 = copy.deepcopy(base).double()
    ref64 = R.learner_logp(p64, obs, acts, masks, th.float64)
    learn32_b1 = np.array([R.learner_logp(policy, {kk: v[i:i + 1] for kk, v in obs.items()}, acts[i:i + 1],
                                          masks[i:i + 1])[0] for i in range(n)])
    dec = DecisionModule(policy).eval()
    o = obs["observation"]
    dec32 = {b: R.decision_logp(dec, o, masks, acts, b) for b in (64, 256)}
    dec32[8] = np.concatenate([R.decision_logp(dec, o[s:s + 8], masks[s:s + 8], acts[s:s + 8], 8)
                               for s in range(0, n, 8)])
    dec64v = R.decision_logp(DecisionModule(p64).eval(), o, masks, acts, 64, th.float64)
    rl32 = R.raw_logits(policy, dec, o, th.float32)
    rl64 = R.raw_logits(p64, DecisionModule(p64).eval(), o, th.float64)
    rng = np.random.default_rng(seed)
    pert = []
    for _ in range(k):
        pp = R.perturbed(base, rng)
        pert.append(R.decision_logp(DecisionModule(pp).eval(), o, masks, acts, 64))
        del pp
    pert = np.array(pert)
    rows: List[Dict[str, Any]] = []
    for i in range(n):
        legal = masks[i]
        rows.append({"i": i, "role": "violator" if i == 0 else "scan", "t": int(d["t"][i]), "e": int(d["e"][i]),
                     "action": int(acts[i]), "n_legal": int(legal.sum()),
                     "stored": float(stored[i]), "gpu_learner": float(gpu[i]), "ref64": float(ref64[i]),
                     "dec64": float(dec64v[i]), "learn32_cpu_batch": float(learn32[i]),
                     "learn32_cpu_b1": float(learn32_b1[i]),
                     "dec32_cpu": {str(b): float(v[i]) for b, v in dec32.items()},
                     "pert_std": float(pert[:, i].std()), "pert_range": float(pert[:, i].max() - pert[:, i].min()),
                     "p_action": float(np.exp(ref64[i])),
                     "raw_absmax_legal": float(np.abs(rl64["raw"][i][legal]).max()),
                     "raw_legal_err32": [float(x) for x in (rl32["raw"][i] - rl64["raw"][i])[legal]]})
    return {"checks": checks, "rows": rows, "module_relerr_fp32_vs_fp64": R.module_error(rl32, rl64),
            "violator_record": {kk: r0.get(kk) for kk in ("index", "abs_dlogp", "tie_margin", "tie_site",
                                                         "judged_by_flip", "abs_dlogp_as_is", "abs_dlogp_flipped",
                                                         "p_action_stored", "entropy_stored", "mask")}}


def summarise(res: Dict[str, Any]) -> Dict[str, Any]:
    rows = res["rows"]
    ref = np.array([r["ref64"] for r in rows])
    paths = {"stored_T2_gpu": np.array([r["stored"] for r in rows]),
             "learner_gpu": np.array([r["gpu_learner"] for r in rows]),
             "learner32_cpu_batch": np.array([r["learn32_cpu_batch"] for r in rows]),
             "learner32_cpu_b1": np.array([r["learn32_cpu_b1"] for r in rows]),
             **{f"decision32_cpu_b{b}": np.array([r["dec32_cpu"][b] for r in rows]) for b in ("8", "64", "256")}}
    err = {kk: v - ref for kk, v in paths.items()}
    cpu = [kk for kk in err if "cpu" in kk]
    fp32_scale = np.sqrt(np.mean(np.stack([e ** 2 for e in err.values()]), 0))
    pert = np.array([r["pert_std"] for r in rows])
    gate = np.abs(paths["stored_T2_gpu"] - paths["learner_gpu"])
    ratio = gate / pert
    oth = slice(1, None)
    mod = []
    for name, v in res["module_relerr_fp32_vs_fp64"].items():
        v = np.array(v, dtype=float)
        if v.size == len(rows) and np.isfinite(v).all() and np.median(v[oth]) > 0:
            mod.append((float(v[0] / np.median(v[oth])), name))
    mod.sort(reverse=True)
    return {
        "violator_err_vs_ref64": {kk: float(v[0]) for kk, v in err.items()},
        "violator_cpu_fp32_err_range": [float(min(err[kk][0] for kk in cpu)), float(max(err[kk][0] for kk in cpu))],
        "others_median_abs_err_vs_ref64": {kk: float(np.median(np.abs(v[oth]))) for kk, v in err.items()},
        "violator_fp32_scale": float(fp32_scale[0]), "others_median_fp32_scale": float(np.median(fp32_scale[oth])),
        "violator_fp32_scale_rank_of_n": [int((fp32_scale > fp32_scale[0]).sum()) + 1, len(rows)],
        "violator_pert_std": float(pert[0]), "others_median_pert_std": float(np.median(pert[oth])),
        "violator_pert_std_rank_of_n": [int((pert > pert[0]).sum()) + 1, len(rows)],
        "violator_gate_d": float(gate[0]), "violator_gate_over_pert": float(ratio[0]),
        "others_gate_over_pert_median_max": [float(np.median(ratio[oth])), float(np.max(ratio[oth]))],
        "violator_stored_closer_to_ref64_than_learner_gpu": bool(abs(err["stored_T2_gpu"][0]) < abs(err["learner_gpu"][0])),
        "violator_sign_agreement_all_fp32_paths": bool(len({np.sign(v[0]) for v in err.values()}) == 1),
        "fp64_two_paths_max_abs_diff": float(max(abs(r["dec64"] - r["ref64"]) for r in rows)),
        "violator_raw_legal_err32_absmax": float(np.max(np.abs(rows[0]["raw_legal_err32"]))),
        "others_median_raw_legal_err32_absmax": float(np.median([np.max(np.abs(r["raw_legal_err32"])) for r in rows[1:]])),
        "module_relerr_violator_over_others_top5": [{"module": nm, "ratio": rt} for rt, nm in mod[:5]],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--updates", default="13900,19300,20490,20630")
    a = ap.parse_args()
    th.set_float32_matmul_precision("highest")
    th.set_num_threads(int(os.environ.get("K9B_THREADS", "3")))
    th.manual_seed(0)
    from main.capacity import load_policy
    t0 = time.time()
    model, _cfg, _ = load_policy(os.path.join(a.inp, "final_model_exception.zip"), a.inp, "cpu")
    template = copy.deepcopy(model.policy)
    del model
    recs = {json.loads(line)["n_updates"]: json.loads(line) for line in open(os.path.join(a.inp, "behaviour_violations.jsonl"))}
    out: Dict[str, Any] = {"k_perturb": a.k, "dumps": {}}
    for u in [int(x) for x in a.updates.split(",")]:
        sd = th.load(os.path.join(a.inp, f"behaviour_violation_u{u}_policy.pt"), map_location="cpu")
        z = np.load(os.path.join(a.inp, f"behaviour_violation_u{u}_rows.npz"))
        res = one_dump(template, sd, z, recs[u], a.k, 20261017 + u)
        res["summary"] = summarise(res)
        res["num_timesteps"] = recs[u]["num_timesteps"]
        out["dumps"][str(u)] = res
        print(f"[replay] u{u} done {time.time() - t0:.0f}s", json.dumps(res["checks"]), flush=True)
        print(json.dumps(res["summary"], indent=1), flush=True)
        with open(a.out, "w") as fh:            # incremental: each dump is written as it lands
            json.dump(out, fh, indent=1)
    out["seconds"] = time.time() - t0
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
