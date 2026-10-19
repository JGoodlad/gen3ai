"""The ORIGIN of the violators' fp32 error: X5's fixed-size presence shift tau (`hypothesis_set.fixed_size_tau`).

`locate_bias.py` puts the whole error in ONE key column of the incoming attention bias, at a value near -12:
a hypothesis key's log-presence log sigma(a + tau). This wraps `hypothesis_set.fixed_size_tau` (looked up as a
module global by `fixed_mass_presence` at call time) and records, per call, the scores ``a``, the candidate
count and mass ``k``, and ``tau``, in an fp32 and a float64 forward of the T2 decision module on the same
weights. Per call it reports, for the violator and the other rows: |tau|, the scores' range over candidates,
tau's fp32 error, and the fp32 error of the logits a + tau. For the species call (call 0) it also reports the
root's CONDITIONING per row: the slope of the constraint at the root, S'(tau) = sum_s pi_s (1 - pi_s), and the
resolution an fp32 sum of the constraint can give tau, ulp(k) / S'(tau) (a sum near k is only known to ~ulp(k),
so the bisection stops telling tau apart inside a band of that width), with its rank correlation against the
measured tau error over the dump's rows. CPU only, pinned code:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<pin_9c033489>/src CUDA_VISIBLE_DEVICES= \
        python3 locate_tau.py --in <copies> --out locate_tau.json
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
from typing import Any, Dict, List

import numpy as np
import torch as th

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("x26_a3_replay_neighbours", os.path.join(_HERE, "replay_neighbours.py"))
RN = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(RN)  # type: ignore[union-attr]
R = RN.R


def capture(template: Any, sd: Dict[str, th.Tensor], obs: np.ndarray, masks: np.ndarray, dtype: th.dtype) -> List[Dict[str, np.ndarray]]:
    from agents.inference.service.decision import DecisionModule
    from agents.model import hypothesis_set as HS
    p = copy.deepcopy(template)
    p.load_state_dict(sd)
    if dtype == th.float64:
        p = p.double()
    calls: List[Dict[str, np.ndarray]] = []
    orig = HS.fixed_size_tau

    def wrapped(scores: th.Tensor, cand: th.Tensor, k: th.Tensor, n_iter: int = HS.BISECTION_ITERS) -> th.Tensor:
        tau = orig(scores, cand, k, n_iter)
        calls.append({"scores": scores.detach().double().numpy().copy(), "cand": cand.detach().numpy().copy(),
                      "k": k.detach().numpy().copy(), "tau": tau.detach().double().numpy().copy()})
        return tau
    HS.fixed_size_tau = wrapped
    try:
        with th.no_grad(), R._ctx(dtype):
            DecisionModule(p).eval()(th.as_tensor(obs).to(dtype), th.as_tensor(masks).bool())
    finally:
        HS.fixed_size_tau = orig
    return calls


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
            assert len(c32) == len(c64)
            calls = []
            for ci, (x32, x64) in enumerate(zip(c32, c64)):
                s64, cand = x64["scores"], x64["cand"].astype(bool)
                tau32, tau64 = x32["tau"], x64["tau"]
                B = s64.shape[0]
                s_flat = s64.reshape(B, -1, s64.shape[-1]) if s64.ndim > 2 else s64[:, None, :]
                c_flat = cand.reshape(B, -1, cand.shape[-1]) if cand.ndim > 2 else cand[:, None, :]
                t32 = tau32.reshape(B, -1)
                t64 = tau64.reshape(B, -1)
                lg32 = x32["scores"].reshape(s_flat.shape) + t32[..., None]
                lg64 = s_flat + t64[..., None]
                lerr = np.where(c_flat, np.abs(lg32 - lg64), 0.0).reshape(B, -1).max(1)
                spread = np.where(c_flat, s_flat, np.nan)
                rng = (np.nanmax(spread, -1) - np.nanmin(spread, -1)) if np.any(c_flat) else np.zeros(t64.shape)
                rng = np.nan_to_num(rng).reshape(B, -1).max(1)
                calls.append({"call": ci, "shape": list(s64.shape),
                              "abs_tau_violator": float(np.abs(t64[0]).max()), "abs_tau_others_median": float(np.median(np.abs(t64[1:]).max(1))),
                              "abs_tau_others_max": float(np.abs(t64[1:]).max()),
                              "score_range_violator": float(rng[0]), "score_range_others_median": float(np.median(rng[1:])),
                              "tau_err32_violator": float(np.abs(t32[0] - t64[0]).max()), "tau_err32_others_median": float(np.median(np.abs(t32[1:] - t64[1:]).max(1))),
                              "logit_err32_violator": float(lerr[0]), "logit_err32_others_median": float(np.median(lerr[1:])),
                              "logit_err32_others_max": float(lerr[1:].max())})
            x64 = c64[0]
            s0 = x64["scores"].reshape(x64["scores"].shape[0], -1)
            cand0 = x64["cand"].reshape(s0.shape).astype(bool)
            t0 = x64["tau"].reshape(-1)
            k0 = x64["k"].reshape(-1).astype(np.float64)
            pi = 1.0 / (1.0 + np.exp(-(s0 + t0[:, None])))
            slope = np.where(cand0, pi * (1 - pi), 0.0).sum(1)
            band = np.spacing(k0.astype(np.float32)).astype(np.float64) / slope
            terr = np.abs(c32[0]["tau"].reshape(-1) - t0)
            from scipy.stats import spearmanr
            rho = spearmanr(band, terr)
            calls[0]["conditioning"] = {"k_violator": float(k0[0]), "slope_violator": float(slope[0]),
                                        "slope_others_median": float(np.median(slope[1:])), "slope_others_min": float(slope[1:].min()),
                                        "band_violator": float(band[0]), "band_others_median": float(np.median(band[1:])),
                                        "band_rank_of_n": [int((band > band[0]).sum()) + 1, int(band.size)],
                                        "spearman_band_vs_tau_err": [float(rho.correlation), float(rho.pvalue)],
                                        "per_row": [{"k": float(a), "slope": float(b), "band": float(c), "tau_err32": float(e), "abs_tau": float(abs(t))}
                                                    for a, b, c, e, t in zip(k0, slope, band, terr, t0)]}
            cc = calls[0]["conditioning"]
            print(f"u{u} {name} call 0 conditioning: k {cc['k_violator']:.0f} slope v {cc['slope_violator']:.3e} (others med {cc['slope_others_median']:.3e}, "
                  f"min {cc['slope_others_min']:.3e}); band ulp(k)/slope v {cc['band_violator']:.2e} rank {cc['band_rank_of_n']} (others med "
                  f"{cc['band_others_median']:.1e}); spearman(band, tau err) {cc['spearman_band_vs_tau_err'][0]:.2f} p {cc['spearman_band_vs_tau_err'][1]:.1e}", flush=True)
            per[name] = calls
            for c in calls:
                print(f"u{u} {name} call {c['call']} {c['shape']}: |tau| v {c['abs_tau_violator']:.4g} (others med {c['abs_tau_others_median']:.3g}, max {c['abs_tau_others_max']:.3g}); "
                      f"score range v {c['score_range_violator']:.4g} (others med {c['score_range_others_median']:.3g}); tau err32 v {c['tau_err32_violator']:.2e} "
                      f"(others {c['tau_err32_others_median']:.1e}); logit err32 v {c['logit_err32_violator']:.2e} (others med {c['logit_err32_others_median']:.1e}, max {c['logit_err32_others_max']:.1e})", flush=True)
        res[str(u)] = per
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
