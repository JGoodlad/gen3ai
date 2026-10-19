"""Do two equally valid fp32 PROGRAMS disagree on the violators' tau by enough to explain the gap?

`locate_tau.py` shows each violator's species constraint (call 0 of `hypothesis_set.fixed_size_tau`) is nearly
flat at its root (k = 1, slope ~1e-5), so an fp32 evaluation pins tau only to ~ulp(k) / slope ~ 1e-2. T2's
compiled CUDA program and the learner's eager one sum the constraint in different orders. This emulates that:

1. ORDER SPREAD: the violator's (and every other row's) scores, cast to fp32, go through the pinned
   `fixed_size_tau` under P random permutations of the candidate axis (the same arithmetic, a different
   summation order). The spread of tau over the permutations is how far two fp32 programs can sit apart.
2. SENSITIVITY: a float64 forward with call 0's tau shifted by +delta (and -delta) on every row gives
   d log pi(taken) / d tau and the max over legal actions, per row.
3. PREDICTION: order spread x sensitivity, against the observed gate gap of each event.
4. DIRECTION: the cosine between the violator's d log pi / d tau over its legal actions and the observed
   mismatch log mu - log pi (the K9(b) record's full distributions), each centred by its probability-weighted
   mean (a log-prob vector is defined up to its normaliser). |cos| near 1 says the mismatch IS a tau shift.

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<pin_9c033489>/src CUDA_VISIBLE_DEVICES= \
        python3 tau_programs.py --in <copies> --out tau_programs.json
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


def forward64(template: Any, sd: Dict[str, th.Tensor], obs: np.ndarray, masks: np.ndarray, shift: float,
              keep: Dict[str, Any]) -> np.ndarray:
    """Float64 decision forward; call 0 of fixed_size_tau gets +shift; call 0's inputs are kept."""
    from agents.inference.service.decision import DecisionModule
    from agents.model import hypothesis_set as HS
    p = copy.deepcopy(template)
    p.load_state_dict(sd)
    p = p.double()
    orig = HS.fixed_size_tau
    n = {"i": 0}

    def wrapped(scores: th.Tensor, cand: th.Tensor, k: th.Tensor, n_iter: int = HS.BISECTION_ITERS) -> th.Tensor:
        tau = orig(scores, cand, k, n_iter)
        if n["i"] == 0:
            keep.setdefault("scores", scores.detach().double().numpy().copy())
            keep.setdefault("cand", cand.detach().numpy().copy())
            keep.setdefault("k", k.detach().numpy().copy())
            keep.setdefault("tau", tau.detach().double().numpy().copy())
            tau = tau + shift
        n["i"] += 1
        return tau
    HS.fixed_size_tau = wrapped
    try:
        with th.no_grad(), R.f64_forward():
            lp, _v = DecisionModule(p).eval()(th.as_tensor(obs).double(), th.as_tensor(masks).bool())
    finally:
        HS.fixed_size_tau = orig
    return lp.double().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--perms", type=int, default=64)
    ap.add_argument("--delta", type=float, default=1e-3)
    a = ap.parse_args()
    th.set_float32_matmul_precision("highest")
    th.set_num_threads(int(os.environ.get("K9B_THREADS", "3")))
    from agents.model import hypothesis_set as HS
    from main.capacity import load_policy
    model, _cfg, _ = load_policy(os.path.join(a.inp, "checkpoint_316000231_steps.zip"), a.inp, "cpu")
    template = copy.deepcopy(model.policy)
    del model
    recs = {json.loads(line)["n_updates"]: json.loads(line) for line in open(os.path.join(a.inp, "behaviour_violations.jsonl"))}
    sds = {k: RN.policy_sd(os.path.join(a.inp, f"checkpoint_{k}_steps.zip")) for k in RN.CKPTS}
    res: Dict[str, Any] = {}
    rng = np.random.default_rng(20261018)
    for u, (lo, hi) in RN.BRACKET.items():
        rec = recs[u]
        gap = float(rec["rows"][0]["abs_dlogp"])
        alpha = (rec["num_timesteps"] - RN.CKPTS[lo]) / (RN.CKPTS[hi] - RN.CKPTS[lo])
        d = RN.RD.rows_of(np.load(os.path.join(a.inp, f"behaviour_violation_u{u}_rows.npz")))
        o, m, acts = d["obs"]["observation"], d["obs"]["action_mask"].astype(bool), d["acts"]
        per: Dict[str, Any] = {"gap": gap}
        for name, sd in {f"ckpt_{lo}": sds[lo], f"ckpt_{hi}": sds[hi], f"interp_{alpha:.3f}": RN.interp(sds[lo], sds[hi], alpha)}.items():
            keep: Dict[str, Any] = {}
            base = forward64(template, sd, o, m, 0.0, keep)
            up = forward64(template, sd, o, m, a.delta, {})
            dn = forward64(template, sd, o, m, -a.delta, {})
            dlp = (up - dn) / (2 * a.delta)
            n = acts.shape[0]
            sens_taken = np.abs(dlp[np.arange(n), acts])
            sens_max = np.array([np.abs(dlp[i][m[i]]).max() for i in range(n)])
            s32 = th.as_tensor(keep["scores"]).float()
            cand = th.as_tensor(keep["cand"]).bool()
            k = th.as_tensor(keep["k"])
            taus = []
            for _ in range(a.perms):
                perm = th.as_tensor(rng.permutation(s32.shape[-1]))
                taus.append(HS.fixed_size_tau(s32[..., perm], cand[..., perm], k).double().numpy().reshape(n, -1)[:, 0])
            taus = np.array(taus)                                        # [P, n]
            spread = taus.max(0) - taus.min(0)
            err_mean = np.abs(taus.mean(0) - keep["tau"].reshape(n, -1)[:, 0])
            pred = spread * sens_taken
            r0 = rec["rows"][0]
            delta = np.array([np.nan if v is None else v for v in r0["dist_stored"]]) - \
                np.array([np.nan if v is None else v for v in r0["dist_recomputed"]])
            ok = m[0] & np.isfinite(delta)
            pw = np.exp(base[0][ok])
            pw = pw / pw.sum()
            va = dlp[0][ok] - float((pw * dlp[0][ok]).sum())
            vb = delta[ok] - float((pw * delta[ok]).sum())
            cos = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-300))
            implied_dtau = float(va @ vb / (va @ va)) if va @ va > 0 else None
            per[name] = {"tau_order_spread_violator": float(spread[0]), "tau_order_spread_others_median": float(np.median(spread[1:])),
                         "tau_order_spread_rank_of_n": [int((spread > spread[0]).sum()) + 1, n],
                         "tau_order_std_violator": float(taus[:, 0].std()),
                         "tau_fp32_mean_err_vs_fp64_violator": float(err_mean[0]),
                         "sens_taken_violator": float(sens_taken[0]), "sens_taken_others_median": float(np.median(sens_taken[1:])),
                         "sens_maxlegal_violator": float(sens_max[0]),
                         "predicted_gap_scale_violator": float(pred[0]), "predicted_gap_scale_others_median": float(np.median(pred[1:])),
                         "predicted_gap_scale_others_max": float(pred[1:].max()),
                         "predicted_rank_of_n": [int((pred > pred[0]).sum()) + 1, n],
                         "observed_gap_over_predicted_scale": gap / float(pred[0]) if pred[0] > 0 else None,
                         "direction_cosine_dlogp_dtau_vs_observed_delta": cos, "implied_tau_difference": implied_dtau,
                         "violator_dlogp_dtau_legal": dlp[0][ok].tolist(), "observed_delta_legal": delta[ok].tolist(),
                         "per_row": [{"spread": float(x), "sens_taken": float(y), "pred": float(z)} for x, y, z in zip(spread, sens_taken, pred)]}
            r = per[name]
            print(f"u{u} {name}: tau order spread v {r['tau_order_spread_violator']:.2e} (std {r['tau_order_std_violator']:.1e}; rank {r['tau_order_spread_rank_of_n']}; others med "
                  f"{r['tau_order_spread_others_median']:.1e}); d logpi/d tau v {r['sens_taken_violator']:.3f} (max legal {r['sens_maxlegal_violator']:.3f}; others med "
                  f"{r['sens_taken_others_median']:.3f}); predicted |gap| scale v {r['predicted_gap_scale_violator']:.2e} rank {r['predicted_rank_of_n']} "
                  f"(others med {r['predicted_gap_scale_others_median']:.1e}, max {r['predicted_gap_scale_others_max']:.1e}); observed/predicted "
                  f"{r['observed_gap_over_predicted_scale']:.2f}; direction cos {cos:+.3f}, implied tau difference {implied_dtau}", flush=True)
        res[str(u)] = per
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
