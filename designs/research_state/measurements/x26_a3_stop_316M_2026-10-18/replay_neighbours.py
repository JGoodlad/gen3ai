"""X26's Amendment-3 stop (u32180 @ 316.80M, 2.97e-3; u32320 @ 318.17M, 8.1e-4): the replay on NEIGHBOUR weights.

The two events have ROW dumps only (the per-process weight-dump cap was spent), so the exact weights that
played them are gone. This replays the dumped rows (the violator + the scan's 32 worst) on the nearest
checkpoints instead, with X26's PINNED code, CPU only:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<pin_9c033489>/src CUDA_VISIBLE_DEVICES= \
        python3 replay_neighbours.py --in <copies> --out replay_neighbours.json

Per dump and per weight set W (each bracketing checkpoint, and the step-linear interpolation between them at
the event's step), it runs `../x26_k9b_tail_refresh_2026-10-17/replay_dumps.py`'s mechanism check unchanged
(ref64, the fp32 learner / decision paths, the one-ulp dither) and `summarise`. What that can say here:

* the CONDITIONING of each row near the true weights (fp32 error vs ref64, dither std) -- a property of the
  row and the local function, which neighbour weights estimate;
* NOT the gate gap itself: neighbour weights move every row by far more than the gap, so "stored - ref64(W)"
  measures the weight distance, not rounding (reported as `weight_distance_*` so it is never misread).

Two further reads, both fp64 (the exact function, no rounding):

* the WEIGHT-PATH SWEEP: full masked log-probs of every row along W(alpha) = (1 - alpha) A + alpha B between
  the two bracketing checkpoints. A discrete site that a row crosses on the path shows as a JUMP (a second
  difference far above the path's smooth curvature); its size is what a flip of that site moves the row by;
* the declared TIE MARGINS (`tie_margins.TieMargins`, keep_calls) of each row's learner forward under each W:
  how near each declared selection / threshold sits to its cutoff, per site.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import io
import json
import os
import time
import zipfile
from typing import Any, Dict, List

import numpy as np
import torch as th

_HERE = os.path.dirname(os.path.abspath(__file__))


def _load(name: str, rel: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, os.path.join(_HERE, "..", rel))
    mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


RD = _load("k9b_replay_dumps_d29f781c", "x26_k9b_tail_refresh_2026-10-17/replay_dumps.py")
R = RD.R

CKPTS = {"316000231": 316_000_231, "317000041": 317_000_041, "320000071": 320_000_071}
#: each event's bracketing checkpoints (before, after)
BRACKET = {32180: ("316000231", "317000041"), 32320: ("317000041", "320000071")}


def policy_sd(path: str) -> Dict[str, th.Tensor]:
    with zipfile.ZipFile(path) as zf:
        return th.load(io.BytesIO(zf.read("policy.pth")), map_location="cpu")  # type: ignore[no-any-return]


def interp(a: Dict[str, th.Tensor], b: Dict[str, th.Tensor], alpha: float) -> Dict[str, th.Tensor]:
    out = {}
    for k, va in a.items():
        vb = b[k]
        if va.is_floating_point():
            out[k] = ((1.0 - alpha) * va.double() + alpha * vb.double()).to(va.dtype)
        else:
            if not th.equal(va, vb):
                raise RuntimeError(f"non-float tensor {k} differs between the checkpoints: cannot interpolate")
            out[k] = va
    return out


def full_logp64(template: Any, sd: Dict[str, th.Tensor], obs: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """The T2 decision forward's full masked log-probs [n, A] in float64 (the exact function)."""
    from agents.inference.service.decision import DecisionModule
    p = copy.deepcopy(template)
    p.load_state_dict(sd)
    p = p.double()
    with th.no_grad(), R.f64_forward():
        lp, _v = DecisionModule(p).eval()(th.as_tensor(obs).double(), th.as_tensor(masks).bool())
    return lp.double().numpy()


def tie_calls(template: Any, sd: Dict[str, th.Tensor], obs: Dict[str, np.ndarray], acts: np.ndarray,
              masks: np.ndarray) -> Dict[str, Any]:
    """Per row: the min relative margin per declared MARGIN site (the learner forward, fp32, as the probe)."""
    from agents.training.rust_rollout.tie_margins import TieMargins
    p = copy.deepcopy(template)
    p.load_state_dict(sd)
    n = acts.shape[0]
    rec = TieMargins(n, keep_calls=True, near_eps=2e-4)
    o = {k: th.as_tensor(v) for k, v in obs.items()}
    p.set_training_mode(True)
    with th.enable_grad(), rec:
        p.evaluate_actions(o, th.as_tensor(acts).long(), action_masks=th.as_tensor(masks))
    per_site: Dict[str, np.ndarray] = {}
    for site, arr in rec.calls or []:
        a = np.asarray(arr, dtype=np.float64).reshape(n, -1).min(-1)
        per_site[site] = np.minimum(per_site.get(site, np.full(n, np.inf)), a)
    return {"min_margin": [float(x) for x in rec.margin], "site": list(rec.site),
            "per_site_min": {s: [float(x) for x in v] for s, v in per_site.items()},
            "undeclared": dict(rec.undeclared)}


def sweep(template: Any, a: Dict[str, th.Tensor], b: Dict[str, th.Tensor], obs: np.ndarray, masks: np.ndarray,
          n_pts: int) -> Dict[str, Any]:
    """fp64 full log-probs along the straight weight path a -> b; jumps per row."""
    alphas = np.linspace(0.0, 1.0, n_pts)
    lps = np.stack([full_logp64(template, interp(a, b, float(al)), obs, masks) for al in alphas])  # [P, n, A]
    lps = np.where(masks[None].astype(bool), lps, np.nan)
    d2 = np.abs(lps[2:] - 2 * lps[1:-1] + lps[:-2])                                     # [P-2, n, A]
    d2max = np.nanmax(d2, axis=2)                                                       # [P-2, n]
    d1 = np.nanmax(np.abs(np.diff(lps, axis=0)), axis=2)                                # [P-1, n]
    rows = []
    for i in range(obs.shape[0]):
        med = float(np.median(d2max[:, i]))
        j = int(np.argmax(d2max[:, i]))
        rows.append({"i": i, "d2_median": med, "d2_max": float(d2max[j, i]), "d2_max_alpha": float(alphas[j + 1]),
                     "d2_max_over_median": float(d2max[j, i] / med) if med > 0 else None,
                     "step_median": float(np.median(d1[:, i])), "step_max": float(d1[:, i].max()),
                     "total_move_absmax": float(np.nanmax(np.abs(lps[-1, i] - lps[0, i])))})
    return {"alphas": alphas.tolist(), "rows": rows, "violator_logp_path": np.nan_to_num(lps[:, 0], nan=-1e9).tolist()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--sweep-points", type=int, default=101)
    a = ap.parse_args()
    th.set_float32_matmul_precision("highest")
    th.set_num_threads(int(os.environ.get("K9B_THREADS", "3")))
    th.manual_seed(0)
    from main.capacity import load_policy
    t0 = time.time()
    model, _cfg, _ = load_policy(os.path.join(a.inp, "checkpoint_316000231_steps.zip"), a.inp, "cpu")
    template = copy.deepcopy(model.policy)
    del model
    sds = {k: policy_sd(os.path.join(a.inp, f"checkpoint_{k}_steps.zip")) for k in CKPTS}
    recs = {json.loads(line)["n_updates"]: json.loads(line) for line in open(os.path.join(a.inp, "behaviour_violations.jsonl"))}
    out: Dict[str, Any] = {"k_perturb": a.k, "dumps": {}}
    for u, (lo, hi) in BRACKET.items():
        rec = recs[u]
        z = np.load(os.path.join(a.inp, f"behaviour_violation_u{u}_rows.npz"))
        d = RD.rows_of(z)
        alpha = (rec["num_timesteps"] - CKPTS[lo]) / (CKPTS[hi] - CKPTS[lo])
        weights = {f"ckpt_{lo}": sds[lo], f"ckpt_{hi}": sds[hi], f"interp_{alpha:.3f}": interp(sds[lo], sds[hi], alpha)}
        per: Dict[str, Any] = {"alpha_event": alpha, "bracket": [lo, hi], "weights": {}}
        for name, sd in weights.items():
            res = RD.one_dump(template, sd, z, rec, a.k, 20261018 + u)
            res["summary"] = RD.summarise(res)
            rows = res["rows"]
            wd = np.array([abs(r["stored"] - r["ref64"]) for r in rows])
            res["weight_distance_abs_stored_minus_ref64"] = {"violator": float(wd[0]), "others_median": float(np.median(wd[1:])),
                                                             "others_max": float(wd[1:].max())}
            gap = float(rec["rows"][0]["abs_dlogp"])
            pert = np.array([r["pert_std"] for r in rows])
            res["event_gap_over_violator_pert_std"] = gap / float(pert[0])
            res["ties"] = tie_calls(template, sd, d["obs"], d["acts"], d["obs"]["action_mask"].astype(bool))
            per["weights"][name] = res
            s = res["summary"]
            print(f"[u{u} {name}] {time.time() - t0:.0f}s checks={json.dumps(res['checks'])}", flush=True)
            print(f"   pert_std v={s['violator_pert_std']:.2e} rank {s['violator_pert_std_rank_of_n']} others_med "
                  f"{s['others_median_pert_std']:.2e}; fp32 scale v={s['violator_fp32_scale']:.2e} rank "
                  f"{s['violator_fp32_scale_rank_of_n']} others_med {s['others_median_fp32_scale']:.2e}; "
                  f"event gap / v pert_std = {res['event_gap_over_violator_pert_std']:.1f}; "
                  f"weight distance v {wd[0]:.3e} others med {np.median(wd[1:]):.3e}; "
                  f"tie min v {res['ties']['min_margin'][0]:.3g} @ {res['ties']['site'][0]} undeclared {res['ties']['undeclared']}",
                  flush=True)
            with open(a.out, "w") as fh:
                json.dump(out | {"dumps": out["dumps"] | {str(u): per}}, fh, indent=1)
        per["sweep"] = sweep(template, sds[lo], sds[hi], d["obs"]["observation"], d["obs"]["action_mask"].astype(bool),
                             a.sweep_points)
        sv = per["sweep"]["rows"]
        oth = [r["d2_max_over_median"] for r in sv[1:] if r["d2_max_over_median"] is not None]
        print(f"[u{u} sweep] {time.time() - t0:.0f}s violator d2max {sv[0]['d2_max']:.2e} @ alpha {sv[0]['d2_max_alpha']:.2f} "
              f"(x{sv[0]['d2_max_over_median']:.1f} its median; others' max ratio {max(oth):.1f}); "
              f"violator step max {sv[0]['step_max']:.2e} median {sv[0]['step_median']:.2e}; "
              f"total move {sv[0]['total_move_absmax']:.3f}", flush=True)
        out["dumps"][str(u)] = per
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1)
    out["seconds"] = time.time() - t0
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
