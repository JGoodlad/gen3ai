"""FP32_ALT at the tau level, on X26's tau-class violators: does the fp32 SPLIT-residual solve land on the float64
root, and do two fp32 PROGRAMS agree on it, where the naive fp32 sum disagreed by the band ulp(k) / S'(tau)?

For each X26 tau-class event dump (u32180, u32320, u35950; the violator + the scan's 32 rows) and each bracketing
checkpoint (read-only copies under ~/.cache/gen3ai):

1. CAPTURE (the A3 diagnosis's `locate_tau` / `tau_programs` method): a float64 decision forward records EVERY
   `fixed_size_tau` call's inputs (call 0 = the species presence over 400 candidates; calls 1.. = the move groups).
2. REF = the float64 SPLIT solve of the fp32-rounded scores (the exact root of what the fp32 programs hold; the
   float64 naive sum is reported beside it: they agree to ~1e-12 on these rows).
3. Per solve in {fp32_naive (pre-T62), fp32_alt (this unit), fp64 (T62)}, on the fp32 scores:
   (a) EAGER |tau - REF|;
   (b) ORDER: P random permutations of the candidate axis (the diagnosis's two-program emulation): the spread;
   (c) COMPILED: an inductor-compiled CPU `fixed_size_tau` (a real second program) against eager.
4. SENSITIVITY (`tau_programs`): d log pi(taken) / d tau from float64 forwards with call 0's tau shifted +-delta, so
   each row's PREDICTED learner-vs-rollout gap scale = order spread x sensitivity, and the violator's rank among
   the 33 rows (the K9(b) class signature: the violator ranked 1st of 33 under the naive fp32 sum).

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<worktree>/src CUDA_VISIBLE_DEVICES= \
        python3 taus.py --out taus.json
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import math
import os
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import torch as th

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "x26_a3_replay_neighbours", os.path.join(_HERE, "..", "x26_a3_stop_316M_2026-10-18", "replay_neighbours.py"))
RN = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(RN)  # type: ignore[union-attr]
R = RN.R

CACHE = os.path.expanduser("~/.cache/gen3ai")
#: event -> (dump dir, [neighbour checkpoints]) — T62's repro set
EVENTS: Dict[int, Tuple[str, List[str]]] = {
    32180: ("x26_a3_stop", ["316000231", "317000041"]),
    32320: ("x26_a3_stop", ["317000041", "320000071"]),
    35950: ("x26_a6_u35950", ["353000151", "354000424"]),
}
#: the policy template (an X26 checkpoint; its weights are replaced) + the run's model_config.json beside it
TEMPLATE = os.path.join(CACHE, "x26_a6_rediag", "u35950", "in", "checkpoint_316000231_steps.zip")
SOLVES = ("fp32_naive", "fp32_alt", "fp64")


class Tap:
    """Wraps `hypothesis_set.fixed_size_tau` (looked up as a module global at call time): keeps every call's
    inputs, and adds ``shift`` to call 0's tau (the sensitivity probe)."""

    def __init__(self, shift: float = 0.0) -> None:
        from agents.model import hypothesis_set as HS
        self.HS, self.orig, self.shift = HS, HS.fixed_size_tau, shift
        self.calls: List[Dict[str, Any]] = []

    def __enter__(self) -> "Tap":
        def wrapped(scores: th.Tensor, cand: th.Tensor, k: th.Tensor, *a: Any, **kw: Any) -> th.Tensor:
            tau = self.orig(scores, cand, k, *a, **kw)
            if len(self.calls) == 0:
                tau = tau + self.shift
            self.calls.append({"scores": scores.detach().double().clone(), "cand": cand.detach().clone(),
                               "k": k.detach().clone(), "solve": kw.get("solve")})
            return tau
        self.HS.fixed_size_tau = wrapped
        return self

    def __exit__(self, *exc: Any) -> None:
        self.HS.fixed_size_tau = self.orig


def set_solve(policy: Any, solve: str) -> None:
    policy.features_extractor.hypothesis_builder.tau_solve = solve


def forward64(policy: Any, obs: np.ndarray, masks: np.ndarray, shift: float = 0.0) -> Tuple[np.ndarray, Tap]:
    from agents.inference.service.decision import DecisionModule
    p = copy.deepcopy(policy).double()
    set_solve(p, "fp32_alt")                      # in float64: the split residual in float64 (the exact root)
    with Tap(shift) as tap, th.no_grad(), R.f64_forward():
        lp, _v = DecisionModule(p).eval()(th.as_tensor(obs).double(), th.as_tensor(masks).bool())
    return lp.double().numpy(), tap


def solve(s: th.Tensor, cand: th.Tensor, k: th.Tensor, how: str, compiled: bool = False) -> np.ndarray:
    from agents.model import hypothesis_set as HS
    import torch._dynamo as dynamo
    if compiled:
        dynamo.reset()
        fn = th.compile(lambda a, b, c: HS.fixed_size_tau(a, b, c, solve=how), fullgraph=True, backend="inductor",
                        dynamic=False)
        out = fn(s, cand, k)
        dynamo.reset()
    else:
        out = HS.fixed_size_tau(s, cand, k, solve=how)
    return out.double().reshape(-1).numpy()


def ulp32(x: np.ndarray) -> np.ndarray:
    return np.array([math.ulp(float(np.float32(abs(v)))) if abs(v) > 0 else 1.4e-45 for v in x.reshape(-1)], dtype=np.float64) * \
        (2 ** 29)        # math.ulp is the float64 ulp; x 2^29 = the float32 ulp of the same binade


def call_report(call: Dict[str, Any], perms: int, rng: np.random.Generator, compiled: bool) -> Dict[str, Any]:
    s64, cand, k = call["scores"], call["cand"], call["k"]
    shp = s64.shape
    s32 = s64.float().reshape(-1, shp[-1])
    cand2 = cand.reshape(-1, shp[-1])
    k1 = k.reshape(-1)
    ref = solve(s32.double(), cand2, k1, "fp32_alt")
    ref_naive64 = solve(s32.double(), cand2, k1, "fp64")
    pi = th.sigmoid(s32.double() + th.as_tensor(ref).unsqueeze(-1))
    slope = th.where(cand2, pi * (1 - pi), th.zeros((), dtype=th.float64)).sum(-1).numpy()
    n = cand2.sum(-1).numpy()
    kk = k1.numpy().astype(np.int64)
    live = (kk > 0) & (kk < n)
    band = np.where(live, np.finfo(np.float32).eps * np.maximum(kk, 1) / np.maximum(slope, 1e-300), 0.0)
    u = ulp32(ref)
    out: Dict[str, Any] = {"rows": int(live.sum()), "band_max": float(band[live].max()) if live.any() else 0.0,
                           "ref_split64_vs_naive64_max": float(np.abs(ref - ref_naive64)[live].max()) if live.any() else 0.0}
    per_row_spread: Dict[str, np.ndarray] = {}
    for how in SOLVES:
        eager = solve(s32, cand2, k1, how)
        taus = []
        for _ in range(perms):
            p = th.as_tensor(rng.permutation(shp[-1]))
            taus.append(solve(s32[:, p], cand2[:, p], k1, how))
        taus_a = np.array(taus)
        spread = np.where(live, taus_a.max(0) - taus_a.min(0), 0.0)
        err = np.where(live, np.abs(eager - ref), 0.0)
        err_orders = np.where(live, np.abs(taus_a - ref).max(0), 0.0)
        r: Dict[str, Any] = {"err_vs_ref_max": float(err.max()), "err_vs_ref_max_ulps": float((err / u).max()),
                             "err_vs_ref_max_over_orders": float(err_orders.max()),
                             "err_over_orders_max_ulps": float((err_orders / u).max()),
                             "order_spread_max": float(spread.max()), "order_spread_max_ulps": float((spread / u).max()),
                             "per_row_err": err.tolist(), "per_row_spread": spread.tolist()}
        if compiled:
            comp = solve(s32, cand2, k1, how, compiled=True)
            ce = np.where(live, np.abs(comp - eager), 0.0)
            r.update(compiled_vs_eager_max=float(ce.max()), compiled_vs_eager_max_ulps=float((ce / u).max()),
                     per_row_compiled_vs_eager=ce.tolist())
        out[how] = r
        per_row_spread[how] = spread
    out["per_row_band"] = band.tolist()
    out["per_row_k"] = kk.tolist()
    out["per_row_slope"] = slope.tolist()
    out["per_row_tau_ref"] = ref.tolist()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--perms", type=int, default=64)
    ap.add_argument("--delta", type=float, default=1e-3)
    ap.add_argument("--no-compiled", action="store_true")
    a = ap.parse_args()
    th.set_float32_matmul_precision("highest")
    th.set_num_threads(int(os.environ.get("TAU_THREADS", "3")))
    from main.capacity import load_policy
    model, _cfg, _ = load_policy(TEMPLATE, os.path.dirname(TEMPLATE), "cpu")
    template = copy.deepcopy(model.policy)
    del model
    rng = np.random.default_rng(20261020)
    res: Dict[str, Any] = {"perms": a.perms, "delta": a.delta, "torch": th.__version__, "solves": list(SOLVES)}
    for u, (d, ckpts) in EVENTS.items():
        z = np.load(os.path.join(CACHE, d, f"behaviour_violation_u{u}_rows.npz"))
        rows = RN.RD.rows_of(z)
        o, m, acts = rows["obs"]["observation"], rows["obs"]["action_mask"].astype(bool), rows["acts"]
        nrow = o.shape[0]
        for c in ckpts:
            t0 = time.time()
            pol = copy.deepcopy(template)
            pol.load_state_dict(RN.policy_sd(os.path.join(CACHE, d, f"checkpoint_{c}_steps.zip")))
            base, tap = forward64(pol, o, m)
            up, _ = forward64(pol, o, m, a.delta)
            dn, _ = forward64(pol, o, m, -a.delta)
            dlp = (up - dn) / (2 * a.delta)
            sens_taken = np.abs(dlp[np.arange(nrow), acts])
            sens_max = np.array([np.abs(dlp[i][m[i]]).max() for i in range(nrow)])
            per: Dict[str, Any] = {"calls": len(tap.calls), "seconds_forward": round(time.time() - t0, 1)}
            for ci, call in enumerate(tap.calls):
                rep = call_report(call, a.perms if ci == 0 else max(8, a.perms // 8), rng,
                                  compiled=(not a.no_compiled) and ci == 0)
                if ci == 0:
                    for how in SOLVES:
                        spread = np.array(rep[how]["per_row_spread"])
                        pred = spread * sens_taken
                        rep[how]["predicted_gap_violator"] = float(pred[0])
                        rep[how]["predicted_gap_others_max"] = float(pred[1:].max())
                        rep[how]["predicted_gap_rank_of_n"] = [int((pred > pred[0]).sum()) + 1, nrow]
                        rep[how]["spread_violator"] = float(spread[0])
                        rep[how]["err_vs_ref_violator"] = float(rep[how]["per_row_err"][0])
                        if "per_row_compiled_vs_eager" in rep[how]:
                            ce = np.array(rep[how]["per_row_compiled_vs_eager"])
                            rep[how]["compiled_vs_eager_violator"] = float(ce[0])
                            rep[how]["predicted_gap_compiled_violator"] = float(ce[0] * sens_taken[0])
                    rep["violator"] = {"k": rep["per_row_k"][0], "slope": rep["per_row_slope"][0],
                                       "band": rep["per_row_band"][0], "tau": rep["per_row_tau_ref"][0],
                                       "sens_taken": float(sens_taken[0]), "sens_maxlegal": float(sens_max[0])}
                    v = rep["violator"]
                    line = (f"u{u}@{c} call0: k={v['k']} slope {v['slope']:.2e} band {v['band']:.2e} tau {v['tau']:.4f} "
                            f"dlogpi/dtau {v['sens_taken']:.3f}")
                    for how in SOLVES:
                        r = rep[how]
                        line += (f"\n   {how:10s} |tau-ref| v {r['err_vs_ref_violator']:.2e} (all {r['err_vs_ref_max']:.2e}, "
                                 f"{r['err_vs_ref_max_ulps']:.1f} ulp; over orders {r['err_over_orders_max_ulps']:.1f} ulp) | "
                                 f"order spread v {r['spread_violator']:.2e} (all {r['order_spread_max']:.2e} = "
                                 f"{r['order_spread_max_ulps']:.1f} ulp) | compiled-eager v "
                                 f"{r.get('compiled_vs_eager_violator', float('nan')):.2e} (all {r.get('compiled_vs_eager_max', float('nan')):.2e}) | "
                                 f"predicted gap v {r['predicted_gap_violator']:.2e} rank {r['predicted_gap_rank_of_n']} "
                                 f"(others max {r['predicted_gap_others_max']:.2e})")
                    print(line, flush=True)
                else:
                    print(f"u{u}@{c} call{ci}: rows {rep['rows']} band max {rep['band_max']:.2e} | " + " | ".join(
                        f"{how} err {rep[how]['err_vs_ref_max']:.2e} ({rep[how]['err_vs_ref_max_ulps']:.1f} ulp) spread "
                        f"{rep[how]['order_spread_max']:.2e}" for how in SOLVES), flush=True)
                for how in SOLVES:          # slim: per-row arrays only for call 0
                    if ci != 0:
                        for key in ("per_row_err", "per_row_spread", "per_row_compiled_vs_eager"):
                            rep[how].pop(key, None)
                if ci != 0:
                    for key in ("per_row_band", "per_row_k", "per_row_slope", "per_row_tau_ref"):
                        rep.pop(key, None)
                per[f"call{ci}"] = rep
            per["seconds"] = round(time.time() - t0, 1)
            res[f"u{u}@{c}"] = per
            with open(a.out, "w") as fh:
                json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
