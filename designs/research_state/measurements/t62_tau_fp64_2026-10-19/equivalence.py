"""T62 NETWORK EQUIVALENCE: is X26's network, evaluated with the float64 tau solve, the SAME network as with the
pre-T62 fp32 solve (the question an X35 fork of X26 adopting T62 raises)?

Per block of real observation rows, the T2 decision forward (`DecisionModule`: masked log pi [B, 11] + value [B])
is run four ways on the same weights:
* OLD  : fp32 eager, `hypothesis_set.TAU_SOLVE_DTYPE` patched to float32 (bit-identical to the pre-T62 code);
* NEW  : fp32 eager, the shipped float64 solve;
* REF  : the float64 forward (the diagnosis's `f64_forward` + a float64 copy): the exact function;
* OLDc / NEWc (--compiled; blocks A and B): the same fp32 forwards compiled by inductor on CPU -- a second fp32 PROGRAM, the
  CPU analogue of T2's compiled rollout against the eager learner (the gap K9(b) already lives with).
Reported (over LEGAL actions): max / p99 |d log pi|, max |d value|, greedy-argmax agreement, overall and on the
saturated rows (call 0's fp32 tau band ulp(k)/S'(tau) > 1e-4: the k = 1 tau class), plus each program's error
against REF.

Blocks: (A) fixture rows (`utils.rust_env.fixture_battles.play_rows`, seeded) @ checkpoint_280000170;
(B) the three tau-class event dumps' rows (violator + scan 32) @ checkpoint_280000170; (C) the same rows at each
event's first neighbour checkpoint. --toy: X34's parent (the toy) runs no `fixed_size_tau` at all (call count).

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<worktree>/src CUDA_VISIBLE_DEVICES= \
        python3 equivalence.py --out equivalence.json [--compiled] [--toy <run dir>]
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch as th

_HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("t62_repro", os.path.join(_HERE, "repro.py"))
RP = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(RP)  # type: ignore[union-attr]
RN, R = RP.RN, RP.TP.R
CACHE = RP.CACHE
P280_DIR = os.path.join(CACHE, "x35_refpop", "pstar", "rb_x26_s2601")
P280 = os.path.join(P280_DIR, "checkpoints", "checkpoint_280000170_steps.zip")
SAT_BAND = 1e-4


class _Tap:
    """Wraps `hypothesis_set.fixed_size_tau`: counts calls, keeps call 0 (the species call) of each forward."""

    def __init__(self) -> None:
        from agents.model import hypothesis_set as HS
        self.HS, self.orig, self.calls, self.first = HS, HS.fixed_size_tau, 0, None

    def __enter__(self) -> "_Tap":
        def wrapped(scores: th.Tensor, cand: th.Tensor, k: th.Tensor, n_iter: int = 64) -> th.Tensor:
            tau = self.orig(scores, cand, k, n_iter)
            if self.calls == 0:
                self.first = (scores.detach().double().clone(), cand.clone(), k.clone(), tau.detach().double().clone())
            self.calls += 1
            return tau
        self.HS.fixed_size_tau = wrapped
        return self

    def __exit__(self, *exc: Any) -> None:
        self.HS.fixed_size_tau = self.orig


def fwd(policy: Any, obs: np.ndarray, masks: np.ndarray, mode: str, compiled: Optional[Any] = None) -> Tuple[np.ndarray, np.ndarray]:
    from agents.inference.service.decision import DecisionModule
    from agents.model import hypothesis_set as HS
    saved = HS.TAU_SOLVE_DTYPE
    HS.TAU_SOLVE_DTYPE = th.float32 if mode == "old" else th.float64
    try:
        with th.no_grad():
            if mode == "ref":
                p = copy.deepcopy(policy).double()
                with R.f64_forward():
                    lp, v = DecisionModule(p).eval()(th.as_tensor(obs).double(), th.as_tensor(masks).bool())
            else:
                mod = compiled if compiled is not None else DecisionModule(policy).eval()
                lp, v = mod(th.as_tensor(obs).float(), th.as_tensor(masks).bool())
    finally:
        HS.TAU_SOLVE_DTYPE = saved
    return lp.double().numpy(), v.double().numpy()


def compare(a: Tuple[np.ndarray, np.ndarray], b: Tuple[np.ndarray, np.ndarray], masks: np.ndarray,
            sel: np.ndarray) -> Dict[str, Any]:
    if not sel.any():
        return {"rows": 0}
    la, lb = a[0][sel], b[0][sel]
    m = masks[sel]
    with np.errstate(invalid="ignore"):                    # -inf - -inf on illegal actions, masked out
        d = np.abs(np.where(m, la - lb, 0.0))
    per_row = d.max(1)
    ga = np.where(m, la, -np.inf).argmax(1)
    gb = np.where(m, lb, -np.inf).argmax(1)
    # rule 8: a greedy flip only counts where the top-2 margin (in b) exceeds 1e-6
    sb = np.sort(np.where(m, lb, -np.inf), 1)
    margin = sb[:, -1] - sb[:, -2]
    decisive = ~np.isfinite(margin) | (margin > 1e-6)
    return {"rows": int(sel.sum()), "max_dlogp": float(per_row.max()), "p99_dlogp": float(np.quantile(per_row, 0.99)),
            "median_dlogp": float(np.median(per_row)), "max_dvalue": float(np.abs(a[1][sel] - b[1][sel]).max()),
            "argmax_agree": int((ga == gb).sum()), "argmax_agree_decisive": f"{int(((ga == gb) | ~decisive).sum())}/{int(sel.sum())}",
            "near_tied_greedy_rows": int((~decisive).sum())}


def block(name: str, policy: Any, obs: np.ndarray, masks: np.ndarray, compiled: bool) -> Dict[str, Any]:
    from agents.inference.service.decision import DecisionModule
    import torch._dynamo as dynamo
    t0 = time.time()
    old = fwd(policy, obs, masks, "old")
    with _Tap() as tap:
        new = fwd(policy, obs, masks, "new")
    ref = fwd(policy, obs, masks, "ref")
    s, cand, k, tau = tap.first
    pi = th.sigmoid(s + tau.unsqueeze(-1))
    slope = th.where(cand, pi * (1 - pi), th.zeros(())).sum(-1).numpy()
    kk = k.numpy().astype(np.int64)
    live = (kk > 0) & (kk < cand.sum(-1).numpy())
    band = np.where(live, np.finfo(np.float32).eps * np.maximum(kk, 1) / np.maximum(slope, 1e-300), 0.0)
    sat = band > SAT_BAND
    allr = np.ones(len(obs), dtype=bool)
    out: Dict[str, Any] = {"rows": int(len(obs)), "saturated_rows": int(sat.sum()), "k1_rows": int((kk == 1).sum()),
                           "band_max": float(band.max()), "band_p99": float(np.quantile(band, 0.99)),
                           "fixed_size_tau_calls_per_forward": tap.calls}
    pairs = {"new_vs_old": (new, old), "old_vs_ref": (old, ref), "new_vs_ref": (new, ref)}
    if compiled:
        res = {}
        for mode in ("old", "new"):
            from agents.model import hypothesis_set as HS
            dynamo.reset()
            HS.TAU_SOLVE_DTYPE = th.float32 if mode == "old" else th.float64
            try:
                from agents.model.compile_regions import CPU_INDUCTOR_OPTIONS     # torch 2.8.0 CPU tile2d codegen bug
                c = th.compile(DecisionModule(policy).eval(), backend="inductor", dynamic=False, options=dict(CPU_INDUCTOR_OPTIONS))
                tc = time.time()
                res[mode] = fwd(policy, obs, masks, mode, compiled=c)
                out[f"compile_plus_first_call_s_{mode}"] = round(time.time() - tc, 1)
            finally:
                HS.TAU_SOLVE_DTYPE = th.float64
                dynamo.reset()
        pairs.update({"oldcompiled_vs_old (the program floor, pre-T62)": (res["old"], old),
                      "newcompiled_vs_new (the program floor, T62)": (res["new"], new),
                      "newcompiled_vs_oldcompiled": (res["new"], res["old"])})
    for pname, (a, b) in pairs.items():
        out[pname] = {"all": compare(a, b, masks, allr), "saturated": compare(a, b, masks, sat),
                      "unsaturated": compare(a, b, masks, ~sat)}
    out["seconds"] = round(time.time() - t0, 1)
    print(f"[{name}] rows {len(obs)} saturated {int(sat.sum())} (k=1 {int((kk == 1).sum())}) "
          f"band max {band.max():.2e} | new-vs-old all {out['new_vs_old']['all']} | sat {out['new_vs_old']['saturated']}", flush=True)
    for pname in pairs:
        if pname != "new_vs_old":
            print(f"   {pname}: all max {out[pname]['all'].get('max_dlogp')} p99 {out[pname]['all'].get('p99_dlogp')} "
                  f"| sat max {out[pname]['saturated'].get('max_dlogp')}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--battles", type=int, default=24)
    ap.add_argument("--seed", type=int, default=6201)
    ap.add_argument("--compiled", action="store_true")
    ap.add_argument("--toy", default=None)
    a = ap.parse_args()
    th.set_float32_matmul_precision("highest")
    th.set_num_threads(int(os.environ.get("T62_THREADS", "3")))
    from main.capacity import load_policy
    from utils.rust_env.fixture_battles import play_rows
    rows = play_rows(a.battles, seed=a.seed)
    fo = np.concatenate([o for o, _ in rows]).astype(np.float32)
    fm = np.concatenate([m for _, m in rows]).astype(bool)
    res: Dict[str, Any] = {"fixture": {"battles": a.battles, "seed": a.seed, "rows": int(len(fo))}, "torch": th.__version__}
    if a.toy:
        from agents.model import hypothesis_set as HS
        ck = sorted(f for f in os.listdir(os.path.join(a.toy, "checkpoints")) if f.endswith(".zip"))[-1]
        tm, _c, _ = load_policy(os.path.join(a.toy, "checkpoints", ck), a.toy, "cpu")
        with _Tap() as tap:
            fwd(tm.policy, fo[:256], fm[:256], "new")
        hb = getattr(tm.policy.features_extractor, "hypothesis_builder", "absent")
        res["toy"] = {"checkpoint": ck, "fixed_size_tau_calls": tap.calls, "hypothesis_builder": repr(hb)[:60]}
        print(f"[toy] {ck}: fixed_size_tau calls {tap.calls}, hypothesis_builder {repr(hb)[:60]}", flush=True)
        del tm
    m280, _c, _ = load_policy(P280, P280_DIR, "cpu")
    pol = m280.policy
    res["A_fixture@280M"] = block("A fixture@280M", pol, fo, fm, a.compiled)
    ev_o, ev_m = [], []
    for u, (d, ckpts) in RP.EVENTS.items():
        z = RN.RD.rows_of(np.load(os.path.join(CACHE, d, f"behaviour_violation_u{u}_rows.npz")))
        ev_o.append(z["obs"]["observation"].astype(np.float32))
        ev_m.append(z["obs"]["action_mask"].astype(bool))
    res["B_event_rows@280M"] = block("B event rows@280M", pol, np.concatenate(ev_o), np.concatenate(ev_m), a.compiled)
    for (u, (d, ckpts)), o, m in zip(RP.EVENTS.items(), ev_o, ev_m):
        p = copy.deepcopy(pol)
        p.load_state_dict(RN.policy_sd(os.path.join(CACHE, d, f"checkpoint_{ckpts[0]}_steps.zip")))
        res[f"C_u{u}@{ckpts[0]}"] = block(f"C u{u}@{ckpts[0]}", p, o, m, False)     # compiled: blocks A, B only
    with open(a.out, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
