"""FP32_ALT at the NETWORK level: on X26's weights and the tau-class rows, is the network under the fp32 split-residual
solve the same function as under T62's float64 solve, and is its learner-vs-rollout gap at T62's level?

Per block of observation rows, on the same weights:
* NAIVE / ALT / FP64 : the T2 decision forward (`DecisionModule`, fp32 eager) with the hypothesis builder's
  ``tau_solve`` = fp32_naive (the pre-T62 sum) / fp32_alt (this unit) / fp64 (T62);
* REF  : the float64 forward (`f64_forward` + a float64 copy, the split solve in float64): the exact function;
* <arm>c (--compiled): the same fp32 forward compiled by inductor on CPU -- a second fp32 PROGRAM, the CPU analogue
  of T2's compiled rollout against the eager learner;
* <arm>L (blocks B, C): the LEARNER path, `policy.evaluate_actions` in train mode (eager), as the K9(b) probe runs
  it -- its taken-action log pi against the compiled decision forward's is the K9(b) judged statistic's analogue.
Reported over LEGAL actions: max / p99 |d log pi|, max |d value|, greedy agreement, overall and on the SATURATED rows
(call 0's naive fp32 band ulp(k)/S'(tau) > 1e-4: the k = 1 tau class), and each arm's error against REF.

Blocks: (A) fixture rows (`utils.rust_env.fixture_battles.play_rows`, seeded) @ P* (X26 checkpoint_280000170);
(B) the three tau-class event dumps' rows (violator + scan 32) @ P*; (C) the same rows at each event's first
neighbour checkpoint (the weights nearest the event).

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=<worktree>/src CUDA_VISIBLE_DEVICES= \
        python3 equivalence.py --out equivalence.json [--compiled]
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
_spec = importlib.util.spec_from_file_location("fp32alt_taus", os.path.join(_HERE, "taus.py"))
TS = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
_spec.loader.exec_module(TS)  # type: ignore[union-attr]
RN, R = TS.RN, TS.R
CACHE = TS.CACHE
P280_DIR = os.path.join(CACHE, "x35_refpop", "pstar", "rb_x26_s2601")
P280 = os.path.join(P280_DIR, "checkpoints", "checkpoint_280000170_steps.zip")
SAT_BAND = 1e-4
ARMS = ("fp32_naive", "fp32_alt", "fp64")


def fwd(policy: Any, obs: np.ndarray, masks: np.ndarray, solve: str, compiled: Optional[Any] = None) -> Tuple[np.ndarray, np.ndarray]:
    from agents.inference.service.decision import DecisionModule
    with th.no_grad():
        if solve == "ref":
            p = copy.deepcopy(policy).double()
            TS.set_solve(p, "fp32_alt")                      # the split solve in float64
            with R.f64_forward():
                lp, v = DecisionModule(p).eval()(th.as_tensor(obs).double(), th.as_tensor(masks).bool())
        else:
            TS.set_solve(policy, solve)
            mod = compiled if compiled is not None else DecisionModule(policy).eval()
            lp, v = mod(th.as_tensor(obs).float(), th.as_tensor(masks).bool())
    return lp.double().numpy(), v.double().numpy()


def learner(policy: Any, obs: Dict[str, np.ndarray], acts: np.ndarray, masks: np.ndarray, solve: str) -> np.ndarray:
    TS.set_solve(policy, solve)
    return R.learner_logp(policy, obs, acts, masks)   # type: ignore[no-any-return]


def compare(a: Tuple[np.ndarray, np.ndarray], b: Tuple[np.ndarray, np.ndarray], masks: np.ndarray,
            sel: np.ndarray) -> Dict[str, Any]:
    if not sel.any():
        return {"rows": 0}
    la, lb = a[0][sel], b[0][sel]
    m = masks[sel]
    with np.errstate(invalid="ignore"):
        d = np.abs(np.where(m, la - lb, 0.0))
    per_row = d.max(1)
    ga = np.where(m, la, -np.inf).argmax(1)
    gb = np.where(m, lb, -np.inf).argmax(1)
    sb = np.sort(np.where(m, lb, -np.inf), 1)
    margin = sb[:, -1] - sb[:, -2]
    decisive = ~np.isfinite(margin) | (margin > 1e-6)           # rule 8: a flip counts only off a near-tie
    return {"rows": int(sel.sum()), "max_dlogp": float(per_row.max()), "p99_dlogp": float(np.quantile(per_row, 0.99)),
            "median_dlogp": float(np.median(per_row)), "max_dvalue": float(np.abs(a[1][sel] - b[1][sel]).max()),
            "argmax_agree_decisive": f"{int(((ga == gb) | ~decisive).sum())}/{int(sel.sum())}",
            "near_tied_greedy_rows": int((~decisive).sum())}


def taken(lp: np.ndarray, acts: np.ndarray) -> np.ndarray:
    return lp[np.arange(len(acts)), acts]


def block(name: str, policy: Any, obs: np.ndarray, masks: np.ndarray, compiled: bool,
          full_obs: Optional[Dict[str, np.ndarray]] = None, acts: Optional[np.ndarray] = None,
          compiled_arms: Tuple[str, ...] = ("fp32_alt", "fp64"), violators: bool = True) -> Dict[str, Any]:
    from agents.inference.service.decision import DecisionModule
    import torch._dynamo as dynamo
    t0 = time.time()
    eager = {arm: fwd(policy, obs, masks, arm) for arm in ARMS}
    with TS.Tap() as tap:
        fwd(policy, obs, masks, "fp32_alt")
    ref = fwd(policy, obs, masks, "ref")
    c0 = tap.calls[0]
    s = c0["scores"].reshape(-1, c0["scores"].shape[-1])
    cand = c0["cand"].reshape(-1, s.shape[-1])
    k = c0["k"].reshape(-1)
    tau = th.as_tensor(TS.solve(s, cand, k, "fp32_alt"))
    pi = th.sigmoid(s + tau.unsqueeze(-1))
    slope = th.where(cand, pi * (1 - pi), th.zeros((), dtype=th.float64)).sum(-1).numpy()
    kk = k.numpy().astype(np.int64)
    live = (kk > 0) & (kk < cand.sum(-1).numpy())
    band = np.where(live, np.finfo(np.float32).eps * np.maximum(kk, 1) / np.maximum(slope, 1e-300), 0.0)
    sat = band > SAT_BAND
    allr = np.ones(len(obs), dtype=bool)
    out: Dict[str, Any] = {"rows": int(len(obs)), "saturated_rows": int(sat.sum()), "k1_rows": int((kk == 1).sum()),
                           "band_max": float(band.max()), "fixed_size_tau_calls_per_forward": len(tap.calls)}
    pairs: Dict[str, Tuple[Any, Any]] = {"alt_vs_fp64": (eager["fp32_alt"], eager["fp64"]),
                                         "naive_vs_fp64": (eager["fp32_naive"], eager["fp64"])}
    for arm in ARMS:
        pairs[f"{arm}_vs_ref"] = (eager[arm], ref)
    comp: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if compiled:
        from agents.model.compile_regions import CPU_INDUCTOR_OPTIONS     # torch 2.8.0 CPU tile2d codegen bug
        for arm in compiled_arms:
            dynamo.reset()
            TS.set_solve(policy, arm)
            try:
                c = th.compile(DecisionModule(policy).eval(), backend="inductor", dynamic=False,
                               options=dict(CPU_INDUCTOR_OPTIONS))
                tc = time.time()
                comp[arm] = fwd(policy, obs, masks, arm, compiled=c)
                out[f"compile_plus_first_call_s_{arm}"] = round(time.time() - tc, 1)
            finally:
                dynamo.reset()
            pairs[f"{arm}_compiled_vs_eager (the program floor)"] = (comp[arm], eager[arm])
        if "fp32_alt" in comp and "fp64" in comp:
            pairs["alt_compiled_vs_fp64_compiled"] = (comp["fp32_alt"], comp["fp64"])
    for pname, (a, b) in pairs.items():
        out[pname] = {"all": compare(a, b, masks, allr), "saturated": compare(a, b, masks, sat),
                      "unsaturated": compare(a, b, masks, ~sat)}
    if full_obs is not None and acts is not None:
        lrn: Dict[str, Any] = {}
        for arm in ARMS:
            lp_l = learner(policy, full_obs, acts, masks, arm)
            row: Dict[str, Any] = {"vs_ref_taken_max": float(np.abs(lp_l - taken(ref[0], acts)).max()),
                                   "vs_eager_decision_taken_max": float(np.abs(lp_l - taken(eager[arm][0], acts)).max())}
            if arm in comp:
                g = np.abs(lp_l - taken(comp[arm][0], acts))
                row.update(learner_vs_compiled_taken_max=float(g.max()), learner_vs_compiled_taken_violators=[
                    float(x) for x in g[::33]] if (len(g) % 33 == 0 and violators) else None,
                    learner_vs_compiled_taken_saturated_max=float(g[sat].max()) if sat.any() else None,
                    learner_vs_compiled_taken_p99=float(np.quantile(g, 0.99)),
                    learner_vs_compiled_taken_p999=float(np.quantile(g, 0.999)),
                    learner_vs_compiled_taken_over_5e5=int((g > 5e-5).sum()),
                    learner_vs_compiled_taken_over_1e4=int((g > 1e-4).sum()),
                    learner_vs_compiled_taken_per_row=[float(x) for x in g])
            lrn[arm] = row
        out["learner_taken_action (the K9(b) statistic's analogue)"] = lrn
    out["seconds"] = round(time.time() - t0, 1)
    print(f"[{name}] rows {len(obs)} saturated {int(sat.sum())} (k=1 {int((kk == 1).sum())}) band max {band.max():.2e}",
          flush=True)
    for pname in pairs:
        al, sa = out[pname]["all"], out[pname]["saturated"]
        print(f"   {pname}: all max {al.get('max_dlogp', float('nan')):.3e} p99 {al.get('p99_dlogp', float('nan')):.3e} "
              f"greedy {al.get('argmax_agree_decisive')} | sat max {sa.get('max_dlogp', float('nan')):.3e}", flush=True)
    if "learner_taken_action (the K9(b) statistic's analogue)" in out:
        for arm, row in out["learner_taken_action (the K9(b) statistic's analogue)"].items():
            print(f"   learner[{arm}]: " + str({k: v for k, v in row.items() if not k.endswith("per_row")}), flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--battles", type=int, default=24)
    ap.add_argument("--seed", type=int, default=6201)          # T62's fixture: the same 915 rows
    ap.add_argument("--compiled", action="store_true")
    ap.add_argument("--blocks", default="A,B,C")
    ap.add_argument("--events", default="", help="block C only: a comma list of event ids (default: every event)")
    a = ap.parse_args()
    blocks = set(a.blocks.split(","))
    th.set_float32_matmul_precision("highest")
    th.set_num_threads(int(os.environ.get("TAU_THREADS", "3")))
    from main.capacity import load_policy
    from utils.rust_env.fixture_battles import play_rows
    res: Dict[str, Any] = {"torch": th.__version__, "arms": list(ARMS)}
    m280, _c, _ = load_policy(P280, P280_DIR, "cpu")
    pol = m280.policy
    if "A" in blocks:
        rows = play_rows(a.battles, seed=a.seed)
        fo = np.concatenate([o for o, _ in rows]).astype(np.float32)
        fm = np.concatenate([m for _, m in rows]).astype(bool)
        res["fixture"] = {"battles": a.battles, "seed": a.seed, "rows": int(len(fo))}
        # the learner path needs the extra obs keys (zeros: they feed only the losses' labels); the taken action is
        # the float64 reference's greedy one (any legal action is a valid K9(b)-style probe)
        from agents.model.extra_obs_keys import zero_extra_obs
        fobs = {"observation": fo, "action_mask": fm.astype(np.float32)}
        fobs.update({k: v.numpy() for k, v in zero_extra_obs(pol.features_extractor, batch=len(fo)).items()})
        ref_lp = fwd(pol, fo, fm, "ref")[0]
        facts = np.where(fm, ref_lp, -np.inf).argmax(1)
        res["A_fixture@280M"] = block("A fixture@280M", pol, fo, fm, a.compiled, fobs, facts, compiled_arms=ARMS,
                                      violators=False)
        _dump(a.out, res)
    ev: List[Tuple[int, str, Dict[str, np.ndarray], np.ndarray, np.ndarray]] = []
    for u, (d, ckpts) in TS.EVENTS.items():
        z = RN.RD.rows_of(np.load(os.path.join(CACHE, d, f"behaviour_violation_u{u}_rows.npz")))
        ev.append((u, d, z["obs"], z["acts"], z["obs"]["action_mask"].astype(bool)))
    if "B" in blocks:
        fobs = {k: np.concatenate([e[2][k] for e in ev]) for k in ev[0][2]}
        acts = np.concatenate([e[3] for e in ev])
        masks = np.concatenate([e[4] for e in ev])
        res["B_event_rows@280M"] = block("B event rows@280M", pol, fobs["observation"].astype(np.float32), masks,
                                         a.compiled, fobs, acts, compiled_arms=ARMS)
        _dump(a.out, res)
    if "C" in blocks:
        want = {int(x) for x in a.events.split(",") if x}
        for u, d, fobs1, acts1, masks1 in ev:
            if want and u not in want:
                continue
            ck = TS.EVENTS[u][1][0]
            p = copy.deepcopy(pol)
            p.load_state_dict(RN.policy_sd(os.path.join(CACHE, d, f"checkpoint_{ck}_steps.zip")))
            res[f"C_u{u}@{ck}"] = block(f"C u{u}@{ck}", p, fobs1["observation"].astype(np.float32), masks1,
                                        a.compiled, fobs1, acts1, compiled_arms=ARMS)
            _dump(a.out, res)


def _dump(path: str, res: Dict[str, Any]) -> None:
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
