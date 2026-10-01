"""K9(b) fp32 TAIL SWEEP (2026-10-01). Sizing arm A2 died on ONE row of 1,024 with |d log pi| 0.0389 (p99
2.6e-6, next-worst 4.5e-6). Two root causes are open:
  (a) NUMERICS: fp32 noise between the two forwards meets a DISCONTINUITY (a discrete op near a tie),
      or a near-zero probability amplified;
  (b) a FAULT: the stored log-prob did not come from this obs under these weights (a race in T2's
      multi-lane replay, a misaligned row, a mask mismatch, stale weights on one slot).

The sweep collects complete-game rollouts on the PRODUCTION path (rust core, T2 graph backend with the
production lane/bucket resolution, a pool of real snapshots so several lanes replay) under ONE fixed
policy (the checkpoint; nothing trains, so every row is current), and runs EVERY row through the learner's
probe forward (`consistency.scan_current`, eager fp32 'highest', train mode, batch_size chunks). For each
row over the fp32 bar (1e-4) it records, on the GPU, while the T2 slot still holds the weights:
  * T2 RECOMPUTE: the row alone (smallest bucket) and inside a 48-row request of real companions —
    does T2 reproduce its own stored value? (yes => deterministic => not a race; no => (b));
  * EAGER fp32 at batch 1;
  * WEIGHT JITTER: 32 eager forwards with every float parameter scaled by (1 + 3e-7 N(0,1)) — a few ulps,
    emulating kernel-order noise — the set of log pi(a) values (two modes a jump apart => a discontinuity);
  * a DISCRETE-OP TRACE (TorchFunctionMode over topk / sort / argmax / max-with-indices / comparisons of
    floats / float->int casts) of the base forward and every jittered one: the op SITES whose discrete
    output changes.
Per fill a summary line; per violator a JSON row + its obs (`violators.npz`). Resumable: fills are
appended; `--fills` is the total wanted. Runs under scripts/ops/gpu_lock.sh with a timeout inside.

    python sweep.py --ckpt <run>/checkpoints/<ckpt>.zip --pool <run>/snapshots --out <dir> --fills 25
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple

import numpy as np
import torch as th
from torch.overrides import TorchFunctionMode

CMP = {"__ge__", "__gt__", "__le__", "__lt__", "__eq__", "__ne__", "ge", "gt", "le", "lt", "eq", "ne",
       "greater", "greater_equal", "less", "less_equal", "isclose"}
CAST = {"long", "int", "round", "floor", "ceil", "to", "type", "trunc"}


def _site() -> str:
    f = sys._getframe(2)
    while f is not None:
        fn = f.f_code.co_filename
        if "/src/agents/" in fn or "/src/main/" in fn:
            return f"{fn.split('/src/', 1)[1]}:{f.f_lineno}"
        f = f.f_back
    return "?"


class DiscreteTrace(TorchFunctionMode):
    """Every DISCRETE output of a forward: indices of topk / sort / argmax / argmin / argsort / max-min
    with a dim, bool results of comparisons with a floating operand, integer results of float casts."""

    def __init__(self) -> None:
        super().__init__()
        self.log: List[Tuple[str, str, np.ndarray]] = []

    def __torch_function__(self, func, types, args=(), kwargs=None):
        out = func(*args, **(kwargs or {}))
        name = getattr(func, "__name__", "")
        rec = None
        if name in ("topk", "sort") and isinstance(out, tuple):
            rec = out[1]
        elif name in ("argmax", "argmin", "argsort"):
            rec = out
        elif name in ("max", "min") and isinstance(out, tuple) and len(out) == 2:
            rec = out[1]
        elif name in CMP and isinstance(out, th.Tensor) and out.dtype == th.bool:
            if any(isinstance(a, th.Tensor) and a.is_floating_point() for a in args):
                rec = out
        elif (name in CAST and isinstance(out, th.Tensor) and not out.is_floating_point()
              and out.dtype != th.bool and args and isinstance(args[0], th.Tensor) and args[0].is_floating_point()):
            rec = out
        if isinstance(rec, th.Tensor):
            self.log.append((name, _site(), rec.detach().cpu().numpy().copy()))
        return out


def trace_diff(a: List[Tuple[str, str, np.ndarray]], b: List[Tuple[str, str, np.ndarray]]) -> Dict[str, Any]:
    if len(a) != len(b):
        return {"control_flow_differs": True, "len": [len(a), len(b)]}
    sites = []
    for (n1, s1, x), (n2, s2, y) in zip(a, b):
        if s1 != s2 or n1 != n2:
            return {"control_flow_differs": True, "at": [n1, s1, n2, s2]}
        if x.shape != y.shape or not np.array_equal(x, y):
            sites.append(f"{n1}@{s1}")
    return {"control_flow_differs": False, "changed": sorted(set(sites))}


def build(a: argparse.Namespace) -> Tuple[Any, Any, Any]:
    from main.rust_core_m5.hooks import ProductionMix
    from main.rust_core_m5.production import CollectorRustArm, load_trainee

    mix = ProductionMix(pool=a.pool, pool_size=a.pool_size, self_play_fraction=a.self_play_fraction)
    args = mix.args
    arm = CollectorRustArm(a.n_envs, int(args.rust_env_threads), args.rust_env_front, args.rust_env_profile,
                           SimpleNamespace(ckpt=a.ckpt), mix, SimpleNamespace(target=a.target), a.seed,
                           device=a.device, backend="graph" if a.device == "cuda" else "eager", buckets=(),
                           name="k9_tail")
    arm.build()
    learner = load_trainee(a.ckpt, a.device)
    learner.rollout_buffer = arm._buf
    learner.behaviour_check = "warn"
    learner._rust_collector = arm.col
    return arm, learner, mix


def t2_logp(col: Any, slot: int, obs: np.ndarray, mask: np.ndarray) -> np.ndarray:
    from agents.inference.service.spec import Priority

    tk = col.svc.submit(int(slot), obs, mask, Priority.ROLLOUT)
    col.svc.flush()
    lp, _v, _g = tk.host()
    return np.asarray(lp, dtype=np.float64)


def eager(policy: Any, obs: Dict[str, np.ndarray], act: np.ndarray, mask: np.ndarray, dev: Any,
          trace: bool = False) -> Tuple[np.ndarray, Any]:
    from stable_baselines3.common.utils import obs_as_tensor

    from agents.training.rust_rollout.consistency import _stashed_logp

    mode = DiscreteTrace() if trace else None
    with th.no_grad():
        if mode is not None:
            with mode:
                _v, lp, _e = policy.evaluate_actions(obs_as_tensor(obs, dev), th.as_tensor(act).long().to(dev),
                                                     action_masks=th.as_tensor(mask).to(dev))
        else:
            _v, lp, _e = policy.evaluate_actions(obs_as_tensor(obs, dev), th.as_tensor(act).long().to(dev),
                                                 action_masks=th.as_tensor(mask).to(dev))
    return _stashed_logp(policy), (None if mode is None else mode.log)


def analyse(arm: Any, learner: Any, row: Dict[str, Any], obs_row: Dict[str, np.ndarray], fill: int,
            rng: np.random.Generator, jitter: int, eps: float) -> Dict[str, Any]:
    col = arm.col
    buf = arm._buf
    dev = learner.device
    pol = learner.policy
    t, e, a = int(row["buffer_t"]), int(row["buffer_e"]), int(row["action"])
    mask = buf.action_masks[t, e][None]
    o1 = {k: v[None] for k, v in obs_row.items()}
    res: Dict[str, Any] = {"fill": fill, **row}
    slot = int(row["slot"]) if row.get("slot", -1) >= 0 else int(col.current_slot)
    mo, mm = buf.observations["observation"][t, e][None], buf.observations["action_mask"][t, e][None]
    res["t2_alone"] = float(t2_logp(col, slot, mo, mm)[0, a])
    n = buf.log_probs.size
    comp = rng.choice(n, 47, replace=False)
    ct, ce = comp // buf.n_envs, comp % buf.n_envs
    bo = np.concatenate([mo, buf.observations["observation"][ct, ce]])
    bm = np.concatenate([mm, buf.observations["action_mask"][ct, ce]])
    res["t2_in_48"] = float(t2_logp(col, slot, bo, bm)[0, a])
    pol.set_training_mode(True)
    try:
        base, tr0 = eager(pol, o1, np.array([a]), mask, dev, trace=True)
        res["eager_b1"] = float(base[0, a])
        res["eager_b1_dist"] = [float(x) if np.isfinite(x) and x > -1e7 else None for x in base[0]]
        res["n_discrete_ops"] = len(tr0)
        saved = {k: p.detach().clone() for k, p in pol.named_parameters() if p.is_floating_point()}
        vals, changed, cf = [], {}, 0
        try:
            for j in range(jitter):
                with th.no_grad():
                    for k, p in pol.named_parameters():
                        if k in saved:
                            p.copy_(saved[k] * (1.0 + eps * th.randn_like(p)))
                lp, tr = eager(pol, o1, np.array([a]), mask, dev, trace=True)
                vals.append(float(lp[0, a]))
                d = trace_diff(tr0, tr)
                if d["control_flow_differs"]:
                    cf += 1
                for s in d.get("changed", []):
                    changed[s] = changed.get(s, 0) + 1
        finally:
            with th.no_grad():
                for k, p in pol.named_parameters():
                    if k in saved:
                        p.copy_(saved[k])
        v = np.asarray(vals)
        res["jitter"] = {"eps": eps, "n": int(v.size), "min": float(v.min()), "max": float(v.max()),
                         "spread": float(v.max() - v.min()), "values": [float(x) for x in v],
                         "control_flow_differs": cf, "changed_sites": changed}
    finally:
        pol.set_training_mode(False)
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fills", type=int, default=25)
    ap.add_argument("--n-envs", type=int, default=48)
    ap.add_argument("--target", type=int, default=98304)
    ap.add_argument("--pool-size", type=int, default=2)
    ap.add_argument("--self-play-fraction", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--jitter", type=int, default=32)
    ap.add_argument("--eps", type=float, default=3e-7)
    ap.add_argument("--max-violators-per-fill", type=int, default=8)
    ap.add_argument("--budget-s", type=float, default=900.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force-analyse", type=int, default=0,
                    help="a SMOKE switch: analyse the worst K rows of each fill even under the bar")
    a = ap.parse_args()
    t_start = time.time()
    th.set_float32_matmul_precision("highest")
    from agents.model.compile_cache import ensure_hermetic_cache
    from agents.training.rust_rollout import consistency as K
    from agents.training.rust_rollout import store as S

    ensure_hermetic_cache("k9 tail sweep")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    done = sum(1 for _ in open(out / "fills.jsonl")) if (out / "fills.jsonl").exists() else 0
    if done >= a.fills:
        print(f"[k9-tail] {done} fills already banked — nothing to do")
        return
    arm, learner, _mix = build(a)
    col = arm.col
    rng = np.random.default_rng([a.seed, done])
    print(f"[k9-tail] built in {time.time() - t_start:.0f}s; T2 {K._route(learner)}", flush=True)
    obs_bank: Dict[str, List[np.ndarray]] = {}
    for fill in range(done, a.fills):
        if time.time() - t_start > a.budget_s:
            print(f"[k9-tail] budget {a.budget_s:.0f}s reached after fill {fill - 1}", flush=True)
            break
        t0 = time.time()
        while not col.ready():
            col.host_step()
        t1 = time.time()
        rep, versions = S.fill_complete(arm._buf, col.log, int(arm._target), current_version=col.version)
        learner._rust_fill = rep
        learner._rust_row_versions = versions
        learner._rust_row_provenance = rep.provenance
        learner._rust_version = col.version
        learner.num_timesteps += int(arm._target)
        ages = int(col.version) - np.asarray(versions, dtype=np.int64)
        sc = K.scan_current(learner, ages)
        t2 = time.time()
        viol = [r for r in sc["worst"] if not r["abs_dlogp"] < sc["bar"]][:a.max_violators_per_fill]
        viol = viol or sc["worst"][:a.force_analyse]
        rows = []
        for j, r in enumerate(viol):
            obs_row = {k: v[j] for k, v in sc["_obs"].items()}
            res = analyse(arm, learner, r, obs_row, fill, rng, a.jitter, a.eps)
            rows.append(res)
            for k, v in obs_row.items():
                obs_bank.setdefault(k, []).append(v)
            with open(out / "violators.jsonl", "a") as f:
                f.write(json.dumps(res) + "\n")
        if rows:
            prev = dict(np.load(out / "violators.npz")) if (out / "violators.npz").exists() else {}
            np.savez_compressed(out / "violators.npz", **{
                k: np.concatenate([prev[k], np.stack(v)]) if k in prev else np.stack(v) for k, v in obs_bank.items()})
            obs_bank = {}
        summ = {"fill": fill, "rows": sc["rows"], "over_bar": sc["over_bar"], "max": sc["max"], "p99": sc["p99"],
                "current_share": rep.current_share, "collect_s": round(t1 - t0, 1), "scan_s": round(t2 - t1, 1),
                "analyse_s": round(time.time() - t2, 1), "top3": [round(r["abs_dlogp"], 8) for r in sc["worst"][:3]]}
        with open(out / "fills.jsonl", "a") as f:
            f.write(json.dumps(summ) + "\n")
        print(f"[k9-tail] fill {fill}: {summ}", flush=True)
    arm.close()


if __name__ == "__main__":
    main()
