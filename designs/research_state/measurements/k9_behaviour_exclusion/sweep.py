"""K9(b) DETERMINISTIC EXCLUSION sweep (`gen3_behaviour_tie_exclusion_v1`, 2026-10-01).

The tail sweep's setup (`../k9_behaviour_tail/sweep.py`: A2's 4.0M checkpoint and its 2 snapshots as the
pool, the production rust path — T2 graph backend, buckets 8 / 48, N = 48, fills of 98,304 rows, nothing
training so every row is current), and per fill:

* EVERY row through the probe forward (`consistency.scan_current`: eager, train mode, ``batch_size``
  chunks) under `tie_margins.TieMargins` — its |d log pi| against T2's stored log-prob AND its tie margin
  (the smallest relative margin over every declared MARGIN site, `agents/model/selection_sites.py`).
  Per fill: the per-row arrays (``fill_<n>.npz``) and a summary line (``fills.jsonl``): the excluded share
  and the max |d| over the JUDGED rows at each epsilon of a grid, and max / p99 |d| per margin decade.
* the ROUNDING SCALE of the margins (``--variant-rows`` random rows): the per-site per-row margin of the
  learner's forward vs the same rows through forwards that differ the ways T2's does — eval mode without
  grad at T2's buckets (48, 8), and a few-ulp weight jitter (``--jitter``; the tail sweep showed such a
  jitter reproduces T2's stored values) — max |m_learner - m_variant| per site.
* ``--faults``: planted faults judged by the REAL probe (`consistency.behaviour_probe`, ``fatal``) at
  ``--eps``: one-step-stale weights, a single wrong-action row NOT at a tie, an obs/mask misalignment;
  each must FATAL on its FIRST update — and the unmodified buffer must pass.
* ``--cost``: the probe forward on ``batch_size`` rows with and without the recorder (CUDA-synchronised).

Resumable (fills are appended; ``--fills`` is the total wanted); ``--budget-s`` bounds the wall inside a
``scripts/ops/gpu_lock.sh`` hold. ``--precision high`` reads TF32 (T2 and the learner both).

    python sweep.py --ckpt <ckpt.zip> --pool <snapshots> --out <dir> --fills 36 [--precision high]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch as th

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "k9_behaviour_tail"))
from sweep import build  # noqa: E402

EPS_GRID = (0.0, 1e-8, 3e-8, 1e-7, 3e-7, 1e-6, 3e-6, 1e-5, 3e-5, 1e-4, 1e-3)
DECADES = (0.0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, np.inf)


def fill_summary(absd: np.ndarray, margin: np.ndarray, bar: float) -> Dict[str, Any]:
    out: Dict[str, Any] = {"rows": int(absd.size), "over_bar": int((absd >= bar).sum()),
                           "max": float(absd.max()), "p99": float(np.quantile(absd, 0.99))}
    out["by_eps"] = {}
    for eps in EPS_GRID:
        ex = ~(margin >= eps) if eps > 0 else ~(margin > 0)
        j = absd[~ex]
        out["by_eps"][f"{eps:g}"] = {"excluded": int(ex.sum()), "max_judged": float(j.max()) if j.size else 0.0,
                                     "over_bar_judged": int((j >= bar).sum())}
    out["by_decade"] = []
    for lo, hi in zip(DECADES[:-1], DECADES[1:]):
        m = (margin >= lo) & (margin < hi)
        if lo == 0:
            m = (margin > 0) & (margin < hi)
        a = absd[m]
        out["by_decade"].append({"lo": lo, "hi": float(hi), "n": int(m.sum()), "max": float(a.max()) if a.size else None,
                                 "p99": float(np.quantile(a, 0.99)) if a.size else None})
    z = absd[margin == 0]
    out["exact_tie"] = {"n": int(z.size), "max": float(z.max()) if z.size else None}
    out["max_margin_over_bar"] = float(margin[absd >= bar].max()) if (absd >= bar).any() else None
    return out


def _site_margins(policy: Any, obs: Dict[str, np.ndarray], acts: np.ndarray, masks: np.ndarray, dev: Any,
                  chunk: int, train: bool, grad: bool) -> Dict[str, np.ndarray]:
    """site -> per-row margin (min over that site's calls in one forward; inf where it did not run)."""
    from stable_baselines3.common.utils import obs_as_tensor

    from agents.training.rust_rollout.tie_margins import TieMargins

    n = acts.shape[0]
    out: Dict[str, np.ndarray] = {}
    was = policy.training
    policy.set_training_mode(train)
    try:
        for s in range(0, n, chunk):
            sl = slice(s, min(n, s + chunk))
            rec = TieMargins(sl.stop - sl.start, keep_calls=True)
            ctx = th.enable_grad() if grad else th.no_grad()
            with ctx, rec:
                policy.evaluate_actions(obs_as_tensor({k: v[sl] for k, v in obs.items()}, dev),
                                        th.as_tensor(acts[sl]).long().to(dev), action_masks=th.as_tensor(masks[sl]).to(dev))
            rec.check()
            for site, per in rec.calls or []:
                arr = out.setdefault(site, np.full(n, np.inf))
                arr[sl] = np.minimum(arr[sl], per)
    finally:
        policy.set_training_mode(was)
    return out


def variants(learner: Any, buf: Any, rng: np.random.Generator, n: int, jitter: float) -> Dict[str, Any]:
    flat = rng.choice(buf.log_probs.size, min(n, buf.log_probs.size), replace=False)
    t, e = flat // buf.n_envs, flat % buf.n_envs
    obs = {k: v[t, e] for k, v in buf.observations.items()}
    acts, masks = buf.actions[t, e].reshape(-1), buf.action_masks[t, e]
    pol, dev = learner.policy, learner.device
    B = int(learner.batch_size)
    base = _site_margins(pol, obs, acts, masks, dev, B, True, True)
    out: Dict[str, Any] = {"rows": int(flat.size)}
    runs = {"eval_b48": (48, False, False, 0.0), "eval_b8": (8, False, False, 0.0), "jitter": (B, True, True, jitter)}
    for name, (chunk, train, grad, jit) in runs.items():
        saved = None
        if jit:
            saved = {k: p.detach().clone() for k, p in pol.named_parameters()}
            g = th.Generator(device=dev).manual_seed(int(rng.integers(1 << 31)))
            with th.no_grad():
                for p in pol.parameters():
                    if p.is_floating_point():
                        p.mul_(1 + jit * th.randn(p.shape, generator=g, device=dev))
        try:
            v = _site_margins(pol, obs, acts, masks, dev, chunk, train, grad)
        finally:
            if saved is not None:
                with th.no_grad():
                    for k, p in pol.named_parameters():
                        p.copy_(saved[k])
        per_site = {}
        overall = 0.0
        for site in sorted(set(base) | set(v)):
            a, b = base.get(site), v.get(site)
            if a is None or b is None:
                per_site[site] = {"missing_in": "variant" if b is None else "learner"}
                continue
            ok = np.isfinite(a) & np.isfinite(b)
            d = np.abs(a[ok] - b[ok])
            if d.size:
                i = int(np.argmax(d))
                per_site[site] = {"n": int(d.size), "max_diff": float(d[i]), "at_margin": float(a[ok][i]),
                                  "p99_diff": float(np.quantile(d, 0.99)),
                                  "rows_where_one_side_is_masked": int((np.isfinite(a) != np.isfinite(b)).sum())}
                overall = max(overall, float(d[i]))
        out[name] = {"max_diff": overall, "per_site": per_site}
    return out


def faults(learner: Any, eps: float) -> Dict[str, Any]:
    """Planted faults through the REAL probe at ``eps`` (the production gate otherwise): each must FATAL on
    its first update; the unmodified buffer must pass. Restores the buffer after each."""
    from agents.training.rust_rollout import consistency as K

    K.BEHAVIOUR_GATES["highest"] = (K.GateCondition("max", 1e-4, 1, eps),
                                    K.GateCondition("excluded_frac", K.FP32_EXCLUDED_CEILING, 1, eps))
    learner.behaviour_check = "fatal"
    learner.behaviour_scan_all = False
    learner.behaviour_dump_dir = None          # never write a dump beside the (read-only) checkpoint
    buf = learner.rollout_buffer
    out: Dict[str, Any] = {"eps": eps}

    def probe(name: str) -> None:
        learner._behaviour_streaks = {}
        try:
            m = K.behaviour_probe(learner)
            out[name] = {"fatal": False, "excluded_frac": m.get("behaviour/excluded_frac"),
                         "max_judged": m.get("behaviour/max_abs_dlogp_judged")}
        except K.BehaviourMismatch as exc:
            out[name] = {"fatal": True, "message": str(exc)[:400]}

    probe("healthy")
    # one-step-stale: every parameter moved by an Adam first step at the recipe lr (|update| = lr per weight)
    saved = {k: p.detach().clone() for k, p in learner.policy.named_parameters()}
    g = th.Generator(device=learner.device).manual_seed(5)
    with th.no_grad():
        for p in learner.policy.parameters():
            if p.is_floating_point():
                p.add_(2.8e-5 * th.sign(th.randn(p.shape, generator=g, device=learner.device)))
    probe("stale_one_adam_step")
    with th.no_grad():
        for k, p in learner.policy.named_parameters():
            p.copy_(saved[k])
    # a single wrong-action row NOT at a tie, among the probe's own rows
    rng = np.random.default_rng([int(getattr(learner, "seed", 0) or 0), int(learner.num_timesteps)])
    flat = K.choose_rows(np.zeros(buf.log_probs.shape, np.int64), int(learner.batch_size), rng)
    from agents.training.rust_rollout.tie_margins import selection_gaps
    t, e = flat // buf.n_envs, flat % buf.n_envs
    gm, _s = selection_gaps(learner.policy, {k: v[t, e] for k, v in buf.observations.items()}, buf.actions[t, e],
                            buf.action_masks[t, e], learner.device)
    cand = [i for i in np.argsort(-gm) if (buf.action_masks[t[i], e[i]] > 0.5).sum() >= 2]
    i = int(cand[0])
    a0 = int(buf.actions[t[i], e[i]].reshape(-1)[0])
    legal = np.flatnonzero(buf.action_masks[t[i], e[i]] > 0.5)
    buf.actions[t[i], e[i]] = int(legal[legal != a0][0])
    probe("one_wrong_action_row")
    out["one_wrong_action_row"]["row_margin"] = float(gm[i])
    buf.actions[t[i], e[i]] = a0
    # an obs/mask misalignment: one env column's observations shifted by one decision
    col = 0
    keep = {k: v[:, col].copy() for k, v in buf.observations.items()}
    keep_m = buf.action_masks[:, col].copy()
    for k, v in buf.observations.items():
        v[:-1, col] = keep[k][1:]
    buf.action_masks[:-1, col] = keep_m[1:]
    probe("obs_mask_misaligned_one_env")
    for k, v in buf.observations.items():
        v[:, col] = keep[k]
    buf.action_masks[:, col] = keep_m
    probe("healthy_after")
    return out


def cost(learner: Any, reps: int = 10) -> Dict[str, Any]:
    from stable_baselines3.common.utils import obs_as_tensor

    from agents.training.rust_rollout.tie_margins import TieMargins

    buf = learner.rollout_buffer
    B = int(learner.batch_size)
    f = np.arange(B)
    t, e = f // buf.n_envs, f % buf.n_envs
    obs = obs_as_tensor({k: v[t, e] for k, v in buf.observations.items()}, learner.device)
    acts = th.as_tensor(buf.actions[t, e].reshape(-1)).long().to(learner.device)
    masks = th.as_tensor(buf.action_masks[t, e]).to(learner.device)
    pol = learner.policy
    pol.set_training_mode(True)

    sync = th.cuda.synchronize if str(learner.device).startswith("cuda") else (lambda: None)

    def once(rec: bool) -> float:
        sync()
        t0 = time.perf_counter()
        m = TieMargins(B) if rec else None
        with th.enable_grad(), (m if m is not None else contextlib.nullcontext()):
            _v, lp, _e = pol.evaluate_actions(obs, acts, action_masks=masks)
        lp.detach().cpu()
        sync()
        return time.perf_counter() - t0
    for _ in range(2):
        once(False), once(True)
    plain, rec = [], []
    for _ in range(reps):
        plain.append(once(False))
        rec.append(once(True))
    return {"rows": B, "reps": reps, "plain_ms_median": 1e3 * float(np.median(plain)),
            "recorder_ms_median": 1e3 * float(np.median(rec)),
            "added_ms_median": 1e3 * float(np.median(np.asarray(rec) - np.asarray(plain)))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pool", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fills", type=int, default=36)
    ap.add_argument("--n-envs", type=int, default=48)
    ap.add_argument("--target", type=int, default=98304)
    ap.add_argument("--pool-size", type=int, default=2)
    ap.add_argument("--self-play-fraction", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--precision", default="highest", choices=("highest", "high"))
    ap.add_argument("--variant-rows", type=int, default=2048)
    ap.add_argument("--jitter", type=float, default=3e-7)
    ap.add_argument("--faults", action="store_true")
    ap.add_argument("--eps", type=float, default=None)
    ap.add_argument("--cost", action="store_true")
    ap.add_argument("--budget-s", type=float, default=900.0)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    t_start = time.time()
    th.set_float32_matmul_precision(a.precision)
    from agents.model.compile_cache import ensure_hermetic_cache
    from agents.training.rust_rollout import consistency as K
    from agents.training.rust_rollout import store as S

    ensure_hermetic_cache("k9 exclusion sweep")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    done = sum(1 for _ in open(out / "fills.jsonl")) if (out / "fills.jsonl").exists() else 0
    want_extra = a.faults or a.cost
    if done >= a.fills and not want_extra:
        print(f"[k9-excl] {done} fills already banked — nothing to do")
        return
    if a.precision == "high":         # TF32 has no exclusion in the gate: give the scan one to record margins
        K.BEHAVIOUR_GATES["high"] = (K.GateCondition("p99", 3.6e-3, 1, 1e-30),
                                     K.GateCondition("max", 0.071, K.TF32_MAX_PERSISTENCE, 1e-30))
    ns = argparse.Namespace(**{**vars(a), "target": a.target})
    arm, learner, _mix = build(ns)
    learner.behaviour_dump_dir = None
    col = arm.col
    rng = np.random.default_rng([a.seed, done])
    print(f"[k9-excl] built in {time.time() - t_start:.0f}s; precision {th.get_float32_matmul_precision()}; "
          f"T2 {K._route(learner)}", flush=True)
    fill = done
    while True:
        if time.time() - t_start > a.budget_s:
            print(f"[k9-excl] budget {a.budget_s:.0f}s reached", flush=True)
            break
        if fill >= a.fills and not want_extra:
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
        if fill >= a.fills:            # an extra fill for the faults / cost reads only
            extra: Dict[str, Any] = {"fill": fill}
            if a.cost:
                extra["cost"] = cost(learner)
            if a.faults:
                extra["faults"] = faults(learner, float(a.eps if a.eps is not None else K.FP32_TIE_EPS))
            with open(out / "extras.jsonl", "a") as f:
                f.write(json.dumps(extra) + "\n")
            print(f"[k9-excl] extras: {json.dumps(extra)[:2000]}", flush=True)
            break
        ages = int(col.version) - np.asarray(versions, dtype=np.int64)
        sc = K.scan_current(learner, ages)
        t2 = time.time()
        absd, margin = sc["_absd"], sc["_margin"]
        sites = sc["_sites"]
        names = sorted(set(sites))
        np.savez_compressed(out / f"fill_{fill:03d}.npz", absd=absd.astype(np.float64), margin=margin,
                            site=np.asarray([names.index(s) for s in sites], np.int16), site_names=np.asarray(names))
        summ = {"fill": fill, "precision": a.precision, **fill_summary(absd, margin, 1e-4),
                "collect_s": round(t1 - t0, 1), "scan_s": round(t2 - t1, 1)}
        if a.variant_rows:
            summ["variants"] = variants(learner, arm._buf, rng, a.variant_rows, a.jitter)
        summ["variants_s"] = round(time.time() - t2, 1)
        with open(out / "fills.jsonl", "a") as f:
            f.write(json.dumps(summ) + "\n")
        v = summ.get("variants", {})
        print(f"[k9-excl] fill {fill}: rows {summ['rows']} over_bar {summ['over_bar']} max {summ['max']:.3g} | "
              f"eps 1e-6: excl {summ['by_eps']['1e-06']['excluded']} max_judged {summ['by_eps']['1e-06']['max_judged']:.3g} | "
              f"var max dm: " + ", ".join(f"{k} {v[k]['max_diff']:.3g}" for k in v if k != "rows")
              + f" | {summ['collect_s']}+{summ['scan_s']}+{summ['variants_s']}s", flush=True)
        fill += 1
    arm.close()


if __name__ == "__main__":
    main()
