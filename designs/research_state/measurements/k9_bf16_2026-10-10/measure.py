"""K9(b) under the bf16 trunk region — ONE (state, behaviour) cell (T25 lever A; `README.md`).

Plays a seeded complete-game Rust-collector rollout at ``--arch endstate`` (``--n-envs`` x ``--n-steps`` rows,
T2 on ``--device`` with ``--t2-backend``, p2 a seeded random policy) whose T2 serves the trunk at
``--behaviour`` precision (``bf16``: design (a), T2 runs the bf16 region; ``fp32``: design (b), today's T2), so
the stored log mu is that precision's. Then, on EVERY row:

* the LEARNER's probe forward (`consistency.probe_forward`: train mode, grad on, nothing saved, chunks of the
  micro-batch ``--chunk``) under `tie_margins.TieMargins` at bf16 AND at fp32 trunk: log pi, the full masked
  log-probs, the row's tie margin and site, and its per-site margins; the flip-judge's ``flippable`` at every
  epsilon of ``EPS_GRID``;
* T2-LIKE bf16 variants (the k9_behaviour_exclusion method: forwards that differ the ways T2's does) with their
  per-site margins: eval / no grad at chunks 2048 (mode + grad alone), 48, 16 (this harness's T2 bucket) and 8
  (on the first ``--b8-rows`` rows), train mode / no grad at 2048, a few-ulp weight jitter (``JITTER``, train /
  grad / 2048), and (``--compiled-rounds``) each trunk round under ``torch.compile`` (inductor) at eval / no grad /
  48 — a proxy for T2's compiled fusion. The same eval-48 and jitter variants at fp32 calibrate R_fp32 on
  this device against the banked CUDA 1.44e-5;
* the TEETH under the bf16 trunk: the one-optimizer-step-stale fault (``lr * sign(N(0,1))`` on every weight,
  seed 5) at each lr of ``STALE_LRS``, through the probe forward with its OWN margins;
* a float64 reference (`learner_golden_fp64.Fp64Mode` on a float64 copy) on the first ``--fp64-rows`` rows.

Writes ``<out>.json`` (the scalars: R per variant and site, timings, the cell) and ``<out>.npz`` (per-row
arrays `derive.py` reads). CPU by default; ``--device cuda`` is the GPU window's re-run (under
``scripts/ops/gpu_lock.sh`` + ``scripts/ops/mem_cap.sh``; never run it without the lease).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict

import numpy as np
import torch as th

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C  # noqa: E402

EPS_GRID = (2e-4, 5e-4, 1e-3, 2e-3, 5e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1)


def site_diff(base: Dict[str, np.ndarray], var: Dict[str, np.ndarray]) -> Dict[str, Any]:
    """Per declared site: max / p99.9 of |m_learner - m_variant| over rows where both are finite; and the
    per-row max over sites (for a per-row read)."""
    per: Dict[str, Any] = {}
    overall = 0.0
    n = next(iter(base.values())).size if base else 0
    row_max = np.zeros(n)
    for s in sorted(set(base) | set(var)):
        a, b = base.get(s), var.get(s)
        if a is None or b is None:
            per[s] = {"missing_in": "variant" if b is None else "learner"}
            continue
        ok = np.isfinite(a) & np.isfinite(b)
        d = np.where(ok, np.abs(np.where(ok, a, 0) - np.where(ok, b, 0)), 0.0)
        row_max = np.maximum(row_max, d)
        if ok.any():
            i = int(np.argmax(d))
            per[s] = {"n": int(ok.sum()), "max_diff": float(d[i]), "at_margin": float(a[i]),
                      "p999_diff": float(np.quantile(d[ok], 0.999)),
                      "one_side_masked": int((np.isfinite(a) != np.isfinite(b)).sum())}
            overall = max(overall, float(d[i]))
    return {"max_diff": overall, "per_site": per, "_row_max": row_max}


def install_compiled_rounds(policy: Any) -> list:
    """Each trunk round's forward replaced (instance-level) by ``torch.compile`` of its own class's forward
    (the bf16 class's autocast region included); returns the rounds (for `uninstall`)."""
    from agents.model.trunk_precision import _round_types
    table = _round_types()
    kinds = set(table) | set(table.values())
    rounds = [m for m in policy.modules() if type(m) in kinds]
    for m in rounds:
        compiled = th.compile(type(m).forward.__get__(m), dynamic=False)

        def fwd(x: Any, bias: Any = None, _c: Any = compiled) -> Any:
            # the tie recorder (a TorchFunctionMode) is popped around the compiled round: the rounds hold no
            # declared site, and dynamo would otherwise trace the recorder's Python
            stack = []
            while th._C._len_torch_function_stack() > 0:
                stack.append(th._C._pop_torch_function_stack())
            try:
                return _c(x, bias)
            finally:
                for md in reversed(stack):
                    th._C._push_on_torch_function_stack(md)
        m.forward = fwd
    return rounds


def uninstall(rounds: list) -> None:
    for m in rounds:
        if "forward" in vars(m):
            del m.forward


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--state", required=True, choices=("fresh", "perturb05", "perturb10", "weights"))
    ap.add_argument("--weights-from", default=None)
    ap.add_argument("--behaviour", required=True, choices=("fp32", "bf16"))
    ap.add_argument("--out", required=True, help="output prefix (writes <out>.json and <out>.npz)")
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--chunk", type=int, default=2048, help="the learner micro-batch (endstate batch_size)")
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--run-seed", type=int, default=1001)
    ap.add_argument("--b8-rows", type=int, default=1024)
    ap.add_argument("--fp64-rows", type=int, default=512)
    ap.add_argument("--compiled-rounds", action="store_true")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--t2-backend", default="eager", choices=("eager", "graph"))
    ap.add_argument("--arch", default="endstate", help="the --arch the learner resolves (the target is endstate)")
    a = ap.parse_args()
    t_start = time.time()
    if a.device == "cpu":
        th.set_num_threads(a.threads)
    th.set_float32_matmul_precision("highest")
    from agents.model.compile_cache import ensure_hermetic_cache
    from agents.model.trunk_precision import trunk_precision_of
    from agents.training.rust_rollout import consistency as K
    from agents.training.rust_rollout.tie_margins import TieMargins

    if a.compiled_rounds or a.t2_backend == "graph":
        ensure_hermetic_cache("k9 bf16 measurement")
    model, env, _args = C.build(state=a.state, weights_from=a.weights_from, seed=a.seed, n_envs=a.n_envs,
                                n_steps=a.n_steps, run_seed=a.run_seed, device=a.device, t2_backend=a.t2_backend,
                                trunk=a.behaviour, arch=a.arch)
    try:
        C.collect(model, env)
        R = C.rows(model)
        n = R["n"]
        pol = model.policy
        p64 = C.fp64_copy(pol)
        res: Dict[str, Any] = {
            "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=C.REPO).stdout.strip(),
            "torch": th.__version__, "device": a.device, "t2_backend": a.t2_backend, "threads": th.get_num_threads(),
            "arch": a.arch, "state": a.state, "weights_from": a.weights_from, "behaviour": a.behaviour, "rows": n, "chunk": a.chunk,
            "n_envs": a.n_envs, "n_steps": a.n_steps, "seed": a.seed, "run_seed": a.run_seed,
            "mu_sha256": hashlib.sha256(R["mu"].tobytes()).hexdigest()[:16],
            "selection_free": bool(K.selection_free(pol)), "eps_grid": list(EPS_GRID)}
        arr: Dict[str, np.ndarray] = {"mu": R["mu"], "acts": R["acts"].numpy(), "masks": R["masks"].numpy().astype(bool)}
        T: Dict[str, float] = {}

        def rec_keep(k: int) -> Any:
            return TieMargins(k, keep_calls=True)

        # ---- the learner probe forward, bf16 trunk and fp32 trunk
        for prec in ("bf16", "fp32"):
            with C.trunk(pol, prec):
                assert trunk_precision_of(pol) == prec
                t0 = time.time()
                lp, full, md = C.forward(model, R, chunk=a.chunk, train=True, grad=True, mode_factory=rec_keep)
                T[f"learner_{prec}_s"] = time.time() - t0
                m, site, per, _f = C.margins_of(md, n)
                arr[f"lp_L{prec}"], arr[f"full_L{prec}"], arr[f"m_L{prec}"] = lp, full, m
                arr[f"site_L{prec}"] = np.asarray(site)
                res[f"_per_L{prec}"] = per
                flips = []
                for eps in (EPS_GRID if prec == "bf16" else (2e-4,)):
                    _lp, _full, mdn = C.forward(model, R, chunk=a.chunk, train=True, grad=True,
                                                mode_factory=lambda k, e=eps: TieMargins(k, near_eps=e))
                    if not np.array_equal(_lp, lp):
                        raise SystemExit(f"the {prec} probe forward is not deterministic (near_eps {eps})")
                    flips.append(C.margins_of(mdn, n)[3])
                arr[f"flip_L{prec}"] = np.stack(flips)
        assert trunk_precision_of(pol) == a.behaviour
        per16 = res.pop("_per_Lbf16")
        per32 = res.pop("_per_Lfp32")

        # ---- T2-like variants (their margins vs the learner's)
        variants = {"eval_b2048": dict(chunk=a.chunk, train=False, grad=False),
                    "train_nograd_b2048": dict(chunk=a.chunk, train=True, grad=False),
                    "eval_b48": dict(chunk=48, train=False, grad=False),
                    "eval_b16": dict(chunk=16, train=False, grad=False),
                    "eval_b8": dict(chunk=8, train=False, grad=False, idx=np.arange(min(n, a.b8_rows))),
                    "jitter": dict(chunk=a.chunk, train=True, grad=True, jit=True)}
        res["variants"] = {}
        for prec in ("bf16", "fp32"):
            base = per16 if prec == "bf16" else per32
            todo = variants if prec == "bf16" else {k: variants[k] for k in ("eval_b48", "jitter")}
            with C.trunk(pol, prec):
                for name, kw in todo.items():
                    kw = dict(kw)
                    jit = kw.pop("jit", False)
                    idx = kw.pop("idx", None)
                    t0 = time.time()
                    if jit:
                        with C.jittered(pol, C.JITTER, seed=77):
                            lp, _full, md = C.forward(model, R, mode_factory=rec_keep, idx=idx, **kw)
                    else:
                        lp, _full, md = C.forward(model, R, mode_factory=rec_keep, idx=idx, **kw)
                    T[f"{prec}_{name}_s"] = time.time() - t0
                    k = n if idx is None else idx.size
                    _m, _s, per_v, _f = C.margins_of(md, k)
                    b = base if idx is None else {s: v[:k] for s, v in base.items()}
                    sd = site_diff(b, per_v)
                    rm = sd.pop("_row_max")
                    full_lp = np.full(n, np.nan)
                    full_lp[:k] = lp
                    arr[f"lp_{prec}_{name}"] = full_lp
                    arr[f"dm_{prec}_{name}"] = np.concatenate([rm, np.full(n - k, np.nan)])
                    res["variants"][f"{prec}_{name}"] = {"rows": int(k), **sd}
                    print(f"[k9-bf16] {prec} {name}: R {sd['max_diff']:.3g}  |dlogp| vs learner max "
                          f"{np.nanmax(np.abs(lp - arr[f'lp_L{prec}'][:k])):.3g} ({T[f'{prec}_{name}_s']:.0f}s)", flush=True)
            if prec == "bf16" and a.compiled_rounds:
                with C.trunk(pol, "bf16"):
                    rounds = install_compiled_rounds(pol)
                    try:
                        t0 = time.time()
                        lp, _full, md = C.forward(model, R, chunk=48, train=False, grad=False, mode_factory=rec_keep)
                        T["bf16_compiled_rounds_b48_s"] = time.time() - t0
                    finally:
                        uninstall(rounds)
                _m, _s, per_v, _f = C.margins_of(md, n)
                sd = site_diff(per16, per_v)
                arr["dm_bf16_compiled_rounds_b48"] = sd.pop("_row_max")
                arr["lp_bf16_compiled_rounds_b48"] = lp
                res["variants"]["bf16_compiled_rounds_b48"] = {"rows": n, **sd}
                print(f"[k9-bf16] bf16 compiled_rounds_b48: R {sd['max_diff']:.3g} |dlogp| vs learner max "
                      f"{np.max(np.abs(lp - arr['lp_Lbf16'])):.3g}", flush=True)

        # ---- the teeth: one-optimizer-step-stale weights, under the bf16 trunk (the probe's own margins)
        with C.trunk(pol, "bf16"):
            for lr in C.STALE_LRS:
                with C.stale_step(pol, lr):
                    lp, full, md = C.forward(model, R, chunk=a.chunk, train=True, grad=True, mode_factory=rec_keep)
                m, _s, _p, _f = C.margins_of(md, n)
                arr[f"lp_stale_{lr:g}"], arr[f"m_stale_{lr:g}"] = lp, m
            # the same fault under the fp32 trunk (today's signal on these rows, for the ratio)
        with C.trunk(pol, "fp32"):
            for lr in C.STALE_LRS:
                with C.stale_step(pol, lr):
                    lp, full, md = C.forward(model, R, chunk=a.chunk, train=True, grad=True, mode_factory=rec_keep)
                m, _s, _p, _f = C.margins_of(md, n)
                arr[f"lp32_stale_{lr:g}"], arr[f"m32_stale_{lr:g}"] = lp, m

        # ---- the float64 reference
        k64 = min(n, a.fp64_rows)
        t0 = time.time()
        arr["lp_fp64"] = np.concatenate([C.fp64_logp(p64, model, R, np.arange(k64)), np.full(n - k64, np.nan)])
        T["fp64_s"] = time.time() - t0
        res["timings_s"] = {k: round(v, 2) for k, v in T.items()}
        res["wall_s"] = round(time.time() - t_start, 1)
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        np.savez_compressed(a.out + ".npz", **arr)
        with open(a.out + ".json", "w") as f:
            json.dump(res, f, indent=1, default=float)
        print(f"[k9-bf16] wrote {a.out}.json/.npz in {res['wall_s']}s", flush=True)
    finally:
        env.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
