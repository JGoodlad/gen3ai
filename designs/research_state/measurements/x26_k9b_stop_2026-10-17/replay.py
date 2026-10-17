"""K9(b) stop of rb_x26_s2601 at update 13,900: CPU replay + decomposition of the violating row.

Run from a checkout of the RUN'S PINNED code (9c033489: `git archive 9c033489 src` + links to data/ and
deps/), CPU only, with this file's directory as the output:

    PYTHONPATH=<pinned>/src CUDA_VISIBLE_DEVICES= python3 replay.py --hold ~/.cache/gen3ai/x26_k9b_hold \
        --run models/rb_x26_s2601 --out replay.json

Inputs (read only): the K9(b) dump (`behaviour_violation_u13900_policy.pt` / `_rows.npz`,
`behaviour_violations.jsonl`), `final_model_exception.zip` (the weights at the FATAL) and
`checkpoint_136000153_steps.zip` (an earlier policy, for the stale-weights contrast).

The dump holds the observations of the 32 WORST rows of the full-buffer scan (90,615 current rows; the
violating row is the worst), not of all 1,024 probe rows: every per-row number below is over those 32.

What it computes, per row (float64 log pi of the taken action unless named):
  stored   the behaviour log-prob (T2, fp32, compiled + CUDA graph, eval mode, bucket 8/64/256, GPU)
  gpu      the learner's recomputation at the FATAL (fp32 'highest', train mode, batch chunk, GPU)
  ref64    THIS script: the learner forward in float64 (eager, CPU) -- the exact function, to ~1e-12
  learn32  the learner forward (evaluate_actions, train mode, grad on), fp32 CPU, batch of 32
  dec32_B  the T2 decision forward (DecisionModule, eval, no_grad), fp32 CPU, at bucket B (rows tiled)
  pert     the fp32 forward's sensitivity: the std / max over K draws of a one-ulp random rounding of
           every weight (each weight moved to a float32 neighbour with prob 1/2) -- a direct measure of how
           far two equally valid fp32 evaluations of THIS row can sit apart
  stale    the same rows under the earlier checkpoint's weights (what a stale-weight fault would look like)
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from typing import Any, Dict

import numpy as np
import torch as th


def _obs(z: Any, prefix: str, idx: np.ndarray) -> Dict[str, np.ndarray]:
    return {k[len(prefix):]: z[k][idx] for k in z.files if k.startswith(prefix)}


class f64_forward(th.overrides.TorchFunctionMode):
    """A float64 forward of code that writes float32 in places (``.float()``, ``dtype=torch.float32``
    factories, float32 constant buffers): EVERY torch call inside sees its float32 tensor operands upcast to
    float64 and any float32 dtype argument replaced by float64, so every intermediate is float64 -- the
    exact function to ~1e-12, a reference for the fp32 evaluations. ``calls`` / ``upcasts`` count the work."""

    def __enter__(self) -> "f64_forward":
        self.calls = 0
        self.upcasts = 0
        self._orig_dtype = th.get_default_dtype()
        th.set_default_dtype(th.float64)
        return super().__enter__()  # type: ignore[no-any-return]

    def __exit__(self, *exc: Any) -> None:
        super().__exit__(*exc)
        th.set_default_dtype(self._orig_dtype)

    def _up(self, x: Any) -> Any:
        if isinstance(x, th.Tensor) and x.dtype in (th.float32, th.float16, th.bfloat16):
            self.upcasts += 1
            return x.double()
        if x is th.float32 or x is th.float16 or x is th.bfloat16:
            return th.float64
        if isinstance(x, (list, tuple)) and not isinstance(x, th.Size):
            y = [self._up(v) for v in x]
            return type(x)(y) if not hasattr(x, "_fields") else type(x)(*y)
        if isinstance(x, dict):
            return {k: self._up(v) for k, v in x.items()}
        return x

    def __torch_function__(self, func: Any, types: Any, args: Any = (), kwargs: Any = None) -> Any:
        self.calls += 1
        if func is th.Tensor.float:            # .float() of a float64 tensor stays float64
            return args[0].double() if args[0].is_floating_point() else func(*args, **(kwargs or {}))
        return func(*self._up(tuple(args)), **self._up(dict(kwargs or {})))


def _ctx(dtype: th.dtype) -> Any:
    import contextlib
    return f64_forward() if dtype == th.float64 else contextlib.nullcontext()


def learner_logp(policy: Any, obs: Dict[str, np.ndarray], acts: np.ndarray, masks: np.ndarray,
                 dtype: th.dtype = th.float32) -> np.ndarray:
    """policy.evaluate_actions as the probe runs it (train mode, grad on, nothing kept)."""
    if dtype == th.float64:
        with f64_forward():
            return _learner(policy, obs, acts, masks, dtype)
    return _learner(policy, obs, acts, masks, dtype)


def _learner(policy: Any, obs: Dict[str, np.ndarray], acts: np.ndarray, masks: np.ndarray,
             dtype: th.dtype) -> np.ndarray:
    o = {k: th.as_tensor(v).to(dtype) if np.issubdtype(v.dtype, np.floating) else th.as_tensor(v)
         for k, v in obs.items()}
    policy.set_training_mode(True)
    with th.enable_grad():
        _v, lp, _e = policy.evaluate_actions(o, th.as_tensor(acts).long(), action_masks=th.as_tensor(masks))
    policy.set_training_mode(False)
    return lp.detach().double().numpy()


def decision_logp(dec: Any, obs: np.ndarray, masks: np.ndarray, acts: np.ndarray, bucket: int,
                  dtype: th.dtype = th.float32) -> np.ndarray:
    """The T2 decision forward (eval, no_grad) at one bucket shape: each row placed at every slot it
    can take in a bucket filled with the other rows (tiled), so the row meets several batch compositions."""
    n = obs.shape[0]
    out = np.empty(n)
    reps = int(np.ceil(bucket / n)) + 1
    tiled_o = np.concatenate([obs] * reps)[:bucket]
    tiled_m = np.concatenate([masks] * reps)[:bucket]
    with th.no_grad(), _ctx(dtype):
        logp, _v = dec(th.as_tensor(tiled_o).to(dtype), th.as_tensor(tiled_m).bool())
    lp = logp.double().numpy()
    for i in range(n):
        out[i] = lp[i, acts[i]] if i < bucket else np.nan
    return out


def raw_logits(policy: Any, dec: Any, obs: np.ndarray, dtype: th.dtype) -> Dict[str, np.ndarray]:
    fe = dec.fe
    stats: Dict[str, float] = {}
    outs: Dict[str, np.ndarray] = {}
    hooks = []
    B = obs.shape[0]

    def mk(name: str) -> Any:
        def h(_m: Any, _i: Any, out: Any) -> None:
            t = out[0] if isinstance(out, tuple) else out
            if th.is_tensor(t) and t.is_floating_point():
                t = t.detach()
                stats[name] = max(stats.get(name, 0.0), float(t[th.isfinite(t)].abs().max()) if t.numel() else 0.0)
                if t.dim() >= 1 and t.shape[0] == B and name not in outs:     # the FIRST call's per-row output
                    outs[name] = t.reshape(B, -1).double().numpy().copy()
        return h
    for name, m in policy.named_modules():
        if name.count(".") <= 2 and name:
            hooks.append(m.register_forward_hook(mk(name)))
    try:
        with th.no_grad(), _ctx(dtype):
            pi, vf = fe({"observation": th.as_tensor(obs).to(dtype)})
            latent_pi = dec.mlp.forward_actor(pi)
            tok_req, valid, team_tokens, move_cells, switch_cells = fe.last_pointer_inputs
            raw = dec.ptr(latent_pi, tok_req, valid, team_tokens, move_cells, switch_cells)
    finally:
        for h in hooks:
            h.remove()
    return {"raw": raw.double().numpy(), "latent_pi_absmax": latent_pi.abs().amax(-1).double().numpy(),
            "module_absmax": stats, "outs": outs}


def module_error(rl32: Dict[str, Any], rl64: Dict[str, Any]) -> Dict[str, Any]:
    """Per module (first call), per row: the fp32 output's relative error against the fp64 one
    (||o32 - o64|| / ||o64||, finite entries only) -- where in the forward a row's rounding grows."""
    res: Dict[str, Any] = {}
    for name, o32 in rl32["outs"].items():
        o64 = rl64["outs"].get(name)
        if o64 is None or o64.shape != o32.shape:
            continue
        fin = np.isfinite(o32) & np.isfinite(o64) & (np.abs(o64) < 1e6)
        num = np.sqrt(np.where(fin, (o32 - o64) ** 2, 0.0).sum(1))
        den = np.sqrt(np.where(fin, o64 ** 2, 0.0).sum(1)) + 1e-30
        res[name] = (num / den).tolist()
    return res


def perturbed(policy: Any, rng: np.random.Generator) -> Any:
    """A copy whose every float weight is moved to a float32 neighbour with probability 1/2 (one ulp)."""
    p = copy.deepcopy(policy)
    with th.no_grad():
        # PARAMETERS only: the float BUFFERS are lookup tables (move base powers, type multipliers, priors)
        # that the forward compares for equality -- a one-ulp move there is a different table, not rounding
        for prm in p.parameters():
            w = prm.data
            if w.dtype != th.float32:
                continue
            g = th.from_numpy(rng.integers(0, 3, size=tuple(w.shape)).astype(np.int8))  # 0 down, 1 keep, 2 up
            up = th.nextafter(w, th.full_like(w, float("inf")))
            dn = th.nextafter(w, th.full_like(w, float("-inf")))
            prm.data = th.where(g == 2, up, th.where(g == 0, dn, w))
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=24, help="perturbation draws")
    ap.add_argument("--no-compile", action="store_true", help="skip the inductor-compiled decision path")
    a = ap.parse_args()
    th.set_float32_matmul_precision("highest")
    th.set_num_threads(int(os.environ.get("K9B_THREADS", "3")))
    th.manual_seed(0)
    from agents.inference.service.decision import DecisionModule
    from main.capacity import load_policy

    t0 = time.time()
    model, _cfg, _ = load_policy(os.path.join(a.hold, "final_model_exception.zip"), a.run, "cpu")
    policy = model.policy
    sd = th.load(os.path.join(a.hold, "behaviour_violation_u13900_policy.pt"), map_location="cpu")
    psd = policy.state_dict()
    weights_equal = set(sd) == set(psd) and all(th.equal(sd[k], psd[k]) for k in sd)
    policy.load_state_dict(sd)
    template = copy.deepcopy(policy)        # never runs a forward: every copy below is taken from it
    rec = json.loads(open(os.path.join(a.hold, "behaviour_violations.jsonl")).readline())
    z = np.load(os.path.join(a.hold, "behaviour_violation_u13900_rows.npz"))
    n = z["scan_t"].shape[0]
    idx = np.arange(n)
    obs = _obs(z, "scan_obs__", idx)
    acts = z["scan_action"].astype(np.int64)
    masks = obs["action_mask"].astype(bool)
    stored = z["scan_logp_stored"]
    gpu = z["scan_logp_recomputed"]
    worst = rec["scan"]["worst"]
    # the dump is aligned with the JSON's worst rows (same order): a structural check of the dump itself
    align = all(abs(w["logp_stored"] - stored[i]) == 0 and w["action"] == acts[i] for i, w in enumerate(worst))
    mask_match = all(w["mask"] == "".join("1" if x else "0" for x in masks[i]) for i, w in enumerate(worst))
    legal_stored_match = all([d is not None for d in w["dist_stored"]] == list(masks[i]) for i, w in enumerate(worst))
    out: Dict[str, Any] = {"weights_dump_equals_final_model_exception": bool(weights_equal),
                           "dump_rows_align_with_json": bool(align), "dump_masks_match_json": bool(mask_match),
                           "stored_dist_legal_set_equals_mask": bool(legal_stored_match), "n_rows": int(n)}
    # 1. the learner path, fp32 and fp64
    learn32 = learner_logp(policy, obs, acts, masks)
    p64 = copy.deepcopy(template).double()
    ref64 = learner_logp(p64, obs, acts, masks, th.float64)
    learn32_single = np.array([learner_logp(policy, {k: v[i:i + 1] for k, v in obs.items()}, acts[i:i + 1],
                                            masks[i:i + 1])[0] for i in range(n)])
    # 2. the T2 decision forward (eval, no_grad), fp32 at each bucket, and in fp64
    dec = DecisionModule(policy).eval()
    dec64 = DecisionModule(p64).eval()
    o = obs["observation"]
    dec32 = {b: decision_logp(dec, o, masks, acts, b) for b in (8, 64, 256)}
    dec32[8] = np.concatenate([decision_logp(dec, o[s:s + 8], masks[s:s + 8], acts[s:s + 8], 8)
                               for s in range(0, n, 8)])
    dec64v = decision_logp(dec64, o, masks, acts, 64, th.float64)
    print(f"[replay] forwards done {time.time() - t0:.0f}s", flush=True)
    # 3. the row's logits
    rl32 = raw_logits(policy, dec, o, th.float32)
    rl64 = raw_logits(p64, dec64, o, th.float64)
    gap32 = rl32["raw"]
    gap64 = rl64["raw"]
    merr = module_error(rl32, rl64)
    # 3b. the T2 path's own kind of forward: the decision module COMPILED by inductor (CPU here; T2 is inductor
    # CUDA + a CUDA graph), at bucket 256 -- one more equally valid fp32 evaluation with fused reductions
    comp = None
    if not a.no_compile:
        try:
            cdec = th.compile(DecisionModule(copy.deepcopy(template)).eval(), dynamic=False)
            comp = decision_logp(cdec, o, masks, acts, 256)
        except Exception as exc:  # recorded, not fatal: a diagnostic path
            out["compiled_error"] = repr(exc)[:500]
    print(f"[replay] compiled done {time.time() - t0:.0f}s", flush=True)
    # 4. sensitivity: one-ulp random rounding of every weight, K draws (fp32 decision forward, bucket 64)
    rng = np.random.default_rng(20261017)
    pert = []
    for k in range(a.k):
        pp = perturbed(template, rng)
        pert.append(decision_logp(DecisionModule(pp).eval(), o, masks, acts, 64))
        del pp
    pert = np.array(pert)
    print(f"[replay] perturbations done {time.time() - t0:.0f}s", flush=True)
    # 5. stale weights: the earlier checkpoint
    stale_path = os.path.join(a.hold, "checkpoint_136000153_steps.zip")
    m2, _c2, _ = load_policy(stale_path, a.run, "cpu")
    stale = learner_logp(m2.policy, obs, acts, masks)
    rows = []
    for i in range(n):
        lg = rl64["raw"][i]
        legal = masks[i]
        rows.append({
            "i": i, "t": int(z["scan_t"][i]), "e": int(z["scan_e"][i]), "action": int(acts[i]),
            "mask": "".join("1" if x else "0" for x in legal), "n_legal": int(legal.sum()),
            "stored": float(stored[i]), "gpu_learner": float(gpu[i]), "ref64": float(ref64[i]),
            "dec64": float(dec64v[i]), "learn32_cpu_b32": float(learn32[i]), "learn32_cpu_b1": float(learn32_single[i]),
            "dec32_cpu": {str(b): float(v[i]) for b, v in dec32.items()},
            "d_stored_gpu": float(stored[i] - gpu[i]), "d_stored_ref64": float(stored[i] - ref64[i]),
            "d_gpu_ref64": float(gpu[i] - ref64[i]),
            "pert_std": float(pert[:, i].std()), "pert_maxdev": float(np.abs(pert[:, i] - ref64[i]).max()),
            "pert_range": float(pert[:, i].max() - pert[:, i].min()),
            "stale_d": float(stale[i] - ref64[i]),
            "raw_logits_legal": [float(x) for x in lg[legal]], "raw_absmax_all": float(np.abs(lg).max()),
            "raw_absmax_legal": float(np.abs(lg[legal]).max()), "latent_pi_absmax": float(rl64["latent_pi_absmax"][i]),
            "p_action": float(np.exp(ref64[i])),
            "compiled32_cpu_b256": None if comp is None else float(comp[i]),
            "raw_legal_err32": [float(x) for x in (gap32[i] - gap64[i])[legal]],
        })
    out["rows"] = rows
    out["module_absmax_fp32_batch"] = rl32["module_absmax"]
    out["module_relerr_fp32_vs_fp64"] = merr
    out["k_perturb"] = a.k
    out["seconds"] = time.time() - t0
    with open(a.out, "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ("rows", "module_absmax_fp32_batch",
                                                                "module_relerr_fp32_vs_fp64")}))
    for r in rows:
        print(f"{r['i']:2d} |st-gpu| {abs(r['d_stored_gpu']):.2e} st-ref {r['d_stored_ref64']:+.2e} "
              f"gpu-ref {r['d_gpu_ref64']:+.2e} l32-ref {r['learn32_cpu_b32'] - r['ref64']:+.2e} "
              f"d256-ref {r['dec32_cpu']['256'] - r['ref64']:+.2e} pert_std {r['pert_std']:.2e} "
              f"pert_max {r['pert_maxdev']:.2e} stale {r['stale_d']:+.2e} nleg {r['n_legal']} "
              f"rawmax {r['raw_absmax_legal']:.1f} p {r['p_action']:.3f}")


if __name__ == "__main__":
    main()
