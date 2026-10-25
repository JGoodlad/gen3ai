"""X26 continuation (rb_x26c_s2601) u1000 canary near-miss: is the clip-margin read (compiled bf16 0.7124 vs M 0.72 on
fixture row 50) a PEAKED / near-tie policy (P) or growing bf16-vs-fp32 trunk drift (D)? CPU only, eager, at the run's
PIN (f4b25238). Run with cwd=<pin>, PYTHONPATH=<pin>/src, CUDA_VISIBLE_DEVICES="":

    python margin_read.py --ckpt 280M=<zip> --ckpt 380M=<zip> --ckpt 478M=<zip> --out <dir> [--dither 8] [--rows 50,31]

Per checkpoint, on the K9 learner golden's 64 committed rows (`compile_regions.r1_batch`, B = 64 = every fixture row
once; batch row == fixture row), with the learner the canary builds (`diag_bf16_canary.build_learner`, bf16 trunk,
decisions in fp32), train mode, no grad, the forward `micro_step` runs (`evaluate_actions_functional`):
  * arms E32 (the trunk under `fp32_scope`: the canary's fp32 reference) and Ebf (the bf16 trunk: eager bf16);
  * per row: |d log pi(taken)| Ebf vs E32; the legal-logit error range (max - min over legal actions of d log pi, which
    equals the range of the logit error since the normaliser cancels); E32's top-2 margin (in log-prob = logit units),
    entropy, p(taken), the taken action's rank;
  * THE TRUNK: per-row relative L2 error Ebf vs E32 of pi_features (the extractor's output), latent_pi and the pointer
    head's token inputs (move tokens on valid seats, team tokens, move / switch cells);
  * correlations across rows of |d log pi| with 1 / top-2 margin (Pearson and Spearman) and with the trunk error;
  * THE DITHER NULL: N seeded half-bf16-ulp weight dithers (relative 2^-9, `diag_bf16_canary.dithered`): E32~ and Ebf~
    per row vs the undithered E32 -- how far the fp32 function alone, and eager bf16, move each row when only the
    rounding of the weights changes;
  * THE DISCRETE-OP TRACE (the K9 tail sweep's `DiscreteTrace` class, per row): which op sites' discrete outputs
    (topk / sort / argmax / max-with-indices / float comparisons / float->int casts) differ between E32 and each other
    arm ON THE ROWS OF INTEREST, and E32's (fp64-free) per-row margin at each topk / argmax / max site.
Nothing here trains or writes under models/; each checkpoint's JSON is written as soon as it finishes (resumable).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.overrides import TorchFunctionMode

CMP = {"__ge__", "__gt__", "__le__", "__lt__", "ge", "gt", "le", "lt", "greater", "greater_equal", "less",
       "less_equal"}


def _site() -> str:
    f = sys._getframe(2)
    while f is not None:
        fn = f.f_code.co_filename
        if "/src/agents/" in fn:
            return f"{fn.split('/src/', 1)[1]}:{f.f_lineno}"
        f = f.f_back
    return "?"


class RowTrace(TorchFunctionMode):
    """Every discrete output of a forward whose leading dim is a multiple of B (batch-major), reshaped [B, -1];
    for topk / argmax / max-with-dim also the per-row MARGIN (k-th minus (k+1)-th value along the op's dim, min over
    the row's other entries). Ops whose leading dim is not a multiple of B are counted, not compared (FINDING)."""

    def __init__(self, B: int) -> None:
        super().__init__()
        self.B = B
        self.log: List[Tuple[str, str, Optional[np.ndarray], Optional[np.ndarray]]] = []
        self.skipped = 0

    def _rows(self, t: torch.Tensor) -> Optional[np.ndarray]:
        if t.dim() == 0 or t.shape[0] % self.B:
            return None
        return t.detach().reshape(self.B, -1).cpu().numpy().copy()

    def __torch_function__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        name = getattr(func, "__name__", "")
        rec, marg = None, None
        x = args[0] if args else None
        try:
            if name in ("topk", "sort") and isinstance(out, tuple):
                rec = out[1]
                if name == "topk" and isinstance(x, torch.Tensor) and x.is_floating_point():
                    k = int(args[1] if len(args) > 1 else kwargs["k"])
                    dim = int(args[2] if len(args) > 2 else kwargs.get("dim", -1))
                    s = torch.sort(x.detach().double(), dim=dim, descending=True).values
                    if s.shape[dim] > k:
                        marg = (s.narrow(dim, k - 1, 1) - s.narrow(dim, k, 1)).abs()
            elif name in ("argmax", "argmin") or (name in ("max", "min") and isinstance(out, tuple) and len(out) == 2):
                rec = out if name in ("argmax", "argmin") else out[1]
                dim = args[1] if len(args) > 1 else kwargs.get("dim", None)
                if isinstance(x, torch.Tensor) and x.is_floating_point() and dim is not None:
                    s = torch.sort(x.detach().double(), dim=int(dim), descending=name in ("argmax", "max")).values
                    if s.shape[int(dim)] > 1:
                        marg = (s.narrow(int(dim), 0, 1) - s.narrow(int(dim), 1, 1)).abs()
            elif name in CMP and isinstance(out, torch.Tensor) and out.dtype == torch.bool:
                if any(isinstance(a, torch.Tensor) and a.is_floating_point() for a in args):
                    rec = out
                    a0, a1 = args[0], args[1] if len(args) > 1 else None
                    if isinstance(a0, torch.Tensor) and a0.is_floating_point() and a1 is not None:
                        b1 = a1.detach().double() if isinstance(a1, torch.Tensor) else float(a1)
                        marg = torch.broadcast_to((a0.detach().double() - b1).abs(), out.shape)
        except Exception:
            rec, marg = None, None
        if isinstance(rec, torch.Tensor):
            r = self._rows(rec)
            if r is None:
                self.skipped += 1
            m = None
            if isinstance(marg, torch.Tensor) and marg.dim() > 0 and marg.shape[0] % self.B == 0:
                m = marg.reshape(self.B, -1).min(dim=1).values.cpu().numpy()
            self.log.append((name, _site(), r, m))
        return out


def row_changes(a: List[Any], b: List[Any], rows: List[int]) -> Dict[str, Any]:
    """Per row of interest: the op sites (name@site) whose discrete output differs between traces ``a`` and ``b``."""
    if len(a) != len(b) or any((x[0], x[1]) != (y[0], y[1]) for x, y in zip(a, b)):
        return {"control_flow_differs": True}
    res: Dict[str, Any] = {"control_flow_differs": False}
    allrows = np.zeros(a[0][2].shape[0] if a and a[0][2] is not None else 0, dtype=int) if a else None
    for r in rows:
        res[str(r)] = sorted({f"{x[0]}@{x[1]}" for x, y in zip(a, b)
                              if x[2] is not None and y[2] is not None and not np.array_equal(x[2][r], y[2][r])})
    if allrows is not None and allrows.size:
        for x, y in zip(a, b):
            if x[2] is not None and y[2] is not None:
                allrows += (x[2] != y[2]).any(axis=1)
        res["rows_with_any_change"] = int((allrows > 0).sum())
        res["changed_ops_per_row"] = allrows.tolist()
    return res


def forward(model: Any, args: Tuple[Any, ...], *, fp32: bool, trace: Optional[RowTrace] = None) -> Dict[str, torch.Tensor]:
    """`micro_step`'s forward (`evaluate_actions_functional`), split so the trunk's outputs are kept."""
    from agents.model import masked_categorical as mc
    from agents.model.trunk_precision import fp32_scope
    import contextlib
    pol, obs, actions, masks = args[0], args[1], args[2], args[3]
    scope = fp32_scope(pol) if fp32 else contextlib.nullcontext()
    tm = trace if trace is not None else contextlib.nullcontext()
    with scope, torch.no_grad(), tm:
        pi, _vf = pol.extract_features(obs)
        lat = pol.mlp_extractor.forward_actor(pi)
        logp = pol.masked_logp(lat, masks)
        mb = mc.mask_bool(masks, logp)
        lp = mc.log_prob(logp, actions.reshape(-1))
        ent = mc.entropy(logp, mb)
        tok, valid, team, mcells, scells = pol.features_extractor.last_pointer_inputs
    f = lambda t: t.detach().double().cpu()  # noqa: E731
    return {"pi": f(pi), "lat": f(lat), "logp": f(logp), "mask": mb.detach().cpu(), "log_prob": f(lp).reshape(-1),
            "entropy": f(ent).reshape(-1), "tok": f(tok), "valid": valid.detach().cpu().bool(), "team": f(team),
            "mcells": f(mcells), "scells": f(scells)}


def rel_rows(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    B = a.shape[0]
    d = (a - b).reshape(B, -1).norm(dim=1)
    n = b.reshape(B, -1).norm(dim=1).clamp(min=1e-30)
    return (d / n).numpy()


def trunk_err(x: Dict[str, torch.Tensor], ref: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    v = ref["valid"]
    tok_x = torch.where(v[..., None], x["tok"], torch.zeros_like(x["tok"]))
    tok_r = torch.where(v[..., None], ref["tok"], torch.zeros_like(ref["tok"]))
    return {"pi_features": rel_rows(x["pi"], ref["pi"]), "latent_pi": rel_rows(x["lat"], ref["lat"]),
            "move_tokens": rel_rows(tok_x, tok_r), "team_tokens": rel_rows(x["team"], ref["team"]),
            "move_cells": rel_rows(x["mcells"], ref["mcells"]), "switch_cells": rel_rows(x["scells"], ref["scells"])}


def policy_shape(ref: Dict[str, torch.Tensor], actions: torch.Tensor) -> Dict[str, np.ndarray]:
    lp, m = ref["logp"], ref["mask"]
    neg = torch.where(m, lp, torch.full_like(lp, -1e30))
    s = torch.sort(neg, dim=1, descending=True).values
    n_legal = m.sum(1)
    top2 = torch.where(n_legal > 1, s[:, 0] - s[:, 1], torch.full_like(s[:, 0], float("inf")))
    a = actions.reshape(-1).long().to(lp.device)  # actions come back on the model device (CUDA); ref stays on CPU
    taken = lp.gather(1, a[:, None]).reshape(-1)
    rank = (neg > taken[:, None]).sum(1)
    spread = torch.where(n_legal > 1, s[:, 0] - torch.where(m, lp, torch.full_like(lp, 1e30)).min(1).values,
                         torch.zeros_like(s[:, 0]))
    return {"top2_margin": top2.numpy(), "legal_logit_spread": spread.numpy(), "entropy": ref["entropy"].numpy(), "p_taken": taken.exp().numpy(),
            "taken_rank": rank.numpy(), "n_legal": n_legal.numpy(), "p_max": s[:, 0].exp().numpy()}


def logit_range(x: Dict[str, torch.Tensor], ref: Dict[str, torch.Tensor]) -> np.ndarray:
    m = ref["mask"]
    d = x["logp"] - ref["logp"]
    hi = torch.where(m, d, torch.full_like(d, -1e30)).max(1).values
    lo = torch.where(m, d, torch.full_like(d, 1e30)).min(1).values
    return (hi - lo).numpy()


def corr(x: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return {"n": int(x.size)}
    rx, ry = np.argsort(np.argsort(x)), np.argsort(np.argsort(y))
    return {"n": int(x.size), "pearson": float(np.corrcoef(x, y)[0, 1]), "spearman": float(np.corrcoef(rx, ry)[0, 1])}


def q(v: np.ndarray) -> Dict[str, float]:
    v = np.asarray(v, dtype=float)
    return {"median": float(np.median(v)), "p90": float(np.quantile(v, 0.9)), "max": float(v.max()),
            "argmax": int(v.argmax())}


def read(label: str, zip_path: str, a: argparse.Namespace) -> Dict[str, Any]:
    from main.ops import diag_bf16_canary as D
    from agents.model import compile_regions as cr
    from agents.training.instrumented_ppo.micro_step import micro_step
    t0 = time.perf_counter()
    B = D.n_fixture_rows()
    model = D.build_learner(Path(zip_path), a.device, precision="bf16", batch=B)
    b = cr.r1_batch(model, B, 0)
    assert list(b.fixture_rows) == list(range(B)), "batch rows are not the fixture rows in order"
    args = cr._r1_args(model, b)
    rows = [int(r) for r in a.rows.split(",")]
    tr32, trbf = RowTrace(B), RowTrace(B)
    E32 = forward(model, args, fp32=True, trace=tr32)
    Ebf = forward(model, args, fp32=False, trace=trbf)
    # determinism + equivalence with the canary's micro_step forward (grad mode, the relocation's reference)
    E32b = forward(model, args, fp32=True)
    ms = D.run_arm(model, micro_step, args, fp32_trunk=True) if a.check_micro else None
    rec: Dict[str, Any] = {"label": label, "zip": zip_path, "B": B,
                           "determinism_E32_rerun_max": float((E32b["log_prob"] - E32["log_prob"]).abs().max()),
                           "E32_vs_micro_step_max": (float((ms.log_prob.double() - E32["log_prob"]).abs().max())
                                                     if ms is not None else None)}
    dlp = (Ebf["log_prob"] - E32["log_prob"]).abs().numpy()
    shape = policy_shape(E32, args[2])
    lr = logit_range(Ebf, E32)
    te = trunk_err(Ebf, E32)
    inv = 1.0 / np.maximum(shape["top2_margin"], 1e-9)
    rec["per_row"] = {"abs_dlogp_Ebf": dlp.tolist(), "logit_err_range_Ebf": lr.tolist(),
                      **{k: np.asarray(v, dtype=float).tolist() for k, v in shape.items()},
                      **{f"trunk_rel_{k}": v.tolist() for k, v in te.items()}}
    rec["summary"] = {"abs_dlogp_Ebf": q(dlp), "logit_err_range_Ebf": q(lr),
                      **{f"trunk_rel_{k}": q(v) for k, v in te.items()},
                      "top2_margin": q(shape["top2_margin"][np.isfinite(shape["top2_margin"])]),
                      "entropy": q(shape["entropy"]), "p_max_median": float(np.median(shape["p_max"])),
                      "legal_logit_spread": q(shape["legal_logit_spread"]),
                      "latent_pi_norm_median": float(E32["lat"].reshape(B, -1).norm(dim=1).median())}
    rec["corr"] = {"dlogp_vs_inv_top2": corr(dlp, inv), "dlogp_vs_logit_range": corr(dlp, lr),
                   "dlogp_vs_trunk_latent": corr(dlp, te["latent_pi"]),
                   "dlogp_vs_trunk_pi": corr(dlp, te["pi_features"]),
                   "dlogp_vs_inv_p_taken": corr(dlp, 1.0 / np.maximum(shape["p_taken"], 1e-12)),
                   "dlogp_vs_logit_spread": corr(dlp, shape["legal_logit_spread"])}
    rec["trace_E32_vs_Ebf"] = row_changes(tr32.log, trbf.log, rows)
    rec["trace_skipped_ops"] = {"E32": tr32.skipped, "Ebf": trbf.skipped, "total": len(tr32.log)}
    # E32's per-row margins at the discrete sites, for the rows of interest (smallest 8 per row)
    marg: Dict[str, Any] = {}
    for r in rows:
        ms_ = sorted(((float(m[r]), f"{n}@{s}") for n, s, _x, m in tr32.log if m is not None and m[r] > 0),
                     key=lambda t: t[0])
        marg[str(r)] = [{"margin": v, "op": o} for v, o in ms_[:8]]
    rec["E32_smallest_nonzero_margins"] = marg   # an exact 0 is structural (padding compared with 0), never a tie
    # the dither null
    if a.dither:
        dE, dB, chg = [], [], {str(r): {} for r in rows}
        nflip = np.zeros(B, dtype=int)
        for s in range(1, a.dither + 1):
            with D.dithered([model], s):
                t32, tbf = RowTrace(B), RowTrace(B)
                x32 = forward(model, args, fp32=True, trace=t32)
                xbf = forward(model, args, fp32=False, trace=tbf)
            dE.append((x32["log_prob"] - E32["log_prob"]).abs().numpy())
            dB.append((xbf["log_prob"] - E32["log_prob"]).abs().numpy())
            for nm, t in (("E32~", t32), ("Ebf~", tbf)):
                rc = row_changes(tr32.log, t.log, rows)
                if "changed_ops_per_row" in rc:
                    nflip += np.asarray(rc["changed_ops_per_row"]) > 0
                for r in rows:
                    for site in rc.get(str(r), []):
                        chg[str(r)][f"{nm} {site}"] = chg[str(r)].get(f"{nm} {site}", 0) + 1
        dE, dB = np.stack(dE), np.stack(dB)
        rec["dither"] = {"n": a.dither, "rel": D.DITHER_REL,
                         "E32~_per_row_max": dE.max(0).tolist(), "Ebf~_per_row_max": dB.max(0).tolist(),
                         "E32~_max": q(dE.max(0)), "Ebf~_max": q(dB.max(0)),
                         "rows": {str(r): {"E32~": dE[:, r].tolist(), "Ebf~": dB[:, r].tolist()} for r in rows},
                         "site_changes_vs_E32_on_rows": chg,
                         "per_row_arm_seeds_with_a_discrete_change": nflip.tolist(),
                         "corr_Ebf_dlogp_vs_E32dither_sensitivity": corr(dlp, dE.max(0)),
                         "corr_Ebf_dlogp_vs_Ebfdither_max": corr(dlp, dB.max(0))}
    if a.compiled:
        rec["compiled"] = compiled_read(zip_path, a, E32["log_prob"].numpy(), rows)
    rec["seconds"] = time.perf_counter() - t0
    rec["rows_of_interest"] = {str(r): {"abs_dlogp_Ebf": float(dlp[r]), "logit_err_range_Ebf": float(lr[r]),
                                        **{k: float(v[r]) for k, v in shape.items()},
                                        **{f"trunk_rel_{k}": float(v[r]) for k, v in te.items()},
                                        "E32_log_prob": float(E32["log_prob"][r]),
                                        "action": int(args[2].reshape(-1)[r]),
                                        "rank_by_dlogp": int((dlp > dlp[r]).sum())} for r in rows}
    del model
    return rec


def compiled_read(zip_path: str, a: argparse.Namespace, e32_64: np.ndarray, rows: List[int]) -> Dict[str, Any]:
    """THE COMPILED ARM, per fixture row, at the canary's declared signature (B = 2048 = the 64 rows tiled, slice 0):
    the production `compile_regions.install` (CUDA: Triton; CPU: Inductor C++), Cbf / Ebf / E32 through `micro_step`
    with its backward, exactly as the canary's rung runs them (`diag_bf16_canary.run_arm`). Per fixture row the max
    over its tiles of |arm - E32|, and |Cbf - Ebf| (the canary's same-precision gap)."""
    from main.ops import diag_bf16_canary as D
    from agents.model import compile_regions as cr
    from agents.training.instrumented_ppo.micro_step import micro_step
    t0 = time.perf_counter()
    if a.device == "cpu":
        from agents.model.compile_cache import ensure_hermetic_cache
        from agents.model import compile_control as cc
        ensure_hermetic_cache("x26c_canary_margin")
        cc.control().install()
        torch._dynamo.config.cache_size_limit = max(int(torch._dynamo.config.cache_size_limit), 64)
    model = D.build_learner(Path(zip_path), a.device, precision="bf16", batch=a.compiled_batch)
    cr.install(model)
    b = cr.r1_batch(model, a.compiled_batch, 0)
    args, fix = cr.relocate_clip_boundary(model, cr._r1_args(model, b))
    fr = np.asarray(b.fixture_rows)
    E32 = D.run_arm(model, micro_step, args, fp32_trunk=True)
    Ebf = D.run_arm(model, micro_step, args, fp32_trunk=False)
    Cbf = D.run_arm(model, model._compiled_micro_step, args, fp32_trunk=False)
    nfix = int(fr.max()) + 1

    def per_fix(x: torch.Tensor, y: torch.Tensor) -> np.ndarray:
        d = (x.double() - y.double()).abs().numpy()
        return np.array([d[fr == u].max() for u in range(nfix)])
    c, e, g = per_fix(Cbf.log_prob, E32.log_prob), per_fix(Ebf.log_prob, E32.log_prob), per_fix(Cbf.log_prob, Ebf.log_prob)
    tile_spread = np.array([float(np.ptp(Cbf.log_prob.double().numpy()[fr == u])) for u in range(nfix)])
    out = {"device": a.device, "batch": a.compiled_batch, "relocated_rows": int(fix[1].sum()),
           "Cbf_vs_E32": q(c), "Ebf_vs_E32": q(e), "Cbf_vs_Ebf": q(g),
           "E32_2048_vs_E32_64_max": float(np.abs(np.array([E32.log_prob.double().numpy()[fr == u][0]
                                                            for u in range(nfix)]) - e32_64).max()),
           "Cbf_tile_spread_max": float(tile_spread.max()),
           "per_row": {"Cbf_vs_E32": c.tolist(), "Ebf_vs_E32": e.tolist(), "Cbf_vs_Ebf": g.tolist()},
           "rows": {str(r): {"Cbf_vs_E32": float(c[r]), "Ebf_vs_E32": float(e[r]), "Cbf_vs_Ebf": float(g[r])}
                    for r in rows},
           "seconds": time.perf_counter() - t0}
    del model
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", action="append", required=True, help="LABEL=path.zip (repeatable)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--dither", type=int, default=8)
    ap.add_argument("--rows", default="50,31")
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--check-micro", action="store_true", help="also run micro_step (grad mode) once per ckpt")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda (ONLY under the GPU lease: gpu_replay.sh)")
    ap.add_argument("--compiled", action="store_true", help="also the compiled arm per row (compiled_read)")
    ap.add_argument("--compiled-batch", type=int, default=2048)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for spec in a.ckpt:
        label, path = spec.split("=", 1)
        p = out / f"read_{label}.json"
        if p.exists() and not a.force:
            print(f"[margin] {label}: done already ({p})", flush=True)
            continue
        rec = read(label, path, a)
        p.write_text(json.dumps(rec, indent=1))
        s = rec["summary"]
        roi = rec["rows_of_interest"]
        print(f"[margin] {label} {rec['seconds']:.0f}s: max|dlogp| Ebf {s['abs_dlogp_Ebf']} | trunk latent "
              f"{s['trunk_rel_latent_pi']} | corr {rec['corr']['dlogp_vs_inv_top2']} | rows "
              + json.dumps({r: {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()}
                            for r, d in roi.items()}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
