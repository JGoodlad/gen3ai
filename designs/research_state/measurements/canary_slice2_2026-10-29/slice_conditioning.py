"""CPU read for the slice-2-only canary refusals (2026-10-29): per slice, the fp32 REFERENCE gradient's conditioning.

For one canary state (a dump, or a checkpoint as the canary's anchor) and each fixture slice (0 = all 64 golden rows,
2 = rows 0-31, 1 = rows 32-63; one copy per row, so B = 64 / 32 / 32), on the rows relocated off the clip boundary
EXACTLY as the canary relocates them (`compile_regions.relocate_clip_boundary`, the run's own margin):

  * per row: fixture row, normalised advantage A, reference ratio rho (after relocation), relocated?, policy
    coefficient c_i = d(policy loss)/d(log pi_i) (0 on a clipped row);
  * per parameter p: ||g_p|| (the full R1 loss's gradient, fp32 eager), the policy part ||sum_i c_i grad_p log pi_i||,
    the cancellation index CI_p = sum_i ||c_i grad_p log pi_i|| / ||g_p||, and the per-row EXPOSURE
    S_ip = ||c_i grad_p log pi_i|| / ||g_p|| — to first order, a forward log-prob shift d on row i alone moves g_p by
    d * S_ip relative (d(rho_i)/rho_i = d), which is how a bf16 arm's per-row forward error reaches a gradient ratio;
  * the same on the RAW (unrelocated) rows, to separate the relocation's reweighting from the rows themselves.

Deviation from the canary, declared: B = one copy per row (the canary tiles to the run's B = 2,048). The mean losses are
tiling-invariant; the advantage normalisation's unbiased std differs by sqrt((n-1)/n * N/(N-1)) (0.984 at n = 32), a
uniform scale on A that moves no relocation decision (sign of A and rho only).

    PYTHONPATH=src python designs/research_state/measurements/canary_slice2_2026-10-29/slice_conditioning.py \\
        --run-dir models/<run> (--dump <x>_policy.pt | --anchor <zip>) [--shift-max 0.3] --params a,b --out <json>
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


def _norms(grads: List[Optional[Any]], names: List[str]) -> Dict[str, float]:
    return {n: (float(g.detach().double().norm()) if g is not None else 0.0) for n, g in zip(names, grads)}


def read_slice(model: Any, slice_: int, relocate: bool, top_params: List[str]) -> Dict[str, Any]:
    import torch

    from agents.model import compile_regions as cr
    from agents.model.compile_gate_probe import grad_parameters
    from agents.model.trunk_precision import fp32_scope
    from agents.training.instrumented_ppo.micro_step import micro_step
    n = 64 if slice_ == 0 else 32
    b = cr.r1_batch(model, n, slice_)
    args0 = cr._r1_args(model, b)
    if relocate:
        args, fix = cr.relocate_clip_boundary(model, args0)
        moved = [bool(x) for x in fix.moved.reshape(-1).tolist()]
    else:
        args, moved = args0, [False] * n
    st = args[-1]
    names_params = grad_parameters(model, model.policy.features_extractor)
    names = [nm for nm, _ in names_params]
    params = [p for _, p in names_params]
    with fp32_scope(model.policy):
        out = micro_step(*args)
        old = args[4].detach().double().reshape(-1)
        lp = out.log_prob.double().reshape(-1)
        adv = out.advantages.detach().double().reshape(-1)
        rho = torch.exp(lp.detach() - old)
        eps = float(st.clip_range)
        active = torch.where(adv > 0, rho < 1 + eps, rho > 1 - eps) & adv.ne(0)
        if st.fork_pg_mask:
            raise SystemExit("fork_pg_mask run: the per-row coefficient formula here assumes the plain mean")
        coef = torch.where(active, -adv * rho / n * float(st.policy_grad_coef), torch.zeros_like(adv))
        g_all = torch.autograd.grad(out.loss, params, retain_graph=True, allow_unused=True)
        tot = _norms(list(g_all), names)
        pol_vec = torch.autograd.grad((coef.detach() * lp).sum(), params, retain_graph=True, allow_unused=True)
        pol = _norms(list(pol_vec), names)
        rest = {nm: (float((g - (q if q is not None else 0)).double().norm()) if g is not None else 0.0)
                for nm, g, q in zip(names, g_all, pol_vec)}
        per_row = []
        for i in range(n):
            gi = torch.autograd.grad(lp[i], params, retain_graph=(i < n - 1), allow_unused=True)
            per_row.append({nm: abs(float(coef[i])) * (float(g.double().norm()) if g is not None else 0.0)
                            for nm, g in zip(names, gi)})
    big = max(tot.values())
    judged = [nm for nm in names if tot[nm] > 1e-3 * big]
    fixture = [int(x) for x in b.fixture_rows] if b.fixture_rows is not None else list(range(n))

    def exposure(nm: str) -> Dict[str, Any]:
        s = [per_row[i][nm] / tot[nm] if tot[nm] > 0 else float("inf") for i in range(n)]
        order = sorted(range(n), key=lambda i: -s[i])
        return {"grad_norm": tot[nm], "grad_norm_rel_to_largest": tot[nm] / big, "judged": nm in judged,
                "policy_part": pol[nm] / tot[nm] if tot[nm] else None,
                "rest_part": rest[nm] / tot[nm] if tot[nm] else None,
                "cancellation_index": sum(per_row[i][nm] for i in range(n)) / tot[nm] if tot[nm] else None,
                "top_rows": [{"fixture_row": fixture[i], "S": s[i], "A": float(adv[i]), "rho": float(rho[i]),
                              "relocated": moved[i]} for i in order[:5]],
                "S_by_fixture_row": {str(fixture[i]): s[i] for i in range(n)}}

    # the judged parameters most exposed to ONE row's forward shift (max_i S_ip), the run-wide ranking
    rank = sorted(judged, key=lambda nm: -max(per_row[i][nm] for i in range(n)) / tot[nm])
    res = {"slice": slice_, "rows": n, "relocated": sum(moved), "relocate": relocate,
           "margin": float(cr.clip_margin_of(model)), "judged_params": len(judged), "largest_grad_norm": big,
           "loss": float(out.loss.detach()),
           "row_table": [{"fixture_row": fixture[i], "A": float(adv[i]), "rho": float(rho[i]), "relocated": moved[i],
                          "active": bool(active[i]), "coef_abs_x_n": abs(float(coef[i])) * n} for i in range(n)],
           "watched": {nm: exposure(nm) for nm in top_params if nm in tot},
           "most_exposed": [{"param": nm, "max_S": max(per_row[i][nm] for i in range(n)) / tot[nm],
                             "CI": sum(per_row[i][nm] for i in range(n)) / tot[nm]} for nm in rank[:15]],
           "median_max_S": sorted(max(per_row[i][nm] for i in range(n)) / tot[nm] for nm in judged)[len(judged) // 2]}
    del out, g_all, pol_vec, per_row
    for p in params:
        p.grad = None
    return res


def _e32_grads(model: Any, rows: List[int], watched: List[str], relocate: bool = True) -> Dict[str, Any]:
    """The fp32 eager R1 gradient of ``watched`` on fixture ``rows`` (one copy each), relocated at the run's margin."""
    import numpy as np
    import torch

    from agents.model import compile_regions as cr
    from agents.model import compile_trainer as ct
    from agents.model.trunk_precision import fp32_scope
    from agents.training.instrumented_ppo.micro_step import micro_step
    real = ct.fixture_index
    ct.fixture_index = lambda batch, n_rows, slice_=0: np.asarray(rows)[np.arange(int(batch)) % len(rows)]
    try:
        b = cr.r1_batch(model, len(rows), 0)
    finally:
        ct.fixture_index = real
    args = cr._r1_args(model, b)
    if relocate:
        args, _ = cr.relocate_clip_boundary(model, args)
    named = dict(model.policy.named_parameters())
    ps = [named[n] for n in watched]
    with fp32_scope(model.policy):
        out = micro_step(*args)
        gs = torch.autograd.grad(out.loss, ps, allow_unused=True)
    return {n: (g.detach().double().clone() if g is not None else torch.zeros_like(p, dtype=torch.float64))
            for n, g, p in zip(watched, gs, ps)}


def _rows(slice_: int) -> List[int]:
    return list(range(64)) if slice_ == 0 else (list(range(32)) if slice_ == 2 else list(range(32, 64)))


def dither_null(model: Any, slice_: int, watched: List[str], seeds: int) -> Dict[str, Any]:
    """d(E32_s, E32) per watched parameter under the canary's half-ulp weight dither (`diag_bf16_canary.dithered`):
    how far the fp32 FUNCTION's own gradient moves under rounding-sized noise (the conditioning floor)."""
    from main.ops.diag_bf16_canary import dithered
    rows = _rows(slice_)
    g0 = _e32_grads(model, rows, watched)
    per: Dict[str, List[float]] = {n: [] for n in watched}
    for s in range(seeds):
        with dithered([model], s):
            gs = _e32_grads(model, rows, watched)
        for n in watched:
            den = float(g0[n].norm())
            per[n].append(float((gs[n] - g0[n]).norm()) / den if den > 0 else float("inf"))
    return {n: {"max": max(v), "median": sorted(v)[len(v) // 2], "all": v} for n, v in per.items()}


def leave_one_out(model: Any, slice_: int, watched: List[str]) -> Dict[str, Any]:
    """Per watched parameter: ||g - g_(-i)|| / ||g|| for each dropped fixture row i (full loss, relocated; the drop
    renormalises the advantages, so on a policy-carried parameter this is approximate)."""
    rows = _rows(slice_)
    g0 = _e32_grads(model, rows, watched)
    out: Dict[str, Dict[str, float]] = {n: {} for n in watched}
    for r in rows:
        gs = _e32_grads(model, [x for x in rows if x != r], watched)
        for n in watched:
            den = float(g0[n].norm())
            out[n][str(r)] = float((gs[n] - g0[n]).norm()) / den if den > 0 else float("inf")
    return {n: dict(sorted(v.items(), key=lambda kv: -kv[1])[:6]) for n, v in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--dump", default="")
    ap.add_argument("--anchor", default="")
    ap.add_argument("--shift-max", type=float, default=None)
    ap.add_argument("--params", default="")
    ap.add_argument("--slices", default="0,2,1")
    ap.add_argument("--raw", action="store_true", help="also read the UNRELOCATED rows")
    ap.add_argument("--dither", type=int, default=0, help="fp32 half-ulp weight-dither null: seeds per slice")
    ap.add_argument("--loo", action="store_true", help="leave-one-row-out: which rows carry each watched gradient")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    import torch
    torch.set_num_threads(a.threads)
    from main.ops import canary_dump_replay as R
    t0 = time.time()
    model, build = R.build_model(Path(a.run_dir), Path(a.dump) if a.dump else None, "cpu", 64,
                                 anchor=Path(a.anchor) if a.anchor else None, shift_max=a.shift_max)
    watched = [p for p in a.params.split(",") if p]
    out: Dict[str, Any] = {"build": {k: v for k, v in build.items() if k != "compile_keys"}, "slices": {}}
    for s in [int(x) for x in a.slices.split(",")]:
        out["slices"][str(s)] = read_slice(model, s, True, watched)
        if a.raw:
            out["slices"][f"{s}_raw"] = read_slice(model, s, False, watched)
        print(f"slice {s} done at {time.time() - t0:.0f}s", flush=True)
    if a.dither:
        out["dither"] = {str(s): dither_null(model, s, watched, a.dither) for s in [int(x) for x in a.slices.split(",")]}
        print(f"dither done at {time.time() - t0:.0f}s", flush=True)
    if a.loo:
        out["loo"] = {str(s): leave_one_out(model, s, watched) for s in [int(x) for x in a.slices.split(",")]}
        print(f"loo done at {time.time() - t0:.0f}s", flush=True)
    out["seconds"] = time.time() - t0
    Path(a.out).write_text(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
