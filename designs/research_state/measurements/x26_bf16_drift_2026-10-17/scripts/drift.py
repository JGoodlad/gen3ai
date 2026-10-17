"""CPU read of the R1 startup gate's bf16 LOSS DRIFT on X26 checkpoints (scratch analysis, agent a57c).

Per checkpoint, on the K9 golden rows RELOCATED exactly as the gate does (fp32 reference forward -> clip_relocation at
the default margin), eager fp32 (E32, the reference) vs eager bf16 (Ebf):
  * drift_loss = |L_bf - L_32| / max(1, |L_32|)   (the gate's DRIFT statistic)
  * per-term |term_bf - term_32| (policy, entropy, win_prob, belief rows, intent ...)
  * the policy term per ROW: -min(A rho, A clip rho)/B, its bf16 change, the rows carrying it
  * grad rel-L2, per-param error on the named heads
  * attribution: every trunk round bf16 ALONE (the others fp32)
  * the dithered null (N seeds, rel 2^-9): the drift statistic's distribution at these weights
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from main.ops import diag_bf16_canary as D


def round_modules(model: Any) -> List[tuple]:
    from agents.model.trunk_precision import _round_types
    tbl = _round_types()
    bf = set(tbl.values())
    return [(n, m) for n, m in model.policy.named_modules() if type(m) in bf]


@contextlib.contextmanager
def only_bf16(model: Any, keep: Optional[set]):
    """Every trunk round fp32 except the names in ``keep`` (None = all bf16, i.e. no change)."""
    from agents.model.trunk_precision import _round_types
    tbl = _round_types()
    inv = {v: k for k, v in tbl.items()}
    swapped = []
    try:
        if keep is not None:
            for n, m in round_modules(model):
                if n not in keep:
                    swapped.append((m, type(m)))
                    m.__class__ = inv[type(m)]
        yield
    finally:
        for m, c in swapped:
            m.__class__ = c


def run(model: Any, call_args: tuple, *, fp32: bool, names: List[str]) -> Dict[str, Any]:
    from agents.model import compile_trainer as ct
    from agents.model.trunk_precision import fp32_scope
    from agents.training.instrumented_ppo.micro_step import micro_step
    params = [p for _, p in ct.grad_parameters(model, model.policy.features_extractor)]
    for p in params:
        p.grad = None
    scope = fp32_scope(model.policy) if fp32 else contextlib.nullcontext()
    with scope:
        out = micro_step(*call_args)
        out.loss.backward()
    gs = [(p.grad if p.grad is not None else torch.zeros_like(p)).detach().double().flatten() for p in params]
    for p in params:
        p.grad = None
    return {"loss": float(out.loss.detach().double()),
            "terms": {k: float(v.detach().double()) for k, v in out.terms.items()},
            "grad": torch.cat(gs), "log_prob": out.log_prob.detach().double().clone(),
            "logp": out.logp.detach().double().clone() if out.logp is not None else None,
            "adv": out.advantages.detach().double().clone(), "values": out.values.detach().double().clone()}


def rel(a: float, b: float) -> float:
    return abs(a - b) / max(1.0, abs(b))


def rel_l2(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm())


def per_param(a: torch.Tensor, b: torch.Tensor, names, sizes) -> Dict[str, float]:
    out = {}
    for n, x, y in zip(names, torch.split(a, list(sizes)), torch.split(b, list(sizes))):
        nb = float(y.norm())
        if nb > 0:
            out[n] = float((x - y).norm()) / nb
    return out


def pg_rows(arm: Dict[str, Any], old: torch.Tensor, clip: float) -> torch.Tensor:
    rho = (arm["log_prob"] - old.double().cpu()).exp()
    a = arm["adv"]
    return -torch.min(a * rho, a * rho.clamp(1 - clip, 1 + clip))   # per row; the term is its mean


def read(model, names, sizes, call_args, rounds, *, attrib: bool, dither: int, shifts=()) -> Dict[str, Any]:
    from agents.model import compile_regions as cr
    st = call_args[-1]
    clip = float(st.clip_range)
    ref0 = run(model, call_args, fp32=True, names=names)
    fix = cr.clip_relocation(ref0["log_prob"].float(), call_args[4], ref0["adv"].float(), clip, cr.R1_CLIP_MARGIN)
    args = call_args[:4] + (fix.old_log_prob.to(call_args[4].device),) + call_args[5:]
    old = args[4]
    e32 = run(model, args, fp32=True, names=names)
    ebf = run(model, args, fp32=False, names=names)
    B = int(old.shape[0])
    r32, rbf = pg_rows(e32, old, clip), pg_rows(ebf, old, clip)
    drow = (rbf - r32) / B
    dlp = (ebf["log_prob"] - e32["log_prob"])
    rho = (e32["log_prob"] - old.double().cpu()).exp()
    top = torch.argsort(drow.abs(), descending=True)[:6]
    from agents.model import compile_regions as cr2
    act = cr2.clip_active(e32["log_prob"].float(), old.float().cpu(), e32["adv"].float(), clip).double()
    S = float((rho * e32["adv"].abs() * act).sum() / B)
    dpg = ebf["terms"]["policy"] - e32["terms"]["policy"]
    rest32 = e32["loss"] - e32["terms"]["policy"]
    restbf = ebf["loss"] - ebf["terms"]["policy"]
    # raw rows (no relocation): the loss is continuous in rho, so its drift needs no relocation
    w32 = run(model, call_args, fp32=True, names=names)
    wbf = run(model, call_args, fp32=False, names=names)
    rho_raw = (w32["log_prob"] - call_args[4].double().cpu()).exp()
    act_raw = cr2.clip_active(w32["log_prob"].float(), call_args[4].float().cpu(), w32["adv"].float(), clip).double()
    S_raw = float((rho_raw * w32["adv"].abs() * act_raw).sum() / B)
    rec0 = {"S": S, "s_pg": abs(dpg) / S if S > 0 else float("nan"), "dpg": dpg,
            "rest_drift": rel(restbf, rest32), "rest_diff": restbf - rest32, "L_rest": rest32, "L_pg": e32["terms"]["policy"],
            "weighted_abs_shift": float((rho * e32["adv"].abs() * act * dlp.abs()).sum() / B) / S if S > 0 else float("nan"),
            "raw": {"drift_loss": rel(wbf["loss"], w32["loss"]), "loss_diff": wbf["loss"] - w32["loss"],
                    "dpg": wbf["terms"]["policy"] - w32["terms"]["policy"], "S": S_raw,
                    "drift_grad": rel_l2(wbf["grad"], w32["grad"]),
                    "max_abs_dlogp": float((wbf["log_prob"] - w32["log_prob"]).abs().max())}}
    rec: Dict[str, Any] = {**rec0,
        "B": B, "relocated": int(fix.moved.sum()), "nearest_unmoved": fix.nearest_unmoved,
        "loss": {"E32": e32["loss"], "Ebf": ebf["loss"]},
        "drift_loss": rel(ebf["loss"], e32["loss"]),
        "loss_diff_signed": ebf["loss"] - e32["loss"],
        "terms_diff": {k: ebf["terms"].get(k, float("nan")) - v for k, v in e32["terms"].items()},
        "terms_E32": e32["terms"],
        "pg_rowsum_diff": float(drow.sum()),
        "pg_abs_rowsum": float(drow.abs().sum()),
        "max_abs_dlogp": float(dlp.abs().max()),
        "rho_absA": {"max": float((rho * e32["adv"].abs()).max()), "p90": float(torch.quantile(rho * e32["adv"].abs(), 0.9)),
                     "rho_max": float(rho.max())},
        "top_rows": [{"row": int(i), "A": float(e32["adv"][i]), "rho": float(rho[i]), "dlogp": float(dlp[i]),
                      "dterm": float(drow[i])} for i in top],
        "drift_grad": rel_l2(ebf["grad"], e32["grad"]),
    }
    pp = per_param(ebf["grad"], e32["grad"], names, sizes)
    rec["param_err_top"] = sorted(pp.items(), key=lambda kv: -kv[1])[:8]
    rec["param_err_watch"] = {k: v for k, v in pp.items() if "move_belief.move_head" in k or "pointer_head.move_proj" in k}
    # logit / activation scale
    lg = e32["logp"]
    if lg is not None:
        mask = torch.as_tensor(args[3]).bool().cpu()
        lgm = torch.where(mask, lg, torch.full_like(lg, float("nan")))
        rng = torch.nan_to_num(lgm, nan=-1e9).max(-1).values - torch.nan_to_num(lgm, nan=1e9).min(-1).values
        rec["logp_range"] = {"p50": float(rng.median()), "max": float(rng.max())}
        p = torch.where(mask, lg.exp(), torch.zeros_like(lg))
        rec["maxp_p50"] = float(p.max(-1).values.median())
    if attrib:
        att = {}
        for n, _ in rounds:
            with only_bf16(model, {n}):
                a1 = run(model, args, fp32=False, names=names)
            att[n] = {"drift_loss": rel(a1["loss"], e32["loss"]), "loss_diff": a1["loss"] - e32["loss"],
                      "max_abs_dlogp": float((a1["log_prob"] - e32["log_prob"]).abs().max()),
                      "drift_grad": rel_l2(a1["grad"], e32["grad"]),
                      "terms_diff": {k: a1["terms"].get(k, float("nan")) - v for k, v in e32["terms"].items()}}
        rec["attrib_only_round_bf16"] = att
        # activation scale at each round's output (fp32 forward)
        acts: Dict[str, Dict[str, float]] = {}
        hooks = []
        for n, m in rounds:
            def h(mod, inp, out, n=n):
                o = out[0] if isinstance(out, tuple) else out
                x = inp[0]
                acts[n] = {"out_absmax": float(o.detach().abs().max()), "out_rms": float(o.detach().float().pow(2).mean().sqrt()),
                           "in_absmax": float(x.detach().abs().max())}
            hooks.append(m.register_forward_hook(h))
        try:
            from agents.model.trunk_precision import fp32_scope
            with torch.no_grad(), fp32_scope(model.policy):
                from agents.training.instrumented_ppo.micro_step import micro_step
                micro_step(*args)
        finally:
            for hk in hooks:
                hk.remove()
        rec["round_acts"] = acts
    if shifts:
        sw = {}
        for sm in shifts:
            fx = cr.clip_relocation(ref0["log_prob"].float(), call_args[4], ref0["adv"].float(), clip, cr.R1_BAR_K * sm)
            a3 = call_args[:4] + (fx.old_log_prob.to(call_args[4].device),) + call_args[5:]
            y32 = run(model, a3, fp32=True, names=names)
            ybf = run(model, a3, fp32=False, names=names)
            o3 = a3[4]
            rr = (y32["log_prob"] - o3.double().cpu()).exp()
            dr = (pg_rows(ybf, o3, clip) - pg_rows(y32, o3, clip)) / B
            sw[str(sm)] = {"margin": cr.R1_BAR_K * sm, "relocated": int(fx.moved.sum()),
                           "drift_loss": rel(ybf["loss"], y32["loss"]), "loss_diff": ybf["loss"] - y32["loss"],
                           "pg_diff": float(dr.sum()), "drift_grad": rel_l2(ybf["grad"], y32["grad"]),
                           "max_abs_dlogp": float((ybf["log_prob"] - y32["log_prob"]).abs().max()),
                           "rho_absA_max": float((rr * y32["adv"].abs()).max()), "rho_max": float(rr.max())}
        rec["shift_sweep"] = sw
    if dither:
        ds = []
        for s in range(dither):
            with D.dithered([model], 1000 + s):
                r0 = run(model, call_args, fp32=True, names=names)
                fx = cr.clip_relocation(r0["log_prob"].float(), call_args[4], r0["adv"].float(), clip, cr.R1_CLIP_MARGIN)
                a2 = call_args[:4] + (fx.old_log_prob.to(call_args[4].device),) + call_args[5:]
                x32 = run(model, a2, fp32=True, names=names)
                xbf = run(model, a2, fp32=False, names=names)
                r32w = run(model, call_args, fp32=True, names=names)
                rbfw = run(model, call_args, fp32=False, names=names)
                ds.append({"raw_drift_loss": rel(rbfw["loss"], r32w["loss"]), "drift_loss": rel(xbf["loss"], x32["loss"]), "drift_grad": rel_l2(xbf["grad"], x32["grad"]),
                           "max_abs_dlogp": float((xbf["log_prob"] - x32["log_prob"]).abs().max()),
                           "terms_diff": {k: xbf["terms"].get(k, float("nan")) - v for k, v in x32["terms"].items()}})
        rec["null"] = ds
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="rb_x26_s2601")
    ap.add_argument("--ckpts", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--attrib", action="store_true")
    ap.add_argument("--dither", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shifts", default="")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    torch.set_num_threads(1)
    run_dir = D.resolve_run(a.run)
    D.select_buffer(run_dir, "golden")
    D.RUN_BATCH = a.batch
    specs = [s for s in a.ckpts.split(",") if s]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    resolved = [D.resolve_checkpoint(run_dir, s) for s in specs]
    t0 = time.perf_counter()
    model = D.build_learner(resolved[0][1], "cpu", batch=a.batch)
    names, sizes = D.param_table(model)
    rounds = round_modules(model)
    print(f"[drift] built in {time.perf_counter() - t0:.1f}s; rounds: {[n for n, _ in rounds]}", flush=True)
    for label, z in resolved:
        f = out / f"b{a.batch}{a.tag}_{label}.json"
        if f.exists():
            print(f"[drift] {label}: exists, skip", flush=True)
            continue
        t1 = time.perf_counter()
        D.load_weights([model], z)
        from agents.model import compile_regions as cr
        b = cr.r1_batch(model, a.batch, 0)
        call_args = cr._r1_args(model, b)
        rec = read(model, names, sizes, call_args, rounds, attrib=a.attrib, dither=a.dither,
                   shifts=[float(x) for x in a.shifts.split(",") if x])
        rec["zip"] = str(z)
        rec["seconds"] = time.perf_counter() - t1
        f.write_text(json.dumps(rec, indent=1, default=float))
        nd = [d["drift_loss"] for d in rec.get("null", [])]
        print(f"[drift] {label}: drift_loss {rec['drift_loss']:.2e} (signed {rec['loss_diff_signed']:+.2e}) grad "
              f"{rec['drift_grad']:.3f} max|dlogp| {rec['max_abs_dlogp']:.3f} pg_rowsum {rec['pg_rowsum_diff']:+.2e} "
              f"reloc {rec['relocated']} S {rec['S']:.3f} s_pg {rec['s_pg']:.2e} rest {rec['rest_drift']:.1e} raw {rec['raw']['drift_loss']:.1e} null {['%.1e' % x for x in nd]} ({rec['seconds']:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
