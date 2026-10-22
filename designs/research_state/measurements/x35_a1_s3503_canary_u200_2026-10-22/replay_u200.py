"""X35 A1 (rb_x35_a1arch_s3503) canary u200 UNCONFIRMED: replay the bf16 gradient rule on the canary's OWN weight dump
(`canary_u200_unconfirmed_policy.pt`) on CPU, at the run's PIN (f4b25238). Run with PYTHONPATH=<pin>/src, cwd=<pin>:

    CUDA_VISIBLE_DEVICES="" PYTHONPATH=<pin>/src python replay_u200.py --zip <ckpt.zip> --pt <dump.pt> --out r.json \
        [--dither 8] [--compiled]

Per slice (0 = all 64 rows, the scheduled check; 2 = rows 0-31; 1 = rows 32-63), on the rows relocated off the PPO clip
boundary exactly as `_r1_judge` does (`compile_regions.relocate_clip_boundary`, B = 2048):
  * arms E32 (eager fp32, the reference), Ebf (eager bf16), and with --compiled Cbf (production `compile_regions.install`
    on CPU Inductor, bf16) and C32 (the same install on an fp32 learner);
  * THE CANARY STATISTIC (`diag_bf16_canary.canary_stat` / `canary_verdict`, the rule itself) per arm vs E32, and for
    every `edge_bias.*` parameter: ||g_ref||, ||g_ref|| / the largest per-parameter norm (the judged floor is 1e-3),
    ||g_arm - g_ref|| absolute and relative;
  * PER-ROW ATTRIBUTION of the s1_map / c2_map weight gradient (eager arms; forward hook + output grad hook):
    dW = sum_rows g_out^T x, split by FIXTURE row; which rows carry a nonzero cell, their share, Ebf's per-row error;
  * a half-bf16-ulp weight-dither null (relative 2^-9, `diag_bf16_canary.dithered`): E32~ vs E32 and Ebf~ vs E32 on
    the edge_bias parameters — how far the fp32 / eager-bf16 gradient of these parameters moves when only the rounding
    of the weights changes.
Nothing here trains or writes under models/.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

FAMS = ("s1_map", "c2_map")
EB = "features_extractor.edge_bias."


def eb_table(arm: Any, ref: Any, names: List[str], sizes: List[int]) -> Dict[str, Any]:
    from main.ops import diag_bf16_canary as D
    sa, sr = D.split(arm.grad, names, sizes), D.split(ref.grad, names, sizes)
    top = max(float(g.double().norm()) for g in sr.values())
    out = {}
    for n in names:
        if not n.startswith(EB):
            continue
        r = sr[n].double()
        nr = float(r.norm())
        e = float((sa[n].double() - r).norm())
        out[n[len(EB):]] = {"ref_norm": nr, "ref_over_top": nr / top if top else None,
                            "abs_err": e, "rel_err": e / nr if nr > 0 else None,
                            "cos": float(torch.nn.functional.cosine_similarity(sa[n].double(), r, dim=0))
                            if nr > 0 else None}
    return {"top_norm": top, "params": out}


class Catch:
    """Forward hooks on the s1 / c2 maps: per call, the input cells and the output's gradient (eager arms only)."""

    def __init__(self, fe: Any) -> None:
        self.rec: Dict[str, List[Dict[str, torch.Tensor]]] = {f: [] for f in FAMS}
        self.hs = []
        for f in FAMS:
            m = getattr(fe.edge_bias, f)
            self.hs.append(m.register_forward_hook(self._hook(f)))

    def _hook(self, f: str):
        def h(_m: Any, inp: Any, out: torch.Tensor) -> None:
            if not out.requires_grad:
                return
            slot: Dict[str, torch.Tensor] = {"x": inp[0].detach().double().cpu().clone()}
            self.rec[f].append(slot)

            def g(grad: torch.Tensor) -> None:
                slot["g"] = grad.detach().double().cpu().clone()
            out.register_hook(g)
        return h

    def clear(self) -> None:
        for f in FAMS:
            self.rec[f] = []

    def remove(self) -> None:
        for h in self.hs:
            h.remove()

    def per_batch_row(self, f: str) -> torch.Tensor:
        """[B, out, in] per batch-row weight gradient, summed over every call of the forward."""
        tot = None
        for s in self.rec[f]:
            if "g" not in s:
                continue
            x, g = s["x"], s["g"]
            B = x.shape[0]
            dw = torch.einsum("bko,bki->boi", g.reshape(B, -1, g.shape[-1]), x.reshape(B, -1, x.shape[-1]))
            tot = dw if tot is None else tot + dw
        assert tot is not None, f"no gradient captured for {f}"
        return tot

    def nonzero_rows(self, f: str) -> torch.Tensor:
        s = self.rec[f][0]
        B = s["x"].shape[0]
        return s["x"].reshape(B, -1).abs().sum(1) > 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", required=True)
    ap.add_argument("--pt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dither", type=int, default=8)
    ap.add_argument("--compiled", action="store_true")
    ap.add_argument("--slices", default="0,2,1")
    a = ap.parse_args()
    torch.set_num_threads(6)
    from main.ops import diag_bf16_canary as D
    from agents.model import compile_regions as cr
    from agents.model.compile_trainer import fixture_index
    from agents.training.instrumented_ppo.micro_step import micro_step
    t0 = time.perf_counter()
    sd = torch.load(a.pt, map_location="cpu")

    def build(prec: str) -> Any:
        m = D.build_learner(Path(a.zip), "cpu", precision=prec, batch=2048)
        m.policy.load_state_dict(sd, strict=True)
        m.policy.set_training_mode(True)
        return m
    model = build("bf16")
    names, sizes = D.param_table(model)
    print(f"[u200] learner built + dump loaded strict {time.perf_counter() - t0:.0f}s", flush=True)
    model32 = None
    if a.compiled:
        from agents.model.compile_cache import ensure_hermetic_cache
        from agents.model import compile_control as cc
        ensure_hermetic_cache("x35_canary_u200")
        cc.control().install()
        torch._dynamo.config.cache_size_limit = max(int(torch._dynamo.config.cache_size_limit), 64)
        cr.install(model)
        model32 = build("fp32")
        cr.install(model32)
    n_fix = D.n_fixture_rows()
    res: Dict[str, Any] = {"zip": a.zip, "pt": a.pt, "batch": 2048, "n_fixture": n_fix, "slices": {}}
    catch = Catch(model.policy.features_extractor)
    for sl in [int(s) for s in a.slices.split(",")]:
        b = cr.r1_batch(model, 2048, sl)
        raw = cr._r1_args(model, b)
        args, fix = cr.relocate_clip_boundary(model, raw)
        idx = torch.as_tensor(fixture_index(2048, n_fix, sl))
        rec: Dict[str, Any] = {"relocated_rows": int(fix.moved.sum()) if hasattr(fix, "moved") else None}
        try:
            rec["relocated_rows"] = int(fix[1].sum())
        except Exception:
            pass
        arms: Dict[str, Any] = {}
        per_row: Dict[str, Dict[str, torch.Tensor]] = {}
        for arm, fp32 in (("E32", True), ("Ebf", False)):
            catch.clear()
            arms[arm] = D.run_arm(model, micro_step, args, fp32_trunk=fp32)
            per_row[arm] = {f: catch.per_batch_row(f) for f in FAMS}
            nz = {f: catch.nonzero_rows(f) for f in FAMS}
        if a.compiled:
            catch.remove()
            t1 = time.perf_counter()
            arms["Cbf"] = D.run_arm(model, model._compiled_micro_step, args, fp32_trunk=False)
            a32 = cr.relocate_clip_boundary(model32, cr._r1_args(model32, cr.r1_batch(model32, 2048, sl)))[0]
            e32b = D.run_arm(model32, micro_step, a32, fp32_trunk=False)
            arms["C32"] = D.run_arm(model32, model32._compiled_micro_step, a32, fp32_trunk=False)
            rec["C32_ref_E32_on_fp32_learner"] = {"vs_E32_same_learner": eb_table(arms["C32"], e32b, names, sizes),
                                                  "global_rel_l2": D.rel_l2(arms["C32"].grad, e32b.grad)}
            print(f"[u200] slice {sl}: compiled arms {time.perf_counter() - t1:.0f}s", flush=True)
            catch = Catch(model.policy.features_extractor)
        ref = arms["E32"]
        for arm in [k for k in arms if k != "E32"]:
            if arm == "C32":
                continue
            rec[arm] = {"canary_stat": D.canary_stat(arms[arm], arms["Ebf"], ref, names, sizes, "trained")
                        if arm != "Ebf" else None,
                        "canary_verdict": D.canary_verdict(arms[arm], arms["Ebf"], ref, names, sizes, "trained")
                        if arm != "Ebf" else None,
                        "edge_bias": eb_table(arms[arm], ref, names, sizes),
                        "global_rel_l2": D.rel_l2(arms[arm].grad, ref.grad),
                        "max_abs_dlogp_vs_E32": float((arms[arm].log_prob - ref.log_prob).abs().max())}
        # per-row attribution (eager arms)
        att: Dict[str, Any] = {}
        for f in FAMS:
            g32, gbf = per_row["E32"][f], per_row["Ebf"][f]
            tot = g32.sum(0)
            rows = sorted(set(idx.tolist()))
            pr = {u: g32[idx == u].sum(0) for u in rows}
            pe = {u: (gbf[idx == u] - g32[idx == u]).sum(0) for u in rows}
            norms = {u: float(pr[u].norm()) for u in rows}
            s = sum(norms.values())
            nzr = sorted(set(idx[nz[f]].tolist()))
            top = sorted(rows, key=lambda u: -norms[u])[:6]
            att[f] = {"sum_norm": float(tot.norm()), "kappa": s / float(tot.norm()) if float(tot.norm()) else None,
                      "fixture_rows_with_nonzero_cell": nzr,
                      "top_rows": [{"row": u, "share_of_abs": norms[u] / s if s else None,
                                    "rel_to_total": norms[u] / float(tot.norm()) if float(tot.norm()) else None,
                                    "Ebf_err_rel_to_total": float(pe[u].norm()) / float(tot.norm())
                                    if float(tot.norm()) else None,
                                    "adv": float(arms["E32"].adv[idx == u][0]),
                                    "relocated": bool(fix[1][(idx == u).nonzero()[0, 0]]) if True else None}
                                   for u in top]}
            if "Cbf" in arms:
                gc = D.split(arms["Cbf"].grad, names, sizes)[EB + f + ".weight"].double().reshape(tot.shape)
                err = gc - tot
                # does the compiled error lie along one row's own contribution? (cosine with each top row)
                att[f]["Cbf_err_norm_rel"] = float(err.norm() / tot.norm())
                att[f]["Cbf_err_cos_with_row"] = {
                    str(u): float(torch.nn.functional.cosine_similarity(err.flatten(), pr[u].flatten(), dim=0))
                    for u in top}
        rec["attribution"] = att
        res["slices"][str(sl)] = rec
        print(f"[u200] slice {sl}: " + json.dumps({k: (v["edge_bias"]["params"].get("c2_map.weight"),
                                                       v["edge_bias"]["params"].get("s1_map.weight"))
                                                   for k, v in rec.items() if isinstance(v, dict) and "edge_bias" in v},
                                                  default=str)[:1500], flush=True)
        # the dither null
        if a.dither:
            nul: Dict[str, List[Dict[str, Any]]] = {"E32~": [], "Ebf~": []}
            for s_ in range(a.dither):
                with D.dithered([model], s_ + 1):
                    for arm, fp32 in (("E32~", True), ("Ebf~", False)):
                        x = D.run_arm(model, micro_step, args, fp32_trunk=fp32)
                        t = eb_table(x, ref, names, sizes)["params"]
                        nul[arm].append({k: t[k]["rel_err"] for k in t})
            summ = {}
            for arm, L in nul.items():
                summ[arm] = {k: {"max": max(d[k] for d in L if d[k] is not None),
                                 "median": sorted(d[k] for d in L if d[k] is not None)[len(L) // 2]}
                             for k in L[0] if L[0][k] is not None}
            rec["dither_null"] = {"n": a.dither, "rel": D.DITHER_REL, "edge_bias_rel_err": summ}
            print(f"[u200] slice {sl}: dither null c2_map.weight {summ['E32~'].get('c2_map.weight')} / "
                  f"{summ['Ebf~'].get('c2_map.weight')}", flush=True)
        Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(f"[u200] wrote {a.out} in {time.perf_counter() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
