"""Derive the K9(b)-under-bf16 numbers from the measured cells (`measure.py`) → ``result.json``.

DECLARED before the numbers were read (the k9_behaviour_exclusion / k9_behaviour_bar rules, unchanged):

* R_bf16 = the largest |m_learner - m_variant| over every row x every declared MARGIN site x every bf16
  T2-like variant x every cell (the learner: the bf16-trunk probe forward); eps_bf16 = round_up_125(10 x R_bf16).
* The excluded share at eps (before the flip-judge: margin < eps; after: and not flippable at eps); the
  ceiling 0.15 must hold at >= 3 x the pooled share and >= 1.5 x the largest 2,048-row block.
* Design (a) HEALTHY = |log pi_bf16-learner - log mu_bf16-T2| over the JUDGED rows (margin >= eps; a
  flip-judged row counted at its as-is value, the conservative side), plus the bf16 learner vs every bf16
  T2-like variant (T2's compiled form is not on the CPU: the variants stand in for it). H = the largest.
  The bar candidates B_k = round_up_125(k x H), k = 4 and 10 (fp32's bar sat at 52x its healthy max).
* TEETH = the smallest 2,048-row micro-batch max |d| under the one-optimizer-step-stale fault (the probe's
  OWN margins; judged rows only — strictly, flip-judged rows left OUT, the conservative side), and the
  wrong-action row's |d| (the sweep's choice: the first other legal action; and the minimum over every
  other legal action). The question: S / B_k >= 10?
* Design (b) = |log pi_bf16-learner - log mu_fp32-T2| over ALL rows (no exclusion: PPO reads every row),
  the clip fraction |exp(d) - 1| > 0.15 and sb3's approx-KL, beside today's fp32 learner on the same rows.
* The reference sanity: on the fp64 rows judged at fp32 eps 2e-4, max / median |lp_fp32 - lp_fp64| vs
  |lp_bf16 - lp_fp32|; the separation must be >= 100x.

    python derive.py --cells ~/gen3ai_archive/k9_bf16_2026-10-10/cells --out result.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import BAR, CEILING, CLIP, STALE_LRS, dist, round_up_125  # noqa: E402

BLOCK = 2048
FP32_EPS = 2e-4
DISCRETE_FP64 = 1e-5
CEILINGS = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30)


def load(cells: str) -> List[Dict[str, Any]]:
    out = []
    for j in sorted(glob.glob(os.path.join(cells, "*.json"))):
        z = np.load(j[:-5] + ".npz")
        meta = json.load(open(j))
        out.append({"meta": meta, "a": {k: z[k] for k in z.files}, "name": os.path.basename(j)[:-5]})
    if not out:
        raise SystemExit(f"no cells under {cells}")
    return out


def blocks_max(x: np.ndarray, keep: np.ndarray) -> List[float]:
    r = []
    for s in range(0, x.size, BLOCK):
        k = keep[s:s + BLOCK]
        r.append(float(x[s:s + BLOCK][k].max()) if k.any() else float("nan"))
    return r


def blocks_frac(ex: np.ndarray) -> List[float]:
    return [float(ex[s:s + BLOCK].mean()) for s in range(0, ex.size, BLOCK)]


def eps_index(meta: Dict[str, Any], eps: float) -> int:
    g = [float(e) for e in meta["eps_grid"]]
    for i, e in enumerate(g):
        if abs(e - eps) <= 1e-12 * max(1.0, eps):
            return i
    raise SystemExit(f"eps {eps:g} is not on the measured grid {g}: re-run measure.py with it")


def wrong_action(full: np.ndarray, acts: np.ndarray, masks: np.ndarray, mu: np.ndarray, keep: np.ndarray):
    first = np.full(mu.size, np.nan)
    mn = np.full(mu.size, np.nan)
    for i in np.flatnonzero(keep):
        legal = np.flatnonzero(masks[i])
        alt = legal[legal != acts[i]]
        if alt.size == 0:
            continue
        d = np.abs(full[i, alt] - mu[i])
        first[i] = d[0]
        mn[i] = d.min()
    return first, mn


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cells = load(a.cells)
    res: Dict[str, Any] = {"what": "K9(b) under the bf16 trunk region (T25 lever A): the re-derivation's numbers",
                           "device": sorted({c["meta"]["device"] for c in cells}),
                           "torch": sorted({c["meta"]["torch"] for c in cells}),
                           "commit": sorted({c["meta"]["commit"] for c in cells}),
                           "cells": {c["name"]: {k: c["meta"][k] for k in ("state", "behaviour", "rows", "selection_free",
                                                                             "weights_from", "mu_sha256", "wall_s")}
                                     for c in cells}}
    # the POOLED numbers are the target's (--arch endstate); a cell at another arch is reported on its own
    other = [c for c in cells if c["meta"].get("arch", "endstate") != "endstate"]
    cells = [c for c in cells if c["meta"].get("arch", "endstate") == "endstate"]
    informative = [c for c in cells if not c["meta"]["selection_free"]]

    # ---------------------------------------------------------------- 1. the rounding scale of the margins
    rs: Dict[str, Any] = {"by_variant": {}, "by_site": {}}
    R16 = R32 = 0.0
    for c in cells:
        for name, v in c["meta"]["variants"].items():
            prec = name.split("_", 1)[0]
            key = f"{name}"
            rs["by_variant"][key] = max(rs["by_variant"].get(key, 0.0), float(v["max_diff"]))
            for s, ps in v["per_site"].items():
                if "max_diff" in ps:
                    k2 = f"{prec} {s}"
                    rs["by_site"][k2] = max(rs["by_site"].get(k2, 0.0), float(ps["max_diff"]))
            if prec == "bf16":
                R16 = max(R16, float(v["max_diff"]))
            else:
                R32 = max(R32, float(v["max_diff"]))
    same_margin = []
    for c in cells:
        A = c["a"]
        m16, m32 = A["m_Lbf16"], A["m_Lfp32"]
        same_margin.append(float(np.mean((m16 == m32) | (np.isinf(m16) & np.isinf(m32)))))
    rs["R_bf16"] = R16
    rs["R_fp32_this_device"] = R32
    rs["R_fp32_banked_cuda"] = 1.4387517728961519e-05
    rs["learner_margin_bf16_equals_fp32_row_frac_min"] = min(same_margin)
    eps16 = round_up_125(10 * R16)
    rs["eps_bf16"] = eps16
    rs["eps_fp32_this_device"] = round_up_125(10 * R32)
    res["rounding_scale"] = rs

    # ---- the excluded share at eps_bf16 and at today's 2e-4
    excl: Dict[str, Any] = {}
    for eps in sorted({eps16, FP32_EPS}):
        per_cell = {}
        pooled_b = pooled_a = n_all = 0
        blocks: List[float] = []
        for c in informative:
            A = c["a"]
            i = eps_index(c["meta"], eps)
            ex = ~(A["m_Lbf16"] >= eps)
            ex_after = ex & ~A["flip_Lbf16"][i]
            pooled_b += int(ex.sum())
            pooled_a += int(ex_after.sum())
            n_all += ex.size
            blocks += blocks_frac(ex_after)
            per_cell[c["name"]] = {"before_flip": float(ex.mean()), "after_flip": float(ex_after.mean()),
                                   "fp32_learner_before_flip_at_2e-4": float((~(A["m_Lfp32"] >= FP32_EPS)).mean()),
                                   "fp32_learner_after_flip_at_2e-4": float((~(A["m_Lfp32"] >= FP32_EPS) & ~A["flip_Lfp32"][0]).mean())}
        pooled = pooled_a / max(1, n_all)
        ceil = next((x for x in CEILINGS if x >= 3 * pooled and x >= 1.5 * max(blocks)), None)
        excl[f"{eps:g}"] = {"per_cell": per_cell, "pooled_before_flip": pooled_b / max(1, n_all),
                            "pooled_after_flip": pooled, "largest_block_after_flip": max(blocks),
                            "ceiling_rule_gives": ceil, "under_0.15": bool(max(blocks) < CEILING)}
    res["excluded"] = excl

    # ---------------------------------------------------------------- 2. design (a): healthy and teeth
    da: Dict[str, Any] = {"eps": eps16, "per_cell": {}}
    H_mu = H_var = 0.0
    all_judged_d: List[np.ndarray] = []
    healthy_blocks: List[float] = []
    stale: Dict[str, List[float]] = {f"{lr:g}": [] for lr in STALE_LRS}
    stale32: Dict[str, List[float]] = {f"{lr:g}": [] for lr in STALE_LRS}
    wa_first: List[np.ndarray] = []
    wa_min: List[np.ndarray] = []
    for c in [c for c in informative if c["meta"]["behaviour"] == "bf16"]:
        A = c["a"]
        i = eps_index(c["meta"], eps16)
        strict = A["m_Lbf16"] >= eps16
        judged = strict | A["flip_Lbf16"][i]
        d = np.abs(A["lp_Lbf16"] - A["mu"])
        all_judged_d.append(d[judged])
        healthy_blocks += blocks_max(d, judged)
        H_mu = max(H_mu, float(d[judged].max()))
        var = {}
        for k in A.files if hasattr(A, "files") else A.keys():
            if k.startswith("lp_bf16_"):
                lv = A[k]
                ok = judged & np.isfinite(lv)
                dv = np.abs(lv - A["lp_Lbf16"])[ok]
                dm = np.abs(lv - A["mu"])[ok]
                var[k[3:]] = {"vs_learner": dist(dv), "vs_mu": dist(dm)}
                H_var = max(H_var, float(dv.max()))
        cell = {"judged_rows": int(judged.sum()), "flip_judged_rows": int((judged & ~strict).sum()),
                "healthy_vs_mu": dist(d[judged]), "healthy_vs_mu_flip_judged_rows": dist(d[judged & ~strict]),
                "healthy_block_max": blocks_max(d, judged), "variants": var,
                "fp32_learner_vs_mu_bf16": dist(np.abs(A["lp_Lfp32"] - A["mu"])[judged])}
        for lr in STALE_LRS:
            k = f"{lr:g}"
            ms = A[f"m_stale_{k}"]
            ds = np.abs(A[f"lp_stale_{k}"] - A["mu"])
            b = blocks_max(ds, ms >= eps16)
            stale[k] += b
            cell[f"stale_{k}"] = {"block_max": b, "rows": dist(ds[ms >= eps16]),
                                  "excluded_share": float((~(ms >= eps16)).mean()),
                                  # the SAME state's fault vs the SAME state's healthy (a per-state bar's best case)
                                  "smallest_block_over_this_cell_healthy_max": min(b) / max(blocks_max(d, judged))}
        f1, fm = wrong_action(A["full_Lbf16"], A["acts"], A["masks"], A["mu"], strict)
        wa_first.append(f1[np.isfinite(f1)])
        wa_min.append(fm[np.isfinite(fm)])
        da["per_cell"][c["name"]] = cell
    # today's fp32 stale signal on the fp32 cells (behaviour fp32, learner fp32), for the ratio
    for c in [c for c in informative if c["meta"]["behaviour"] == "fp32"]:
        A = c["a"]
        for lr in STALE_LRS:
            k = f"{lr:g}"
            ms = A[f"m32_stale_{k}"]
            stale32[k] += blocks_max(np.abs(A[f"lp32_stale_{k}"] - A["mu"]), ms >= FP32_EPS)
    H = max(H_mu, H_var)
    h32 = []
    for c in [c for c in informative if c["meta"]["behaviour"] == "fp32"]:
        A = c["a"]
        keep = (A["m_Lfp32"] >= FP32_EPS) | A["flip_Lfp32"][0]
        h32.append(float(np.abs(A["lp_Lfp32"] - A["mu"])[keep].max()))
    H32 = max(h32) if h32 else float("nan")
    da["fp32_today_same_harness"] = {
        "healthy_max": H32, "bar": BAR, "headroom": BAR / H32 if h32 else None,
        **{f"stale_{lr}_smallest_block": (min(b) if b else None) for lr, b in stale32.items()},
        **{f"stale_{lr}_over_bar": (min(b) / BAR if b else None) for lr, b in stale32.items()},
        **{f"stale_{lr}_over_healthy": (min(b) / H32 if b and h32 else None) for lr, b in stale32.items()}}
    da["best_case_bar_just_above_H"] = {"bar": round_up_125(H),
                                        **{f"stale_{lr}_over_bar": (min(b) / round_up_125(H) if b else None)
                                           for lr, b in stale.items()}}
    da["healthy"] = {"H_vs_mu": H_mu, "H_vs_T2_like_variants": H_var, "H": H,
                     "rows": dist(np.concatenate(all_judged_d)) if all_judged_d else {},
                     "largest_block": max(healthy_blocks) if healthy_blocks else None,
                     "blocks": len(healthy_blocks)}
    bars = {}
    for kmul in (4, 10):
        B = round_up_125(kmul * H)
        row = {"bar": B}
        for lr, b in stale.items():
            S = min(b) if b else float("nan")
            row[f"stale_{lr}_smallest_block"] = S
            row[f"stale_{lr}_over_bar"] = S / B if B else None
            row[f"stale_{lr}_caught_all_blocks"] = bool(b) and all(x >= B for x in b)
        wf = np.concatenate(wa_first) if wa_first else np.zeros(0)
        wm = np.concatenate(wa_min) if wa_min else np.zeros(0)
        row["wrong_action_first_alt_caught_frac"] = float((wf >= B).mean()) if wf.size else None
        row["wrong_action_min_alt_caught_frac"] = float((wm >= B).mean()) if wm.size else None
        bars[f"k{kmul}"] = row
    da["bars"] = bars
    da["stale_blocks"] = stale
    da["stale_fp32_today_blocks"] = stale32
    da["stale_over_H"] = {lr: (min(b) / H if b and H else None) for lr, b in stale.items()}
    da["wrong_action"] = {"first_alt": dist(np.concatenate(wa_first)) if wa_first else {},
                          "min_alt": dist(np.concatenate(wa_min)) if wa_min else {},
                          "min_alt_quantiles": ({f"p{q:g}": float(np.quantile(np.concatenate(wa_min), q / 100))
                                                 for q in (0.1, 1, 5, 10, 50)} if wa_min else {})}
    res["design_a"] = da

    # ---------------------------------------------------------------- 3. design (b): the epoch-0 ratio noise
    db: Dict[str, Any] = {"per_cell": {}}
    pooled16: List[np.ndarray] = []
    pooled32: List[np.ndarray] = []
    for c in [c for c in informative if c["meta"]["behaviour"] == "fp32"]:
        A = c["a"]
        d16 = A["lp_Lbf16"] - A["mu"]
        d32 = A["lp_Lfp32"] - A["mu"]
        pooled16.append(d16)
        pooled32.append(d32)

        def ratio_stats(d: np.ndarray) -> Dict[str, Any]:
            r = np.exp(d)
            return {"abs_dlogp": dist(np.abs(d)), "clip_frac": float((np.abs(r - 1) > CLIP).mean()),
                    "mean_abs_r_minus_1": float(np.abs(r - 1).mean()),
                    "approx_kl": float(((r - 1) - d).mean()), "max_abs_r_minus_1": float(np.abs(r - 1).max())}
        db["per_cell"][c["name"]] = {"bf16_learner": ratio_stats(d16), "fp32_learner_today": ratio_stats(d32),
                                     "fp32_gate_today": {"judged_max": float(np.abs(d32)[A["m_Lfp32"] >= FP32_EPS].max()),
                                                         "excluded": float((~(A["m_Lfp32"] >= FP32_EPS)).mean())}}
    if pooled16:
        d16, d32 = np.concatenate(pooled16), np.concatenate(pooled32)
        db["pooled"] = {"bf16_learner_abs_dlogp": dist(np.abs(d16)), "fp32_learner_abs_dlogp": dist(np.abs(d32)),
                        "bf16_clip_frac": float((np.abs(np.exp(d16) - 1) > CLIP).mean()),
                        "rows_to_reach_clip": "|d log pi| >= log(1.15) = 0.1398 for the ratio to leave the band",
                        "bf16_max_over_clip_edge": float(np.abs(d16).max() / np.log(1 + CLIP))}
    res["design_b"] = db

    # ---------------------------------------------------------------- 4. the reference sanity (fp64)
    ref: Dict[str, Any] = {}
    e32: List[np.ndarray] = []
    e16: List[np.ndarray] = []
    for c in informative:
        A = c["a"]
        ok = np.isfinite(A["lp_fp64"]) & (A["m_Lfp32"] >= FP32_EPS)
        e32.append(np.abs(A["lp_Lfp32"] - A["lp_fp64"])[ok])
        e16.append(np.abs(A["lp_Lbf16"] - A["lp_Lfp32"])[ok])
    if e32:
        x32, x16 = np.concatenate(e32), np.concatenate(e16)
        disc = x32 >= DISCRETE_FP64
        ok = ~disc
        ratio = x16[ok] / np.maximum(x32[ok], 1e-12)
        ref = {"rows": int(x32.size), "fp32_vs_fp64": dist(x32), "bf16_vs_fp32": dist(x16),
               "discrete_divergence_rule": f"|lp_fp32 - lp_fp64| >= {DISCRETE_FP64:g} (10x fp32's healthy per-row "
                                           "max ~1e-6): fp64 resolved a discrete op differently — counted, not judged",
               "discrete_divergence_rows": int(disc.sum()),
               "discrete_divergence_values": sorted(float(v) for v in x32[disc])[-10:],
               "fp32_vs_fp64_continuous": dist(x32[ok]), "bf16_vs_fp32_on_the_same_rows": dist(x16[ok]),
               "separation_max": float(x16[ok].max() / max(x32[ok].max(), 1e-300)),
               "separation_median": float(np.median(x16[ok]) / max(np.median(x32[ok]), 1e-300)),
               "per_row_ratio_bf16_over_fp32err": {"p1": float(np.quantile(ratio, 0.01)),
                                                   "p50": float(np.median(ratio))},
               "rows_where_bf16_err_is_not_100x_fp32_err": int((x16[ok] < 100 * x32[ok]).sum()),
               "fp32_is_a_valid_reference": bool(x16[ok].max() / max(x32[ok].max(), 1e-300) >= 100)}
    res["reference_fp64"] = ref
    res["other_arch"] = {}
    for c in other:
        A, m = c["a"], c["meta"]
        d16, d32 = A["lp_Lbf16"] - A["mu"], A["lp_Lfp32"] - A["mu"]
        o: Dict[str, Any] = {"arch": m["arch"], "behaviour": m["behaviour"], "weights_from": m["weights_from"],
                             "selection_free": m["selection_free"],
                             "bf16_learner_vs_mu": dist(np.abs(d16)), "fp32_learner_vs_mu": dist(np.abs(d32)),
                             "bf16_clip_frac": float((np.abs(np.exp(d16) - 1) > CLIP).mean()),
                             "bf16_max_abs_r_minus_1": float(np.abs(np.exp(d16) - 1).max()),
                             "bf16_approx_kl": float(((np.exp(d16) - 1) - d16).mean()),
                             "R_by_variant": {k: v["max_diff"] for k, v in m["variants"].items()},
                             "margin_bf16_equals_fp32_frac": float(np.mean((A["m_Lbf16"] == A["m_Lfp32"]) |
                                                                           (np.isinf(A["m_Lbf16"]) & np.isinf(A["m_Lfp32"])))),
                             "excluded_at_2e-4": float((~(A["m_Lbf16"] >= FP32_EPS)).mean())}
        for lr in STALE_LRS:
            k = f"{lr:g}"
            key, mk = ((f"lp_stale_{k}", f"m_stale_{k}") if m["behaviour"] == "bf16"
                       else (f"lp32_stale_{k}", f"m32_stale_{k}"))
            o[f"stale_{k}_block_max"] = blocks_max(np.abs(A[key] - A["mu"]), A[mk] >= FP32_EPS)
        j = A["m_Lbf16"] >= FP32_EPS
        o["healthy_judged_max_same_precision"] = float(np.abs((d16 if m["behaviour"] == "bf16" else d32)[j]).max())
        ok = np.isfinite(A["lp_fp64"]) & (A["m_Lfp32"] >= FP32_EPS)
        o["fp64"] = {"fp32_vs_fp64": dist(np.abs(A["lp_Lfp32"] - A["lp_fp64"])[ok]),
                     "bf16_vs_fp32": dist(np.abs(A["lp_Lbf16"] - A["lp_Lfp32"])[ok])}
        res["other_arch"][c["name"]] = o
    res["bar_fp32_today"] = BAR
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("rounding_scale",)}, indent=1, default=float)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
