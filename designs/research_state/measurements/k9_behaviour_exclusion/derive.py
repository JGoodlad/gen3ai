"""Derive K9(b)'s deterministic rule from the exclusion sweep (`sweep.py`): epsilon, the excluded-share
ceiling, and the verification, into ``result.json`` (re-derived by `consistency_test`).

DECLARED CRITERIA (written before the numbers were read; owner direction 2026-10-01):

* EPSILON = ``SAFETY`` x the MEASURED ROUNDING SCALE of the margins, rounded UP to a 1-2-5 value. The
  rounding scale R is the largest |m_learner - m_variant| (relative margin, per declared site and row)
  between the learner's probe forward and any of the T2-like forwards (eval mode without grad at T2's
  buckets 48 and 8; a few-ulp weight jitter of the learner forward), over every fill of the sweep —
  the max, not just at the rows that flipped.
* every row whose margin is >= epsilon is JUDGED: the verification requires ZERO such rows over the
  bar on every healthy row of the sweep, and reports the headroom (bar / their max |d|).
* THE CEILING on the excluded share: the smallest of ``CEILINGS`` that is >= ``CEIL_X_POOLED`` x the
  pooled healthy share AND >= ``CEIL_X_BLOCK`` x the largest share in any probe-sized block (1,024
  random current rows of one fill; ``BLOCKS_PER_FILL`` per fill).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

SAFETY = 10.0
CEILINGS = (0.02, 0.05, 0.10, 0.15, 0.20, 0.30)
CEIL_X_POOLED, CEIL_X_BLOCK = 3.0, 1.5
BLOCK, BLOCKS_PER_FILL = 1024, 200
BAR_FP32 = 1e-4


def round_up_125(x: float) -> float:
    e = math.floor(math.log10(x))
    for m in (1, 2, 5, 10):
        if m * 10 ** e >= x * (1 - 1e-12):
            return float(m * 10 ** e)
    return float(10 ** (e + 1))


def load(d: Path) -> Dict[str, Any]:
    fills = [json.loads(x) for x in open(d / "fills.jsonl")]
    arrs = []
    for f in fills:
        z = np.load(d / f"fill_{f['fill']:03d}.npz")
        arrs.append((z["absd"], z["margin"], z["site"], z["site_names"]))
    extras = [json.loads(x) for x in open(d / "extras.jsonl")] if (d / "extras.jsonl").exists() else []
    return {"fills": fills, "arrs": arrs, "extras": extras}


def rounding_scale(fills: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_var: Dict[str, float] = {}
    by_site: Dict[str, float] = {}
    for f in fills:
        for name, v in f.get("variants", {}).items():
            if name == "rows":
                continue
            by_var[name] = max(by_var.get(name, 0.0), v["max_diff"])
            for site, x in v["per_site"].items():
                if "max_diff" in x:
                    by_site[site] = max(by_site.get(site, 0.0), x["max_diff"])
    return {"R": max(by_var.values()), "by_variant": by_var, "by_site": dict(sorted(by_site.items(), key=lambda kv: -kv[1])),
            "rows_compared": int(sum(f["variants"]["rows"] for f in fills if "variants" in f))}


def verify(arrs: List[Any], eps: float, bar: float, seed: int = 0) -> Dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = ex_n = 0
    max_judged = 0.0
    over_judged = 0
    over_all = 0
    max_margin_over_bar = 0.0
    block_shares: List[float] = []
    exact = 0
    site_ex: Dict[str, int] = {}
    for absd, margin, site, names in arrs:
        ex = ~(margin >= eps)
        n += absd.size
        ex_n += int(ex.sum())
        exact += int((margin == 0).sum())
        j = absd[~ex]
        max_judged = max(max_judged, float(j.max()))
        over_judged += int((j >= bar).sum())
        over_all += int((absd >= bar).sum())
        if (absd >= bar).any():
            max_margin_over_bar = max(max_margin_over_bar, float(margin[absd >= bar].max()))
        for k, c in zip(*np.unique(names[site[ex]], return_counts=True)):
            site_ex[str(k)] = site_ex.get(str(k), 0) + int(c)
        for _ in range(BLOCKS_PER_FILL):
            idx = rng.choice(absd.size, BLOCK, replace=False)
            block_shares.append(float(ex[idx].mean()))
    bs = np.asarray(block_shares)
    return {"eps": eps, "rows": n, "excluded": ex_n, "excluded_frac": ex_n / n, "exact_tie_rows": exact,
            "excluded_by_site": dict(sorted(site_ex.items(), key=lambda kv: -kv[1])),
            "judged_max_abs_dlogp": max_judged, "judged_over_bar": over_judged, "headroom": bar / max_judged,
            "all_rows_over_bar": over_all, "largest_margin_of_a_row_over_bar": max_margin_over_bar if over_all else None,
            "block": {"rows": BLOCK, "n": int(bs.size), "max": float(bs.max()), "p999": float(np.quantile(bs, 0.999)),
                      "median": float(np.median(bs))}}


def decade_table(arrs: List[Any]) -> List[Dict[str, Any]]:
    edges = [0.0, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, np.inf]
    out = []
    absd = np.concatenate([a for a, *_ in arrs])
    m = np.concatenate([b for _a, b, *_ in arrs])
    z = absd[m == 0]
    out.append({"margin": "exact 0", "n": int(z.size), "max_abs_dlogp": float(z.max()) if z.size else None})
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = (m > lo) & (m < hi) if lo == 0 else (m >= lo) & (m < hi)
        a = absd[s]
        out.append({"margin": f"[{lo:g}, {hi:g})", "n": int(s.sum()), "max_abs_dlogp": float(a.max()) if a.size else None,
                    "over_1e-4": int((a >= 1e-4).sum())})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", required=True)
    ap.add_argument("--tf32", default=None)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    D = load(Path(a.fp32))
    rs = rounding_scale(D["fills"])
    eps = round_up_125(SAFETY * rs["R"])
    v = verify(D["arrs"], eps, BAR_FP32)
    pooled, block = v["excluded_frac"], v["block"]["max"]
    ceiling = next(c for c in CEILINGS if c >= CEIL_X_POOLED * pooled and c >= CEIL_X_BLOCK * block)
    res: Dict[str, Any] = {
        "what": "K9(b) fp32 DETERMINISTIC rule (gen3_behaviour_tie_exclusion_v1): epsilon, ceiling, verification",
        "checkpoint": "A2's 4.0M checkpoint + its 2 snapshots (~/gen3ai_archive/k6_k8/k9tail/a2), as the tail sweep",
        "path": "rust core, T2 graph backend (buckets 8/48, 7 lanes), N = 48, fills of 98,304 rows; learner probe "
                "forward eager, train mode, grad on, 1,024-row chunks; torch 2.8.0+cu126, RTX 3080 Ti",
        "criteria": {"safety": SAFETY, "eps": "round_up_125(safety x R)", "ceilings": CEILINGS,
                     "ceiling_rule": f">= {CEIL_X_POOLED} x pooled share and >= {CEIL_X_BLOCK} x the largest "
                                     f"{BLOCK}-row block share"},
        "fills": len(D["fills"]), "rounding_scale": rs, "eps": eps, "ceiling": ceiling,
        "verification": v, "by_margin": decade_table(D["arrs"]),
        "eps_grid": {f"{e:g}": {k: verify(D["arrs"], e, BAR_FP32)[k] for k in ("excluded_frac", "judged_max_abs_dlogp",
                                                                              "judged_over_bar")}
                     for e in (1e-6, 1e-5, 3e-5, 1e-4, 2e-4, 1e-3)},
        "extras": D["extras"],
    }
    if a.tf32:
        T = load(Path(a.tf32))
        rt = rounding_scale(T["fills"])
        eps_t = round_up_125(SAFETY * rt["R"])
        res["tf32"] = {"fills": len(T["fills"]), "rounding_scale": rt, "eps": eps_t,
                       "verification_at_eps": verify(T["arrs"], eps_t, 0.071),
                       "by_margin": decade_table(T["arrs"]),
                       "eps_grid": {f"{e:g}": {k: verify(T["arrs"], e, 0.071)[k] for k in
                                               ("excluded_frac", "judged_max_abs_dlogp", "judged_over_bar")}
                                    for e in (1e-5, 1e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1)},
                       "extras": T["extras"]}
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(json.dumps({k: res[k] for k in ("eps", "ceiling", "fills")}, indent=1))
    print(json.dumps(res["verification"], indent=1, default=float))


if __name__ == "__main__":
    main()
