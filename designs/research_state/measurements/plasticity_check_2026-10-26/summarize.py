"""Tables and the verdict of the plasticity check, from ``results/*.json`` (written by ``plasticity_probe.py``).

    python summarize.py results/

Pure arithmetic over the committed JSONs; no model is loaded. The verdict rule is the README's §2 (the strategy
doc's registered bars on the end-to-end P*-label fit, with the stated downgrade), applied mechanically.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

ORDER = ["fresh0", "fresh1", "x26_010M", "x26_025M", "x26_100M", "x26_175M", "x26_280M", "x26_380M", "x26c_480M",
         "c0_s3501_300M"]
BUDGET = 400


def auc_norm(curve: Dict[int, float], budget: int) -> float:
    """Area under loss(t) / loss(0) over t in [0, budget] (trapezoid), divided by the budget: 1.0 = never moved."""
    pts = sorted((int(k), float(v)) for k, v in curve.items() if int(k) <= budget)
    if pts[0][0] != 0 or pts[-1][0] != budget:
        raise ValueError(f"curve does not span 0..{budget}: {pts}")
    l0 = pts[0][1]
    area = sum((t1 - t0) * (a + b) / 2 for (t0, a), (t1, b) in zip(pts, pts[1:]))
    return area / l0 / budget


def refit_table(res: Dict[str, dict]) -> Dict[str, dict]:
    """Per net: the seed-mean held-out loss at the budget, its seeds, and the seed-mean normalised AUC."""
    by: Dict[str, List[dict]] = {}
    for r in res.values():
        by.setdefault(r["net"], []).append(r)
    out = {}
    for net, rs in by.items():
        finals = [r["held_curve"][str(BUDGET)] for r in sorted(rs, key=lambda r: r["seed"])]
        aucs = [auc_norm(r["held_curve"], BUDGET) for r in sorted(rs, key=lambda r: r["seed"])]
        out[net] = {"final": sum(finals) / len(finals), "finals": finals, "auc": sum(aucs) / len(aucs), "aucs": aucs}
    return out


def head_auc(rec: List[dict], start: float) -> float:
    """Frozen head: the normalised AUC of the TRAIN-loss curve at steps 0 / 30 / 100 / 300 / 1000 (seed mean)."""
    vals = []
    for r in rec:
        curve = {0: start}
        curve.update({int(k): v["train"] for k, v in r["curve"].items()})
        vals.append(auc_norm(curve, 1000))
    return sum(vals) / len(vals)


def classify(ratio: float, dormant_ratio: float) -> str:
    """The strategy doc's registered bars (§2)."""
    if ratio >= 1.25 and dormant_ratio >= 2.0:
        return "LOSS"
    if ratio < 1.10:
        return "NO-LOSS"
    return "UNCLEAR"


def main(argv: Optional[List[str]] = None) -> int:
    d = Path((argv or sys.argv[1:])[0])
    st = json.loads((d / "structure.json").read_text())
    heads = json.loads((d / "heads.json").read_text())
    refits = {t: refit_table(json.loads((d / f"refit_{t}.json").read_text()))
              for t in ("t_net", "t_lin", "win", "pstar") if (d / f"refit_{t}.json").exists()}

    print("## structure (unfam bank)")
    print("| net | dormant τ=0 | τ=0.025 | τ=0.1 | srank latent_pi /512 | srank vf /128 | PR latent_pi | PR vf | "
          "tanh-1 sat. units / mean | tanh-2 sat. units / mean | ‖W‖ total | ‖W‖ policy MLP | grad norm mean / median | "
          "mean abs grad cos | grad Gram PR |")
    print("|" + "---|" * 16)
    for k in ORDER:
        r, u = st[k], st[k]["unfam"]
        dp, rk, tn, g = u["dormant_pooled"], u["ranks"], u["tanh"], st[k]["grad"]
        t1, t2 = (tn[n] for n in sorted(tn))
        print(f"| {k} | {dp['d0.0']:.3f} | {dp['d0.025']:.3f} | {dp['d0.1']:.3f} | {rk['latent_pi']['srank_kumar']} | "
              f"{rk['vf']['srank_kumar']} | {rk['latent_pi']['pr']:.1f} | {rk['vf']['pr']:.2f} | "
              f"{t1['saturated_units']:.3f} / {t1['mean_sat_frac']:.3f} | {t2['saturated_units']:.3f} / "
              f"{t2['mean_sat_frac']:.3f} | {r['weight_norm_total_excl_ridealong']:.1f} | "
              f"{r['weight_norm_policy_mlp']:.1f} | {g['mean_norm']:.1f} / {g['median_norm']:.1f} | "
              f"{g['mean_abs_offdiag_cos']:.3f} | {g['gram_pr']:.1f} |")
    print("\n## dormant τ=0.025 on the other banks (fixture / golden)")
    for k in ORDER:
        print(k, st[k]["fixture"]["dormant_pooled"]["d0.025"], st[k]["golden"]["dormant_pooled"]["d0.025"],
              "srank lat", st[k]["fixture"]["ranks"]["latent_pi"]["srank_kumar"],
              st[k]["golden"]["ranks"]["latent_pi"]["srank_kumar"])

    print("\n## end-to-end refits: held-out loss at step 400 (seeds) | normalised AUC")
    print("| net | " + " | ".join(refits) + " |")
    print("|---|" + "---|" * len(refits))
    for k in ORDER:
        cells = []
        for t, tab in refits.items():
            if k in tab:
                x = tab[k]
                cells.append(f"{x['final']:.3f} ({', '.join(f'{v:.3f}' for v in x['finals'])}) · AUC {x['auc']:.3f}")
            else:
                cells.append("—")
        print(f"| {k} | " + " | ".join(cells) + " |")

    print("\n## frozen heads: held-out loss at 300 steps | train loss at 1000 | AUC (win / species)")
    tasks = list(next(iter(heads["reads"].values())))
    for k in ORDER:
        rec = heads["reads"][k]
        row = []
        for t in tasks:
            held = sum(x["curve"]["300"]["held"] for x in rec[t]) / len(rec[t])
            tr = sum(x["curve"]["1000"]["train"] for x in rec[t]) / len(rec[t])
            extra = f" AUC {sum(x['held_auc'] for x in rec[t]) / len(rec[t]):.3f}" if "held_auc" in rec[t][0] else ""
            row.append(f"{t}: {held:.3f} / {tr:.3f}{extra}")
        print(k, " | ".join(row))

    dorm = {k: st[k]["unfam"]["dormant_pooled"]["d0.025"] for k in ORDER}
    dratio = dorm["x26_280M"] / dorm["x26_025M"]
    print(f"\n## the registered bars (P* = x26_280M vs x26_025M); near-dormant ratio {dratio:.3f}")
    for t, tab in refits.items():
        if "x26_025M" in tab and "x26_280M" in tab:
            ratio = tab["x26_280M"]["auc"] / tab["x26_025M"]["auc"]
            print(f"end-to-end {t}: AUC ratio {ratio:.3f} -> {classify(ratio, dratio)}")
    if "pstar" in heads["reads"]["x26_280M"]:
        bank = d / "bank_meta.json"
        start = json.loads(bank.read_text()).get("pstar_train_start_ce") if bank.exists() else None
        if start is not None:
            a25 = head_auc(heads["reads"]["x26_025M"]["pstar"], start)
            a280 = head_auc(heads["reads"]["x26_280M"]["pstar"], start)
            print(f"frozen pstar: AUC 25M {a25:.3f} P* {a280:.3f} ratio {a280 / a25:.3f} -> "
                  f"{classify(a280 / a25, dratio)}")
    for t in ("t_net",):
        a25, a280 = (head_auc(heads["reads"][k][t], 1.0) for k in ("x26_025M", "x26_280M"))
        print(f"frozen {t}: AUC 25M {a25:.3f} P* {a280:.3f} ratio {a280 / a25:.3f} -> {classify(a280 / a25, dratio)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
