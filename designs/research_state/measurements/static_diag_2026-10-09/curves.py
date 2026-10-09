"""H1 (attention slow to learn?) — learning curves, legacy vs static. DESCRIPTIVE / exploratory.

Reads, read-only, each finished screen seed's TensorBoard events and `eval_results.jsonl`:
  * the in-loop bots eval (8 training bots + random, 100 games each, every 2M steps);
  * TB: train/value_loss, train/explained_variance, train/entropy_loss, train/approx_kl,
    win_prob/brier, win_prob/brier_contested, train/clip_fraction.
Per seed, each TB scalar is averaged in 1M-step bins (a resumed child's re-written steps: the
LAST-written value per step wins). Per arm: mean and sd over seeds per bin, and per seed the OLS slope
over the last 5M (10M..15M), arm mean ± sd.

Writes curves.json next to this file. No GPU, no model load.
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

MODELS = Path("/home/goodlad/dev/gen3ai/models")
TAGS = ["train/value_loss", "train/explained_variance", "train/entropy_loss", "train/approx_kl",
        "win_prob/brier", "win_prob/brier_contested", "train/clip_fraction", "eval/win_rate_vs_bots"]
BIN = 1_000_000
BOTS = ["heuristic", "heuristic2", "staller", "staller_v2", "aggressive", "aggressive_v2",
        "setup_sweep", "setup_sweep_v2"]          # `random` left out: saturated at ~1.0


def runs():
    out = {"legacy": [], "static": []}
    for arm in out:
        for d in sorted(MODELS.glob(f"rb_st_{arm}_s1*")):
            if (d / "final_model.zip").exists():
                out[arm].append(d)
    return out


def tb_series(run: Path):
    per_tag = {t: {} for t in TAGS}
    files = sorted(glob.glob(str(run / "tb" / "*")))
    for f in files:                                   # file order == child order (timestamp names)
        e = EventAccumulator(f, size_guidance={"scalars": 0})
        e.Reload()
        have = set(e.Tags()["scalars"])
        for t in TAGS:
            if t in have:
                for ev in e.Scalars(t):
                    per_tag[t][int(ev.step)] = float(ev.value)   # later child overwrites
    return per_tag


def binned(series: dict, lo=0, hi=15_000_000):
    edges = np.arange(lo, hi + BIN, BIN)
    out = []
    steps = np.array(sorted(series))
    vals = np.array([series[s] for s in steps])
    for a, b in zip(edges[:-1], edges[1:]):
        sel = (steps >= a) & (steps < b) & np.isfinite(vals)
        out.append(float(vals[sel].mean()) if sel.any() else None)
    return out


def slope(series: dict, lo=10_000_000, hi=15_100_000):
    steps = np.array(sorted(s for s in series if lo <= s < hi), dtype=float)
    if len(steps) < 3:
        return None
    v = np.array([series[int(s)] for s in steps])
    ok = np.isfinite(v)
    if ok.sum() < 3:
        return None
    return float(np.polyfit(steps[ok] / 1e6, v[ok], 1)[0])       # per 1M steps


def evals(run: Path):
    rows = {}
    for line in (run / "eval_results.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        rows[int(round(r["step"] / 1e6))] = {
            "bots8_mean": float(np.mean([r["bots"][b] for b in BOTS])),
            "pool_mean": (float(np.mean([s["win_rate"] for s in r["sentinels"]])) if r["sentinels"] else None)}
    return rows


def summarise(vals):
    v = np.array([x for x in vals if x is not None], dtype=float)
    if len(v) == 0:
        return None
    return {"mean": round(float(v.mean()), 5), "sd": round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 5),
            "n": int(len(v))}


def main():
    R = runs()
    out = {"schema": "static_diag_curves_v1", "tag": "DESCRIPTIVE", "runs": {a: [p.name for p in v] for a, v in R.items()},
           "bins_M": list(range(0, 15)), "tb": {}, "tb_slope_last5M_per_M": {}, "bots_eval": {}}
    per = {a: {p.name: tb_series(p) for p in v} for a, v in R.items()}
    for t in TAGS:
        out["tb"][t] = {}
        out["tb_slope_last5M_per_M"][t] = {}
        for a in R:
            B = [binned(per[a][n][t]) for n in per[a]]
            out["tb"][t][a] = [summarise([b[i] for b in B]) for i in range(15)]
            sl = [slope(per[a][n][t]) for n in per[a]]
            out["tb_slope_last5M_per_M"][t][a] = {"per_seed": sl, "summary": summarise(sl)}
    for a in R:
        E = {p.name: evals(p) for p in R[a]}
        out["bots_eval"][a] = {"per_seed": E}
        steps = sorted({s for e in E.values() for s in e})
        out["bots_eval"][a]["by_step"] = {s: summarise([E[n].get(s, {}).get("bots8_mean") for n in E]) for s in steps}
        sl = []
        for n, e in E.items():
            pts = [(s, e[s]["bots8_mean"]) for s in (10, 12, 14) if s in e]
            sl.append(float(np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)[0]) if len(pts) == 3 else None)
        out["bots_eval"][a]["slope_10_14M_per_M"] = {"per_seed": sl, "summary": summarise(sl)}
    Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).with_name("curves.json")).write_text(
        json.dumps(out, indent=1) + "\n")
    # console digest
    for s in sorted(out["bots_eval"]["legacy"]["by_step"]):
        L, S = out["bots_eval"]["legacy"]["by_step"][s], out["bots_eval"]["static"]["by_step"].get(s)
        print(f"bots8 @{s}M legacy {L['mean']:.4f}±{L['sd']:.4f}  static {S['mean']:.4f}±{S['sd']:.4f}  "
              f"Δ {S['mean'] - L['mean']:+.4f}")
    for t in TAGS:
        row = []
        for i in (1, 2, 4, 7, 10, 14):
            L, S = out["tb"][t]["legacy"][i], out["tb"][t]["static"][i]
            if L and S:
                row.append(f"{i}M L {L['mean']:.4g} S {S['mean']:.4g}")
        print(t, " | ".join(row))
        print("   slope last5M", out["tb_slope_last5M_per_M"][t]["legacy"]["summary"],
              out["tb_slope_last5M_per_M"][t]["static"]["summary"])
    print("bots slope 10-14M", out["bots_eval"]["legacy"]["slope_10_14M_per_M"]["summary"],
          out["bots_eval"]["static"]["slope_10_14M_per_M"]["summary"])


if __name__ == "__main__":
    main()
