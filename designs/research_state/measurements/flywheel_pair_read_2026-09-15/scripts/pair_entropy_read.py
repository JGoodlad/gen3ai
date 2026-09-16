#!/usr/bin/env python3
"""THE PAIR READ — POLICY ENTROPY. Arm S and arm W (registration sec 8.3).

Same code path as the arm-S standalone read
(``measurements/flywheel_armS_reads_2026-09-14/scripts/entropy_read.py``); this adds arm W and
the MATCHED-STEP S-vs-W comparison the registration calls a FINDING rather than a bar.

``train/entropy_loss`` in nats, read from the TENSORBOARD EVENTS (never the child log's table,
which is a rendering and undersamples the rollouts).  Reported:

  * H_start / H_end as MEDIAN-20 (the window is pinned in this file, BEFORE the number is read)
  * the late slope in nats per 1M steps with its residual se
  * the self-play crossing step, read from the events (standing rule 15: a windowed statistic
    never crosses an opponent-regime boundary, and the crossing is a PER-ARM lottery)
  * the H_end comparison against the `ent05` arm's 1.086 and v8's 1.07-1.11 plateau, with the
    0.074-nat three-seed replicate floor

`--ent-coef` is LIVE on a resume (model_build.py:543), unlike `--lr`, so 0.05 holds for BOTH arms'
whole runs and each series is one regime in the coefficient.

🚨 Registration sec 8.3: a difference between arm S's H and arm W's at matched steps is a FINDING,
not a bar — the entropy coefficient is matched, so entropy is downstream of the reward composition.
The replicate floor in hand is 0.074 nats (three seeds, measured at 10M).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/goodlad/dev/gen3ai/src")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from main.ops import tb_read  # noqa: E402

MODELS = Path("/home/goodlad/dev/gen3ai/models")
TAG = "train/entropy_loss"
MEDIAN_WINDOW = 20           # pinned before the number is read
LATE_FRACTION = 1.0 / 3.0    # the last third of the post-crossing span

# Established comparators, quoted with their provenance.
ENT05_H_END = 1.0861          # ai_v12_28_ladder_ent05 @10M [ledger 2026-09-12 · VERDICT · ent05 PASSES]
V8_PLATEAU = (1.07, 1.11)     # entropy_forensics_v8_vs_ours_2026-09-12
REPLICATE_FLOOR_NATS = 0.074  # three seeds, measured at 10M


def ols_slope(x: np.ndarray, y: np.ndarray) -> dict:
    n = len(x)
    if n < 3:
        return {"n": int(n), "slope": None, "se": None}
    xc = x - x.mean()
    sxx = float((xc * xc).sum())
    slope = float((xc * (y - y.mean())).sum() / sxx)
    icept = float(y.mean() - slope * x.mean())
    resid = y - (icept + slope * x)
    s2 = float((resid ** 2).sum() / (n - 2))
    se = float(np.sqrt(s2 / sxx))
    return {"n": int(n), "slope_nats_per_Mstep": round(slope, 5), "se": round(se, 5),
            "t": round(slope / se, 2) if se > 0 else None,
            "span_Mstep": [round(float(x[0]), 3), round(float(x[-1]), 3)]}


def read(run: str, human: str) -> dict:
    run_dir = MODELS / run
    series = tb_read.load(run_dir, [TAG, "train/selfplay_fraction"])
    pts = series.get(TAG) or []
    if not pts:
        return {"run": run, "error": f"NO SCALAR FOUND for {TAG} — absence is not a zero"}
    steps = np.array([p[0] for p in pts], dtype=float)
    # 🚨 SB3 logs `train/entropy_loss` as the NEGATIVE mean policy entropy (it is the term that
    # enters the loss). H in nats is its negation; every number below is H, never entropy_loss.
    vals = -np.array([p[1] for p in pts], dtype=float)
    order = np.argsort(steps)
    steps, vals = steps[order], vals[order]

    crossing = tb_read.selfplay_crossing_step(run_dir)
    cross_step = crossing.get("step") if isinstance(crossing, dict) else None

    rec: dict = {
        "run": run, "human_description": human, "tag": TAG,
        "n_points": int(len(steps)),
        "span_steps": [int(steps[0]), int(steps[-1])],
        "selfplay_crossing": crossing,
        "median_window": MEDIAN_WINDOW,
        "H_start_median20": round(float(np.median(vals[:MEDIAN_WINDOW])), 4),
        "H_end_median20": round(float(np.median(vals[-MEDIAN_WINDOW:])), 4),
        "H_min": round(float(vals.min()), 4), "H_max": round(float(vals.max()), 4),
        "sign_convention": ("H = -train/entropy_loss (SB3 logs the loss term, i.e. the NEGATIVE "
                            "mean policy entropy). Every H below is in nats and positive."),
    }

    # the LATE slope, struck strictly inside the post-crossing regime (rule 15)
    mask = steps > (cross_step or 0)
    s_post, v_post = steps[mask], vals[mask]
    rec["post_crossing_points"] = int(len(s_post))
    if len(s_post) >= 6:
        k = max(6, int(len(s_post) * LATE_FRACTION))
        rec["late_slope"] = ols_slope(s_post[-k:] / 1e6, v_post[-k:])
        rec["late_slope"]["window_note"] = ("the last third of the POST-CROSSING span only — "
                                            "rule 15: a windowed statistic never crosses the "
                                            "self-play boundary")
        rec["post_crossing_slope_all"] = ols_slope(s_post / 1e6, v_post)

    # a coarse trajectory, one point per ~5M steps, so the README can show the shape
    traj = []
    for lo in range(0, int(steps[-1]) + 1, 5_000_000):
        m = (steps >= lo) & (steps < lo + 5_000_000)
        if m.sum():
            traj.append({"step_bucket_M": lo // 1_000_000, "n": int(m.sum()),
                         "median_H": round(float(np.median(vals[m])), 4)})
    rec["trajectory_5M_buckets"] = traj

    h = rec["H_end_median20"]
    rec["comparators"] = {
        "ent05_H_end_10M": ENT05_H_END,
        "delta_vs_ent05": round(h - ENT05_H_END, 4),
        "replicate_floor_nats": REPLICATE_FLOOR_NATS,
        "vs_ent05_reads": ("NOT READ (below the 0.074-nat floor)"
                           if abs(h - ENT05_H_END) < REPLICATE_FLOOR_NATS
                           else "a difference larger than the 0.074-nat floor"),
        "v8_plateau": list(V8_PLATEAU),
        "inside_v8_plateau": bool(V8_PLATEAU[0] <= h <= V8_PLATEAU[1]),
        "distance_to_v8_plateau": (0.0 if V8_PLATEAU[0] <= h <= V8_PLATEAU[1]
                                   else round(min(abs(h - V8_PLATEAU[0]), abs(h - V8_PLATEAU[1])), 4)),
        "floor_caveat": ("both comparators were measured at 10M depth; arm S is at 75M and there is "
                         "no 75M replicate — registration sec 9.1. No PASS/FAIL attaches to H."),
    }
    return rec


def main() -> int:
    out = {"generated_at": _dt.datetime.now().astimezone().isoformat(),
           "instrument": "TensorBoard events via main.ops.tb_read (never the child log table)",
           "runs": {}}
    for run, human in (("ai_v13_01_flywheel_shaped",
                        "arm S — the flywheel pair's SHAPED arm (--ent-coef 0.05)"),
                       ("ai_v13_02_flywheel_winprob",
                        "arm W — the flywheel pair's WIN-PROB arm (--ent-coef 0.05)"),
                       ("ai_v12_02_winprob_critic",
                        "the 75M win-prob run at --ent-coef 0.02 (the '0.02 leg')"),
                       ("ai_v12_28_ladder_ent05",
                        "the 10M ent05 arm — the registered entropy comparator")):
        try:
            out["runs"][run] = read(run, human)
        except Exception as exc:                      # noqa: BLE001 — a failed read is reported
            out["runs"][run] = {"run": run, "error": f"{type(exc).__name__}: {exc}"}
    # ---- the MATCHED-STEP pair comparison (registration sec 8.3: a FINDING, never a bar) ----
    S, W = out["runs"].get("ai_v13_01_flywheel_shaped"), out["runs"].get("ai_v13_02_flywheel_winprob")
    if S and W and "trajectory_5M_buckets" in S and "trajectory_5M_buckets" in W:
        sb = {r["step_bucket_M"]: r["median_H"] for r in S["trajectory_5M_buckets"]}
        wb = {r["step_bucket_M"]: r["median_H"] for r in W["trajectory_5M_buckets"]}
        common = sorted(set(sb) & set(wb))
        rows = [{"step_bucket_M": b, "armS": sb[b], "armW": wb[b],
                 "S_minus_W": round(sb[b] - wb[b], 4)} for b in common]
        post = [r for r in rows if r["step_bucket_M"] >= 5]     # strictly post-crossing (4.13M)
        d = [r["S_minus_W"] for r in post]
        out["pair_matched_step"] = {
            "buckets": rows,
            "post_crossing_buckets": len(post),
            "mean_S_minus_W_post_crossing": round(float(np.mean(d)), 4) if d else None,
            "max_abs_S_minus_W_post_crossing": round(float(np.max(np.abs(d))), 4) if d else None,
            "H_end_delta": round(S["H_end_median20"] - W["H_end_median20"], 4),
            "replicate_floor_nats": REPLICATE_FLOOR_NATS,
            "reads": ("a FINDING only if it clears the 0.074-nat floor; the floor was measured on "
                      "three seeds at 10M and is imported to 75M (registration sec 9.1). No "
                      "PASS/FAIL attaches to H either way."),
        }
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("entropy_read.json")
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
