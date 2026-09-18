#!/usr/bin/env python3
"""THE 75M RUN-LEVEL FLOOR — POLICY ENTROPY. W_b vs W (the floor) beside S vs W (the finding).

Every function below is VERBATIM from the pair read's ``pair_entropy_read.py`` (itself verbatim
from arm S's ``entropy_read.py``). What is added is W_b -- arm W's TOKEN-EXACT SEED REPLICATE --
so the FLAT-vs-DECAYING trajectory finding can be read against a 75M RUN-LEVEL floor for the
first time. |slope_W - slope_W_b| is that floor; ONE PAIR BOUNDS IT AND DOES NOT ESTIMATE IT.

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
                        "arm S -- the flywheel pair's SHAPED arm (--ent-coef 0.05, seed 1001)"),
                       ("ai_v13_02_flywheel_winprob",
                        "arm W -- the flywheel pair's WIN-PROB arm (--ent-coef 0.05, seed 1001)"),
                       ("ai_v13_04_flywheel_winprob_b",
                        "W_b -- arm W's TOKEN-EXACT SEED REPLICATE (--ent-coef 0.05, seed 1002)"),
                       ("ai_v12_02_winprob_critic",
                        "the 75M win-prob run at --ent-coef 0.02 (the '0.02 leg')"),
                       ("ai_v12_28_ladder_ent05",
                        "the 10M ent05 arm -- the registered entropy comparator")):
        try:
            out["runs"][run] = read(run, human)
        except Exception as exc:                      # noqa: BLE001 -- a failed read is reported
            out["runs"][run] = {"run": run, "error": f"{type(exc).__name__}: {exc}"}

    def matched(a_key: str, b_key: str, label: str) -> dict:
        A, B = out["runs"].get(a_key), out["runs"].get(b_key)
        if not (A and B and "trajectory_5M_buckets" in A and "trajectory_5M_buckets" in B):
            return {"label": label, "error": "one side has no trajectory"}
        ab = {r["step_bucket_M"]: r["median_H"] for r in A["trajectory_5M_buckets"]}
        bb = {r["step_bucket_M"]: r["median_H"] for r in B["trajectory_5M_buckets"]}
        common = sorted(set(ab) & set(bb))
        rows = [{"step_bucket_M": b, "a": ab[b], "b": bb[b], "a_minus_b": round(ab[b] - bb[b], 4)}
                for b in common]
        post = [r for r in rows if r["step_bucket_M"] >= 5]   # strictly post-crossing (4.128768M)
        d = [r["a_minus_b"] for r in post]
        sa = A.get("post_crossing_slope_all", {}).get("slope_nats_per_Mstep")
        sbv = B.get("post_crossing_slope_all", {}).get("slope_nats_per_Mstep")
        return {"label": label, "buckets": rows, "post_crossing_buckets": len(post),
                "mean_a_minus_b_post_crossing": round(float(np.mean(d)), 4) if d else None,
                "max_abs_a_minus_b_post_crossing": round(float(np.max(np.abs(d))), 4) if d else None,
                "H_end_delta": round(A["H_end_median20"] - B["H_end_median20"], 4),
                "post_crossing_slope_a": A.get("post_crossing_slope_all"),
                "post_crossing_slope_b": B.get("post_crossing_slope_all"),
                "slope_delta_nats_per_Mstep": (round(sa - sbv, 6) if sa is not None and sbv is not None else None),
                "total_H_change_over_span_a": (round(sa * (A["span_steps"][1] - (A["selfplay_crossing"] or {}).get("step", 0)) / 1e6, 4)
                                               if sa is not None else None),
                "total_H_change_over_span_b": (round(sbv * (B["span_steps"][1] - (B["selfplay_crossing"] or {}).get("step", 0)) / 1e6, 4)
                                               if sbv is not None else None),
                "replicate_floor_nats": REPLICATE_FLOOR_NATS}

    out["pair_matched_step_S_vs_W"] = matched("ai_v13_01_flywheel_shaped",
                                              "ai_v13_02_flywheel_winprob",
                                              "THE FINDING -- arm S minus arm W (flat vs decaying)")
    out["floor_matched_step_W_vs_Wb"] = matched("ai_v13_02_flywheel_winprob",
                                                "ai_v13_04_flywheel_winprob_b",
                                                "THE 75M RUN-LEVEL FLOOR -- arm W minus W_b (seed 1001 minus 1002)")
    out["S_vs_Wb"] = matched("ai_v13_01_flywheel_shaped",
                             "ai_v13_04_flywheel_winprob_b",
                             "arm S minus W_b -- the treatment contrast against the OTHER seed")

    f = out["floor_matched_step_W_vs_Wb"]; p = out["pair_matched_step_S_vs_W"]
    if f.get("slope_delta_nats_per_Mstep") is not None and p.get("slope_delta_nats_per_Mstep") is not None:
        fl = abs(f["slope_delta_nats_per_Mstep"]); dd = abs(p["slope_delta_nats_per_Mstep"])
        # the slope deltas carry their own OLS se's, so clause (b) is evaluated on the delta CI
        sa, sb = p["post_crossing_slope_a"], p["post_crossing_slope_b"]
        se = float(np.sqrt((sa["se"] or 0) ** 2 + (sb["se"] or 0) ** 2))
        lo, hi = dd - 1.96 * se, dd + 1.96 * se
        out["floor_verdict_entropy_trajectory"] = {
            "finding_abs_slope_delta_S_minus_W": round(dd, 6),
            "finding_ci95": [round(lo, 6), round(hi, 6)],
            "floor_abs_slope_delta_W_minus_Wb": round(fl, 6),
            "clause_a_abs_delta_gt_floor": bool(dd > fl),
            "clause_b_ci_excludes_floor_point": bool(not (lo <= fl <= hi)),
            "verdict": ("OUTSIDE THE 75M FLOOR" if (dd > fl and not (lo <= fl <= hi))
                        else "WITHIN FLOOR at n = 2"),
            "H_end_floor_nats": abs(f["H_end_delta"]),
            "H_end_finding_nats": abs(p["H_end_delta"]),
            "H_end_verdict": ("OUTSIDE THE 75M FLOOR" if abs(p["H_end_delta"]) > abs(f["H_end_delta"])
                              else "WITHIN FLOOR at n = 2"),
            "caveats": ["one replicate pair BOUNDS a floor; no CI attaches to it (rules 19/22)",
                        "WITHIN FLOOR is NEVER 'equivalent' (rule 6)",
                        "no PASS/FAIL attaches to H under any outcome (registration sec 8.3)"],
        }

    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("floor_entropy_read.json")
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps(out.get("floor_verdict_entropy_trajectory"), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
