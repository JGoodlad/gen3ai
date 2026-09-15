#!/usr/bin/env python3
"""ARM S's registered CRITIC rows — READ BOTH WAYS, and both LABELLED.

Registration sec 8.4 / hazard H5: on every win-prob arm ``values`` and ``win_probs`` are the same
tensor, so the v6 path's QC scalar ``max_abs_values_minus_winprobs`` reads ~0.  ON ARM S THEY ARE
DIFFERENT READOUTS and that scalar is LARGE BY CONSTRUCTION — which is not a defect:

  S-V   ``values``     the ACTUAL critic — the distributional E[Z] in raw shaped-return units
                       (PopArt ON, gamma 0.9999). This is the treatment's own row.
  S-WP  ``win_probs``  the AUXILIARY win-prob head at ``--win-prob-coef 0.05`` — a diagnostic,
                       NOT the value function, and the column the v6 tool reads by DEFAULT.

🚨 A CONSTRAINT THE REGISTRATION DID NOT ANTICIPATE, found here: only the RANK-based rows survive
the switch. ``cond.opp_class_auc.*`` is an AUC and is invariant to any monotone rescaling of V, so
it is genuinely readable both ways. The calibration family (``cond.calibration_slope/intercept``)
regresses the outcome on ``logit(V)`` and every ``gate.*`` row is a Murphy decomposition of a
PROBABILITY forecast — neither is defined on a raw shaped-return column. Those rows are therefore
reported S-WP ONLY, and the reason is printed beside them rather than left for a reader to infer.

Reads two independent offline full-capture draws (rule 19: the eval-draw component is above
sampling noise, so two draws BOUND a floor — they never give it a CI).  Levels only: the one
available 75M control tree is at the OTHER sentinel regime and a different games-per-opponent, so
no delta is taken against it (see the README).
"""
from __future__ import annotations

import datetime as _dt
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/goodlad/dev/gen3ai-wt/armS-reads/src")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("POKESIM_SIM_BRIDGE_BIN",
                      "/home/goodlad/dev/gen3ai/src/rust_sim/target/release/sim_bridge")

from main.ops import conditioning_meters as CM          # noqa: E402
from main.ops import critic_read as CR                   # noqa: E402
from main.ops import critic_readouts as R                # noqa: E402

COND_ROWS = ("cond.opp_class_auc.t4_10", "cond.opp_class_auc.t1_3", "cond.opp_class_auc.t1",
             "cond.calibration_slope.all", "cond.calibration_intercept.all",
             "cond.calibration_slope.t1_3",
             "cond.spread_ratio.t4_10", "cond.spread_ratio.all")
RANK_BASED = {"cond.opp_class_auc.t4_10", "cond.opp_class_auc.t1_3", "cond.opp_class_auc.t1"}
GATE_ROWS = (("resolution", "bot"), ("resolution", "all"), ("resolution", "pool"),
             ("ece", "all"), ("ece", "bot"), ("skill", "bot"), ("reliability", "all"))

# The v6 two-pair floors in hand. Both are 10M, four-node-era, CONTROL-vs-CONTROL floors and are
# imported to 75M for want of anything at this depth (registration sec 8.4 / 9.1).
FLOORS = {"first_draw": 0.0220, "second_draw": 0.0245}


def say(msg: str) -> None:
    print(f"  [{msg}]", flush=True)


def read_one(shadow: Path, step: int, label: str, v_column: str, *, with_gate: bool) -> dict:
    rec: dict = {"label": label, "shadow_run": str(shadow), "step": step, "v_column": v_column}
    cond = CM.conditioning_block(str(shadow), step, ladder="refit", v_column=v_column, say=say)
    pts = cond.get("points") or {}
    # `conditioning_block` publishes the raw battle-clustered draws, not intervals — a delta between
    # two runs is the difference of two independent bootstraps and cannot be assembled from two
    # published CIs, which is why it hands back the draws. Here only LEVELS are wanted, so the
    # percentile interval is taken directly off each row's own draws.
    draws = cond.get("_draws") or {}
    ci = {}
    for key, arr in draws.items():
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if a.size >= 100:
            ci[key] = [round(float(np.percentile(a, 2.5)), 5),
                       round(float(np.percentile(a, 97.5)), 5)]
    rec["conditioning"] = {}
    for k in COND_ROWS:
        v = pts.get(k)
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            rec["conditioning"][k] = {"value": None, "omitted": (cond.get("omitted") or {}).get(k)}
            continue
        entry = {"value": round(float(v), 5), "ci": ci.get(k)}
        if v_column == "values" and k not in RANK_BASED:
            entry["NOT_READABLE"] = ("this row needs V on a PROBABILITY scale — the calibration "
                                     "family regresses on logit(V) and the spread rows are in V's "
                                     "own units. On the raw shaped-return column the number is "
                                     "arithmetic, not a measurement. Read the S-WP row instead.")
        rec["conditioning"][k] = entry
    fr = cond.get("frame") or {}
    rec["frame"] = {k: fr.get(k) for k in
                    ("n_states", "n_battles", "n_teams_seen", "max_abs_values_minus_winprobs",
                     "v_column", "v_column_note", "opponents", "n_team_cells")
                    if k in fr}
    rec["omitted"] = cond.get("omitted")
    if with_gate:
        gate = CR.gate_block(shadow, step, boot=400, bins=10, seed=0, say=say)
        rec["gate"] = {}
        for metric, stratum in GATE_ROWS:
            st = (gate.get("strata") or {}).get(stratum) or {}
            e = st.get(metric)
            if e is None:
                continue
            rec["gate"][f"gate.{metric}.{stratum}"] = {
                "value": round(float(e["point"]), 5), "ci": [round(float(x), 5) for x in e["ci"]],
                "n_states": st.get("n_states"), "n_battles": st.get("n_battles")}
        rec["gate_note"] = ("🚨 the gate family is computed by the scaffolding gauge on the npz's "
                            "`win_probs` column ALWAYS — a Murphy decomposition of a probability "
                            "forecast has no meaning on a raw shaped-return column. On arm S every "
                            "gate row is therefore an S-WP row (the AUXILIARY head), whatever the "
                            "conditioning block was asked to read.")
        rec["gate_coverage"] = gate.get("coverage")
    return rec


def main() -> int:
    traces = Path("/home/goodlad/.claude/jobs/9ab51de6/tmp/armS_reads/traces")
    step = 74000016
    out = {"generated_at": _dt.datetime.now().astimezone().isoformat(),
           "arm": "ai_v13_01_flywheel_shaped (arm S — the flywheel pair's SHAPED arm)",
           "step_read": step,
           "step_note": ("the run's LAST EVALUATED step. The final weights are at 75,005,952 but "
                         "the run has no eval_traces cycle there, and eval_trace_gen deliberately "
                         "replays the bit-exact `eval_traces/step_<N>/snapshot.zip` that played "
                         "the live cycle rather than a checkpoint that merely shares a step."),
           "floors": FLOORS,
           "floor_provenance": ("v6 two-pair control-vs-control floors measured at 10M, four-node "
                                "era; imported to 75M for want of anything at this depth"),
           "draws": {}}
    for draw in ("draw1", "draw2"):
        shadow = traces / draw
        man = shadow / "eval_traces" / f"step_{step}" / "eval_manifest.json"
        if not man.exists():
            out["draws"][draw] = {"error": f"no manifest at {man}"}
            continue
        m = json.loads(man.read_text())
        gb = m.get("generated_by") or {}
        out["draws"][draw] = {
            "manifest": {k: gb.get(k) for k in
                         ("seed", "games_per_opponent", "sentinels_used", "sentinel_steps",
                          "capture", "battles_played", "complete", "eval_sentinel_greedy",
                          "eval_sentinel_greedy_source", "checkpoint_sha", "reproducible",
                          "wall_seconds", "impl")},
            "S-WP": read_one(shadow, step, "S-WP — the AUXILIARY win-prob head at "
                             "--win-prob-coef 0.05 (a DIAGNOSTIC, not the value function); this is "
                             "the column the v6 tool reads by DEFAULT",
                             "win_probs", with_gate=True),
            "S-V": read_one(shadow, step, "S-V — the ACTUAL critic: the distributional E[Z] in raw "
                            "shaped-return units (PopArt ON, gamma 0.9999). The treatment's own row",
                            "values", with_gate=False),
        }

    # the eval-DRAW spread of every row that exists on both draws (rule 19: two draws BOUND a
    # floor at the eval-draw level; they never give it a CI).
    spread: dict = {}
    d1, d2 = out["draws"].get("draw1"), out["draws"].get("draw2")
    if d1 and d2 and "error" not in d1 and "error" not in d2:
        for side in ("S-WP", "S-V"):
            for k, e1 in (d1[side].get("conditioning") or {}).items():
                e2 = (d2[side].get("conditioning") or {}).get(k) or {}
                if e1.get("value") is None or e2.get("value") is None:
                    continue
                spread[f"{side}|{k}"] = {"draw1": e1["value"], "draw2": e2["value"],
                                         "abs_draw_spread": round(abs(e1["value"] - e2["value"]), 5)}
            for k, e1 in (d1[side].get("gate") or {}).items():
                e2 = (d2[side].get("gate") or {}).get(k) or {}
                if not e2:
                    continue
                spread[f"{side}|{k}"] = {"draw1": e1["value"], "draw2": e2["value"],
                                         "abs_draw_spread": round(abs(e1["value"] - e2["value"]), 5)}
    out["eval_draw_spread"] = spread
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("critic_levels.json")
    dest.write_text(json.dumps(out, indent=1, default=float))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
