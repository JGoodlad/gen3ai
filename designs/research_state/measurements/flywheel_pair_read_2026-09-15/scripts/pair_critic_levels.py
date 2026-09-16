#!/usr/bin/env python3
"""THE PAIR READ — the CRITIC rows, both arms, BOTH WAYS on arm S, every column LABELLED.

Same library functions ``main.ops.critic_read`` itself calls
(``conditioning_meters.conditioning_block`` + ``critic_read.gate_block``), run on the two
independent offline full-capture draws of EACH arm at its last evaluated step.

The columns, and why the labels are load-bearing:

  W-V   arm W, ``win_probs``  the ACTUAL critic — V(s) = sigmoid(win-prob logit) in [0,1]. On a
                              ``--critic winprob`` run ``values`` and ``win_probs`` are the SAME
                              TENSOR (``max_abs_values_minus_winprobs`` ~ 0).
  S-WP  arm S, ``win_probs``  the AUXILIARY win-prob head at ``--win-prob-coef 0.05`` — a
                              DIAGNOSTIC, not the value function. The column the v6 tool reads
                              by DEFAULT, which is why it must be named.
  S-V   arm S, ``values``     the ACTUAL critic — the distributional E[Z] in raw shaped-return
                              units (PopArt ON, gamma 0.9999). The treatment's own row.

Two registered contrasts, and they are DIFFERENT QUESTIONS:

  ROW A  (S-WP) - (W-V)   head-vs-head: the same readout family, a DIAGNOSTIC on one side and a
                          CRITIC on the other. Every row is defined (both columns are
                          probabilities), so calibration and the gate family live here.
  ROW B  (S-V)  - (W-V)   critic-vs-critic: the treatment's own contrast. 🚨 RANK-BASED ROWS ONLY
                          (``cond.opp_class_auc.*``) — an AUC is invariant to a monotone rescaling
                          of V, but the calibration family regresses on ``logit(V)`` and every
                          ``gate.*`` row is a Murphy decomposition of a PROBABILITY forecast, and
                          a raw shaped-return column is neither.

Floors are the v6 two-pair CONTROL-vs-CONTROL floors, measured at 10M in the four-node era and
IMPORTED to 75M for want of anything at this depth (registration sec 8.4 / 9.1).
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
os.environ.setdefault("POKESIM_SIM_BRIDGE_BIN",
                      "/home/goodlad/dev/gen3ai/src/rust_sim/target/release/sim_bridge")

from main.ops import conditioning_meters as CM          # noqa: E402
from main.ops import critic_read as CR                   # noqa: E402

COND_ROWS = ("cond.opp_class_auc.t4_10", "cond.opp_class_auc.t1_3", "cond.opp_class_auc.t1",
             "cond.calibration_slope.all", "cond.calibration_intercept.all",
             "cond.calibration_slope.t1_3",
             "cond.spread_ratio.t4_10", "cond.spread_ratio.all")
RANK_BASED = {"cond.opp_class_auc.t4_10", "cond.opp_class_auc.t1_3", "cond.opp_class_auc.t1"}
GATE_ROWS = (("resolution", "bot"), ("resolution", "all"), ("resolution", "pool"),
             ("ece", "all"), ("ece", "bot"), ("skill", "bot"), ("reliability", "all"))

# 🚨 10M, four-node-era, CONTROL-vs-CONTROL floors. The MAX of the two v6 draws per key is taken
# (rule 3: a floor is the MAX over the replicates in hand, never the mean).
FLOOR_FILES = ("designs/research_state/measurements/critic_ladder_reads/"
               "vf15_b_vs_ctrl10M_2026-09-12/hp800_floor_v6.json",
               "designs/research_state/measurements/critic_ladder_reads/"
               "vf15_b_vs_ctrl10M_2026-09-12/hp800b_floor_v6.json")
ARMS = {
    "armW": ("/home/goodlad/.claude/jobs/9ab51de6/tmp/pair_read/traces",
             "ai_v13_02_flywheel_winprob (arm W — the pair's WIN-PROB arm)"),
    "armS": ("/home/goodlad/.claude/jobs/9ab51de6/tmp/armS_reads/traces",
             "ai_v13_01_flywheel_shaped (arm S — the pair's SHAPED arm)"),
}
STEP = 74000016


def say(msg: str) -> None:
    print(f"  [{msg}]", flush=True)


def load_floors() -> dict:
    merged: dict = {}
    for rel in FLOOR_FILES:
        f = json.loads((Path("/home/goodlad/dev/gen3ai") / rel).read_text())["floors"]
        for k, v in f.items():
            merged[k] = max(merged.get(k, 0.0), float(v))
    return merged


def read_one(shadow: Path, label: str, v_column: str, *, with_gate: bool) -> dict:
    rec: dict = {"label": label, "shadow_run": str(shadow), "step": STEP, "v_column": v_column}
    cond = CM.conditioning_block(str(shadow), STEP, ladder="refit", v_column=v_column, say=say)
    pts = cond.get("points") or {}
    draws = cond.get("_draws") or {}
    ci, keep = {}, {}
    for key, arr in draws.items():
        a = np.asarray(arr, dtype=float)
        a = a[np.isfinite(a)]
        if a.size >= 100:
            ci[key] = [round(float(np.percentile(a, 2.5)), 5),
                       round(float(np.percentile(a, 97.5)), 5)]
            keep[key] = a
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
                                     "own units. On a raw shaped-return column the number is "
                                     "arithmetic, not a measurement.")
        rec["conditioning"][k] = entry
    fr = cond.get("frame") or {}
    rec["frame"] = {k: fr.get(k) for k in
                    ("n_states", "n_battles", "n_teams_seen", "max_abs_values_minus_winprobs",
                     "v_column", "v_column_note", "opponents", "n_team_cells")
                    if k in fr}
    rec["omitted"] = cond.get("omitted")
    rec["_draws"] = keep
    if with_gate:
        gate = CR.gate_block(shadow, STEP, boot=400, bins=10, seed=0, say=say)
        gdraws = gate.get("_draws") or {}
        rec["gate"], rec["_gate_draws"] = {}, {}
        for metric, stratum in GATE_ROWS:
            st = (gate.get("strata") or {}).get(stratum) or {}
            e = st.get(metric)
            if e is None:
                continue
            rec["gate"][f"gate.{metric}.{stratum}"] = {
                "value": round(float(e["point"]), 5), "ci": [round(float(x), 5) for x in e["ci"]],
                "n_states": st.get("n_states"), "n_battles": st.get("n_battles")}
            d = gdraws.get(f"{metric}.{stratum}")
            if d is not None:
                rec["_gate_draws"][f"gate.{metric}.{stratum}"] = np.asarray(d, dtype=float)
        rec["gate_note"] = ("🚨 the gate family is computed by the scaffolding gauge on the npz's "
                            "`win_probs` column ALWAYS — a Murphy decomposition of a probability "
                            "forecast has no meaning on a raw shaped-return column. On arm S every "
                            "gate row is therefore an S-WP row (the AUXILIARY head).")
        rec["gate_coverage"] = gate.get("coverage")
    return rec


def delta_block(a: dict, b: dict, floors: dict, label: str, rank_only: bool) -> dict:
    """a MINUS b, with the delta's own CI from the two independent bootstraps."""
    out: dict = {"label": label, "rank_rows_only": rank_only, "rows": {}}
    for family in ("_draws", "_gate_draws"):
        da, db = a.get(family) or {}, b.get(family) or {}
        for k in sorted(set(da) & set(db)):
            if rank_only and family == "_draws" and k not in RANK_BASED:
                continue
            if rank_only and family == "_gate_draws":
                continue
            if family == "_draws" and k not in COND_ROWS:
                continue
            xa, xb = np.asarray(da[k], float), np.asarray(db[k], float)
            n = min(len(xa), len(xb))
            d = float(np.mean(xa[:n]) - np.mean(xb[:n]))
            diff = xa[:n] - xb[:n]        # two INDEPENDENT bootstraps, differenced draw-wise
            lo, hi = float(np.percentile(diff, 2.5)), float(np.percentile(diff, 97.5))
            fl = floors.get(k)
            detected = (fl is not None and abs(d) > fl and not (lo <= 0 <= hi))
            out["rows"][k] = {
                "delta": round(d, 5), "ci95": [round(lo, 5), round(hi, 5)],
                "floor_10M_imported": (round(fl, 5) if fl is not None else None),
                "verdict": ("CANDIDATE — clears the imported floor and the CI excludes zero; "
                            "rule 22 binds, n=1 per arm" if detected else
                            "NOT DETECTED — never 'equivalent' (rule 6: equivalence needs the "
                            "delta's own CI INSIDE the bar, and this bar is imported from 10M)"),
            }
    return out


def main() -> int:
    floors = load_floors()
    out: dict = {
        "generated_at": _dt.datetime.now().astimezone().isoformat(),
        "step_read": STEP,
        "step_note": ("BOTH arms' LAST EVALUATED step, and it is the SAME step on both: each run's "
                      "final weights are at 75,005,952 and neither has an eval_traces cycle there, "
                      "so both are read 1,005,936 steps (1.3%) short of the end — an exactly "
                      "matched distance, which is what the arm-S note asked the pair read to check."),
        "floor_provenance": ("MAX over the two v6 two-pair CONTROL-vs-CONTROL floor files "
                            "(hp800_floor_v6.json, hp800b_floor_v6.json), both measured at 10M in "
                            "the four-node era and IMPORTED to 75M — there is no 75M floor and "
                            "none is affordable (registration sec 9.1)."),
        "floors_used": {k: round(v, 5) for k, v in sorted(floors.items())
                        if k in set(COND_ROWS) | {f"gate.{m}.{s}" for m, s in GATE_ROWS}},
        "arms": {}, "contrasts": {}, "eval_draw_spread": {},
    }
    cache: dict = {}
    for arm, (root, human) in ARMS.items():
        out["arms"][arm] = {"human_description": human, "draws": {}}
        for draw in ("draw1", "draw2"):
            shadow = Path(root) / draw
            man = shadow / "eval_traces" / f"step_{STEP}" / "eval_manifest.json"
            if not man.exists():
                out["arms"][arm]["draws"][draw] = {"error": f"no manifest at {man}"}
                continue
            m = json.loads(man.read_text())
            gb = m.get("generated_by") or {}
            rec = {"manifest": {k: gb.get(k) for k in
                                ("seed", "games_per_opponent", "sentinels_used", "sentinel_steps",
                                 "capture", "battles_played", "battles_expected", "complete",
                                 "eval_sentinel_greedy", "eval_sentinel_greedy_source",
                                 "checkpoint_sha", "reproducible", "wall_seconds", "impl")}}
            if arm == "armW":
                col = read_one(shadow, "W-V — the ACTUAL critic: V(s) = sigmoid(win-prob logit) in "
                               "[0,1]; on a --critic winprob run `values` and `win_probs` are the "
                               "SAME TENSOR", "win_probs", with_gate=True)
                rec["W-V"] = col
                cache[(arm, draw, "W-V")] = col
            else:
                wp = read_one(shadow, "S-WP — the AUXILIARY win-prob head at --win-prob-coef 0.05 "
                              "(a DIAGNOSTIC, not the value function); the column the v6 tool reads "
                              "by DEFAULT", "win_probs", with_gate=True)
                vv = read_one(shadow, "S-V — the ACTUAL critic: the distributional E[Z] in raw "
                              "shaped-return units (PopArt ON, gamma 0.9999)", "values",
                              with_gate=False)
                rec["S-WP"], rec["S-V"] = wp, vv
                cache[(arm, draw, "S-WP")] = wp
                cache[(arm, draw, "S-V")] = vv
            out["arms"][arm]["draws"][draw] = rec

    for draw in ("draw1", "draw2"):
        w = cache.get(("armW", draw, "W-V"))
        swp, sv = cache.get(("armS", draw, "S-WP")), cache.get(("armS", draw, "S-V"))
        if w and swp:
            out["contrasts"][f"ROW_A_{draw}"] = delta_block(
                swp, w, floors,
                "ROW A — (arm S's AUXILIARY win-prob head) MINUS (arm W's CRITIC). Head-vs-head: "
                "the same readout family, a diagnostic at coef 0.05 on one side and the value "
                "function on the other. Both columns are probabilities, so every row is defined.",
                rank_only=False)
        if w and sv:
            out["contrasts"][f"ROW_B_{draw}"] = delta_block(
                sv, w, floors,
                "ROW B — (arm S's ACTUAL critic, the distributional E[Z]) MINUS (arm W's CRITIC). "
                "Critic-vs-critic, the treatment's own contrast. RANK-BASED ROWS ONLY: an AUC is "
                "invariant to a monotone rescaling of V; the calibration family and every gate.* "
                "row are not defined on a raw shaped-return column.",
                rank_only=True)

    for arm in ARMS:
        d1 = out["arms"][arm]["draws"].get("draw1") or {}
        d2 = out["arms"][arm]["draws"].get("draw2") or {}
        for side in ("W-V", "S-WP", "S-V"):
            if side not in d1 or side not in d2:
                continue
            for fam in ("conditioning", "gate"):
                for k, e1 in (d1[side].get(fam) or {}).items():
                    e2 = (d2[side].get(fam) or {}).get(k) or {}
                    if e1.get("value") is None or e2.get("value") is None:
                        continue
                    out["eval_draw_spread"][f"{arm}|{side}|{k}"] = {
                        "draw1": e1["value"], "draw2": e2["value"],
                        "abs_draw_spread": round(abs(e1["value"] - e2["value"]), 5),
                        "floor_10M_imported": (round(floors[k], 5) if k in floors else None)}

    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if not k.startswith("_")}
        if isinstance(o, list):
            return [strip(x) for x in o]
        return o

    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("pair_critic_levels.json")
    dest.write_text(json.dumps(strip(out), indent=1, default=float))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
