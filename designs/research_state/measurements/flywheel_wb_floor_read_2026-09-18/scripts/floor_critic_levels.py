#!/usr/bin/env python3
"""THE 75M RUN-LEVEL FLOOR — the CRITIC rows, W_b against W, both columns the ACTUAL critic.

Same library functions ``main.ops.critic_read`` itself calls
(``conditioning_meters.conditioning_block`` + ``critic_read.gate_block``), on the two independent
offline full-capture draws of EACH arm at its last evaluated step — the SAME two seeds
(20260910 / 20260911) and the SAME spec the pair read used on arm W and arm S.

🚨 **THIS PAIR IS LIKE-FOR-LIKE AND THE PAIR READ'S ECE ROW WAS NOT.** On a ``--critic winprob``
run ``values`` and ``win_probs`` are the SAME TENSOR, so on BOTH sides here the column read is the
run's actual value function and ``max_abs_values_minus_winprobs`` is 0.0. The pair read's
calibration family was ROW A — arm S's AUXILIARY win-prob head at ``--win-prob-coef 0.05`` against
arm W's CRITIC, a DIAGNOSTIC against a VALUE FUNCTION — because arm S's real critic lives on a
shaped-return scale where ECE is not defined. That caveat is a property of the FINDING and no
floor measured here can remove it; what this read supplies is the run-to-run floor the finding
had to be judged against, measured critic-vs-critic on two seeds of one config.

  Wb-V  W_b, ``win_probs``   the ACTUAL critic of arm W's TOKEN-EXACT SEED REPLICATE (seed 1002)
  W-V   arm W, ``win_probs`` the ACTUAL critic (seed 1001)

The contrast |Wb-V - W-V| IS the 75M RUN-LEVEL FLOOR for every row below. ONE PAIR BOUNDS IT AND
DOES NOT ESTIMATE IT (rules 19/22): there is no CI on the floor, and rule 19's own decomposition
says these rows are RUN-level, so the eval-draw spread printed beside them never substitutes.

The 10M four-node CONTROL-vs-CONTROL floors are still loaded and printed, so every row can be
read against the imported bar AND the measured one.
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
    "armWb": ("/home/goodlad/.claude/jobs/9ab51de6/tmp/wb_read/traces",
              "ai_v13_04_flywheel_winprob_b (W_b — arm W's TOKEN-EXACT SEED REPLICATE, seed 1002)"),
    "armW": ("/home/goodlad/.claude/jobs/9ab51de6/tmp/pair_read/traces",
             "ai_v13_02_flywheel_winprob (arm W — the pair's WIN-PROB arm, seed 1001)"),
}
#: the pair read's own banked contrasts, joined for the RE-READ (never recomputed here)
PAIR_JSON = ("/home/goodlad/dev/gen3ai/designs/research_state/measurements/"
             "flywheel_pair_read_2026-09-15/out/pair_critic_levels.json")
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
        "step_note": ("Each arm's LAST EVALUATED step, and it is the SAME step on both: every run "
                      "in this family has its final weights at 75,005,952 and no eval_traces cycle "
                      "there, so all three (S, W, W_b) are read 1,005,936 steps (1.34%) short of "
                      "the end — an exactly MATCHED distance from the end on every arm."),
        "like_for_like_note": ("BOTH columns here are the run's ACTUAL critic (on --critic winprob, "
                               "`values` and `win_probs` are the same tensor). The pair read's "
                               "calibration rows were a CRITIC against a DIAGNOSTIC; this floor is "
                               "critic-vs-critic, which is what makes it a clean floor and NOT a "
                               "substitute for the comparison the pair could not make."),
        "floor_provenance_imported": ("MAX over the two v6 two-pair CONTROL-vs-CONTROL floor files "
                                      "(hp800_floor_v6.json, hp800b_floor_v6.json), both measured "
                                      "at 10M in the four-node era and IMPORTED to 75M by the pair "
                                      "read for want of anything at this depth."),
        "floors_used_imported": {k: round(v, 5) for k, v in sorted(floors.items())
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
            col = read_one(shadow, f"{arm} V — the ACTUAL critic: V(s) = sigmoid(win-prob logit) "
                           "in [0,1]; on a --critic winprob run `values` and `win_probs` are the "
                           "SAME TENSOR", "win_probs", with_gate=True)
            rec["V"] = col
            cache[(arm, draw)] = col
            out["arms"][arm]["draws"][draw] = rec

    for draw in ("draw1", "draw2"):
        b, w = cache.get(("armWb", draw)), cache.get(("armW", draw))
        if b and w:
            out["contrasts"][f"FLOOR_Wb_minus_W_{draw}"] = delta_block(
                b, w, floors,
                "THE 75M RUN-LEVEL FLOOR — W_b's CRITIC minus arm W's CRITIC. Two seeds of ONE "
                "config (the argv multiset differs by 1001 -> 1002 plus the run name), same pin, "
                "same declared dose, same eval regime. Both columns are the value function, so "
                "every row including the whole calibration family is defined on both sides. "
                "|delta| is the floor; the 'verdict' column below is the IMPORTED-floor verdict "
                "and is printed only to show what the 10M bar would have said.",
                rank_only=False)

    for arm in ARMS:
        d1 = out["arms"][arm]["draws"].get("draw1") or {}
        d2 = out["arms"][arm]["draws"].get("draw2") or {}
        if "V" not in d1 or "V" not in d2:
            continue
        for fam in ("conditioning", "gate"):
            for k, e1 in (d1["V"].get(fam) or {}).items():
                e2 = (d2["V"].get(fam) or {}).get(k) or {}
                if e1.get("value") is None or e2.get("value") is None:
                    continue
                out["eval_draw_spread"][f"{arm}|V|{k}"] = {
                    "draw1": e1["value"], "draw2": e2["value"],
                    "abs_draw_spread": round(abs(e1["value"] - e2["value"]), 5),
                    "floor_10M_imported": (round(floors[k], 5) if k in floors else None)}

    # ---- THE RE-READ: the pair's own banked S-vs-W deltas against the floor measured here ----
    try:
        pair = json.loads(Path(PAIR_JSON).read_text())
    except Exception as exc:                            # noqa: BLE001
        out["re_read_vs_pair"] = {"error": f"{type(exc).__name__}: {exc}"}
        pair = None
    if pair:
        reread: dict = {
            "source": PAIR_JSON,
            "rule": ("OUTSIDE THE 75M FLOOR iff |S - W| > |W - W_b| AND the S-W CI excludes the "
                     "|W - W_b| POINT (PREDICTION.md sec 2). Otherwise WITHIN FLOOR at n = 2, "
                     "which is NEVER 'equivalent' (rule 6)."),
            "rows": {},
        }
        for draw in ("draw1", "draw2"):
            fl_rows = (out["contrasts"].get(f"FLOOR_Wb_minus_W_{draw}") or {}).get("rows") or {}
            pr_rows = (pair.get("contrasts", {}).get(f"ROW_A_{draw}") or {}).get("rows") or {}
            for k in sorted(set(fl_rows) & set(pr_rows)):
                dd = abs(pr_rows[k]["delta"])
                lo, hi = pr_rows[k]["ci95"]
                fl = abs(fl_rows[k]["delta"])
                inside = (lo <= fl <= hi) or (lo <= -fl <= hi)
                reread["rows"][f"{k}|{draw}"] = {
                    "finding_S_minus_W_ROW_A": pr_rows[k]["delta"],
                    "finding_ci95": [lo, hi],
                    "floor_abs_W_minus_Wb": round(fl, 5),
                    "floor_signed_Wb_minus_W": fl_rows[k]["delta"],
                    "imported_10M_floor": fl_rows[k]["floor_10M_imported"],
                    "clause_a_abs_delta_gt_floor": bool(dd > fl),
                    "clause_b_ci_excludes_floor_point": bool(not inside),
                    "verdict": ("OUTSIDE THE 75M FLOOR" if (dd > fl and not inside)
                                else "WITHIN FLOOR at n = 2"),
                    "standing_caveat": ("the pair's ROW A is a CRITIC (arm W) against a DIAGNOSTIC "
                                        "(arm S's auxiliary win-prob head at coef 0.05). No floor "
                                        "converts it into a critic-vs-critic statement."),
                }
        out["re_read_vs_pair"] = reread

    def strip(o):
        if isinstance(o, dict):
            return {k: strip(v) for k, v in o.items() if not k.startswith("_")}
        if isinstance(o, list):
            return [strip(x) for x in o]
        return o

    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("floor_critic_levels.json")
    dest.write_text(json.dumps(strip(out), indent=1, default=float))
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
