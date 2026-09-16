#!/usr/bin/env python3
"""THE PAIR READ — STRENGTH. Arm S vs arm W, plus the free third leg and the v8-line shape row.

Same code paths as the arm-S standalone read
(``measurements/flywheel_armS_reads_2026-09-14/scripts/strength_read.py``); this adds arm W,
the pair delta, and the COMMON-STEP-SET refit the registration's sec 8.1 correction 1 requires.

Instrument: ``<run>/snapshot_ladder/ladder.json`` (dense, +/-10) — never ``eval/elo`` (+/-29)
[UNDERSTANDING sec 3.2 rule 1]. Every refit goes through ``fit_ladder(..., write=False)``, so
NOTHING under ``models/`` is written.

Runs:
  armS        ai_v13_01_flywheel_shaped    arm S — the pair's SHAPED arm (the era's reward/critic
                                           composition transplanted onto the current pin)
  armW        ai_v13_02_flywheel_winprob   arm W — the pair's WIN-PROB arm (V(s) = P(win|s), the
                                           value loss IS the win-prob head's BCE)
  winprob75M  ai_v12_02_winprob_critic     the free third leg: the 75M win-prob run at
                                           --ent-coef 0.02, pin f971caf2 (REFIT, rule 24)
  v8control   ai_v8_03_zarch_control_0718  the v8-line control — SHAPE ONLY (rule 4)

Rule 24 (a committed ladder.json carries the RECIPE it was fitted with) is enforced by reporting
committed-vs-refit for every run and by using ONLY the current-code refit in every cross-run row.
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

from agents.training import snapshot_ladder as sl  # noqa: E402

MODELS = Path("/home/goodlad/dev/gen3ai/models")
RUNS = {
    "armS": ("ai_v13_01_flywheel_shaped",
             "arm S — the flywheel pair's SHAPED arm (the era's reward/critic composition, 75M)"),
    "armW": ("ai_v13_02_flywheel_winprob",
             "arm W — the flywheel pair's WIN-PROB arm (V(s) = P(win|s), BCE value loss, 75M)"),
    "winprob75M": ("ai_v12_02_winprob_critic",
                   "the 75M win-prob run at --ent-coef 0.02 (the '0.02 leg'; pin f971caf2)"),
    "v8control": ("ai_v8_03_zarch_control_0718",
                  "the v8-line control (different architecture/obs/action era — SHAPE only, rule 4)"),
}
# The registered floor: the MAX pairwise |Delta| over the three same-argv 10M controls
# (45.0 / 35.3 / 9.7). It belongs to a DIFFERENT DEPTH (10M, four nodes) and is imported for
# want of anything at 75M — registration sec 8.1 / 9.1.
FLOOR_ELO = 45.0


def nodes_of(ladder: dict) -> list[tuple[int, float]]:
    return sorted((int(k), float(v)) for k, v in ladder["ratings"].items() if str(k).isdigit())


def node_se(ladder: dict) -> dict[int, float]:
    se = ladder.get("se") or {}
    return {int(k): float(v) for k, v in se.items() if str(k).isdigit()} if isinstance(se, dict) else {}


def ols_slope(steps_m: np.ndarray, elo: np.ndarray) -> dict:
    n = len(steps_m)
    if n < 3:
        return {"n": int(n), "slope_elo_per_Mstep": None, "se": None, "note": "fewer than 3 nodes"}
    x = steps_m - steps_m.mean()
    sxx = float((x * x).sum())
    slope = float((x * (elo - elo.mean())).sum() / sxx)
    intercept = float(elo.mean() - slope * steps_m.mean())
    resid = elo - (intercept + slope * steps_m)
    s2 = float((resid ** 2).sum() / (n - 2))
    se = float(np.sqrt(s2 / sxx))
    return {"n": int(n), "slope_elo_per_Mstep": round(slope, 4), "se": round(se, 4),
            "t": round(slope / se, 2) if se > 0 else None,
            "resid_rms": round(float(np.sqrt(s2)), 3),
            "span_Mstep": [round(float(steps_m[0]), 3), round(float(steps_m[-1]), 3)]}


def read_run(name: str, human: str) -> dict:
    run_dir = MODELS / name
    committed = json.loads((run_dir / "snapshot_ladder" / "ladder.json").read_text())
    cnodes = nodes_of(committed)
    n = len(cnodes)

    refit = sl.fit_ladder(str(run_dir), first_n=n, write=False)
    rnodes = nodes_of(refit)
    rse = node_se(refit)
    steps = np.array([s for s, _ in rnodes], dtype=float)
    elos = np.array([r for _, r in rnodes], dtype=float)

    rec: dict = {
        "run": name,
        "human_description": human,
        "committed_ladder_json": {
            "computed_at": committed.get("computed_at"),
            "n_nodes": n,
            "n_frozen_pairs_measured": committed.get("n_frozen_pairs_measured"),
            "n_pairs_possible": committed.get("n_pairs_possible"),
            "eval_sentinel_edges_dropped": committed.get("eval_sentinel_edges_dropped"),
            "recipe_stamp_present": committed.get("eval_sentinel_edges_dropped") is not None,
            "newest_node_step": cnodes[-1][0],
            "newest_node_elo": round(cnodes[-1][1], 1),
            "second_newest_elo": round(cnodes[-2][1], 1) if n >= 2 else None,
        },
        "current_code_refit": {
            "n_nodes": len(rnodes),
            "n_frozen_pairs_measured": refit.get("n_frozen_pairs_measured"),
            "eval_sentinel_edges_dropped": refit.get("eval_sentinel_edges_dropped"),
            "newest_node_step": int(steps[-1]),
            "newest_node_elo": round(float(elos[-1]), 1),
            "se_newest": rse.get(int(steps[-1])),
            "second_newest_step": int(steps[-2]) if n >= 2 else None,
            "second_newest_elo": round(float(elos[-2]), 1) if n >= 2 else None,
            "se_second_newest": rse.get(int(steps[-2])) if n >= 2 else None,
            "span_steps": [int(steps[0]), int(steps[-1])],
        },
        "committed_minus_refit_newest": round(cnodes[-1][1] - float(elos[-1]), 1),
        "nodes_current_code_refit": [{"step": int(s), "elo": round(float(r), 1),
                                      "se": rse.get(int(s))}
                                     for s, r in zip(steps, elos)],
    }

    curve = []
    for k in (4, 6, 8, 10, 12, 16, 20):
        if k > n:
            continue
        fit = sl.fit_ladder(str(run_dir), first_n=k, write=False)
        kn = nodes_of(fit)
        kse = node_se(fit).get(kn[-1][0], float("nan"))
        curve.append({"nodes": k, "newest_node_step": kn[-1][0],
                      "newest_node_elo": round(kn[-1][1], 1), "se": round(float(kse), 2),
                      "se_delta": round(float(np.sqrt(2.0) * kse), 2),
                      "ci95_delta": round(float(1.96 * np.sqrt(2.0) * kse), 1),
                      "n_frozen_pairs": fit.get("n_frozen_pairs_measured")})
    rec["resolution_curve"] = curve

    s_m, keep = steps / 1e6, n - 1
    s_k, e_k = s_m[:keep], elos[:keep]
    third = keep // 3
    rec["slopes_newest_node_dropped"] = {
        "nodes_used": int(keep),
        "late_third": ols_slope(s_k[-max(3, third):], e_k[-max(3, third):]),
        "middle_third": (ols_slope(s_k[third:2 * third], e_k[third:2 * third]) if keep >= 9
                         else ols_slope(s_k, e_k)),
        "all_nodes": ols_slope(s_k, e_k),
    }
    lt = rec["slopes_newest_node_dropped"]["late_third"]
    md = rec["slopes_newest_node_dropped"]["middle_third"]
    if lt.get("slope_elo_per_Mstep") is not None and md.get("slope_elo_per_Mstep") is not None:
        d = lt["slope_elo_per_Mstep"] - md["slope_elo_per_Mstep"]
        sd = float(np.sqrt(lt["se"] ** 2 + md["se"] ** 2))
        rec["slopes_newest_node_dropped"]["late_minus_middle"] = {
            "delta": round(d, 4), "se": round(sd, 4),
            "ci95": [round(d - 1.96 * sd, 3), round(d + 1.96 * sd, 3)],
            "covers_zero": bool(abs(d) <= 1.96 * sd)}

    adj = np.abs(np.diff(elos))
    rec["adjacent_node_spread"] = {
        "n": int(len(adj)), "max_abs_delta": round(float(adj.max()), 1),
        "median_abs_delta": round(float(np.median(adj)), 1),
        "mean_abs_delta": round(float(adj.mean()), 1),
        "caveat": ("bounds WITHIN-run wobble only — it is dominated by early LEARNING and is NOT a "
                   "run-to-run floor (registration sec 9.1)")}
    return rec


def cross(a: dict, b: dict, label: str, note: str, key: str = "current_code_refit") -> dict:
    ra, rb = a[key], b[key]
    out: dict = {}
    for tag, sa in (("newest", "se_newest"), ("second_newest", "se_second_newest")):
        ka = "newest_node_elo" if tag == "newest" else "second_newest_elo"
        d = ra[ka] - rb[ka]
        se = float(np.sqrt((ra[sa] or 0) ** 2 + (rb[sa] or 0) ** 2))
        claimable = FLOOR_ELO + 1.96 * se
        out[tag] = {"a_elo": ra[ka], "b_elo": rb[ka],
                    "a_step": ra["newest_node_step"] if tag == "newest" else ra["second_newest_step"],
                    "b_step": rb["newest_node_step"] if tag == "newest" else rb["second_newest_step"],
                    "delta_elo": round(d, 1), "se_delta": round(se, 2),
                    "ci95": [round(d - 1.96 * se, 1), round(d + 1.96 * se, 1)],
                    "smallest_claimable_abs_delta": round(claimable, 1),
                    "verdict": ("CANDIDATE" if abs(d) > claimable else "NOT DETECTED"),
                    "verdict_note": ("NOT DETECTED is never 'equivalent' — rule 6: equivalence needs "
                                     "the delta's own CI INSIDE the bar, and the 45.0 floor is "
                                     "imported from 10M four-node depth")}
    out["matched_count"] = [ra["n_nodes"], rb["n_nodes"]]
    out["step_spans"] = [ra["span_steps"], rb["span_steps"]]
    out["label"] = label
    out["confounds"] = note
    return out


def common_step_refit(a_name: str, b_name: str) -> dict:
    """The registration's sec 8.1 correction 1: refit BOTH arms on the COMMON step set, so node k
    of one is the same STEP as node k of the other and not merely the same ordinal."""
    a_steps = [s for s, _ in nodes_of(json.loads(
        (MODELS / a_name / "snapshot_ladder" / "ladder.json").read_text()))]
    b_steps = [s for s, _ in nodes_of(json.loads(
        (MODELS / b_name / "snapshot_ladder" / "ladder.json").read_text()))]
    common = sorted(set(a_steps) & set(b_steps))
    if len(common) < 3:
        return {"n_common": len(common), "note": "too few common steps to fit"}
    out: dict = {"n_common": len(common), "common_steps": common,
                 "reportable": len(common) >= 12,
                 "floor_note": ("the registration says n >= 12 is the floor below which the read "
                                "is not reported at all")}
    fits = {}
    for key, name in (("a", a_name), ("b", b_name)):
        f = sl.fit_ladder(str(MODELS / name), steps=common, write=False)
        nn = nodes_of(f)
        se = node_se(f)
        fits[key] = {"run": name, "n_nodes": len(nn),
                     "nodes": [{"step": s, "elo": round(r, 1), "se": se.get(s)} for s, r in nn],
                     "newest_step": nn[-1][0], "newest_elo": round(nn[-1][1], 1),
                     "se_newest": se.get(nn[-1][0]),
                     "second_newest_step": nn[-2][0], "second_newest_elo": round(nn[-2][1], 1),
                     "se_second_newest": se.get(nn[-2][0])}
    out["fits"] = fits
    for tag, ke, ks in (("newest", "newest_elo", "se_newest"),
                        ("second_newest", "second_newest_elo", "se_second_newest")):
        d = fits["a"][ke] - fits["b"][ke]
        se = float(np.sqrt((fits["a"][ks] or 0) ** 2 + (fits["b"][ks] or 0) ** 2))
        claimable = FLOOR_ELO + 1.96 * se
        out[tag] = {"delta_elo": round(d, 1), "se_delta": round(se, 2),
                    "ci95": [round(d - 1.96 * se, 1), round(d + 1.96 * se, 1)],
                    "smallest_claimable_abs_delta": round(claimable, 1),
                    "verdict": ("CANDIDATE" if abs(d) > claimable else "NOT DETECTED")}
    return out


def main() -> int:
    out = {"generated_at": _dt.datetime.now().astimezone().isoformat(),
           "instrument": "snapshot_ladder/ladder.json (dense, +/-10); refits via fit_ladder(write=False)",
           "floor_elo": FLOOR_ELO,
           "floor_provenance": ("MAX pairwise |Delta| over the three same-argv 10M controls "
                                "(45.0 / 35.3 / 9.7) — a 10M four-node floor imported to 75M"),
           "runs": {}}
    for key, (name, human) in RUNS.items():
        p = MODELS / name / "snapshot_ladder" / "ladder.json"
        out["runs"][key] = read_run(name, human) if p.exists() else {"run": name, "error": "no ladder.json"}

    r = out["runs"]
    out["cross_run"] = {
        "armS_vs_armW": cross(
            r["armS"], r["armW"], "THE PAIR — arm S (shaped) minus arm W (win-prob)",
            "MATCHED on pin (6eb9c776), dose (4.578e-8 by declaration), length (75,005,952), seed "
            "(1001), --ent-coef (0.05), eval regime (--eval-sentinel-greedy, RECORDED true on both), "
            "architecture and ecology; both crossed into self-play at the identical 4,128,768 and "
            "both retain 20 ladder nodes; both ladder.json files are CURRENT-RECIPE (rule 24 "
            "satisfied on both sides). The treatment is the critic objective. The binding "
            "limitation is n=1 per arm — rule 22: any direction is a CANDIDATE, never a family "
            "verdict."),
        "armS_vs_winprob75M": cross(
            r["armS"], r["winprob75M"], "arm S minus the 0.02 leg (REFIT)",
            "DIFFERENT pin (6eb9c776 vs f971caf2), DIFFERENT --ent-coef (0.05 vs 0.02), DIFFERENT "
            "critic objective. The 0.02 leg's committed ladder.json is STALE-RECIPE (rule 24) and "
            "is REFIT here. ladder.json is UNAFFECTED by the 2026-09-07 eval-sentinel regime "
            "boundary; eval/elo and win_rate_vs_pool are NOT and are not read here."),
        "armW_vs_winprob75M": cross(
            r["armW"], r["winprob75M"], "arm W minus the 0.02 leg (REFIT)",
            "SAME critic objective (winprob) on both sides, so this row isolates the ENTROPY "
            "COEFFICIENT axis at full length — but it still carries a pin confound (6eb9c776 vs "
            "f971caf2) and the 0.02 leg's stale-recipe trap (REFIT here, rule 24). Ladder-only: "
            "the 0.02 leg spans the 2026-09-07 opponent-regime boundary, which ladder.json is "
            "immune to and eval/elo is not."),
        "armS_vs_v8control": cross(
            r["armS"], r["v8control"], "arm S minus the v8 control",
            "SHAPE ONLY (rule 4): different architecture, obs dim, action-space era, reward "
            "composition, clip range (0.10 vs 0.15) and ecology; 20 nodes over 22-72M against 10 "
            "nodes over 150.9-248.0M. The Elo row must NOT be quoted as a difference."),
        "armW_vs_v8control": cross(
            r["armW"], r["v8control"], "arm W minus the v8 control",
            "SHAPE ONLY (rule 4), same confound set as the arm-S row. Must NOT be quoted."),
    }
    out["common_step_refit_armS_armW"] = common_step_refit(
        "ai_v13_01_flywheel_shaped", "ai_v13_02_flywheel_winprob")

    # The node-by-node curves at matched COUNT, side by side by ordinal.
    sn = r["armS"]["nodes_current_code_refit"]
    wn = r["armW"]["nodes_current_code_refit"]
    out["node_by_node_matched_count"] = [
        {"k": i + 1,
         "armS_step": sn[i]["step"] if i < len(sn) else None,
         "armS_elo": sn[i]["elo"] if i < len(sn) else None,
         "armW_step": wn[i]["step"] if i < len(wn) else None,
         "armW_elo": wn[i]["elo"] if i < len(wn) else None,
         "S_minus_W": (round(sn[i]["elo"] - wn[i]["elo"], 1)
                       if i < len(sn) and i < len(wn) else None)}
        for i in range(max(len(sn), len(wn)))]

    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("pair_strength_read.json")
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["cross_run"]["armS_vs_armW"], indent=1))
    print(json.dumps(out["common_step_refit_armS_armW"].get("newest"), indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
