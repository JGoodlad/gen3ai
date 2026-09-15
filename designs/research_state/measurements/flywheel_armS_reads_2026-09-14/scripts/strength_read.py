#!/usr/bin/env python3
"""ARM S run-end STRENGTH read + the two registered external ladder comparators.

Instrument: ``<run>/snapshot_ladder/ladder.json`` (dense, +/-10) — never ``eval/elo`` (+/-29).
Every refit goes through ``fit_ladder(..., first_n=N, write=False)``; ``first_n`` forces
``write=False``, so NOTHING under ``models/`` is written.

Runs, each read identically:
  armS        ai_v13_01_flywheel_shaped     the flywheel pair's SHAPED arm (the era's composition)
  winprob75M  ai_v12_02_winprob_critic      the 75M win-prob run at --ent-coef 0.02 (the "0.02 leg")
  v8control   ai_v8_03_zarch_control_0718   the v8-line control (SHAPE comparison only, rule 4)

Per designs/research_state/flywheel_era_pair_2026-09-12.md sec 8.1 / 8.2 and UNDERSTANDING sec 3.2:
  * the headline rating at run end and the node count
  * the COMMITTED ladder.json against a CURRENT-CODE refit of the same node set.  These differ
    whenever the committed file predates `3e6875a5` (which dropped the eval-sentinel edges from
    the fit); the difference is reported rather than smoothed over, because a cross-run Elo taken
    between a stale committed file and a fresh one is not a comparison.
  * the se(Delta) resolution curve, refit at increasing first_n
  * the within-run LATE slope against the MIDDLE slope, with the NEWEST node DROPPED
    (sec 3.2 rule 2: the newest node of any fit is systematically inflated)
  * the adjacent-node |Delta| spread — the only within-run floor proxy the pair has (sec 9.1)
  * the cross-run deltas at matched snapshot COUNT, each against the registered resolution
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
    "winprob75M": ("ai_v12_02_winprob_critic",
                   "the 75M win-prob run at --ent-coef 0.02 (the ' 0.02 leg'; pin f971caf2)"),
    "v8control": ("ai_v8_03_zarch_control_0718",
                  "the v8-line control (different architecture/obs/action era — SHAPE only, rule 4)"),
}
# The registered floor: the MAX pairwise |Delta| over the three same-argv 10M controls
# (45.0 / 35.3 / 9.7).  It belongs to a DIFFERENT DEPTH (10M, four nodes) and is imported for
# want of anything at 75M — registration sec 8.1 / 9.1.
FLOOR_ELO = 45.0


def nodes_of(ladder: dict) -> list[tuple[int, float]]:
    """(step, rating) for every SNAPSHOT node, ascending. Keys are bare step strings."""
    return sorted((int(k), float(v)) for k, v in ladder["ratings"].items() if str(k).isdigit())


def node_se(ladder: dict) -> dict[int, float]:
    se = ladder.get("se") or {}
    return {int(k): float(v) for k, v in se.items() if str(k).isdigit()} if isinstance(se, dict) else {}


def ols_slope(steps_m: np.ndarray, elo: np.ndarray) -> dict:
    """Slope in Elo per 1M steps with the residual-based se of the slope."""
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

    # THE CURRENT-CODE REFIT of the same node count. This is the object every cross-run
    # comparison uses, because the committed files were written by different code versions.
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
            "pairs_by_source": committed.get("pairs_by_source"),
            "newest_node_step": cnodes[-1][0],
            "newest_node_elo": round(cnodes[-1][1], 1),
            "second_newest_elo": round(cnodes[-2][1], 1) if n >= 2 else None,
        },
        "current_code_refit": {
            "n_nodes": len(rnodes),
            "n_frozen_pairs_measured": refit.get("n_frozen_pairs_measured"),
            "eval_sentinel_edges_dropped": refit.get("eval_sentinel_edges_dropped"),
            "pairs_by_source": refit.get("pairs_by_source"),
            "newest_node_step": int(steps[-1]),
            "newest_node_elo": round(float(elos[-1]), 1),
            "se_newest": rse.get(int(steps[-1])),
            "second_newest_step": int(steps[-2]) if n >= 2 else None,
            "second_newest_elo": round(float(elos[-2]), 1) if n >= 2 else None,
            "se_second_newest": rse.get(int(steps[-2])) if n >= 2 else None,
            "span_steps": [int(steps[0]), int(steps[-1])],
        },
        "committed_minus_refit_newest": round(cnodes[-1][1] - float(elos[-1]), 1),
        "nodes_current_code_refit": [{"step": int(s), "elo": round(float(r), 1)}
                                     for s, r in zip(steps, elos)],
    }

    # ---- resolution curve -----------------------------------------------------------------
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

    # ---- slopes, newest node DROPPED ------------------------------------------------------
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
    lt, md = rec["slopes_newest_node_dropped"]["late_third"], rec["slopes_newest_node_dropped"]["middle_third"]
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


def cross(a: dict, b: dict, label: str, note: str) -> dict:
    ra, rb = a["current_code_refit"], b["current_code_refit"]
    out = {}
    for tag, ka, kb, sa, sb in (("newest", "newest_node_elo", "newest_node_elo", "se_newest", "se_newest"),
                                ("second_newest", "second_newest_elo", "second_newest_elo",
                                 "se_second_newest", "se_second_newest")):
        d = ra[ka] - rb[kb]
        se = float(np.sqrt((ra[sa] or 0) ** 2 + (rb[sb] or 0) ** 2))
        claimable = FLOOR_ELO + 1.96 * se
        out[tag] = {"delta_elo": round(d, 1), "se_delta": round(se, 2),
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
        "armS_vs_winprob75M": cross(
            r["armS"], r["winprob75M"], "arm S minus the 0.02 leg",
            "DIFFERENT pin (6eb9c776 vs f971caf2), DIFFERENT --ent-coef (0.05 vs 0.02), DIFFERENT "
            "critic objective, and DIFFERENT step spans at the same node count (22-72M vs 36-74M). "
            "ladder.json is UNAFFECTED by the 2026-09-07 eval-sentinel regime boundary; eval/elo "
            "and win_rate_vs_pool are NOT and are not read here. This is NOT the pair's read."),
        "armS_vs_v8control": cross(
            r["armS"], r["v8control"], "arm S minus the v8 control",
            "SHAPE ONLY (rule 4): different architecture, obs dim, action-space era, reward "
            "composition, clip range (0.10 vs 0.15) and ecology; 20 nodes over 22-72M against 10 "
            "nodes over 150.9-248.0M — a different DEPTH and a different COUNT. Both are far above "
            "the bot anchors, so a direct match is required before any gap is quoted as a "
            "difference. The Elo row below is printed for completeness and must NOT be quoted."),
    }
    dest = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("strength_read.json")
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps(out["cross_run"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
