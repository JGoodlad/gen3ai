"""Roll ``analyze.py``'s per-run JSON up into the README's headline numbers.

    python .../summary.py <result.json> <out-summary.json>
"""
from __future__ import annotations

import json
import statistics as st
import sys
from pathlib import Path


def q(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    return {"n": len(xs), "min": xs[0], "median": st.median(xs), "max": xs[-1]}


def main(argv):
    res = json.loads(Path(argv[0]).read_text())["runs"]
    first_launch, restart, toy_all = [], [], []
    dry = []
    recompiles, reserved_drift, train_drift = [], [], []
    for run, r in res.items():
        for k, p in enumerate(r["procs"]):
            lg = p["log"]
            if lg.get("prewarm_s") is None:
                continue
            if run.startswith("rb_toy"):
                toy_all.append(lg["prewarm_s"])
            else:
                (restart if k > 0 else first_launch).append(lg["prewarm_s"])
                if lg.get("dry_update_s"):
                    dry.append(lg["dry_update_s"])
            hs = [h for h in p["hourly"] if h.get("train_s_median")]
            if len(hs) >= 2:
                train_drift.append(round(100 * (hs[-1]["train_s_median"] / hs[0]["train_s_median"] - 1), 2))
                if hs[0].get("cuda_reserved_mib") and hs[-1].get("cuda_reserved_mib"):
                    reserved_drift.append(hs[-1]["cuda_reserved_mib"] - hs[0]["cuda_reserved_mib"])
            for h in p["hourly"]:
                if h.get("recompiles_after_lock") is not None:
                    recompiles.append(h["recompiles_after_lock"])
    seams = [s for r in res.values() for s in r["seams"]]
    launcher = [s for s in seams if not s["human_gap"]]
    out = {
        # a FIRST launch is cold unless the shared compile cache already holds its key (the same
        # pinned commit launched before): bimodal, ~55-90 s warm vs ~165-420 s cold
        "prewarm_s": {"first_launch_production": q(first_launch),
                      "first_launch_le_120s": sum(1 for x in first_launch if x <= 120),
                      "first_launch_gt_120s": sum(1 for x in first_launch if x > 120),
                      "same_run_restart_production": q(restart), "toy_any": q(toy_all)},
        "dry_first_update_s_production": q(dry),
        "train_s_first_to_last_hour_pct": q(train_drift),
        "cuda_reserved_first_to_last_hour_mib": q(reserved_drift),
        "recompiles_after_lock_hourly_medians_max": max(recompiles) if recompiles else None,
        "launcher_restart_gaps_s": [{"run": run, "prev_end": s["prev_end"], "gap_s": s["gap_s"],
                                     "steps_lost": s["steps_prev_end"] - s["steps_resume_first"],
                                     "prewarm_s": s["prewarm_s"], "first_cycles_excess_s": s["first_cycles_excess_s"]}
                                    for run, r in res.items() for s in r["seams"] if not s["human_gap"]],
        "launcher_restart_gap_s_quantiles": q([s["gap_s"] for s in launcher]),
        "learning_jump_over_sd": q([j["jump_over_sd"] for s in seams for j in s["learning"].values()
                                    if j.get("jump_over_sd") is not None]),
        "learning_jump_over_sd_gt2": [(run, s["to"][-30:], tag, j["jump_over_sd"])
                                      for run, r in res.items() for s in r["seams"]
                                      for tag, j in s["learning"].items()
                                      if (j.get("jump_over_sd") or 0) > 2],
    }
    Path(argv[1]).write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "launcher_restart_gaps_s"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
